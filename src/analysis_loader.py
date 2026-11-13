"""Results loading, gold scoring, missing-seed aggregation, alignment, and context-token reporting (P5b).

Implements:
1. Split-aware JSONL results loader (`main` or `pilot`) rejecting stub rows, scoring SQuAD EM
   and token F1 via `src/metrics.py`, computing per-question accuracy as the mean over AVAILABLE
   seeds for each `(model, condition, distractor_type, n, qid)`, and aligning across conditions
   and models by `qid` intersection.
2. Context token-length summary (`Qwen3` and `Llama-3.1` token counts from
   `data/frozen/main_seed{S}.jsonl` or `data/inputs/{split}_seed{S}.jsonl`) for `closed_book` (0),
   `oracle` (`gold`), and `hard_negative` / `random_length_matched` nested prefixes `n in {1, 2, 4, 8}`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.harness import VALID_SPLITS, load_result_rows
from src.metrics import exact_match, token_f1

REPO_ROOT = Path(__file__).resolve().parent.parent

ConditionKey = tuple[str, str | None, int]


def resolve_gold_path(
    split: str,
    *,
    repo_root: Path = REPO_ROOT,
    gold_path: Path | str | None = None,
) -> Path:
    """Resolve the gold-answer JSON path for `split` (`main`, `pilot`, or `reserve`)."""
    if gold_path is not None:
        return Path(gold_path)
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {VALID_SPLITS}.")

    if split == "main":
        frozen_manifest = repo_root / "data" / "frozen" / "manifest.json"
        frozen_main = repo_root / "data" / "frozen" / "main.json"
        if frozen_manifest.exists() and frozen_main.exists():
            return frozen_main
        if frozen_main.exists() and not (repo_root / "data" / "sampling" / "main.json").exists():
            return frozen_main
        return repo_root / "data" / "sampling" / "main.json"

    return repo_root / "data" / "sampling" / f"{split}.json"


def load_gold_answers(
    split: str = "main",
    *,
    repo_root: Path = REPO_ROOT,
    gold_path: Path | str | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Load canonical question order and `qid -> gold_answers` for `split`."""
    path = resolve_gold_path(split, repo_root=repo_root, gold_path=gold_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold answers file not found: {path}")

    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"Gold answers file must be a non-empty JSON list: {path}")

    qid_order: list[str] = []
    gold_by_qid: dict[str, list[str]] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}:{idx}: expected JSON object")
        qid = str(item["question_id"])
        answers = [str(a) for a in item["gold_answers"]]
        if not answers:
            raise ValueError(f"{path}: question {qid!r} has empty gold_answers")
        if qid in gold_by_qid:
            raise ValueError(f"{path}: duplicate question_id {qid!r}")
        qid_order.append(qid)
        gold_by_qid[qid] = answers

    return qid_order, gold_by_qid


def _canonical_condition_key(row: Mapping[str, Any]) -> ConditionKey:
    """Extract canonical `(condition, distractor_type, n)` tuple from a result row."""
    condition = str(row["condition"])
    raw_dtype = row.get("distractor_type")
    dtype = str(raw_dtype) if raw_dtype is not None else None
    n = int(row["n"])
    return (condition, dtype, n)


def _format_condition_label(cond_key: ConditionKey) -> str:
    """Format canonical `(condition, distractor_type, n)` as a readable label."""
    condition, dtype, n = cond_key
    if condition in ("closed_book", "oracle"):
        return condition
    return f"{dtype}_n{n}"


def load_and_score_results(
    results_dir: Path | str,
    split: str = "main",
    *,
    gold_path: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
    allow_stub: bool = False,
    models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load JSONL result rows for `split`, score EM/F1, and average over available seeds.

    Cleanly handles questions missing 1 or 2 seeds by computing the mean over all
    available seeds for each `(model, condition, distractor_type, n, qid)`.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {VALID_SPLITS}.")

    res_path = Path(results_dir)
    qid_order, gold_by_qid = load_gold_answers(
        split, repo_root=repo_root, gold_path=gold_path
    )
    raw_rows = load_result_rows(res_path, allow_stub=allow_stub, repo_root=repo_root)

    model_filter = set(models) if models is not None else None
    filtered_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("split") != split:
            continue
        if model_filter is not None and row.get("model") not in model_filter:
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError(
            f"No result rows found for split={split!r} in {res_path}"
            + (f" matching models={sorted(model_filter)!r}" if model_filter else "")
            + "."
        )

    # Group by model -> condition_key -> qid -> seed
    grouped: dict[str, dict[ConditionKey, dict[str, dict[int | None, dict[str, Any]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for row in filtered_rows:
        qid = str(row["qid"])
        if qid not in gold_by_qid:
            raise ValueError(f"Result row contains unknown qid {qid!r} for split={split!r}.")
        model = str(row["model"])
        cond_key = _canonical_condition_key(row)
        raw_seed = row.get("seed")
        seed: int | None = int(raw_seed) if raw_seed is not None else None

        if cond_key[0] in ("closed_book", "oracle") and seed is not None:
            raise ValueError(
                f"Seed-independent condition {cond_key[0]!r} has non-null seed={seed!r} for qid={qid!r}."
            )
        if cond_key[0] == "rag" and seed is None:
            raise ValueError(
                f"RAG condition {cond_key!r} must specify an integer seed for qid={qid!r}."
            )

        seed_bucket = grouped[model][cond_key][qid]
        if seed in seed_bucket:
            raise ValueError(
                f"Duplicate result row for model={model!r}, condition={cond_key!r}, "
                f"seed={seed!r}, qid={qid!r}."
            )

        extracted = str(row["extracted_answer"])
        gold_answers = gold_by_qid[qid]
        seed_bucket[seed] = {
            "em": exact_match(extracted, gold_answers),
            "f1": token_f1(extracted, gold_answers),
            "valid": bool(row.get("valid", True)),
        }

    per_question: dict[str, dict[ConditionKey, dict[str, dict[str, Any]]]] = {}
    condition_maps_em: dict[str, dict[Any, Any]] = {}
    condition_maps_f1: dict[str, dict[Any, Any]] = {}
    seed_coverage: dict[str, dict[str, dict[str, Any]]] = {}

    for model, cond_dict in grouped.items():
        per_question[model] = {}
        condition_maps_em[model] = {}
        condition_maps_f1[model] = {}
        seed_coverage[model] = {}

        for cond_key, qid_map in cond_dict.items():
            cond, dtype, n = cond_key
            label = _format_condition_label(cond_key)
            per_q_cond: dict[str, dict[str, Any]] = {}
            em_map: dict[str, float] = {}
            f1_map: dict[str, float] = {}
            counts_by_n_seeds: dict[int, int] = defaultdict(int)
            partial_qids: list[str] = []

            for qid in qid_order:
                if qid not in qid_map:
                    continue
                seed_entries = qid_map[qid]
                avail_seeds = sorted(
                    seed_entries.keys(),
                    key=lambda s: (-1 if s is None else int(s)),
                )
                em_mean = float(np.mean([seed_entries[s]["em"] for s in avail_seeds]))
                f1_mean = float(np.mean([seed_entries[s]["f1"] for s in avail_seeds]))
                n_seeds = len(avail_seeds)
                counts_by_n_seeds[n_seeds] += 1
                if cond == "rag" and n_seeds < 3:
                    partial_qids.append(qid)

                per_q_cond[qid] = {
                    "em": em_mean,
                    "f1": f1_mean,
                    "n_seeds": n_seeds,
                    "seeds": avail_seeds,
                    "seed_scores": dict(seed_entries),
                }
                em_map[qid] = em_mean
                f1_map[qid] = f1_mean

            per_question[model][cond_key] = per_q_cond
            missing_qids = [q for q in qid_order if q not in per_q_cond]
            seed_coverage[model][label] = {
                "condition_key": cond_key,
                "n_questions_present": len(per_q_cond),
                "counts_by_n_seeds": dict(sorted(counts_by_n_seeds.items())),
                "partial_seed_qids": partial_qids,
                "missing_all_seeds_qids": missing_qids,
            }

            if cond in ("closed_book", "oracle"):
                condition_maps_em[model][cond] = em_map
                condition_maps_f1[model][cond] = f1_map
            else:
                assert dtype is not None
                condition_maps_em[model].setdefault(dtype, {})[n] = em_map
                condition_maps_f1[model].setdefault(dtype, {})[n] = f1_map

    return {
        "split": split,
        "results_dir": str(res_path),
        "qid_order": qid_order,
        "gold_by_qid": gold_by_qid,
        "models": list(grouped.keys()),
        "per_question": per_question,
        "condition_maps": {
            "em": condition_maps_em,
            "f1": condition_maps_f1,
        },
        "seed_coverage": seed_coverage,
    }


def align_conditions_by_qid(
    loaded_results: Mapping[str, Any],
    *,
    models: Sequence[str] | None = None,
    condition_keys: Sequence[ConditionKey] | None = None,
    metric: str = "em",
) -> dict[str, Any]:
    """Align per-question scores across models and conditions by `qid` intersection.

    Preserves canonical `qid_order` from the gold split file.
    """
    if metric not in ("em", "f1"):
        raise ValueError(f"metric must be 'em' or 'f1', got {metric!r}.")

    per_question: Mapping[str, Mapping[ConditionKey, Mapping[str, Mapping[str, Any]]]] = (
        loaded_results["per_question"]
    )
    qid_order: list[str] = list(loaded_results["qid_order"])
    target_models = list(models) if models is not None else list(loaded_results["models"])
    if not target_models:
        raise ValueError("No models available to align.")

    for m in target_models:
        if m not in per_question:
            raise KeyError(f"Model {m!r} not found in loaded_results.")

    if condition_keys is not None:
        target_keys = list(condition_keys)
    else:
        seen_keys: list[ConditionKey] = []
        for m in target_models:
            for ck in per_question[m]:
                if ck not in seen_keys:
                    seen_keys.append(ck)
        target_keys = seen_keys

    if not target_keys:
        raise ValueError("No condition keys found to align.")

    for m in target_models:
        for ck in target_keys:
            if ck not in per_question[m]:
                raise KeyError(f"Model {m!r} is missing required condition {ck!r}.")

    aligned_qids = [
        qid
        for qid in qid_order
        if all(qid in per_question[m][ck] for m in target_models for ck in target_keys)
    ]
    if not aligned_qids:
        raise ValueError(
            "Question-ID intersection across requested models and conditions is empty."
        )

    aligned_set = set(aligned_qids)
    dropped_qids = [qid for qid in qid_order if qid not in aligned_set]

    by_qid: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, Any]] = {}
    model_conditions: dict[str, dict[str, Any]] = {}

    for m in target_models:
        by_qid_m: dict[str, Any] = {}
        vec_m: dict[str, Any] = {}
        for ck in target_keys:
            cond, dtype, n = ck
            q_scores = {
                qid: float(per_question[m][ck][qid][metric])
                for qid in aligned_qids
            }
            q_arr = np.asarray([q_scores[qid] for qid in aligned_qids], dtype=np.float64)
            if cond in ("closed_book", "oracle"):
                by_qid_m[cond] = q_scores
                vec_m[cond] = q_arr
            else:
                assert dtype is not None
                by_qid_m.setdefault(dtype, {})[n] = q_scores
                vec_m.setdefault(dtype, {})[n] = q_arr
        by_qid[m] = by_qid_m
        vectors[m] = vec_m
        model_conditions[m] = vec_m

    return {
        "split": loaded_results["split"],
        "metric": metric,
        "models": target_models,
        "condition_keys": target_keys,
        "qids": aligned_qids,
        "n_questions": len(aligned_qids),
        "dropped_qids": dropped_qids,
        "by_qid": by_qid,
        "vectors": vectors,
        "model_conditions": model_conditions,
    }


def _resolve_token_input_dir(
    split: str,
    *,
    repo_root: Path = REPO_ROOT,
    inputs_dir: Path | str | None = None,
) -> Path:
    """Resolve input JSONL directory for context-token summaries."""
    if inputs_dir is not None:
        return Path(inputs_dir)
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {VALID_SPLITS}.")
    if split == "main" and (repo_root / "data" / "frozen" / "manifest.json").exists():
        return repo_root / "data" / "frozen"
    return repo_root / "data" / "inputs"


def _token_stats(values: Sequence[int]) -> dict[str, float | int]:
    """Compute summary statistics (mean, median, sd, min, max) for token counts."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize empty token count sequence.")
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def summarize_context_tokens(
    split: str = "main",
    *,
    repo_root: Path = REPO_ROOT,
    inputs_dir: Path | str | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    n_levels: Sequence[int] = (0, 1, 2, 4, 8),
) -> dict[str, Any]:
    """Compute mean and median context token lengths (`Qwen3` and `Llama-3.1`) by condition and n.

    Reports:
    - `closed_book` (n=0): 0 context tokens for every question.
    - `oracle` (n=0): `token_lengths[tok]["gold"]` per unique question.
    - `hard_negative` and `random_length_matched` for each `n in n_levels` > 0:
      `gold + sum(distractors[:n])` across loaded `(question_id, seed)` rows.
    """
    target_dir = _resolve_token_input_dir(split, repo_root=repo_root, inputs_dir=inputs_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Inputs directory does not exist: {target_dir}")

    loaded_files: list[str] = []
    seeds_loaded: list[int] = []
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}

    for s in seeds:
        path = target_dir / f"{split}_seed{int(s)}.jsonl"
        if not path.exists():
            continue
        seed_rows: list[dict[str, Any]] = []
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            seed_rows.append(obj)
        rows_by_seed[int(s)] = seed_rows
        loaded_files.append(str(path))
        seeds_loaded.append(int(s))

    if not rows_by_seed:
        raise FileNotFoundError(
            f"No {split}_seed*.jsonl files found in {target_dir} for seeds={list(seeds)}."
        )

    max_rag_n = max((int(n) for n in n_levels if int(n) > 0), default=0)
    oracle_tokens: dict[str, dict[str, int]] = {"qwen3": {}, "llama3": {}}
    all_rows: list[dict[str, Any]] = []

    for s in seeds_loaded:
        for row in rows_by_seed[s]:
            qid = str(row["question_id"])
            tok_lengths = row["token_lengths"]
            for tok in ("qwen3", "llama3"):
                gold_len = int(tok_lengths[tok]["gold"])
                hard_lens = tok_lengths[tok]["hard"]
                rand_lens = tok_lengths[tok]["random"]
                if len(hard_lens) < max_rag_n or len(rand_lens) < max_rag_n:
                    raise ValueError(
                        f"Question {qid!r} seed={s} has fewer than {max_rag_n} distractor lengths for {tok}."
                    )
                if qid in oracle_tokens[tok]:
                    if oracle_tokens[tok][qid] != gold_len:
                        raise ValueError(
                            f"Inconsistent gold {tok} token length for {qid!r} across seeds: "
                            f"{oracle_tokens[tok][qid]} vs {gold_len}."
                        )
                else:
                    oracle_tokens[tok][qid] = gold_len
            all_rows.append(row)

    unique_qids = list(oracle_tokens["qwen3"].keys())
    n_unique = len(unique_qids)
    n_seed_rows = len(all_rows)

    summary_rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    by_tuple: dict[ConditionKey, dict[str, Any]] = {}

    def _record(
        cond: str,
        dtype: str | None,
        n: int,
        qwen3_vals: Sequence[int],
        llama3_vals: Sequence[int],
        n_q: int,
    ) -> None:
        label = _format_condition_label((cond, dtype, n))
        q_stats = _token_stats(qwen3_vals)
        l_stats = _token_stats(llama3_vals)
        entry: dict[str, Any] = {
            "label": label,
            "condition": cond,
            "distractor_type": dtype,
            "n": int(n),
            "n_rows": len(qwen3_vals),
            "n_questions": int(n_q),
            "qwen3": q_stats,
            "llama3": l_stats,
            "Qwen3": q_stats,
            "Llama-3.1": l_stats,
        }
        summary_rows.append(entry)
        by_label[label] = entry
        by_tuple[(cond, dtype, int(n))] = entry

    # closed_book: 0 context tokens
    _record(
        "closed_book",
        None,
        0,
        [0] * n_unique,
        [0] * n_unique,
        n_unique,
    )

    # oracle (n=0): gold passage only
    if 0 in n_levels:
        _record(
            "oracle",
            None,
            0,
            [oracle_tokens["qwen3"][q] for q in unique_qids],
            [oracle_tokens["llama3"][q] for q in unique_qids],
            n_unique,
        )

    rag_ns = sorted(int(n) for n in n_levels if int(n) > 0)
    for dtype, field_key in (
        ("hard_negative", "hard"),
        ("random_length_matched", "random"),
    ):
        for n in rag_ns:
            qwen3_vals = [
                int(r["token_lengths"]["qwen3"]["gold"])
                + sum(int(x) for x in r["token_lengths"]["qwen3"][field_key][:n])
                for r in all_rows
            ]
            llama3_vals = [
                int(r["token_lengths"]["llama3"]["gold"])
                + sum(int(x) for x in r["token_lengths"]["llama3"][field_key][:n])
                for r in all_rows
            ]
            row_qids = {str(r["question_id"]) for r in all_rows}
            _record("rag", dtype, n, qwen3_vals, llama3_vals, len(row_qids))

    return {
        "split": split,
        "inputs_dir": str(target_dir),
        "input_files": loaded_files,
        "seeds_loaded": seeds_loaded,
        "n_unique_questions": n_unique,
        "n_seed_rows": n_seed_rows,
        "rows": summary_rows,
        "by_condition": by_label,
        "by_tuple": by_tuple,
    }
