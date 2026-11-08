"""Distractor selection, leak filtering, length matching, and prompt-input construction (Step p2c).

Hard Rule 11 compliance:
- Rejects KMP_DUPLICATE_LIB_OK if set in environment.
- Imports src.omp_guard before any transformers/torch imports.
- Enforces offline mode (HF_HUB_OFFLINE=1, local_files_only=True).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import string
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_INITIAL_KMP_DUPLICATE_LIB_OK = (
    os.environ.get("KMP_DUPLICATE_LIB_OK")
    if "src.omp_guard" not in sys.modules and "omp_guard" not in sys.modules
    else None
)
if _INITIAL_KMP_DUPLICATE_LIB_OK is not None:
    raise RuntimeError(
        "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK was set in the environment prior to omp_guard."
    )

os.environ["HF_HUB_OFFLINE"] = "1"

import src.omp_guard  # noqa: F401, E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

DEFAULT_PREREG_PATH = REPO_ROOT / "configs" / "prereg.yaml"
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_SAMPLING_DIR = REPO_ROOT / "data" / "sampling"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "data" / "candidates" / "hard_candidates.jsonl"
DEFAULT_INPUTS_DIR = REPO_ROOT / "data" / "inputs"
DEFAULT_QWEN3_TOKENIZER_PATH = REPO_ROOT / "models" / "hf" / "Qwen__Qwen3-8B"
DEFAULT_LLAMA3_TOKENIZER_PATH = (
    REPO_ROOT / "models" / "hf" / "unsloth__Meta-Llama-3.1-8B-Instruct"
)

SPLITS = ("main", "pilot", "reserve")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def assert_no_kmp_duplicate_lib_ok() -> None:
    """Enforce AGENTS.md Hard Rule 11: caller must never set KMP_DUPLICATE_LIB_OK."""
    if _INITIAL_KMP_DUPLICATE_LIB_OK is not None:
        raise RuntimeError(
            "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK is set in environment."
        )


def sha256_bytes(data: bytes) -> str:
    """Compute hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    return sha256_bytes(path.read_bytes())


def normalize_squad(text: str) -> str:
    """Official SQuAD v2.0 normalization: lowercase, strip punctuation, remove articles, fix whitespace."""
    lowered = text.lower()
    no_punc = lowered.translate(_PUNC_TABLE)
    no_articles = _ARTICLE_RE.sub(" ", no_punc)
    return " ".join(no_articles.split())


def has_answer_leak(
    normalized_passage: str,
    normalized_gold_answers: list[str],
) -> bool:
    """Return True if any SQuAD-normalized gold answer appears as a substring of normalized_passage.

    Note: If a gold answer normalizes to the empty string (e.g. an answer consisting purely of
    an article or punctuation), Python's `"" in normalized_passage` is True, which conservatively
    rejects the passage unless only non-empty normalized strings are checked. Here we check
    verbatim substring membership `ans in normalized_passage` for every normalized gold answer;
    if `ans == ""`, it is treated as leaking if and only if all normalized gold answers are
    checked directly. To avoid vacuous empty-string rejection when other valid answers exist,
    WAIT: if `ans == ""`, `"" in s` is always True! We check non-empty normalized answers, and if
    ALL gold answers normalize to empty string, we reject.
    """
    if not normalized_gold_answers:
        return False
    non_empty = [ans for ans in normalized_gold_answers if ans]
    if not non_empty:
        return True
    return any(ans in normalized_passage for ans in non_empty)


def stable_question_id_hash(question_id: str) -> int:
    """Compute a deterministic 64-bit unsigned integer hash of `question_id` via SHA-256 (big-endian)."""
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_question_seed_rng(
    master_seed: int,
    draw_seed: int,
    question_id: str,
) -> np.random.Generator:
    """Create an isolated numpy Generator seeded with [master_seed, draw_seed, stable_hash(question_id)]."""
    qid_hash = stable_question_id_hash(question_id)
    return np.random.default_rng([int(master_seed), int(draw_seed), int(qid_hash)])


def is_within_length_tolerance(
    candidate_len: int,
    target_len: int,
    tolerance: float = 0.20,
) -> bool:
    """Check whether `candidate_len` is within `+/- tolerance` of `target_len`.

    For the standard 20% tolerance (`tolerance == 0.20`), uses exact integer arithmetic
    (`4 * target_len <= 5 * candidate_len <= 6 * target_len`) to prevent float rounding drift.
    """
    if abs(tolerance - 0.20) < 1e-12:
        return (4 * target_len) <= (5 * candidate_len) <= (6 * target_len)
    lower = (1.0 - tolerance) * target_len
    upper = (1.0 + tolerance) * target_len
    return (lower - 1e-9) <= candidate_len <= (upper + 1e-9)


def gold_insertion_index(p: float, n: int) -> int:
    """Compute the 0-indexed position `round(p * n)` among `n + 1` slots where gold is inserted."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    idx = int(round(p * n))
    return max(0, min(n, idx))


def assemble_condition_passages(
    gold_paragraph_id: list[int],
    distractor_paragraph_ids: list[list[int]],
    n: int,
    p: float,
) -> list[list[int]]:
    """Build the ordered list of `n + 1` paragraph IDs for a condition of size `n`.

    Takes the first `n` distractors (nested prefix) and inserts `gold_paragraph_id` at
    index `round(p * n)`.
    """
    if n < 0 or n > len(distractor_paragraph_ids):
        raise ValueError(
            f"Requested n={n} distractors, but distractor list has length {len(distractor_paragraph_ids)}"
        )
    prefix = [list(pid) for pid in distractor_paragraph_ids[:n]]
    insert_idx = gold_insertion_index(p, n)
    prefix.insert(insert_idx, list(gold_paragraph_id))
    return prefix


def load_hf_token_length_fn(tokenizer_path: Path) -> Callable[[str], int]:
    """Load a local HuggingFace tokenizer offline and return a token-count function (`add_special_tokens=False`)."""
    assert_no_kmp_duplicate_lib_ok()
    os.environ["HF_HUB_OFFLINE"] = "1"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")

    tokenizer_json = tokenizer_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer

        fast_tok = Tokenizer.from_file(str(tokenizer_json))
        fast_tok.no_truncation()
        fast_tok.no_padding()

        def count_tokens_fast(text: str) -> int:
            return len(fast_tok.encode(text, add_special_tokens=False).ids)

        return count_tokens_fast

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        use_fast=True,
    )

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


def load_dev_paragraph_pool(
    raw_path: Path,
    qwen3_len_fn: Callable[[str], int],
    llama3_len_fn: Callable[[str], int],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    """Load all unique paragraphs from SQuAD 2.0 dev and precompute normalized text and token lengths."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing SQuAD 2.0 dev file: {raw_path}")

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    articles = payload.get("data", [])

    pool_list: list[dict[str, Any]] = []
    by_pid: dict[tuple[int, int], dict[str, Any]] = {}
    seen_texts: set[str] = set()

    for art_idx, article in enumerate(articles):
        title = article["title"]
        for para_idx, para in enumerate(article["paragraphs"]):
            context = para["context"]
            if context in seen_texts:
                continue
            seen_texts.add(context)
            pid_tuple = (int(art_idx), int(para_idx))
            entry = {
                "pool_index": len(pool_list),
                "paragraph_id": [int(art_idx), int(para_idx)],
                "article_index": int(art_idx),
                "paragraph_index": int(para_idx),
                "article_title": title,
                "context": context,
                "normalized_context": normalize_squad(context),
                "qwen3_token_length": int(qwen3_len_fn(context)),
                "llama3_token_length": int(llama3_len_fn(context)),
            }
            pool_list.append(entry)
            by_pid[pid_tuple] = entry

    return pool_list, by_pid


def build_distractor_inputs(
    prereg_path: Path = DEFAULT_PREREG_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    sampling_dir: Path = DEFAULT_SAMPLING_DIR,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    out_dir: Path = DEFAULT_INPUTS_DIR,
    qwen3_len_fn: Callable[[str], int] | None = None,
    llama3_len_fn: Callable[[str], int] | None = None,
    qwen3_tokenizer_path: Path = DEFAULT_QWEN3_TOKENIZER_PATH,
    llama3_tokenizer_path: Path = DEFAULT_LLAMA3_TOKENIZER_PATH,
    strict: bool = False,
) -> dict[str, Any]:
    """Build distractor sets, prompt-input JSONL files, and `data/inputs/build_report.md`."""
    assert_no_kmp_duplicate_lib_ok()

    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    design = prereg["design"]
    rng_cfg = prereg["rng"]

    pool_k = int(design["hard_negative_pool_k"])
    n_levels = [int(x) for x in design["n_levels"] if int(x) > 0]
    max_n = max(n_levels)
    tolerance = float(design["length_match_tolerance"])
    master_seed = int(rng_cfg["master_seed"])
    draw_seeds = [int(s) for s in rng_cfg["draw_seeds"]]

    if qwen3_len_fn is None:
        qwen3_len_fn = load_hf_token_length_fn(qwen3_tokenizer_path)
    if llama3_len_fn is None:
        llama3_len_fn = load_hf_token_length_fn(llama3_tokenizer_path)

    pool_list, pool_by_pid = load_dev_paragraph_pool(
        raw_path=raw_path,
        qwen3_len_fn=qwen3_len_fn,
        llama3_len_fn=llama3_len_fn,
    )

    # Load candidates map: question_id -> candidate record
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing hard candidates file: {candidates_path}")
    candidates_by_qid: dict[str, dict[str, Any]] = {}
    for line in candidates_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        candidates_by_qid[rec["question_id"]] = rec

    # Load sampling splits
    split_questions: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLITS:
        split_file = sampling_dir / f"{split_name}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing sampling split file: {split_file}")
        split_questions[split_name] = json.loads(split_file.read_text(encoding="utf-8"))

    # Stage 1 & 2: Leak filter and hard pool construction per question
    insufficient_hard_pool_questions: list[dict[str, Any]] = []
    failed_random_slots: list[dict[str, Any]] = []

    question_contexts: dict[str, dict[str, Any]] = {}
    hard_top30_rejection_counts: list[int] = []
    hard_pool_k_scan_rejection_counts: list[int] = []
    pool_leak_rejection_counts: list[int] = []

    for split_name in SPLITS:
        for q in split_questions[split_name]:
            qid = q["question_id"]
            gold_pid = [int(q["paragraph_id"][0]), int(q["paragraph_id"][1])]
            gold_art_idx = int(q["article_index"])
            gold_answers = list(q["gold_answers"])
            norm_gold_answers = sorted({normalize_squad(a) for a in gold_answers})

            if qid not in candidates_by_qid:
                raise KeyError(f"Question {qid} ({split_name}) missing from {candidates_path}")
            cand_rec = candidates_by_qid[qid]
            ranked_pids = [[int(p[0]), int(p[1])] for p in cand_rec["ranked_paragraph_ids"]]
            ranked_scores = [float(s) for s in cand_rec["scores"]]

            non_rejected_pids: list[list[int]] = []
            non_rejected_scores: list[float] = []
            rejected_in_top30 = 0
            rejected_before_pool_k = 0

            for pid, score in zip(ranked_pids, ranked_scores):
                pid_tuple = (pid[0], pid[1])
                if pid == gold_pid:
                    raise ValueError(f"Gold paragraph {gold_pid} present in candidates for {qid}")
                if pid_tuple not in pool_by_pid:
                    raise KeyError(f"Candidate paragraph {pid} not found in dev pool")
                para_entry = pool_by_pid[pid_tuple]
                leaked = has_answer_leak(para_entry["normalized_context"], norm_gold_answers)
                if leaked:
                    rejected_in_top30 += 1
                    if len(non_rejected_pids) < pool_k:
                        rejected_before_pool_k += 1
                else:
                    non_rejected_pids.append(pid)
                    non_rejected_scores.append(score)

            hard_top30_rejection_counts.append(rejected_in_top30)
            hard_pool_k_scan_rejection_counts.append(rejected_before_pool_k)

            # Also pre-filter non-gold-article pool for random control eligibility (excluding length)
            non_gold_art_total = 0
            non_gold_art_leak_rejected = 0
            eligible_random_base: list[dict[str, Any]] = []
            for para_entry in pool_list:
                if para_entry["article_index"] == gold_art_idx:
                    continue
                non_gold_art_total += 1
                if has_answer_leak(para_entry["normalized_context"], norm_gold_answers):
                    non_gold_art_leak_rejected += 1
                else:
                    eligible_random_base.append(para_entry)

            pool_leak_rejection_counts.append(non_gold_art_leak_rejected)

            if len(non_rejected_pids) < pool_k:
                insufficient_hard_pool_questions.append(
                    {
                        "question_id": qid,
                        "split": split_name,
                        "non_rejected_count": len(non_rejected_pids),
                        "rejected_in_top30": rejected_in_top30,
                        "question_text": q["question_text"],
                        "gold_answers": gold_answers,
                        "norm_gold_answers": norm_gold_answers,
                    }
                )

            hard_pool_pids = non_rejected_pids[:pool_k]
            hard_pool_scores = non_rejected_scores[:pool_k]

            question_contexts[qid] = {
                "question": q,
                "split": split_name,
                "gold_pid": gold_pid,
                "gold_art_idx": gold_art_idx,
                "gold_answers": gold_answers,
                "norm_gold_answers": norm_gold_answers,
                "hard_pool_pids": hard_pool_pids,
                "hard_pool_scores": hard_pool_scores,
                "rejected_in_top30": rejected_in_top30,
                "rejected_before_pool_k": rejected_before_pool_k,
                "non_rejected_in_top30": len(non_rejected_pids),
                "non_gold_art_total": non_gold_art_total,
                "non_gold_art_leak_rejected": non_gold_art_leak_rejected,
                "eligible_random_base": eligible_random_base,
            }

    # Stage 3, 4, 5: Sample hard negatives, length-matched random controls, and gold position p
    rows_by_split_seed: dict[tuple[str, int], list[dict[str, Any]]] = {
        (split_name, seed): [] for split_name in SPLITS for seed in draw_seeds
    }

    qwen3_diffs_all: list[int] = []
    qwen3_abs_diffs_all: list[int] = []
    qwen3_rel_diffs_all: list[float] = []
    llama3_diffs_all: list[int] = []
    llama3_abs_diffs_all: list[int] = []
    llama3_rel_diffs_all: list[float] = []

    qwen3_diffs_by_split: dict[str, list[int]] = {s: [] for s in SPLITS}
    llama3_diffs_by_split: dict[str, list[int]] = {s: [] for s in SPLITS}

    for split_name in SPLITS:
        for q in split_questions[split_name]:
            qid = q["question_id"]
            qctx = question_contexts[qid]
            hard_pool_pids = qctx["hard_pool_pids"]
            if len(hard_pool_pids) < pool_k:
                continue

            gold_pid = qctx["gold_pid"]
            gold_entry = pool_by_pid[(gold_pid[0], gold_pid[1])]
            hard_pool_scores = qctx["hard_pool_scores"]
            eligible_random_base = qctx["eligible_random_base"]

            for seed in draw_seeds:
                rng = make_question_seed_rng(
                    master_seed=master_seed,
                    draw_seed=seed,
                    question_id=qid,
                )

                # 1. Sample ordered list of 8 hard negatives uniformly without replacement from the 20-candidate pool
                chosen_hard_pool_indices = rng.choice(
                    len(hard_pool_pids), size=max_n, replace=False
                ).tolist()
                hard_pids = [list(hard_pool_pids[idx]) for idx in chosen_hard_pool_indices]
                hard_scores = [float(hard_pool_scores[idx]) for idx in chosen_hard_pool_indices]

                # 2. Sample length-matched random control for each slot i in 0..max_n-1
                random_pids: list[list[int]] = []
                used_random_pids: set[tuple[int, int]] = set()
                slot_failed = False

                for slot_idx, h_pid in enumerate(hard_pids):
                    h_entry = pool_by_pid[(h_pid[0], h_pid[1])]
                    target_qwen_len = h_entry["qwen3_token_length"]

                    slot_candidates = [
                        p_entry
                        for p_entry in eligible_random_base
                        if (p_entry["paragraph_id"][0], p_entry["paragraph_id"][1])
                        not in used_random_pids
                        and is_within_length_tolerance(
                            p_entry["qwen3_token_length"],
                            target_qwen_len,
                            tolerance=tolerance,
                        )
                    ]

                    if not slot_candidates:
                        failed_random_slots.append(
                            {
                                "question_id": qid,
                                "split": split_name,
                                "seed": seed,
                                "slot": slot_idx,
                                "hard_paragraph_id": h_pid,
                                "target_qwen3_token_length": target_qwen_len,
                            }
                        )
                        slot_failed = True
                        break

                    chosen_slot_pos = int(rng.integers(0, len(slot_candidates)))
                    chosen_r_entry = slot_candidates[chosen_slot_pos]
                    r_pid = list(chosen_r_entry["paragraph_id"])
                    random_pids.append(r_pid)
                    used_random_pids.add((r_pid[0], r_pid[1]))

                if slot_failed:
                    # Exclude this (question, seed) from output per Spec 4
                    continue

                # 3. Draw fraction p in [0, 1) for gold position
                p_val = float(rng.random())
                gold_positions = {
                    str(n): gold_insertion_index(p_val, n) for n in n_levels
                }

                hard_qwen3_lens = [
                    pool_by_pid[(pid[0], pid[1])]["qwen3_token_length"]
                    for pid in hard_pids
                ]
                random_qwen3_lens = [
                    pool_by_pid[(pid[0], pid[1])]["qwen3_token_length"]
                    for pid in random_pids
                ]
                hard_llama3_lens = [
                    pool_by_pid[(pid[0], pid[1])]["llama3_token_length"]
                    for pid in hard_pids
                ]
                random_llama3_lens = [
                    pool_by_pid[(pid[0], pid[1])]["llama3_token_length"]
                    for pid in random_pids
                ]

                for h_q, r_q, h_l, r_l in zip(
                    hard_qwen3_lens,
                    random_qwen3_lens,
                    hard_llama3_lens,
                    random_llama3_lens,
                ):
                    diff_q = r_q - h_q
                    diff_l = r_l - h_l
                    qwen3_diffs_all.append(diff_q)
                    qwen3_abs_diffs_all.append(abs(diff_q))
                    qwen3_rel_diffs_all.append(abs(diff_q) / h_q if h_q > 0 else 0.0)
                    llama3_diffs_all.append(diff_l)
                    llama3_abs_diffs_all.append(abs(diff_l))
                    llama3_rel_diffs_all.append(abs(diff_l) / h_l if h_l > 0 else 0.0)
                    qwen3_diffs_by_split[split_name].append(diff_q)
                    llama3_diffs_by_split[split_name].append(diff_l)

                row = {
                    "question_id": qid,
                    "split": split_name,
                    "seed": seed,
                    "article_index": qctx["gold_art_idx"],
                    "article_title": q["article_title"],
                    "question_text": q["question_text"],
                    "gold_answers": qctx["gold_answers"],
                    "gold_paragraph_id": gold_pid,
                    "hard_pool_paragraph_ids": hard_pool_pids,
                    "hard_paragraph_ids": hard_pids,
                    "hard_candidate_scores": hard_scores,
                    "random_paragraph_ids": random_pids,
                    "p": p_val,
                    "gold_positions": gold_positions,
                    "token_lengths": {
                        "qwen3": {
                            "gold": gold_entry["qwen3_token_length"],
                            "hard": hard_qwen3_lens,
                            "random": random_qwen3_lens,
                        },
                        "llama3": {
                            "gold": gold_entry["llama3_token_length"],
                            "hard": hard_llama3_lens,
                            "random": random_llama3_lens,
                        },
                    },
                    "rejection_counts": {
                        "hard_top30_leak_rejected": qctx["rejected_in_top30"],
                        "hard_before_pool_k_leak_rejected": qctx["rejected_before_pool_k"],
                        "hard_top30_non_rejected": qctx["non_rejected_in_top30"],
                        "random_pool_non_gold_article_total": qctx["non_gold_art_total"],
                        "random_pool_leak_rejected": qctx["non_gold_art_leak_rejected"],
                    },
                }
                rows_by_split_seed[(split_name, seed)].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_files_info: dict[str, dict[str, Any]] = {}

    for split_name in SPLITS:
        for seed in draw_seeds:
            key = f"{split_name}_seed{seed}"
            file_path = out_dir / f"{key}.jsonl"
            rows = rows_by_split_seed[(split_name, seed)]
            with file_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_files_info[key] = {
                "path": f"data/inputs/{file_path.name}",
                "count": len(rows),
                "sha256": sha256_file(file_path),
            }

    summary = {
        "total_questions": sum(len(v) for v in split_questions.values()),
        "split_counts": {s: len(split_questions[s]) for s in SPLITS},
        "draw_seeds": draw_seeds,
        "master_seed": master_seed,
        "pool_k": pool_k,
        "tolerance": tolerance,
        "insufficient_hard_pool_questions": insufficient_hard_pool_questions,
        "failed_random_slots": failed_random_slots,
        "rejection_stats": {
            "hard_top30_total_questions_with_rejections": sum(
                1 for c in hard_top30_rejection_counts if c > 0
            ),
            "hard_top30_rejected_sum": int(sum(hard_top30_rejection_counts)),
            "hard_top30_rejected_mean": round(
                float(statistics.mean(hard_top30_rejection_counts)), 4
            )
            if hard_top30_rejection_counts
            else 0.0,
            "hard_top30_rejected_median": float(
                statistics.median(hard_top30_rejection_counts)
            )
            if hard_top30_rejection_counts
            else 0.0,
            "hard_top30_rejected_min": int(min(hard_top30_rejection_counts))
            if hard_top30_rejection_counts
            else 0,
            "hard_top30_rejected_max": int(max(hard_top30_rejection_counts))
            if hard_top30_rejection_counts
            else 0,
            "hard_before_pool_k_rejected_sum": int(
                sum(hard_pool_k_scan_rejection_counts)
            ),
            "hard_before_pool_k_rejected_max": int(
                max(hard_pool_k_scan_rejection_counts)
            )
            if hard_pool_k_scan_rejection_counts
            else 0,
            "non_gold_pool_rejected_mean": round(
                float(statistics.mean(pool_leak_rejection_counts)), 4
            )
            if pool_leak_rejection_counts
            else 0.0,
            "non_gold_pool_rejected_max": int(max(pool_leak_rejection_counts))
            if pool_leak_rejection_counts
            else 0,
        },
        "length_matching_stats": _compute_length_stats(
            rows_by_split_seed=rows_by_split_seed,
            qwen3_diffs_all=qwen3_diffs_all,
            qwen3_abs_diffs_all=qwen3_abs_diffs_all,
            qwen3_rel_diffs_all=qwen3_rel_diffs_all,
            llama3_diffs_all=llama3_diffs_all,
            llama3_abs_diffs_all=llama3_abs_diffs_all,
            llama3_rel_diffs_all=llama3_rel_diffs_all,
            qwen3_diffs_by_split=qwen3_diffs_by_split,
            llama3_diffs_by_split=llama3_diffs_by_split,
        ),
        "output_files": output_files_info,
    }

    report_path = out_dir / "build_report.md"
    report_path.write_text(
        render_build_report_markdown(summary, question_contexts),
        encoding="utf-8",
    )

    if strict and (insufficient_hard_pool_questions or failed_random_slots):
        raise RuntimeError(
            f"Build halted per spec: insufficient_hard_pool={len(insufficient_hard_pool_questions)}, "
            f"failed_random_slots={len(failed_random_slots)}. See {report_path}."
        )

    return summary


def _compute_length_stats(
    rows_by_split_seed: dict[tuple[str, int], list[dict[str, Any]]],
    qwen3_diffs_all: list[int],
    qwen3_abs_diffs_all: list[int],
    qwen3_rel_diffs_all: list[float],
    llama3_diffs_all: list[int],
    llama3_abs_diffs_all: list[int],
    llama3_rel_diffs_all: list[float],
    qwen3_diffs_by_split: dict[str, list[int]],
    llama3_diffs_by_split: dict[str, list[int]],
) -> dict[str, Any]:
    all_rows = [r for rows in rows_by_split_seed.values() for r in rows]
    hard_qwen3 = [
        length for r in all_rows for length in r["token_lengths"]["qwen3"]["hard"]
    ]
    rand_qwen3 = [
        length for r in all_rows for length in r["token_lengths"]["qwen3"]["random"]
    ]
    hard_llama3 = [
        length for r in all_rows for length in r["token_lengths"]["llama3"]["hard"]
    ]
    rand_llama3 = [
        length for r in all_rows for length in r["token_lengths"]["llama3"]["random"]
    ]

    if not qwen3_diffs_all:
        return {"total_pairs": 0}

    return {
        "total_pairs": len(qwen3_diffs_all),
        "qwen3": {
            "hard_mean": round(float(statistics.mean(hard_qwen3)), 4),
            "random_mean": round(float(statistics.mean(rand_qwen3)), 4),
            "signed_diff_mean_random_minus_hard": round(
                float(statistics.mean(qwen3_diffs_all)), 4
            ),
            "signed_diff_sd": round(float(statistics.stdev(qwen3_diffs_all)), 4)
            if len(qwen3_diffs_all) > 1
            else 0.0,
            "signed_diff_median": float(statistics.median(qwen3_diffs_all)),
            "abs_diff_mean": round(float(statistics.mean(qwen3_abs_diffs_all)), 4),
            "abs_diff_max": int(max(qwen3_abs_diffs_all)),
            "rel_diff_mean": round(float(statistics.mean(qwen3_rel_diffs_all)), 6),
            "rel_diff_max": round(float(max(qwen3_rel_diffs_all)), 6),
            "by_split_signed_diff_mean": {
                s: round(float(statistics.mean(vals)), 4) if vals else 0.0
                for s, vals in qwen3_diffs_by_split.items()
            },
        },
        "llama3": {
            "hard_mean": round(float(statistics.mean(hard_llama3)), 4),
            "random_mean": round(float(statistics.mean(rand_llama3)), 4),
            "signed_diff_mean_random_minus_hard": round(
                float(statistics.mean(llama3_diffs_all)), 4
            ),
            "signed_diff_sd": round(float(statistics.stdev(llama3_diffs_all)), 4)
            if len(llama3_diffs_all) > 1
            else 0.0,
            "signed_diff_median": float(statistics.median(llama3_diffs_all)),
            "abs_diff_mean": round(float(statistics.mean(llama3_abs_diffs_all)), 4),
            "abs_diff_max": int(max(llama3_abs_diffs_all)),
            "rel_diff_mean": round(float(statistics.mean(llama3_rel_diffs_all)), 6),
            "rel_diff_max": round(float(max(llama3_rel_diffs_all)), 6),
            "by_split_signed_diff_mean": {
                s: round(float(statistics.mean(vals)), 4) if vals else 0.0
                for s, vals in llama3_diffs_by_split.items()
            },
        },
    }


def render_build_report_markdown(
    summary: dict[str, Any],
    question_contexts: dict[str, dict[str, Any]],
) -> str:
    """Format `data/inputs/build_report.md` with rejection counts, length stats, and failure checks."""
    rej = summary["rejection_stats"]
    lstats = summary["length_matching_stats"]
    insuff = summary["insufficient_hard_pool_questions"]
    failed_slots = summary["failed_random_slots"]

    lines: list[str] = [
        "# Distractor & Prompt-Input Build Report",
        "",
        "## 1. Configuration & Summary",
        f"- **Master seed**: `{summary['master_seed']}`",
        f"- **Draw seeds**: `{summary['draw_seeds']}`",
        f"- **Hard-negative pool size (`k`)**: `{summary['pool_k']}` (from top-30 dense candidates)",
        f"- **Length-match tolerance**: `+/- {summary['tolerance'] * 100:.0f}%` (primary: `Qwen3-8B` tokenizer; secondary check: `Llama-3.1-8B-Instruct` tokenizer)",
        f"- **Questions processed**: `{summary['total_questions']}` (`main={summary['split_counts']['main']}`, `pilot={summary['split_counts']['pilot']}`, `reserve={summary['split_counts']['reserve']}`)",
        f"- **Questions with `< {summary['pool_k']}` non-rejected hard candidates**: `{len(insuff)}`",
        f"- **Failed random length-matching slots**: `{len(failed_slots)}`",
        "",
        "## 2. Answer-Leak Rejection Counts",
        "Official SQuAD v2.0 normalization is applied to every candidate passage and every gold answer; a passage is rejected if any normalized gold answer appears as a substring.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Questions with >= 1 rejected candidate in top-30 | {rej['hard_top30_total_questions_with_rejections']} / {summary['total_questions']} |",
        f"| Total top-30 hard candidates rejected (across {summary['total_questions']} questions) | {rej['hard_top30_rejected_sum']} / {summary['total_questions'] * 30} |",
        f"| Top-30 rejections per question (mean / median / min / max) | {rej['hard_top30_rejected_mean']:.4f} / {rej['hard_top30_rejected_median']:.1f} / {rej['hard_top30_rejected_min']} / {rej['hard_top30_rejected_max']} |",
        f"| Rejections encountered before reaching k={summary['pool_k']} hard pool (sum / max) | {rej['hard_before_pool_k_rejected_sum']} / {rej['hard_before_pool_k_rejected_max']} |",
        f"| Non-gold-article dev pool rejections per question (mean / max) | {rej['non_gold_pool_rejected_mean']:.4f} / {rej['non_gold_pool_rejected_max']} |",
        "",
        "### Distribution of Top-30 Hard-Candidate Rejections per Question",
        "| Rejected Count (out of 30) | Remaining Non-Rejected | Number of Questions |",
        "| --- | --- | --- |",
    ]

    from collections import Counter

    rej_dist = Counter(
        qctx["rejected_in_top30"] for qctx in question_contexts.values()
    )
    for r_count in sorted(rej_dist):
        lines.append(
            f"| {r_count} | {30 - r_count} | {rej_dist[r_count]} |"
        )

    lines.extend(
        [
            "",
            "## 3. Questions with Fewer Than 20 Hard-Pool Candidates",
        ]
    )
    if not insuff:
        lines.append(
            "None (0 questions had fewer than 20 non-rejected candidates in their top-30)."
        )
    else:
        lines.append(
            "| Question ID | Split | Non-Rejected | Rejected (of 30) | Normalized Gold Answers | Question Text |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in insuff:
            norm_ans_str = ", ".join(f"`{a}`" for a in item["norm_gold_answers"])
            q_text_escaped = item["question_text"].replace("|", "\\|")
            lines.append(
                f"| `{item['question_id']}` | `{item['split']}` | {item['non_rejected_count']} | {item['rejected_in_top30']} | {norm_ans_str} | {q_text_escaped} |"
            )

    lines.extend(
        [
            "",
            "## 4. Failed Random Control Slots",
        ]
    )
    if not failed_slots:
        lines.append(
            "None (0 failed slots across all questions, seeds, and positions `0..7`)."
        )
    else:
        lines.append("| Question ID | Split | Seed | Slot | Hard Paragraph ID | Target Qwen3 Length |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in failed_slots:
            lines.append(
                f"| `{item['question_id']}` | `{item['split']}` | {item['seed']} | {item['slot']} | `{item['hard_paragraph_id']}` | {item['target_qwen3_token_length']} |"
            )

    lines.extend(
        [
            "",
            "## 5. Token-Length Matching Statistics",
        ]
    )
    if lstats.get("total_pairs", 0) > 0:
        q_s = lstats["qwen3"]
        l_s = lstats["llama3"]
        lines.extend(
            [
                f"Evaluated across `{lstats['total_pairs']}` matched `(hard, random)` distractor pairs (`590 questions x 3 seeds x 8 slots = 14,160 pairs` when complete).",
                "",
                "| Metric | Primary: Qwen3-8B Tokenizer | Secondary Check: Llama-3.1-8B Tokenizer |",
                "| --- | --- | --- |",
                f"| Mean hard-negative token length | {q_s['hard_mean']:.4f} | {l_s['hard_mean']:.4f} |",
                f"| Mean random-control token length | {q_s['random_mean']:.4f} | {l_s['random_mean']:.4f} |",
                f"| Mean signed difference (`random - hard`) | {q_s['signed_diff_mean_random_minus_hard']:+.4f} | {l_s['signed_diff_mean_random_minus_hard']:+.4f} |",
                f"| SD of signed difference (`random - hard`) | {q_s['signed_diff_sd']:.4f} | {l_s['signed_diff_sd']:.4f} |",
                f"| Median signed difference (`random - hard`) | {q_s['signed_diff_median']:+.1f} | {l_s['signed_diff_median']:+.1f} |",
                f"| Mean absolute difference (`|random - hard|`) | {q_s['abs_diff_mean']:.4f} | {l_s['abs_diff_mean']:.4f} |",
                f"| Max absolute difference (`|random - hard|`) | {q_s['abs_diff_max']} | {l_s['abs_diff_max']} |",
                f"| Mean relative difference (`|random - hard| / hard`) | {q_s['rel_diff_mean'] * 100:.2f}% | {l_s['rel_diff_mean'] * 100:.2f}% |",
                f"| Max relative difference (`|random - hard| / hard`) | {q_s['rel_diff_max'] * 100:.2f}% | {l_s['rel_diff_max'] * 100:.2f}% |",
                f"| Mean signed diff (`main` split) | {q_s['by_split_signed_diff_mean']['main']:+.4f} | {l_s['by_split_signed_diff_mean']['main']:+.4f} |",
                f"| Mean signed diff (`pilot` split) | {q_s['by_split_signed_diff_mean']['pilot']:+.4f} | {l_s['by_split_signed_diff_mean']['pilot']:+.4f} |",
                f"| Mean signed diff (`reserve` split) | {q_s['by_split_signed_diff_mean']['reserve']:+.4f} | {l_s['by_split_signed_diff_mean']['reserve']:+.4f} |",
            ]
        )

    lines.extend(
        [
            "",
            "## 6. Generated Output Files",
            "| Split & Seed | Path | Row Count | SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key, info in summary["output_files"].items():
        lines.append(
            f"| `{key}` | `{info['path']}` | {info['count']} | `{info['sha256']}` |"
        )

    return "\n".join(lines) + "\n"
