"""Tests for distractor construction, leak filtering, length matching, and prompt-input files (Step p2c)."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"

import scripts.embed_dev_pool  # noqa: F401, E402
import src.omp_guard  # noqa: F401, E402
from src.distractors import (  # noqa: E402
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_INPUTS_DIR,
    DEFAULT_LLAMA3_TOKENIZER_PATH,
    DEFAULT_PREREG_PATH,
    DEFAULT_QWEN3_TOKENIZER_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_SAMPLING_DIR,
    SPLITS,
    assemble_condition_passages,
    assert_no_kmp_duplicate_lib_ok,
    build_distractor_inputs,
    gold_insertion_index,
    has_answer_leak,
    is_within_length_tolerance,
    load_dev_paragraph_pool,
    load_hf_token_length_fn,
    make_question_seed_rng,
    normalize_squad,
    sha256_file,
)

requires_raw_squad = pytest.mark.skipif(
    not DEFAULT_RAW_PATH.exists(),
    reason=f"Raw SQuAD 2.0 file not present at {DEFAULT_RAW_PATH}; skipping raw-data tests.",
)


def _check_omp_guard_before_transformers(py_path: Path) -> None:
    """Verify via AST that src.omp_guard is imported at top-level before any torch/transformers import."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    saw_omp_guard = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("src.omp_guard", "omp_guard"):
                    saw_omp_guard = True
                if alias.name.split(".")[0] in ("torch", "transformers", "sentence_transformers"):
                    assert saw_omp_guard, f"{py_path} imports {alias.name} before omp_guard"
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in ("src.omp_guard", "omp_guard"):
                saw_omp_guard = True
            if node.module.split(".")[0] in ("torch", "transformers", "sentence_transformers"):
                assert saw_omp_guard, f"{py_path} imports from {node.module} before omp_guard"
    assert saw_omp_guard, f"{py_path} must import src.omp_guard at top level"


def test_hard_rule_11_omp_guard_and_kmp_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify omp_guard precedes transformers imports and KMP_DUPLICATE_LIB_OK is rejected."""
    import src.distractors as distractors_mod

    _check_omp_guard_before_transformers(REPO_ROOT / "src" / "distractors.py")
    _check_omp_guard_before_transformers(REPO_ROOT / "scripts" / "build_inputs.py")

    monkeypatch.setattr(distractors_mod, "_INITIAL_KMP_DUPLICATE_LIB_OK", "TRUE")
    with pytest.raises(RuntimeError, match="KMP_DUPLICATE_LIB_OK"):
        assert_no_kmp_duplicate_lib_ok()


def test_squad_normalization_and_leak_filter_unit() -> None:
    """Unit test official SQuAD normalization and answer-leak substring rejection."""
    assert normalize_squad("The   Quick, Brown-Fox! (A test)") == "quick brownfox test"
    assert normalize_squad("Maciot de Bethencourt.") == "maciot de bethencourt"

    norm_gold = [normalize_squad("Maciot de Bethencourt"), normalize_squad("Bethencourt")]
    leaking_passage = normalize_squad(
        "In 1418, Maciot de Bethencourt sold the rights to the islands."
    )
    clean_passage = normalize_squad(
        "The Normans were descendants of Norse raiders who settled in Normandy."
    )

    assert has_answer_leak(leaking_passage, norm_gold) is True
    assert has_answer_leak(clean_passage, norm_gold) is False


def test_rng_determinism_and_seed_sensitivity_unit() -> None:
    """Verify per-(question_id, draw_seed) RNG is deterministic and differs across seeds/questions."""
    rng_a1 = make_question_seed_rng(4207, 0, "56de49a8cffd8e1900b4b7a9")
    rng_a2 = make_question_seed_rng(4207, 0, "56de49a8cffd8e1900b4b7a9")
    rng_b = make_question_seed_rng(4207, 1, "56de49a8cffd8e1900b4b7a9")
    rng_c = make_question_seed_rng(4207, 0, "56e1c720e3433e140042316e")

    draw_a1 = rng_a1.choice(20, size=8, replace=False).tolist()
    draw_a2 = rng_a2.choice(20, size=8, replace=False).tolist()
    draw_b = rng_b.choice(20, size=8, replace=False).tolist()
    draw_c = rng_c.choice(20, size=8, replace=False).tolist()

    assert draw_a1 == draw_a2
    assert draw_a1 != draw_b
    assert draw_a1 != draw_c


def test_gold_insertion_index_and_nested_prefixes_unit() -> None:
    """Unit test round(p * n) gold insertion and nested prefix preservation."""
    gold_pid = [0, 5]
    distractors = [[1, i] for i in range(8)]
    p = 0.37

    for n in (1, 2, 4, 8):
        assembled = assemble_condition_passages(gold_pid, distractors, n=n, p=p)
        expected_idx = round(p * n)
        assert len(assembled) == n + 1
        assert assembled[expected_idx] == gold_pid
        without_gold = [pid for pid in assembled if pid != gold_pid]
        assert without_gold == distractors[:n]

    # Verify nested prefixes across n = 1, 2, 4, 8
    assert distractors[:1] == distractors[:2][:1]
    assert distractors[:2] == distractors[:4][:2]
    assert distractors[:4] == distractors[:8][:4]


def test_synthetic_failure_modes_stop_and_report(tmp_path: Path) -> None:
    """Verify builder records and stops when <20 hard candidates remain or a random slot has 0 matches."""
    prereg_path = DEFAULT_PREREG_PATH
    raw_path = tmp_path / "raw.json"
    sampling_dir = tmp_path / "sampling"
    sampling_dir.mkdir(parents=True)
    candidates_path = tmp_path / "hard_candidates.jsonl"
    out_dir = tmp_path / "inputs"

    # Build a synthetic raw dataset where most top-30 candidates leak the gold answer "unicorn"
    articles = [
        {
            "title": "Art0",
            "paragraphs": [
                {"context": "Gold context about unicorn.", "qas": []},
            ]
            + [
                {"context": f"Leaking candidate {i} mentioning unicorn.", "qas": []}
                for i in range(1, 20)
            ]
            + [
                {"context": f"Clean candidate {i} with safe words only.", "qas": []}
                for i in range(20, 32)
            ],
        },
        {
            "title": "Art1",
            "paragraphs": [
                {"context": f"Other article paragraph {j} with safe words only.", "qas": []}
                for j in range(20)
            ],
        },
    ]
    raw_path.write_text(json.dumps({"data": articles}), encoding="utf-8")

    q_rec = {
        "question_id": "q_leak_heavy",
        "article_index": 0,
        "article_title": "Art0",
        "paragraph_id": [0, 0],
        "paragraph_index": 0,
        "question_text": "What animal?",
        "gold_answers": ["unicorn"],
    }
    (sampling_dir / "main.json").write_text(json.dumps([q_rec]), encoding="utf-8")
    (sampling_dir / "pilot.json").write_text(json.dumps([]), encoding="utf-8")
    (sampling_dir / "reserve.json").write_text(json.dumps([]), encoding="utf-8")

    # 19 leaking + 11 clean = only 11 non-rejected (< 20 required)
    cand_row = {
        "question_id": "q_leak_heavy",
        "split": "main",
        "gold_paragraph_id": [0, 0],
        "ranked_paragraph_ids": [[0, i] for i in range(1, 31)],
        "scores": [1.0 - i * 0.01 for i in range(30)],
    }
    candidates_path.write_text(json.dumps(cand_row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="insufficient_hard_pool=1"):
        build_distractor_inputs(
            prereg_path=prereg_path,
            raw_path=raw_path,
            sampling_dir=sampling_dir,
            candidates_path=candidates_path,
            out_dir=out_dir,
            qwen3_len_fn=lambda s: len(s.split()),
            llama3_len_fn=lambda s: len(s.split()),
            strict=True,
        )

    report_text = (out_dir / "build_report.md").read_text(encoding="utf-8")
    assert "q_leak_heavy" in report_text


@pytest.fixture(scope="module")
def built_inputs_environment() -> dict[str, Any]:
    """Ensure real distractor inputs are built and return loaded pool, questions, and rows."""
    if not DEFAULT_RAW_PATH.exists():
        pytest.skip(f"Missing raw SQuAD 2.0 file at {DEFAULT_RAW_PATH}")

    qwen3_fn = load_hf_token_length_fn(DEFAULT_QWEN3_TOKENIZER_PATH)
    llama3_fn = load_hf_token_length_fn(DEFAULT_LLAMA3_TOKENIZER_PATH)
    pool_list, pool_by_pid = load_dev_paragraph_pool(
        DEFAULT_RAW_PATH, qwen3_len_fn=qwen3_fn, llama3_len_fn=llama3_fn
    )

    expected_files = [
        DEFAULT_INPUTS_DIR / f"{split}_seed{seed}.jsonl"
        for split in SPLITS
        for seed in (0, 1, 2)
    ]
    if not all(p.exists() for p in expected_files) or not (
        DEFAULT_INPUTS_DIR / "build_report.md"
    ).exists():
        summary = build_distractor_inputs(
            qwen3_len_fn=qwen3_fn,
            llama3_len_fn=llama3_fn,
        )
    else:
        summary = {}

    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        for seed in (0, 1, 2):
            key = f"{split}_seed{seed}"
            fpath = DEFAULT_INPUTS_DIR / f"{key}.jsonl"
            rows_by_key[key] = [
                json.loads(line)
                for line in fpath.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    return {
        "pool_list": pool_list,
        "pool_by_pid": pool_by_pid,
        "rows_by_key": rows_by_key,
        "qwen3_fn": qwen3_fn,
        "llama3_fn": llama3_fn,
        "summary": summary,
    }


@requires_raw_squad
def test_all_paragraph_ids_exist_in_dev_pool_and_split_counts(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify every gold, hard-pool, hard, and random paragraph ID exists in the 1,204 dev pool."""
    pool_by_pid = built_inputs_environment["pool_by_pid"]
    rows_by_key = built_inputs_environment["rows_by_key"]
    assert len(pool_by_pid) == 1204

    expected_counts = {
        "main_seed0": 475,
        "main_seed1": 476,
        "main_seed2": 475,
        "pilot_seed0": 45,
        "pilot_seed1": 45,
        "pilot_seed2": 44,
        "reserve_seed0": 37,
        "reserve_seed1": 39,
        "reserve_seed2": 39,
    }
    for key, exp_count in expected_counts.items():
        rows = rows_by_key[key]
        assert len(rows) == exp_count
        for row in rows:
            assert tuple(row["gold_paragraph_id"]) in pool_by_pid
            assert len(row["hard_pool_paragraph_ids"]) == 20
            for pid in row["hard_pool_paragraph_ids"]:
                assert tuple(pid) in pool_by_pid
            for pid in row["hard_paragraph_ids"]:
                assert tuple(pid) in pool_by_pid
            for pid in row["random_paragraph_ids"]:
                assert tuple(pid) in pool_by_pid


@requires_raw_squad
def test_gold_never_among_distractors(built_inputs_environment: dict[str, Any]) -> None:
    """Verify gold_paragraph_id never appears in hard_pool, hard_paragraph_ids, or random_paragraph_ids."""
    rows_by_key = built_inputs_environment["rows_by_key"]
    for key, rows in rows_by_key.items():
        for row in rows:
            gold_pid = row["gold_paragraph_id"]
            assert (
                gold_pid not in row["hard_pool_paragraph_ids"]
            ), f"Gold in hard_pool for {row['question_id']} ({key})"
            assert (
                gold_pid not in row["hard_paragraph_ids"]
            ), f"Gold in hard_paragraph_ids for {row['question_id']} ({key})"
            assert (
                gold_pid not in row["random_paragraph_ids"]
            ), f"Gold in random_paragraph_ids for {row['question_id']} ({key})"


@requires_raw_squad
def test_no_answer_leak_in_any_distractor(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify no hard-pool, hard, or random distractor contains any normalized gold answer substring."""
    pool_by_pid = built_inputs_environment["pool_by_pid"]
    rows_by_key = built_inputs_environment["rows_by_key"]

    for key, rows in rows_by_key.items():
        for row in rows:
            norm_gold = [normalize_squad(a) for a in row["gold_answers"]]
            for list_name in (
                "hard_pool_paragraph_ids",
                "hard_paragraph_ids",
                "random_paragraph_ids",
            ):
                for pid in row[list_name]:
                    norm_ctx = pool_by_pid[tuple(pid)]["normalized_context"]
                    assert not has_answer_leak(
                        norm_ctx, norm_gold
                    ), f"Answer leak detected in {list_name} passage {pid} for {row['question_id']} ({key})"


@requires_raw_squad
def test_nested_prefix_property_and_no_duplicates(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify n=1 within n=2 within n=4 within n=8 in identical order without duplicate distractors."""
    rows_by_key = built_inputs_environment["rows_by_key"]
    for key, rows in rows_by_key.items():
        for row in rows:
            hard = row["hard_paragraph_ids"]
            rand = row["random_paragraph_ids"]
            pool_set = {tuple(p) for p in row["hard_pool_paragraph_ids"]}

            assert len(hard) == 8
            assert len(rand) == 8
            assert len({tuple(p) for p in hard}) == 8, f"Duplicate in hard list ({key})"
            assert len({tuple(p) for p in rand}) == 8, f"Duplicate in random list ({key})"
            assert all(tuple(p) in pool_set for p in hard)

            for seq in (hard, rand):
                p1, p2, p4, p8 = seq[:1], seq[:2], seq[:4], seq[:8]
                assert p2[:1] == p1
                assert p4[:2] == p2
                assert p8[:4] == p4


@requires_raw_squad
def test_length_tolerance_for_every_slot(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify every random passage is within +/-20% Qwen3 token length of its matched hard negative."""
    pool_by_pid = built_inputs_environment["pool_by_pid"]
    rows_by_key = built_inputs_environment["rows_by_key"]

    for key, rows in rows_by_key.items():
        for row in rows:
            hard_pids = row["hard_paragraph_ids"]
            rand_pids = row["random_paragraph_ids"]
            hard_q_lens = row["token_lengths"]["qwen3"]["hard"]
            rand_q_lens = row["token_lengths"]["qwen3"]["random"]

            assert len(hard_q_lens) == 8
            assert len(rand_q_lens) == 8

            for slot_idx in range(8):
                h_pid = tuple(hard_pids[slot_idx])
                r_pid = tuple(rand_pids[slot_idx])
                h_len = hard_q_lens[slot_idx]
                r_len = rand_q_lens[slot_idx]

                assert h_len == pool_by_pid[h_pid]["qwen3_token_length"]
                assert r_len == pool_by_pid[r_pid]["qwen3_token_length"]
                assert is_within_length_tolerance(r_len, h_len, tolerance=0.20), (
                    f"Slot {slot_idx} length tolerance violated for {row['question_id']} ({key}): "
                    f"hard={h_len}, random={r_len}"
                )


@requires_raw_squad
def test_random_passages_never_from_gold_article(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify no random distractor comes from the same article as the gold passage."""
    rows_by_key = built_inputs_environment["rows_by_key"]
    for key, rows in rows_by_key.items():
        for row in rows:
            gold_art = row["gold_paragraph_id"][0]
            assert gold_art == row["article_index"]
            for r_pid in row["random_paragraph_ids"]:
                assert (
                    r_pid[0] != gold_art
                ), f"Random passage {r_pid} from gold article {gold_art} in {row['question_id']} ({key})"


@requires_raw_squad
def test_gold_insertion_index_across_all_rows(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify p is in [0, 1) and gold_positions[n] == round(p * n) for every row and n in {1,2,4,8}."""
    rows_by_key = built_inputs_environment["rows_by_key"]
    for key, rows in rows_by_key.items():
        for row in rows:
            p = row["p"]
            assert 0.0 <= p < 1.0
            gold_pid = row["gold_paragraph_id"]
            for n in (1, 2, 4, 8):
                expected_pos = gold_insertion_index(p, n)
                assert expected_pos == round(p * n)
                assert 0 <= expected_pos <= n
                assert row["gold_positions"][str(n)] == expected_pos

                assembled_hard = assemble_condition_passages(
                    gold_pid, row["hard_paragraph_ids"], n=n, p=p
                )
                assembled_rand = assemble_condition_passages(
                    gold_pid, row["random_paragraph_ids"], n=n, p=p
                )
                assert assembled_hard[expected_pos] == gold_pid
                assert assembled_rand[expected_pos] == gold_pid


@requires_raw_squad
def test_different_seeds_give_different_lists(
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify seed0, seed1, and seed2 produce distinct distractor draws and p values."""
    rows_by_key = built_inputs_environment["rows_by_key"]
    for split in SPLITS:
        r0 = rows_by_key[f"{split}_seed0"]
        r1 = rows_by_key[f"{split}_seed1"]
        r2 = rows_by_key[f"{split}_seed2"]

        hard_0 = [r["hard_paragraph_ids"] for r in r0]
        hard_1 = [r["hard_paragraph_ids"] for r in r1]
        hard_2 = [r["hard_paragraph_ids"] for r in r2]

        rand_0 = [r["random_paragraph_ids"] for r in r0]
        rand_1 = [r["random_paragraph_ids"] for r in r1]
        rand_2 = [r["random_paragraph_ids"] for r in r2]

        p_0 = [r["p"] for r in r0]
        p_1 = [r["p"] for r in r1]
        p_2 = [r["p"] for r in r2]

        assert hard_0 != hard_1 and hard_1 != hard_2 and hard_0 != hard_2
        assert rand_0 != rand_1 and rand_1 != rand_2 and rand_0 != rand_2
        assert p_0 != p_1 and p_1 != p_2 and p_0 != p_2


@requires_raw_squad
def test_determinism_two_runs_identical_hashes(
    tmp_path: Path,
    built_inputs_environment: dict[str, Any],
) -> None:
    """Verify running build_distractor_inputs a second time produces identical SHA-256 hashes."""
    out_dir_2 = tmp_path / "inputs_run2"
    summary_2 = build_distractor_inputs(
        out_dir=out_dir_2,
        qwen3_len_fn=built_inputs_environment["qwen3_fn"],
        llama3_len_fn=built_inputs_environment["llama3_fn"],
    )

    for split in SPLITS:
        for seed in (0, 1, 2):
            key = f"{split}_seed{seed}"
            orig_file = DEFAULT_INPUTS_DIR / f"{key}.jsonl"
            new_file = out_dir_2 / f"{key}.jsonl"
            assert sha256_file(new_file) == sha256_file(orig_file)
            assert summary_2["output_files"][key]["sha256"] == sha256_file(orig_file)

    assert sha256_file(out_dir_2 / "build_report.md") == sha256_file(
        DEFAULT_INPUTS_DIR / "build_report.md"
    )
