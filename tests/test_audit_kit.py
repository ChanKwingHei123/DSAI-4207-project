"""Unit and integration tests for the blind ambiguity-audit kit (Step p2d)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_agreement import (  # noqa: E402
    compute_cohens_kappa,
    evaluate_audit_agreement,
)
from scripts.build_audit_kit import (  # noqa: E402
    AUDIT_SAMPLE_SIZE,
    CSV_COLUMNS,
    DEFAULT_AUDIT_DIR,
    DEFAULT_MAIN_SAMPLING_PATH,
    DEFAULT_MAIN_SEED0_PATH,
    DEFAULT_PREREG_PATH,
    DEFAULT_RAW_PATH,
    build_audit_kit,
    format_gold_answers,
    load_paragraph_texts,
)
from scripts.make_sampling_frame import largest_remainder_allocate, sha256_file  # noqa: E402

requires_raw_squad = pytest.mark.skipif(
    not DEFAULT_RAW_PATH.exists(),
    reason=f"Raw SQuAD 2.0 dev file is absent at {DEFAULT_RAW_PATH}",
)

FORBIDDEN_MODEL_OR_SCORE_FIELDS = {
    "prediction",
    "model_output",
    "generated_text",
    "raw_output",
    "answer_pred",
    "exact_match",
    "token_f1",
    "f1",
    "em",
    "gold_score",
    "hard_candidate_scores",
    "scores",
    "rank",
    "ranks",
    "similarity",
    "article_title",
}


def test_audit_rubric_exact_text() -> None:
    """Verify docs/audit_rubric.md matches the required specification verbatim."""
    rubric_path = REPO_ROOT / "docs" / "audit_rubric.md"
    assert rubric_path.exists(), "docs/audit_rubric.md is missing"
    expected = (
        "Flag = 1 if any of the 8 passages validly answers the question under a "
        "reasonable alternative reading, or gives a correct answer string that "
        "matches none of the listed gold answers. Otherwise 0."
    )
    assert rubric_path.read_text(encoding="utf-8").strip() == expected


@requires_raw_squad
def test_no_model_output_or_score_or_title_fields_present() -> None:
    """Verify no model outputs, retrieval scores, ranks, similarity values, or article titles appear in audit files."""
    # Also verify no results/*.jsonl model outputs exist in the repo
    results_dir = REPO_ROOT / "results"
    if results_dir.exists():
        assert list(results_dir.glob("*.jsonl")) == [], (
            "No model outputs may exist anywhere during the blind ambiguity-audit task"
        )

    sample_path = DEFAULT_AUDIT_DIR / "audit_sample.json"
    csv_a_path = DEFAULT_AUDIT_DIR / "annotator_A.csv"
    csv_b_path = DEFAULT_AUDIT_DIR / "annotator_B.csv"
    key_path = DEFAULT_AUDIT_DIR / "key.json"

    for p in (sample_path, csv_a_path, csv_b_path, key_path):
        assert p.exists(), f"Expected audit file missing: {p}"

    sample_data = json.loads(sample_path.read_text(encoding="utf-8"))
    assert len(sample_data) == AUDIT_SAMPLE_SIZE
    for rec in sample_data:
        assert set(rec.keys()).isdisjoint(FORBIDDEN_MODEL_OR_SCORE_FIELDS)

    key_data = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(key_data) == AUDIT_SAMPLE_SIZE
    for qid, mapping in key_data.items():
        assert set(mapping.keys()) == {str(i) for i in range(1, 9)}
        for idx_str, pid in mapping.items():
            assert isinstance(pid, list) and len(pid) == 2
            assert all(isinstance(x, int) for x in pid)

    for csv_path in (csv_a_path, csv_b_path):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert tuple(reader.fieldnames) == CSV_COLUMNS
            assert set(reader.fieldnames).isdisjoint(FORBIDDEN_MODEL_OR_SCORE_FIELDS)
            rows = list(reader)
            assert len(rows) == AUDIT_SAMPLE_SIZE
            for row in rows:
                assert row["flagged"] == ""
                assert row["which_passage"] == ""
                assert row["note"] == ""


@requires_raw_squad
def test_both_sheets_hold_same_100_questions_different_row_order_and_match_key() -> None:
    """Verify annotator_A.csv and annotator_B.csv hold the same 100 questions in different row orders and match key.json & main_seed0.jsonl."""
    _, para_text_by_pid = load_paragraph_texts(DEFAULT_RAW_PATH)
    main_questions = json.loads(DEFAULT_MAIN_SAMPLING_PATH.read_text(encoding="utf-8"))
    seed0_by_qid = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in DEFAULT_MAIN_SEED0_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    sample_data = json.loads(
        (DEFAULT_AUDIT_DIR / "audit_sample.json").read_text(encoding="utf-8")
    )
    key_data = json.loads((DEFAULT_AUDIT_DIR / "key.json").read_text(encoding="utf-8"))

    with (DEFAULT_AUDIT_DIR / "annotator_A.csv").open("r", encoding="utf-8", newline="") as f:
        rows_a = list(csv.DictReader(f))
    with (DEFAULT_AUDIT_DIR / "annotator_B.csv").open("r", encoding="utf-8", newline="") as f:
        rows_b = list(csv.DictReader(f))

    assert len(rows_a) == AUDIT_SAMPLE_SIZE == 100
    assert len(rows_b) == AUDIT_SAMPLE_SIZE == 100

    qids_a = [r["question_id"] for r in rows_a]
    qids_b = [r["question_id"] for r in rows_b]
    qids_sample = [r["question_id"] for r in sample_data]

    assert len(set(qids_a)) == 100
    assert set(qids_a) == set(qids_b) == set(qids_sample) == set(key_data.keys())
    assert qids_a != qids_b, "annotator_A.csv and annotator_B.csv must have different row orders"

    # Verify all 100 sampled questions exist in main_seed0.jsonl and are stratified by article
    prereg = yaml.safe_load(DEFAULT_PREREG_PATH.read_text(encoding="utf-8"))
    master_seed = int(prereg["rng"]["master_seed"])
    rng = np.random.default_rng([master_seed, 1, 0])

    eligible_main = [q for q in main_questions if q["question_id"] in seed0_by_qid]
    num_articles = max(int(q["article_index"]) for q in eligible_main) + 1
    pools = [[] for _ in range(num_articles)]
    for q in eligible_main:
        pools[int(q["article_index"])].append(q)
    expected_allocs = largest_remainder_allocate([len(p) for p in pools], 100)
    actual_allocs = [
        sum(1 for r in sample_data if r["article_index"] == idx)
        for idx in range(num_articles)
    ]
    assert actual_allocs == expected_allocs

    # Verify RNG draw with [master_seed, 1, 0] matches audit_sample.json
    expected_qids: list[str] = []
    for pool, k in zip(pools, expected_allocs):
        if k > 0:
            chosen = set(rng.choice(len(pool), size=k, replace=False).tolist())
            expected_qids.extend(
                pool[i]["question_id"] for i in range(len(pool)) if i in chosen
            )
    assert qids_sample == expected_qids

    # Verify every row in A and B has identical content per question_id and matches key.json + seed0 hard negatives
    by_qid_a = {r["question_id"]: r for r in rows_a}
    by_qid_b = {r["question_id"]: r for r in rows_b}

    unpermuted_count = 0
    for qid in qids_sample:
        ra = by_qid_a[qid]
        rb = by_qid_b[qid]
        assert ra == rb, f"Row content mismatch between A and B for {qid}"

        s0 = seed0_by_qid[qid]
        assert ra["question"] == s0["question_text"]
        assert ra["gold_answers"] == format_gold_answers(s0["gold_answers"])
        gold_pid = (s0["gold_paragraph_id"][0], s0["gold_paragraph_id"][1])
        assert ra["gold_passage"] == para_text_by_pid[gold_pid]

        hard_pids = [tuple(pid) for pid in s0["hard_paragraph_ids"]]
        q_key = key_data[qid]
        key_pids = [tuple(q_key[str(i)]) for i in range(1, 9)]

        assert set(key_pids) == set(hard_pids)
        assert len(key_pids) == 8
        if key_pids == hard_pids:
            unpermuted_count += 1

        for i in range(1, 9):
            expected_text = para_text_by_pid[key_pids[i - 1]]
            assert ra[f"passage_{i}"] == expected_text

    # Across 100 questions (8! = 40,320 permutations each), virtually none should remain in original order
    assert unpermuted_count < 5


@requires_raw_squad
def test_build_audit_kit_determinism(tmp_path: Path) -> None:
    """Verify two runs of build_audit_kit produce byte-identical files."""
    out_2 = tmp_path / "audit_run2"
    build_audit_kit(out_dir=out_2)

    for fname in ("audit_sample.json", "annotator_A.csv", "annotator_B.csv", "key.json"):
        assert sha256_file(out_2 / fname) == sha256_file(DEFAULT_AUDIT_DIR / fname)


def test_kappa_constructed_case_equals_hand_computed_value(tmp_path: Path) -> None:
    """Verify Cohen's kappa and percent agreement on a constructed 10-item case equal hand-computed values.

    Constructed 10-question case:
      - n_11 (both 1): 2 questions (q1, q2)
      - n_10 (A=1, B=0): 1 question (q3)
      - n_01 (A=0, B=1): 2 questions (q4, q5)
      - n_00 (both 0): 5 questions (q6..q10)
    Total N = 10:
      - Observed agreement: p_o = (2 + 5) / 10 = 0.70 (70.0%)
      - Marginals:
          P(A=1) = 3/10 = 0.30, P(A=0) = 7/10 = 0.70
          P(B=1) = 4/10 = 0.40, P(B=0) = 6/10 = 0.60
      - Expected agreement by chance:
          p_e = (0.30 * 0.40) + (0.70 * 0.60) = 0.12 + 0.42 = 0.54
      - Hand-computed Cohen's kappa:
          kappa = (0.70 - 0.54) / (1.0 - 0.54) = 0.16 / 0.46 = 8 / 23 ≈ 0.34782608695652173
    """
    flags_a = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    flags_b = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]

    stats = compute_cohens_kappa(flags_a, flags_b)
    assert stats["n"] == 10
    assert stats["counts"] == {"n_11": 2, "n_10": 1, "n_01": 2, "n_00": 5}
    assert stats["agreed_count"] == 7
    assert stats["disagreed_count"] == 3
    assert stats["percent_agreement"] == pytest.approx(70.0, abs=1e-12)
    assert stats["proportion_agreement"] == pytest.approx(0.70, abs=1e-12)
    assert stats["expected_agreement"] == pytest.approx(0.54, abs=1e-12)
    assert stats["cohens_kappa"] == pytest.approx(8.0 / 23.0, abs=1e-12)


def test_exclusions_come_only_from_adjudicated_flags(tmp_path: Path) -> None:
    """Verify exclusions.json is written ONLY from rows with adjudicated_flag == 1, regardless of annotator flags."""
    csv_a = tmp_path / "annotator_A.csv"
    csv_b = tmp_path / "annotator_B.csv"
    exclusions_path = tmp_path / "exclusions.json"

    # Case 1: Annotators flag questions, but NO adjudicated_flag column is filled yet -> exclusions.json must NOT be created
    fieldnames_no_adj = list(CSV_COLUMNS)
    rows_a_data = [
        {"question_id": "q1", "question": "Q1?", "flagged": "1", "which_passage": "2", "note": "Both flagged"},
        {"question_id": "q2", "question": "Q2?", "flagged": "1", "which_passage": "5", "note": "Only A flagged"},
        {"question_id": "q3", "question": "Q3?", "flagged": "0", "which_passage": "", "note": "Only B flagged"},
        {"question_id": "q4", "question": "Q4?", "flagged": "0", "which_passage": "", "note": "Neither flagged"},
    ]
    # Give B a different row order
    rows_b_data = [
        {"question_id": "q4", "question": "Q4?", "flagged": "0", "which_passage": "", "note": ""},
        {"question_id": "q3", "question": "Q3?", "flagged": "1", "which_passage": "1", "note": "Alt reading"},
        {"question_id": "q2", "question": "Q2?", "flagged": "0", "which_passage": "", "note": ""},
        {"question_id": "q1", "question": "Q1?", "flagged": "1", "which_passage": "2", "note": "Agree"},
    ]

    for path, data in ((csv_a, rows_a_data), (csv_b, rows_b_data)):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_no_adj)
            writer.writeheader()
            for r in data:
                full_row = {col: "" for col in fieldnames_no_adj}
                full_row.update(r)
                writer.writerow(full_row)

    res_before_adj = evaluate_audit_agreement(
        csv_a_path=csv_a,
        csv_b_path=csv_b,
        exclusions_out_path=exclusions_path,
    )
    assert res_before_adj["percent_agreement"] == pytest.approx(50.0)
    assert [d["question_id"] for d in res_before_adj["disagreements"]] == ["q2", "q3"]
    assert res_before_adj["exclusions"] is None
    assert not exclusions_path.exists(), (
        "exclusions.json must not be written before adjudicated_flag is provided"
    )

    # Case 2: Fill adjudicated_flag column such that:
    # - q1 (both A=1, B=1) gets adjudicated_flag = 0 -> MUST NOT be excluded!
    # - q2 (A=1, B=0) gets adjudicated_flag = 1 -> MUST be excluded!
    # - q3 (A=0, B=1) gets adjudicated_flag = 0 -> MUST NOT be excluded!
    # - q4 (A=0, B=0) gets adjudicated_flag = 1 -> MUST be excluded!
    fieldnames_with_adj = list(CSV_COLUMNS) + ["adjudicated_flag"]
    adj_map = {"q1": "0", "q2": "1", "q3": "0", "q4": "1"}

    with csv_a.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_with_adj)
        writer.writeheader()
        for r in rows_a_data:
            full_row = {col: "" for col in fieldnames_with_adj}
            full_row.update(r)
            full_row["adjudicated_flag"] = adj_map[r["question_id"]]
            writer.writerow(full_row)

    res_after_adj = evaluate_audit_agreement(
        csv_a_path=csv_a,
        csv_b_path=csv_b,
        exclusions_out_path=exclusions_path,
    )
    assert exclusions_path.exists()
    written_exclusions: list[dict[str, Any]] = json.loads(
        exclusions_path.read_text(encoding="utf-8")
    )
    excluded_qids = [item["question_id"] for item in written_exclusions]
    assert excluded_qids == ["q2", "q4"]
    assert all(item["adjudicated_flag"] == 1 for item in written_exclusions)
