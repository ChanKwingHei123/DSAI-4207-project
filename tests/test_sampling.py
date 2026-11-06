"""Unit and integration tests for `scripts/make_sampling_frame.py`."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_sampling_frame import (  # noqa: E402
    DEFAULT_OUT_DIR,
    DEFAULT_PREREG_PATH,
    DEFAULT_RAW_PATH,
    SEED,
    build_sampling_frame,
    largest_remainder_allocate,
    sha256_file,
)

requires_raw_squad = pytest.mark.skipif(
    not DEFAULT_RAW_PATH.exists(),
    reason=f"Raw SQuAD 2.0 dev file is absent at {DEFAULT_RAW_PATH}",
)


def test_largest_remainder_allocate_unit() -> None:
    """Verify largest-remainder proportional allocation math and deterministic tie-breaking."""
    counts = [100, 100, 100]
    # 10 seats across 3 equal strata -> remainders are equal, tie-broken by index 0 then 1
    assert largest_remainder_allocate(counts, 10) == [4, 3, 3]
    assert largest_remainder_allocate([96, 176, 297], 50) == [8, 16, 26]


def test_missing_raw_file_raises_with_download_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify missing raw file prints the download command and raises FileNotFoundError."""
    missing_raw = tmp_path / "nonexistent" / "dev-v2.0.json"
    with pytest.raises(FileNotFoundError):
        build_sampling_frame(
            raw_path=missing_raw,
            prereg_path=DEFAULT_PREREG_PATH,
            out_dir=tmp_path / "out",
        )
    captured = capsys.readouterr()
    assert "curl -fsSLo" in captured.err
    assert "dev-v2.0.json" in captured.err


@requires_raw_squad
def test_sampling_frame_disjointness_and_exact_counts() -> None:
    """Verify main, pilot, and reserve splits exist, match prereg counts, and are disjoint."""
    prereg = yaml.safe_load(DEFAULT_PREREG_PATH.read_text(encoding="utf-8"))
    expected_main = int(prereg["design"]["n_questions"])
    expected_pilot = int(prereg["design"]["pilot_questions"])
    expected_reserve = int(prereg["design"]["reserve_questions"])

    main_data = json.loads((DEFAULT_OUT_DIR / "main.json").read_text(encoding="utf-8"))
    pilot_data = json.loads((DEFAULT_OUT_DIR / "pilot.json").read_text(encoding="utf-8"))
    reserve_data = json.loads((DEFAULT_OUT_DIR / "reserve.json").read_text(encoding="utf-8"))

    assert len(main_data) == expected_main == 500
    assert len(pilot_data) == expected_pilot == 50
    assert len(reserve_data) == expected_reserve == 40

    main_ids = {q["question_id"] for q in main_data}
    pilot_ids = {q["question_id"] for q in pilot_data}
    reserve_ids = {q["question_id"] for q in reserve_data}

    assert len(main_ids) == expected_main
    assert len(pilot_ids) == expected_pilot
    assert len(reserve_ids) == expected_reserve

    assert main_ids.isdisjoint(pilot_ids), "main and pilot splits overlap"
    assert main_ids.isdisjoint(reserve_ids), "main and reserve splits overlap"
    assert pilot_ids.isdisjoint(reserve_ids), "pilot and reserve splits overlap"


@requires_raw_squad
def test_sampling_frame_determinism_identical_hashes(tmp_path: Path) -> None:
    """Verify two independent runs produce byte-identical output files and hashes."""
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    manifest_a = build_sampling_frame(
        raw_path=DEFAULT_RAW_PATH,
        prereg_path=DEFAULT_PREREG_PATH,
        out_dir=out_a,
        seed=SEED,
    )
    manifest_b = build_sampling_frame(
        raw_path=DEFAULT_RAW_PATH,
        prereg_path=DEFAULT_PREREG_PATH,
        out_dir=out_b,
        seed=SEED,
    )

    assert manifest_a == manifest_b
    for fname in ("main.json", "pilot.json", "reserve.json", "manifest.json"):
        hash_a = sha256_file(out_a / fname)
        hash_b = sha256_file(out_b / fname)
        hash_repo = sha256_file(DEFAULT_OUT_DIR / fname)
        assert hash_a == hash_b == hash_repo, f"Determinism hash mismatch for {fname}"


@requires_raw_squad
def test_per_article_counts_equal_sequential_largest_remainder() -> None:
    """Verify per-article counts in main, pilot, and reserve equal largest-remainder allocations."""
    prereg = yaml.safe_load(DEFAULT_PREREG_PATH.read_text(encoding="utf-8"))
    n_main = int(prereg["design"]["n_questions"])
    n_pilot = int(prereg["design"]["pilot_questions"])
    n_reserve = int(prereg["design"]["reserve_questions"])

    raw_articles = json.loads(DEFAULT_RAW_PATH.read_text(encoding="utf-8"))["data"]
    initial_counts: list[int] = []
    for art in raw_articles:
        c = sum(
            1
            for p in art["paragraphs"]
            for qa in p["qas"]
            if not qa.get("is_impossible", False) and len(qa.get("answers", [])) > 0
        )
        initial_counts.append(c)

    expected_main_alloc = largest_remainder_allocate(initial_counts, n_main)
    rem_after_main = [c - m for c, m in zip(initial_counts, expected_main_alloc)]
    expected_pilot_alloc = largest_remainder_allocate(rem_after_main, n_pilot)
    rem_after_pilot = [c - p for c, p in zip(rem_after_main, expected_pilot_alloc)]
    expected_reserve_alloc = largest_remainder_allocate(rem_after_pilot, n_reserve)

    main_data = json.loads((DEFAULT_OUT_DIR / "main.json").read_text(encoding="utf-8"))
    pilot_data = json.loads((DEFAULT_OUT_DIR / "pilot.json").read_text(encoding="utf-8"))
    reserve_data = json.loads((DEFAULT_OUT_DIR / "reserve.json").read_text(encoding="utf-8"))

    num_articles = len(raw_articles)
    actual_main_alloc = [
        sum(1 for q in main_data if q["article_index"] == idx) for idx in range(num_articles)
    ]
    actual_pilot_alloc = [
        sum(1 for q in pilot_data if q["article_index"] == idx) for idx in range(num_articles)
    ]
    actual_reserve_alloc = [
        sum(1 for q in reserve_data if q["article_index"] == idx) for idx in range(num_articles)
    ]

    assert actual_main_alloc == expected_main_alloc
    assert actual_pilot_alloc == expected_pilot_alloc
    assert actual_reserve_alloc == expected_reserve_alloc


@requires_raw_squad
def test_every_question_keeps_all_gold_answers_and_none_unanswerable() -> None:
    """Verify every sampled question is answerable and preserves all its SQuAD gold answers."""
    raw_articles = json.loads(DEFAULT_RAW_PATH.read_text(encoding="utf-8"))["data"]
    raw_lookup: dict[str, dict[str, Any]] = {}
    for art_idx, art in enumerate(raw_articles):
        for para_idx, para in enumerate(art["paragraphs"]):
            for qa in para["qas"]:
                raw_lookup[qa["id"]] = {
                    "is_impossible": bool(qa.get("is_impossible", False)),
                    "answers": [a["text"] for a in qa.get("answers", [])],
                    "article_index": art_idx,
                    "article_title": art["title"],
                    "paragraph_id": [art_idx, para_idx],
                    "question_text": qa["question"],
                }

    multi_answer_verified_count = 0
    for split_name in ("main.json", "pilot.json", "reserve.json"):
        records = json.loads((DEFAULT_OUT_DIR / split_name).read_text(encoding="utf-8"))
        for rec in records:
            qid = rec["question_id"]
            assert qid in raw_lookup, f"Unknown question_id {qid} in {split_name}"
            ref = raw_lookup[qid]
            assert ref["is_impossible"] is False, f"Unanswerable question {qid} included in {split_name}"
            assert len(ref["answers"]) >= 1, f"Question {qid} has no gold answers"
            assert rec["gold_answers"] == ref["answers"], (
                f"Gold answers mismatch for {qid}: {rec['gold_answers']} != {ref['answers']}"
            )
            assert rec["article_index"] == ref["article_index"]
            assert rec["article_title"] == ref["article_title"]
            assert rec["paragraph_id"] == ref["paragraph_id"]
            assert rec["question_text"] == ref["question_text"]
            if len(rec["gold_answers"]) > 1:
                multi_answer_verified_count += 1

    assert multi_answer_verified_count > 500, (
        "Expected most SQuAD 2.0 dev questions to carry multiple human gold answers"
    )
