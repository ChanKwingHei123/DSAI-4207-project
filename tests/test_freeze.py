"""Tests for the P7 input freeze (`scripts/freeze_inputs.py` and `data/frozen/`)."""

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

from scripts.freeze_inputs import (  # noqa: E402
    DEFAULT_EXCLUSIONS_PATH,
    DEFAULT_INPUTS_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_PREREG_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_SAMPLING_DIR,
    freeze_inputs,
    load_jsonl_raw_lines,
    sha256_file,
)


def _make_synthetic_freeze_repo(
    tmp_path: Path,
    *,
    n_questions: int,
    main_items: list[dict[str, Any]],
    reserve_items: list[dict[str, Any]],
    main_missing_seeds: dict[str, set[int]],
    reserve_missing_seeds: dict[str, set[int]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Path]:
    """Create a synthetic repository layout in `tmp_path` for testing `freeze_inputs`."""
    configs_dir = tmp_path / "configs"
    sampling_dir = tmp_path / "data" / "sampling"
    inputs_dir = tmp_path / "data" / "inputs"
    audit_dir = tmp_path / "data" / "audit"
    for d in (configs_dir, sampling_dir, inputs_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    prereg_path = configs_dir / "prereg.yaml"
    prereg_path.write_text(
        yaml.safe_dump(
            {
                "design": {"n_questions": n_questions},
                "rng": {"draw_seeds": [0, 1, 2]},
            }
        ),
        encoding="utf-8",
    )

    raw_path = tmp_path / "data" / "raw" / "dev-v2.0.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text('{"data": []}\n', encoding="utf-8")
    raw_sha = sha256_file(raw_path)

    (sampling_dir / "main.json").write_text(
        json.dumps(main_items, indent=2) + "\n", encoding="utf-8"
    )
    (sampling_dir / "reserve.json").write_text(
        json.dumps(reserve_items, indent=2) + "\n", encoding="utf-8"
    )

    art_titles = {
        int(q["article_index"]): str(q["article_title"])
        for q in (main_items + reserve_items)
    }
    (sampling_dir / "manifest.json").write_text(
        json.dumps(
            {
                "raw_file_sha256": raw_sha,
                "per_article_counts": [
                    {"article_index": idx, "article_title": title}
                    for idx, title in sorted(art_titles.items())
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    exclusions_path = audit_dir / "exclusions.json"
    exclusions_path.write_text(json.dumps(exclusions, indent=2) + "\n", encoding="utf-8")

    for split_name, items, missing_map in (
        ("main", main_items, main_missing_seeds),
        ("reserve", reserve_items, reserve_missing_seeds),
    ):
        for seed in (0, 1, 2):
            lines: list[str] = []
            for idx, q in enumerate(items):
                qid = str(q["question_id"])
                if seed in missing_map.get(qid, set()):
                    continue
                row = {
                    "question_id": qid,
                    "split": split_name,
                    "seed": seed,
                    "article_index": q["article_index"],
                    "article_title": q["article_title"],
                    "question_text": q["question_text"],
                    "gold_answers": q["gold_answers"],
                    "gold_paragraph_id": q["paragraph_id"],
                    # Distinct whitespace/key formatting verifies byte-for-byte copy
                    "marker": f"{split_name}-seed{seed}-idx{idx}",
                }
                lines.append(json.dumps(row, separators=(",", ":")) + "\n")
            (inputs_dir / f"{split_name}_seed{seed}.jsonl").write_text(
                "".join(lines), encoding="utf-8"
            )

    return {
        "prereg_path": prereg_path,
        "raw_path": raw_path,
        "sampling_dir": sampling_dir,
        "inputs_dir": inputs_dir,
        "exclusions_path": exclusions_path,
    }


def _make_q(qid: str, art_idx: int, art_title: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "article_index": art_idx,
        "article_title": art_title,
        "paragraph_id": [art_idx, 0],
        "paragraph_index": 0,
        "question_text": f"Question {qid}?",
        "gold_answers": [f"ans_{qid}"],
    }


def test_two_pass_prevents_cross_article_poaching(tmp_path: Path) -> None:
    """Pass 1 must allocate same-article reserves before Pass 2 cross-article fallbacks."""
    main_items = [
        _make_q("m0", 0, "Art0"),  # incomplete (missing seed 1), no Art0 reserve
        _make_q("m1", 1, "Art1"),  # retained
        _make_q("m2", 2, "Art2"),  # audit-excluded, Art2 has reserve r_art2 (first in reserve.json!)
        _make_q("m3", 3, "Art3"),  # retained
    ]
    # Note: r_art2 appears BEFORE r_art4 in reserve.json.
    # Under 1-pass greedy, m0 would consume r_art2, forcing m2 onto r_art4.
    # Under 2-pass, Pass 1 assigns r_art2 -> m2, and Pass 2 assigns r_art4 -> m0.
    reserve_items = [
        _make_q("r_art2", 2, "Art2"),
        _make_q("r_art4", 4, "Art4"),
    ]
    paths = _make_synthetic_freeze_repo(
        tmp_path,
        n_questions=4,
        main_items=main_items,
        reserve_items=reserve_items,
        main_missing_seeds={"m0": {1}},
        reserve_missing_seeds={},
        exclusions=[{"question_id": "m2", "adjudicated_flag": 1}],
    )

    out_dir = tmp_path / "data" / "frozen"
    res = freeze_inputs(**paths, out_dir=out_dir)
    manifest = res["manifest"]

    assert manifest["final_n"] == 4
    assert manifest["question_ids"] == ["r_art4", "m1", "r_art2", "m3"]
    assert manifest["counts"]["same_article_replacements"] == 1
    assert manifest["counts"]["cross_article_replacements"] == 1

    rep_by_old = {r["old_qid"]: r for r in manifest["replacements"]}
    assert rep_by_old["m2"]["new_qid"] == "r_art2"
    assert rep_by_old["m2"]["same_article"] is True
    assert rep_by_old["m2"]["pass"] == 1
    assert rep_by_old["m0"]["new_qid"] == "r_art4"
    assert rep_by_old["m0"]["same_article"] is False
    assert rep_by_old["m0"]["pass"] == 2


def test_freeze_determinism_overlap_ineligible_reserve_and_byte_copy(tmp_path: Path) -> None:
    """Verify determinism, overlap accounting, ineligible reserve filtering, and byte-exact seed lines."""
    main_items = [
        _make_q("m0", 0, "Art0"),  # both missing seed 2 AND audit-excluded (overlap)
        _make_q("m1", 1, "Art1"),  # retained
        _make_q("m2", 2, "Art2"),  # audit-excluded (adjudicated_flag=1)
        _make_q("m3", 3, "Art3"),  # retained (listed in exclusions with adjudicated_flag=0)
    ]
    reserve_items = [
        _make_q("r_bad", 0, "Art0"),  # ineligible: missing seed 0
        _make_q("r_good0", 0, "Art0"),
        _make_q("r_good9", 9, "Art9"),
    ]
    paths = _make_synthetic_freeze_repo(
        tmp_path,
        n_questions=4,
        main_items=main_items,
        reserve_items=reserve_items,
        main_missing_seeds={"m0": {2}},
        reserve_missing_seeds={"r_bad": {0}},
        exclusions=[
            {"question_id": "m0", "adjudicated_flag": 1},
            {"question_id": "m2", "adjudicated_flag": 1},
            {"question_id": "m3", "adjudicated_flag": 0},
        ],
    )

    out_a = tmp_path / "frozen_a"
    out_b = tmp_path / "frozen_b"
    res_a = freeze_inputs(**paths, out_dir=out_a)
    res_b = freeze_inputs(**paths, out_dir=out_b)

    assert res_a["manifest_sha256"] == res_b["manifest_sha256"]
    for fname in (
        "main.json",
        "main_seed0.jsonl",
        "main_seed1.jsonl",
        "main_seed2.jsonl",
        "freeze_report.md",
        "manifest.json",
        "manifest.sha256",
    ):
        assert (out_a / fname).read_bytes() == (out_b / fname).read_bytes()

    c = res_a["manifest"]["counts"]
    assert c["final_n"] == 4
    assert c["retained"] == 2
    assert c["replaced"] == 2
    assert c["incomplete_seed_exclusions"] == 1
    assert c["audit_exclusions"] == 2
    assert c["overlap"] == 1
    assert c["eligible_reserve"] == 2
    assert c["ineligible_reserve"] == 1
    assert res_a["manifest"]["question_ids"] == ["r_good0", "m1", "r_good9", "m3"]

    # Verify byte-for-byte line copying from source main_seed{S} / reserve_seed{S}
    for s in (0, 1, 2):
        main_src = load_jsonl_raw_lines(paths["inputs_dir"] / f"main_seed{s}.jsonl")
        res_src = load_jsonl_raw_lines(paths["inputs_dir"] / f"reserve_seed{s}.jsonl")
        frozen_lines = (out_a / f"main_seed{s}.jsonl").read_bytes().splitlines()
        assert frozen_lines == [
            res_src["r_good0"],
            main_src["m1"],
            res_src["r_good9"],
            main_src["m3"],
        ]


def test_freeze_insufficient_reserve_raises(tmp_path: Path) -> None:
    """Raise RuntimeError if eligible reserve questions cannot cover all exclusions."""
    main_items = [_make_q("m0", 0, "Art0"), _make_q("m1", 1, "Art1")]
    reserve_items = [_make_q("r0", 0, "Art0")]
    paths = _make_synthetic_freeze_repo(
        tmp_path,
        n_questions=2,
        main_items=main_items,
        reserve_items=reserve_items,
        main_missing_seeds={"m0": {0}, "m1": {1}},
        reserve_missing_seeds={},
        exclusions=[],
    )
    with pytest.raises(RuntimeError, match="Insufficient eligible reserve"):
        freeze_inputs(**paths, out_dir=tmp_path / "frozen")


def test_real_frozen_split_integrity_and_zero_exclusions() -> None:
    """Verify real data/frozen/ split has N=500, 3 complete seeds, 0 exclusions, and valid manifest."""
    assert DEFAULT_OUT_DIR.exists(), f"Missing frozen output dir: {DEFAULT_OUT_DIR}"

    prereg = yaml.safe_load(DEFAULT_PREREG_PATH.read_text(encoding="utf-8"))
    expected_n = int(prereg["design"]["n_questions"])
    assert expected_n == 500

    frozen_main = json.loads((DEFAULT_OUT_DIR / "main.json").read_text(encoding="utf-8"))
    assert len(frozen_main) == 500
    frozen_qids = [str(q["question_id"]) for q in frozen_main]
    assert len(set(frozen_qids)) == 500

    # Zero overlap with pilot
    pilot_qids = {
        str(q["question_id"])
        for q in json.loads((DEFAULT_SAMPLING_DIR / "pilot.json").read_text(encoding="utf-8"))
    }
    assert set(frozen_qids).isdisjoint(pilot_qids)

    # Zero audit-excluded IDs
    exclusions = json.loads(DEFAULT_EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    audit_excluded = {
        str(e["question_id"]) for e in exclusions if int(e.get("adjudicated_flag", 0)) == 1
    }
    assert len(audit_excluded) == 8
    assert set(frozen_qids).isdisjoint(audit_excluded)

    # Zero incomplete-seed IDs and identical ID sets/order across all 3 seeds
    main_sampling_qids = {
        str(q["question_id"])
        for q in json.loads((DEFAULT_SAMPLING_DIR / "main.json").read_text(encoding="utf-8"))
    }
    reserve_sampling_qids = {
        str(q["question_id"])
        for q in json.loads((DEFAULT_SAMPLING_DIR / "reserve.json").read_text(encoding="utf-8"))
    }
    retained_qids = set(frozen_qids) & main_sampling_qids
    replaced_qids = set(frozen_qids) & reserve_sampling_qids
    assert len(retained_qids) == 482
    assert len(replaced_qids) == 18

    for seed in (0, 1, 2):
        frozen_seed_path = DEFAULT_OUT_DIR / f"main_seed{seed}.jsonl"
        frozen_lines = load_jsonl_raw_lines(frozen_seed_path)
        assert list(frozen_lines.keys()) == frozen_qids
        assert len(frozen_lines) == 500

        main_src_lines = load_jsonl_raw_lines(DEFAULT_INPUTS_DIR / f"main_seed{seed}.jsonl")
        res_src_lines = load_jsonl_raw_lines(DEFAULT_INPUTS_DIR / f"reserve_seed{seed}.jsonl")

        for qid in frozen_qids:
            if qid in retained_qids:
                assert qid in main_src_lines
                assert frozen_lines[qid] == main_src_lines[qid]
            else:
                assert qid in replaced_qids
                assert qid in res_src_lines
                assert frozen_lines[qid] == res_src_lines[qid]

    # Manifest & digest checks
    manifest_path = DEFAULT_OUT_DIR / "manifest.json"
    manifest_sha_path = DEFAULT_OUT_DIR / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["final_n"] == 500
    assert manifest["question_ids"] == frozen_qids
    assert manifest["counts"] == {
        "final_n": 500,
        "original_main": 500,
        "retained": 482,
        "replaced": 18,
        "incomplete_seed_exclusions": 10,
        "audit_exclusions": 8,
        "overlap": 0,
        "reserve_total": 40,
        "eligible_reserve": 39,
        "ineligible_reserve": 1,
        "same_article_replacements": 8,
        "cross_article_replacements": 10,
        "unused_reserve": 21,
    }
    assert len(manifest["replacements"]) == 18
    assert sum(r["before"] for r in manifest["per_article_counts"]) == 500
    assert sum(r["after"] for r in manifest["per_article_counts"]) == 500

    sampling_manifest = json.loads((DEFAULT_SAMPLING_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_dev_sha256"] == sampling_manifest["raw_file_sha256"]
    if DEFAULT_RAW_PATH.exists():
        assert manifest["raw_dev_sha256"] == sha256_file(DEFAULT_RAW_PATH)
    assert manifest["sampling_manifest_sha256"] == sha256_file(DEFAULT_SAMPLING_DIR / "manifest.json")
    assert manifest["exclusions_sha256"] == sha256_file(DEFAULT_EXCLUSIONS_PATH)

    expected_manifest_sha = manifest_sha_path.read_text(encoding="utf-8").strip().split()[0].lower()
    assert sha256_file(manifest_path) == expected_manifest_sha
    for key in ("main", "main_seed0", "main_seed1", "main_seed2", "freeze_report"):
        info = manifest["output_files"][key]
        assert sha256_file(REPO_ROOT / info["path"]) == info["sha256"]

    report_text = (DEFAULT_OUT_DIR / "freeze_report.md").read_text(encoding="utf-8")
    assert "Pass 1" in report_text and "Pass 2" in report_text
    for rep in manifest["replacements"]:
        assert rep["old_qid"] in report_text
        assert rep["new_qid"] in report_text
