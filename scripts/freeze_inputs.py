#!/usr/bin/env python3
"""Execute the P7 input freeze (Decision D12 + P7 delta) to produce `data/frozen/`.

Reads `data/sampling/{main,reserve,manifest}.json`, `data/audit/exclusions.json`,
and `data/inputs/{main,reserve}_seed{0,1,2}.jsonl` without modifying them, then
applies the deterministic 2-pass replacement rule:
  - Pass 1 (Same-article): iterate excluded/incomplete `main.json` questions in order
    and assign the first unused eligible reserve question (present in all 3 reserve
    seeds and not audit-excluded) from the same `article_index`.
  - Pass 2 (Cross-article fallback): iterate remaining unassigned excluded/incomplete
    `main.json` questions in order and assign the next unused eligible reserve
    question in `reserve.json` order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREG_PATH = REPO_ROOT / "configs" / "prereg.yaml"
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_SAMPLING_DIR = REPO_ROOT / "data" / "sampling"
DEFAULT_INPUTS_DIR = REPO_ROOT / "data" / "inputs"
DEFAULT_EXCLUSIONS_PATH = REPO_ROOT / "data" / "audit" / "exclusions.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "frozen"


def sha256_bytes(data: bytes) -> str:
    """Compute hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON file, raising FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_raw_lines(path: Path) -> dict[str, bytes]:
    """Load `{question_id: raw_line_bytes}` from a JSONL file without trailing newlines."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL file: {path}")
    by_qid: dict[str, bytes] = {}
    for line_no, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line.decode("utf-8"))
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Malformed row missing 'question_id' in {path}:{line_no}")
        if qid in by_qid:
            raise ValueError(f"Duplicate question_id {qid!r} in {path}:{line_no}")
        by_qid[qid] = raw_line
    return by_qid


def _format_exclusion_reason(missing_seeds: Sequence[int], audit_flagged: bool) -> str:
    if missing_seeds and audit_flagged:
        seeds_str = ",".join(str(s) for s in missing_seeds)
        return f"incomplete_seeds (missing: {seeds_str}) + audit_exclusion"
    if missing_seeds:
        seeds_str = ",".join(str(s) for s in missing_seeds)
        return f"incomplete_seeds (missing: {seeds_str})"
    if audit_flagged:
        return "audit_exclusion"
    raise ValueError("Excluded question must have missing_seeds or audit_flagged=True")


def _select_two_pass_replacements(
    excluded_items: Sequence[dict[str, Any]],
    eligible_reserve: Sequence[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], bool, int]]:
    """Assign replacement reserve items using the 2-pass rule.

    Returns `{old_qid: (reserve_item, same_article, pass_number)}`.
    """
    used_reserve_ids: set[str] = set()
    assignments: dict[str, tuple[dict[str, Any], bool, int]] = {}

    # Pass 1: Same-article replacements in main.json order.
    for item in excluded_items:
        old_q = item["question"]
        old_qid = str(old_q["question_id"])
        old_art = int(old_q["article_index"])
        for res_q in eligible_reserve:
            res_qid = str(res_q["question_id"])
            if res_qid in used_reserve_ids:
                continue
            if int(res_q["article_index"]) == old_art:
                used_reserve_ids.add(res_qid)
                assignments[old_qid] = (res_q, True, 1)
                break

    # Pass 2: Cross-article fallbacks in main.json order from remaining reserve.json order.
    for item in excluded_items:
        old_q = item["question"]
        old_qid = str(old_q["question_id"])
        if old_qid in assignments:
            continue
        chosen: dict[str, Any] | None = None
        for res_q in eligible_reserve:
            res_qid = str(res_q["question_id"])
            if res_qid not in used_reserve_ids:
                chosen = res_q
                break
        if chosen is None:
            raise RuntimeError(
                f"Insufficient eligible reserve questions to replace {old_qid!r}: "
                f"needed {len(excluded_items)}, eligible={len(eligible_reserve)}"
            )
        used_reserve_ids.add(str(chosen["question_id"]))
        assignments[old_qid] = (chosen, False, 2)

    return assignments


def _compute_per_article_counts(
    main_questions: Sequence[Mapping[str, Any]],
    frozen_questions: Sequence[Mapping[str, Any]],
    sampling_manifest: Mapping[str, Any],
    replacements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build ordered per-article counts before and after freeze replacement."""
    article_titles: dict[int, str] = {}
    for entry in sampling_manifest.get("per_article_counts", []):
        article_titles[int(entry["article_index"])] = str(entry["article_title"])
    for q in list(main_questions) + list(frozen_questions):
        article_titles.setdefault(int(q["article_index"]), str(q["article_title"]))

    before_counts: dict[int, int] = {idx: 0 for idx in article_titles}
    after_counts: dict[int, int] = {idx: 0 for idx in article_titles}
    excluded_counts: dict[int, int] = {idx: 0 for idx in article_titles}
    added_counts: dict[int, int] = {idx: 0 for idx in article_titles}

    for q in main_questions:
        before_counts[int(q["article_index"])] += 1
    for q in frozen_questions:
        after_counts[int(q["article_index"])] += 1
    for rep in replacements:
        excluded_counts[int(rep["old_article_index"])] += 1
        added_counts[int(rep["new_article_index"])] += 1

    table: list[dict[str, Any]] = []
    for art_idx in sorted(article_titles):
        before = before_counts[art_idx]
        after = after_counts[art_idx]
        table.append(
            {
                "article_index": art_idx,
                "article_title": article_titles[art_idx],
                "before": before,
                "excluded": excluded_counts[art_idx],
                "added": added_counts[art_idx],
                "after": after,
                "delta": after - before,
            }
        )
    return table


def _render_freeze_report(
    *,
    counts: Mapping[str, int],
    replacements: Sequence[Mapping[str, Any]],
    per_article_counts: Sequence[Mapping[str, Any]],
    ineligible_reserve: Sequence[Mapping[str, Any]],
    raw_dev_sha256: str,
    sampling_manifest_sha256: str,
    exclusions_sha256: str,
    output_files: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render deterministic human-readable `data/frozen/freeze_report.md`."""
    lines: list[str] = [
        "# P7 Frozen Main Split Report (Decision D12 + P7 Delta)",
        "",
        "## 1. Freeze Policy & Two-Pass Replacement Rule",
        "- **Exclusion rule (Decision D12)**: Exclude every question in `data/sampling/main.json` that is either:",
        "  1. Missing from any of `data/inputs/main_seed{0,1,2}.jsonl` (P2c incomplete-seed exclusion due to failed random length-match slots), or",
        "  2. Listed in `data/audit/exclusions.json` with `adjudicated_flag == 1` (blind audit exclusion).",
        "- **Eligible reserve pool**: Questions in `data/sampling/reserve.json` present in all three `data/inputs/reserve_seed{0,1,2}.jsonl` files and not audit-excluded.",
        "- **Two-pass replacement rule**:",
        "  - **Pass 1 (Same-article replacements)**: Iterate over all excluded/incomplete main questions in `main.json` order; if an unused eligible reserve question from the same `article_index` exists, assign the first such reserve question in `reserve.json` order.",
        "  - **Pass 2 (Cross-article fallbacks)**: Iterate over remaining unassigned excluded/incomplete main questions in `main.json` order; assign each to the next unused eligible reserve question in `reserve.json` order.",
        "",
        "## 2. Summary Counts",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| **Final frozen `main` N** | **{counts['final_n']}** |",
        f"| Original `main.json` questions | {counts['original_main']} |",
        f"| Retained original `main.json` questions | {counts['retained']} |",
        f"| Total excluded & replaced questions | {counts['replaced']} |",
        f"| Incomplete-seed exclusions (missing >=1 seed) | {counts['incomplete_seed_exclusions']} |",
        f"| Audit exclusions (`adjudicated_flag == 1`) | {counts['audit_exclusions']} |",
        f"| Overlap (incomplete-seed AND audit exclusion) | {counts['overlap']} |",
        f"| Reserve pool total (`reserve.json`) | {counts['reserve_total']} |",
        f"| Eligible reserve questions (complete in all 3 seeds) | {counts['eligible_reserve']} |",
        f"| Ineligible reserve questions | {counts['ineligible_reserve']} |",
        f"| Same-article replacements (Pass 1) | {counts['same_article_replacements']} |",
        f"| Cross-article fallbacks (Pass 2) | {counts['cross_article_replacements']} |",
        f"| Unused eligible reserve remaining | {counts['unused_reserve']} |",
        "",
    ]

    if ineligible_reserve:
        lines.extend(
            [
                "### Ineligible Reserve Questions",
                "| reserve_qid | article | reason |",
                "| --- | --- | --- |",
            ]
        )
        for item in ineligible_reserve:
            art = f"{item['article_index']} ({item['article_title']})"
            lines.append(f"| `{item['question_id']}` | {art} | {item['reason']} |")
        lines.append("")

    lines.extend(
        [
            "## 3. Complete Replacement Table",
            "| old_qid | old_article | reason | new_qid | new_article | same_article |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rep in replacements:
        old_art = f"{rep['old_article_index']} ({rep['old_article_title']})"
        new_art = f"{rep['new_article_index']} ({rep['new_article_title']})"
        same_str = "true" if rep["same_article"] else "false"
        lines.append(
            f"| `{rep['old_qid']}` | {old_art} | {rep['reason']} "
            f"| `{rep['new_qid']}` | {new_art} | {same_str} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 4. Per-Article Question Counts (Before vs After Replacement)",
            "| article_index | article_title | before | excluded | added | after | delta |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    total_before = sum(int(r["before"]) for r in per_article_counts)
    total_excluded = sum(int(r["excluded"]) for r in per_article_counts)
    total_added = sum(int(r["added"]) for r in per_article_counts)
    total_after = sum(int(r["after"]) for r in per_article_counts)
    for row in per_article_counts:
        delta_str = f"{int(row['delta']):+d}" if row["delta"] != 0 else "0"
        lines.append(
            f"| {row['article_index']} | {row['article_title']} | {row['before']} "
            f"| {row['excluded']} | {row['added']} | {row['after']} | {delta_str} |"
        )
    lines.append(
        f"| **Total** | **All {len(per_article_counts)} articles** | **{total_before}** "
        f"| **{total_excluded}** | **{total_added}** | **{total_after}** | **{total_after - total_before:+d}** |"
    )
    lines.append("")

    lines.extend(
        [
            "## 5. Input & Output SHA-256 Digests",
            "| Artifact | Path | Count | SHA-256 |",
            "| --- | --- | ---: | --- |",
            f"| Raw SQuAD 2.0 dev | `data/raw/dev-v2.0.json` | - | `{raw_dev_sha256}` |",
            f"| Sampling manifest | `data/sampling/manifest.json` | - | `{sampling_manifest_sha256}` |",
            f"| Audit exclusions | `data/audit/exclusions.json` | {counts['audit_exclusions']} | `{exclusions_sha256}` |",
        ]
    )
    for key in ("main", "main_seed0", "main_seed1", "main_seed2"):
        info = output_files[key]
        lines.append(
            f"| Frozen `{key}` | `{info['path']}` | {info['count']} | `{info['sha256']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def freeze_inputs(
    *,
    prereg_path: Path = DEFAULT_PREREG_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    sampling_dir: Path = DEFAULT_SAMPLING_DIR,
    inputs_dir: Path = DEFAULT_INPUTS_DIR,
    exclusions_path: Path = DEFAULT_EXCLUSIONS_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Execute the deterministic P7 input freeze and write `data/frozen/` outputs."""
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    expected_n = int(prereg["design"]["n_questions"])
    draw_seeds = [int(s) for s in prereg["rng"]["draw_seeds"]]

    sampling_manifest_path = sampling_dir / "manifest.json"
    sampling_manifest = load_json(sampling_manifest_path)
    main_questions: list[dict[str, Any]] = load_json(sampling_dir / "main.json")
    reserve_questions: list[dict[str, Any]] = load_json(sampling_dir / "reserve.json")
    exclusions_raw: list[dict[str, Any]] = load_json(exclusions_path)

    sampling_manifest_sha256 = sha256_file(sampling_manifest_path)
    exclusions_sha256 = sha256_file(exclusions_path)
    if raw_path.exists():
        raw_dev_sha256 = sha256_file(raw_path)
        manifest_raw_sha = sampling_manifest.get("raw_file_sha256")
        if manifest_raw_sha and raw_dev_sha256 != manifest_raw_sha:
            raise ValueError(
                f"Raw file SHA-256 mismatch: {raw_dev_sha256} != {manifest_raw_sha}"
            )
    elif "raw_file_sha256" in sampling_manifest:
        raw_dev_sha256 = str(sampling_manifest["raw_file_sha256"])
    else:
        raise FileNotFoundError(f"Missing raw SQuAD file: {raw_path}")

    main_seed_lines: dict[int, dict[str, bytes]] = {
        s: load_jsonl_raw_lines(inputs_dir / f"main_seed{s}.jsonl") for s in draw_seeds
    }
    reserve_seed_lines: dict[int, dict[str, bytes]] = {
        s: load_jsonl_raw_lines(inputs_dir / f"reserve_seed{s}.jsonl") for s in draw_seeds
    }

    audit_excluded_ids: list[str] = []
    seen_audit: set[str] = set()
    for entry in exclusions_raw:
        if int(entry.get("adjudicated_flag", 0)) == 1:
            qid = str(entry["question_id"])
            if qid not in seen_audit:
                seen_audit.add(qid)
                audit_excluded_ids.append(qid)
    audit_excluded_set = set(audit_excluded_ids)

    # Identify main exclusions in main.json order
    excluded_items: list[dict[str, Any]] = []
    incomplete_seed_ids: list[str] = []
    audit_in_main_ids: list[str] = []
    overlap_ids: list[str] = []

    for pos, q in enumerate(main_questions):
        qid = str(q["question_id"])
        missing_seeds = [s for s in draw_seeds if qid not in main_seed_lines[s]]
        audit_flagged = qid in audit_excluded_set
        if missing_seeds:
            incomplete_seed_ids.append(qid)
        if audit_flagged:
            audit_in_main_ids.append(qid)
        if missing_seeds and audit_flagged:
            overlap_ids.append(qid)
        if missing_seeds or audit_flagged:
            excluded_items.append(
                {
                    "position": pos,
                    "question": q,
                    "missing_seeds": missing_seeds,
                    "audit_flagged": audit_flagged,
                    "reason": _format_exclusion_reason(missing_seeds, audit_flagged),
                }
            )

    # Identify eligible vs ineligible reserve questions in reserve.json order
    eligible_reserve: list[dict[str, Any]] = []
    ineligible_reserve: list[dict[str, Any]] = []
    for q in reserve_questions:
        qid = str(q["question_id"])
        missing_seeds = [s for s in draw_seeds if qid not in reserve_seed_lines[s]]
        audit_flagged = qid in audit_excluded_set
        if missing_seeds or audit_flagged:
            ineligible_reserve.append(
                {
                    "question_id": qid,
                    "article_index": int(q["article_index"]),
                    "article_title": str(q["article_title"]),
                    "missing_seeds": missing_seeds,
                    "audit_flagged": audit_flagged,
                    "reason": _format_exclusion_reason(missing_seeds, audit_flagged),
                }
            )
        else:
            eligible_reserve.append(q)

    assignments = _select_two_pass_replacements(excluded_items, eligible_reserve)

    # Assemble frozen main.json and seed JSONLs preserving main.json positions
    frozen_main: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    frozen_seed_raw_lines: dict[int, list[bytes]] = {s: [] for s in draw_seeds}
    excl_by_qid = {str(item["question"]["question_id"]): item for item in excluded_items}

    for pos, q in enumerate(main_questions):
        qid = str(q["question_id"])
        if qid in assignments:
            new_q, same_article, pass_num = assignments[qid]
            new_qid = str(new_q["question_id"])
            excl_info = excl_by_qid[qid]
            frozen_main.append(new_q)
            for s in draw_seeds:
                frozen_seed_raw_lines[s].append(reserve_seed_lines[s][new_qid])
            replacements.append(
                {
                    "position": pos,
                    "old_qid": qid,
                    "old_article_index": int(q["article_index"]),
                    "old_article_title": str(q["article_title"]),
                    "reason": excl_info["reason"],
                    "missing_seeds": excl_info["missing_seeds"],
                    "audit_flagged": excl_info["audit_flagged"],
                    "new_qid": new_qid,
                    "new_article_index": int(new_q["article_index"]),
                    "new_article_title": str(new_q["article_title"]),
                    "same_article": bool(same_article),
                    "pass": pass_num,
                }
            )
        else:
            frozen_main.append(q)
            for s in draw_seeds:
                frozen_seed_raw_lines[s].append(main_seed_lines[s][qid])

    if len(frozen_main) != expected_n:
        raise RuntimeError(
            f"Frozen main split has {len(frozen_main)} questions, expected {expected_n}"
        )

    final_qids = [str(q["question_id"]) for q in frozen_main]
    if len(set(final_qids)) != expected_n:
        raise RuntimeError("Duplicate question_id detected in frozen main split")
    if set(final_qids) & audit_excluded_set:
        raise RuntimeError("Audit-excluded question_id leaked into frozen main split")

    per_article_counts = _compute_per_article_counts(
        main_questions=main_questions,
        frozen_questions=frozen_main,
        sampling_manifest=sampling_manifest,
        replacements=replacements,
    )

    same_article_count = sum(1 for r in replacements if r["same_article"])
    cross_article_count = len(replacements) - same_article_count
    counts = {
        "final_n": len(frozen_main),
        "original_main": len(main_questions),
        "retained": len(main_questions) - len(replacements),
        "replaced": len(replacements),
        "incomplete_seed_exclusions": len(incomplete_seed_ids),
        "audit_exclusions": len(audit_in_main_ids),
        "overlap": len(overlap_ids),
        "reserve_total": len(reserve_questions),
        "eligible_reserve": len(eligible_reserve),
        "ineligible_reserve": len(ineligible_reserve),
        "same_article_replacements": same_article_count,
        "cross_article_replacements": cross_article_count,
        "unused_reserve": len(eligible_reserve) - len(replacements),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    main_json_path = out_dir / "main.json"
    main_json_path.write_text(
        json.dumps(frozen_main, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    output_files: dict[str, dict[str, Any]] = {
        "main": {
            "path": "data/frozen/main.json",
            "count": len(frozen_main),
            "sha256": sha256_file(main_json_path),
        }
    }

    for s in draw_seeds:
        seed_path = out_dir / f"main_seed{s}.jsonl"
        payload = b"\n".join(frozen_seed_raw_lines[s]) + b"\n"
        seed_path.write_bytes(payload)
        output_files[f"main_seed{s}"] = {
            "path": f"data/frozen/main_seed{s}.jsonl",
            "count": len(frozen_seed_raw_lines[s]),
            "sha256": sha256_file(seed_path),
        }

    report_text = _render_freeze_report(
        counts=counts,
        replacements=replacements,
        per_article_counts=per_article_counts,
        ineligible_reserve=ineligible_reserve,
        raw_dev_sha256=raw_dev_sha256,
        sampling_manifest_sha256=sampling_manifest_sha256,
        exclusions_sha256=exclusions_sha256,
        output_files=output_files,
    )
    report_path = out_dir / "freeze_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    output_files["freeze_report"] = {
        "path": "data/frozen/freeze_report.md",
        "sha256": sha256_file(report_path),
    }

    manifest: dict[str, Any] = {
        "decision": "D12+P7",
        "replacement_rule": "two_pass_same_article_then_cross_article_fallback",
        "replacement_rule_description": (
            "Pass 1 iterates excluded/incomplete main.json questions in order and assigns "
            "the first unused eligible reserve question from the same article_index in "
            "reserve.json order. Pass 2 iterates remaining unassigned exclusions in "
            "main.json order and assigns each to the next unused eligible reserve question "
            "in reserve.json order."
        ),
        "raw_dev_sha256": raw_dev_sha256,
        "sampling_manifest_sha256": sampling_manifest_sha256,
        "exclusions_sha256": exclusions_sha256,
        "final_n": len(frozen_main),
        "counts": counts,
        "incomplete_seed_excluded_ids": incomplete_seed_ids,
        "audit_excluded_ids": audit_in_main_ids,
        "overlap_excluded_ids": overlap_ids,
        "ineligible_reserve_records": ineligible_reserve,
        "question_ids": final_qids,
        "replacements": replacements,
        "per_article_counts": per_article_counts,
        "output_files": output_files,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = sha256_file(manifest_path)

    manifest_sha_path = out_dir / "manifest.sha256"
    manifest_sha_path.write_text(
        f"{manifest_sha256}  data/frozen/manifest.json\n",
        encoding="utf-8",
    )

    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "out_dir": out_dir,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute P7 input freeze (Decision D12 + 2-pass replacement) to produce data/frozen/."
    )
    parser.add_argument("--prereg-path", type=Path, default=DEFAULT_PREREG_PATH)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--sampling-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--exclusions-path", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = freeze_inputs(
        prereg_path=args.prereg_path,
        raw_path=args.raw_path,
        sampling_dir=args.sampling_dir,
        inputs_dir=args.inputs_dir,
        exclusions_path=args.exclusions_path,
        out_dir=args.out_dir,
    )
    manifest = result["manifest"]
    c = manifest["counts"]
    print(
        f"[Freeze Inputs] final_n={c['final_n']} retained={c['retained']} "
        f"replaced={c['replaced']} (incomplete={c['incomplete_seed_exclusions']}, "
        f"audit={c['audit_exclusions']}, overlap={c['overlap']}) | "
        f"same_article={c['same_article_replacements']} cross_article={c['cross_article_replacements']}"
    )
    print(f"  manifest.json sha256: {result['manifest_sha256']}")
    for key in ("main", "main_seed0", "main_seed1", "main_seed2"):
        info = manifest["output_files"][key]
        print(f"  {key:10s}: count={info['count']:3d} | sha256={info['sha256']} | {info['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
