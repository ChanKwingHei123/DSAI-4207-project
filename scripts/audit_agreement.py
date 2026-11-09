#!/usr/bin/env python3
"""Compute inter-annotator agreement (Cohen's kappa, percent agreement, disagreements) and write exclusions.json.

Reads completed `data/audit/annotator_A.csv` and `data/audit/annotator_B.csv`, aligns rows by
`question_id`, reports Cohen's kappa, percent agreement, and the list of disagreements.
When an `adjudicated_flag` column is filled (in an adjudicated CSV or in the annotator sheet(s)),
writes `data/audit/exclusions.json` containing ONLY questions with `adjudicated_flag == 1`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DIR = REPO_ROOT / "data" / "audit"
DEFAULT_CSV_A = DEFAULT_AUDIT_DIR / "annotator_A.csv"
DEFAULT_CSV_B = DEFAULT_AUDIT_DIR / "annotator_B.csv"
DEFAULT_EXCLUSIONS_PATH = DEFAULT_AUDIT_DIR / "exclusions.json"


def parse_binary_flag(val: Any, field_name: str, question_id: str) -> int:
    """Parse a binary 0/1 flag value strictly."""
    s = str(val if val is not None else "").strip()
    if s in ("0", "0.0"):
        return 0
    if s in ("1", "1.0"):
        return 1
    raise ValueError(
        f"Invalid or empty {field_name} value {val!r} for question_id={question_id!r}; expected 0 or 1"
    )


def parse_optional_binary_flag(val: Any, field_name: str, question_id: str) -> int | None:
    """Parse an optional binary 0/1 flag value; return None if blank."""
    s = str(val if val is not None else "").strip()
    if s == "":
        return None
    return parse_binary_flag(s, field_name=field_name, question_id=question_id)


def compute_cohens_kappa(flags_a: list[int], flags_b: list[int]) -> dict[str, Any]:
    """Compute binary percent agreement, confusion counts, and Cohen's kappa.

    Formula:
      p_o = (n_11 + n_00) / N
      p_e = ((n_11 + n_10)/N) * ((n_11 + n_01)/N) + ((n_01 + n_00)/N) * ((n_10 + n_00)/N)
      kappa = (p_o - p_e) / (1 - p_e)   (1.0 when p_e == 1.0 and p_o == 1.0)
    """
    if len(flags_a) != len(flags_b):
        raise ValueError(
            f"Length mismatch between annotator A ({len(flags_a)}) and annotator B ({len(flags_b)})"
        )
    n = len(flags_a)
    if n == 0:
        raise ValueError("Cannot compute agreement on empty flag lists")

    n_11 = 0
    n_10 = 0
    n_01 = 0
    n_00 = 0

    for a, b in zip(flags_a, flags_b):
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError(f"Flags must be 0 or 1, got ({a}, {b})")
        if a == 1 and b == 1:
            n_11 += 1
        elif a == 1 and b == 0:
            n_10 += 1
        elif a == 0 and b == 1:
            n_01 += 1
        else:
            n_00 += 1

    agreed = n_11 + n_00
    p_o = agreed / n
    p_a1 = (n_11 + n_10) / n
    p_b1 = (n_11 + n_01) / n
    p_a0 = (n_01 + n_00) / n
    p_b0 = (n_10 + n_00) / n
    p_e = (p_a1 * p_b1) + (p_a0 * p_b0)

    if abs(1.0 - p_e) < 1e-15:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-15 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "n": n,
        "agreed_count": agreed,
        "disagreed_count": n - agreed,
        "percent_agreement": p_o * 100.0,
        "proportion_agreement": p_o,
        "expected_agreement": p_e,
        "cohens_kappa": kappa,
        "counts": {
            "n_11": n_11,
            "n_10": n_10,
            "n_01": n_01,
            "n_00": n_00,
        },
    }


def _read_csv_by_qid(csv_path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    by_qid: dict[str, dict[str, str]] = {}
    for row in rows:
        qid = (row.get("question_id") or "").strip()
        if not qid:
            raise ValueError(f"Row with missing question_id in {csv_path}")
        if qid in by_qid:
            raise ValueError(f"Duplicate question_id {qid!r} in {csv_path}")
        by_qid[qid] = row
    return rows, by_qid


def evaluate_audit_agreement(
    csv_a_path: Path = DEFAULT_CSV_A,
    csv_b_path: Path = DEFAULT_CSV_B,
    adjudicated_csv_path: Path | None = None,
    exclusions_out_path: Path | None = DEFAULT_EXCLUSIONS_PATH,
) -> dict[str, Any]:
    """Read both completed CSVs, compute agreement and disagreements, and write exclusions.json if adjudicated_flag is present."""
    rows_a, by_qid_a = _read_csv_by_qid(csv_a_path)
    rows_b, by_qid_b = _read_csv_by_qid(csv_b_path)

    qids_a = set(by_qid_a.keys())
    qids_b = set(by_qid_b.keys())
    if qids_a != qids_b:
        only_a = sorted(qids_a - qids_b)
        only_b = sorted(qids_b - qids_a)
        raise ValueError(
            f"Question sets in {csv_a_path.name} and {csv_b_path.name} do not match: "
            f"only_in_A={only_a[:5]}, only_in_B={only_b[:5]}"
        )

    # Preserve canonical question_id ordering sorted by question_id (or order in A)
    ordered_qids = [r["question_id"].strip() for r in rows_a]

    flags_a: list[int] = []
    flags_b: list[int] = []
    disagreements: list[dict[str, Any]] = []

    for qid in ordered_qids:
        ra = by_qid_a[qid]
        rb = by_qid_b[qid]
        fa = parse_binary_flag(ra.get("flagged"), field_name="flagged (A)", question_id=qid)
        fb = parse_binary_flag(rb.get("flagged"), field_name="flagged (B)", question_id=qid)
        flags_a.append(fa)
        flags_b.append(fb)

        if fa != fb:
            disagreements.append(
                {
                    "question_id": qid,
                    "question": ra.get("question", ""),
                    "annotator_A_flagged": fa,
                    "annotator_A_which_passage": (ra.get("which_passage") or "").strip(),
                    "annotator_A_note": (ra.get("note") or "").strip(),
                    "annotator_B_flagged": fb,
                    "annotator_B_which_passage": (rb.get("which_passage") or "").strip(),
                    "annotator_B_note": (rb.get("note") or "").strip(),
                }
            )

    stats = compute_cohens_kappa(flags_a, flags_b)

    # Check for adjudicated_flag column
    adj_by_qid: dict[str, int] = {}
    if adjudicated_csv_path is not None:
        _, by_qid_adj = _read_csv_by_qid(adjudicated_csv_path)
        for qid in ordered_qids:
            if qid in by_qid_adj and "adjudicated_flag" in by_qid_adj[qid]:
                val = parse_optional_binary_flag(
                    by_qid_adj[qid].get("adjudicated_flag"),
                    field_name="adjudicated_flag",
                    question_id=qid,
                )
                if val is not None:
                    adj_by_qid[qid] = val
    else:
        for qid in ordered_qids:
            ra = by_qid_a[qid]
            rb = by_qid_b[qid]
            val_a = (
                parse_optional_binary_flag(
                    ra.get("adjudicated_flag"),
                    field_name="adjudicated_flag (A)",
                    question_id=qid,
                )
                if "adjudicated_flag" in ra
                else None
            )
            val_b = (
                parse_optional_binary_flag(
                    rb.get("adjudicated_flag"),
                    field_name="adjudicated_flag (B)",
                    question_id=qid,
                )
                if "adjudicated_flag" in rb
                else None
            )
            if val_a is not None and val_b is not None and val_a != val_b:
                raise ValueError(
                    f"Conflicting adjudicated_flag between A ({val_a}) and B ({val_b}) for question_id={qid!r}"
                )
            resolved = val_a if val_a is not None else val_b
            if resolved is not None:
                adj_by_qid[qid] = resolved

    exclusions: list[dict[str, Any]] | None = None
    if adj_by_qid:
        # Exclusions come ONLY from questions with adjudicated_flag == 1
        exclusions = []
        for qid in ordered_qids:
            if adj_by_qid.get(qid) == 1:
                ra = by_qid_a[qid]
                rb = by_qid_b[qid]
                exclusions.append(
                    {
                        "question_id": qid,
                        "question": ra.get("question", ""),
                        "adjudicated_flag": 1,
                        "annotator_A_flagged": parse_binary_flag(
                            ra.get("flagged"), field_name="flagged (A)", question_id=qid
                        ),
                        "annotator_B_flagged": parse_binary_flag(
                            rb.get("flagged"), field_name="flagged (B)", question_id=qid
                        ),
                        "annotator_A_which_passage": (ra.get("which_passage") or "").strip(),
                        "annotator_B_which_passage": (rb.get("which_passage") or "").strip(),
                        "annotator_A_note": (ra.get("note") or "").strip(),
                        "annotator_B_note": (rb.get("note") or "").strip(),
                    }
                )
        if exclusions_out_path is not None:
            exclusions_out_path.parent.mkdir(parents=True, exist_ok=True)
            exclusions_out_path.write_text(
                json.dumps(exclusions, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    return {
        "n": stats["n"],
        "agreed_count": stats["agreed_count"],
        "disagreed_count": stats["disagreed_count"],
        "percent_agreement": stats["percent_agreement"],
        "proportion_agreement": stats["proportion_agreement"],
        "expected_agreement": stats["expected_agreement"],
        "cohens_kappa": stats["cohens_kappa"],
        "counts": stats["counts"],
        "disagreements": disagreements,
        "adjudicated_count": len(adj_by_qid),
        "exclusions": exclusions,
        "exclusions_path": str(exclusions_out_path)
        if exclusions is not None and exclusions_out_path is not None
        else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa, percent agreement, and disagreements; write exclusions.json when adjudicated_flag is present."
    )
    parser.add_argument("--csv-a", type=Path, default=DEFAULT_CSV_A)
    parser.add_argument("--csv-b", type=Path, default=DEFAULT_CSV_B)
    parser.add_argument(
        "--adjudicated-csv",
        type=Path,
        default=None,
        help="Optional CSV containing an adjudicated_flag column (if not added directly to annotator_A.csv or annotator_B.csv).",
    )
    parser.add_argument(
        "--exclusions-out",
        type=Path,
        default=DEFAULT_EXCLUSIONS_PATH,
        help="Output path for exclusions.json (written when adjudicated_flag is present).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_audit_agreement(
        csv_a_path=args.csv_a,
        csv_b_path=args.csv_b,
        adjudicated_csv_path=args.adjudicated_csv,
        exclusions_out_path=args.exclusions_out,
    )

    c = result["counts"]
    print("=== Ambiguity Audit Agreement Summary ===")
    print(f"Total questions evaluated: {result['n']}")
    print(
        f"Confusion matrix (A vs B): n_11={c['n_11']}, n_10={c['n_10']}, n_01={c['n_01']}, n_00={c['n_00']}"
    )
    print(f"Percent agreement:         {result['percent_agreement']:.2f}% ({result['agreed_count']}/{result['n']})")
    print(f"Cohen's kappa:             {result['cohens_kappa']:.4f}")
    print(f"Disagreements count:       {result['disagreed_count']}")

    if result["disagreements"]:
        print("\n=== Disagreements ===")
        for d in result["disagreements"]:
            print(
                f"  - {d['question_id']}: A={d['annotator_A_flagged']} (passage={d['annotator_A_which_passage']!r}) "
                f"| B={d['annotator_B_flagged']} (passage={d['annotator_B_which_passage']!r}) "
                f"| Q: {d['question']}"
            )

    if result["exclusions"] is not None:
        print(
            f"\n[Adjudication] Wrote {len(result['exclusions'])} excluded question(s) "
            f"(adjudicated_flag == 1) to {result['exclusions_path']}"
        )
    else:
        print(
            "\n[Adjudication] No adjudicated_flag column values found yet; exclusions.json was not written."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
