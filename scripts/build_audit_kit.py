#!/usr/bin/env python3
"""Build the blind ambiguity-audit kit (Step p2d).

Samples 100 questions stratified by article from `data/sampling/main.json` restricted to
questions that have a row in `data/inputs/main_seed0.jsonl`, using `np.random.default_rng([master_seed, 1, 0])`.
Writes:
- `data/audit/audit_sample.json`: sampled 100 questions (no scores, ranks, article titles, or model outputs)
- `data/audit/annotator_A.csv` and `data/audit/annotator_B.csv`: same 100 questions in different row orders,
  with 8 seed-0 hard-negative passages shown in randomized order without retrieval scores, ranks,
  similarity values, or article titles, plus empty columns `flagged`, `which_passage`, `note`.
- `data/audit/key.json`: maps each question's displayed passage indices (`"1"`..`"8"`) to their `[article_index, paragraph_index]` IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_sampling_frame import largest_remainder_allocate, sha256_file  # noqa: E402

DEFAULT_PREREG_PATH = REPO_ROOT / "configs" / "prereg.yaml"
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_MAIN_SAMPLING_PATH = REPO_ROOT / "data" / "sampling" / "main.json"
DEFAULT_MAIN_SEED0_PATH = REPO_ROOT / "data" / "inputs" / "main_seed0.jsonl"
DEFAULT_AUDIT_DIR = REPO_ROOT / "data" / "audit"
AUDIT_SAMPLE_SIZE = 100

CSV_COLUMNS: tuple[str, ...] = (
    "question_id",
    "question",
    "gold_answers",
    "gold_passage",
    "passage_1",
    "passage_2",
    "passage_3",
    "passage_4",
    "passage_5",
    "passage_6",
    "passage_7",
    "passage_8",
    "flagged",
    "which_passage",
    "note",
)


def load_paragraph_texts(raw_path: Path) -> tuple[int, dict[tuple[int, int], str]]:
    """Load mapping from `(article_index, paragraph_index)` to paragraph text (`context`) from SQuAD 2.0 dev."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing SQuAD 2.0 dev file: {raw_path}")

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    articles = payload.get("data", [])
    by_pid: dict[tuple[int, int], str] = {}

    for art_idx, article in enumerate(articles):
        for para_idx, para in enumerate(article.get("paragraphs", [])):
            by_pid[(int(art_idx), int(para_idx))] = str(para["context"])

    return len(articles), by_pid


def format_gold_answers(gold_answers: list[str]) -> str:
    """Format all gold answers into a single pipe-delimited string preserving order."""
    return " | ".join(str(ans) for ans in gold_answers)


def build_audit_kit(
    prereg_path: Path = DEFAULT_PREREG_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    main_sampling_path: Path = DEFAULT_MAIN_SAMPLING_PATH,
    main_seed0_path: Path = DEFAULT_MAIN_SEED0_PATH,
    out_dir: Path = DEFAULT_AUDIT_DIR,
    target_n: int = AUDIT_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Sample 100 questions stratified by article and write audit_sample.json, annotator_A.csv, annotator_B.csv, and key.json."""
    if not prereg_path.exists():
        raise FileNotFoundError(f"Missing prereg config: {prereg_path}")
    if not main_sampling_path.exists():
        raise FileNotFoundError(f"Missing main sampling file: {main_sampling_path}")
    if not main_seed0_path.exists():
        raise FileNotFoundError(f"Missing main_seed0 inputs file: {main_seed0_path}")

    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    master_seed = int(prereg["rng"]["master_seed"])
    seed_sequence = [master_seed, 1, 0]
    rng = np.random.default_rng(seed_sequence)

    num_articles, para_text_by_pid = load_paragraph_texts(raw_path)

    main_questions: list[dict[str, Any]] = json.loads(
        main_sampling_path.read_text(encoding="utf-8")
    )
    seed0_by_qid: dict[str, dict[str, Any]] = {}
    for line in main_seed0_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            seed0_by_qid[rec["question_id"]] = rec

    # Filter main.json to questions that have a row in data/inputs/main_seed0.jsonl
    eligible_main = [q for q in main_questions if q["question_id"] in seed0_by_qid]
    if len(eligible_main) < target_n:
        raise ValueError(
            f"Only {len(eligible_main)} eligible questions in main_seed0, fewer than target_n={target_n}"
        )

    # Group eligible questions by article_index preserving order in main.json
    max_art_idx = max(
        num_articles - 1,
        max(int(q["article_index"]) for q in eligible_main),
    )
    article_pools: list[list[dict[str, Any]]] = [[] for _ in range(max_art_idx + 1)]
    for q in eligible_main:
        article_pools[int(q["article_index"])].append(q)

    counts = [len(pool) for pool in article_pools]
    allocations = largest_remainder_allocate(counts, target_n)

    sampled_questions: list[dict[str, Any]] = []
    for pool, k in zip(article_pools, allocations):
        if k > 0:
            chosen_idx = set(rng.choice(len(pool), size=k, replace=False).tolist())
            sampled_questions.extend(pool[i] for i in range(len(pool)) if i in chosen_idx)

    assert len(sampled_questions) == target_n, (
        f"Expected {target_n} sampled questions, got {len(sampled_questions)}"
    )

    # Construct audit_sample.json records, CSV base rows, and hidden key.json
    audit_sample_records: list[dict[str, Any]] = []
    csv_base_rows: list[dict[str, str]] = []
    key_mapping: dict[str, dict[str, list[int]]] = {}

    for q in sampled_questions:
        qid = str(q["question_id"])
        seed0_row = seed0_by_qid[qid]
        q_text = str(q["question_text"])
        gold_answers = [str(a) for a in q["gold_answers"]]
        gold_pid = [int(seed0_row["gold_paragraph_id"][0]), int(seed0_row["gold_paragraph_id"][1])]
        gold_passage_text = para_text_by_pid[(gold_pid[0], gold_pid[1])]

        hard_pids: list[list[int]] = [
            [int(pid[0]), int(pid[1])] for pid in seed0_row["hard_paragraph_ids"]
        ]
        if len(hard_pids) != 8:
            raise ValueError(
                f"Expected 8 hard_paragraph_ids for question {qid}, got {len(hard_pids)}"
            )

        # Permute the 8 hard negatives in randomized order
        perm = rng.permutation(8).tolist()
        displayed_pids = [hard_pids[idx] for idx in perm]
        displayed_texts = [para_text_by_pid[(pid[0], pid[1])] for pid in displayed_pids]

        key_mapping[qid] = {
            str(disp_idx): displayed_pids[disp_idx - 1] for disp_idx in range(1, 9)
        }

        audit_sample_records.append(
            {
                "question_id": qid,
                "article_index": int(q["article_index"]),
                "paragraph_id": gold_pid,
                "question_text": q_text,
                "gold_answers": gold_answers,
            }
        )

        row_dict: dict[str, str] = {
            "question_id": qid,
            "question": q_text,
            "gold_answers": format_gold_answers(gold_answers),
            "gold_passage": gold_passage_text,
        }
        for disp_idx in range(1, 9):
            row_dict[f"passage_{disp_idx}"] = displayed_texts[disp_idx - 1]
        row_dict["flagged"] = ""
        row_dict["which_passage"] = ""
        row_dict["note"] = ""
        csv_base_rows.append(row_dict)

    # Create two distinct randomized row orders for annotator_A and annotator_B
    order_a = rng.permutation(len(csv_base_rows)).tolist()
    order_b = rng.permutation(len(csv_base_rows)).tolist()
    while order_b == order_a and len(csv_base_rows) > 1:
        order_b = rng.permutation(len(csv_base_rows)).tolist()

    rows_a = [csv_base_rows[i] for i in order_a]
    rows_b = [csv_base_rows[i] for i in order_b]

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "audit_sample.json"
    csv_a_path = out_dir / "annotator_A.csv"
    csv_b_path = out_dir / "annotator_B.csv"
    key_path = out_dir / "key.json"

    sample_path.write_text(
        json.dumps(audit_sample_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for csv_path, rows in ((csv_a_path, rows_a), (csv_b_path, rows_b)):
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    key_path.write_text(
        json.dumps(key_mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "seed_sequence": seed_sequence,
        "eligible_main_count": len(eligible_main),
        "sampled_count": len(sampled_questions),
        "allocations_by_article": allocations,
        "files": {
            "audit_sample": {
                "path": str(sample_path.relative_to(REPO_ROOT))
                if sample_path.is_relative_to(REPO_ROOT)
                else str(sample_path),
                "sha256": sha256_file(sample_path),
            },
            "annotator_A": {
                "path": str(csv_a_path.relative_to(REPO_ROOT))
                if csv_a_path.is_relative_to(REPO_ROOT)
                else str(csv_a_path),
                "sha256": sha256_file(csv_a_path),
            },
            "annotator_B": {
                "path": str(csv_b_path.relative_to(REPO_ROOT))
                if csv_b_path.is_relative_to(REPO_ROOT)
                else str(csv_b_path),
                "sha256": sha256_file(csv_b_path),
            },
            "key": {
                "path": str(key_path.relative_to(REPO_ROOT))
                if key_path.is_relative_to(REPO_ROOT)
                else str(key_path),
                "sha256": sha256_file(key_path),
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the blind ambiguity-audit kit (audit_sample.json, annotator_A.csv, annotator_B.csv, key.json)."
    )
    parser.add_argument("--prereg-path", type=Path, default=DEFAULT_PREREG_PATH)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--main-sampling-path", type=Path, default=DEFAULT_MAIN_SAMPLING_PATH
    )
    parser.add_argument("--main-seed0-path", type=Path, default=DEFAULT_MAIN_SEED0_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--target-n", type=int, default=AUDIT_SAMPLE_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_audit_kit(
        prereg_path=args.prereg_path,
        raw_path=args.raw_path,
        main_sampling_path=args.main_sampling_path,
        main_seed0_path=args.main_seed0_path,
        out_dir=args.out_dir,
        target_n=args.target_n,
    )
    print(
        f"[Audit Kit Built] eligible_main={summary['eligible_main_count']} "
        f"sampled={summary['sampled_count']} seed={summary['seed_sequence']}"
    )
    for name, info in summary["files"].items():
        print(f"  {name:14s}: sha256={info['sha256']} | {info['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
