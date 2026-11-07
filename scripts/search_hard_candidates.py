#!/usr/bin/env python3
"""Process 2 of dense retrieval: build FAISS IndexFlatIP, retrieve top-30 hard negatives, and run gold rank diagnostics.

Hard Rule 11 compliance:
- Loads precomputed .npy embeddings from disk and runs FAISS search in this process.
- NEVER imports torch or sentence_transformers in this process.
- NEVER sets KMP_DUPLICATE_LIB_OK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = REPO_ROOT / "results" / "index"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "data" / "candidates" / "hard_candidates.jsonl"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 30

# Diagnostic sanity thresholds (diagnostic only; never tuned on)
LOW_RECALL_AT_1_THRESHOLD = 0.50
LOW_RECALL_AT_5_THRESHOLD = 0.75


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_process_isolation() -> None:
    """Enforce AGENTS.md Hard Rule 11 in the FAISS search process."""
    if "KMP_DUPLICATE_LIB_OK" in os.environ:
        raise RuntimeError(
            "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK is set in environment."
        )
    for forbidden in ("torch", "sentence_transformers"):
        if forbidden in sys.modules:
            raise RuntimeError(
                f"Hard Rule 11 violation: {forbidden} imported in FAISS search process."
            )


def compute_rank_summary(ranks: list[int]) -> dict[str, Any]:
    """Compute recall@1, recall@5, MRR, and rank summary stats from 1-indexed gold ranks."""
    n = len(ranks)
    if n == 0:
        return {
            "total_questions": 0,
            "hits_at_1": 0,
            "hits_at_5": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "rank_min": 0,
            "rank_median": 0.0,
            "rank_max": 0,
        }
    hits_1 = sum(1 for r in ranks if r <= 1)
    hits_5 = sum(1 for r in ranks if r <= 5)
    mrr = sum(1.0 / r for r in ranks) / n
    return {
        "total_questions": n,
        "hits_at_1": hits_1,
        "hits_at_5": hits_5,
        "recall_at_1": round(hits_1 / n, 6),
        "recall_at_5": round(hits_5 / n, 6),
        "mrr": round(mrr, 6),
        "rank_min": int(min(ranks)),
        "rank_median": float(statistics.median(ranks)),
        "rank_max": int(max(ranks)),
    }


def run_faiss_search_stage(
    index_dir: Path = DEFAULT_INDEX_DIR,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Build FAISS inner-product index, write top-k non-gold candidates, and log diagnostics."""
    assert_process_isolation()

    para_emb_path = index_dir / "paragraph_embeddings.npy"
    q_emb_path = index_dir / "question_embeddings.npy"
    para_meta_path = index_dir / "paragraphs.json"
    q_meta_path = index_dir / "questions.json"

    for required_path in (para_emb_path, q_emb_path, para_meta_path, q_meta_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed embedding artifact: {required_path}. "
                "Run scripts/embed_dev_pool.py first."
            )

    para_embeddings = np.ascontiguousarray(np.load(para_emb_path), dtype=np.float32)
    question_embeddings = np.ascontiguousarray(np.load(q_emb_path), dtype=np.float32)
    paragraphs = json.loads(para_meta_path.read_text(encoding="utf-8"))
    questions = json.loads(q_meta_path.read_text(encoding="utf-8"))

    num_paragraphs, dim = para_embeddings.shape
    num_questions, q_dim = question_embeddings.shape
    if dim != q_dim:
        raise ValueError(f"Dimension mismatch: paragraphs={dim} vs questions={q_dim}")
    if num_paragraphs != len(paragraphs):
        raise ValueError(
            f"Paragraph count mismatch: embeddings={num_paragraphs} vs metadata={len(paragraphs)}"
        )
    if num_questions != len(questions):
        raise ValueError(
            f"Question count mismatch: embeddings={num_questions} vs metadata={len(questions)}"
        )
    if top_k >= num_paragraphs:
        raise ValueError(
            f"top_k={top_k} must be strictly less than paragraph pool size ({num_paragraphs})"
        )

    # Build and save FAISS inner-product index
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(dim)
    index.add(para_embeddings)

    faiss_index_path = index_dir / "paragraphs.faiss"
    faiss.write_index(index, str(faiss_index_path))

    # Search full pool so we obtain both exact top-k excluding gold and exact unfiltered gold rank
    scores_matrix, indices_matrix = index.search(question_embeddings, num_paragraphs)

    para_id_by_pool_idx: list[list[int]] = [
        [int(p["paragraph_id"][0]), int(p["paragraph_id"][1])] for p in paragraphs
    ]

    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    all_gold_ranks: list[int] = []
    split_gold_ranks: dict[str, list[int]] = {"main": [], "pilot": [], "reserve": []}

    with candidates_path.open("w", encoding="utf-8") as out_f:
        for q_idx, q_meta in enumerate(questions):
            q_id = q_meta["question_id"]
            split_name = q_meta["split"]
            gold_pid = [
                int(q_meta["gold_paragraph_id"][0]),
                int(q_meta["gold_paragraph_id"][1]),
            ]

            row_scores = scores_matrix[q_idx]
            row_indices = indices_matrix[q_idx]

            # Deterministic sort: descending score, tie-breaking by ascending (article_idx, paragraph_idx)
            ranked_items = sorted(
                (
                    (float(row_scores[pos]), para_id_by_pool_idx[int(row_indices[pos])])
                    for pos in range(num_paragraphs)
                ),
                key=lambda item: (-item[0], item[1][0], item[1][1]),
            )

            gold_rank: int | None = None
            filtered_pids: list[list[int]] = []
            filtered_scores: list[float] = []

            for rank_1idx, (score, pid) in enumerate(ranked_items, start=1):
                if pid == gold_pid:
                    gold_rank = rank_1idx
                    continue
                if len(filtered_pids) < top_k:
                    filtered_pids.append(pid)
                    filtered_scores.append(round(score, 8))

            if gold_rank is None:
                raise ValueError(
                    f"Gold paragraph {gold_pid} for question {q_id} not found in paragraph pool."
                )
            if len(filtered_pids) != top_k:
                raise RuntimeError(
                    f"Expected {top_k} candidates for question {q_id}, got {len(filtered_pids)}"
                )

            all_gold_ranks.append(gold_rank)
            if split_name in split_gold_ranks:
                split_gold_ranks[split_name].append(gold_rank)

            record = {
                "question_id": q_id,
                "split": split_name,
                "gold_paragraph_id": gold_pid,
                "ranked_paragraph_ids": filtered_pids,
                "scores": filtered_scores,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    overall_diag = compute_rank_summary(all_gold_ranks)
    by_split_diag = {
        split_name: compute_rank_summary(ranks)
        for split_name, ranks in split_gold_ranks.items()
    }

    low_r1 = overall_diag["recall_at_1"] < LOW_RECALL_AT_1_THRESHOLD
    low_r5 = overall_diag["recall_at_5"] < LOW_RECALL_AT_5_THRESHOLD
    unexpectedly_low = bool(low_r1 or low_r5)
    flag_reasons: list[str] = []
    if low_r1:
        flag_reasons.append(
            f"recall@1 ({overall_diag['recall_at_1']:.4f}) < {LOW_RECALL_AT_1_THRESHOLD:.2f}"
        )
    if low_r5:
        flag_reasons.append(
            f"recall@5 ({overall_diag['recall_at_5']:.4f}) < {LOW_RECALL_AT_5_THRESHOLD:.2f}"
        )

    index_sha256 = sha256_file(faiss_index_path)
    candidates_sha256 = sha256_file(candidates_path)

    manifest = {
        "model_name": model_name,
        "dim": int(dim),
        "paragraph_count": int(num_paragraphs),
        "question_count": int(num_questions),
        "top_k": int(top_k),
        "index_type": "IndexFlatIP",
        "sha256": index_sha256,
        "files": {
            "faiss_index": {
                "path": "results/index/paragraphs.faiss",
                "sha256": index_sha256,
            },
            "paragraph_embeddings": {
                "path": "results/index/paragraph_embeddings.npy",
                "sha256": sha256_file(para_emb_path),
            },
            "question_embeddings": {
                "path": "results/index/question_embeddings.npy",
                "sha256": sha256_file(q_emb_path),
            },
            "paragraphs_meta": {
                "path": "results/index/paragraphs.json",
                "sha256": sha256_file(para_meta_path),
            },
            "questions_meta": {
                "path": "results/index/questions.json",
                "sha256": sha256_file(q_meta_path),
            },
            "hard_candidates": {
                "path": "data/candidates/hard_candidates.jsonl",
                "sha256": candidates_sha256,
            },
        },
        "diagnostics": {
            "note": "Diagnostic only (unfiltered retrieval of gold paragraph across full dev pool; not tuned on).",
            "overall": overall_diag,
            "by_split": by_split_diag,
            "thresholds": {
                "recall_at_1_min": LOW_RECALL_AT_1_THRESHOLD,
                "recall_at_5_min": LOW_RECALL_AT_5_THRESHOLD,
            },
            "unexpectedly_low": unexpectedly_low,
            "flag_reasons": flag_reasons,
        },
    }

    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS index and retrieve top-30 hard-negative candidates (Process 2)."
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_faiss_search_stage(
        index_dir=args.index_dir,
        candidates_path=args.candidates_path,
        model_name=args.model_name,
        top_k=args.top_k,
    )
    diag = manifest["diagnostics"]["overall"]
    flag_str = (
        f"FLAGGED ({', '.join(manifest['diagnostics']['flag_reasons'])})"
        if manifest["diagnostics"]["unexpectedly_low"]
        else "OK (not flagged)"
    )
    print(
        f"[Process 2: FAISS Search OK] index_sha256={manifest['sha256']} "
        f"candidates_sha256={manifest['files']['hard_candidates']['sha256']}"
    )
    print(
        f"[Gold Retrieval Diagnostic] N={diag['total_questions']} | "
        f"Recall@1={diag['recall_at_1']:.4f} ({diag['hits_at_1']}/{diag['total_questions']}) | "
        f"Recall@5={diag['recall_at_5']:.4f} ({diag['hits_at_5']}/{diag['total_questions']}) | "
        f"MRR={diag['mrr']:.4f} | Status={flag_str}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
