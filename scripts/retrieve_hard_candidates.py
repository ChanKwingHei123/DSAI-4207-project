#!/usr/bin/env python3
"""Orchestrator for dense retrieval of hard-negative candidates (AGENTS.md Hard Rule 11).

Spawns two strictly isolated OS subprocesses:
1. scripts/embed_dev_pool.py (sentence-transformers embedding -> disk .npy)
2. scripts/search_hard_candidates.py (disk .npy -> FAISS IndexFlatIP search -> hard_candidates.jsonl)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_SCRIPT = REPO_ROOT / "scripts" / "embed_dev_pool.py"
SEARCH_SCRIPT = REPO_ROOT / "scripts" / "search_hard_candidates.py"

DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_SAMPLING_DIR = REPO_ROOT / "data" / "sampling"
DEFAULT_INDEX_DIR = REPO_ROOT / "results" / "index"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "data" / "candidates" / "hard_candidates.jsonl"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 100


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embeddings_valid(index_dir: Path) -> bool:
    """Return True if precomputed .npy and metadata files exist and match manifest.json hashes."""
    manifest_path = index_dir / "manifest.json"
    para_emb = index_dir / "paragraph_embeddings.npy"
    q_emb = index_dir / "question_embeddings.npy"
    para_meta = index_dir / "paragraphs.json"
    q_meta = index_dir / "questions.json"

    for p in (manifest_path, para_emb, q_emb, para_meta, q_meta):
        if not p.exists():
            return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        return (
            _sha256_file(para_emb) == files["paragraph_embeddings"]["sha256"]
            and _sha256_file(q_emb) == files["question_embeddings"]["sha256"]
            and _sha256_file(para_meta) == files["paragraphs_meta"]["sha256"]
            and _sha256_file(q_meta) == files["questions_meta"]["sha256"]
        )
    except Exception:
        return False


def run_retrieval_pipeline(
    raw_path: Path = DEFAULT_RAW_PATH,
    sampling_dir: Path = DEFAULT_SAMPLING_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 64,
    reuse_embeddings_if_valid: bool = False,
) -> dict[str, Any]:
    """Run Process 1 (embedding) and Process 2 (FAISS search) in separate OS processes."""
    if "KMP_DUPLICATE_LIB_OK" in os.environ:
        raise RuntimeError(
            "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK is set in environment."
        )

    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)

    if not (reuse_embeddings_if_valid and _embeddings_valid(index_dir)):
        cmd_embed = [
            sys.executable,
            str(EMBED_SCRIPT),
            "--raw-path",
            str(raw_path),
            "--sampling-dir",
            str(sampling_dir),
            "--index-dir",
            str(index_dir),
            "--model-name",
            model_name,
            "--batch-size",
            str(batch_size),
        ]
        res_embed = subprocess.run(cmd_embed, env=env, check=False)
        if res_embed.returncode != 0:
            raise RuntimeError(
                f"Embedding subprocess failed with exit code {res_embed.returncode}"
            )

    cmd_search = [
        sys.executable,
        str(SEARCH_SCRIPT),
        "--index-dir",
        str(index_dir),
        "--candidates-path",
        str(candidates_path),
        "--model-name",
        model_name,
        "--top-k",
        str(top_k),
    ]
    res_search = subprocess.run(cmd_search, env=env, check=False)
    if res_search.returncode != 0:
        raise RuntimeError(
            f"FAISS search subprocess failed with exit code {res_search.returncode}"
        )

    manifest_path = index_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-process dense retrieval of hard-negative candidates."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--sampling-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_retrieval_pipeline(
            raw_path=args.raw_path,
            sampling_dir=args.sampling_dir,
            index_dir=args.index_dir,
            candidates_path=args.candidates_path,
            model_name=args.model_name,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    diag = manifest["diagnostics"]["overall"]
    by_split = manifest["diagnostics"]["by_split"]
    print("\n=== Dense Index & Hard-Negative Retrieval Summary ===")
    print(f"Model name:                 {manifest['model_name']}")
    print(f"Embedding dimension:        {manifest['dim']}")
    print(f"Paragraph pool count:       {manifest['paragraph_count']}")
    print(f"Question count:             {manifest['question_count']}")
    print(f"Top-K non-gold candidates:  {manifest['top_k']}")
    print(f"FAISS index SHA-256:        {manifest['sha256']}")
    print(
        f"hard_candidates.jsonl SHA:  {manifest['files']['hard_candidates']['sha256']}"
    )

    print("\n=== Diagnostic Only: Unfiltered Gold Paragraph Retrieval ===")
    print(
        f"Overall (N={diag['total_questions']:3d}): "
        f"Recall@1 = {diag['recall_at_1']:.4f} ({diag['hits_at_1']}/{diag['total_questions']}) | "
        f"Recall@5 = {diag['recall_at_5']:.4f} ({diag['hits_at_5']}/{diag['total_questions']}) | "
        f"MRR = {diag['mrr']:.4f} | "
        f"Gold rank (min/median/max) = {diag['rank_min']} / {diag['rank_median']} / {diag['rank_max']}"
    )
    for split_name in ("main", "pilot", "reserve"):
        s_diag = by_split[split_name]
        print(
            f"  {split_name:7s} (N={s_diag['total_questions']:3d}): "
            f"Recall@1 = {s_diag['recall_at_1']:.4f} ({s_diag['hits_at_1']}/{s_diag['total_questions']}) | "
            f"Recall@5 = {s_diag['recall_at_5']:.4f} ({s_diag['hits_at_5']}/{s_diag['total_questions']}) | "
            f"MRR = {s_diag['mrr']:.4f}"
        )
    if manifest["diagnostics"]["unexpectedly_low"]:
        print(
            f"FLAG: Gold retrieval recall is unexpectedly low: "
            f"{', '.join(manifest['diagnostics']['flag_reasons'])}"
        )
    else:
        print("Flag status:                OK (Recall@1 and Recall@5 within expected range)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
