#!/usr/bin/env python3
"""Process 1 of dense retrieval: embed SQuAD 2.0 dev paragraphs and sampled questions.

Hard Rule 11 compliance:
- Computes embeddings with sentence-transformers and saves .npy arrays to disk.
- NEVER imports faiss in this process.
- NEVER sets KMP_DUPLICATE_LIB_OK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_INITIAL_KMP_DUPLICATE_LIB_OK = os.environ.get("KMP_DUPLICATE_LIB_OK")
if _INITIAL_KMP_DUPLICATE_LIB_OK is not None:
    raise RuntimeError(
        "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK was set in the environment prior to omp_guard."
    )

import src.omp_guard  # noqa: F401, E402
import numpy as np  # noqa: E402

DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_SAMPLING_DIR = REPO_ROOT / "data" / "sampling"
DEFAULT_INDEX_DIR = REPO_ROOT / "results" / "index"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SPLITS = ("main", "pilot", "reserve")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def assert_no_kmp_duplicate_lib_ok() -> None:
    """Enforce AGENTS.md Hard Rule 11: caller must never set KMP_DUPLICATE_LIB_OK."""
    if _INITIAL_KMP_DUPLICATE_LIB_OK is not None:
        raise RuntimeError(
            "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK is set in environment."
        )


def load_dev_paragraphs(raw_path: Path) -> list[dict[str, Any]]:
    """Extract all unique paragraphs from SQuAD 2.0 dev in (article_index, paragraph_index) order."""
    if not raw_path.exists():
        print(
            f"ERROR: Raw dataset file is missing: {raw_path}\n"
            "Run the following command to download it:\n"
            f"  mkdir -p {raw_path.parent} && curl -fsSLo {raw_path} {SQUAD_DEV_URL}",
            file=sys.stderr,
        )
        raise FileNotFoundError(f"Missing raw SQuAD 2.0 dev file: {raw_path}")

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    articles = payload.get("data", [])

    paragraphs: list[dict[str, Any]] = []
    seen_texts: dict[str, list[int]] = {}

    for art_idx, article in enumerate(articles):
        title = article["title"]
        for para_idx, para in enumerate(article["paragraphs"]):
            context = para["context"]
            para_id = [art_idx, para_idx]
            if context in seen_texts:
                # Deduplicate identical paragraph text if any ever occurred while logging canonical ID
                continue
            seen_texts[context] = para_id
            paragraphs.append(
                {
                    "pool_index": len(paragraphs),
                    "paragraph_id": para_id,
                    "article_index": art_idx,
                    "paragraph_index": para_idx,
                    "article_title": title,
                    "context": context,
                    "text_sha256": sha256_bytes(context.encode("utf-8")),
                }
            )

    return paragraphs


def load_sampled_questions(
    sampling_dir: Path, raw_path: Path
) -> list[dict[str, Any]]:
    """Load questions across main, pilot, and reserve splits (building sampling frame if needed)."""
    missing_splits = [
        s for s in SPLITS if not (sampling_dir / f"{s}.json").exists()
    ]
    if missing_splits:
        from scripts.make_sampling_frame import build_sampling_frame

        build_sampling_frame(raw_path=raw_path, out_dir=sampling_dir)

    questions: list[dict[str, Any]] = []
    for split_name in SPLITS:
        split_path = sampling_dir / f"{split_name}.json"
        records = json.loads(split_path.read_text(encoding="utf-8"))
        for rec in records:
            questions.append(
                {
                    "query_index": len(questions),
                    "question_id": rec["question_id"],
                    "split": split_name,
                    "gold_paragraph_id": list(rec["paragraph_id"]),
                    "article_index": int(rec["article_index"]),
                    "article_title": rec["article_title"],
                    "question_text": rec["question_text"],
                }
            )
    return questions


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts into L2-normalized float32 embeddings using sentence-transformers."""
    assert_no_kmp_duplicate_lib_ok()
    assert "faiss" not in sys.modules, "Hard Rule 11 violation: faiss imported in embedding process"

    import torch
    from sentence_transformers import SentenceTransformer

    torch.use_deterministic_algorithms(True, warn_only=True)
    model = SentenceTransformer(
        model_name,
        local_files_only=True,
    )
    model.eval()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Verify unit L2 norms
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not L2-normalized to unit length.")

    return embeddings


def run_embedding_stage(
    raw_path: Path = DEFAULT_RAW_PATH,
    sampling_dir: Path = DEFAULT_SAMPLING_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Compute and save paragraph and question embeddings + metadata to index_dir."""
    assert_no_kmp_duplicate_lib_ok()

    paragraphs = load_dev_paragraphs(raw_path)
    questions = load_sampled_questions(sampling_dir, raw_path=raw_path)

    para_texts = [p["context"] for p in paragraphs]
    question_texts = [q["question_text"] for q in questions]

    para_embeddings = embed_texts(para_texts, model_name=model_name, batch_size=batch_size)
    question_embeddings = embed_texts(question_texts, model_name=model_name, batch_size=batch_size)

    index_dir.mkdir(parents=True, exist_ok=True)

    para_emb_path = index_dir / "paragraph_embeddings.npy"
    q_emb_path = index_dir / "question_embeddings.npy"
    para_meta_path = index_dir / "paragraphs.json"
    q_meta_path = index_dir / "questions.json"

    np.save(para_emb_path, para_embeddings)
    np.save(q_emb_path, question_embeddings)

    # Store compact metadata (without duplicating raw context text in paragraphs.json)
    para_meta_records = [
        {
            "pool_index": p["pool_index"],
            "paragraph_id": p["paragraph_id"],
            "article_index": p["article_index"],
            "paragraph_index": p["paragraph_index"],
            "article_title": p["article_title"],
            "text_sha256": p["text_sha256"],
        }
        for p in paragraphs
    ]
    q_meta_records = [
        {
            "query_index": q["query_index"],
            "question_id": q["question_id"],
            "split": q["split"],
            "gold_paragraph_id": q["gold_paragraph_id"],
            "article_index": q["article_index"],
            "article_title": q["article_title"],
        }
        for q in questions
    ]

    para_meta_path.write_text(
        json.dumps(para_meta_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    q_meta_path.write_text(
        json.dumps(q_meta_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "model_name": model_name,
        "dim": int(para_embeddings.shape[1]),
        "paragraph_count": int(para_embeddings.shape[0]),
        "question_count": int(question_embeddings.shape[0]),
        "paragraph_embeddings_path": str(para_emb_path),
        "paragraph_embeddings_sha256": sha256_file(para_emb_path),
        "question_embeddings_path": str(q_emb_path),
        "question_embeddings_sha256": sha256_file(q_emb_path),
        "paragraphs_meta_sha256": sha256_file(para_meta_path),
        "questions_meta_sha256": sha256_file(q_meta_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed SQuAD 2.0 dev paragraphs and sampled questions (Process 1)."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--sampling-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        info = run_embedding_stage(
            raw_path=args.raw_path,
            sampling_dir=args.sampling_dir,
            index_dir=args.index_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
    except FileNotFoundError:
        return 1

    print(
        f"[Process 1: Embeddings OK] model={info['model_name']} "
        f"paragraphs={info['paragraph_count']} questions={info['question_count']} "
        f"dim={info['dim']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
