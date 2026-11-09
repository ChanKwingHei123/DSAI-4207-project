"""Tests for dense retrieval of hard-negative candidates (Step p2b)."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.embed_dev_pool import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_RAW_PATH,
    DEFAULT_SAMPLING_DIR,
    assert_no_kmp_duplicate_lib_ok,
    load_dev_paragraphs,
)
from scripts.retrieve_hard_candidates import (  # noqa: E402
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_INDEX_DIR,
    EMBED_SCRIPT,
    SEARCH_SCRIPT,
    run_retrieval_pipeline,
)
from scripts.search_hard_candidates import (  # noqa: E402
    assert_process_isolation,
    run_faiss_search_stage,
)

requires_raw_squad = pytest.mark.skipif(
    not DEFAULT_RAW_PATH.exists(),
    reason=f"Raw SQuAD 2.0 file not present at {DEFAULT_RAW_PATH}; skipping raw-data tests.",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _imported_top_level_modules(script_path: Path) -> set[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_hard_rule_11_ast_and_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Process 1 never imports faiss, Process 2 never imports torch/sentence_transformers, and KMP_DUPLICATE_LIB_OK is rejected."""
    import scripts.embed_dev_pool as embed_mod

    embed_imports = _imported_top_level_modules(EMBED_SCRIPT)
    search_imports = _imported_top_level_modules(SEARCH_SCRIPT)

    assert "faiss" not in embed_imports, "embed_dev_pool.py must never import faiss"
    assert "torch" not in search_imports, "search_hard_candidates.py must never import torch"
    assert (
        "sentence_transformers" not in search_imports
    ), "search_hard_candidates.py must never import sentence_transformers"

    monkeypatch.setattr(embed_mod, "_INITIAL_KMP_DUPLICATE_LIB_OK", "TRUE")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    with pytest.raises(RuntimeError, match="KMP_DUPLICATE_LIB_OK"):
        assert_no_kmp_duplicate_lib_ok()
    with pytest.raises(RuntimeError, match="KMP_DUPLICATE_LIB_OK"):
        assert_process_isolation()


def test_faiss_search_stage_synthetic_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit test FAISS search, gold exclusion, and recall diagnostics on synthetic embeddings."""
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    candidates_path = tmp_path / "candidates" / "hard_candidates.jsonl"

    # Create 6 orthogonal-ish 4D unit vectors as paragraph embeddings
    raw_vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.43588989, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw_vecs /= np.linalg.norm(raw_vecs, axis=1, keepdims=True)

    # 2 queries:
    # q0 aligns with p0 (gold=[0,0] -> rank 1); top-3 excluding gold should be [0,1], [1,0], [1,1]
    # q1 aligns with p2 (gold=[0,1] -> rank 2); top-3 excluding gold should be [1,0], [0,0], [1,1]
    q_vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    para_ids = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    para_meta = [
        {
            "pool_index": i,
            "paragraph_id": pid,
            "article_index": pid[0],
            "paragraph_index": pid[1],
            "article_title": f"Art_{pid[0]}",
            "text_sha256": f"sha_{i}",
        }
        for i, pid in enumerate(para_ids)
    ]
    q_meta = [
        {
            "query_index": 0,
            "question_id": "q0",
            "split": "main",
            "gold_paragraph_id": [0, 0],
            "article_index": 0,
            "article_title": "Art_0",
        },
        {
            "query_index": 1,
            "question_id": "q1",
            "split": "pilot",
            "gold_paragraph_id": [0, 1],
            "article_index": 0,
            "article_title": "Art_0",
        },
    ]

    np.save(index_dir / "paragraph_embeddings.npy", raw_vecs)
    np.save(index_dir / "question_embeddings.npy", q_vecs)
    (index_dir / "paragraphs.json").write_text(json.dumps(para_meta), encoding="utf-8")
    (index_dir / "questions.json").write_text(json.dumps(q_meta), encoding="utf-8")

    manifest = run_faiss_search_stage(
        index_dir=index_dir,
        candidates_path=candidates_path,
        model_name="synthetic-test",
        top_k=3,
    )

    rows = [
        json.loads(line)
        for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["gold_paragraph_id"] not in rows[0]["ranked_paragraph_ids"]
    assert rows[1]["gold_paragraph_id"] not in rows[1]["ranked_paragraph_ids"]
    assert rows[0]["ranked_paragraph_ids"][0] == [0, 1]
    assert rows[1]["ranked_paragraph_ids"][0] == [1, 0]

    # q0 gold rank = 1, q1 gold rank = 2 -> recall@1 = 0.5, recall@5 = 1.0
    assert manifest["diagnostics"]["overall"]["recall_at_1"] == 0.5
    assert manifest["diagnostics"]["overall"]["recall_at_5"] == 1.0


@pytest.fixture(scope="module")
def ensured_retrieval_artifacts() -> dict:
    """Ensure the retrieval pipeline has been executed and return the index manifest."""
    if not DEFAULT_RAW_PATH.exists():
        pytest.skip(f"Missing raw SQuAD 2.0 file at {DEFAULT_RAW_PATH}")
    manifest_path = DEFAULT_INDEX_DIR / "manifest.json"
    if not manifest_path.exists() or not DEFAULT_CANDIDATES_PATH.exists():
        return run_retrieval_pipeline()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@requires_raw_squad
def test_gold_paragraph_never_in_candidates(ensured_retrieval_artifacts: dict) -> None:
    """Verify every question has 100 candidates, gold paragraph is excluded, and scores are sorted."""
    rows = [
        json.loads(line)
        for line in DEFAULT_CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 590

    for row in rows:
        q_id = row["question_id"]
        gold_pid = row["gold_paragraph_id"]
        ranked_pids = row["ranked_paragraph_ids"]
        scores = row["scores"]

        assert "gold_score" in row and isinstance(row["gold_score"], float)
        assert len(ranked_pids) == 100, f"Question {q_id} does not have 100 candidates"
        assert len(scores) == 100, f"Question {q_id} does not have 100 scores"
        assert (
            gold_pid not in ranked_pids
        ), f"Gold paragraph {gold_pid} leaked into candidates for {q_id}"

        # No duplicate candidate paragraph IDs within a question's top-100
        pid_tuples = [tuple(pid) for pid in ranked_pids]
        assert len(set(pid_tuples)) == 100, f"Duplicate candidate IDs found for {q_id}"

        # Scores must be monotonically non-increasing
        for i in range(len(scores) - 1):
            assert (
                scores[i] >= scores[i + 1] - 1e-7
            ), f"Scores not sorted for {q_id} at position {i}: {scores[i]} < {scores[i + 1]}"


@requires_raw_squad
def test_candidate_ids_exist_in_dev_pool_and_questions_match_splits(
    ensured_retrieval_artifacts: dict,
) -> None:
    """Verify every candidate paragraph ID exists in the 1,204 dev pool and questions match splits."""
    paragraphs = load_dev_paragraphs(DEFAULT_RAW_PATH)
    valid_pool_pids = {tuple(p["paragraph_id"]) for p in paragraphs}
    assert len(valid_pool_pids) == 1204

    expected_questions: list[tuple[str, str, tuple[int, int]]] = []
    for split_name in ("main", "pilot", "reserve"):
        split_records = json.loads(
            (DEFAULT_SAMPLING_DIR / f"{split_name}.json").read_text(encoding="utf-8")
        )
        for rec in split_records:
            expected_questions.append(
                (rec["question_id"], split_name, tuple(rec["paragraph_id"]))
            )

    rows = [
        json.loads(line)
        for line in DEFAULT_CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == len(expected_questions) == 590

    for row, (exp_qid, exp_split, exp_gold_pid) in zip(rows, expected_questions):
        assert row["question_id"] == exp_qid
        assert row["split"] == exp_split
        assert tuple(row["gold_paragraph_id"]) == exp_gold_pid
        assert tuple(row["gold_paragraph_id"]) in valid_pool_pids

        for pid in row["ranked_paragraph_ids"]:
            assert (
                tuple(pid) in valid_pool_pids
            ), f"Candidate paragraph ID {pid} not in SQuAD dev pool"


@requires_raw_squad
def test_index_manifest_and_diagnostics(ensured_retrieval_artifacts: dict) -> None:
    """Verify results/index/manifest.json schema, hashes, and diagnostic recall metrics."""
    manifest = ensured_retrieval_artifacts
    assert manifest["model_name"] == DEFAULT_MODEL_NAME
    assert manifest["dim"] == 384
    assert manifest["paragraph_count"] == 1204
    assert manifest["question_count"] == 590
    assert manifest["top_k"] == 100

    faiss_path = DEFAULT_INDEX_DIR / "paragraphs.faiss"
    assert faiss_path.exists()
    assert sha256_file(faiss_path) == manifest["sha256"]
    assert (
        sha256_file(DEFAULT_CANDIDATES_PATH)
        == manifest["files"]["hard_candidates"]["sha256"]
    )

    diag = manifest["diagnostics"]["overall"]
    assert diag["total_questions"] == 590
    assert 0.0 < diag["recall_at_1"] <= diag["recall_at_5"] <= 1.0
    assert not manifest["diagnostics"]["unexpectedly_low"]


@requires_raw_squad
def test_retrieval_pipeline_determinism(
    tmp_path: Path,
    ensured_retrieval_artifacts: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a fresh two-process run produces identical index and candidate SHA-256 hashes."""
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    alt_index_dir = tmp_path / "index_run2"
    alt_candidates_path = tmp_path / "candidates_run2" / "hard_candidates.jsonl"

    manifest2 = run_retrieval_pipeline(
        raw_path=DEFAULT_RAW_PATH,
        sampling_dir=DEFAULT_SAMPLING_DIR,
        index_dir=alt_index_dir,
        candidates_path=alt_candidates_path,
    )

    assert manifest2["sha256"] == ensured_retrieval_artifacts["sha256"]
    assert (
        manifest2["files"]["paragraph_embeddings"]["sha256"]
        == ensured_retrieval_artifacts["files"]["paragraph_embeddings"]["sha256"]
    )
    assert (
        manifest2["files"]["question_embeddings"]["sha256"]
        == ensured_retrieval_artifacts["files"]["question_embeddings"]["sha256"]
    )
    assert (
        sha256_file(alt_candidates_path)
        == sha256_file(DEFAULT_CANDIDATES_PATH)
    )
