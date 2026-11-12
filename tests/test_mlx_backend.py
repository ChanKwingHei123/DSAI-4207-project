"""Unit and integration tests for the P4b MLXBackend (`src/backends/mlx_backend.py`).

Model-weight tests skip cleanly when `models/mlx/` is absent so lightweight worktrees
without local model weights still pass `make test`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.omp_guard  # noqa: F401, E402
from scripts.run_units import main as run_units_main  # noqa: E402
from src.backends.mlx_backend import (  # noqa: E402
    SUPPORTED_MLX_MODELS,
    MLXBackend,
    get_active_mlx_model_name,
    unload_active_mlx_model,
)
from src.harness import (  # noqa: E402
    build_messages,
    compute_study_hashes,
    load_runtime_config,
    parse_run_unit,
    render_closed_book_text,
    run_unit,
)

MLX_MODELS_DIR = REPO_ROOT / "models" / "mlx"
HAS_MLX_WEIGHTS = (
    MLX_MODELS_DIR.exists()
    and (MLX_MODELS_DIR / "qwen3-4b-4bit").exists()
    and (MLX_MODELS_DIR / "qwen3-8b-4bit").exists()
    and (MLX_MODELS_DIR / "llama31-8b-4bit").exists()
)

requires_mlx_weights = pytest.mark.skipif(
    not HAS_MLX_WEIGHTS,
    reason="models/mlx/ weights are absent in this worktree.",
)


def _make_synthetic_input_row(qid: str, question_text: str, gold_answer: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "split": "pilot",
        "seed": 0,
        "article_index": 0,
        "article_title": "Test_Article",
        "question_text": question_text,
        "gold_answers": [gold_answer],
        "gold_paragraph_id": [0, 0],
        "gold_score": 0.85,
        "hard_pool_paragraph_ids": [[1, i] for i in range(20)],
        "hard_paragraph_ids": [[1, i] for i in range(8)],
        "hard_candidate_scores": [0.5 - 0.01 * i for i in range(8)],
        "random_paragraph_ids": [[2, i] for i in range(8)],
        "random_length_match_windows": [5] * 8,
        "p": 0.5,
        "gold_positions": {"1": 0, "2": 1, "4": 2, "8": 4},
        "token_lengths": {
            "qwen3": {"gold": 40, "hard": [40] * 8, "random": [40] * 8},
            "llama3": {"gold": 38, "hard": [38] * 8, "random": [38] * 8},
        },
        "rejection_counts": {
            "candidate_depth": 100,
            "hard_top_k_leak_rejected": 0,
            "hard_before_pool_k_leak_rejected": 0,
            "hard_top_k_non_rejected": 100,
            "random_pool_non_gold_article_total": 1000,
            "random_pool_leak_rejected": 0,
        },
    }


@pytest.fixture()
def mlx_sandbox_repo(tmp_path: Path) -> Path:
    """Create an isolated temporary repository with 3 pilot questions for MLXBackend tests."""
    for subdir in (
        "configs",
        "data/raw",
        "data/inputs",
        "data/sampling",
    ):
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)

    for rel in (
        "configs/prereg.yaml",
        "configs/prompt_v1.yaml",
        "configs/extraction_v1.yaml",
        "configs/runtime.yaml",
    ):
        shutil.copyfile(REPO_ROOT / rel, tmp_path / rel)

    # Synthetic dev-v2.0.json with gold passage [0, 0]
    articles = [
        {
            "title": "Test_Article",
            "paragraphs": [
                {
                    "context": (
                        "Paris is the capital and most populous city of France. "
                        "Alexander Fleming discovered penicillin in 1928. "
                        "Water boils at 100 degrees Celsius at standard atmospheric pressure."
                    ),
                    "qas": [],
                }
            ],
        },
        {
            "title": "Hard_Article",
            "paragraphs": [{"context": f"Hard distractor {i}.", "qas": []} for i in range(20)],
        },
        {
            "title": "Random_Article",
            "paragraphs": [{"context": f"Random distractor {i}.", "qas": []} for i in range(20)],
        },
    ]
    (tmp_path / "data" / "raw" / "dev-v2.0.json").write_text(
        json.dumps({"version": "v2.0", "data": articles}),
        encoding="utf-8",
    )

    (tmp_path / "data" / "sampling" / "manifest.json").write_text(
        json.dumps({"seed": 4207, "split_counts": {"pilot": 3, "main": 0, "reserve": 0}}),
        encoding="utf-8",
    )

    questions = [
        ("pilot_q1", "What is the capital of France?", "Paris"),
        ("pilot_q2", "Who discovered penicillin in 1928?", "Alexander Fleming"),
        (
            "pilot_q3",
            "At what temperature in degrees Celsius does water boil at standard atmospheric pressure?",
            "100",
        ),
    ]
    (tmp_path / "data" / "sampling" / "pilot.json").write_text(
        json.dumps([{"question_id": qid} for qid, _, _ in questions]),
        encoding="utf-8",
    )

    rows = [_make_synthetic_input_row(qid, qtext, ans) for qid, qtext, ans in questions]
    for seed in (0, 1, 2):
        seed_file = tmp_path / "data" / "inputs" / f"pilot_seed{seed}.jsonl"
        seed_file.write_text(
            "".join(json.dumps(r) + "\n" for r in rows),
            encoding="utf-8",
        )

    return tmp_path


def test_mlx_backend_metadata_resolution_from_runtime_yaml() -> None:
    """Verify MLXBackend resolves paths and metadata strictly from configs/runtime.yaml."""
    runtime_cfg = load_runtime_config(REPO_ROOT / "configs" / "runtime.yaml")
    for model_name in SUPPORTED_MLX_MODELS:
        backend = MLXBackend(model_name=model_name)
        info = runtime_cfg["models"][model_name]
        assert backend.model_name == model_name
        assert backend.backend_name == "mlx"
        assert backend.is_stub is False
        assert backend.model_id == f"{info['hf_repo_id']}@{info['commit_sha']}"
        assert backend.mlx_rel_path == info["mlx_path"]
        if model_name.startswith("Qwen3"):
            assert backend.enable_thinking is False
        else:
            assert backend.enable_thinking is None

    with pytest.raises(ValueError, match="Unsupported model"):
        MLXBackend(model_name="Unknown-Model-7B")


def test_mlx_backend_validation_guards() -> None:
    """Verify MLXBackend rejects system prompts, empty messages, and non-greedy decoding."""
    backend = MLXBackend("Qwen3-4B")
    with pytest.raises(ValueError, match="messages must not be empty"):
        backend.format_prompt([])
    with pytest.raises(ValueError, match="System prompt is forbidden"):
        backend.format_prompt(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        )
    with pytest.raises(ValueError, match="Expected single user message"):
        backend.format_prompt(
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ]
        )
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        backend.generate(["prompt"], max_new_tokens=0, greedy=True)
    with pytest.raises(ValueError, match="greedy=True"):
        backend.generate(["prompt"], max_new_tokens=32, greedy=False)


@requires_mlx_weights
def test_mlx_chat_template_formatting_and_no_think_tags() -> None:
    """Verify Qwen3 chat template disables thinking (`<think>\\n\\n</think>`) and output has no `<think>`."""
    try:
        qwen_backend = MLXBackend("Qwen3-4B")
        user_text = render_closed_book_text("What is the capital of France?")
        messages = build_messages(user_text)

        rendered_qwen = qwen_backend.format_prompt(messages)
        # Qwen3 with enable_thinking=False pre-fills empty `<think>\n\n</think>\n\n` in the prompt
        assert "<|im_start|>user\n" in rendered_qwen
        assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in rendered_qwen

        outputs = qwen_backend.generate([rendered_qwen], max_new_tokens=32, greedy=True)
        assert len(outputs) == 1
        raw_out = outputs[0]
        assert "<think>" not in raw_out
        assert "</think>" not in raw_out
        assert "Paris" in raw_out

        # Switching to Llama-3.1-8B-Instruct must evict Qwen3-4B and apply Llama chat template
        llama_backend = MLXBackend("Llama-3.1-8B-Instruct")
        rendered_llama = llama_backend.format_prompt(messages)
        assert get_active_mlx_model_name() == "Llama-3.1-8B-Instruct"
        assert "<|start_header_id|>user<|end_header_id|>" in rendered_llama
        assert "<|start_header_id|>assistant<|end_header_id|>" in rendered_llama
    finally:
        unload_active_mlx_model()


@requires_mlx_weights
def test_mlx_repeat_determinism_and_row_fields_through_run_unit(
    mlx_sandbox_repo: Path,
) -> None:
    """Verify repeat determinism across two `run_unit` runs on 3 prompts using `tmp_path`."""
    try:
        unit = parse_run_unit("Qwen3-4B__oracle__none__n0__seednone__pilot")
        backend = MLXBackend("Qwen3-4B", repo_root=mlx_sandbox_repo)

        run1_dir = mlx_sandbox_repo / "scratch" / "run1"
        run2_dir = mlx_sandbox_repo / "scratch" / "run2"

        res1 = run_unit(
            unit=unit,
            backend=backend,
            max_questions=3,
            results_dir=run1_dir,
            dry_run=True,
            repo_root=mlx_sandbox_repo,
        )
        res2 = run_unit(
            unit=unit,
            backend=backend,
            max_questions=3,
            results_dir=run2_dir,
            dry_run=True,
            repo_root=mlx_sandbox_repo,
        )

        assert res1["written_count"] == 3
        assert res2["written_count"] == 3
        assert not (mlx_sandbox_repo / "results").exists()

        rows1 = [
            json.loads(line)
            for line in res1["output_path"].read_text(encoding="utf-8").splitlines()
        ]
        rows2 = [
            json.loads(line)
            for line in res2["output_path"].read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows1) == 3
        assert len(rows2) == 3

        expected_hashes = compute_study_hashes(repo_root=mlx_sandbox_repo)

        for r1, r2 in zip(rows1, rows2, strict=True):
            assert r1["qid"] == r2["qid"]
            assert r1["prompt_sha256"] == r2["prompt_sha256"]
            assert r1["raw_output"] == r2["raw_output"]
            assert r1["extracted_answer"] == r2["extracted_answer"]

            assert r1["backend"] == "mlx"
            assert r1["is_stub"] is False
            assert r1["seconds"] > 0.0
            assert r1["tokens_per_second"] > 0.0
            assert "<think>" not in r1["raw_output"]
            assert "</think>" not in r1["raw_output"]
            assert r1["config_hash"] == expected_hashes["config_hash"]
            assert r1["prereg_hash"] == expected_hashes["prereg_hash"]
            assert r1["manifest_hash"] == expected_hashes["manifest_hash"]

        # Verify CLI --backend mlx --dry-run writes only to scratch/dryrun and never results/
        rc = run_units_main(
            [
                "--units",
                "Qwen3-4B__oracle__none__n0__seednone__pilot",
                "--questions",
                "2",
                "--backend",
                "mlx",
                "--dry-run",
                "--repo-root",
                str(mlx_sandbox_repo),
            ]
        )
        assert rc == 0
        dryrun_file = (
            mlx_sandbox_repo
            / "scratch"
            / "dryrun"
            / "pilot"
            / "Qwen3-4B__oracle__none__n0__seednone__pilot.jsonl"
        )
        assert dryrun_file.exists()
        assert not (mlx_sandbox_repo / "results").exists()
    finally:
        unload_active_mlx_model()
