"""Tests for configs/runtime.yaml and docs/MODELS.md model provenance and hashes."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import re
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_YAML_PATH = REPO_ROOT / "configs" / "runtime.yaml"
PREREG_YAML_PATH = REPO_ROOT / "configs" / "prereg.yaml"
MODELS_DOC_PATH = REPO_ROOT / "docs" / "MODELS.md"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def runtime_cfg() -> dict[str, Any]:
    assert RUNTIME_YAML_PATH.exists(), f"Missing {RUNTIME_YAML_PATH}"
    data = yaml.safe_load(RUNTIME_YAML_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_runtime_config_schema_and_versions(runtime_cfg: dict[str, Any]) -> None:
    """Verify runtime.yaml schema, quantization metadata, and mlx / mlx-lm versions."""
    prereg = yaml.safe_load(PREREG_YAML_PATH.read_text(encoding="utf-8"))
    expected_models = prereg["models"]

    env_block = runtime_cfg.get("environment", {})
    assert env_block.get("mlx_version") == importlib.metadata.version("mlx")
    assert env_block.get("mlx_lm_version") == importlib.metadata.version("mlx-lm")

    models_block = runtime_cfg.get("models", {})
    assert list(models_block.keys()) == expected_models

    expected_bpw = {
        "Qwen3-4B": 4.500,
        "Qwen3-8B": 4.501,
        "Llama-3.1-8B-Instruct": 4.500,
    }
    expected_repos = {
        "Qwen3-4B": "Qwen/Qwen3-4B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Llama-3.1-8B-Instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    }

    for model_name in expected_models:
        entry = models_block[model_name]
        assert entry["hf_repo_id"] == expected_repos[model_name]
        assert COMMIT_SHA_RE.match(str(entry["commit_sha"])), (
            f"Invalid commit_sha for {model_name}: {entry.get('commit_sha')}"
        )
        assert isinstance(entry.get("local_hf_path"), str) and entry["local_hf_path"].startswith("models/hf/")
        assert isinstance(entry.get("mlx_path"), str) and entry["mlx_path"].startswith("models/mlx/")

        quant = entry.get("quantization", {})
        assert quant.get("bits") == 4
        assert quant.get("group_size") == 64
        assert float(quant.get("bits_per_weight")) == pytest.approx(expected_bpw[model_name], abs=1e-6)

        sha_block = entry.get("sha256", {})
        for copy_key in ("hf", "mlx"):
            copy_entry = sha_block.get(copy_key, {})
            st_map = copy_entry.get("safetensors", {})
            assert isinstance(st_map, dict) and len(st_map) >= 1
            for fname, digest in st_map.items():
                assert fname.endswith(".safetensors")
                assert SHA256_RE.match(str(digest)), f"Invalid sha256 for {model_name}/{copy_key}/{fname}"
            tok_digest = copy_entry.get("tokenizer.json")
            assert SHA256_RE.match(str(tok_digest)), (
                f"Invalid tokenizer.json sha256 for {model_name}/{copy_key}"
            )


def test_models_doc_provenance_and_license_statement(runtime_cfg: dict[str, Any]) -> None:
    """Verify docs/MODELS.md includes commit SHAs, unsloth mirror reason, and Llama 3.1 license."""
    assert MODELS_DOC_PATH.exists(), f"Missing {MODELS_DOC_PATH}"
    doc_text = MODELS_DOC_PATH.read_text(encoding="utf-8")

    assert "unsloth/Meta-Llama-3.1-8B-Instruct" in doc_text
    assert "declined" in doc_text.lower()
    assert "Llama 3.1 Community License" in doc_text

    for model_name, entry in runtime_cfg["models"].items():
        assert model_name in doc_text
        assert entry["commit_sha"] in doc_text
        assert entry["hf_repo_id"] in doc_text


def _verify_hashes_against_root(repo_root: Path, runtime_cfg: dict[str, Any]) -> None:
    models_root = repo_root / "models"
    if not models_root.exists():
        pytest.skip("models/ directory is absent; skipping on-disk hash verification.")

    for model_name, entry in runtime_cfg["models"].items():
        hf_dir = repo_root / entry["local_hf_path"]
        mlx_dir = repo_root / entry["mlx_path"]
        if not hf_dir.exists() or not mlx_dir.exists():
            pytest.skip(f"Model directory missing ({hf_dir} or {mlx_dir}); skipping on-disk check.")

        for copy_key, base_dir in (("hf", hf_dir), ("mlx", mlx_dir)):
            copy_sha = entry["sha256"][copy_key]
            recorded_st: dict[str, str] = copy_sha["safetensors"]

            # Ensure exact set of *.safetensors files on disk is recorded
            disk_st_files = {p.name for p in base_dir.glob("*.safetensors")}
            assert set(recorded_st.keys()) == disk_st_files, (
                f"Safetensors file set mismatch for {model_name} ({copy_key}): "
                f"recorded={set(recorded_st.keys())} vs disk={disk_st_files}"
            )

            for fname, expected_digest in sorted(recorded_st.items()):
                fpath = base_dir / fname
                actual_digest = _sha256_file(fpath)
                assert actual_digest == expected_digest, (
                    f"SHA-256 mismatch for {fpath}: actual={actual_digest} != expected={expected_digest}"
                )

            tok_path = base_dir / "tokenizer.json"
            assert tok_path.exists(), f"Missing {tok_path}"
            actual_tok_digest = _sha256_file(tok_path)
            assert actual_tok_digest == copy_sha["tokenizer.json"], (
                f"SHA-256 mismatch for {tok_path}: "
                f"actual={actual_tok_digest} != expected={copy_sha['tokenizer.json']}"
            )


def test_recorded_hashes_match_files_on_disk(runtime_cfg: dict[str, Any]) -> None:
    """Verify every recorded sha256 in runtime.yaml matches the file on disk; skip if models/ is absent."""
    _verify_hashes_against_root(REPO_ROOT, runtime_cfg)


def test_skip_when_models_dir_absent(tmp_path: Path, runtime_cfg: dict[str, Any]) -> None:
    """Verify hash check skips cleanly with pytest.skip when models/ is absent."""
    with pytest.raises(pytest.skip.Exception, match="models/ directory is absent"):
        _verify_hashes_against_root(tmp_path, runtime_cfg)

