"""Unit tests for code-level guardrails (`scripts/check_env.py` and `scripts/check_invariants.py`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_env import check_environment  # noqa: E402
from scripts.check_invariants import check_invariants, sha256_bytes, sha256_file  # noqa: E402


@pytest.fixture()
def sandbox_repo(tmp_path: Path) -> Path:
    """Create a temporary copy of the repository configs, hashes, and requirement files."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "data" / "frozen").mkdir(parents=True)

    for rel in (
        "configs/prereg.yaml",
        "configs/prompt_v1.yaml",
        "tests/prereg.sha256",
        "tests/prompt.sha256",
        "requirements.txt",
        "requirements-mac.txt",
    ):
        dst = tmp_path / rel
        shutil.copyfile(REPO_ROOT / rel, dst)
        dst.chmod(0o644)

    return tmp_path


def test_real_repo_prereg_and_prompt_contents() -> None:
    """Verify exact values in configs/prereg.yaml and configs/prompt_v1.yaml."""
    prereg_path = REPO_ROOT / "configs" / "prereg.yaml"
    prompt_path = REPO_ROOT / "configs" / "prompt_v1.yaml"

    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    assert prereg == {
        "hypothesis": "H1",
        "primary_model": "Qwen3-8B",
        "models": ["Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B-Instruct"],
        "thresholds": {
            "hard_drop_min_pts": 5,
            "gap_min_pts": 3,
            "ci_level": 0.95,
            "ci_must_exclude_zero": True,
        },
        "design": {
            "n_levels": [0, 1, 2, 4, 8],
            "distractor_types": ["hard_negative", "random_length_matched"],
            "length_match_tolerance": 0.20,
            "seeds": 3,
            "n_questions": 500,
            "pilot_questions": 50,
            "reserve_questions": 40,
            "hard_negative_pool_k": 20,
            "nested_prefixes": True,
        },
        "decoding": {"greedy": True, "max_new_tokens": 32},
        "metrics": {"primary": "exact_match_squad_normalized", "secondary": "token_f1"},
        "statistics": {
            "bootstrap_resamples": 10000,
            "unit": "question",
            "average_seeds_per_question": True,
            "secondary_correction": "holm",
        },
        "power_rule": {"extend_primary_comparison_to": 800, "if_power_below": 0.80},
        "length_match": {
            "tokenizer": "Qwen3",
            "secondary_check_tokenizer": "Llama-3.1",
            "tolerance": 0.20,
        },
        "prompt": {
            "template_file": "configs/prompt_v1.yaml",
            "chat_template": True,
            "qwen3_enable_thinking": False,
            "system_prompt": None,
        },
        "primary_comparison": {
            "n": 8,
            "model": "Qwen3-8B",
            "baseline_condition": "oracle",
            "distractor_types": ["hard_negative", "random_length_matched"],
        },
        "rng": {
            "master_seed": 4207,
            "draw_seeds": [0, 1, 2],
        },
    }

    prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    assert prompt == {
        "closed_book": "Question: {q}\nAnswer with the shortest possible answer, no explanation.",
        "rag": {
            "header": "Context:\n",
            "passage_line": "[Document {i}] {text}\n",
            "footer": "\nQuestion: {q}\nAnswer with the shortest text span that answers the question, no explanation.",
        },
    }


def test_rag_prompt_rendering_matches_fixture_bytes() -> None:
    """Verify applying the RAG rendering rule to AAA/BBB/CCC/QQQ matches fixture byte for byte."""
    prompt_path = REPO_ROOT / "configs" / "prompt_v1.yaml"
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "rag_prompt_3passages.txt"

    prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    rag_tpl = prompt["rag"]
    passages = ["AAA", "BBB", "CCC"]
    question = "QQQ"

    rendered = (
        rag_tpl["header"]
        + "".join(
            rag_tpl["passage_line"].format(i=i, text=text)
            for i, text in enumerate(passages, start=1)
        )
        + rag_tpl["footer"].format(q=question)
    )
    assert rendered.encode("utf-8") == fixture_path.read_bytes()


def test_real_repo_invariants_pass() -> None:
    """Verify the real repository passes all environment and invariant checks."""
    errors = check_invariants(repo_root=REPO_ROOT, run_env_check=True)
    assert errors == []


def test_failure_mode_a_prereg_hash_mismatch(sandbox_repo: Path) -> None:
    """Failure mode (a1): modifying configs/prereg.yaml triggers prereg hash mismatch."""
    prereg_file = sandbox_repo / "configs" / "prereg.yaml"
    prereg_file.write_text(
        prereg_file.read_text(encoding="utf-8").replace("hard_drop_min_pts: 5", "hard_drop_min_pts: 2"),
        encoding="utf-8",
    )
    errors = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("prereg hash mismatch" in e for e in errors), f"Expected prereg mismatch in {errors}"


def test_failure_mode_a_prompt_hash_mismatch(sandbox_repo: Path) -> None:
    """Failure mode (a2): modifying configs/prompt_v1.yaml triggers prompt hash mismatch."""
    prompt_file = sandbox_repo / "configs" / "prompt_v1.yaml"
    prompt_file.write_text(
        prompt_file.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    errors = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("prompt hash mismatch" in e for e in errors), f"Expected prompt mismatch in {errors}"


def test_failure_mode_b_frozen_manifest_hash_mismatch(sandbox_repo: Path) -> None:
    """Failure mode (b): data/frozen/manifest.json hash differing from manifest.sha256 fails."""
    manifest_json = sandbox_repo / "data" / "frozen" / "manifest.json"
    manifest_sha = sandbox_repo / "data" / "frozen" / "manifest.sha256"

    manifest_json.write_text('{"pilot_size": 50, "main_size": 500}\n', encoding="utf-8")
    true_hash = sha256_file(manifest_json)
    manifest_sha.write_text(f"{true_hash}  data/frozen/manifest.json\n", encoding="utf-8")

    # Passes when matching
    assert check_invariants(repo_root=sandbox_repo, run_env_check=False) == []

    # Fails when manifest.json is altered
    manifest_json.write_text('{"pilot_size": 50, "main_size": 300}\n', encoding="utf-8")
    errors = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("frozen manifest hash mismatch" in e for e in errors), (
        f"Expected frozen manifest hash mismatch in {errors}"
    )


@pytest.mark.parametrize("missing_key", ["config_hash", "prereg_hash", "manifest_hash"])
def test_failure_mode_c_results_jsonl_missing_hashes(
    sandbox_repo: Path, missing_key: str
) -> None:
    """Failure mode (c): any results/*.jsonl row lacking config_hash, prereg_hash, or manifest_hash fails."""
    run_file = sandbox_repo / "results" / "run_001.jsonl"
    valid_row = {
        "question_id": "q1",
        "config_hash": "abc123",
        "prereg_hash": "def456",
        "manifest_hash": "ghi789",
    }
    bad_row = dict(valid_row)
    del bad_row[missing_key]

    run_file.write_text(
        json.dumps(valid_row) + "\n" + json.dumps(bad_row) + "\n",
        encoding="utf-8",
    )

    errors = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any(missing_key in e and "run_001.jsonl:2" in e for e in errors), (
        f"Expected missing {missing_key} error in {errors}"
    )


def test_failure_mode_d_ledger_append_allowed_and_rewrite_or_truncate_rejected(
    sandbox_repo: Path,
) -> None:
    """Failure mode (d): LEDGER.csv allows appending lines but fails if prior lines change or shrink."""
    run_file = sandbox_repo / "results" / "qwen3_8b.jsonl"
    row1 = json.dumps(
        {"id": 1, "config_hash": "c1", "prereg_hash": "p1", "manifest_hash": "m1"}
    ) + "\n"
    row2 = json.dumps(
        {"id": 2, "config_hash": "c1", "prereg_hash": "p1", "manifest_hash": "m1"}
    ) + "\n"
    initial_bytes = (row1 + row2).encode("utf-8")
    run_file.write_bytes(initial_bytes)

    ledger_file = sandbox_repo / "results" / "LEDGER.csv"
    ledger_file.write_text(
        f"file,line_count,sha256\nresults/qwen3_8b.jsonl,2,{sha256_bytes(initial_bytes)}\n",
        encoding="utf-8",
    )

    # 1. Exact recorded state passes
    assert check_invariants(repo_root=sandbox_repo, run_env_check=False) == []

    # 2. Appending a 3rd valid line passes
    row3 = json.dumps(
        {"id": 3, "config_hash": "c1", "prereg_hash": "p1", "manifest_hash": "m1"}
    ) + "\n"
    run_file.write_bytes(initial_bytes + row3.encode("utf-8"))
    assert check_invariants(repo_root=sandbox_repo, run_env_check=False) == []

    # 3. Modifying previously recorded line 1 (while keeping 3 lines) fails
    tampered_row1 = json.dumps(
        {"id": 999, "config_hash": "c1", "prereg_hash": "p1", "manifest_hash": "m1"}
    ) + "\n"
    run_file.write_bytes((tampered_row1 + row2 + row3).encode("utf-8"))
    errors_rewrite = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("previously recorded 2 lines were modified" in e for e in errors_rewrite), (
        f"Expected rewrite violation in {errors_rewrite}"
    )

    # 4. Truncating file to 1 line fails
    run_file.write_bytes(row1.encode("utf-8"))
    errors_truncate = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("line count shrank from 2 to 1" in e for e in errors_truncate), (
        f"Expected truncation violation in {errors_truncate}"
    )

    # 5. Deleting the recorded results file fails
    run_file.unlink()
    errors_deleted = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("was deleted or is missing" in e for e in errors_deleted), (
        f"Expected deleted file violation in {errors_deleted}"
    )


def test_failure_mode_e_env_guard_wrong_conda_env(sandbox_repo: Path) -> None:
    """Failure mode (e1): CONDA_DEFAULT_ENV != 'rag' fails environment guard."""
    errors = check_environment(
        repo_root=sandbox_repo,
        env={"CONDA_DEFAULT_ENV": "base"},
    )
    assert any("CONDA_DEFAULT_ENV must be 'rag'" in e for e in errors), (
        f"Expected CONDA_DEFAULT_ENV error in {errors}"
    )


def test_failure_mode_e_env_guard_wrong_python_version(sandbox_repo: Path) -> None:
    """Failure mode (e2): Python minor version != 3.12 fails environment guard."""
    errors = check_environment(
        repo_root=sandbox_repo,
        env={"CONDA_DEFAULT_ENV": "rag"},
        py_version_info=(3, 14),
    )
    assert any("Python minor version must be 3.12" in e for e in errors), (
        f"Expected Python version error in {errors}"
    )


def test_failure_mode_e_env_guard_package_pin_mismatch_core_and_mac(
    sandbox_repo: Path,
) -> None:
    """Failure mode (e3): Version mismatch in requirements.txt or requirements-mac.txt fails."""
    # Core mismatch
    errors_core = check_environment(
        repo_root=sandbox_repo,
        env={"CONDA_DEFAULT_ENV": "rag"},
        platform_sys="linux",
        machine_arch="x86_64",
        version_getter=lambda pkg: "0.0.0" if pkg == "numpy" else (
            "3.1.1" if pkg == "pandas" else
            "1.18.1" if pkg == "scipy" else
            "1.9.1" if pkg == "scikit-learn" else
            "6.0.3" if pkg == "pyyaml" else
            "4.70.1" if pkg == "tqdm" else
            "9.2.0" if pkg == "pytest" else
            "3.10.9" if pkg == "matplotlib" else
            "1.14.3" if pkg == "faiss-cpu" else
            "2.15.0" if pkg == "torch" else
            "5.22.1" if pkg == "transformers" else
            "6.2.0"
        ),
        py_version_info=(3, 12),
    )
    assert any("Core package 'numpy' version mismatch" in e for e in errors_core), (
        f"Expected numpy pin mismatch in {errors_core}"
    )

    # Mac-only mismatch on darwin arm64, ignored on linux x86_64
    def mock_versions_bad_mlx(pkg: str) -> str:
        mapping = {
            "numpy": "2.5.4",
            "pandas": "3.1.1",
            "scipy": "1.18.1",
            "scikit-learn": "1.9.1",
            "pyyaml": "6.0.3",
            "tqdm": "4.70.1",
            "pytest": "9.2.0",
            "matplotlib": "3.10.9",
            "faiss-cpu": "1.14.3",
            "torch": "2.15.0",
            "transformers": "5.22.1",
            "sentence-transformers": "6.2.0",
            "mlx": "0.1.0",
            "mlx-lm": "0.32.2",
        }
        return mapping[pkg]

    assert (
        check_environment(
            repo_root=sandbox_repo,
            env={"CONDA_DEFAULT_ENV": "rag"},
            platform_sys="linux",
            machine_arch="x86_64",
            version_getter=mock_versions_bad_mlx,
            py_version_info=(3, 12),
        )
        == []
    )

    errors_mac = check_environment(
        repo_root=sandbox_repo,
        env={"CONDA_DEFAULT_ENV": "rag"},
        platform_sys="darwin",
        machine_arch="arm64",
        version_getter=mock_versions_bad_mlx,
        py_version_info=(3, 12),
    )
    assert any("Mac package 'mlx' version mismatch" in e for e in errors_mac), (
        f"Expected mlx pin mismatch in {errors_mac}"
    )


def test_cli_exit_code_nonzero_when_env_invalid() -> None:
    """Verify scripts/check_invariants.py exits non-zero when CONDA_DEFAULT_ENV != 'rag'."""
    env = {"PATH": "/usr/bin:/bin", "CONDA_DEFAULT_ENV": "wrong_env"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_invariants.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "CONDA_DEFAULT_ENV must be 'rag'" in proc.stderr


def test_manifest_hash_rules_pilot_vs_main_when_frozen_exists(sandbox_repo: Path) -> None:
    """When data/frozen/manifest.json exists, pilot accepts sampling hash and main requires frozen hash."""
    sampling_dir = sandbox_repo / "data" / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    sampling_manifest = sampling_dir / "manifest.json"
    sampling_manifest.write_text('{"split_counts": {"pilot": 50, "main": 500}}\n', encoding="utf-8")
    sampling_hash = sha256_file(sampling_manifest)

    frozen_manifest = sandbox_repo / "data" / "frozen" / "manifest.json"
    frozen_manifest.write_text('{"final_n": 500}\n', encoding="utf-8")
    frozen_hash = sha256_file(frozen_manifest)
    (sandbox_repo / "data" / "frozen" / "manifest.sha256").write_text(
        f"{frozen_hash}  data/frozen/manifest.json\n",
        encoding="utf-8",
    )
    assert sampling_hash != frozen_hash

    pilot_dir = sandbox_repo / "results" / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_file = pilot_dir / "Qwen3-8B__oracle__none__n0__seednone__pilot.jsonl"
    pilot_file.write_text(
        json.dumps(
            {
                "qid": "p1",
                "split": "pilot",
                "config_hash": "c1",
                "prereg_hash": "p1",
                "manifest_hash": sampling_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main_file = sandbox_repo / "results" / "Qwen3-8B__oracle__none__n0__seednone__main.jsonl"
    main_file.write_text(
        json.dumps(
            {
                "qid": "m1",
                "split": "main",
                "config_hash": "c1",
                "prereg_hash": "p1",
                "manifest_hash": frozen_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Valid state passes
    assert check_invariants(repo_root=sandbox_repo, run_env_check=False) == []

    # Main row using sampling hash (instead of frozen hash) fails
    main_file.write_text(
        json.dumps(
            {
                "qid": "m1",
                "split": "main",
                "config_hash": "c1",
                "prereg_hash": "p1",
                "manifest_hash": sampling_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    errs_main = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("manifest_hash mismatch" in e for e in errs_main), errs_main

    # Restore valid main row, tamper pilot row manifest_hash -> fails
    main_file.write_text(
        json.dumps(
            {
                "qid": "m1",
                "split": "main",
                "config_hash": "c1",
                "prereg_hash": "p1",
                "manifest_hash": frozen_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pilot_file.write_text(
        json.dumps(
            {
                "qid": "p1",
                "split": "pilot",
                "config_hash": "c1",
                "prereg_hash": "p1",
                "manifest_hash": "deadbeef",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    errs_pilot = check_invariants(repo_root=sandbox_repo, run_env_check=False)
    assert any("pilot manifest_hash mismatch" in e for e in errs_pilot), errs_pilot


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def test_prereg_v1_tag_byte_identity_and_tampering_cli(tmp_path: Path) -> None:
    """Verify prereg-v1 tag enforces byte-identity across configs and data/frozen/ via CLI."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "data" / "frozen").mkdir(parents=True)

    for rel in (
        "configs/prereg.yaml",
        "configs/prompt_v1.yaml",
        "configs/extraction_v1.yaml",
        "configs/runtime.yaml",
        "tests/prereg.sha256",
        "tests/prompt.sha256",
        "requirements.txt",
        "requirements-mac.txt",
    ):
        dst = tmp_path / rel
        shutil.copyfile(REPO_ROOT / rel, dst)

    frozen_main = tmp_path / "data" / "frozen" / "main.json"
    frozen_seed0 = tmp_path / "data" / "frozen" / "main_seed0.jsonl"
    frozen_manifest = tmp_path / "data" / "frozen" / "manifest.json"
    frozen_sha = tmp_path / "data" / "frozen" / "manifest.sha256"

    frozen_main.write_text('[{"question_id": "q1"}]\n', encoding="utf-8")
    frozen_seed0.write_text('{"question_id": "q1", "seed": 0}\n', encoding="utf-8")
    manifest_payload = {
        "final_n": 1,
        "output_files": {
            "main": {"path": "data/frozen/main.json", "sha256": sha256_file(frozen_main)},
            "main_seed0": {
                "path": "data/frozen/main_seed0.jsonl",
                "sha256": sha256_file(frozen_seed0),
            },
        },
    }
    frozen_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    frozen_sha.write_text(
        f"{sha256_file(frozen_manifest)}  data/frozen/manifest.json\n",
        encoding="utf-8",
    )

    _run_git(tmp_path, "init")
    _run_git(tmp_path, "add", ".")
    _run_git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "init",
    )
    _run_git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "tag",
        "-a",
        "prereg-v1",
        "-m",
        "prereg freeze",
    )

    cli_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_invariants.py"),
        "--repo-root",
        str(tmp_path),
    ]

    # Clean tagged state passes both API and CLI
    assert check_invariants(repo_root=tmp_path, run_env_check=False) == []
    ok_proc = subprocess.run(cli_cmd, capture_output=True, text=True, check=False)
    assert ok_proc.returncode == 0, ok_proc.stderr

    # Each tampered file triggers non-zero failure
    for rel_tamper, append_text in (
        ("configs/prereg.yaml", "\n# tampered\n"),
        ("configs/prompt_v1.yaml", "\n# tampered\n"),
        ("configs/extraction_v1.yaml", "\n# tampered\n"),
        ("configs/runtime.yaml", "\n# tampered\n"),
        ("data/frozen/main.json", "\n"),
    ):
        target = tmp_path / rel_tamper
        orig_bytes = target.read_bytes()
        try:
            target.write_bytes(orig_bytes + append_text.encode("utf-8"))
            errs = check_invariants(repo_root=tmp_path, run_env_check=False)
            assert any("prereg-v1" in e and rel_tamper in e for e in errs), (
                f"Expected prereg-v1 error for {rel_tamper}, got {errs}"
            )
            fail_proc = subprocess.run(cli_cmd, capture_output=True, text=True, check=False)
            assert fail_proc.returncode != 0
        finally:
            target.write_bytes(orig_bytes)

    # Extra file under data/frozen/ fails
    extra_file = tmp_path / "data" / "frozen" / "extra.txt"
    extra_file.write_text("extra\n", encoding="utf-8")
    try:
        errs_extra = check_invariants(repo_root=tmp_path, run_env_check=False)
        assert any("Extra working-tree file" in e and "extra.txt" in e for e in errs_extra), errs_extra
        assert subprocess.run(cli_cmd, capture_output=True, text=True, check=False).returncode != 0
    finally:
        extra_file.unlink()

    # Missing file under data/frozen/ fails
    orig_seed0 = frozen_seed0.read_bytes()
    frozen_seed0.unlink()
    try:
        errs_missing = check_invariants(repo_root=tmp_path, run_env_check=False)
        assert any(
            "Missing working-tree file" in e and "main_seed0.jsonl" in e for e in errs_missing
        ), errs_missing
        assert subprocess.run(cli_cmd, capture_output=True, text=True, check=False).returncode != 0
    finally:
        frozen_seed0.write_bytes(orig_seed0)

