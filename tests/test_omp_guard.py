"""Tests for centralized OpenMP import-order guard (`src/omp_guard.py`)."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARDED_TARGETS = {"torch", "transformers", "sentence_transformers"}
GUARD_MODULE_NAMES = {"omp_guard", "src.omp_guard"}


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env["HF_HUB_OFFLINE"] = "1"
    return env


def test_omp_guard_exact_import_order_and_contents() -> None:
    """Verify src/omp_guard.py imports only numpy, scipy, scipy.sparse, sklearn, sklearn.utils in order."""
    guard_path = REPO_ROOT / "src" / "omp_guard.py"
    assert guard_path.exists(), f"Missing {guard_path}"

    tree = ast.parse(guard_path.read_text(encoding="utf-8"), filename=str(guard_path))
    imported_names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # Module docstring
            continue
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        else:
            pytest.fail(
                f"src/omp_guard.py must only contain a docstring and import statements, found {type(node).__name__}"
            )

    assert imported_names == [
        "numpy",
        "scipy",
        "scipy.sparse",
        "sklearn",
        "sklearn.utils",
    ]


def test_a_omp_guard_first_then_torch_and_sentence_transformers_succeeds() -> None:
    """(a) Fresh interpreter: import omp_guard first, then torch + sentence_transformers, and encode offline."""
    code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import src.omp_guard
import torch
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
except Exception as exc:
    print("CACHE_ABSENT:" + str(exc))
    sys.exit(77)

emb = model.encode(["OpenMP import order guard verification sentence."], convert_to_numpy=True, normalize_embeddings=True)
assert emb.shape == (1, 384), f"Unexpected shape: {{emb.shape}}"
print("ENCODE_OK", emb.shape)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 77 or "CACHE_ABSENT:" in proc.stdout:
        pytest.skip(
            "Local cache for sentence-transformers/all-MiniLM-L6-v2 is absent; skipping offline encode test."
        )

    assert proc.returncode == 0, (
        f"Subprocess failed with rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "ENCODE_OK (1, 384)" in proc.stdout


def test_b_negative_control_without_omp_guard_aborts_with_omp_error_15() -> None:
    """(b) Negative control in fresh interpreter WITHOUT omp_guard: import torch then sentence_transformers."""
    code = """
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
emb = model.encode(["Negative control without omp_guard."], convert_to_numpy=True, normalize_embeddings=True)
print("UNEXPECTED_SUCCESS", emb.shape)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        pytest.xfail(
            "Negative control did not abort with OMP Error #15 on this platform/runtime "
            "(single OpenMP runtime linked)."
        )

    assert proc.returncode != 0
    assert (
        "OMP: Error #15" in proc.stderr
    ), f"Expected OMP: Error #15 in stderr, got rc={proc.returncode}, stderr={proc.stderr!r}"


def _find_import_lines(py_path: Path) -> tuple[list[int], list[tuple[int, str]]]:
    """Return (omp_guard_import_lines, target_ml_import_lines) for a Python source file."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    guard_lines: list[int] = []
    target_lines: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                top = name.split(".")[0]
                if name in GUARD_MODULE_NAMES or top == "omp_guard":
                    guard_lines.append(node.lineno)
                if top in GUARDED_TARGETS:
                    target_lines.append((node.lineno, name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            top = mod.split(".")[0]
            if mod in GUARD_MODULE_NAMES or top == "omp_guard":
                guard_lines.append(node.lineno)
            elif mod == "src":
                for alias in node.names:
                    if alias.name == "omp_guard":
                        guard_lines.append(node.lineno)
            if top in GUARDED_TARGETS:
                target_lines.append((node.lineno, mod))

    return sorted(guard_lines), sorted(target_lines, key=lambda x: x[0])


def test_c_static_every_script_and_src_file_imports_omp_guard_first() -> None:
    """(c) Every file under scripts/ and src/ importing torch, transformers or sentence_transformers imports omp_guard before them."""
    checked_files: list[str] = []
    violations: list[str] = []

    for folder in ("scripts", "src"):
        dir_path = REPO_ROOT / folder
        if not dir_path.exists():
            continue
        for py_path in sorted(dir_path.rglob("*.py")):
            if py_path.name == "omp_guard.py":
                continue
            guard_lines, target_lines = _find_import_lines(py_path)
            if not target_lines:
                continue
            rel_path = str(py_path.relative_to(REPO_ROOT))
            checked_files.append(rel_path)
            first_target_line, first_target_mod = target_lines[0]
            if not guard_lines:
                violations.append(
                    f"{rel_path}: imports {first_target_mod!r} on line {first_target_line} without importing omp_guard"
                )
            elif guard_lines[0] >= first_target_line:
                violations.append(
                    f"{rel_path}: imports omp_guard on line {guard_lines[0]} AFTER {first_target_mod!r} on line {first_target_line}"
                )

    assert checked_files, "Expected at least one file under scripts/ or src/ importing torch/transformers/sentence_transformers"
    assert not violations, "OMP guard import-order violations:\n  " + "\n  ".join(violations)
