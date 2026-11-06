#!/usr/bin/env python3
"""Verify the dedicated `rag` conda environment against pinned specifications."""

from __future__ import annotations

import os
import importlib.metadata
from pathlib import Path
import platform
import sys

import faiss
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy
import sklearn
import torch
import tqdm
import transformers
import yaml
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_pins(req_path: Path) -> dict[str, str]:
    """Parse exact `pkg==version` pins from a requirements file."""
    pins: dict[str, str] = {}
    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise AssertionError(f"Unpinned requirement line in {req_path.name}: {line}")
        pkg, ver = line.split("==", 1)
        pins[pkg.strip()] = ver.strip()
    return pins


def get_installed_version(pkg: str) -> str:
    """Resolve installed package version across pip and conda-forge metadata."""
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        if pkg == "faiss-cpu":
            try:
                return importlib.metadata.version("faiss")
            except importlib.metadata.PackageNotFoundError:
                return getattr(faiss, "__version__", "unknown")
        raise


def main() -> None:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    arch = platform.machine()
    sys_plat = sys.platform
    print(f"Python version: {py_ver} ({sys.executable})")
    print(f"Platform: {sys_plat} | Architecture: {arch}")
    assert (sys.version_info.major, sys.version_info.minor) == (3, 12), (
        f"Expected Python 3.12, got {py_ver}"
    )

    print("-" * 60)
    core_pins = parse_pins(REPO_ROOT / "requirements.txt")
    for pkg, expected_ver in core_pins.items():
        installed_ver = get_installed_version(pkg)
        print(f"[core] {pkg}: installed={installed_ver}, pinned={expected_ver}")
        assert installed_ver == expected_ver, (
            f"Version mismatch for {pkg}: installed {installed_ver} != pinned {expected_ver}"
        )

    is_arm64_macos = sys_plat == "darwin" and arch == "arm64"
    if is_arm64_macos:
        mac_pins = parse_pins(REPO_ROOT / "requirements-mac.txt")
        for pkg, expected_ver in mac_pins.items():
            installed_ver = get_installed_version(pkg)
            print(f"[mac]  {pkg}: installed={installed_ver}, pinned={expected_ver}")
            assert installed_ver == expected_ver, (
                f"Version mismatch for {pkg}: installed {installed_ver} != pinned {expected_ver}"
            )

        import mlx.core as mx
        import mlx_lm  # noqa: F401

        default_dev = mx.default_device()
        print("-" * 60)
        print(f"MLX default device: {default_dev}")
        assert default_dev.type == mx.DeviceType.gpu, (
            f"Expected MLX default device to be GPU, got {default_dev}"
        )

    print("-" * 60)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sentence = "Controlled dose-response study of irrelevant passages in small open-weight RAG readers."
    try:
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embedding = model.encode(
        [sentence], convert_to_numpy=True, normalize_embeddings=True
    )
    l2_norm = float(np.linalg.norm(embedding[0]))
    print("Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding shape: {embedding.shape} | dtype: {embedding.dtype} | L2 norm: {l2_norm:.6f}")
    assert embedding.shape == (1, 384), f"Expected shape (1, 384), got {embedding.shape}"
    assert np.isclose(l2_norm, 1.0, atol=1e-5), f"Expected L2 norm ~1.0, got {l2_norm}"

    print("-" * 60)
    print("ALL ENVIRONMENT CHECKS PASSED.")


if __name__ == "__main__":
    main()
