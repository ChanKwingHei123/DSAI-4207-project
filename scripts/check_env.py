#!/usr/bin/env python3
"""Environment guardrail check: enforces active `rag` conda env and exact package pins."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from typing import Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_pins(req_path: Path) -> dict[str, str]:
    """Parse exact `pkg==version` lines from a requirements file."""
    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {req_path}")
    pins: dict[str, str] = {}
    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"Unpinned requirement line in {req_path.name}: {line}")
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
                import faiss  # type: ignore

                return getattr(faiss, "__version__", "unknown")
        raise


def check_environment(
    repo_root: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
    platform_sys: str | None = None,
    machine_arch: str | None = None,
    version_getter: Callable[[str], str] = get_installed_version,
    py_version_info: tuple[int, int] | None = None,
) -> list[str]:
    """Return a list of environment guardrail violations (empty if all pass)."""
    errors: list[str] = []
    environ = os.environ if env is None else env
    sys_plat = sys.platform if platform_sys is None else platform_sys
    arch = platform.machine() if machine_arch is None else machine_arch
    py_ver = (
        (sys.version_info.major, sys.version_info.minor)
        if py_version_info is None
        else py_version_info
    )

    active_env = environ.get("CONDA_DEFAULT_ENV", "")
    if active_env != "rag":
        errors.append(
            f"CONDA_DEFAULT_ENV must be 'rag', got {active_env!r}. "
            "Activate with: conda activate rag"
        )

    if py_ver != (3, 12):
        errors.append(f"Python minor version must be 3.12, got {py_ver[0]}.{py_ver[1]}")

    req_core = repo_root / "requirements.txt"
    try:
        core_pins = parse_pins(req_core)
    except Exception as exc:
        errors.append(str(exc))
        core_pins = {}

    for pkg, expected_ver in core_pins.items():
        try:
            installed_ver = version_getter(pkg)
        except Exception:
            errors.append(f"Required core package {pkg!r} (pinned=={expected_ver}) is not installed")
            continue
        if installed_ver != expected_ver:
            errors.append(
                f"Core package {pkg!r} version mismatch: installed={installed_ver} != pinned={expected_ver}"
            )

    if sys_plat == "darwin" and arch == "arm64":
        req_mac = repo_root / "requirements-mac.txt"
        try:
            mac_pins = parse_pins(req_mac)
        except Exception as exc:
            errors.append(str(exc))
            mac_pins = {}

        for pkg, expected_ver in mac_pins.items():
            try:
                installed_ver = version_getter(pkg)
            except Exception:
                errors.append(
                    f"Required Mac package {pkg!r} (pinned=={expected_ver}) is not installed"
                )
                continue
            if installed_ver != expected_ver:
                errors.append(
                    f"Mac package {pkg!r} version mismatch: installed={installed_ver} != pinned={expected_ver}"
                )

    return errors


def main() -> int:
    errors = check_environment()
    if errors:
        for err in errors:
            print(f"[ENV GUARD FAIL] {err}", file=sys.stderr)
        return 1
    print("[ENV GUARD OK] CONDA_DEFAULT_ENV='rag' and all pinned package versions match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
