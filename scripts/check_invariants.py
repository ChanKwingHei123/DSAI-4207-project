#!/usr/bin/env python3
"""Project invariant checks (AGENTS.md Hard Rules 1, 2, 10)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_env import check_environment  # noqa: E402

REQUIRED_RESULT_HASH_KEYS = ("config_hash", "prereg_hash", "manifest_hash")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_expected_sha256(hash_file: Path) -> str:
    tokens = hash_file.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError(f"Empty sha256 file: {hash_file}")
    return tokens[0].lower()


def check_file_hash_pair(target_file: Path, hash_file: Path, label: str) -> list[str]:
    errors: list[str] = []
    if not target_file.exists():
        errors.append(f"Missing {label} file: {target_file}")
        return errors
    if not hash_file.exists():
        errors.append(f"Missing {label} hash file: {hash_file}")
        return errors
    try:
        expected = read_expected_sha256(hash_file)
    except Exception as exc:
        errors.append(f"Invalid {label} hash file {hash_file}: {exc}")
        return errors

    actual = sha256_file(target_file)
    if actual != expected:
        errors.append(
            f"{label} hash mismatch for {target_file.name}: actual={actual} != expected={expected}"
        )
    return errors


def check_results_jsonl_rows(results_dir: Path) -> list[str]:
    errors: list[str] = []
    if not results_dir.exists():
        return errors

    for jsonl_path in sorted(results_dir.rglob("*.jsonl")):
        rel_path = jsonl_path.relative_to(results_dir)
        raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        for line_no, raw_line in enumerate(raw_lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                errors.append(f"{rel_path}:{line_no}: empty line in JSONL results file")
                continue
            try:
                record: Any = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"{rel_path}:{line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(record, dict):
                errors.append(f"{rel_path}:{line_no}: JSONL row must be a JSON object")
                continue
            if record.get("is_stub") is True or str(record.get("backend", "")).strip().lower() == "stub":
                errors.append(
                    f"{rel_path}:{line_no}: stub output (is_stub=True) is forbidden under results/"
                )
            for key in REQUIRED_RESULT_HASH_KEYS:
                val = record.get(key)
                if not isinstance(val, str) or not val.strip():
                    errors.append(
                        f"{rel_path}:{line_no}: missing or empty required field {key!r}"
                    )
    return errors


def check_results_ledger(repo_root: Path, results_dir: Path) -> list[str]:
    errors: list[str] = []
    ledger_path = results_dir / "LEDGER.csv"
    if not ledger_path.exists():
        return errors

    with ledger_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            errors.append(f"Empty or headerless ledger file: {ledger_path}")
            return errors

        for row_idx, raw_row in enumerate(reader, start=2):
            row = {k.strip().lower().replace(" ", "_"): (v or "").strip() for k, v in raw_row.items() if k}
            rel_file = row.get("file", "")
            line_count_str = row.get("line_count", "")
            expected_sha = row.get("sha256", "").lower()

            if not rel_file or not line_count_str or not expected_sha:
                errors.append(
                    f"LEDGER.csv:{row_idx}: row must have non-empty file, line_count, and sha256"
                )
                continue

            try:
                recorded_lines = int(line_count_str)
                if recorded_lines < 0:
                    raise ValueError("negative line_count")
            except ValueError:
                errors.append(
                    f"LEDGER.csv:{row_idx}: invalid line_count {line_count_str!r} for {rel_file}"
                )
                continue

            candidate = repo_root / rel_file
            if not candidate.exists():
                alt_candidate = results_dir / rel_file
                candidate = alt_candidate if alt_candidate.exists() else candidate

            if not candidate.exists():
                errors.append(
                    f"LEDGER.csv:{row_idx}: recorded results file {rel_file!r} was deleted or is missing"
                )
                continue

            raw_bytes_lines = candidate.read_bytes().splitlines(keepends=True)
            current_lines = len(raw_bytes_lines)
            if current_lines < recorded_lines:
                errors.append(
                    f"LEDGER.csv:{row_idx}: {rel_file} line count shrank from "
                    f"{recorded_lines} to {current_lines} (append-only violation)"
                )
                continue

            prefix_bytes = b"".join(raw_bytes_lines[:recorded_lines])
            prefix_sha = sha256_bytes(prefix_bytes)
            if prefix_sha != expected_sha:
                errors.append(
                    f"LEDGER.csv:{row_idx}: {rel_file} previously recorded {recorded_lines} lines "
                    f"were modified (prefix sha256={prefix_sha} != recorded sha256={expected_sha})"
                )

    return errors


def check_invariants(
    repo_root: Path = REPO_ROOT,
    run_env_check: bool = True,
    env_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Run all project invariant checks and return a list of failure messages."""
    errors: list[str] = []

    if run_env_check:
        kwargs = env_kwargs or {}
        errors.extend(check_environment(repo_root=repo_root, **kwargs))

    # (a) prereg.yaml and prompt_v1.yaml hash verification
    errors.extend(
        check_file_hash_pair(
            repo_root / "configs" / "prereg.yaml",
            repo_root / "tests" / "prereg.sha256",
            label="prereg",
        )
    )
    errors.extend(
        check_file_hash_pair(
            repo_root / "configs" / "prompt_v1.yaml",
            repo_root / "tests" / "prompt.sha256",
            label="prompt",
        )
    )

    # (b) data/frozen/manifest.json hash verification (when manifest.json exists)
    frozen_manifest = repo_root / "data" / "frozen" / "manifest.json"
    if frozen_manifest.exists():
        errors.extend(
            check_file_hash_pair(
                frozen_manifest,
                repo_root / "data" / "frozen" / "manifest.sha256",
                label="frozen manifest",
            )
        )

    # (c) results/*.jsonl rows carry config_hash, prereg_hash, manifest_hash
    results_dir = repo_root / "results"
    errors.extend(check_results_jsonl_rows(results_dir))

    # (d) results/LEDGER.csv append-only verification
    errors.extend(check_results_ledger(repo_root, results_dir))

    return errors


def main() -> int:
    errors = check_invariants()
    if errors:
        for err in errors:
            print(f"[INVARIANT FAIL] {err}", file=sys.stderr)
        return 1
    print("[INVARIANTS OK] prereg, prompt, frozen manifest, results rows, and LEDGER checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
