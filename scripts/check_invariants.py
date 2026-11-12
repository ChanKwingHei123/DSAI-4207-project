#!/usr/bin/env python3
"""Project invariant checks (AGENTS.md Hard Rules 1, 2, 10)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_env import check_environment  # noqa: E402

REQUIRED_RESULT_HASH_KEYS = ("config_hash", "prereg_hash", "manifest_hash")
PREREG_TAG_NAME = "prereg-v1"
PREREG_TAG_CONFIG_FILES = (
    "configs/prereg.yaml",
    "configs/prompt_v1.yaml",
    "configs/extraction_v1.yaml",
    "configs/runtime.yaml",
)


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


def check_results_jsonl_rows(
    results_dir: Path,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    if not results_dir.exists():
        return errors

    frozen_hash: str | None = None
    sampling_hash: str | None = None
    if repo_root is not None:
        frozen_manifest = repo_root / "data" / "frozen" / "manifest.json"
        sampling_manifest = repo_root / "data" / "sampling" / "manifest.json"
        if frozen_manifest.exists():
            frozen_hash = sha256_file(frozen_manifest)
        if sampling_manifest.exists():
            sampling_hash = sha256_file(sampling_manifest)

    for jsonl_path in sorted(results_dir.rglob("*.jsonl")):
        rel_path = jsonl_path.relative_to(results_dir)
        is_pilot_path = bool(rel_path.parts) and rel_path.parts[0] == "pilot"
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

            manifest_val = record.get("manifest_hash")
            if isinstance(manifest_val, str) and manifest_val.strip():
                if is_pilot_path:
                    row_split = record.get("split")
                    if row_split is not None and row_split != "pilot":
                        errors.append(
                            f"{rel_path}:{line_no}: pilot results path contains row with split={row_split!r}"
                        )
                    expected_pilot = sampling_hash if sampling_hash is not None else frozen_hash
                    if expected_pilot is not None and manifest_val != expected_pilot:
                        errors.append(
                            f"{rel_path}:{line_no}: pilot manifest_hash mismatch "
                            f"(actual={manifest_val} != expected={expected_pilot})"
                        )
                else:
                    expected_main = frozen_hash if frozen_hash is not None else sampling_hash
                    if expected_main is not None and manifest_val != expected_main:
                        errors.append(
                            f"{rel_path}:{line_no}: manifest_hash mismatch "
                            f"(actual={manifest_val} != expected={expected_main})"
                        )
    return errors


def check_frozen_manifest_outputs(repo_root: Path, frozen_manifest: Path) -> list[str]:
    """Verify SHA-256 of any output files declared in `data/frozen/manifest.json`."""
    errors: list[str] = []
    try:
        manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in frozen manifest {frozen_manifest}: {exc}")
        return errors

    if not isinstance(manifest, dict):
        errors.append(f"Frozen manifest {frozen_manifest} must be a JSON object")
        return errors

    output_files = manifest.get("output_files")
    if isinstance(output_files, dict):
        for name, entry in sorted(output_files.items()):
            if not isinstance(entry, dict):
                continue
            rel_path = entry.get("path")
            expected_sha = entry.get("sha256")
            if not isinstance(rel_path, str) or not isinstance(expected_sha, str):
                continue
            target = repo_root / rel_path
            if not target.exists():
                errors.append(
                    f"Missing frozen output file declared in manifest ({name}): {rel_path}"
                )
                continue
            actual_sha = sha256_file(target)
            if actual_sha != expected_sha.lower():
                errors.append(
                    f"Frozen output hash mismatch for {rel_path}: "
                    f"actual={actual_sha} != expected={expected_sha}"
                )
    return errors


def _read_git_blob(repo_root: Path, tag_name: str, rel_path: str) -> tuple[bytes | None, str | None]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "blob", f"refs/tags/{tag_name}:{rel_path}"],
        capture_output=True,
        text=False,
        check=False,
    )
    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        return None, err_msg or f"git cat-file exited {proc.returncode}"
    return proc.stdout, None


def check_prereg_tag_byte_identity(repo_root: Path) -> list[str]:
    """When git tag `prereg-v1` exists, enforce byte-identity of frozen configs and `data/frozen/`."""
    errors: list[str] = []
    if not (repo_root / ".git").exists():
        return errors

    rev_proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG_NAME}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if rev_proc.returncode == 1:
        return errors
    if rev_proc.returncode != 0:
        errors.append(
            f"Failed to verify git tag {PREREG_TAG_NAME!r}: "
            f"{rev_proc.stderr.strip() or f'exit {rev_proc.returncode}'}"
        )
        return errors

    for rel_path in PREREG_TAG_CONFIG_FILES:
        tag_bytes, err = _read_git_blob(repo_root, PREREG_TAG_NAME, rel_path)
        if err is not None or tag_bytes is None:
            errors.append(
                f"Tag {PREREG_TAG_NAME!r} is missing required config file {rel_path}: {err}"
            )
            continue
        wt_path = repo_root / rel_path
        if not wt_path.exists():
            errors.append(
                f"Working-tree file {rel_path} is missing (must match tag {PREREG_TAG_NAME!r})"
            )
            continue
        wt_bytes = wt_path.read_bytes()
        if wt_bytes != tag_bytes:
            errors.append(
                f"Tag {PREREG_TAG_NAME!r} byte-identity violation for {rel_path}: "
                f"working_tree={sha256_bytes(wt_bytes)} != tag={sha256_bytes(tag_bytes)}"
            )

    ls_proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            f"refs/tags/{PREREG_TAG_NAME}",
            "--",
            "data/frozen",
        ],
        capture_output=True,
        text=False,
        check=False,
    )
    if ls_proc.returncode != 0:
        err_msg = ls_proc.stderr.decode("utf-8", errors="replace").strip()
        errors.append(
            f"Failed to list data/frozen/ in tag {PREREG_TAG_NAME!r}: {err_msg}"
        )
        return errors

    tag_frozen_files = {
        p for p in ls_proc.stdout.decode("utf-8").split("\0") if p
    }
    if not tag_frozen_files:
        errors.append(f"Tag {PREREG_TAG_NAME!r} contains no files under data/frozen/")
        return errors

    frozen_dir = repo_root / "data" / "frozen"
    wt_frozen_files: set[str] = set()
    if frozen_dir.exists():
        wt_frozen_files = {
            p.relative_to(repo_root).as_posix()
            for p in frozen_dir.rglob("*")
            if p.is_file() or p.is_symlink()
        }

    for missing in sorted(tag_frozen_files - wt_frozen_files):
        errors.append(
            f"Missing working-tree file under data/frozen/ relative to tag {PREREG_TAG_NAME!r}: {missing}"
        )
    for extra in sorted(wt_frozen_files - tag_frozen_files):
        errors.append(
            f"Extra working-tree file under data/frozen/ not present in tag {PREREG_TAG_NAME!r}: {extra}"
        )

    for rel_path in sorted(tag_frozen_files & wt_frozen_files):
        tag_bytes, err = _read_git_blob(repo_root, PREREG_TAG_NAME, rel_path)
        if err is not None or tag_bytes is None:
            errors.append(
                f"Failed to read {rel_path} from tag {PREREG_TAG_NAME!r}: {err}"
            )
            continue
        wt_bytes = (repo_root / rel_path).read_bytes()
        if wt_bytes != tag_bytes:
            errors.append(
                f"Tag {PREREG_TAG_NAME!r} byte-identity violation for {rel_path}: "
                f"working_tree={sha256_bytes(wt_bytes)} != tag={sha256_bytes(tag_bytes)}"
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
        errors.extend(check_frozen_manifest_outputs(repo_root, frozen_manifest))

    # (c) results/*.jsonl rows carry config_hash, prereg_hash, manifest_hash
    results_dir = repo_root / "results"
    errors.extend(check_results_jsonl_rows(results_dir, repo_root=repo_root))

    # (d) results/LEDGER.csv append-only verification
    errors.extend(check_results_ledger(repo_root, results_dir))

    # (e) prereg-v1 tag byte-identity verification (when git tag exists)
    errors.extend(check_prereg_tag_byte_identity(repo_root))

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify project invariants (configs, frozen inputs, results hashes, LEDGER, prereg-v1 tag)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory to verify (default: project root).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = check_invariants(repo_root=args.repo_root.resolve())
    if errors:
        for err in errors:
            print(f"[INVARIANT FAIL] {err}", file=sys.stderr)
        return 1
    print(
        "[INVARIANTS OK] prereg, prompt, frozen manifest, results rows, LEDGER, "
        "and prereg-v1 tag checks passed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
