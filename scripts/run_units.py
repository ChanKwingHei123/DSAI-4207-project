#!/usr/bin/env python3
"""Run inference units for the DSAI4207 RAG irrelevant-context study (P3).

Example dry-run / stub command:
    python scripts/run_units.py --units Qwen3-8B__oracle__none__n0__seednone__pilot --questions 20 --backend stub

Guardrails (AGENTS.md Hard Rule 2):
- `--results-dir` defaults to `results` for official runs.
- `--backend stub` forces `results-dir = scratch/stub` (unless a non-`results/` scratch/tmp
  directory is explicitly provided) and refuses any path under `results/`.
- `--dry-run` (any backend) forces `results-dir = scratch/dryrun` and refuses any path under `results/`.
- Only official runs (a real backend, no `--dry-run`) may write to `results/`
  (pilot units go to `results/pilot/`), and only those append to `results/LEDGER.csv`.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.omp_guard  # noqa: F401, E402
from src.backends.mlx_backend import MLXBackend, unload_active_mlx_model  # noqa: E402
from src.harness import (  # noqa: E402
    Backend,
    StubBackend,
    is_path_under_results,
    load_paragraph_lookup,
    parse_units_list,
    resolve_results_dir,
    run_unit,
)


def make_backend(
    backend_name: str,
    model_name: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> Backend:
    """Instantiate the requested backend for `model_name`."""
    norm = backend_name.strip().lower()
    if norm == "stub":
        return StubBackend(model_name=model_name)
    if norm == "mlx":
        return MLXBackend(model_name=model_name, repo_root=repo_root)
    raise ValueError(f"Unknown backend {backend_name!r}; expected 'stub' or 'mlx'.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute RAG study inference units with resume and ledger tracking."
    )
    parser.add_argument(
        "--units",
        nargs="+",
        required=True,
        help="Space- or comma-separated list of run units to execute.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Maximum number of questions to process per unit (e.g., 20 for dry-run).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="stub",
        choices=("stub", "mlx"),
        help="Inference backend to use ('stub' for tests/dry-run, 'mlx' in P4).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help=(
            "Target output directory (default: 'results' for official runs; forced to "
            "'scratch/stub' for --backend stub and 'scratch/dryrun' for --dry-run; "
            "any path under 'results/' is refused for stub or dry-run)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force output to scratch/dryrun and refuse writing under results/.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root path (used for sandboxed testing).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    explicit_results_dir = any(
        tok == "--results-dir" or tok.startswith("--results-dir=") for tok in raw_argv
    )
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    repo_root = Path(args.repo_root).resolve()
    backend_name = str(args.backend).strip().lower()
    is_stub = backend_name == "stub"
    dry_run = bool(args.dry_run)

    requested_dir = args.results_dir if explicit_results_dir else None

    try:
        units = parse_units_list(args.units)
        resolved_dir = resolve_results_dir(
            requested_dir,
            backend_name=backend_name,
            is_stub=is_stub,
            dry_run=dry_run,
            repo_root=repo_root,
        )
        if (is_stub or dry_run) and is_path_under_results(resolved_dir, repo_root=repo_root):
            raise ValueError(
                f"Refusing to write stub or dry-run outputs under results/: {resolved_dir}"
            )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    raw_data_path = repo_root / "data" / "raw" / "dev-v2.0.json"
    needs_paragraphs = any(u.condition != "closed_book" for u in units)
    paragraph_lookup = (
        load_paragraph_lookup(raw_data_path)
        if needs_paragraphs and raw_data_path.exists()
        else None
    )

    active_backend: Backend | None = None
    for unit in units:
        if active_backend is None or active_backend.model_name != unit.model:
            if active_backend is not None and hasattr(active_backend, "unload"):
                active_backend.unload()
            try:
                active_backend = make_backend(
                    backend_name,
                    unit.model,
                    repo_root=repo_root,
                )
            except Exception as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 2

        summary = run_unit(
            unit=unit,
            backend=active_backend,
            max_questions=args.questions,
            results_dir=resolved_dir,
            dry_run=dry_run,
            repo_root=repo_root,
            paragraph_lookup=paragraph_lookup,
        )
        print(
            f"[UNIT DONE] {summary['unit_id']} -> {summary['output_path']} "
            f"(written={summary['written_count']}, "
            f"resumed_skipped={summary['skipped_resume_count']}, "
            f"p2c_skipped={summary['skipped_p2c_count']})"
        )

    if active_backend is not None and hasattr(active_backend, "unload"):
        active_backend.unload()
    else:
        unload_active_mlx_model()

    return 0


if __name__ == "__main__":
    sys.exit(main())
