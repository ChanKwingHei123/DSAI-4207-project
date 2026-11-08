#!/usr/bin/env python3
"""CLI entry point to build distractor sets and prompt-input JSONL files (Step p2c).

Hard Rule 11 compliance:
- Rejects KMP_DUPLICATE_LIB_OK if set in environment.
- Imports src.omp_guard before any transformers/torch imports.
- Enforces HF_HUB_OFFLINE=1 (no network use).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "KMP_DUPLICATE_LIB_OK" in os.environ:
    raise RuntimeError(
        "Hard Rule 11 violation: KMP_DUPLICATE_LIB_OK was set in the environment prior to omp_guard."
    )

os.environ["HF_HUB_OFFLINE"] = "1"

import src.omp_guard  # noqa: F401, E402
from src.distractors import (  # noqa: E402
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_INPUTS_DIR,
    DEFAULT_LLAMA3_TOKENIZER_PATH,
    DEFAULT_PREREG_PATH,
    DEFAULT_QWEN3_TOKENIZER_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_SAMPLING_DIR,
    build_distractor_inputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build distractor sets, prompt-input JSONL files, and build_report.md."
    )
    parser.add_argument("--prereg-path", type=Path, default=DEFAULT_PREREG_PATH)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--sampling-dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument(
        "--qwen3-tokenizer-path", type=Path, default=DEFAULT_QWEN3_TOKENIZER_PATH
    )
    parser.add_argument(
        "--llama3-tokenizer-path", type=Path, default=DEFAULT_LLAMA3_TOKENIZER_PATH
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise RuntimeError if any question has <20 hard candidates or any random slot fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_distractor_inputs(
        prereg_path=args.prereg_path,
        raw_path=args.raw_path,
        sampling_dir=args.sampling_dir,
        candidates_path=args.candidates_path,
        out_dir=args.out_dir,
        qwen3_tokenizer_path=args.qwen3_tokenizer_path,
        llama3_tokenizer_path=args.llama3_tokenizer_path,
        strict=args.strict,
    )

    q_s = summary["length_matching_stats"]["qwen3"]
    l_s = summary["length_matching_stats"]["llama3"]
    rej = summary["rejection_stats"]
    insuff_count = len(summary["insufficient_hard_pool_questions"])
    failed_slot_count = len(summary["failed_random_slots"])

    print(
        f"[Build Inputs] questions={summary['total_questions']} "
        f"files={len(summary['output_files'])} "
        f"top30_rejections_total={rej['hard_top30_rejected_sum']} "
        f"(mean={rej['hard_top30_rejected_mean']:.4f}, max={rej['hard_top30_rejected_max']}) "
        f"insufficient_hard_pool={insuff_count} "
        f"failed_random_slots={failed_slot_count}"
    )
    print(
        f"[Length Match Qwen3] hard_mean={q_s['hard_mean']:.2f} "
        f"random_mean={q_s['random_mean']:.2f} "
        f"signed_diff_mean={q_s['signed_diff_mean_random_minus_hard']:+.4f} "
        f"abs_diff_mean={q_s['abs_diff_mean']:.4f} "
        f"rel_diff_max={q_s['rel_diff_max'] * 100:.2f}%"
    )
    print(
        f"[Length Match Llama-3.1] hard_mean={l_s['hard_mean']:.2f} "
        f"random_mean={l_s['random_mean']:.2f} "
        f"signed_diff_mean={l_s['signed_diff_mean_random_minus_hard']:+.4f} "
        f"abs_diff_mean={l_s['abs_diff_mean']:.4f} "
        f"rel_diff_max={l_s['rel_diff_max'] * 100:.2f}%"
    )
    for key, info in summary["output_files"].items():
        print(f"  {key:13s}: count={info['count']:3d} | sha256={info['sha256']} | {info['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
