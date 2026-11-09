#!/usr/bin/env python3
"""Validate converted MLX models sequentially (one model per process, offline).

Checks per model:
  (a) 5 hand-written closed-book factual questions: print prompts and outputs.
  (b) Determinism: run the same 20 prompts twice, assert identical outputs.
  (c) Verify Qwen outputs contain no <think> block; print repr() of rendered prompt
      for one question per model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# Enforce offline execution before any HF / MLX imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.omp_guard  # noqa: F401, E402
import yaml  # noqa: E402

FIVE_FACTUAL_QUESTIONS: list[str] = [
    "What is the capital of France?",
    "Which planet is known as the Red Planet?",
    "What is the chemical symbol for gold?",
    "Who wrote the play Hamlet?",
    "In what year did World War II end?",
]

TWENTY_DETERMINISM_QUESTIONS: list[str] = FIVE_FACTUAL_QUESTIONS + [
    "What is the largest ocean on Earth?",
    "What is the speed of light in vacuum in kilometers per second (approximate)?",
    "Who painted the Mona Lisa?",
    "What is the square root of 144?",
    "Which element has the atomic number 1?",
    "What is the capital city of Japan?",
    "How many continents are there on Earth?",
    "What is the boiling point of water at standard atmospheric pressure in degrees Celsius?",
    "Who developed the theory of general relativity?",
    "What currency is used in the United Kingdom?",
    "Which organ in the human body pumps blood?",
    "What is the longest river in Africa?",
    "In which country are the ancient pyramids of Giza located?",
    "What gas do plants primarily absorb during photosynthesis?",
    "How many sides does a standard hexagon have?",
]


def load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg = yaml.safe_load((REPO_ROOT / "configs" / "prereg.yaml").read_text(encoding="utf-8"))
    prompt_cfg = yaml.safe_load((REPO_ROOT / "configs" / "prompt_v1.yaml").read_text(encoding="utf-8"))
    runtime_cfg = yaml.safe_load((REPO_ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
    return prereg, prompt_cfg, runtime_cfg


def render_closed_book_prompt(
    tokenizer: Any,
    model_name: str,
    question: str,
    closed_book_template: str,
) -> str:
    user_content = closed_book_template.format(q=question)
    messages = [{"role": "user", "content": user_content}]
    chat_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_name.startswith("Qwen3"):
        chat_kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **chat_kwargs)
    return str(rendered)


def run_single_model(model_name: str) -> int:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    prereg, prompt_cfg, runtime_cfg = load_configs()
    max_new_tokens = int(prereg["decoding"]["max_new_tokens"])
    assert max_new_tokens == 32, f"Expected max_new_tokens=32, got {max_new_tokens}"
    assert prereg["decoding"]["greedy"] is True

    model_info = runtime_cfg["models"][model_name]
    mlx_dir = REPO_ROOT / model_info["mlx_path"]
    if not mlx_dir.exists():
        raise FileNotFoundError(f"Missing MLX model directory: {mlx_dir}")

    print("=" * 80)
    print(f"MODEL VALIDATION: {model_name}")
    print(f"  MLX Path  : {model_info['mlx_path']}")
    print(f"  HF Repo   : {model_info['hf_repo_id']} @ {model_info['commit_sha']}")
    print(f"  PID       : {os.getpid()} (HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')})")
    print("=" * 80)

    model, tokenizer = load(str(mlx_dir))
    sampler = make_sampler(temp=0.0)
    closed_book_tpl = prompt_cfg["closed_book"]
    is_qwen = model_name.startswith("Qwen3")

    # (c) Print repr() of the rendered prompt for one question per model
    sample_q = FIVE_FACTUAL_QUESTIONS[0]
    sample_rendered = render_closed_book_prompt(tokenizer, model_name, sample_q, closed_book_tpl)
    print(f"\n[{model_name}] (c) Rendered prompt repr() for Question 1:")
    print(f"  RENDERED_PROMPT_REPR[{model_name}]: {repr(sample_rendered)}")

    # (a) 5 hand-written closed-book factual questions: print prompts and outputs
    print(f"\n[{model_name}] (a) 5 Hand-Written Closed-Book Factual Questions:")
    five_outputs: list[str] = []
    for idx, q in enumerate(FIVE_FACTUAL_QUESTIONS, start=1):
        rendered_prompt = render_closed_book_prompt(tokenizer, model_name, q, closed_book_tpl)
        out = generate(
            model,
            tokenizer,
            prompt=rendered_prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        mx.eval()
        five_outputs.append(out)
        print(f"  Q{idx}: {q}")
        print(f"      Prompt (repr): {repr(rendered_prompt)}")
        print(f"      Output (repr): {repr(out)}")
        print(f"      Output       : {out.strip()}")
        if is_qwen and ("<think>" in out or "</think>" in out):
            raise AssertionError(
                f"[{model_name}] Q{idx} output unexpectedly contained a <think> block: {repr(out)}"
            )

    # (b) Determinism: run the same 20 prompts twice, assert identical outputs
    print(f"\n[{model_name}] (b) Determinism Check (20 prompts x 2 passes):")
    rendered_twenty = [
        render_closed_book_prompt(tokenizer, model_name, q, closed_book_tpl)
        for q in TWENTY_DETERMINISM_QUESTIONS
    ]

    # Reuse the first 5 outputs from (a) as pass 1 for Q1..Q5 or run full 20 in pass 1 and pass 2
    pass1_outputs: list[str] = []
    for idx, rendered_prompt in enumerate(rendered_twenty):
        if idx < len(five_outputs):
            out = five_outputs[idx]
        else:
            out = generate(
                model,
                tokenizer,
                prompt=rendered_prompt,
                max_tokens=max_new_tokens,
                sampler=sampler,
                verbose=False,
            )
            mx.eval()
        if is_qwen and ("<think>" in out or "</think>" in out):
            raise AssertionError(
                f"[{model_name}] Pass 1 prompt {idx + 1} contained <think> block: {repr(out)}"
            )
        pass1_outputs.append(out)

    pass2_outputs: list[str] = []
    for idx, rendered_prompt in enumerate(rendered_twenty):
        out = generate(
            model,
            tokenizer,
            prompt=rendered_prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        mx.eval()
        if is_qwen and ("<think>" in out or "</think>" in out):
            raise AssertionError(
                f"[{model_name}] Pass 2 prompt {idx + 1} contained <think> block: {repr(out)}"
            )
        pass2_outputs.append(out)

    mismatches = [
        (i + 1, p1, p2)
        for i, (p1, p2) in enumerate(zip(pass1_outputs, pass2_outputs, strict=True))
        if p1 != p2
    ]
    if mismatches:
        raise AssertionError(f"[{model_name}] Determinism failed on {len(mismatches)} prompts: {mismatches}")

    print(f"  Determinism PASS: 20/20 prompts produced byte-identical outputs across Pass 1 and Pass 2.")
    if is_qwen:
        print(f"  No-<think> PASS : 0/40 Qwen outputs contained '<think>' or '</think>'.")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate converted MLX models sequentially.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run validation for a single model in this process. Omit to run all models sequentially in child processes.",
    )
    args = parser.parse_args()

    prereg, _, _ = load_configs()
    all_models: list[str] = list(prereg["models"])

    if args.model is not None:
        if args.model not in all_models:
            raise ValueError(f"Unknown model {args.model!r}; expected one of {all_models}")
        return run_single_model(args.model)

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    for model_name in all_models:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--model", model_name]
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            print(f"[ERROR] Validation failed for {model_name} (exit code {proc.returncode})", file=sys.stderr)
            return proc.returncode

    print("ALL 3 MODELS VALIDATED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
