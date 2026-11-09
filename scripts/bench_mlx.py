#!/usr/bin/env python3
"""Benchmark converted MLX models sequentially (one model per process, offline).

Per model:
  - 30 synthetic RAG prompts at n=0 (~70 tokens, 1 passage) built from real SQuAD dev paragraphs
  - 30 synthetic RAG prompts at n=8 (~1,400 tokens, 9 passages) built from real SQuAD dev paragraphs
  - Greedy decoding, max 32 new tokens, chat template applied (Qwen3 enable_thinking=False)
  - Reports:
      * Prompt token count (mean, min, max)
      * Seconds per generation (median, p90, mean)
      * Prompt-processing tokens/s (median, mean)
      * Generation tokens/s (median, mean)
      * Peak memory (GB)
  - Prints estimated wall-clock time for 39,000 generations and compressed 21,000 generations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Enforce offline execution before any HF / MLX imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.omp_guard  # noqa: F401, E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

NUM_PROMPTS_PER_LEVEL = 30
TARGET_N0_TOKENS = 70
TARGET_N8_TOKENS = 1400


def load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg = yaml.safe_load((REPO_ROOT / "configs" / "prereg.yaml").read_text(encoding="utf-8"))
    prompt_cfg = yaml.safe_load((REPO_ROOT / "configs" / "prompt_v1.yaml").read_text(encoding="utf-8"))
    runtime_cfg = yaml.safe_load((REPO_ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
    return prereg, prompt_cfg, runtime_cfg


def render_rag_prompt(
    tokenizer: Any,
    model_name: str,
    passages: list[str],
    question: str,
    rag_tpl: dict[str, str],
) -> str:
    user_content = (
        rag_tpl["header"]
        + "".join(
            rag_tpl["passage_line"].format(i=i, text=text)
            for i, text in enumerate(passages, start=1)
        )
        + rag_tpl["footer"].format(q=question)
    )
    messages = [{"role": "user", "content": user_content}]
    chat_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_name.startswith("Qwen3"):
        chat_kwargs["enable_thinking"] = False
    return str(tokenizer.apply_chat_template(messages, **chat_kwargs))


def build_benchmark_prompts(
    tokenizer: Any,
    model_name: str,
    rag_tpl: dict[str, str],
    n_prompts: int = NUM_PROMPTS_PER_LEVEL,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Build 30 prompts at n=0 (~70 tokens) and 30 prompts at n=8 (~1,400 tokens) from real dev paragraphs."""
    dev_path = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing SQuAD dev file: {dev_path}")

    raw_data = json.loads(dev_path.read_text(encoding="utf-8"))["data"]
    para_records: list[tuple[str, str]] = []
    for article in raw_data:
        for para in article["paragraphs"]:
            context = para["context"].strip()
            answerable_qs = [
                qa["question"].strip()
                for qa in para["qas"]
                if not qa.get("is_impossible", False) and qa.get("question", "").strip()
            ]
            if answerable_qs:
                para_records.append((context, answerable_qs[0]))

    # 1. Select 30 distinct real dev (paragraph, question) pairs whose rendered n=0 prompt is closest to 70 tokens
    n0_candidates: list[tuple[int, int, str, int]] = []
    for idx, (context, q) in enumerate(para_records):
        rendered = render_rag_prompt(tokenizer, model_name, [context], q, rag_tpl)
        tok_len = len(tokenizer.encode(rendered))
        n0_candidates.append((abs(tok_len - TARGET_N0_TOKENS), idx, rendered, tok_len))

    n0_candidates.sort(key=lambda item: (item[0], item[1]))
    n0_prompts = [(rendered, tok_len) for _, _, rendered, tok_len in n0_candidates[:n_prompts]]

    # 2. Build 30 distinct 9-passage (n=8: 1 gold + 8 distractors) prompts around ~1,400 tokens
    # Measure single-paragraph token lengths to pick 9-paragraph windows near 1,400 total tokens
    para_tok_lens = [len(tokenizer.encode(ctx)) for ctx, _ in para_records]
    # Sort paragraphs by token length so contiguous/strided groups of 9 span various lengths
    order = sorted(range(len(para_records)), key=lambda i: (para_tok_lens[i], i))

    n8_candidates: list[tuple[int, int, str, int]] = []
    # Generate candidate 9-passage sets across the sorted pool
    step = max(1, (len(order) - 9) // 180)
    for start in range(0, len(order) - 9, step):
        group_indices = order[start : start + 9]
        passages = [para_records[i][0] for i in group_indices]
        question = para_records[group_indices[0]][1]
        rendered = render_rag_prompt(tokenizer, model_name, passages, question, rag_tpl)
        tok_len = len(tokenizer.encode(rendered))
        n8_candidates.append((abs(tok_len - TARGET_N8_TOKENS), start, rendered, tok_len))

    n8_candidates.sort(key=lambda item: (item[0], item[1]))
    n8_prompts = [(rendered, tok_len) for _, _, rendered, tok_len in n8_candidates[:n_prompts]]

    return n0_prompts, n8_prompts


def benchmark_prompt_set(
    model: Any,
    tokenizer: Any,
    prompts_with_len: list[tuple[str, int]],
    max_new_tokens: int,
) -> dict[str, float]:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    # Warm-up generation (1 prompt)
    warmup_prompt = prompts_with_len[0][0]
    for _ in stream_generate(
        model,
        tokenizer,
        prompt=warmup_prompt,
        max_tokens=max_new_tokens,
        sampler=sampler,
    ):
        pass
    mx.eval()
    mx.reset_peak_memory()

    latencies: list[float] = []
    prompt_tps_list: list[float] = []
    gen_tps_list: list[float] = []
    prompt_tokens_list: list[int] = []
    gen_tokens_list: list[int] = []
    peak_mem_gb = 0.0

    for rendered_prompt, _ in prompts_with_len:
        t0 = time.perf_counter()
        last_resp = None
        for resp in stream_generate(
            model,
            tokenizer,
            prompt=rendered_prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        ):
            last_resp = resp
        mx.eval()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        if last_resp is not None:
            prompt_tps_list.append(float(last_resp.prompt_tps))
            gen_tps_list.append(float(last_resp.generation_tps))
            prompt_tokens_list.append(int(last_resp.prompt_tokens))
            gen_tokens_list.append(int(last_resp.generation_tokens))
            peak_mem_gb = max(peak_mem_gb, float(last_resp.peak_memory))

    mx_peak_gb = mx.get_peak_memory() / 1e9
    peak_mem_gb = max(peak_mem_gb, mx_peak_gb)

    lat_arr = np.asarray(latencies, dtype=np.float64)
    ptps_arr = np.asarray(prompt_tps_list, dtype=np.float64)
    gtps_arr = np.asarray(gen_tps_list, dtype=np.float64)
    ptok_arr = np.asarray(prompt_tokens_list, dtype=np.float64)
    gtok_arr = np.asarray(gen_tokens_list, dtype=np.float64)

    return {
        "n_prompts": int(len(prompts_with_len)),
        "mean_prompt_tokens": float(np.mean(ptok_arr)),
        "min_prompt_tokens": int(np.min(ptok_arr)),
        "max_prompt_tokens": int(np.max(ptok_arr)),
        "mean_gen_tokens": float(np.mean(gtok_arr)),
        "sec_per_gen_median": float(np.median(lat_arr)),
        "sec_per_gen_p90": float(np.percentile(lat_arr, 90)),
        "sec_per_gen_mean": float(np.mean(lat_arr)),
        "prompt_tps_median": float(np.median(ptps_arr)),
        "prompt_tps_mean": float(np.mean(ptps_arr)),
        "gen_tps_median": float(np.median(gtps_arr)),
        "gen_tps_mean": float(np.mean(gtps_arr)),
        "peak_memory_gb": float(peak_mem_gb),
    }


def run_single_model(model_name: str, json_out: Path | None = None) -> int:
    from mlx_lm import load

    prereg, prompt_cfg, runtime_cfg = load_configs()
    max_new_tokens = int(prereg["decoding"]["max_new_tokens"])
    model_info = runtime_cfg["models"][model_name]
    mlx_dir = REPO_ROOT / model_info["mlx_path"]

    print("=" * 80)
    print(f"MLX BENCHMARK: {model_name} (PID {os.getpid()})")
    print(f"  MLX Path : {model_info['mlx_path']}")
    print("=" * 80)

    model, tokenizer = load(str(mlx_dir))
    rag_tpl = prompt_cfg["rag"]

    n0_prompts, n8_prompts = build_benchmark_prompts(tokenizer, model_name, rag_tpl)
    stats_n0 = benchmark_prompt_set(model, tokenizer, n0_prompts, max_new_tokens)
    stats_n8 = benchmark_prompt_set(model, tokenizer, n8_prompts, max_new_tokens)

    result = {
        "model": model_name,
        "mlx_path": model_info["mlx_path"],
        "n0": stats_n0,
        "n8": stats_n8,
    }

    for label, key in (("n=0 (~70 tok)", "n0"), ("n=8 (~1,400 tok)", "n8")):
        s = result[key]
        print(
            f"  [{model_name}] {label}: "
            f"prompt_tokens={s['mean_prompt_tokens']:.1f} (min={s['min_prompt_tokens']}, max={s['max_prompt_tokens']}), "
            f"gen_tokens={s['mean_gen_tokens']:.1f}, "
            f"sec/gen median={s['sec_per_gen_median']:.3f}s (p90={s['sec_per_gen_p90']:.3f}s), "
            f"prompt_tps={s['prompt_tps_median']:.1f} tok/s, "
            f"gen_tps={s['gen_tps_median']:.1f} tok/s, "
            f"peak_mem={s['peak_memory_gb']:.2f} GB"
        )

    if json_out is not None:
        json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0


def format_hours(seconds: float) -> str:
    hours = seconds / 3600.0
    mins = seconds / 60.0
    return f"{hours:.2f} h ({mins:.1f} min)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLX models sequentially.")
    parser.add_argument("--model", type=str, default=None, help="Single model to benchmark in this process.")
    parser.add_argument("--json-out", type=Path, default=None, help="Path to write JSON summary for single model.")
    args = parser.parse_args()

    prereg, _, _ = load_configs()
    all_models: list[str] = list(prereg["models"])

    if args.model is not None:
        if args.model not in all_models:
            raise ValueError(f"Unknown model {args.model!r}; expected one of {all_models}")
        return run_single_model(args.model, args.json_out)

    import tempfile

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    all_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for model_name in all_models:
            out_file = Path(tmpdir) / f"{model_name}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                model_name,
                "--json-out",
                str(out_file),
            ]
            proc = subprocess.run(cmd, env=env, check=False)
            if proc.returncode != 0:
                print(f"[ERROR] Benchmark failed for {model_name} (exit code {proc.returncode})", file=sys.stderr)
                return proc.returncode
            all_results.append(json.loads(out_file.read_text(encoding="utf-8")))

    print("\n" + "=" * 108)
    print("BENCHMARK SUMMARY TABLE (30 prompts at n=0 and 30 prompts at n=8 per model, greedy max_new_tokens=32)")
    print("=" * 108)
    header = (
        f"{'Model':<22} | {'Cond':<5} | {'Tok (mean)':>10} | "
        f"{'Med s/gen':>9} | {'P90 s/gen':>9} | "
        f"{'Prompt tok/s':>12} | {'Gen tok/s':>9} | {'Peak Mem':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        m = r["model"]
        for cond_label, key in (("n=0", "n0"), ("n=8", "n8")):
            s = r[key]
            print(
                f"{m:<22} | {cond_label:<5} | {s['mean_prompt_tokens']:>10.1f} | "
                f"{s['sec_per_gen_median']:>8.3f}s | {s['sec_per_gen_p90']:>8.3f}s | "
                f"{s['prompt_tps_median']:>12.1f} | {s['gen_tps_median']:>9.1f} | "
                f"{s['peak_memory_gb']:>7.2f} GB"
            )

    # Estimate wall-clock times for 39,000 generations (13,000/model) and compressed 21,000 (7,000/model)
    # In the full grid (26 generations/question):
    #   closed-book (1), n=0 (1 doc, 1), n=1 (2 docs, 6), n=2 (3 docs, 6), n=4 (5 docs, 6), n=8 (9 docs, 6).
    #   Average passages per prompt across the 26 draws is (0 + 1 + 6*2 + 6*3 + 6*5 + 6*9)/26 = 115/26 = 4.423 docs.
    #   Linear interpolation weight between n=0 (1 doc) and n=8 (9 docs): w_n8 = (4.423 - 1) / 8 = 0.4279, w_n0 = 0.5721.
    w_n8 = (115.0 / 26.0 - 1.0) / 8.0
    w_n0 = 1.0 - w_n8

    mean_n0_med = float(np.mean([r["n0"]["sec_per_gen_median"] for r in all_results]))
    mean_n8_med = float(np.mean([r["n8"]["sec_per_gen_median"] for r in all_results]))
    weighted_sec_per_gen = w_n0 * mean_n0_med + w_n8 * mean_n8_med
    midpoint_sec_per_gen = 0.5 * (mean_n0_med + mean_n8_med)

    print("\n" + "=" * 108)
    print("ESTIMATED RUNTIME FOR 39,000 GENERATIONS (FULL GRID) AND 21,000 GENERATIONS (COMPRESSED GRID)")
    print("=" * 108)
    print(f"Across-model average median s/gen : n=0: {mean_n0_med:.3f}s | n=8: {mean_n8_med:.3f}s")
    print(f"Dose-weighted mean s/gen (w_n0={w_n0:.3f}, w_n8={w_n8:.3f}): {weighted_sec_per_gen:.3f}s")
    print(f"50/50 midpoint mean s/gen         : {midpoint_sec_per_gen:.3f}s")
    print("-" * 108)
    for total_gens, label in ((39000, "Full 39,000 generations (13,000/model)"), (21000, "Compressed 21,000 generations (7,000/model)")):
        t_n0 = total_gens * mean_n0_med
        t_weighted = total_gens * weighted_sec_per_gen
        t_mid = total_gens * midpoint_sec_per_gen
        t_n8 = total_gens * mean_n8_med
        print(f"{label}:")
        print(f"  - Dose-weighted estimate : {format_hours(t_weighted)}")
        print(f"  - 50/50 midpoint estimate: {format_hours(t_mid)}")
        print(f"  - Range [all n=0 .. n=8] : {format_hours(t_n0)} .. {format_hours(t_n8)}")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
