#!/usr/bin/env python3
"""Build the deterministic article-stratified SQuAD 2.0 dev sampling frame (main, pilot, reserve)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_PREREG_PATH = REPO_ROOT / "configs" / "prereg.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "sampling"
SEED = 4207
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def largest_remainder_allocate(counts: list[int], target_total: int) -> list[int]:
    """Allocate `target_total` items across strata proportionally via largest-remainder method.

    Uses exact integer arithmetic (`(c * target_total) // total` and `(c * target_total) % total`)
    with deterministic tie-breaking by ascending stratum index.
    """
    total_available = sum(counts)
    if target_total < 0 or target_total > total_available:
        raise ValueError(
            f"Cannot allocate target_total={target_total} from total_available={total_available}"
        )
    if target_total == 0:
        return [0] * len(counts)

    quotas = [(c * target_total) // total_available for c in counts]
    remainders = [(c * target_total) % total_available for c in counts]
    shortfall = target_total - sum(quotas)

    # Sort strata by descending remainder, tie-breaking by ascending stratum (article) index
    ranked_indices = sorted(range(len(counts)), key=lambda idx: (-remainders[idx], idx))
    for idx in ranked_indices[:shortfall]:
        quotas[idx] += 1

    assert sum(quotas) == target_total, f"Sum of quotas {sum(quotas)} != {target_total}"
    assert all(0 <= q <= c for q, c in zip(quotas, counts)), "Quota exceeded available count"
    return quotas


def build_sampling_frame(
    raw_path: Path = DEFAULT_RAW_PATH,
    prereg_path: Path = DEFAULT_PREREG_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = SEED,
) -> dict[str, Any]:
    """Parse SQuAD 2.0 dev, stratify answerable questions by article, and write sampling frame."""
    if not raw_path.exists():
        print(
            f"ERROR: Raw dataset file is missing: {raw_path}\n"
            "Run the following command to download it:\n"
            f"  mkdir -p {raw_path.parent} && curl -fsSLo {raw_path} {SQUAD_DEV_URL}",
            file=sys.stderr,
        )
        raise FileNotFoundError(f"Missing raw SQuAD 2.0 dev file: {raw_path}")

    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    design = prereg["design"]
    n_main = int(design["n_questions"])
    n_pilot = int(design["pilot_questions"])
    n_reserve = int(design["reserve_questions"])

    raw_sha256 = sha256_file(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    articles = payload.get("data", [])

    num_articles = len(articles)
    num_paragraphs = 0
    num_questions = 0
    num_answerable = 0

    article_pools: list[list[dict[str, Any]]] = []
    article_meta: list[dict[str, Any]] = []

    for art_idx, article in enumerate(articles):
        title = article["title"]
        paragraphs = article["paragraphs"]
        num_paragraphs += len(paragraphs)

        art_total_q = 0
        art_answerable_pool: list[dict[str, Any]] = []

        for para_idx, para in enumerate(paragraphs):
            for qa in para["qas"]:
                art_total_q += 1
                num_questions += 1

                is_impossible = bool(qa.get("is_impossible", False))
                raw_answers = qa.get("answers", [])
                if is_impossible or not raw_answers:
                    continue

                gold_answers = [ans["text"] for ans in raw_answers if "text" in ans]
                if not gold_answers:
                    continue

                num_answerable += 1
                art_answerable_pool.append(
                    {
                        "question_id": qa["id"],
                        "article_index": art_idx,
                        "article_title": title,
                        "paragraph_id": [art_idx, para_idx],
                        "paragraph_index": para_idx,
                        "question_text": qa["question"],
                        "gold_answers": gold_answers,
                    }
                )

        article_pools.append(art_answerable_pool)
        article_meta.append(
            {
                "article_index": art_idx,
                "article_title": title,
                "num_paragraphs": len(paragraphs),
                "total_questions": art_total_q,
                "answerable_questions": len(art_answerable_pool),
            }
        )

    ans_counts_initial = [len(pool) for pool in article_pools]
    ans_min = min(ans_counts_initial)
    ans_median = float(statistics.median(ans_counts_initial))
    ans_max = max(ans_counts_initial)

    # Single continuing RNG stream seeded with 4207
    rng = np.random.default_rng(seed)

    def draw_split(
        pools: list[list[dict[str, Any]]], target_n: int
    ) -> tuple[list[dict[str, Any]], list[int], list[list[dict[str, Any]]]]:
        current_counts = [len(p) for p in pools]
        allocations = largest_remainder_allocate(current_counts, target_n)
        drawn_questions: list[dict[str, Any]] = []
        remaining_pools: list[list[dict[str, Any]]] = []

        for pool, k in zip(pools, allocations):
            if k > 0:
                chosen_idx = set(rng.choice(len(pool), size=k, replace=False).tolist())
                drawn_questions.extend(pool[i] for i in range(len(pool)) if i in chosen_idx)
                remaining_pools.append([pool[i] for i in range(len(pool)) if i not in chosen_idx])
            else:
                remaining_pools.append(list(pool))

        return drawn_questions, allocations, remaining_pools

    # Draw main first, then pilot from remaining, then reserve from what remains
    main_questions, main_alloc, pools_after_main = draw_split(article_pools, n_main)
    pilot_questions, pilot_alloc, pools_after_pilot = draw_split(pools_after_main, n_pilot)
    reserve_questions, reserve_alloc, pools_after_reserve = draw_split(
        pools_after_pilot, n_reserve
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    split_files = {
        "main": (out_dir / "main.json", main_questions),
        "pilot": (out_dir / "pilot.json", pilot_questions),
        "reserve": (out_dir / "reserve.json", reserve_questions),
    }

    output_hashes: dict[str, dict[str, Any]] = {}
    for split_name, (file_path, records) in split_files.items():
        file_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_hashes[split_name] = {
            "path": f"data/sampling/{file_path.name}",
            "count": len(records),
            "sha256": sha256_file(file_path),
        }

    per_article_table: list[dict[str, Any]] = []
    for idx, meta in enumerate(article_meta):
        row = dict(meta)
        row["main"] = main_alloc[idx]
        row["pilot"] = pilot_alloc[idx]
        row["reserve"] = reserve_alloc[idx]
        row["remaining_unselected"] = len(pools_after_reserve[idx])
        per_article_table.append(row)

    manifest = {
        "raw_file": "data/raw/dev-v2.0.json",
        "raw_file_sha256": raw_sha256,
        "seed": seed,
        "method": "stratified_largest_remainder",
        "raw_statistics": {
            "num_articles": num_articles,
            "num_paragraphs": num_paragraphs,
            "num_questions": num_questions,
            "num_answerable_questions": num_answerable,
            "answerable_questions_per_article": {
                "min": ans_min,
                "median": ans_median,
                "max": ans_max,
            },
        },
        "split_counts": {
            "main": len(main_questions),
            "pilot": len(pilot_questions),
            "reserve": len(reserve_questions),
        },
        "output_files": output_hashes,
        "per_article_counts": per_article_table,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    try:
        manifest = build_sampling_frame()
    except FileNotFoundError:
        return 1

    stats = manifest["raw_statistics"]
    apa = stats["answerable_questions_per_article"]
    print("=== SQuAD 2.0 Dev Raw Statistics ===")
    print(f"Raw file SHA-256:             {manifest['raw_file_sha256']}")
    print(f"Number of articles:           {stats['num_articles']}")
    print(f"Number of paragraphs (pool):  {stats['num_paragraphs']}")
    print(f"Number of questions:          {stats['num_questions']}")
    print(f"Answerable questions:         {stats['num_answerable_questions']}")
    print(
        f"Answerable / article (min / median / max): "
        f"{apa['min']} / {apa['median']} / {apa['max']}"
    )
    print("\n=== Sampling Frame Split Outputs ===")
    for split_name, info in manifest["output_files"].items():
        print(f"  {split_name:7s}: count={info['count']:3d} | sha256={info['sha256']} | {info['path']}")

    print("\n=== First 10 Rows of Per-Article Allocation Table ===")
    header = f"{'Idx':>3} | {'Article Title':<32} | {'Ans':>4} | {'Main':>4} | {'Pilot':>5} | {'Res':>3}"
    print(header)
    print("-" * len(header))
    for row in manifest["per_article_counts"][:10]:
        print(
            f"{row['article_index']:3d} | {row['article_title'][:32]:<32} | "
            f"{row['answerable_questions']:4d} | {row['main']:4d} | "
            f"{row['pilot']:5d} | {row['reserve']:3d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
