"""Synthetic-truth and loader tests for P5b (`src/stats.py` and `src/analysis_loader.py`).

Covers all P5b requirements:
(a) Residual decision class `(v)` (`class_v`) and complete real-line partition in `evaluate_h1`.
(b) Holm-Bonferroni step-down correction against hand-computed 4-contrast examples
    (monotone adjusted p-values, stopping rule, and Holm-adjusted percentile CIs).
(c) Results loading, scoring, missing-seed handling (1 or 2 missing seeds), and `qid` alignment.
(d) Paired scale & family contrasts, closed-book correctness stratification (within-stratum
    and independent two-sample between-stratum bootstrap), and combined exploratory Holm family.
(e) Context-token length summary (`Qwen3` and `Llama-3.1`) matching hand-computed sums for
    nested prefixes `n in {0, 1, 2, 4, 8}` (plus read-only check on `data/frozen/`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis_loader import (  # noqa: E402
    align_conditions_by_qid,
    load_and_score_results,
    summarize_context_tokens,
)
from src.stats import (  # noqa: E402
    bootstrap_two_sided_p_value,
    compute_all_exploratory_contrasts,
    compute_closed_book_stratified_contrasts,
    compute_scale_and_family_contrasts,
    evaluate_h1,
    holm_bonferroni,
    per_question_accuracy,
)


# ---------------------------------------------------------------------------
# (a) Residual Decision Class (v) and Complete Real-Line Partition
# ---------------------------------------------------------------------------


def test_evaluate_h1_residual_class_v_from_bootstrap_summary() -> None:
    """hard_drop = 3.5 pts with 95% CI [+1.2, +5.8] triggers residual class (v)."""
    boot = {
        "hard_drop": {
            "estimate": 0.035,
            "ci_low": 0.012,
            "ci_high": 0.058,
            "estimate_pts": 3.5,
            "ci_low_pts": 1.2,
            "ci_high_pts": 5.8,
        },
        "random_drop": {
            "estimate": 0.010,
            "ci_low": -0.005,
            "ci_high": 0.025,
            "estimate_pts": 1.0,
            "ci_low_pts": -0.5,
            "ci_high_pts": 2.5,
        },
        "gap": {
            "estimate": 0.025,
            "ci_low": 0.002,
            "ci_high": 0.048,
            "estimate_pts": 2.5,
            "ci_low_pts": 0.2,
            "ci_high_pts": 4.8,
        },
    }
    res = evaluate_h1(bootstrap_result=boot)
    assert res["h1_met"] is False
    assert res["decision_class"] == "(v)"
    assert res["decision_label"] == "class_v"
    assert res["decision_description"] == "H1 not supported; effect detectable but below practical threshold"
    assert res["sub_reason"] == "subthreshold_hard_drop_2_to_5"
    assert res["matched_classes"] == {
        "(i)": False,
        "(ii)": False,
        "(iii)": False,
        "(iv)": False,
        "(v)": True,
    }


def test_evaluate_h1_class_v_end_to_end_synthetic_data() -> None:
    """Construct N=500 synthetic data with hard_drop = 3.5 pts (CI > 0) triggering class (v)."""
    rng = np.random.default_rng(4207)
    n_q = 500
    oracle = np.full(n_q, 0.80, dtype=np.float64)
    jitter = rng.normal(0.0, 0.04, size=n_q)
    jitter -= np.mean(jitter)

    hard_8 = oracle - (0.035 + jitter)
    rand_8 = oracle - (0.010 + 0.5 * jitter)

    res = evaluate_h1(oracle, hard_8, rand_8, n_resamples=2000, seed=4207)
    assert math.isclose(res["bootstrap"]["hard_drop"]["estimate_pts"], 3.5, abs_tol=1e-9)
    assert res["bootstrap"]["hard_drop"]["ci_low_pts"] > 0.0
    assert res["decision_class"] == "(v)"
    assert res["decision_label"] == "class_v"
    assert res["sub_reason"] == "subthreshold_hard_drop_2_to_5"


def test_evaluate_h1_residual_hard_drop_ge_5_gap_unmet_maps_to_class_v() -> None:
    """When hard_drop >= 5 pts (CI > 0) fails H1 on gap and misses (ii), maps to class (v)."""
    # Subcase: 0 < gap < 3 pts with gap_ci_low > 0 (class (i) fails on gap < 3, class (ii) fails on gap CI > 0)
    boot = {
        "hard_drop": {"estimate_pts": 6.0, "ci_low_pts": 3.5, "ci_high_pts": 8.5},
        "random_drop": {"estimate_pts": 4.0, "ci_low_pts": 1.5, "ci_high_pts": 6.5},
        "gap": {"estimate_pts": 2.0, "ci_low_pts": 0.4, "ci_high_pts": 3.6},
    }
    res = evaluate_h1(bootstrap_result=boot)
    assert res["h1_met"] is False
    assert res["decision_class"] == "(v)"
    assert res["decision_label"] == "class_v"
    assert res["sub_reason"] == "hard_drop_met_but_gap_criterion_unmet"


def test_evaluate_h1_complete_real_line_partition_has_zero_gaps() -> None:
    """Exhaustive grid across the real line never produces 'unclassified' and maps (iii) cleanly."""
    # Check small positive hard_drop (< 2) and CI-including-0 cases map to (iii)
    for boot_case in (
        {
            "hard_drop": {"estimate_pts": 1.2, "ci_low_pts": 0.2, "ci_high_pts": 2.2},
            "random_drop": {"estimate_pts": 0.5, "ci_low_pts": -0.3, "ci_high_pts": 1.3},
            "gap": {"estimate_pts": 0.7, "ci_low_pts": -0.2, "ci_high_pts": 1.6},
        },
        {
            "hard_drop": {"estimate_pts": 3.5, "ci_low_pts": -0.2, "ci_high_pts": 7.2},
            "random_drop": {"estimate_pts": 1.0, "ci_low_pts": -1.0, "ci_high_pts": 3.0},
            "gap": {"estimate_pts": 2.5, "ci_low_pts": -0.5, "ci_high_pts": 5.5},
        },
        {
            "hard_drop": {"estimate_pts": -1.5, "ci_low_pts": -3.0, "ci_high_pts": -0.1},
            "random_drop": {"estimate_pts": 0.2, "ci_low_pts": -0.8, "ci_high_pts": 1.2},
            "gap": {"estimate_pts": -1.7, "ci_low_pts": -3.2, "ci_high_pts": -0.2},
        },
    ):
        res = evaluate_h1(bootstrap_result=boot_case)
        assert res["decision_class"] == "(iii)"
        assert res["decision_label"] == "class_iii"

    # Exhaustive sweep across representative regimes of the 3D real line
    hard_est_vals = [-3.0, 0.0, 1.0, 2.0, 3.5, 5.0, 7.0]
    hard_low_vals = [-4.0, -0.5, 0.0, 0.5, 3.0]
    rand_est_vals = [-4.0, 0.0, 2.0, 5.0, 7.0]
    rand_high_vals = [-1.0, 0.0, 4.0, 8.0]
    gap_est_vals = [-3.0, 0.0, 2.0, 3.0, 5.0]
    gap_low_vals = [-2.0, 0.0, 0.5, 2.0]

    for h_e in hard_est_vals:
        for h_lo in hard_low_vals:
            h_hi = max(h_e + 1.0, h_lo + 1.0)
            for r_e in rand_est_vals:
                for r_hi in rand_high_vals:
                    r_lo = min(r_e - 1.0, r_hi - 1.0)
                    for g_e in gap_est_vals:
                        for g_lo in gap_low_vals:
                            g_hi = max(g_e + 1.0, g_lo + 1.0)
                            res = evaluate_h1(
                                bootstrap_result={
                                    "hard_drop": {"estimate_pts": h_e, "ci_low_pts": h_lo, "ci_high_pts": h_hi},
                                    "random_drop": {"estimate_pts": r_e, "ci_low_pts": r_lo, "ci_high_pts": r_hi},
                                    "gap": {"estimate_pts": g_e, "ci_low_pts": g_lo, "ci_high_pts": g_hi},
                                }
                            )
                            assert res["decision_class"] in {"(i)", "(ii)", "(iii)", "(iv)", "(v)"}
                            assert any(res["matched_classes"].values())


# ---------------------------------------------------------------------------
# (b) Holm-Bonferroni Step-Down Correction
# ---------------------------------------------------------------------------


def test_holm_bonferroni_hand_computed_4_contrast_stopping_and_monotonicity() -> None:
    """Verify Holm step-down stopping rule and monotonic adjusted p-values on 4-contrast example."""
    # Unsorted raw p-values at alpha=0.05 (m=4):
    # idx 0: 0.021 -> sorted rank 3 (alpha/2 = 0.025)
    # idx 1: 0.008 -> sorted rank 1 (alpha/4 = 0.0125)
    # idx 2: 0.040 -> sorted rank 4 (alpha/1 = 0.050)
    # idx 3: 0.020 -> sorted rank 2 (alpha/3 = 0.01667) -> FIRST NON-SIGNIFICANT -> STOPS HERE!
    #
    # Hand-computed sorted adjusted p-values:
    # k=1 (idx 1, p=0.008): min(1, 4*0.008) = 0.032 <= 0.05 -> sig=True
    # k=2 (idx 3, p=0.020): max(0.032, min(1, 3*0.020)) = 0.060 > 0.05 -> sig=False (STOP)
    # k=3 (idx 0, p=0.021): max(0.060, min(1, 2*0.021)) = max(0.060, 0.042) = 0.060 -> sig=False
    #                        (even though 0.021 <= 0.05/2 = 0.025, stopping rule prevents rejection!)
    # k=4 (idx 2, p=0.040): max(0.060, min(1, 1*0.040)) = 0.060 -> sig=False
    raw_p = [0.021, 0.008, 0.040, 0.020]
    out = holm_bonferroni(raw_p, alpha=0.05)

    assert out["m"] == 4
    assert out["n_significant"] == 1
    assert out["first_non_significant_rank"] == 2
    assert out["significant"] == [False, True, False, False]
    np.testing.assert_allclose(out["adjusted_p_values"], [0.060, 0.032, 0.060, 0.060], atol=1e-12)
    assert [r["rank"] for r in out["results"]] == [3, 1, 4, 2]
    np.testing.assert_allclose(
        out["holm_ci_levels"],
        [1.0 - 0.05 / 2, 1.0 - 0.05 / 4, 1.0 - 0.05 / 1, 1.0 - 0.05 / 3],
        atol=1e-12,
    )

    # Case 2: Monotonicity when rank 2 rejects (p=0.015 <= 0.05/3), forcing rank 3 p~ = max(0.045, 0.042) = 0.045
    raw_p_all_sig = [0.021, 0.008, 0.040, 0.015]
    out_sig = holm_bonferroni(raw_p_all_sig, alpha=0.05)
    assert out_sig["n_significant"] == 4
    assert out_sig["first_non_significant_rank"] is None
    assert out_sig["significant"] == [True, True, True, True]
    np.testing.assert_allclose(out_sig["adjusted_p_values"], [0.045, 0.032, 0.045, 0.045], atol=1e-12)


def test_holm_bonferroni_from_bootstrap_replicates_and_adjusted_cis() -> None:
    """Verify two-sided bootstrap p-value and Holm-adjusted percentile CIs at 1 - alpha / (m - k + 1)."""
    # 1,000 replicates per contrast with exact known negative counts:
    # c_a: 2 negative -> p = 2 * 2/1000 = 0.004 (rank 1, alpha/3 = 0.01667, CI 98.333%)
    # c_b: 12 negative -> p = 2 * 12/1000 = 0.024 (rank 2, alpha/2 = 0.025, CI 97.5%)
    # c_c: 40 negative -> p = 2 * 40/1000 = 0.080 (rank 3, alpha/1 = 0.05, CI 95.0%)
    def _make_reps(n_neg: int, total: int = 1000) -> np.ndarray:
        vals = np.linspace(0.001, 0.100, total)
        vals[:n_neg] = -np.linspace(0.010, 0.001, n_neg)
        return vals

    reps_a = _make_reps(2)
    reps_b = _make_reps(12)
    reps_c = _make_reps(40)

    assert math.isclose(bootstrap_two_sided_p_value(reps_a), 0.004, abs_tol=1e-12)
    assert math.isclose(bootstrap_two_sided_p_value(reps_b), 0.024, abs_tol=1e-12)
    assert math.isclose(bootstrap_two_sided_p_value(reps_c), 0.080, abs_tol=1e-12)
    assert bootstrap_two_sided_p_value(np.zeros(100)) == 1.0

    out = holm_bonferroni(
        {
            "c_c": {"replicates": reps_c},
            "c_a": {"replicates": reps_a},
            "c_b": {"replicates": reps_b},
        },
        alpha=0.05,
    )
    by_name = out["results_by_name"]
    assert by_name["c_a"]["rank"] == 1
    assert by_name["c_b"]["rank"] == 2
    assert by_name["c_c"]["rank"] == 3
    assert by_name["c_a"]["significant"] is True
    assert by_name["c_b"]["significant"] is True
    assert by_name["c_c"]["significant"] is False
    np.testing.assert_allclose(
        [by_name["c_a"]["p_value_adj"], by_name["c_b"]["p_value_adj"], by_name["c_c"]["p_value_adj"]],
        [0.012, 0.048, 0.080],
        atol=1e-12,
    )

    # Check Holm-adjusted percentile CIs against direct np.percentile
    alpha_1 = 0.05 / 3.0
    exp_lo_a, exp_hi_a = np.percentile(reps_a, [100.0 * (alpha_1 / 2.0), 100.0 * (1.0 - alpha_1 / 2.0)])
    assert math.isclose(by_name["c_a"]["holm_ci_low"], float(exp_lo_a), rel_tol=1e-12)
    assert math.isclose(by_name["c_a"]["holm_ci_high"], float(exp_hi_a), rel_tol=1e-12)
    assert by_name["c_a"]["holm_ci_low"] > 0.0  # matches significant=True

    alpha_2 = 0.05 / 2.0
    exp_lo_b, exp_hi_b = np.percentile(reps_b, [100.0 * (alpha_2 / 2.0), 100.0 * (1.0 - alpha_2 / 2.0)])
    assert math.isclose(by_name["c_b"]["holm_ci_low"], float(exp_lo_b), rel_tol=1e-12)
    assert math.isclose(by_name["c_b"]["holm_ci_high"], float(exp_hi_b), rel_tol=1e-12)
    assert by_name["c_b"]["holm_ci_low"] > 0.0  # boundary parity: p=0.024 < 0.025 -> CI > 0

    alpha_3 = 0.05 / 1.0
    exp_lo_c, exp_hi_c = np.percentile(reps_c, [100.0 * (alpha_3 / 2.0), 100.0 * (1.0 - alpha_3 / 2.0)])
    assert math.isclose(by_name["c_c"]["holm_ci_low"], float(exp_lo_c), rel_tol=1e-12)
    assert by_name["c_c"]["holm_ci_high"], float(exp_hi_c)
    assert by_name["c_c"]["holm_ci_low"] < 0.0  # includes 0 -> non-significant


# ---------------------------------------------------------------------------
# (c) Missing-Seed Handling & Results Alignment Loader
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_missing_seed_per_question_accuracy_and_alignment(tmp_path: Path) -> None:
    """When a question has only 2 of 3 seeds present, mean uses the 2 available seeds and aligns cleanly."""
    # Direct unit test on per_question_accuracy(expected_seeds=None)
    ragged = {
        "q0": [1.0, 1.0, 0.0],
        "q1": [1.0, 0.0],       # 2 seeds -> mean = 0.5
        "q2": [1.0],            # 1 seed -> mean = 1.0
    }
    np.testing.assert_allclose(per_question_accuracy(ragged, expected_seeds=None), [2.0 / 3.0, 0.5, 1.0])
    nan_padded = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, np.nan], [1.0, np.nan, np.nan]])
    np.testing.assert_allclose(per_question_accuracy(nan_padded, expected_seeds=None), [2.0 / 3.0, 0.5, 1.0])

    # End-to-end JSONL results loader test with missing seeds
    gold_path = tmp_path / "gold_main.json"
    qids = ["q0", "q1", "q2", "q3"]
    gold_items = [{"question_id": q, "gold_answers": [f"the {q} answer", f"{q} answer"]} for q in qids]
    gold_path.write_text(json.dumps(gold_items), encoding="utf-8")

    results_dir = tmp_path / "res"

    def _row(model: str, cond: str, dtype: str | None, n: int, seed: int | None, qid: str, ans: str) -> dict[str, Any]:
        return {
            "qid": qid,
            "model": model,
            "condition": cond,
            "distractor_type": dtype,
            "n": n,
            "seed": seed,
            "split": "main",
            "backend": "mlx",
            "is_stub": False,
            "extracted_answer": ans,
            "valid": True,
        }

    for model in ("Qwen3-8B", "Qwen3-4B"):
        _write_jsonl(
            results_dir / f"{model}__closed_book__none__n0__seednone__main.jsonl",
            [_row(model, "closed_book", None, 0, None, q, f"{q} answer" if q == "q0" else "wrong") for q in qids],
        )
        _write_jsonl(
            results_dir / f"{model}__oracle__none__n0__seednone__main.jsonl",
            [_row(model, "oracle", None, 0, None, q, f"{q} answer") for q in qids],
        )
        for seed in (0, 1, 2):
            hard_rows = []
            rand_rows = []
            for q in qids:
                # q1 missing seed 1 in hard_negative n=8 for Qwen3-8B (seeds 0->correct, 2->wrong => mean 0.5)
                if model == "Qwen3-8B" and q == "q1" and seed == 1:
                    continue
                # q3 missing ALL seeds in hard_negative n=8 for Qwen3-4B -> excluded by alignment intersection
                if model == "Qwen3-4B" and q == "q3":
                    continue
                # q2 has only 1 seed (seed 2) in random_length_matched n=8 for Qwen3-8B (partial match -> F1=2/3, EM=0)
                if model == "Qwen3-8B" and q == "q2" and seed in (0, 1):
                    pass  # omitted from random below
                else:
                    rand_ans = f"{q}" if (model == "Qwen3-8B" and q == "q2") else f"{q} answer"
                    rand_rows.append(_row(model, "rag", "random_length_matched", 8, seed, q, rand_ans))

                if model == "Qwen3-8B" and q == "q1":
                    ans_h = f"{q} answer" if seed == 0 else "wrong"
                else:
                    ans_h = f"{q} answer"
                hard_rows.append(_row(model, "rag", "hard_negative", 8, seed, q, ans_h))

            _write_jsonl(
                results_dir / f"{model}__rag__hard_negative__n8__seed{seed}__main.jsonl",
                hard_rows,
            )
            _write_jsonl(
                results_dir / f"{model}__rag__random_length_matched__n8__seed{seed}__main.jsonl",
                rand_rows,
            )

    loaded = load_and_score_results(results_dir, split="main", gold_path=gold_path)
    q8_hard = loaded["per_question"]["Qwen3-8B"][("rag", "hard_negative", 8)]
    assert q8_hard["q1"]["n_seeds"] == 2
    assert q8_hard["q1"]["seeds"] == [0, 2]
    assert math.isclose(q8_hard["q1"]["em"], 0.5, abs_tol=1e-12)
    # Seed 0 ("q1 answer" -> F1=1.0), seed 2 ("wrong" -> F1=0.0) -> mean 0.5
    assert math.isclose(q8_hard["q1"]["f1"], 0.5, abs_tol=1e-12)

    q8_rand = loaded["per_question"]["Qwen3-8B"][("rag", "random_length_matched", 8)]
    assert q8_rand["q2"]["n_seeds"] == 1
    assert q8_rand["q2"]["seeds"] == [2]
    assert q8_rand["q2"]["em"] == 0.0
    # gold "q2 answer" (2 tokens), pred "q2" (1 token): P=1, R=0.5 -> F1 = 2/3
    assert math.isclose(q8_rand["q2"]["f1"], 2.0 / 3.0, abs_tol=1e-12)

    aligned = align_conditions_by_qid(loaded, metric="em")
    assert aligned["qids"] == ["q0", "q1", "q2"]
    assert aligned["dropped_qids"] == ["q3"]
    assert aligned["n_questions"] == 3
    np.testing.assert_allclose(
        aligned["vectors"]["Qwen3-8B"]["hard_negative"][8],
        [1.0, 0.5, 1.0],
    )

    # Stub rows must be rejected when allow_stub=False
    _write_jsonl(
        results_dir / "Qwen3-8B__oracle__none__n0__seednone__main_stub.jsonl",
        [{"qid": "q0", "model": "Qwen3-8B", "condition": "oracle", "distractor_type": None,
          "n": 0, "seed": None, "split": "main", "backend": "stub", "is_stub": True,
          "extracted_answer": "q0 answer"}],
    )
    with pytest.raises(ValueError, match="stub"):
        load_and_score_results(results_dir, split="main", gold_path=gold_path)


# ---------------------------------------------------------------------------
# (d) Scale, Family, and Closed-Book Stratified Contrasts
# ---------------------------------------------------------------------------


def test_scale_and_family_contrasts_recover_planted_effects() -> None:
    """Verify scale (Qwen3-4B - Qwen3-8B) and family (Qwen3-8B - Llama-3.1-8B-Instruct) contrasts."""
    rng = np.random.default_rng(4207)
    n_q = 500
    qids = [f"q{i:03d}" for i in range(n_q)]

    # Plant exact paired question vectors per model
    def _build_model_conds(o_val: float, rand_drop: float, hard_drop: float, scale: float) -> dict[str, Any]:
        noise = rng.normal(0.0, scale, size=n_q)
        noise -= np.mean(noise)
        o = np.full(n_q, o_val) + noise
        r = o - (rand_drop + 0.3 * noise)
        h = o - (hard_drop + 0.6 * noise)
        return {
            "oracle": dict(zip(qids, o)),
            "hard_negative": {8: dict(zip(qids, h))},
            "random_length_matched": {8: dict(zip(qids, r))},
        }

    # Qwen3-8B: hard_drop=10 pts, random_drop=4 pts -> gap=6 pts
    # Qwen3-4B: hard_drop=16 pts, random_drop=5 pts -> gap=11 pts
    # Llama-3.1-8B-Instruct: hard_drop=6 pts, random_drop=3 pts -> gap=3 pts
    model_conds = {
        "Qwen3-4B": _build_model_conds(0.75, 0.05, 0.16, 0.02),
        "Qwen3-8B": _build_model_conds(0.80, 0.04, 0.10, 0.02),
        "Llama-3.1-8B-Instruct": _build_model_conds(0.82, 0.03, 0.06, 0.02),
    }

    res = compute_scale_and_family_contrasts(
        model_conds, n_resamples=2000, seed=4207, return_replicates=True
    )
    assert res["n_questions"] == 500
    c = res["contrasts"]

    # Scale: Qwen3-4B - Qwen3-8B
    assert c["scale_gap"]["direction"] == "Qwen3-4B - Qwen3-8B"
    assert math.isclose(c["scale_gap"]["estimate_pts"], 5.0, abs_tol=1e-9)
    assert math.isclose(c["scale_hard_drop"]["estimate_pts"], 6.0, abs_tol=1e-9)
    assert math.isclose(c["scale_random_drop"]["estimate_pts"], 1.0, abs_tol=1e-9)

    # Family: Qwen3-8B - Llama-3.1-8B-Instruct
    assert c["family_gap"]["direction"] == "Qwen3-8B - Llama-3.1-8B-Instruct"
    assert math.isclose(c["family_gap"]["estimate_pts"], 3.0, abs_tol=1e-9)
    assert math.isclose(c["family_hard_drop"]["estimate_pts"], 4.0, abs_tol=1e-9)
    assert math.isclose(c["family_random_drop"]["estimate_pts"], 1.0, abs_tol=1e-9)

    # Replicate-level algebraic identity proves shared resample indices across all contrasts
    np.testing.assert_allclose(
        c["scale_hard_drop"]["replicates"] - c["scale_random_drop"]["replicates"],
        c["scale_gap"]["replicates"],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        c["family_hard_drop"]["replicates"] - c["family_random_drop"]["replicates"],
        c["family_gap"]["replicates"],
        atol=1e-14,
    )
    assert res["holm_gap"]["m"] == 2
    assert res["holm_gap"]["results_by_name"]["scale_gap"]["significant"] is True
    assert res["holm_gap"]["results_by_name"]["family_gap"]["significant"] is True
    assert res["holm_all"]["m"] == 6


def test_closed_book_stratified_contrasts_and_combined_exploratory_family() -> None:
    """Verify within-stratum CIs, independent two-sample between-stratum diffs, and combined Holm."""
    rng = np.random.default_rng(4207)
    n_correct = 160
    n_wrong = 240
    n_q = n_correct + n_wrong

    cb = np.concatenate([np.ones(n_correct), np.zeros(n_wrong)])
    j_corr = rng.normal(0.0, 0.02, size=n_correct)
    j_corr -= np.mean(j_corr)
    j_wrong = rng.normal(0.0, 0.02, size=n_wrong)
    j_wrong -= np.mean(j_wrong)

    # Correct stratum: hard_drop=3 pts, random_drop=1 pt -> gap=2 pts
    # Wrong stratum:   hard_drop=14 pts, random_drop=4 pts -> gap=10 pts
    # Between strata (wrong - correct): gap=+8 pts, hard_drop=+11 pts, random_drop=+3 pts
    oracle = np.concatenate([0.90 + j_corr, 0.75 + j_wrong])
    random_8 = np.concatenate([0.90 + j_corr - 0.01, 0.75 + j_wrong - 0.04])
    hard_8 = np.concatenate([0.90 + j_corr - 0.03, 0.75 + j_wrong - 0.14])

    res = compute_closed_book_stratified_contrasts(
        cb, oracle, hard_8, random_8, model_name="Qwen3-8B", n_resamples=2000, seed=4207, return_replicates=True
    )
    assert res["n_correct"] == n_correct
    assert res["n_wrong"] == n_wrong

    s_corr = res["strata"]["closed_book_correct"]
    s_wrong = res["strata"]["closed_book_wrong"]
    assert math.isclose(s_corr["gap"]["estimate_pts"], 2.0, abs_tol=1e-9)
    assert math.isclose(s_corr["hard_drop"]["estimate_pts"], 3.0, abs_tol=1e-9)
    assert math.isclose(s_wrong["gap"]["estimate_pts"], 10.0, abs_tol=1e-9)
    assert math.isclose(s_wrong["hard_drop"]["estimate_pts"], 14.0, abs_tol=1e-9)

    between = res["between_strata"]
    assert between["gap"]["direction"] == "closed_book_wrong - closed_book_correct"
    assert math.isclose(between["gap"]["estimate_pts"], 8.0, abs_tol=1e-9)
    assert math.isclose(between["hard_drop"]["estimate_pts"], 11.0, abs_tol=1e-9)
    assert math.isclose(between["random_drop"]["estimate_pts"], 3.0, abs_tol=1e-9)
    assert between["gap"]["ci_low_pts"] > 0.0
    assert between["hard_drop"]["ci_low_pts"] > 0.0
    assert res["holm_primary"]["m"] == 2
    assert res["holm_primary"]["n_significant"] == 2
    assert res["holm_all"]["m"] == 3

    # Independent RNG streams between strata even when n_correct == n_wrong
    cb_eq = np.array([1.0] * 100 + [0.0] * 100)
    o_eq = np.full(200, 0.80)
    r_eq = np.full(200, 0.78)
    h_eq = np.full(200, 0.72) + rng.normal(0.0, 0.05, size=200)
    res_eq = compute_closed_book_stratified_contrasts(
        cb_eq, o_eq, h_eq, r_eq, n_resamples=500, seed=4207, return_replicates=True
    )
    assert not np.array_equal(
        res_eq["strata"]["closed_book_correct"]["resample_indices"],
        res_eq["strata"]["closed_book_wrong"]["resample_indices"],
    )

    # Combined multi-model exploratory suite
    qids = [f"q{i:03d}" for i in range(n_q)]
    model_conds = {}
    for m, shift in (("Qwen3-4B", 0.02), ("Qwen3-8B", 0.0), ("Llama-3.1-8B-Instruct", -0.02)):
        model_conds[m] = {
            "closed_book": dict(zip(qids, cb)),
            "oracle": dict(zip(qids, oracle)),
            "hard_negative": {8: dict(zip(qids, hard_8 - shift))},
            "random_length_matched": {8: dict(zip(qids, random_8))},
        }
    all_exp = compute_all_exploratory_contrasts(model_conds, n_resamples=1000, seed=4207)
    assert all_exp["holm_core_gap_family"]["m"] == 5  # scale_gap + family_gap + 3 stratified_gap
    assert all_exp["holm_full_family"]["m"] == 15     # 6 scale/family + 3 * 3 stratified


# ---------------------------------------------------------------------------
# (e) Context-Token Summary Matching Hand-Computed Nested Prefix Sums
# ---------------------------------------------------------------------------


def test_context_token_summary_matches_hand_computed_nested_prefixes(tmp_path: Path) -> None:
    """Verify mean and median context token lengths for closed_book, oracle, and n in {1,2,4,8}."""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 2 questions x 2 seeds = 4 rows
    # q1 gold=(100, 90), hard=[10,20,30,40,50,60,70,80], rand=[11,19,31,39,51,59,71,79]
    # q2 gold=(200, 180), hard=[5,5,5,5,5,5,5,5], rand=[6,4,6,4,6,4,6,4]
    seed0_rows = [
        {
            "question_id": "q1",
            "token_lengths": {
                "qwen3": {"gold": 100, "hard": [10, 20, 30, 40, 50, 60, 70, 80], "random": [11, 19, 31, 39, 51, 59, 71, 79]},
                "llama3": {"gold": 90, "hard": [9, 18, 27, 36, 45, 54, 63, 72], "random": [10, 17, 28, 35, 46, 53, 64, 71]},
            },
        },
        {
            "question_id": "q2",
            "token_lengths": {
                "qwen3": {"gold": 200, "hard": [5, 5, 5, 5, 5, 5, 5, 5], "random": [6, 4, 6, 4, 6, 4, 6, 4]},
                "llama3": {"gold": 180, "hard": [4, 4, 4, 4, 4, 4, 4, 4], "random": [5, 3, 5, 3, 5, 3, 5, 3]},
            },
        },
    ]
    seed1_rows = [
        {
            "question_id": "q1",
            "token_lengths": {
                "qwen3": {"gold": 100, "hard": [20, 20, 20, 20, 20, 20, 20, 20], "random": [21, 19, 21, 19, 21, 19, 21, 19]},
                "llama3": {"gold": 90, "hard": [18, 18, 18, 18, 18, 18, 18, 18], "random": [19, 17, 19, 17, 19, 17, 19, 17]},
            },
        },
        {
            "question_id": "q2",
            "token_lengths": {
                "qwen3": {"gold": 200, "hard": [10, 10, 10, 10, 10, 10, 10, 10], "random": [10, 10, 10, 10, 10, 10, 10, 10]},
                "llama3": {"gold": 180, "hard": [8, 8, 8, 8, 8, 8, 8, 8], "random": [8, 8, 8, 8, 8, 8, 8, 8]},
            },
        },
    ]
    _write_jsonl(inputs_dir / "main_seed0.jsonl", seed0_rows)
    _write_jsonl(inputs_dir / "main_seed1.jsonl", seed1_rows)

    summary = summarize_context_tokens(split="main", inputs_dir=inputs_dir, seeds=(0, 1, 2))
    assert summary["n_unique_questions"] == 2
    assert summary["n_seed_rows"] == 4
    by_cond = summary["by_condition"]

    # closed_book = 0
    assert by_cond["closed_book"]["qwen3"]["mean"] == 0.0
    assert by_cond["closed_book"]["qwen3"]["median"] == 0.0
    assert by_cond["closed_book"]["llama3"]["mean"] == 0.0

    # oracle = gold only: qwen3 [100, 200] -> mean 150, median 150; llama3 [90, 180] -> mean 135
    assert by_cond["oracle"]["n_rows"] == 2
    assert by_cond["oracle"]["qwen3"]["mean"] == pytest.approx(150.0)
    assert by_cond["oracle"]["qwen3"]["median"] == pytest.approx(150.0)
    assert by_cond["oracle"]["llama3"]["mean"] == pytest.approx(135.0)
    assert by_cond["oracle"]["llama3"]["median"] == pytest.approx(135.0)

    # hard_negative n=1 qwen3: [110, 205, 120, 210] -> sorted [110, 120, 205, 210]
    # mean = 645/4 = 161.25, median = (120+205)/2 = 162.5
    assert by_cond["hard_negative_n1"]["n_rows"] == 4
    assert by_cond["hard_negative_n1"]["qwen3"]["mean"] == pytest.approx(161.25)
    assert by_cond["hard_negative_n1"]["qwen3"]["median"] == pytest.approx(162.5)

    # hard_negative n=2 qwen3: [130, 210, 140, 220] -> mean = 700/4 = 175.0, median = 175.0
    assert by_cond["hard_negative_n2"]["qwen3"]["mean"] == pytest.approx(175.0)
    assert by_cond["hard_negative_n2"]["qwen3"]["median"] == pytest.approx(175.0)

    # hard_negative n=4 qwen3: [200, 220, 180, 240] -> sorted [180, 200, 220, 240] -> mean=210, median=210
    assert by_cond["hard_negative_n4"]["qwen3"]["mean"] == pytest.approx(210.0)
    assert by_cond["hard_negative_n4"]["qwen3"]["median"] == pytest.approx(210.0)

    # hard_negative n=8 qwen3: [460, 240, 260, 280] -> sorted [240, 260, 280, 460] -> mean=310, median=270
    assert by_cond["hard_negative_n8"]["qwen3"]["mean"] == pytest.approx(310.0)
    assert by_cond["hard_negative_n8"]["qwen3"]["median"] == pytest.approx(270.0)

    # random_length_matched n=8 llama3:
    # q1_s0: 90 + (10+17+28+35+46+53+64+71) = 90 + 324 = 414
    # q2_s0: 180 + 4*(5+3) = 212
    # q1_s1: 90 + 4*(19+17) = 234
    # q2_s1: 180 + 8*8 = 244
    # sorted [212, 234, 244, 414] -> mean = 1104/4 = 276.0, median = (234+244)/2 = 239.0
    assert by_cond["random_length_matched_n8"]["llama3"]["mean"] == pytest.approx(276.0)
    assert by_cond["random_length_matched_n8"]["llama3"]["median"] == pytest.approx(239.0)


def test_context_token_summary_frozen_main_read_only() -> None:
    """Verify context-token summary runs cleanly against read-only data/frozen/ main inputs."""
    frozen_manifest = REPO_ROOT / "data" / "frozen" / "manifest.json"
    if not frozen_manifest.exists():
        pytest.skip("data/frozen/manifest.json not present")
    summary = summarize_context_tokens(split="main", repo_root=REPO_ROOT)
    assert summary["n_unique_questions"] == 500
    assert summary["n_seed_rows"] == 1500
    by_cond = summary["by_condition"]
    assert by_cond["closed_book"]["qwen3"]["mean"] == 0.0
    # Nested prefixes strictly increase mean context token counts
    means = [
        by_cond["oracle"]["qwen3"]["mean"],
        by_cond["hard_negative_n1"]["qwen3"]["mean"],
        by_cond["hard_negative_n2"]["qwen3"]["mean"],
        by_cond["hard_negative_n4"]["qwen3"]["mean"],
        by_cond["hard_negative_n8"]["qwen3"]["mean"],
    ]
    assert all(a < b for a, b in zip(means, means[1:]))
