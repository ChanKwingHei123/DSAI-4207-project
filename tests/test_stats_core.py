"""Unit and synthetic statistical validation tests for P5a (`src/metrics.py` and `src/stats.py`).

Covers all required P5a checks:
1. Official SQuAD normalization on known SQuAD examples; exact_match against ANY gold answer;
   token_f1 (max over gold answers) and exact 2PR/(P+R) verification on hand-computed cases.
2. Per-question accuracy averaged over the 3 seeds; unit of resampling = QUESTION.
3. Paired bootstrap (10,000 resamples, fixed seed) with identical resample indices across
   hard drop, random drop, and gap.
4. Synthetic data of known truth:
   - Planted effect recovered;
   - Null gives ~5% false positives over repeated draws;
   - 95% CI coverage near 95% over repeated draws;
   - Each decision-table class (i), (ii), (iii), (iv) triggered by a constructed case;
   - Dynamic reading of `primary_comparison.n` from `prereg.yaml` (never hardcoded).
5. Power helper at N=500 and N=800 and the extend-to-800 rule when gap power < 0.80.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (  # noqa: E402
    compute_exact,
    compute_f1,
    exact_match,
    get_tokens,
    normalize_answer,
    token_f1,
)
from src.stats import (  # noqa: E402
    DEFAULT_PREREG_PATH,
    evaluate_h1,
    estimate_power,
    load_prereg,
    paired_bootstrap,
    per_question_accuracy,
)


# ---------------------------------------------------------------------------
# 1. Official SQuAD Normalization, Exact Match, and Hand-Computed Token F1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Denver Broncos!", "denver broncos"),
        ("  An   apple, a pear, and THE banana.  ", "apple pear and banana"),
        ("1990s - 2000s (approx.)", "1990s 2000s approx"),
        ("St. John's Wort", "st johns wort"),
        ("A", ""),
        ("the   , . ! ?  an", ""),
        ("Super Bowl 50", "super bowl 50"),
        ("Cam Newton, quarterback of the Carolina Panthers", "cam newton quarterback of carolina panthers"),
    ],
)
def test_normalize_answer_matches_known_squad_examples(raw: str, expected: str) -> None:
    """Verify official SQuAD normalization pipeline (lower -> punc -> articles -> whitespace)."""
    assert normalize_answer(raw) == expected


def test_normalize_order_punctuation_before_articles() -> None:
    """In official SQuAD, punctuation is stripped before article removal so `'the'` -> `the` -> removed."""
    assert normalize_answer("'The' United States") == "united states"
    assert get_tokens("  The,   quick-brown fox! ") == ["quickbrown", "fox"]


def test_exact_match_against_any_gold_answer() -> None:
    """Verify exact_match returns 1.0 when prediction matches ANY normalized gold answer."""
    gold_answers = ["19th century", "the nineteenth century", "1800s"]
    assert exact_match("The 19th Century!", gold_answers) == 1.0
    assert exact_match("nineteenth century", gold_answers) == 1.0
    assert exact_match("the 1800s.", gold_answers) == 1.0
    assert exact_match("20th century", gold_answers) == 0.0
    assert compute_exact("The Apollo 11", "apollo 11!") == 1


def test_token_f1_hand_computed_formula_2pr_over_p_plus_r() -> None:
    """Verify token_f1 equals 2*P*R / (P + R) on hand-computed token multiset examples."""
    # Case 1: gold = "new york city" (3 tokens), pred = "new york" (2 tokens)
    # Overlap = 2 ("new", "york") -> P = 2/2 = 1.0, R = 2/3, F1 = 2*(1)*(2/3) / (1 + 2/3) = 4/5 = 0.8
    p1 = 2.0 / 2.0
    r1 = 2.0 / 3.0
    expected_f1_1 = (2.0 * p1 * r1) / (p1 + r1)
    assert math.isclose(compute_f1("The New York City!", "new york"), expected_f1_1, rel_tol=1e-12)
    assert math.isclose(expected_f1_1, 0.8, rel_tol=1e-12)

    # Case 2: duplicate tokens in multiset
    # gold = "red red blue green" (4 tokens), pred = "red blue blue" (3 tokens)
    # Counter intersection = {"red": 1, "blue": 1} -> num_same = 2
    # P = 2/3, R = 2/4 = 1/2 -> F1 = 2*(2/3)*(1/2) / (2/3 + 1/2) = (2/3) / (7/6) = 4/7
    p2 = 2.0 / 3.0
    r2 = 2.0 / 4.0
    expected_f1_2 = (2.0 * p2 * r2) / (p2 + r2)
    assert math.isclose(compute_f1("red red blue green", "red blue blue"), expected_f1_2, rel_tol=1e-12)
    assert math.isclose(expected_f1_2, 4.0 / 7.0, rel_tol=1e-12)

    # Case 3: max over multiple gold answers
    # Pred = "john f kennedy" (3 tokens)
    # Gold A = "kennedy" (1 token) -> P=1/3, R=1/1 -> F1 = 0.5
    # Gold B = "president john f. kennedy" (4 tokens) -> overlap=3 -> P=3/3=1.0, R=3/4 -> F1 = 6/7
    gold_list = ["Kennedy", "President John F. Kennedy", "JFK"]
    assert math.isclose(token_f1("The John F Kennedy", gold_list), 6.0 / 7.0, rel_tol=1e-12)

    # Case 4: zero overlap and empty-after-normalization edge cases
    assert compute_f1("apple orange", "banana grape") == 0.0
    assert compute_f1("the!", "a?") == 1.0  # both normalize to empty token list
    assert compute_f1("the!", "apple") == 0.0


def test_metrics_reject_invalid_gold_inputs() -> None:
    """Ensure passing a bare string or empty sequence as gold_answers raises an error."""
    with pytest.raises(TypeError):
        exact_match("ans", "not_a_list")
    with pytest.raises(ValueError):
        token_f1("ans", [])


# ---------------------------------------------------------------------------
# 2. Per-Question Accuracy (Mean Over 3 Seeds, Unit = Question)
# ---------------------------------------------------------------------------


def test_per_question_accuracy_averages_three_seeds() -> None:
    """Verify per-question accuracy averages across the 3 seeds and preserves question count."""
    raw_scores = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    q_acc = per_question_accuracy(raw_scores, expected_seeds=3)
    assert q_acc.shape == (4,)
    np.testing.assert_allclose(q_acc, [1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0])

    # Mapping input keyed by question_id
    by_qid = {
        "q1": [1.0, 1.0, 1.0],
        "q2": [0.0, 1.0, 0.0],
    }
    q_acc_map = per_question_accuracy(by_qid, expected_seeds=3)
    np.testing.assert_allclose(q_acc_map, [1.0, 1.0 / 3.0])

    # Wrong seed count raises ValueError
    with pytest.raises(ValueError, match="Expected 3 seeds"):
        per_question_accuracy(np.ones((4, 2)), expected_seeds=3)


def test_bootstrap_unit_of_resampling_is_question_not_seed() -> None:
    """Verify passing 2D (N_questions, 3_seeds) arrays resamples N questions, not 3*N seed rows."""
    rng = np.random.default_rng(123)
    n_questions = 40
    oracle_2d = rng.integers(0, 2, size=(n_questions, 3)).astype(np.float64)
    hard_2d = rng.integers(0, 2, size=(n_questions, 3)).astype(np.float64)
    random_2d = rng.integers(0, 2, size=(n_questions, 3)).astype(np.float64)

    res_from_2d = paired_bootstrap(
        oracle_2d, hard_2d, random_2d, n_resamples=500, seed=4207, return_replicates=True
    )
    res_from_1d = paired_bootstrap(
        per_question_accuracy(oracle_2d),
        per_question_accuracy(hard_2d),
        per_question_accuracy(random_2d),
        n_resamples=500,
        seed=4207,
        return_replicates=True,
    )

    assert res_from_2d["unit"] == "question"
    assert res_from_2d["n_questions"] == n_questions
    assert res_from_2d["resample_indices"].shape == (500, n_questions)
    np.testing.assert_allclose(
        res_from_2d["hard_drop"]["replicates"], res_from_1d["hard_drop"]["replicates"]
    )


# ---------------------------------------------------------------------------
# 3. Paired Bootstrap: Shared Resample Indices & Synthetic Truth Tests
# ---------------------------------------------------------------------------


def test_paired_bootstrap_uses_identical_resample_indices_for_all_three_quantities() -> None:
    """In every single bootstrap replicate b, hard_drop[b] - random_drop[b] must equal gap[b]."""
    rng = np.random.default_rng(4207)
    n_questions = 200
    oracle = rng.choice([0.0, 1.0], size=n_questions, p=[0.2, 0.8])
    hard_8 = rng.choice([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], size=n_questions)
    random_8 = rng.choice([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], size=n_questions)

    res = paired_bootstrap(
        oracle, hard_8, random_8, n_resamples=10000, ci_level=0.95, seed=4207, return_replicates=True
    )

    rep_hard = res["hard_drop"]["replicates"]
    rep_rand = res["random_drop"]["replicates"]
    rep_gap = res["gap"]["replicates"]

    assert rep_hard.shape == (10000,)
    assert rep_rand.shape == (10000,)
    assert rep_gap.shape == (10000,)

    # Algebraic identity holds replicate-by-replicate iff the exact same resample indices were used
    np.testing.assert_allclose(rep_hard - rep_rand, rep_gap, atol=1e-15)


def test_synthetic_planted_effect_recovered() -> None:
    """Verify planted hard drop (8 pts), random drop (3 pts), and gap (5 pts) are recovered."""
    rng = np.random.default_rng(2026)
    n_questions = 500

    # Construct synthetic paired data with exact sample means:
    # oracle = 0.80, random_8 = 0.77 (random_drop = 0.03 = 3 pts), hard_8 = 0.72 (hard_drop = 0.08 = 8 pts, gap = 5 pts)
    oracle = np.zeros(n_questions, dtype=np.float64)
    oracle[:400] = 1.0  # 400 / 500 = 0.80

    random_8 = oracle.copy()
    # Drop 15 questions from 1 -> 0 so mean drops by 15/500 = 0.03 (3.0 pts)
    random_8[:15] = 0.0

    hard_8 = random_8.copy()
    # Drop an additional 25 questions from 1 -> 0 so mean drops by another 25/500 = 0.05 (gap = 5.0 pts, hard_drop = 8.0 pts)
    hard_8[15:40] = 0.0

    # Shuffle question order deterministically
    perm = rng.permutation(n_questions)
    oracle, hard_8, random_8 = oracle[perm], hard_8[perm], random_8[perm]

    res = paired_bootstrap(oracle, hard_8, random_8, n_resamples=10000, ci_level=0.95, seed=4207)

    assert math.isclose(res["hard_drop"]["estimate_pts"], 8.0, abs_tol=1e-9)
    assert math.isclose(res["random_drop"]["estimate_pts"], 3.0, abs_tol=1e-9)
    assert math.isclose(res["gap"]["estimate_pts"], 5.0, abs_tol=1e-9)

    assert res["hard_drop"]["ci_low_pts"] > 5.0
    assert res["gap"]["ci_low_pts"] > 3.0
    assert res["hard_drop"]["ci_low_pts"] < 8.0 < res["hard_drop"]["ci_high_pts"]
    assert res["gap"]["ci_low_pts"] < 5.0 < res["gap"]["ci_high_pts"]


def test_synthetic_null_gives_about_5_percent_false_positives() -> None:
    """Under the null (true drop = 0, true gap = 0), 95% CI excludes 0 ~5% of the time across repeated draws."""
    rng = np.random.default_rng(98765)
    n_trials = 300
    n_questions = 250
    n_boot = 1500

    fp_hard = 0
    fp_gap = 0

    for trial in range(n_trials):
        # Null DGP: oracle, hard, random have identical mean (0.75) with zero mean paired differences
        base = rng.binomial(1, 0.75, size=n_questions).astype(np.float64)
        noise_hard = rng.choice([-1.0 / 3.0, 0.0, 1.0 / 3.0], size=n_questions, p=[0.2, 0.6, 0.2])
        noise_rand = rng.choice([-1.0 / 3.0, 0.0, 1.0 / 3.0], size=n_questions, p=[0.2, 0.6, 0.2])

        oracle = base
        hard_n = base - noise_hard
        random_n = base - noise_rand

        res = paired_bootstrap(
            oracle, hard_n, random_n, n_resamples=n_boot, ci_level=0.95, seed=10000 + trial
        )
        # Two-sided 95% CI excludes 0 when ci_low > 0 or ci_high < 0
        if res["hard_drop"]["ci_low"] > 0.0 or res["hard_drop"]["ci_high"] < 0.0:
            fp_hard += 1
        if res["gap"]["ci_low"] > 0.0 or res["gap"]["ci_high"] < 0.0:
            fp_gap += 1

    fp_rate_hard = fp_hard / n_trials
    fp_rate_gap = fp_gap / n_trials

    # With 300 trials at alpha=0.05, SE = sqrt(0.05 * 0.95 / 300) ~= 0.0126; [0.02, 0.085] is well within +/-3 SE
    assert 0.02 <= fp_rate_hard <= 0.085, f"Null false positive rate for hard_drop was {fp_rate_hard:.3f}"
    assert 0.02 <= fp_rate_gap <= 0.085, f"Null false positive rate for gap was {fp_rate_gap:.3f}"


def test_synthetic_ci_coverage_near_95_percent() -> None:
    """Over repeated draws from a known DGP with true hard_drop=0.06 and gap=0.04, 95% CI covers truth ~95% of the time."""
    rng = np.random.default_rng(54321)
    n_trials = 300
    n_questions = 250
    n_boot = 1500

    true_hard_drop = 0.06
    true_random_drop = 0.02
    true_gap = true_hard_drop - true_random_drop  # 0.04

    covered_hard = 0
    covered_gap = 0

    for trial in range(n_trials):
        oracle = np.full(n_questions, 0.80, dtype=np.float64)
        # Paired differences with known population expectation
        d_hard = rng.normal(loc=true_hard_drop, scale=0.25, size=n_questions)
        d_rand = rng.normal(loc=true_random_drop, scale=0.25, size=n_questions)

        hard_n = oracle - d_hard
        random_n = oracle - d_rand

        res = paired_bootstrap(
            oracle, hard_n, random_n, n_resamples=n_boot, ci_level=0.95, seed=50000 + trial
        )
        if res["hard_drop"]["ci_low"] <= true_hard_drop <= res["hard_drop"]["ci_high"]:
            covered_hard += 1
        if res["gap"]["ci_low"] <= true_gap <= res["gap"]["ci_high"]:
            covered_gap += 1

    cov_hard = covered_hard / n_trials
    cov_gap = covered_gap / n_trials

    assert 0.91 <= cov_hard <= 0.985, f"95% CI coverage for hard_drop was {cov_hard:.3f}"
    assert 0.91 <= cov_gap <= 0.985, f"95% CI coverage for gap was {cov_gap:.3f}"


# ---------------------------------------------------------------------------
# 4. H1 Evaluation and Decision-Table Classes (i), (ii), (iii), (iv)
# ---------------------------------------------------------------------------


def test_evaluate_h1_reads_primary_n_from_prereg_yaml(tmp_path: Path) -> None:
    """Verify evaluate_h1 reads primary_comparison.n dynamically from config rather than hardcoding 8."""
    real_prereg = load_prereg(DEFAULT_PREREG_PATH)
    assert real_prereg["primary_comparison"]["n"] == 8

    # Create a temporary prereg YAML with primary_comparison.n = 4
    custom_prereg = dict(real_prereg)
    custom_prereg["primary_comparison"] = dict(real_prereg["primary_comparison"], n=4)
    custom_prereg_path = tmp_path / "prereg_n4.yaml"
    custom_prereg_path.write_text(yaml.safe_dump(custom_prereg), encoding="utf-8")

    n_q = 200
    oracle = np.full(n_q, 0.80)
    hard_4 = np.full(n_q, 0.70)   # 10 pt drop
    rand_4 = np.full(n_q, 0.78)   # 2 pt drop -> 8 pt gap

    # Providing only n=4 in conditions succeeds when prereg has n=4, and fails when prereg has n=8
    conditions_only_n4 = {
        "oracle": oracle,
        "hard_negative": {4: hard_4},
        "random_length_matched": {4: rand_4},
    }
    res_n4 = evaluate_h1(conditions=conditions_only_n4, prereg_path=custom_prereg_path, n_resamples=500)
    assert res_n4["primary_n"] == 4
    assert res_n4["h1_met"] is True
    assert res_n4["decision_class"] == "(i)"

    with pytest.raises(KeyError, match="primary_comparison.n=8"):
        evaluate_h1(conditions=conditions_only_n4, prereg_path=DEFAULT_PREREG_PATH, n_resamples=500)


def test_decision_table_all_four_classes_triggered_by_constructed_cases() -> None:
    """Construct synthetic cases that unambiguously trigger each decision class (i), (ii), (iii), (iv)."""
    rng = np.random.default_rng(4207)
    n_q = 500
    oracle = np.full(n_q, 0.80, dtype=np.float64)
    jitter = rng.normal(0.0, 0.05, size=n_q)
    jitter -= np.mean(jitter)  # exact zero sample mean jitter

    # Class (i): Both criteria met (hard_drop = 8 pts >= 5 with CI > 0, gap = 6 pts >= 3 with CI > 0)
    hard_i = oracle - (0.08 + jitter)
    rand_i = oracle - (0.02 + jitter * 0.5)
    out_i = evaluate_h1(oracle, hard_i, rand_i, n_resamples=2000, seed=4207)
    assert out_i["h1_met"] is True
    assert out_i["decision_class"] == "(i)"
    assert out_i["matched_classes"] == {"(i)": True, "(ii)": False, "(iii)": False, "(iv)": False}

    # Class (ii): Both drops >= 5 pts (hard_drop = 7 pts, random_drop = 6.8 pts) and gap CI includes 0 (gap = 0.2 pts)
    hard_ii = oracle - (0.07 + jitter)
    rand_ii = oracle - (0.068 - jitter)
    out_ii = evaluate_h1(oracle, hard_ii, rand_ii, n_resamples=2000, seed=4207)
    assert out_ii["h1_met"] is False
    assert out_ii["decision_class"] == "(ii)"
    assert out_ii["matched_classes"] == {"(i)": False, "(ii)": True, "(iii)": False, "(iv)": False}

    # Class (iii): Hard drop < 2 pts or its CI includes 0 (hard_drop = 1.0 pt, random_drop = 0.5 pts)
    hard_iii = oracle - (0.01 + jitter)
    rand_iii = oracle - (0.005 + jitter * 0.5)
    out_iii = evaluate_h1(oracle, hard_iii, rand_iii, n_resamples=2000, seed=4207)
    assert out_iii["h1_met"] is False
    assert out_iii["decision_class"] == "(iii)"
    assert out_iii["matched_classes"] == {"(i)": False, "(ii)": False, "(iii)": True, "(iv)": False}

    # Class (iv): Random-drop CI entirely below 0 (random passages raise accuracy by 5 pts; hard_drop = 3 pts)
    hard_iv = oracle - (0.03 + jitter * 0.2)
    rand_iv = oracle - (-0.05 + jitter * 0.2)  # random_drop = -5 pts, entire 95% CI < 0
    out_iv = evaluate_h1(oracle, hard_iv, rand_iv, n_resamples=2000, seed=4207)
    assert out_iv["h1_met"] is False
    assert out_iv["decision_class"] == "(iv)"
    assert out_iv["matched_classes"] == {"(i)": False, "(ii)": False, "(iii)": False, "(iv)": True}


# ---------------------------------------------------------------------------
# 6. Power Helper and Extend-to-800 Contingency Rule
# ---------------------------------------------------------------------------


def test_power_helper_low_vs_high_variance_and_extend_to_800_rule() -> None:
    """Verify estimate_power computes power at N=500 and N=800 and triggers extend_to_800 iff gap power < 0.80."""
    rng = np.random.default_rng(4207)
    n_pilot = 50

    # 1. Low-variance pilot (SD of paired gap difference ~ 0.15 -> SE at N=500 is ~0.0067 -> high power > 0.95)
    pilot_oracle_low = np.full(n_pilot, 0.80)
    pilot_hard_low = pilot_oracle_low - rng.normal(0.05, 0.15, size=n_pilot)
    pilot_rand_low = pilot_oracle_low - rng.normal(0.02, 0.15, size=n_pilot)

    res_low = estimate_power(
        pilot_oracle_low,
        pilot_hard_low,
        pilot_rand_low,
        n_simulations=400,
        n_bootstrap=800,
        seed=4207,
    )
    assert res_low["n_pilot_questions"] == 50
    assert res_low["power_hard_drop_500"] > 0.90
    assert res_low["power_gap_500"] >= 0.80
    assert res_low["extend_to_800"] is False
    assert res_low["recommended_n"] == 500

    # 2. High-variance pilot (SD of paired gap difference ~ 0.45 -> SE at N=500 is ~0.020 -> gap power for 3 pts < 0.80)
    pilot_oracle_high = np.full(n_pilot, 0.80)
    pilot_hard_high = pilot_oracle_high - rng.normal(0.05, 0.35, size=n_pilot)
    pilot_rand_high = pilot_oracle_high - rng.normal(0.02, 0.35, size=n_pilot)

    res_high = estimate_power(
        pilot_oracle_high,
        pilot_hard_high,
        pilot_rand_high,
        n_simulations=400,
        n_bootstrap=800,
        seed=4207,
    )
    assert res_high["power_gap_500"] < 0.80
    assert res_high["power_gap_800"] > res_high["power_gap_500"]
    assert res_high["power_hard_drop_800"] >= res_high["power_hard_drop_500"]
    assert res_high["extend_to_800"] is True
    assert res_high["recommended_n"] == 800
