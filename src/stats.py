"""Core statistical functions for the DSAI4207 RAG irrelevant-context study (P5a).

Implements ONLY Items 2, 3, 4, and 6:
- Item 2: Per-question accuracy = mean over the 3 seeds; unit of resampling = question.
- Item 3: Paired bootstrap (10,000 resamples, fixed seed) for hard drop, random drop,
  and gap using the exact same question resample indices for all three quantities.
- Item 4: H1 evaluation reading configs/prereg.yaml dynamically (using primary_comparison.n)
  and classifying outcomes into decision-table classes (i), (ii), (iii), (iv).
- Item 6: Simulation-based power helper for the 5-pt drop and 3-pt gap at N=500 and N=800
  with the extend-to-800 rule (if estimated power < 0.80 for the gap).

Note: Exploratory contrasts and Holm step-down correction are explicitly out of scope for P5a
and belong to P5b.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREG_PATH = REPO_ROOT / "configs" / "prereg.yaml"


def load_prereg(prereg_path: Path | str = DEFAULT_PREREG_PATH) -> dict[str, Any]:
    """Load pre-registration configuration from YAML."""
    path = Path(prereg_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def per_question_accuracy(
    scores_by_seed: np.ndarray | Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    expected_seeds: int | None = 3,
) -> np.ndarray:
    """Compute per-question accuracy as the mean over seeds (Item 2).

    Parameters
    ----------
    scores_by_seed:
        Either a 2D array-like of shape (n_questions, n_seeds) or a mapping from
        question_id to a sequence of per-seed scores.
    expected_seeds:
        If not None, validates that each question has exactly `expected_seeds` seed scores.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (n_questions,) containing per-question mean accuracy.
    """
    if isinstance(scores_by_seed, Mapping):
        if len(scores_by_seed) == 0:
            raise ValueError("scores_by_seed mapping must not be empty.")
        rows = [list(scores_by_seed[qid]) for qid in scores_by_seed]
        if expected_seeds is not None:
            for qid, row in zip(scores_by_seed.keys(), rows):
                if len(row) != expected_seeds:
                    raise ValueError(
                        f"Question {qid!r} has {len(row)} seeds; expected {expected_seeds}."
                    )
        arr = np.asarray(rows, dtype=np.float64)
    else:
        arr = np.asarray(scores_by_seed, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(
            f"scores_by_seed must be 2D of shape (n_questions, n_seeds), got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError("scores_by_seed must contain at least one question.")
    if expected_seeds is not None and arr.shape[1] != expected_seeds:
        raise ValueError(
            f"Expected {expected_seeds} seeds per question (axis 1), got {arr.shape[1]}."
        )

    return np.mean(arr, axis=1, dtype=np.float64)


def _as_question_vector(
    values: np.ndarray | Sequence[Any] | Mapping[str, Any],
    name: str,
    expected_seeds: int | None = None,
) -> np.ndarray:
    """Coerce condition input to a 1D per-question float64 vector.

    If a 2D (n_questions, n_seeds) matrix is provided, averages across seeds per question
    so that the unit of resampling is always the question.
    """
    if isinstance(values, Mapping):
        first_val = next(iter(values.values()), None)
        if isinstance(first_val, (Sequence, np.ndarray)) and not isinstance(first_val, str):
            return per_question_accuracy(values, expected_seeds=expected_seeds)
        arr = np.asarray(list(values.values()), dtype=np.float64)
    else:
        arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 2:
        return per_question_accuracy(arr, expected_seeds=expected_seeds)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (n_questions,) or 2D (n_questions, n_seeds), got {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _summarize_contrast(
    point_estimate: float,
    replicates: np.ndarray,
    ci_level: float,
    return_replicates: bool = False,
) -> dict[str, Any]:
    """Compute percentile CI summary in both proportion [0, 1] and percentage-point [0, 100] units."""
    alpha = 1.0 - ci_level
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(replicates, [low_pct, high_pct])
    ci_low_f = float(ci_low)
    ci_high_f = float(ci_high)
    est_f = float(point_estimate)

    summary: dict[str, Any] = {
        "estimate": est_f,
        "ci_low": ci_low_f,
        "ci_high": ci_high_f,
        "estimate_pts": est_f * 100.0,
        "ci_low_pts": ci_low_f * 100.0,
        "ci_high_pts": ci_high_f * 100.0,
    }
    if return_replicates:
        summary["replicates"] = replicates
    return summary


def paired_bootstrap(
    oracle: np.ndarray | Sequence[Any],
    hard_n: np.ndarray | Sequence[Any],
    random_n: np.ndarray | Sequence[Any],
    *,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 4207,
    return_replicates: bool = False,
) -> dict[str, Any]:
    """Paired bootstrap over questions for hard drop, random drop, and gap (Item 3).

    Uses the EXACT SAME question resample indices across all three quantities:
      (a) hard_drop   = acc(oracle)   - acc(hard_n)
      (b) random_drop = acc(oracle)   - acc(random_n)
      (c) gap         = acc(random_n) - acc(hard_n)

    Parameters
    ----------
    oracle, hard_n, random_n:
        Per-question accuracy arrays (1D of length N, or 2D of shape (N, n_seeds) which
        will be averaged over seeds first) aligned by question.
    n_resamples:
        Number of bootstrap resamples (default 10,000 per prereg.yaml).
    ci_level:
        Confidence level for percentile intervals (default 0.95).
    seed:
        Fixed RNG seed for deterministic resampling (default 4207 per prereg.yaml).
    return_replicates:
        If True, includes raw bootstrap replicate arrays and resample indices in the output.
    """
    oracle_vec = _as_question_vector(oracle, "oracle")
    hard_vec = _as_question_vector(hard_n, "hard_n")
    random_vec = _as_question_vector(random_n, "random_n")

    n_questions = oracle_vec.shape[0]
    if hard_vec.shape[0] != n_questions or random_vec.shape[0] != n_questions:
        raise ValueError(
            f"All conditions must have the same number of questions: "
            f"oracle={n_questions}, hard_n={hard_vec.shape[0]}, random_n={random_vec.shape[0]}."
        )
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")

    # Per-question paired differences
    diff_hard = oracle_vec - hard_vec
    diff_random = oracle_vec - random_vec
    diff_gap = random_vec - hard_vec

    # Point estimates over questions
    hard_drop_est = float(np.mean(diff_hard))
    random_drop_est = float(np.mean(diff_random))
    gap_est = float(np.mean(diff_gap))

    # Draw question resample indices ONCE and apply to all three quantities
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n_questions, size=(n_resamples, n_questions))

    boot_hard_drop = np.mean(diff_hard[resample_indices], axis=1)
    boot_random_drop = np.mean(diff_random[resample_indices], axis=1)
    boot_gap = np.mean(diff_gap[resample_indices], axis=1)

    result: dict[str, Any] = {
        "unit": "question",
        "n_questions": int(n_questions),
        "n_resamples": int(n_resamples),
        "ci_level": float(ci_level),
        "seed": int(seed),
        "oracle_acc": float(np.mean(oracle_vec)),
        "hard_acc": float(np.mean(hard_vec)),
        "random_acc": float(np.mean(random_vec)),
        "hard_drop": _summarize_contrast(
            hard_drop_est, boot_hard_drop, ci_level, return_replicates=return_replicates
        ),
        "random_drop": _summarize_contrast(
            random_drop_est, boot_random_drop, ci_level, return_replicates=return_replicates
        ),
        "gap": _summarize_contrast(
            gap_est, boot_gap, ci_level, return_replicates=return_replicates
        ),
    }
    if return_replicates:
        result["resample_indices"] = resample_indices
    return result


def _extract_primary_condition_vectors(
    conditions: Mapping[Any, Any],
    primary_n: int,
    baseline_condition: str,
    distractor_types: Sequence[str],
    expected_seeds: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (oracle, hard_n, random_n) vectors for `primary_n` from a conditions mapping."""
    if baseline_condition not in conditions:
        raise KeyError(f"Baseline condition {baseline_condition!r} not found in conditions.")
    oracle_raw = conditions[baseline_condition]

    hard_type = distractor_types[0] if len(distractor_types) > 0 else "hard_negative"
    random_type = distractor_types[1] if len(distractor_types) > 1 else "random_length_matched"

    def _lookup_distractor(dtype: str, short_prefix: str) -> Any:
        # 1. Nested mapping: conditions[dtype][primary_n] or conditions[dtype][str(primary_n)]
        if dtype in conditions and isinstance(conditions[dtype], Mapping):
            sub = conditions[dtype]
            if primary_n in sub:
                return sub[primary_n]
            if str(primary_n) in sub:
                return sub[str(primary_n)]
        if short_prefix in conditions and isinstance(conditions[short_prefix], Mapping):
            sub = conditions[short_prefix]
            if primary_n in sub:
                return sub[primary_n]
            if str(primary_n) in sub:
                return sub[str(primary_n)]
        # 2. Tuple key: (dtype, primary_n) or (short_prefix, primary_n)
        for key in (
            (dtype, primary_n),
            (short_prefix, primary_n),
            f"{dtype}_{primary_n}",
            f"{short_prefix}_{primary_n}",
            f"{short_prefix}_n{primary_n}",
        ):
            if key in conditions:
                return conditions[key]
        raise KeyError(
            f"Could not find condition for distractor_type={dtype!r} at primary_comparison.n={primary_n}."
        )

    hard_raw = _lookup_distractor(hard_type, "hard")
    random_raw = _lookup_distractor(random_type, "random")

    return (
        _as_question_vector(oracle_raw, baseline_condition, expected_seeds=expected_seeds),
        _as_question_vector(hard_raw, f"{hard_type}_{primary_n}", expected_seeds=expected_seeds),
        _as_question_vector(random_raw, f"{random_type}_{primary_n}", expected_seeds=expected_seeds),
    )


def evaluate_h1(
    oracle: np.ndarray | Sequence[Any] | None = None,
    hard_n: np.ndarray | Sequence[Any] | None = None,
    random_n: np.ndarray | Sequence[Any] | None = None,
    *,
    conditions: Mapping[Any, Any] | None = None,
    bootstrap_result: Mapping[str, Any] | None = None,
    prereg_path: Path | str = DEFAULT_PREREG_PATH,
    seed: int | None = None,
    n_resamples: int | None = None,
) -> dict[str, Any]:
    """Evaluate pre-registered hypothesis H1 and decision-table class (Item 4).

    Reads `configs/prereg.yaml` dynamically:
    - Uses `primary_comparison.n` (never hardcoded).
    - Reads `thresholds.hard_drop_min_pts` (5), `thresholds.gap_min_pts` (3),
      `thresholds.ci_level` (0.95), `thresholds.ci_must_exclude_zero` (True).
    - H1 criteria met iff:
        hard_drop >= hard_drop_min_pts AND hard_drop CI lower bound > 0
        AND gap >= gap_min_pts AND gap CI lower bound > 0.
    - Classifies into the decision-table class:
        "(i)"   : both H1 criteria met;
        "(ii)"  : both drops >= 5 pts and the gap CI includes 0;
        "(iii)" : hard drop < 2 pts or its CI includes 0;
        "(iv)"  : random-drop CI entirely below 0.
    """
    prereg = load_prereg(prereg_path)
    primary_n = int(prereg["primary_comparison"]["n"])
    baseline_cond = str(prereg["primary_comparison"]["baseline_condition"])
    distractor_types = list(prereg["primary_comparison"]["distractor_types"])

    thresholds = prereg["thresholds"]
    hard_drop_min_pts = float(thresholds["hard_drop_min_pts"])
    gap_min_pts = float(thresholds["gap_min_pts"])
    ci_level = float(thresholds["ci_level"])
    ci_must_exclude_zero = bool(thresholds["ci_must_exclude_zero"])
    robust_drop_max_pts = 2.0

    boot_resamples = (
        int(n_resamples)
        if n_resamples is not None
        else int(prereg["statistics"]["bootstrap_resamples"])
    )
    boot_seed = int(seed) if seed is not None else int(prereg["rng"]["master_seed"])

    if bootstrap_result is not None:
        boot = dict(bootstrap_result)
    else:
        if conditions is not None:
            oracle_vec, hard_vec, random_vec = _extract_primary_condition_vectors(
                conditions=conditions,
                primary_n=primary_n,
                baseline_condition=baseline_cond,
                distractor_types=distractor_types,
            )
        elif oracle is not None and hard_n is not None and random_n is not None:
            oracle_vec = _as_question_vector(oracle, "oracle")
            hard_vec = _as_question_vector(hard_n, f"hard_{primary_n}")
            random_vec = _as_question_vector(random_n, f"random_{primary_n}")
        else:
            raise ValueError(
                "Must provide either (oracle, hard_n, random_n), `conditions`, or `bootstrap_result`."
            )

        boot = paired_bootstrap(
            oracle_vec,
            hard_vec,
            random_vec,
            n_resamples=boot_resamples,
            ci_level=ci_level,
            seed=boot_seed,
        )

    hard_drop_pts = float(boot["hard_drop"]["estimate_pts"])
    hard_ci_low_pts = float(boot["hard_drop"]["ci_low_pts"])
    hard_ci_high_pts = float(boot["hard_drop"]["ci_high_pts"])

    random_drop_pts = float(boot["random_drop"]["estimate_pts"])
    random_ci_low_pts = float(boot["random_drop"]["ci_low_pts"])
    random_ci_high_pts = float(boot["random_drop"]["ci_high_pts"])

    gap_pts = float(boot["gap"]["estimate_pts"])
    gap_ci_low_pts = float(boot["gap"]["ci_low_pts"])
    gap_ci_high_pts = float(boot["gap"]["ci_high_pts"])

    # H1 sub-criteria
    hard_drop_meets_min = hard_drop_pts >= hard_drop_min_pts
    hard_drop_excludes_zero = hard_ci_low_pts > 0.0
    hard_criterion_met = bool(
        hard_drop_meets_min and (hard_drop_excludes_zero if ci_must_exclude_zero else True)
    )

    gap_meets_min = gap_pts >= gap_min_pts
    gap_excludes_zero = gap_ci_low_pts > 0.0
    gap_criterion_met = bool(
        gap_meets_min and (gap_excludes_zero if ci_must_exclude_zero else True)
    )

    h1_met = bool(hard_criterion_met and gap_criterion_met)

    # Decision-table predicates
    class_i = h1_met
    class_ii = bool(
        (hard_drop_pts >= hard_drop_min_pts)
        and (random_drop_pts >= hard_drop_min_pts)
        and (gap_ci_low_pts <= 0.0 <= gap_ci_high_pts)
    )
    class_iii = bool(
        (hard_drop_pts < robust_drop_max_pts)
        or (hard_ci_low_pts <= 0.0 <= hard_ci_high_pts)
    )
    class_iv = bool(random_ci_high_pts < 0.0)

    matched_classes = {
        "(i)": class_i,
        "(ii)": class_ii,
        "(iii)": class_iii,
        "(iv)": class_iv,
    }

    # Primary decision class assignment:
    # (i) both criteria met;
    # (ii) both drops >= 5 and gap CI includes 0;
    # (iv) random-drop CI entirely below 0 (checked before (iii) so random-boost cases
    #      with small hard drop are not masked by (iii));
    # (iii) hard drop < 2 pts or its CI includes 0.
    if class_i:
        decision_class = "(i)"
    elif class_ii:
        decision_class = "(ii)"
    elif class_iv:
        decision_class = "(iv)"
    elif class_iii:
        decision_class = "(iii)"
    else:
        decision_class = "unclassified"

    return {
        "hypothesis": str(prereg.get("hypothesis", "H1")),
        "primary_model": str(prereg.get("primary_model", "Qwen3-8B")),
        "primary_n": primary_n,
        "h1_met": h1_met,
        "hard_criterion_met": hard_criterion_met,
        "gap_criterion_met": gap_criterion_met,
        "decision_class": decision_class,
        "matched_classes": matched_classes,
        "thresholds": {
            "hard_drop_min_pts": hard_drop_min_pts,
            "gap_min_pts": gap_min_pts,
            "robust_drop_max_pts": robust_drop_max_pts,
            "ci_level": ci_level,
            "ci_must_exclude_zero": ci_must_exclude_zero,
        },
        "bootstrap": boot,
    }


def estimate_power(
    pilot_oracle: np.ndarray | Sequence[Any],
    pilot_hard: np.ndarray | Sequence[Any],
    pilot_random: np.ndarray | Sequence[Any],
    *,
    prereg_path: Path | str = DEFAULT_PREREG_PATH,
    n_main: int | None = None,
    n_extended: int | None = None,
    n_simulations: int = 1000,
    n_bootstrap: int = 2000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Estimate statistical power for the 5-pt drop and 3-pt gap at N=500 and N=800 (Item 6).

    Given pilot per-question outcomes:
    1. Computes the empirical per-question paired differences on the pilot set:
       `d_hard = pilot_oracle - pilot_hard` and `d_gap = pilot_random - pilot_hard`.
    2. Centers the pilot paired differences and shifts them to the pre-registered target
       effect sizes (`hard_drop_min_pts` = 5 pts = 0.05, `gap_min_pts` = 3 pts = 0.03),
       preserving the pilot's empirical paired variance and higher moments.
    3. For each sample size N in {n_main (500), n_extended (800)}, simulates `n_simulations`
       synthetic experiments of N questions drawn with replacement from the shifted pilot pool,
       computes paired bootstrap 95% CIs (`n_bootstrap` resamples), and estimates power as the
       fraction of simulations where the 95% CI lower bound exceeds 0 (`ci_low > 0`).
    4. Applies the pre-registered `extend_to_800` contingency rule:
       `extend_to_800 = (power_gap_500 < power_rule.if_power_below)` (0.80).
    """
    prereg = load_prereg(prereg_path)

    target_hard_drop = float(prereg["thresholds"]["hard_drop_min_pts"]) / 100.0
    target_gap = float(prereg["thresholds"]["gap_min_pts"]) / 100.0
    ci_level = float(prereg["thresholds"]["ci_level"])

    main_size = int(n_main) if n_main is not None else int(prereg["design"]["n_questions"])
    extended_size = (
        int(n_extended)
        if n_extended is not None
        else int(prereg["power_rule"]["extend_primary_comparison_to"])
    )
    power_threshold = float(prereg["power_rule"]["if_power_below"])
    sim_seed = int(seed) if seed is not None else int(prereg["rng"]["master_seed"])

    oracle_vec = _as_question_vector(pilot_oracle, "pilot_oracle")
    hard_vec = _as_question_vector(pilot_hard, "pilot_hard")
    random_vec = _as_question_vector(pilot_random, "pilot_random")

    n_pilot = oracle_vec.shape[0]
    if hard_vec.shape[0] != n_pilot or random_vec.shape[0] != n_pilot:
        raise ValueError(
            f"Pilot arrays must have identical question counts: "
            f"oracle={n_pilot}, hard={hard_vec.shape[0]}, random={random_vec.shape[0]}."
        )
    if n_pilot < 2:
        raise ValueError(f"Pilot must contain at least 2 questions, got {n_pilot}.")
    if n_simulations <= 0 or n_bootstrap <= 0:
        raise ValueError("n_simulations and n_bootstrap must be positive.")

    pilot_diff_hard = oracle_vec - hard_vec
    pilot_diff_gap = random_vec - hard_vec

    # Shift centered pilot differences to target effect sizes (5-pt drop and 3-pt gap)
    shifted_hard = (pilot_diff_hard - np.mean(pilot_diff_hard)) + target_hard_drop
    shifted_gap = (pilot_diff_gap - np.mean(pilot_diff_gap)) + target_gap

    alpha = 1.0 - ci_level
    low_pct = 100.0 * (alpha / 2.0)

    rng = np.random.default_rng(sim_seed)

    def _simulate_for_n(sample_size: int) -> dict[str, float]:
        # Draw (n_simulations, sample_size) question indices from the pilot pool
        sim_indices = rng.integers(0, n_pilot, size=(n_simulations, sample_size))
        sim_hard = shifted_hard[sim_indices]
        sim_gap = shifted_gap[sim_indices]

        # Build bootstrap multinomial weight matrix W of shape (sample_size, n_bootstrap)
        # Each column b is a normalized count vector from resampling `sample_size` items
        boot_probs = np.full(sample_size, 1.0 / sample_size, dtype=np.float64)
        boot_counts = rng.multinomial(sample_size, boot_probs, size=n_bootstrap)
        weights = (boot_counts / float(sample_size)).T  # (sample_size, n_bootstrap)

        # Exact bootstrap replicate means for all simulations via matrix multiplication
        boot_means_hard = sim_hard @ weights  # (n_simulations, n_bootstrap)
        boot_means_gap = sim_gap @ weights    # (n_simulations, n_bootstrap)

        ci_low_hard = np.percentile(boot_means_hard, low_pct, axis=1)
        ci_low_gap = np.percentile(boot_means_gap, low_pct, axis=1)

        power_hard = float(np.mean(ci_low_hard > 0.0))
        power_gap = float(np.mean(ci_low_gap > 0.0))
        return {
            "n_questions": int(sample_size),
            "power_hard_drop": power_hard,
            "power_gap": power_gap,
        }

    res_main = _simulate_for_n(main_size)
    res_extended = _simulate_for_n(extended_size)

    extend_to_800 = bool(res_main["power_gap"] < power_threshold)
    recommended_n = extended_size if extend_to_800 else main_size

    return {
        "n_pilot_questions": int(n_pilot),
        "pilot_sd_hard_drop": float(np.std(pilot_diff_hard, ddof=1)),
        "pilot_sd_gap": float(np.std(pilot_diff_gap, ddof=1)),
        "target_hard_drop_pts": target_hard_drop * 100.0,
        "target_gap_pts": target_gap * 100.0,
        "power_threshold": power_threshold,
        "n_simulations": int(n_simulations),
        "n_bootstrap": int(n_bootstrap),
        "seed": sim_seed,
        f"N_{main_size}": res_main,
        f"N_{extended_size}": res_extended,
        "power_hard_drop_500": res_main["power_hard_drop"],
        "power_gap_500": res_main["power_gap"],
        "power_hard_drop_800": res_extended["power_hard_drop"],
        "power_gap_800": res_extended["power_gap"],
        "extend_to_800": extend_to_800,
        "recommended_n": int(recommended_n),
    }
