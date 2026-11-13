"""Statistical functions for the DSAI4207 RAG irrelevant-context study (P5a + P5b).

P5a core:
- Per-question accuracy = mean over available seeds; unit of resampling = question.
- Paired bootstrap (10,000 resamples, fixed seed) for hard drop, random drop, and gap
  using the exact same question resample indices for all three quantities.
- H1 evaluation reading configs/prereg.yaml dynamically (using primary_comparison.n)
  and classifying outcomes into decision-table classes (i)-(v) with zero real-line gaps.
- Simulation-based power helper for the 5-pt drop and 3-pt gap at N=500 and N=800
  with the extend-to-800 rule (if estimated power < 0.80 for the gap).

P5b exploratory analysis:
- Holm-Bonferroni step-down correction (monotonic adjusted p-values, stopping rule,
  and Holm-adjusted percentile CIs at level 1 - alpha / (m - k + 1)).
- Paired scale (Qwen3-4B vs Qwen3-8B) and family (Qwen3-8B vs Llama-3.1-8B-Instruct)
  contrasts at n=8 using 10,000 shared bootstrap resamples.
- Closed-book correctness stratification (closed_book_correct vs closed_book_wrong)
  with within-stratum CIs and independent two-sample bootstrap between-stratum diffs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
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


def _mean_available_rows(rows: Sequence[Sequence[float]], labels: Sequence[Any] | None = None) -> np.ndarray:
    """Compute per-row mean ignoring NaNs for ragged or NaN-padded seed lists."""
    means: list[float] = []
    for idx, row in enumerate(rows):
        arr_row = np.asarray(row, dtype=np.float64)
        if arr_row.ndim != 1 or arr_row.size == 0:
            label = labels[idx] if labels is not None else idx
            raise ValueError(f"Question {label!r} has 0 seed scores.")
        valid = arr_row[~np.isnan(arr_row)]
        if valid.size == 0:
            label = labels[idx] if labels is not None else idx
            raise ValueError(f"Question {label!r} has no non-NaN seed scores.")
        means.append(float(np.mean(valid, dtype=np.float64)))
    return np.asarray(means, dtype=np.float64)


def per_question_accuracy(
    scores_by_seed: np.ndarray | Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    expected_seeds: int | None = 3,
) -> np.ndarray:
    """Compute per-question accuracy as the mean over available seeds (Item 2 / P5b).

    Parameters
    ----------
    scores_by_seed:
        Either a 2D array-like of shape (n_questions, n_seeds), a ragged sequence of
        per-question seed score sequences (when `expected_seeds is None`), or a mapping
        from question_id to a sequence of per-seed scores.
    expected_seeds:
        If not None, validates that each question has exactly `expected_seeds` seed scores.
        If None, averages over all available (non-NaN) seeds per question.

    Returns
    -------
    np.ndarray
        1D float64 array of shape (n_questions,) containing per-question mean accuracy.
    """
    if isinstance(scores_by_seed, Mapping):
        if len(scores_by_seed) == 0:
            raise ValueError("scores_by_seed mapping must not be empty.")
        qids = list(scores_by_seed.keys())
        rows = [list(scores_by_seed[qid]) for qid in qids]
        if expected_seeds is not None:
            for qid, row in zip(qids, rows):
                if len(row) != expected_seeds:
                    raise ValueError(
                        f"Question {qid!r} has {len(row)} seeds; expected {expected_seeds}."
                    )
            arr = np.asarray(rows, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"scores_by_seed must be 2D of shape (n_questions, n_seeds), got shape {arr.shape}."
                )
            return np.mean(arr, axis=1, dtype=np.float64)
        return _mean_available_rows(rows, labels=qids)

    try:
        arr = np.asarray(scores_by_seed, dtype=np.float64)
    except ValueError:
        if expected_seeds is None and isinstance(scores_by_seed, Sequence):
            if len(scores_by_seed) == 0:
                raise ValueError("scores_by_seed must contain at least one question.")
            return _mean_available_rows(scores_by_seed)
        raise

    if arr.ndim != 2:
        raise ValueError(
            f"scores_by_seed must be 2D of shape (n_questions, n_seeds), got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError("scores_by_seed must contain at least one question.")
    if expected_seeds is not None:
        if arr.shape[1] != expected_seeds:
            raise ValueError(
                f"Expected {expected_seeds} seeds per question (axis 1), got {arr.shape[1]}."
            )
        return np.mean(arr, axis=1, dtype=np.float64)

    if arr.shape[1] == 0:
        raise ValueError("scores_by_seed must have at least one seed column.")
    if np.any(np.isnan(arr)):
        if np.any(np.all(np.isnan(arr), axis=1)):
            raise ValueError("Each question must have at least one non-NaN seed score.")
        return np.nanmean(arr, axis=1, dtype=np.float64)
    return np.mean(arr, axis=1, dtype=np.float64)


def _as_question_vector(
    values: np.ndarray | Sequence[Any] | Mapping[str, Any],
    name: str,
    expected_seeds: int | None = None,
) -> np.ndarray:
    """Coerce condition input to a 1D per-question float64 vector.

    If a 2D (n_questions, n_seeds) matrix or ragged per-question mapping/sequence is
    provided, averages across available seeds per question so the unit of resampling is
    always the question.
    """
    if isinstance(values, Mapping):
        if len(values) == 0:
            raise ValueError(f"{name} must not be empty.")
        has_seq = any(
            isinstance(v, (Sequence, np.ndarray)) and not isinstance(v, (str, bytes))
            for v in values.values()
        )
        if has_seq:
            return per_question_accuracy(values, expected_seeds=expected_seeds)
        arr = np.asarray(list(values.values()), dtype=np.float64)
    else:
        try:
            arr = np.asarray(values, dtype=np.float64)
        except ValueError:
            if expected_seeds is None and isinstance(values, Sequence):
                return per_question_accuracy(values, expected_seeds=None)
            raise

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


def _lookup_distractor_raw(
    conditions: Mapping[Any, Any],
    dtype: str,
    short_prefix: str,
    n: int,
) -> Any:
    """Look up raw distractor condition data for `(dtype, n)` across supported key styles."""
    if dtype in conditions and isinstance(conditions[dtype], Mapping):
        sub = conditions[dtype]
        if n in sub:
            return sub[n]
        if str(n) in sub:
            return sub[str(n)]
    if short_prefix in conditions and isinstance(conditions[short_prefix], Mapping):
        sub = conditions[short_prefix]
        if n in sub:
            return sub[n]
        if str(n) in sub:
            return sub[str(n)]
    for key in (
        ("rag", dtype, n),
        (dtype, n),
        (short_prefix, n),
        f"{dtype}_{n}",
        f"{short_prefix}_{n}",
        f"{short_prefix}_n{n}",
    ):
        if key in conditions:
            return conditions[key]
    raise KeyError(
        f"Could not find condition for distractor_type={dtype!r} at primary_comparison.n={n}."
    )


def _lookup_baseline_raw(conditions: Mapping[Any, Any], cond_name: str) -> Any:
    """Look up raw baseline condition (`oracle` or `closed_book`) across supported key styles."""
    for key in (cond_name, (cond_name, None, 0), (cond_name, 0)):
        if key in conditions:
            return conditions[key]
    raise KeyError(f"Baseline condition {cond_name!r} not found in conditions.")


def _extract_primary_raw_conditions(
    conditions: Mapping[Any, Any],
    primary_n: int,
    baseline_condition: str,
    distractor_types: Sequence[str],
) -> tuple[Any, Any, Any]:
    """Extract raw `(oracle, hard_n, random_n)` entries from a conditions mapping."""
    oracle_raw = _lookup_baseline_raw(conditions, baseline_condition)
    hard_type = distractor_types[0] if len(distractor_types) > 0 else "hard_negative"
    random_type = distractor_types[1] if len(distractor_types) > 1 else "random_length_matched"
    hard_raw = _lookup_distractor_raw(conditions, hard_type, "hard", primary_n)
    random_raw = _lookup_distractor_raw(conditions, random_type, "random", primary_n)
    return oracle_raw, hard_raw, random_raw


def _extract_primary_condition_vectors(
    conditions: Mapping[Any, Any],
    primary_n: int,
    baseline_condition: str,
    distractor_types: Sequence[str],
    expected_seeds: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (oracle, hard_n, random_n) vectors for `primary_n` from a conditions mapping."""
    hard_type = distractor_types[0] if len(distractor_types) > 0 else "hard_negative"
    random_type = distractor_types[1] if len(distractor_types) > 1 else "random_length_matched"
    oracle_raw, hard_raw, random_raw = _extract_primary_raw_conditions(
        conditions, primary_n, baseline_condition, distractor_types
    )
    return (
        _as_question_vector(oracle_raw, baseline_condition, expected_seeds=expected_seeds),
        _as_question_vector(hard_raw, f"{hard_type}_{primary_n}", expected_seeds=expected_seeds),
        _as_question_vector(random_raw, f"{random_type}_{primary_n}", expected_seeds=expected_seeds),
    )


_DECISION_CLASS_METADATA: dict[str, tuple[str, str]] = {
    "(i)": (
        "class_i",
        "H1 supported; hard drop >= 5 pts and gap >= 3 pts with 95% CIs excluding 0",
    ),
    "(ii)": (
        "class_ii",
        "H1 not supported; volume/length effect (both drops >= 5 pts and gap CI includes 0)",
    ),
    "(iii)": (
        "class_iii",
        "H1 not supported; negligible or non-significant hard-negative drop (< 2 pts or CI includes 0)",
    ),
    "(iv)": (
        "class_iv",
        "H1 not supported; random passages improve accuracy (random-drop CI entirely below 0)",
    ),
    "(v)": (
        "class_v",
        "H1 not supported; effect detectable but below practical threshold",
    ),
}


def _extract_metric_pts(entry: Mapping[str, Any], key: str) -> float:
    """Read a summary statistic in percentage points from `*_pts` or proportion key."""
    pts_key = f"{key}_pts"
    if pts_key in entry:
        return float(entry[pts_key])
    return 100.0 * float(entry[key])


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
    """Evaluate pre-registered hypothesis H1 and decision-table class (Item 4 + P5b).

    Reads `configs/prereg.yaml` dynamically:
    - Uses `primary_comparison.n` (never hardcoded).
    - Reads `thresholds.hard_drop_min_pts` (5), `thresholds.gap_min_pts` (3),
      `thresholds.ci_level` (0.95), `thresholds.ci_must_exclude_zero` (True).
    - H1 criteria met iff:
        hard_drop >= hard_drop_min_pts AND hard_drop CI lower bound > 0
        AND gap >= gap_min_pts AND gap CI lower bound > 0.
    - Partitions the entire real line into decision classes (i)-(v):
        "(i)"   : both H1 criteria met;
        "(ii)"  : both drops >= 5 pts and the gap CI includes 0;
        "(iv)"  : random-drop CI entirely below 0;
        "(iii)" : hard drop < 2 pts or hard-drop CI lower bound <= 0;
        "(v)"   : residual non-H1 class when none of (i)-(iv) match (hard_drop >= 2 pts
                  with CI lower bound > 0, either 2 <= hard_drop < 5 or hard_drop >= 5
                  with gap criterion / class (ii) unmet).
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

    hard_drop_pts = _extract_metric_pts(boot["hard_drop"], "estimate")
    hard_ci_low_pts = _extract_metric_pts(boot["hard_drop"], "ci_low")
    hard_ci_high_pts = _extract_metric_pts(boot["hard_drop"], "ci_high")

    random_drop_pts = _extract_metric_pts(boot["random_drop"], "estimate")
    random_ci_low_pts = _extract_metric_pts(boot["random_drop"], "ci_low")
    random_ci_high_pts = _extract_metric_pts(boot["random_drop"], "ci_high")

    gap_pts = _extract_metric_pts(boot["gap"], "estimate")
    gap_ci_low_pts = _extract_metric_pts(boot["gap"], "ci_low")
    gap_ci_high_pts = _extract_metric_pts(boot["gap"], "ci_high")

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

    # Decision-table predicates (complete partition of R with zero gaps)
    class_i = h1_met
    class_ii = bool(
        (hard_drop_pts >= hard_drop_min_pts)
        and (random_drop_pts >= hard_drop_min_pts)
        and (gap_ci_low_pts <= 0.0 <= gap_ci_high_pts)
    )
    class_iii = bool(
        (hard_drop_pts < robust_drop_max_pts)
        or (hard_ci_low_pts <= 0.0)
    )
    class_iv = bool(random_ci_high_pts < 0.0)
    class_v = bool(
        (not class_i)
        and (not class_ii)
        and (not class_iii)
        and (not class_iv)
        and (hard_drop_pts >= robust_drop_max_pts)
        and (hard_ci_low_pts > 0.0)
    )

    matched_classes = {
        "(i)": class_i,
        "(ii)": class_ii,
        "(iii)": class_iii,
        "(iv)": class_iv,
        "(v)": class_v,
    }

    # Primary decision class assignment:
    # (i) both criteria met;
    # (ii) both drops >= 5 and gap CI includes 0;
    # (iv) random-drop CI entirely below 0 (checked before (iii) so random-boost cases
    #      with small hard drop are not masked by (iii));
    # (iii) hard drop < 2 pts or hard-drop CI lower bound <= 0;
    # (v) residual non-H1 class (hard_drop >= 2 pts with CI > 0, none of (i)-(iv) matched).
    if class_i:
        decision_class = "(i)"
    elif class_ii:
        decision_class = "(ii)"
    elif class_iv:
        decision_class = "(iv)"
    elif class_iii:
        decision_class = "(iii)"
    else:
        decision_class = "(v)"

    decision_label, decision_description = _DECISION_CLASS_METADATA[decision_class]
    sub_reason: str | None = None
    if decision_class == "(v)":
        sub_reason = (
            "subthreshold_hard_drop_2_to_5"
            if hard_drop_pts < hard_drop_min_pts
            else "hard_drop_met_but_gap_criterion_unmet"
        )

    return {
        "hypothesis": str(prereg.get("hypothesis", "H1")),
        "primary_model": str(prereg.get("primary_model", "Qwen3-8B")),
        "primary_n": primary_n,
        "h1_met": h1_met,
        "hard_criterion_met": hard_criterion_met,
        "gap_criterion_met": gap_criterion_met,
        "decision_class": decision_class,
        "decision_label": decision_label,
        "decision_description": decision_description,
        "sub_reason": sub_reason,
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


# ---------------------------------------------------------------------------
# P5b: Bootstrap P-Values, Holm Step-Down Correction, and Exploratory Contrasts
# ---------------------------------------------------------------------------


def bootstrap_two_sided_p_value(replicates: np.ndarray | Sequence[float]) -> float:
    """Compute unsmoothed two-sided percentile tail-inversion bootstrap p-value.

    Formula:
        p = min(1.0, 2.0 * min(mean(d_boot <= 0.0), mean(d_boot >= 0.0)))
    Returns 1.0 when all replicates are identically 0.0.
    """
    arr = np.asarray(replicates, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"replicates must be a non-empty 1D array, got shape {arr.shape}.")
    if np.any(np.isnan(arr)):
        raise ValueError("replicates must not contain NaN values.")
    if np.all(arr == 0.0):
        return 1.0
    p_le = float(np.mean(arr <= 0.0))
    p_ge = float(np.mean(arr >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _strip_replicates_inplace(obj: Any) -> None:
    """Recursively remove heavy `replicates` / `resample_indices` arrays when not requested."""
    if isinstance(obj, dict):
        obj.pop("replicates", None)
        obj.pop("resample_indices", None)
        for val in obj.values():
            _strip_replicates_inplace(val)
    elif isinstance(obj, list):
        for item in obj:
            _strip_replicates_inplace(item)


def holm_bonferroni(
    p_values_or_contrasts: Sequence[float | Mapping[str, Any]] | Mapping[str, float | Mapping[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply Holm-Bonferroni step-down multiple-testing correction (P5b).

    Parameters
    ----------
    p_values_or_contrasts:
        Sequence or mapping of either raw two-sided p-values or contrast dicts
        (each having `"p_value"` and/or bootstrap `"replicates"` / `"bootstrap_diffs"`).
    alpha:
        Family-wise error rate level in (0, 1) (default 0.05).

    Returns
    -------
    dict[str, Any]
        Contains `alpha`, `m`, `n_significant`, `first_non_significant_rank`,
        `adjusted_p_values`, `significant`, `holm_alphas`, `holm_ci_levels`,
        `sorted_order`, `results` (in original order), and `results_by_name`.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    alpha_f = float(alpha)

    if isinstance(p_values_or_contrasts, Mapping):
        items: list[tuple[str, Any]] = [(str(k), v) for k, v in p_values_or_contrasts.items()]
    elif isinstance(p_values_or_contrasts, Sequence) and not isinstance(
        p_values_or_contrasts, (str, bytes)
    ):
        items = []
        for idx, val in enumerate(p_values_or_contrasts):
            if isinstance(val, Mapping):
                name = str(val.get("name") or val.get("contrast") or val.get("label") or f"contrast_{idx}")
            else:
                name = f"contrast_{idx}"
            items.append((name, val))
    else:
        raise TypeError(
            "p_values_or_contrasts must be a Sequence or Mapping of p-values or contrast dicts."
        )

    m = len(items)
    if m == 0:
        raise ValueError("p_values_or_contrasts must not be empty.")

    names: list[str] = []
    raw_p: list[float] = []
    replicates_list: list[np.ndarray | None] = []
    metadata_list: list[dict[str, Any]] = []

    for name, item in items:
        names.append(name)
        if isinstance(item, Mapping):
            reps_raw = item.get("replicates")
            if reps_raw is None:
                reps_raw = item.get("bootstrap_diffs")
            reps_arr: np.ndarray | None = None
            if reps_raw is not None:
                reps_arr = np.asarray(reps_raw, dtype=np.float64)
                if reps_arr.ndim != 1 or reps_arr.size == 0:
                    raise ValueError(
                        f"Contrast {name!r} replicates must be a non-empty 1D array, got {reps_arr.shape}."
                    )
            if "p_value" in item and item["p_value"] is not None:
                p_val = float(item["p_value"])
            elif reps_arr is not None:
                p_val = bootstrap_two_sided_p_value(reps_arr)
            else:
                raise ValueError(
                    f"Contrast {name!r} must provide 'p_value' and/or 'replicates'."
                )
            meta = {
                k: v
                for k, v in item.items()
                if k not in ("replicates", "bootstrap_diffs", "p_value", "name")
            }
        elif isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool):
            p_val = float(item)
            reps_arr = None
            meta = {}
        else:
            raise TypeError(
                f"Unsupported item type for {name!r}: expected float or contrast Mapping, got {type(item).__name__}."
            )

        if math.isnan(p_val) or not (0.0 <= p_val <= 1.0):
            raise ValueError(f"p-value for {name!r} must be in [0, 1], got {p_val}.")

        raw_p.append(p_val)
        replicates_list.append(reps_arr)
        metadata_list.append(meta)

    # Sort in ascending raw p-value order (stable on original index)
    order = sorted(range(m), key=lambda idx: (raw_p[idx], idx))

    ranks = [0] * m
    adj_p_values = [0.0] * m
    significant_flags = [False] * m
    holm_alphas = [0.0] * m
    holm_ci_levels = [0.0] * m
    holm_ci_lows: list[float | None] = [None] * m
    holm_ci_highs: list[float | None] = [None] * m

    running_max = 0.0
    stopped = False
    first_non_sig_rank: int | None = None

    for k_zero, orig_idx in enumerate(order):
        k = k_zero + 1
        mult = m - k + 1
        step_alpha = alpha_f / float(mult)
        ci_level_k = 1.0 - step_alpha
        p_val = raw_p[orig_idx]

        adj_p = max(running_max, min(1.0, float(mult) * p_val))
        running_max = adj_p

        if (not stopped) and (p_val <= step_alpha):
            is_sig = True
        else:
            if not stopped:
                first_non_sig_rank = k
            stopped = True
            is_sig = False

        ranks[orig_idx] = k
        adj_p_values[orig_idx] = float(adj_p)
        significant_flags[orig_idx] = bool(is_sig)
        holm_alphas[orig_idx] = float(step_alpha)
        holm_ci_levels[orig_idx] = float(ci_level_k)

        reps = replicates_list[orig_idx]
        if reps is not None:
            low_pct = 100.0 * (step_alpha / 2.0)
            high_pct = 100.0 * (1.0 - step_alpha / 2.0)
            c_low, c_high = np.percentile(reps, [low_pct, high_pct])
            holm_ci_lows[orig_idx] = float(c_low)
            holm_ci_highs[orig_idx] = float(c_high)

    results: list[dict[str, Any]] = []
    results_by_name: dict[str, dict[str, Any]] = {}
    for idx in range(m):
        entry: dict[str, Any] = {
            **metadata_list[idx],
            "name": names[idx],
            "index": idx,
            "rank": ranks[idx],
            "p_value": raw_p[idx],
            "p_value_adj": adj_p_values[idx],
            "holm_alpha": holm_alphas[idx],
            "holm_ci_level": holm_ci_levels[idx],
            "significant": significant_flags[idx],
        }
        if holm_ci_lows[idx] is not None and holm_ci_highs[idx] is not None:
            c_lo = float(holm_ci_lows[idx])
            c_hi = float(holm_ci_highs[idx])
            entry["holm_ci_low"] = c_lo
            entry["holm_ci_high"] = c_hi
            entry["holm_ci_low_pts"] = c_lo * 100.0
            entry["holm_ci_high_pts"] = c_hi * 100.0
        results.append(entry)
        results_by_name[names[idx]] = entry

    return {
        "alpha": alpha_f,
        "m": m,
        "n_significant": int(sum(significant_flags)),
        "first_non_significant_rank": first_non_sig_rank,
        "adjusted_p_values": adj_p_values,
        "significant": significant_flags,
        "holm_alphas": holm_alphas,
        "holm_ci_levels": holm_ci_levels,
        "sorted_order": [names[i] for i in order],
        "results": results,
        "results_by_name": results_by_name,
    }


def _align_named_raw_inputs(
    named_raws: Mapping[str, Any],
) -> tuple[list[str] | None, dict[str, np.ndarray]]:
    """Align multiple condition inputs either by common `qid` keys or equal vector length."""
    if not named_raws:
        raise ValueError("named_raws must not be empty.")

    keys = list(named_raws.keys())
    are_mappings = [isinstance(named_raws[k], Mapping) for k in keys]
    if any(are_mappings) and not all(are_mappings):
        raise TypeError(
            "Cannot mix question-ID Mapping inputs with sequence/array inputs in the same contrast."
        )

    if all(are_mappings):
        mappings: dict[str, Mapping[str, Any]] = {
            k: named_raws[k] for k in keys  # type: ignore[assignment]
        }
        first_map = mappings[keys[0]]
        if not first_map:
            raise ValueError(f"Condition {keys[0]!r} mapping must not be empty.")
        common_qids = [
            str(qid)
            for qid in first_map.keys()
            if all(qid in mappings[k] for k in keys[1:])
        ]
        if not common_qids:
            raise ValueError("No common question IDs found across the provided conditions.")
        aligned_vecs = {
            k: _as_question_vector(
                {qid: mappings[k][qid] for qid in common_qids},
                k,
                expected_seeds=None,
            )
            for k in keys
        }
        return common_qids, aligned_vecs

    aligned_vecs = {
        k: _as_question_vector(named_raws[k], k, expected_seeds=None) for k in keys
    }
    lengths = {k: v.shape[0] for k, v in aligned_vecs.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All inputs must have identical question counts, got {lengths}.")
    return None, aligned_vecs


def _summarize_resampled_diff(
    diff_values: np.ndarray,
    resample_indices: np.ndarray,
    ci_level: float,
    *,
    name: str,
    **metadata: Any,
) -> dict[str, Any]:
    """Compute point estimate, percentile CI, and two-sided bootstrap p-value from resample indices."""
    point_est = float(np.mean(diff_values))
    replicates = np.mean(diff_values[resample_indices], axis=1)
    summary = _summarize_contrast(
        point_est,
        replicates,
        ci_level,
        return_replicates=True,
    )
    summary["name"] = name
    summary["p_value"] = bootstrap_two_sided_p_value(replicates)
    summary.update(metadata)
    return summary


def compute_scale_and_family_contrasts(
    model_conditions: Mapping[str, Mapping[Any, Any]],
    *,
    primary_n: int | None = None,
    prereg_path: Path | str = DEFAULT_PREREG_PATH,
    n_resamples: int | None = None,
    ci_level: float | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
    scale_models: tuple[str, str] = ("Qwen3-4B", "Qwen3-8B"),
    family_models: tuple[str, str] = ("Qwen3-8B", "Llama-3.1-8B-Instruct"),
    return_replicates: bool = False,
) -> dict[str, Any]:
    """Compute paired scale and family contrasts at `primary_n` using shared bootstrap resamples.

    Sign conventions:
    - Scale contrast: `Qwen3-4B - Qwen3-8B` (positive => 4B has larger drop/gap).
    - Family contrast: `Qwen3-8B - Llama-3.1-8B-Instruct` (positive => Qwen3-8B has larger drop/gap).
    """
    prereg = load_prereg(prereg_path)
    target_n = int(primary_n) if primary_n is not None else int(prereg["primary_comparison"]["n"])
    baseline_cond = str(prereg["primary_comparison"]["baseline_condition"])
    distractor_types = list(prereg["primary_comparison"]["distractor_types"])
    boot_resamples = (
        int(n_resamples)
        if n_resamples is not None
        else int(prereg["statistics"]["bootstrap_resamples"])
    )
    boot_ci_level = float(ci_level) if ci_level is not None else float(prereg["thresholds"]["ci_level"])
    boot_seed = int(seed) if seed is not None else int(prereg["rng"]["master_seed"])

    if boot_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {boot_resamples}.")
    if not (0.0 < boot_ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {boot_ci_level}.")

    required_models: list[str] = []
    for m in (*scale_models, *family_models):
        if m not in model_conditions:
            raise KeyError(f"Model {m!r} missing from model_conditions.")
        if m not in required_models:
            required_models.append(m)

    raw_slots: dict[str, Any] = {}
    for m in required_models:
        o_raw, h_raw, r_raw = _extract_primary_raw_conditions(
            model_conditions[m], target_n, baseline_cond, distractor_types
        )
        raw_slots[f"{m}::oracle"] = o_raw
        raw_slots[f"{m}::hard"] = h_raw
        raw_slots[f"{m}::random"] = r_raw

    common_qids, aligned = _align_named_raw_inputs(raw_slots)
    n_questions = next(iter(aligned.values())).shape[0]

    rng = np.random.default_rng(boot_seed)
    resample_indices = rng.integers(0, n_questions, size=(boot_resamples, n_questions))

    per_model_diffs: dict[str, dict[str, np.ndarray]] = {}
    per_model_summaries: dict[str, dict[str, Any]] = {}
    for m in required_models:
        o_vec = aligned[f"{m}::oracle"]
        h_vec = aligned[f"{m}::hard"]
        r_vec = aligned[f"{m}::random"]
        d_hard = o_vec - h_vec
        d_rand = o_vec - r_vec
        d_gap = r_vec - h_vec
        per_model_diffs[m] = {
            "hard_drop": d_hard,
            "random_drop": d_rand,
            "gap": d_gap,
        }
        per_model_summaries[m] = {
            "oracle_acc": float(np.mean(o_vec)),
            "hard_acc": float(np.mean(h_vec)),
            "random_acc": float(np.mean(r_vec)),
            "hard_drop": _summarize_resampled_diff(
                d_hard, resample_indices, boot_ci_level, name=f"{m}_hard_drop", model=m, metric="hard_drop"
            ),
            "random_drop": _summarize_resampled_diff(
                d_rand, resample_indices, boot_ci_level, name=f"{m}_random_drop", model=m, metric="random_drop"
            ),
            "gap": _summarize_resampled_diff(
                d_gap, resample_indices, boot_ci_level, name=f"{m}_gap", model=m, metric="gap"
            ),
        }

    contrasts: dict[str, dict[str, Any]] = {}
    for comp_label, (model_a, model_b) in (
        ("scale", scale_models),
        ("family", family_models),
    ):
        direction = f"{model_a} - {model_b}"
        for metric in ("gap", "hard_drop", "random_drop"):
            c_name = f"{comp_label}_{metric}"
            diff_vec = per_model_diffs[model_a][metric] - per_model_diffs[model_b][metric]
            contrasts[c_name] = _summarize_resampled_diff(
                diff_vec,
                resample_indices,
                boot_ci_level,
                name=c_name,
                comparison=comp_label,
                metric=metric,
                model_a=model_a,
                model_b=model_b,
                direction=direction,
            )

    holm_gap = holm_bonferroni(
        {"scale_gap": contrasts["scale_gap"], "family_gap": contrasts["family_gap"]},
        alpha=alpha,
    )
    holm_all = holm_bonferroni(contrasts, alpha=alpha)

    result: dict[str, Any] = {
        "n": target_n,
        "n_questions": int(n_questions),
        "qids": common_qids,
        "n_resamples": boot_resamples,
        "ci_level": boot_ci_level,
        "seed": boot_seed,
        "alpha": float(alpha),
        "per_model": per_model_summaries,
        "contrasts": contrasts,
        "holm_gap": holm_gap,
        "holm_all": holm_all,
    }
    if return_replicates:
        result["resample_indices"] = resample_indices
    else:
        _strip_replicates_inplace(result)
    return result


def compute_closed_book_stratified_contrasts(
    closed_book: np.ndarray | Sequence[Any] | Mapping[str, Any] | None = None,
    oracle: np.ndarray | Sequence[Any] | Mapping[str, Any] | None = None,
    hard_n: np.ndarray | Sequence[Any] | Mapping[str, Any] | None = None,
    random_n: np.ndarray | Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    conditions: Mapping[Any, Any] | None = None,
    model_name: str = "Qwen3-8B",
    primary_n: int | None = None,
    prereg_path: Path | str = DEFAULT_PREREG_PATH,
    n_resamples: int | None = None,
    ci_level: float | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
    return_replicates: bool = False,
) -> dict[str, Any]:
    """Stratify questions by closed-book EM (correct=1 vs wrong=0) and compare strata (P5b).

    Within each stratum, computes `oracle`, `hard n`, `random n` accuracy and `hard_drop`,
    `random_drop`, `gap` with bootstrap CIs using independent RNG streams per stratum.
    Between-stratum differences (`closed_book_wrong - closed_book_correct`) use an
    independent two-sample bootstrap over the two strata, with Holm correction.
    """
    prereg = load_prereg(prereg_path)
    target_n = int(primary_n) if primary_n is not None else int(prereg["primary_comparison"]["n"])
    baseline_cond = str(prereg["primary_comparison"]["baseline_condition"])
    distractor_types = list(prereg["primary_comparison"]["distractor_types"])
    boot_resamples = (
        int(n_resamples)
        if n_resamples is not None
        else int(prereg["statistics"]["bootstrap_resamples"])
    )
    boot_ci_level = float(ci_level) if ci_level is not None else float(prereg["thresholds"]["ci_level"])
    boot_seed = int(seed) if seed is not None else int(prereg["rng"]["master_seed"])

    if boot_resamples <= 0:
        raise ValueError(f"n_resamples must be positive, got {boot_resamples}.")
    if not (0.0 < boot_ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {boot_ci_level}.")

    if conditions is not None:
        cb_raw = _lookup_baseline_raw(conditions, "closed_book")
        o_raw, h_raw, r_raw = _extract_primary_raw_conditions(
            conditions, target_n, baseline_cond, distractor_types
        )
    elif (
        closed_book is not None
        and oracle is not None
        and hard_n is not None
        and random_n is not None
    ):
        cb_raw, o_raw, h_raw, r_raw = closed_book, oracle, hard_n, random_n
    else:
        raise ValueError(
            "Must provide either (closed_book, oracle, hard_n, random_n) or `conditions`."
        )

    common_qids, aligned = _align_named_raw_inputs(
        {
            "closed_book": cb_raw,
            "oracle": o_raw,
            "hard_n": h_raw,
            "random_n": r_raw,
        }
    )
    cb_vec = aligned["closed_book"]
    o_vec = aligned["oracle"]
    h_vec = aligned["hard_n"]
    r_vec = aligned["random_n"]

    if not np.all(np.isclose(cb_vec, 0.0) | np.isclose(cb_vec, 1.0)):
        raise ValueError(
            "closed_book values must be binary 0 or 1 EM outcomes for stratification."
        )

    correct_mask = cb_vec >= 0.5
    wrong_mask = ~correct_mask
    n_correct = int(np.sum(correct_mask))
    n_wrong = int(np.sum(wrong_mask))
    if n_correct == 0 or n_wrong == 0:
        raise ValueError(
            f"Both closed-book strata must be non-empty for {model_name!r}: "
            f"closed_book_correct={n_correct}, closed_book_wrong={n_wrong}."
        )

    # Independent RNG streams for the two independent strata
    ss_correct, ss_wrong = np.random.SeedSequence(boot_seed).spawn(2)
    stratum_rngs = {
        "closed_book_correct": np.random.default_rng(ss_correct),
        "closed_book_wrong": np.random.default_rng(ss_wrong),
    }
    stratum_masks = {
        "closed_book_correct": correct_mask,
        "closed_book_wrong": wrong_mask,
    }

    strata_results: dict[str, dict[str, Any]] = {}
    for s_name in ("closed_book_correct", "closed_book_wrong"):
        mask = stratum_masks[s_name]
        n_s = int(np.sum(mask))
        o_s = o_vec[mask]
        h_s = h_vec[mask]
        r_s = r_vec[mask]

        resample_idx_s = stratum_rngs[s_name].integers(0, n_s, size=(boot_resamples, n_s))
        d_hard_s = o_s - h_s
        d_rand_s = o_s - r_s
        d_gap_s = r_s - h_s

        stratum_dict: dict[str, Any] = {
            "n_questions": n_s,
            "qids": [q for q, flag in zip(common_qids, mask) if flag] if common_qids is not None else None,
            "oracle_acc": float(np.mean(o_s)),
            "hard_acc": float(np.mean(h_s)),
            "random_acc": float(np.mean(r_s)),
            "oracle": _summarize_resampled_diff(
                o_s, resample_idx_s, boot_ci_level, name=f"{model_name}_{s_name}_oracle"
            ),
            "hard_n": _summarize_resampled_diff(
                h_s, resample_idx_s, boot_ci_level, name=f"{model_name}_{s_name}_hard_n"
            ),
            "random_n": _summarize_resampled_diff(
                r_s, resample_idx_s, boot_ci_level, name=f"{model_name}_{s_name}_random_n"
            ),
            "hard_drop": _summarize_resampled_diff(
                d_hard_s, resample_idx_s, boot_ci_level, name=f"{model_name}_{s_name}_hard_drop"
            ),
            "random_drop": _summarize_resampled_diff(
                d_rand_s, resample_idx_s, boot_ci_level, name=f"{model_name}_{s_name}_random_drop"
            ),
            "gap": _summarize_resampled_diff(
                d_gap_s, resample_idx_s, boot_ci_level, name=f"{model_name}_{s_name}_gap"
            ),
        }
        if return_replicates:
            stratum_dict["resample_indices"] = resample_idx_s
        strata_results[s_name] = stratum_dict

    between_strata: dict[str, dict[str, Any]] = {}
    direction = "closed_book_wrong - closed_book_correct"
    for metric in ("gap", "hard_drop", "random_drop"):
        wrong_entry = strata_results["closed_book_wrong"][metric]
        correct_entry = strata_results["closed_book_correct"][metric]
        point_diff = float(wrong_entry["estimate"]) - float(correct_entry["estimate"])
        rep_diff = wrong_entry["replicates"] - correct_entry["replicates"]

        summary = _summarize_contrast(
            point_diff,
            rep_diff,
            boot_ci_level,
            return_replicates=True,
        )
        summary["name"] = f"{model_name}_stratified_{metric}"
        summary["model"] = model_name
        summary["metric"] = metric
        summary["direction"] = direction
        summary["p_value"] = bootstrap_two_sided_p_value(rep_diff)
        between_strata[metric] = summary

    holm_primary = holm_bonferroni(
        {"gap": between_strata["gap"], "hard_drop": between_strata["hard_drop"]},
        alpha=alpha,
    )
    holm_all = holm_bonferroni(between_strata, alpha=alpha)

    result: dict[str, Any] = {
        "model": model_name,
        "n": target_n,
        "n_questions": int(cb_vec.shape[0]),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "qids": common_qids,
        "n_resamples": boot_resamples,
        "ci_level": boot_ci_level,
        "seed": boot_seed,
        "alpha": float(alpha),
        "strata": strata_results,
        "between_strata": between_strata,
        "holm_primary": holm_primary,
        "holm_all": holm_all,
    }
    if not return_replicates:
        _strip_replicates_inplace(result)
    return result


def compute_all_exploratory_contrasts(
    model_conditions: Mapping[str, Mapping[Any, Any]],
    *,
    primary_n: int | None = None,
    prereg_path: Path | str = DEFAULT_PREREG_PATH,
    n_resamples: int | None = None,
    ci_level: float | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
    scale_models: tuple[str, str] = ("Qwen3-4B", "Qwen3-8B"),
    family_models: tuple[str, str] = ("Qwen3-8B", "Llama-3.1-8B-Instruct"),
    return_replicates: bool = False,
) -> dict[str, Any]:
    """Run all P5b exploratory contrasts (scale, family, per-model closed-book stratification)
    and apply Holm correction across both the core gap family and the full exploratory family.
    """
    scale_and_family = compute_scale_and_family_contrasts(
        model_conditions,
        primary_n=primary_n,
        prereg_path=prereg_path,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        alpha=alpha,
        scale_models=scale_models,
        family_models=family_models,
        return_replicates=True,
    )

    stratified_by_model: dict[str, dict[str, Any]] = {}
    for m, conds in model_conditions.items():
        stratified_by_model[m] = compute_closed_book_stratified_contrasts(
            conditions=conds,
            model_name=m,
            primary_n=primary_n,
            prereg_path=prereg_path,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed,
            alpha=alpha,
            return_replicates=True,
        )

    core_gap_family: dict[str, Mapping[str, Any]] = {
        "scale_gap": scale_and_family["contrasts"]["scale_gap"],
        "family_gap": scale_and_family["contrasts"]["family_gap"],
    }
    for m, strat in stratified_by_model.items():
        core_gap_family[f"{m}_stratified_gap"] = strat["between_strata"]["gap"]

    full_family: dict[str, Mapping[str, Any]] = dict(scale_and_family["contrasts"])
    for m, strat in stratified_by_model.items():
        for metric in ("gap", "hard_drop", "random_drop"):
            full_family[f"{m}_stratified_{metric}"] = strat["between_strata"][metric]

    holm_core_gap_family = holm_bonferroni(core_gap_family, alpha=alpha)
    holm_full_family = holm_bonferroni(full_family, alpha=alpha)

    result: dict[str, Any] = {
        "alpha": float(alpha),
        "scale_and_family": scale_and_family,
        "stratified_by_model": stratified_by_model,
        "holm_core_gap_family": holm_core_gap_family,
        "holm_full_family": holm_full_family,
    }
    if not return_replicates:
        _strip_replicates_inplace(result)
    return result


from src.analysis_loader import (  # noqa: E402, F401
    align_conditions_by_qid,
    load_and_score_results,
    load_gold_answers,
    summarize_context_tokens,
)
