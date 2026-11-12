"""Tests for the pilot evaluation & acceptance script (P6 v2, `scripts/evaluate_pilot.py`).

All fixtures are synthetic and live under pytest's `tmp_path`; no models are loaded and nothing
is written to `results/`.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.evaluate_pilot as ep  # noqa: E402
from src.harness import compute_study_hashes  # noqa: E402
from src.stats import load_prereg, paired_bootstrap  # noqa: E402

PREREG = load_prereg(REPO_ROOT / "configs" / "prereg.yaml")
SPECS = ep.pilot_unit_specs(PREREG)
FAKE_HASHES = {"config_hash": "c" * 64, "prereg_hash": "p" * 64, "manifest_hash": "m" * 64}
N_Q = 20

# answer_fn(model, unit_key, idx) -> (extracted_answer, valid, seconds)
AnswerFn = Callable[[str, str, int], tuple[str, bool, float]]


def qid(i: int) -> str:
    return f"q{i:03d}"


def default_answer(model: str, unit: str, i: int) -> tuple[str, bool, float]:
    """Oracle 90 % correct, hard 70 %, random 85 %, closed_book 30 %; all valid; 1 s/gen."""
    correct_upto = {"oracle": 18, "hard": 14, "random": 17, "closed_book": 6}[unit]
    answer = f"answer {i}" if i < correct_upto else "wrong"
    return answer, True, 1.0


def build_dataset(
    root: Path,
    *,
    answer_fn: AnswerFn = default_answer,
    drop: set[tuple[str, str, str]] | None = None,
    hashes: dict[str, str] | None = None,
    mutate: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
    n_q: int = N_Q,
) -> tuple[Path, Path]:
    """Write synthetic pilot.json and 12 unit JSONL files. Returns (results_dir, sampling_file)."""
    drop = drop or set()
    hashes = hashes or FAKE_HASHES
    sampling = root / "data" / "sampling" / "pilot.json"
    sampling.parent.mkdir(parents=True, exist_ok=True)
    sampling.write_text(
        json.dumps(
            [
                {"question_id": qid(i), "gold_answers": [f"The Answer {i}.", f"answer {i}"]}
                for i in range(n_q)
            ]
        ),
        encoding="utf-8",
    )
    results_dir = root / "results"
    for model in ep.MODELS:
        for key, spec in SPECS.items():
            path = spec.run_unit(model).output_path(results_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for i in range(n_q):
                if (model, key, qid(i)) in drop:
                    continue
                ans, valid, secs = answer_fn(model, key, i)
                row: dict[str, Any] = {
                    "qid": qid(i),
                    "model": model,
                    "condition": spec.condition,
                    "distractor_type": spec.distractor_type,
                    "n": spec.n,
                    "seed": spec.seed,
                    "split": "pilot",
                    "backend": "mlx",
                    "is_stub": False,
                    "extracted_answer": ans,
                    "raw_output": ans,
                    "valid": valid,
                    "seconds": secs,
                    "tokens_per_second": 10.0 / secs,
                    **hashes,
                }
                if mutate is not None:
                    row = mutate(model, key, row)
                lines.append(json.dumps(row))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results_dir, sampling


def run_eval(root: Path, **kwargs: Any) -> dict[str, Any]:
    results_dir, sampling = build_dataset(root, **kwargs)
    return ep.evaluate_pilot(
        results_dir,
        sampling,
        repo_root=root,
        prereg_path=REPO_ROOT / "configs" / "prereg.yaml",
        expected_hashes=FAKE_HASHES,
        power_n_simulations=50,
        power_n_bootstrap=100,
    )


# ---------------------------------------------------------------------------
# (a) valid-answer rate and verdict rule
# ---------------------------------------------------------------------------


def test_valid_rates_per_unit_and_pooled(tmp_path: Path) -> None:
    # Qwen3-8B closed_book: 6/20 invalid (70 %); everything else valid.
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        ans, _, s = default_answer(model, unit, i)
        return ans, not (model == "Qwen3-8B" and unit == "closed_book" and i < 6), s

    r = run_eval(tmp_path, answer_fn=fn)["models"]["Qwen3-8B"]
    assert r["valid_units"]["closed_book"]["valid_pct"] == pytest.approx(70.0)
    assert r["valid_units"]["closed_book"]["pass"] is False
    assert r["valid_overall"]["n_valid"] == 74 and r["valid_overall"]["n_rows"] == 80
    assert r["valid_overall"]["valid_pct"] == pytest.approx(92.5)
    assert r["valid_context"]["valid_pct"] == pytest.approx(100.0)
    # Closed-book abstentions alone do not fail the model.
    assert r["verdict_pass"] is True


def test_valid_rate_boundary_85_exact_passes_below_fails(tmp_path: Path) -> None:
    # 17/20 = 85.0 % valid in oracle -> PASS; 16/20 = 80 % in hard -> FAIL.
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        ans, _, s = default_answer(model, unit, i)
        if unit == "oracle":
            return ans, i >= 3, s
        if unit == "hard" and model == "Qwen3-4B":
            return ans, i >= 4, s
        return ans, True, s

    models = run_eval(tmp_path, answer_fn=fn)["models"]
    assert models["Qwen3-8B"]["valid_units"]["oracle"]["valid_pct"] == pytest.approx(85.0)
    assert models["Qwen3-8B"]["valid_units"]["oracle"]["pass"] is True
    assert models["Qwen3-8B"]["verdict_pass"] is True
    q4 = models["Qwen3-4B"]
    assert q4["valid_units"]["hard"]["pass"] is False
    # Pooled overall (73/80) and context (53/60 = 88.3 %) pass, but a single context unit fails.
    assert q4["valid_overall"]["pass"] is True
    assert q4["valid_context"]["pass"] is True
    assert q4["verdict_checks"]["valid_each_context_unit"] is False
    assert q4["verdict_pass"] is False


def test_context_pooled_failure_fails_model(tmp_path: Path) -> None:
    # Llama: every context unit at 85 % (each passes) but closed_book 0 % valid ->
    # overall 63.75 % fails even though the context units pass.
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        ans, _, s = default_answer(model, unit, i)
        if model != "Llama-3.1-8B-Instruct":
            return ans, True, s
        return ans, (unit != "closed_book" and i >= 3), s

    r = run_eval(tmp_path, answer_fn=fn)["models"]["Llama-3.1-8B-Instruct"]
    assert r["verdict_checks"]["valid_each_context_unit"] is True
    assert r["verdict_checks"]["valid_context_pooled_3_units"] is True
    assert r["valid_overall"]["valid_pct"] == pytest.approx(63.75)
    assert r["verdict_checks"]["valid_overall_4_units"] is False
    assert r["verdict_pass"] is False


# ---------------------------------------------------------------------------
# (b) accuracy, (c) headroom
# ---------------------------------------------------------------------------


def test_em_f1_and_gold_normalization(tmp_path: Path) -> None:
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        if unit == "oracle":
            # "the answer i!" normalizes to "answer i" (EM 1); "answer" alone gives F1 2/3.
            return (f"the answer {i}!" if i < 10 else "answer"), True, 1.0
        return default_answer(model, unit, i)

    r = run_eval(tmp_path, answer_fn=fn)["models"]["Qwen3-8B"]
    assert r["acc_all"]["oracle"]["em_pct"] == pytest.approx(50.0)
    # F1: 10 * 1.0 + 10 * (2PR/(P+R) with P=1, R=1/2 -> 2/3)
    assert r["acc_all"]["oracle"]["f1_pct"] == pytest.approx(100.0 * (10 + 10 * 2 / 3) / 20)
    assert r["acc_all"]["closed_book"]["em_pct"] == pytest.approx(30.0)


def test_headroom_pass_and_fail(tmp_path: Path) -> None:
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        if unit == "oracle" and model == "Qwen3-4B":
            # 19/20 = 95 % correct (headroom 5 -> FAIL)
            return (f"answer {i}" if i < 19 else "wrong"), True, 1.0
        return default_answer(model, unit, i)

    models = run_eval(tmp_path, answer_fn=fn)["models"]
    # Default oracle 18/20 = 90 % -> headroom exactly 10.0 -> PASS.
    assert models["Qwen3-8B"]["headroom_pts"] == pytest.approx(10.0)
    assert models["Qwen3-8B"]["headroom_pass"] is True
    assert models["Qwen3-4B"]["headroom_pts"] == pytest.approx(5.0)
    assert models["Qwen3-4B"]["headroom_pass"] is False
    assert models["Qwen3-4B"]["verdict_pass"] is False


def test_paired_intersection_excludes_p2c_missing_rows(tmp_path: Path) -> None:
    # Remove q000 (correct everywhere) from the hard and random seed-0 units of every model.
    drop = {(m, k, qid(0)) for m in ep.MODELS for k in ("hard", "random")}
    r = run_eval(tmp_path, drop=drop)["models"]["Qwen3-8B"]
    assert r["acc_all"]["oracle"]["n"] == N_Q
    assert r["acc_all"]["hard"]["n"] == N_Q - 1
    assert len(r["paired_qids"]) == N_Q - 1
    assert r["acc_paired"]["oracle"]["n"] == N_Q - 1
    assert r["acc_paired"]["oracle"]["em_pct"] == pytest.approx(100.0 * 17 / 19)
    # Primary headroom uses all oracle rows; paired headroom is displayed alongside.
    assert r["headroom_pts"] == pytest.approx(10.0)
    assert r["headroom_paired_pts"] == pytest.approx(100.0 - 100.0 * 17 / 19)
    assert r["unit_files"]["hard"]["missing_qids"] == [qid(0)]
    assert r["unit_files"]["oracle"]["missing_qids"] == []


# ---------------------------------------------------------------------------
# (d) paired drops
# ---------------------------------------------------------------------------


def test_paired_drops_match_stats_paired_bootstrap(tmp_path: Path) -> None:
    r = run_eval(tmp_path)["models"]["Qwen3-8B"]
    oracle = np.array([1.0 if i < 18 else 0.0 for i in range(N_Q)])
    hard = np.array([1.0 if i < 14 else 0.0 for i in range(N_Q)])
    random = np.array([1.0 if i < 17 else 0.0 for i in range(N_Q)])
    expected = paired_bootstrap(
        oracle,
        hard,
        random,
        n_resamples=int(PREREG["statistics"]["bootstrap_resamples"]),
        ci_level=float(PREREG["thresholds"]["ci_level"]),
        seed=int(PREREG["rng"]["master_seed"]),
    )
    got = r["drops"]["em"]
    assert got["n_resamples"] == 10000
    for key in ("hard_drop", "random_drop", "gap"):
        for field in ("estimate_pts", "ci_low_pts", "ci_high_pts"):
            assert got[key][field] == pytest.approx(expected[key][field])
    assert got["hard_drop"]["estimate_pts"] == pytest.approx(20.0)
    assert got["random_drop"]["estimate_pts"] == pytest.approx(5.0)
    assert got["gap"]["estimate_pts"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# (e) throughput
# ---------------------------------------------------------------------------


def test_throughput_stats_and_tripwire(tmp_path: Path) -> None:
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        ans, valid, _ = default_answer(model, unit, i)
        if model == "Llama-3.1-8B-Instruct":
            return ans, valid, 4.0  # overall mean 4.0 > 3.0 -> tripped
        return ans, valid, float(i + 1)  # 1..20 s

    models = run_eval(tmp_path, answer_fn=fn)["models"]
    s = models["Qwen3-8B"]["throughput_units"]["oracle"]["seconds"]
    vals = np.arange(1, 21, dtype=float)
    assert s["mean"] == pytest.approx(10.5)
    assert s["median"] == pytest.approx(10.5)
    assert s["p90"] == pytest.approx(float(np.percentile(vals, 90)))
    assert models["Qwen3-8B"]["throughput_overall"]["n_rows"] == 80
    assert models["Qwen3-8B"]["tripwire_tripped"] is True  # mean 10.5 s
    assert models["Llama-3.1-8B-Instruct"]["tripwire_tripped"] is True
    tps = models["Llama-3.1-8B-Instruct"]["throughput_overall"]["tokens_per_second"]
    assert tps["mean"] == pytest.approx(2.5)


def test_tripwire_not_tripped_at_or_below_threshold(tmp_path: Path) -> None:
    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        ans, valid, _ = default_answer(model, unit, i)
        return ans, valid, ep.TRIPWIRE_SEC_PER_GEN

    models = run_eval(tmp_path, answer_fn=fn)["models"]
    assert all(r["tripwire_tripped"] is False for r in models.values())


# ---------------------------------------------------------------------------
# (f) power
# ---------------------------------------------------------------------------


def test_power_section_and_extend_rule(tmp_path: Path) -> None:
    ev = run_eval(tmp_path)
    for r in ev["models"].values():
        pw = r["power"]
        assert pw["n_pilot_questions"] == N_Q
        assert pw["n_simulations"] == 50 and pw["n_bootstrap"] == 100
        assert set(["power_hard_drop_500", "power_gap_500", "power_hard_drop_800", "power_gap_800"]) <= set(pw)
        assert pw["extend_to_800"] == (pw["power_gap_500"] < 0.80)
        assert pw["recommended_n"] == (800 if pw["extend_to_800"] else 500)


# ---------------------------------------------------------------------------
# Loud failures on untrustworthy inputs
# ---------------------------------------------------------------------------


def test_missing_unit_file_raises(tmp_path: Path) -> None:
    results_dir, sampling = build_dataset(tmp_path)
    SPECS["random"].run_unit("Qwen3-4B").output_path(results_dir).unlink()
    with pytest.raises(ep.PilotDataError, match="Missing pilot result file"):
        ep.evaluate_pilot(results_dir, sampling, repo_root=tmp_path, expected_hashes=FAKE_HASHES,
                          prereg_path=REPO_ROOT / "configs" / "prereg.yaml",
                          power_n_simulations=10, power_n_bootstrap=10)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda m, k, r: {**r, "is_stub": True, "backend": "stub"}, "stub"),
        (lambda m, k, r: {**r, "prereg_hash": "x" * 64}, "prereg_hash"),
        (lambda m, k, r: {**r, "qid": "q999"}, "not in pilot sampling file"),
        (lambda m, k, r: {**r, "seed": 1} if k == "hard" else r, "seed"),
        (lambda m, k, r: {**r, "model": "Other"} if k == "oracle" else r, "model"),
        (lambda m, k, r: {**r, "valid": "yes"}, "valid"),
    ],
)
def test_untrustworthy_rows_raise(tmp_path: Path, mutation: Any, match: str) -> None:
    with pytest.raises(ep.PilotDataError, match=match):
        run_eval(tmp_path, mutate=mutation)


def test_duplicate_qid_raises(tmp_path: Path) -> None:
    results_dir, sampling = build_dataset(tmp_path)
    path = SPECS["oracle"].run_unit("Qwen3-8B").output_path(results_dir)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    with path.open("a", encoding="utf-8") as f:
        f.write(first + "\n")
    with pytest.raises(ep.PilotDataError, match="duplicate qid"):
        ep.evaluate_pilot(results_dir, sampling, repo_root=tmp_path, expected_hashes=FAKE_HASHES,
                          prereg_path=REPO_ROOT / "configs" / "prereg.yaml",
                          power_n_simulations=10, power_n_bootstrap=10)


# ---------------------------------------------------------------------------
# End-to-end main(): report content, append-only guard
# ---------------------------------------------------------------------------


def _sandbox_repo(tmp_path: Path) -> dict[str, str]:
    for name in ("prereg.yaml", "prompt_v1.yaml", "extraction_v1.yaml", "runtime.yaml"):
        dst = tmp_path / "configs" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "configs" / name, dst)
    manifest = tmp_path / "data" / "sampling" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "data" / "sampling" / "manifest.json", manifest)
    return compute_study_hashes(tmp_path)


def test_main_writes_report_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ep, "POWER_N_SIMULATIONS", 50)
    monkeypatch.setattr(ep, "POWER_N_BOOTSTRAP", 100)
    hashes = _sandbox_repo(tmp_path)
    drop = {(m, k, qid(0)) for m in ep.MODELS for k in ("hard", "random")}
    build_dataset(tmp_path, hashes=hashes, drop=drop)

    rc = ep.main(["--repo-root", str(tmp_path)])
    assert rc == 0
    report_path = tmp_path / "results" / "pilot" / "acceptance_report.md"
    text = report_path.read_text(encoding="utf-8")
    assert (
        "| Model | Valid Answer Rate (>=85%) | Oracle EM | Oracle Headroom (>=10 pts) "
        "| Overall Verdict (PASS/FAIL) |"
    ) in text
    assert "single seed (seed 0)" in text and "conservative" in text
    assert "closed_book abstentions" in text
    assert "extend-to-800" in text
    assert "**All models PASS acceptance.**" in text
    assert f"`{qid(0)}`" in text  # P2c-excluded qid listed
    assert hashes["prereg_hash"] in text
    for m in ep.MODELS:
        assert f"| {m} |" in text
    capsys.readouterr()

    before = report_path.read_bytes()
    assert ep.main(["--repo-root", str(tmp_path)]) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert report_path.read_bytes() == before


def test_main_reports_failure_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ep, "POWER_N_SIMULATIONS", 20)
    monkeypatch.setattr(ep, "POWER_N_BOOTSTRAP", 50)
    hashes = _sandbox_repo(tmp_path)

    def fn(model: str, unit: str, i: int) -> tuple[str, bool, float]:
        ans, _, s = default_answer(model, unit, i)
        return ans, not (model == "Qwen3-4B" and unit == "random" and i < 10), s

    build_dataset(tmp_path, hashes=hashes, answer_fn=fn)
    out = tmp_path / "scratch" / "report.md"
    assert ep.main(["--repo-root", str(tmp_path), "--output", str(out)]) == 1
    text = out.read_text(encoding="utf-8")
    assert "Acceptance FAILED for: Qwen3-4B" in text
    assert "no model is substituted" in text
    assert "**FAIL**" in text


def test_main_hash_mismatch_returns_error(tmp_path: Path) -> None:
    _sandbox_repo(tmp_path)
    build_dataset(tmp_path, hashes=FAKE_HASHES)
    assert ep.main(["--repo-root", str(tmp_path)]) == 2
    assert not (tmp_path / "results" / "pilot" / "acceptance_report.md").exists()
