#!/usr/bin/env python3
"""Pilot evaluation & model acceptance report (P6 v2) for the DSAI4207 RAG study.

Reads the 12 official pilot result files (3 models x 4 pilot units) and the pilot gold
answers, then computes per model and unit:

(a) valid-short-answer rate (PASS/FAIL vs 85.0 %), pooled over all 4 units, pooled over the
    3 context-bearing units, and per unit;
(b) SQuAD EM and token F1 (src/metrics.py) on all present rows and on the paired
    intersection of qids present in all 4 units;
(c) oracle headroom = 100 - oracle EM % (PASS/FAIL vs 10.0 pts), primary on all oracle rows;
(d) paired hard_drop / random_drop / gap with 95 % paired-bootstrap CIs (src/stats.py);
(e) throughput diagnostics (mean / median / p90 of seconds and tokens_per_second) and the
    3.0 s/gen tripwire evaluated on the overall per-model mean;
(f) power for the 5-pt hard drop and 3-pt gap at N=500 / N=800 (src/stats.estimate_power)
    and the extend-to-800 rule.

Acceptance thresholds (85 % valid answers, 10 pts oracle headroom) are taken verbatim from the
pre-registered proposal (docs/reference/proposal_redacted.tex, "acceptance checks"); they are
fixed constants and must never be tuned. Decision rule (human-approved, P6 v2):

    PASS iff  valid rate pooled over all 4 units            >= 85.0 %
         AND  valid rate pooled over the 3 context units    >= 85.0 %
         AND  each context unit (oracle, hard n=8, random n=8) >= 85.0 %
         AND  oracle headroom (all oracle rows)             >= 10.0 pts

The report is written once; an existing report is never overwritten (results/ is append-only).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.harness import (  # noqa: E402
    RunUnit,
    compute_study_hashes,
    load_result_rows,
    sha256_file,
)
from src.metrics import exact_match, token_f1  # noqa: E402
from src.stats import estimate_power, load_prereg, paired_bootstrap  # noqa: E402

# Fixed acceptance thresholds (proposal, Weeks 8-9 acceptance checks). Never tune.
VALID_RATE_MIN_PCT = 85.0
HEADROOM_MIN_PTS = 10.0
TRIPWIRE_SEC_PER_GEN = 3.0

MODELS: tuple[str, ...] = ("Qwen3-8B", "Qwen3-4B", "Llama-3.1-8B-Instruct")
UNIT_KEYS: tuple[str, ...] = ("closed_book", "oracle", "hard", "random")
CONTEXT_UNIT_KEYS: tuple[str, ...] = ("oracle", "hard", "random")
HASH_KEYS: tuple[str, ...] = ("config_hash", "prereg_hash", "manifest_hash")

# Power simulation sizes (src/stats.estimate_power defaults). Tests may monkeypatch these.
POWER_N_SIMULATIONS = 1000
POWER_N_BOOTSTRAP = 2000

DEFAULT_SAMPLING_FILE = Path("data") / "sampling" / "pilot.json"
DEFAULT_OUTPUT = Path("results") / "pilot" / "acceptance_report.md"


class PilotDataError(ValueError):
    """Raised when pilot result files are missing, inconsistent, or untrustworthy."""


@dataclass(frozen=True)
class PilotUnitSpec:
    key: str
    label: str
    condition: str
    distractor_type: str | None
    n: int
    seed: int | None

    def run_unit(self, model: str) -> RunUnit:
        return RunUnit(
            model=model,
            condition=self.condition,
            distractor_type=self.distractor_type,
            n=self.n,
            seed=self.seed,
            split="pilot",
        )


def pilot_unit_specs(prereg: Mapping[str, Any]) -> dict[str, PilotUnitSpec]:
    """Build the 4 pilot units, reading the primary n and the first draw seed from prereg."""
    primary_n = int(prereg["primary_comparison"]["n"])
    seed0 = int(prereg["rng"]["draw_seeds"][0])
    hard_type, random_type = list(prereg["primary_comparison"]["distractor_types"])[:2]
    return {
        "closed_book": PilotUnitSpec("closed_book", "closed_book", "closed_book", None, 0, None),
        "oracle": PilotUnitSpec("oracle", "oracle (n=0)", "oracle", None, 0, None),
        "hard": PilotUnitSpec(
            "hard", f"hard_negative n={primary_n} seed={seed0}", "rag", hard_type, primary_n, seed0
        ),
        "random": PilotUnitSpec(
            "random",
            f"random_length_matched n={primary_n} seed={seed0}",
            "rag",
            random_type,
            primary_n,
            seed0,
        ),
    }


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------


def load_gold(sampling_file: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return (canonical qid order, qid -> gold answers) from data/sampling/pilot.json."""
    if not sampling_file.exists():
        raise PilotDataError(f"Missing pilot sampling file: {sampling_file}")
    items = json.loads(sampling_file.read_text(encoding="utf-8"))
    order: list[str] = []
    gold: dict[str, list[str]] = {}
    for item in items:
        qid = str(item["question_id"])
        answers = [str(a) for a in item["gold_answers"]]
        if not answers:
            raise PilotDataError(f"Pilot question {qid} has no gold answers.")
        if qid in gold:
            raise PilotDataError(f"Duplicate qid {qid} in {sampling_file}")
        order.append(qid)
        gold[qid] = answers
    return order, gold


def load_unit_rows(
    path: Path,
    *,
    model: str,
    spec: PilotUnitSpec,
    gold: Mapping[str, Sequence[str]],
    expected_hashes: Mapping[str, str],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Load and strictly validate one pilot unit file."""
    if not path.exists():
        raise PilotDataError(f"Missing pilot result file: {path}")
    try:
        rows = load_result_rows(path, allow_stub=False, repo_root=repo_root)
    except ValueError as exc:
        raise PilotDataError(str(exc)) from exc
    if not rows:
        raise PilotDataError(f"Pilot result file is empty: {path}")

    seen: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        where = f"{path.name}:{idx}"
        qid = str(row.get("qid", ""))
        if qid in seen:
            raise PilotDataError(f"{where}: duplicate qid {qid}")
        seen.add(qid)
        if qid not in gold:
            raise PilotDataError(f"{where}: qid {qid} not in pilot sampling file")
        if row.get("split") != "pilot":
            raise PilotDataError(f"{where}: split={row.get('split')!r}, expected 'pilot'")
        expected_fields = {
            "model": model,
            "condition": spec.condition,
            "distractor_type": spec.distractor_type,
            "n": spec.n,
            "seed": spec.seed,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise PilotDataError(
                    f"{where}: {field}={row.get(field)!r} does not match unit ({expected!r})"
                )
        for key in HASH_KEYS:
            if row.get(key) != expected_hashes[key]:
                raise PilotDataError(
                    f"{where}: {key}={row.get(key)!r} != current {expected_hashes[key]!r}"
                )
        if not isinstance(row.get("valid"), bool):
            raise PilotDataError(f"{where}: 'valid' must be a boolean")
        for num_key in ("seconds", "tokens_per_second"):
            if not isinstance(row.get(num_key), (int, float)):
                raise PilotDataError(f"{where}: '{num_key}' must be numeric")
        if not isinstance(row.get("extracted_answer"), str):
            raise PilotDataError(f"{where}: 'extracted_answer' must be a string")
    return rows


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def pct(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        raise PilotDataError("Cannot compute a percentage over zero rows.")
    return 100.0 * float(numerator) / float(denominator)


def valid_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_valid = sum(1 for r in rows if r["valid"] is True)
    rate = pct(n_valid, len(rows))
    return {
        "n_rows": len(rows),
        "n_valid": n_valid,
        "valid_pct": rate,
        "pass": bool(rate >= VALID_RATE_MIN_PCT),
    }


def distribution(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def throughput_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "seconds": distribution([float(r["seconds"]) for r in rows]),
        "tokens_per_second": distribution([float(r["tokens_per_second"]) for r in rows]),
    }


def score_rows(
    rows: Sequence[Mapping[str, Any]], gold: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, float]]:
    """Return qid -> {'em': 0/1, 'f1': [0, 1]} scored on `extracted_answer`."""
    return {
        str(r["qid"]): {
            "em": exact_match(str(r["extracted_answer"]), gold[str(r["qid"])]),
            "f1": token_f1(str(r["extracted_answer"]), gold[str(r["qid"])]),
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: str,
    results_dir: Path,
    *,
    specs: Mapping[str, PilotUnitSpec],
    qid_order: Sequence[str],
    gold: Mapping[str, Sequence[str]],
    expected_hashes: Mapping[str, str],
    prereg: Mapping[str, Any],
    prereg_path: Path,
    repo_root: Path,
    power_n_simulations: int,
    power_n_bootstrap: int,
) -> dict[str, Any]:
    unit_rows: dict[str, list[dict[str, Any]]] = {}
    unit_files: dict[str, dict[str, Any]] = {}
    for key in UNIT_KEYS:
        spec = specs[key]
        path = spec.run_unit(model).output_path(results_dir)
        rows = load_unit_rows(
            path,
            model=model,
            spec=spec,
            gold=gold,
            expected_hashes=expected_hashes,
            repo_root=repo_root,
        )
        unit_rows[key] = rows
        present = {str(r["qid"]) for r in rows}
        unit_files[key] = {
            "path": path,
            "sha256": sha256_file(path),
            "n_rows": len(rows),
            "missing_qids": [q for q in qid_order if q not in present],
        }

    # (a) valid-answer rates
    valid_units = {k: valid_summary(unit_rows[k]) for k in UNIT_KEYS}
    valid_overall = valid_summary([r for k in UNIT_KEYS for r in unit_rows[k]])
    valid_context = valid_summary([r for k in CONTEXT_UNIT_KEYS for r in unit_rows[k]])
    context_units_pass = all(valid_units[k]["pass"] for k in CONTEXT_UNIT_KEYS)

    # (b) accuracy (all present rows + paired intersection)
    scores = {k: score_rows(unit_rows[k], gold) for k in UNIT_KEYS}
    acc_all: dict[str, dict[str, Any]] = {}
    for k in UNIT_KEYS:
        n_rows = len(scores[k])
        acc_all[k] = {
            "n": n_rows,
            "em_pct": pct(sum(s["em"] for s in scores[k].values()), n_rows),
            "f1_pct": pct(sum(s["f1"] for s in scores[k].values()), n_rows),
        }
    paired_qids = [q for q in qid_order if all(q in scores[k] for k in UNIT_KEYS)]
    if len(paired_qids) < 2:
        raise PilotDataError(f"{model}: paired intersection has {len(paired_qids)} qids (< 2).")
    vectors = {
        metric: {
            k: np.asarray([scores[k][q][metric] for q in paired_qids], dtype=np.float64)
            for k in UNIT_KEYS
        }
        for metric in ("em", "f1")
    }
    acc_paired = {
        k: {
            "n": len(paired_qids),
            "em_pct": 100.0 * float(np.mean(vectors["em"][k])),
            "f1_pct": 100.0 * float(np.mean(vectors["f1"][k])),
        }
        for k in UNIT_KEYS
    }

    # (c) oracle headroom: primary on all oracle rows, paired shown alongside
    headroom_pts = 100.0 - acc_all["oracle"]["em_pct"]
    headroom_paired_pts = 100.0 - acc_paired["oracle"]["em_pct"]
    headroom_pass = bool(headroom_pts >= HEADROOM_MIN_PTS)

    # (d) paired drops with 95 % paired-bootstrap CIs
    n_resamples = int(prereg["statistics"]["bootstrap_resamples"])
    ci_level = float(prereg["thresholds"]["ci_level"])
    boot_seed = int(prereg["rng"]["master_seed"])
    drops = {
        metric: paired_bootstrap(
            vectors[metric]["oracle"],
            vectors[metric]["hard"],
            vectors[metric]["random"],
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=boot_seed,
        )
        for metric in ("em", "f1")
    }

    # (e) throughput
    throughput_units = {k: throughput_summary(unit_rows[k]) for k in UNIT_KEYS}
    throughput_overall = throughput_summary([r for k in UNIT_KEYS for r in unit_rows[k]])
    tripwire_tripped = bool(throughput_overall["seconds"]["mean"] > TRIPWIRE_SEC_PER_GEN)

    # (f) power (paired seed-0 EM outcomes)
    power = estimate_power(
        vectors["em"]["oracle"],
        vectors["em"]["hard"],
        vectors["em"]["random"],
        prereg_path=prereg_path,
        n_simulations=power_n_simulations,
        n_bootstrap=power_n_bootstrap,
    )

    verdict_checks = {
        "valid_overall_4_units": valid_overall["pass"],
        "valid_context_pooled_3_units": valid_context["pass"],
        "valid_each_context_unit": context_units_pass,
        "oracle_headroom": headroom_pass,
    }
    return {
        "model": model,
        "unit_files": unit_files,
        "valid_units": valid_units,
        "valid_overall": valid_overall,
        "valid_context": valid_context,
        "acc_all": acc_all,
        "acc_paired": acc_paired,
        "paired_qids": paired_qids,
        "headroom_pts": headroom_pts,
        "headroom_paired_pts": headroom_paired_pts,
        "headroom_pass": headroom_pass,
        "drops": drops,
        "throughput_units": throughput_units,
        "throughput_overall": throughput_overall,
        "tripwire_tripped": tripwire_tripped,
        "power": power,
        "verdict_checks": verdict_checks,
        "verdict_pass": bool(all(verdict_checks.values())),
    }


def evaluate_pilot(
    results_dir: Path,
    sampling_file: Path,
    *,
    repo_root: Path = REPO_ROOT,
    prereg_path: Path | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    models: Sequence[str] = MODELS,
    power_n_simulations: int | None = None,
    power_n_bootstrap: int | None = None,
) -> dict[str, Any]:
    """Evaluate all pilot units for all models; raises PilotDataError on any inconsistency."""
    p_prereg = prereg_path or (repo_root / "configs" / "prereg.yaml")
    prereg = load_prereg(p_prereg)
    hashes = dict(expected_hashes) if expected_hashes is not None else compute_study_hashes(repo_root)
    specs = pilot_unit_specs(prereg)
    qid_order, gold = load_gold(sampling_file)
    n_sims = int(power_n_simulations if power_n_simulations is not None else POWER_N_SIMULATIONS)
    n_boot = int(power_n_bootstrap if power_n_bootstrap is not None else POWER_N_BOOTSTRAP)

    per_model = {
        m: evaluate_model(
            m,
            results_dir,
            specs=specs,
            qid_order=qid_order,
            gold=gold,
            expected_hashes=hashes,
            prereg=prereg,
            prereg_path=p_prereg,
            repo_root=repo_root,
            power_n_simulations=n_sims,
            power_n_bootstrap=n_boot,
        )
        for m in models
    }
    return {
        "primary_model": str(prereg["primary_model"]),
        "prereg": prereg,
        "hashes": hashes,
        "specs": specs,
        "sampling_file": sampling_file,
        "sampling_sha256": sha256_file(sampling_file),
        "n_pilot_questions": len(qid_order),
        "models": per_model,
        "all_pass": bool(all(r["verdict_pass"] for r in per_model.values())),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _pf(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _ci(summary: Mapping[str, Any]) -> str:
    return (
        f"{summary['estimate_pts']:+.2f} "
        f"[{summary['ci_low_pts']:+.2f}, {summary['ci_high_pts']:+.2f}]"
    )


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def render_report(ev: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> str:
    specs: Mapping[str, PilotUnitSpec] = ev["specs"]
    models: Mapping[str, Any] = ev["models"]
    primary = ev["primary_model"]
    prereg = ev["prereg"]
    lines: list[str] = []
    add = lines.append

    add("# Pilot acceptance report (P6 v2)")
    add("")
    add(
        f"Pilot split: {ev['n_pilot_questions']} questions (`{_rel(ev['sampling_file'], repo_root)}`). "
        f"Models: {', '.join(models)}. Primary model: **{primary}**. "
        "Units per model: closed_book, oracle (n=0), "
        f"{specs['hard'].label}, {specs['random'].label}."
    )
    add("")
    add(
        f"Acceptance thresholds (fixed in the pre-registered proposal; not tuned): valid short "
        f"answers >= {VALID_RATE_MIN_PCT:.1f} %, oracle headroom >= {HEADROOM_MIN_PTS:.1f} pts. "
        f"A model PASSES iff the valid rate pooled over all 4 units >= {VALID_RATE_MIN_PCT:.1f} %, "
        f"the valid rate pooled over the 3 context-bearing units (oracle, hard, random) "
        f">= {VALID_RATE_MIN_PCT:.1f} %, each context-bearing unit individually >= "
        f"{VALID_RATE_MIN_PCT:.1f} %, and oracle headroom (all oracle rows) >= "
        f"{HEADROOM_MIN_PTS:.1f} pts."
    )
    add("")

    # Executive summary
    add("## 1. Executive summary")
    add("")
    add(
        "| Model | Valid Answer Rate (>=85%) | Oracle EM | Oracle Headroom (>=10 pts) "
        "| Overall Verdict (PASS/FAIL) |"
    )
    add("|---|---|---|---|---|")
    for m, r in models.items():
        vo = r["valid_overall"]
        add(
            f"| {m} | {vo['valid_pct']:.2f} % ({vo['n_valid']}/{vo['n_rows']}) {_pf(vo['pass'])} "
            f"| {r['acc_all']['oracle']['em_pct']:.2f} % (n={r['acc_all']['oracle']['n']}) "
            f"| {r['headroom_pts']:.2f} pts {_pf(r['headroom_pass'])} "
            f"| **{_pf(r['verdict_pass'])}** |"
        )
    add("")
    add("Verdict components:")
    add("")
    add(
        "| Model | Valid, all 4 units pooled | Valid, 3 context units pooled "
        "| Each context unit >= 85 % | Oracle headroom | Verdict |"
    )
    add("|---|---|---|---|---|---|")
    for m, r in models.items():
        c = r["verdict_checks"]
        add(
            f"| {m} | {r['valid_overall']['valid_pct']:.2f} % {_pf(c['valid_overall_4_units'])} "
            f"| {r['valid_context']['valid_pct']:.2f} % {_pf(c['valid_context_pooled_3_units'])} "
            f"| {_pf(c['valid_each_context_unit'])} "
            f"| {r['headroom_pts']:.2f} pts {_pf(c['oracle_headroom'])} "
            f"| **{_pf(r['verdict_pass'])}** |"
        )
    add("")
    if ev["all_pass"]:
        add("**All models PASS acceptance.**")
    else:
        failed = [m for m, r in models.items() if not r["verdict_pass"]]
        add(
            f"**Acceptance FAILED for: {', '.join(failed)}.** Per the P6 brief, no model is "
            "substituted and no threshold, N, prompt or condition is changed; this is reported "
            "for a human decision."
        )
    add("")

    # Per-unit valid rates
    add("## 2. Valid-short-answer rate per unit")
    add("")
    add(
        "> [!NOTE]\n"
        "> closed_book abstentions (e.g. \"I don't know\") are expected when no context passage is "
        "provided and are counted as invalid by `configs/extraction_v1.yaml`. They do not fail a "
        f"model as long as its overall rate and its context-bearing units are >= "
        f"{VALID_RATE_MIN_PCT:.1f} %."
    )
    add("")
    add("| Model | Unit | Valid / Rows | Valid % | >= 85 % |")
    add("|---|---|---|---|---|")
    for m, r in models.items():
        for k in UNIT_KEYS:
            v = r["valid_units"][k]
            flag = _pf(v["pass"]) if v["pass"] else "**FAIL**"
            add(f"| {m} | {specs[k].label} | {v['n_valid']}/{v['n_rows']} | {v['valid_pct']:.2f} | {flag} |")
        vc, vo = r["valid_context"], r["valid_overall"]
        add(f"| {m} | *3 context units pooled* | {vc['n_valid']}/{vc['n_rows']} | {vc['valid_pct']:.2f} | {_pf(vc['pass'])} |")
        add(f"| {m} | *all 4 units pooled* | {vo['n_valid']}/{vo['n_rows']} | {vo['valid_pct']:.2f} | {_pf(vo['pass'])} |")
    add("")

    # Accuracy
    add("## 3. Accuracy per condition (EM primary, token F1 secondary)")
    add("")
    add(
        "EM and F1 are scored on `extracted_answer` against any gold answer using "
        "`src/metrics.py` (SQuAD normalization), with invalid answers scored as-is. *All rows* = every row "
        "present in the unit file; *paired* = the intersection of qids present in all 4 units."
    )
    add("")
    add("| Model | Unit | n (all) | EM % (all) | F1 % (all) | n (paired) | EM % (paired) | F1 % (paired) |")
    add("|---|---|---|---|---|---|---|---|")
    for m, r in models.items():
        for k in UNIT_KEYS:
            a, p = r["acc_all"][k], r["acc_paired"][k]
            add(
                f"| {m} | {specs[k].label} | {a['n']} | {a['em_pct']:.2f} | {a['f1_pct']:.2f} "
                f"| {p['n']} | {p['em_pct']:.2f} | {p['f1_pct']:.2f} |"
            )
    add("")
    add("### Oracle headroom")
    add("")
    add("| Model | Oracle EM % (all rows, n) | Headroom pts (primary) | Oracle EM % (paired, n) | Headroom pts (paired) | Verdict (>= 10 pts) |")
    add("|---|---|---|---|---|---|")
    for m, r in models.items():
        a, p = r["acc_all"]["oracle"], r["acc_paired"]["oracle"]
        add(
            f"| {m} | {a['em_pct']:.2f} (n={a['n']}) | {r['headroom_pts']:.2f} "
            f"| {p['em_pct']:.2f} (n={p['n']}) | {r['headroom_paired_pts']:.2f} "
            f"| {_pf(r['headroom_pass'])} |"
        )
    add("")

    # Paired drops
    n_res = int(prereg["statistics"]["bootstrap_resamples"])
    add("## 4. Paired differences (seed 0, paired intersection)")
    add("")
    add(
        f"Percentage points with {float(prereg['thresholds']['ci_level']) * 100:.0f} % paired-bootstrap "
        f"CIs (`src/stats.paired_bootstrap`, {n_res} resamples over questions, seed "
        f"{int(prereg['rng']['master_seed'])}, the same resample indices for all three contrasts). "
        "hard_drop = acc(oracle) - acc(hard); random_drop = acc(oracle) - acc(random); "
        "gap = acc(random) - acc(hard). These pilot figures are descriptive; they are not a test of H1."
    )
    add("")
    add("| Model | Metric | n | hard_drop [95% CI] | random_drop [95% CI] | gap [95% CI] |")
    add("|---|---|---|---|---|---|")
    for m, r in models.items():
        for metric, label in (("em", "EM"), ("f1", "F1")):
            d = r["drops"][metric]
            add(
                f"| {m} | {label} | {d['n_questions']} | {_ci(d['hard_drop'])} "
                f"| {_ci(d['random_drop'])} | {_ci(d['gap'])} |"
            )
    add("")

    # Throughput
    add("## 5. Throughput diagnostics")
    add("")
    add(
        f"Tripwire: the overall per-model **mean** seconds per generation must stay "
        f"<= {TRIPWIRE_SEC_PER_GEN:.1f} s (total wall-clock time scales with the mean). "
        "`seconds` is the backend generation time per row."
    )
    add("")
    add("| Model | Unit | n | s/gen mean | s/gen median | s/gen p90 | tok/s mean | tok/s median | tok/s p90 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for m, r in models.items():
        entries = [(specs[k].label, r["throughput_units"][k]) for k in UNIT_KEYS]
        entries.append(("*overall*", r["throughput_overall"]))
        for label, t in entries:
            s, tps = t["seconds"], t["tokens_per_second"]
            add(
                f"| {m} | {label} | {t['n_rows']} | {s['mean']:.3f} | {s['median']:.3f} | {s['p90']:.3f} "
                f"| {tps['mean']:.2f} | {tps['median']:.2f} | {tps['p90']:.2f} |"
            )
    add("")
    add("| Model | Overall mean s/gen | Tripwire (<= 3.0 s/gen) |")
    add("|---|---|---|")
    for m, r in models.items():
        status = "TRIPPED" if r["tripwire_tripped"] else "OK (below tripwire)"
        add(f"| {m} | {r['throughput_overall']['seconds']['mean']:.3f} | {status} |")
    add("")

    # Power
    pw_primary = models[primary]["power"] if primary in models else None
    add("## 6. Statistical power & N-extension rule")
    add("")
    add(
        "> [!IMPORTANT]\n"
        "> Pilot power is estimated from a **single seed (seed 0)** of paired per-question EM "
        "outcomes. Per-question outcomes are therefore binary (0/1) rather than averages over 3 "
        "seeds, so their variance is higher than in the main study. The estimates are "
        "**conservative** relative to the 3-seed main analysis."
    )
    add("")
    if pw_primary is not None:
        add(
            f"Method: `src/stats.estimate_power`. The pilot paired differences are centered and shifted to "
            f"the pre-registered targets ({pw_primary['target_hard_drop_pts']:.0f}-pt hard drop, "
            f"{pw_primary['target_gap_pts']:.0f}-pt gap). Power is the share of "
            f"{pw_primary['n_simulations']} simulations whose 95 % bootstrap CI "
            f"({pw_primary['n_bootstrap']} resamples) has a lower bound > 0. Simulation seed "
            f"{pw_primary['seed']}. Rule: extend the primary comparison to N=800 iff gap power at "
            f"N=500 < {pw_primary['power_threshold']:.2f}."
        )
        add("")
    add("| Model | n pilot (paired) | SD hard diff | SD gap diff | Power hard N=500 | Power gap N=500 | Power hard N=800 | Power gap N=800 | extend-to-800 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for m, r in models.items():
        pw = r["power"]
        tag = " (primary, decisive)" if m == primary else " (informational)"
        add(
            f"| {m}{tag} | {pw['n_pilot_questions']} | {pw['pilot_sd_hard_drop']:.4f} "
            f"| {pw['pilot_sd_gap']:.4f} | {pw['power_hard_drop_500']:.3f} | {pw['power_gap_500']:.3f} "
            f"| {pw['power_hard_drop_800']:.3f} | {pw['power_gap_800']:.3f} "
            f"| {'YES' if pw['extend_to_800'] else 'no'} |"
        )
    add("")
    if pw_primary is not None:
        if pw_primary["extend_to_800"]:
            add(
                f"**Decision ({primary}):** gap power at N=500 = {pw_primary['power_gap_500']:.3f} "
                f"< {pw_primary['power_threshold']:.2f}, so the pre-registered rule says **extend the "
                f"primary comparison to N={pw_primary['recommended_n']}**. This needs a human decision "
                "before the main runs."
            )
        else:
            add(
                f"**Decision ({primary}):** gap power at N=500 = {pw_primary['power_gap_500']:.3f} "
                f">= {pw_primary['power_threshold']:.2f}, so **N stays at "
                f"{pw_primary['recommended_n']}** (no extension)."
            )
        add("")

    # Data integrity
    add("## 7. Data integrity & provenance")
    add("")
    h = ev["hashes"]
    add(
        f"All rows were checked against the current hashes: config_hash `{h['config_hash']}`, "
        f"prereg_hash `{h['prereg_hash']}`, manifest_hash `{h['manifest_hash']}`. Stub rows are "
        "rejected. Model, condition, distractor_type, n, seed and split were verified per row."
    )
    add("")
    add(f"Gold answers: `{_rel(ev['sampling_file'], repo_root)}` (sha256 `{ev['sampling_sha256']}`).")
    add("")
    add("| Model | Unit | File | Rows | Missing pilot qids (P2c exclusions) | sha256 |")
    add("|---|---|---|---|---|---|")
    for m, r in models.items():
        for k in UNIT_KEYS:
            f = r["unit_files"][k]
            missing = ", ".join(f"`{q}`" for q in f["missing_qids"]) or "none"
            add(
                f"| {m} | {specs[k].label} | `{_rel(f['path'], repo_root)}` | {f['n_rows']} "
                f"| {missing} | `{f['sha256']}` |"
            )
    add("")
    add("Generated by `scripts/evaluate_pilot.py`. Every number above is computed from the files listed in this section.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pilot evaluation & model acceptance report (P6).")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-dir", type=Path, default=None, help="Default: <repo>/results")
    parser.add_argument("--sampling-file", type=Path, default=None, help="Default: <repo>/data/sampling/pilot.json")
    parser.add_argument("--output", type=Path, default=None, help="Default: <repo>/results/pilot/acceptance_report.md")
    return parser


def _resolve(path: Path | None, default: Path, repo_root: Path) -> Path:
    p = path if path is not None else default
    return (p if p.is_absolute() else repo_root / p).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = Path(args.repo_root).resolve()
    results_dir = _resolve(args.results_dir, Path("results"), repo_root)
    sampling_file = _resolve(args.sampling_file, DEFAULT_SAMPLING_FILE, repo_root)
    output = _resolve(args.output, DEFAULT_OUTPUT, repo_root)

    if output.exists():
        print(
            f"[ERROR] Refusing to overwrite existing report {output} (results/ is append-only).",
            file=sys.stderr,
        )
        return 2

    try:
        ev = evaluate_pilot(results_dir, sampling_file, repo_root=repo_root)
    except PilotDataError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    report = render_report(ev, repo_root=repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write(report)

    for m, r in ev["models"].items():
        print(
            f"[{_pf(r['verdict_pass'])}] {m}: valid={r['valid_overall']['valid_pct']:.2f}% "
            f"context_valid={r['valid_context']['valid_pct']:.2f}% "
            f"oracle_EM={r['acc_all']['oracle']['em_pct']:.2f}% headroom={r['headroom_pts']:.2f}pts "
            f"mean_s/gen={r['throughput_overall']['seconds']['mean']:.3f}"
        )
    print(f"[REPORT] {output}")
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
