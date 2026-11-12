# Pilot acceptance report (P6 v2)

Pilot split: 50 questions (`data/sampling/pilot.json`). Models: Qwen3-8B, Qwen3-4B, Llama-3.1-8B-Instruct. Primary model: **Qwen3-8B**. Units per model: closed_book, oracle (n=0), hard_negative n=8 seed=0, random_length_matched n=8 seed=0.

Acceptance thresholds (fixed in the pre-registered proposal; not tuned): valid short answers >= 85.0 %, oracle headroom >= 10.0 pts.

**Overall model verdict rule (Stage B approved):** A model PASSES overall if (a) its oracle headroom is >= 10.0 percentage points AND (b) its pooled valid-answer rate across all 4 units OR across the 3 context-bearing units (oracle, hard_negative n=8, random_length_matched n=8) is >= 85.0 %. Any single unit below 85.0 % is flagged in the per-unit table for diagnostic visibility, but does not fail the model if either pooled rate passes.

## 1. Executive summary

| Model | Valid Answer Rate (>=85%) | Oracle EM | Oracle Headroom (>=10 pts) | Overall Verdict (PASS/FAIL) |
|---|---|---|---|---|
| Qwen3-8B | pooled_4=85.35% (PASS), pooled_3=85.14% (PASS) | 64.00 % (n=50) | 36.00 pts PASS | **PASS** |
| Qwen3-4B | pooled_4=78.28% (FAIL), pooled_3=81.08% (FAIL) | 66.00 % (n=50) | 34.00 pts PASS | **FAIL** |
| Llama-3.1-8B-Instruct | pooled_4=88.38% (PASS), pooled_3=88.51% (PASS) | 60.00 % (n=50) | 40.00 pts PASS | **PASS** |

Verdict components:

| Model | Valid, 4 units pooled (>=85%) | Valid, 3 context units pooled (>=85%) | Either pooled >= 85% | Oracle headroom (>=10 pts) | Single unit dips (<85%) | Verdict |
|---|---|---|---|---|---|---|
| Qwen3-8B | 85.35 % PASS | 85.14 % PASS | PASS | 36.00 pts PASS | hard_negative n=8 seed=0 | **PASS** |
| Qwen3-4B | 78.28 % FAIL | 81.08 % FAIL | FAIL | 34.00 pts PASS | closed_book, hard_negative n=8 seed=0, random_length_matched n=8 seed=0 | **FAIL** |
| Llama-3.1-8B-Instruct | 88.38 % PASS | 88.51 % PASS | PASS | 40.00 pts PASS | none | **PASS** |

**Acceptance FAILED for: Qwen3-4B.** Per the P6 brief, no model is substituted and no threshold, N, prompt or condition is changed; this is reported for a human decision.

## 2. Valid-short-answer rate per unit & invalid breakdown

> [!NOTE]
> closed_book abstentions (e.g. "I don't know") are expected when no context passage is provided and are counted as invalid by `configs/extraction_v1.yaml`. They do not fail a model as long as its pooled rate (across all 4 units or across the 3 context-bearing units) is >= 85.0 %.

| Model | Unit | Valid / Rows | Valid % | >= 85 % | Diagnostic Note |
|---|---|---|---|---|---|
| Qwen3-8B | closed_book | 43/50 | 86.00 | PASS | OK |
| Qwen3-8B | oracle (n=0) | 43/50 | 86.00 | PASS | OK |
| Qwen3-8B | hard_negative n=8 seed=0 | 40/49 | 81.63 | **< 85%** | dip (flagged for visibility; does not fail model if pooled passes) |
| Qwen3-8B | random_length_matched n=8 seed=0 | 43/49 | 87.76 | PASS | OK |
| Qwen3-8B | *3 context units pooled* | 126/148 | 85.14 | PASS | context benchmark |
| Qwen3-8B | *all 4 units pooled* | 169/198 | 85.35 | PASS | overall benchmark |
| Qwen3-4B | closed_book | 35/50 | 70.00 | **< 85%** | dip (flagged for visibility; does not fail model if pooled passes) |
| Qwen3-4B | oracle (n=0) | 43/50 | 86.00 | PASS | OK |
| Qwen3-4B | hard_negative n=8 seed=0 | 39/49 | 79.59 | **< 85%** | dip (flagged for visibility; does not fail model if pooled passes) |
| Qwen3-4B | random_length_matched n=8 seed=0 | 38/49 | 77.55 | **< 85%** | dip (flagged for visibility; does not fail model if pooled passes) |
| Qwen3-4B | *3 context units pooled* | 120/148 | 81.08 | FAIL | context benchmark |
| Qwen3-4B | *all 4 units pooled* | 155/198 | 78.28 | FAIL | overall benchmark |
| Llama-3.1-8B-Instruct | closed_book | 44/50 | 88.00 | PASS | OK |
| Llama-3.1-8B-Instruct | oracle (n=0) | 47/50 | 94.00 | PASS | OK |
| Llama-3.1-8B-Instruct | hard_negative n=8 seed=0 | 42/49 | 85.71 | PASS | OK |
| Llama-3.1-8B-Instruct | random_length_matched n=8 seed=0 | 42/49 | 85.71 | PASS | OK |
| Llama-3.1-8B-Instruct | *3 context units pooled* | 131/148 | 88.51 | PASS | context benchmark |
| Llama-3.1-8B-Instruct | *all 4 units pooled* | 175/198 | 88.38 | PASS | overall benchmark |

### Invalid answer breakdown (`valid == False`)

Breakdown of non-valid answers by cause (`>10 words only` vs. `forbidden prefix` vs. `empty` vs. `both >10 words & prefix`):

| Model | Unit | Total Invalid | >10 words only | Forbidden prefix | Empty | Both (>10 words & prefix) |
|---|---|---|---|---|---|---|
| Qwen3-8B | closed_book | 7 | 7 | 0 | 0 | 0 |
| Qwen3-8B | oracle (n=0) | 7 | 7 | 0 | 0 | 0 |
| Qwen3-8B | hard_negative n=8 seed=0 | 9 | 9 | 0 | 0 | 0 |
| Qwen3-8B | random_length_matched n=8 seed=0 | 6 | 6 | 0 | 0 | 0 |
| Qwen3-8B | *3 context units pooled* | 22 | 22 | 0 | 0 | 0 |
| Qwen3-8B | *all 4 units pooled* | 29 | 29 | 0 | 0 | 0 |
| Qwen3-4B | closed_book | 15 | 15 | 0 | 0 | 0 |
| Qwen3-4B | oracle (n=0) | 7 | 6 | 0 | 0 | 1 |
| Qwen3-4B | hard_negative n=8 seed=0 | 10 | 10 | 0 | 0 | 0 |
| Qwen3-4B | random_length_matched n=8 seed=0 | 11 | 10 | 0 | 0 | 1 |
| Qwen3-4B | *3 context units pooled* | 28 | 26 | 0 | 0 | 2 |
| Qwen3-4B | *all 4 units pooled* | 43 | 41 | 0 | 0 | 2 |
| Llama-3.1-8B-Instruct | closed_book | 6 | 4 | 2 | 0 | 0 |
| Llama-3.1-8B-Instruct | oracle (n=0) | 3 | 3 | 0 | 0 | 0 |
| Llama-3.1-8B-Instruct | hard_negative n=8 seed=0 | 7 | 7 | 0 | 0 | 0 |
| Llama-3.1-8B-Instruct | random_length_matched n=8 seed=0 | 7 | 7 | 0 | 0 | 0 |
| Llama-3.1-8B-Instruct | *3 context units pooled* | 17 | 17 | 0 | 0 | 0 |
| Llama-3.1-8B-Instruct | *all 4 units pooled* | 23 | 21 | 2 | 0 | 0 |

## 3. Accuracy per condition (EM primary, token F1 secondary)

EM and F1 are scored on `extracted_answer` against any gold answer using `src/metrics.py` (SQuAD normalization), with invalid answers scored as-is. *All rows* = every row present in the unit file; *paired* = the intersection of qids present in all 4 units.

| Model | Unit | n (all) | EM % (all) | F1 % (all) | n (paired) | EM % (paired) | F1 % (paired) |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | closed_book | 50 | 10.00 | 27.11 | 49 | 10.20 | 27.66 |
| Qwen3-8B | oracle (n=0) | 50 | 64.00 | 83.33 | 49 | 65.31 | 83.40 |
| Qwen3-8B | hard_negative n=8 seed=0 | 49 | 61.22 | 75.00 | 49 | 61.22 | 75.00 |
| Qwen3-8B | random_length_matched n=8 seed=0 | 49 | 71.43 | 83.69 | 49 | 71.43 | 83.69 |
| Qwen3-4B | closed_book | 50 | 6.00 | 21.74 | 49 | 6.12 | 22.19 |
| Qwen3-4B | oracle (n=0) | 50 | 66.00 | 86.95 | 49 | 65.31 | 86.69 |
| Qwen3-4B | hard_negative n=8 seed=0 | 49 | 48.98 | 72.89 | 49 | 48.98 | 72.89 |
| Qwen3-4B | random_length_matched n=8 seed=0 | 49 | 51.02 | 75.38 | 49 | 51.02 | 75.38 |
| Llama-3.1-8B-Instruct | closed_book | 50 | 10.00 | 24.22 | 49 | 10.20 | 24.72 |
| Llama-3.1-8B-Instruct | oracle (n=0) | 50 | 60.00 | 86.02 | 49 | 61.22 | 86.14 |
| Llama-3.1-8B-Instruct | hard_negative n=8 seed=0 | 49 | 53.06 | 77.73 | 49 | 53.06 | 77.73 |
| Llama-3.1-8B-Instruct | random_length_matched n=8 seed=0 | 49 | 57.14 | 80.98 | 49 | 57.14 | 80.98 |

### Oracle headroom

| Model | Oracle EM % (all rows, n) | Headroom pts (primary) | Oracle EM % (paired, n) | Headroom pts (paired) | Verdict (>= 10 pts) |
|---|---|---|---|---|---|
| Qwen3-8B | 64.00 (n=50) | 36.00 | 65.31 (n=49) | 34.69 | PASS |
| Qwen3-4B | 66.00 (n=50) | 34.00 | 65.31 (n=49) | 34.69 | PASS |
| Llama-3.1-8B-Instruct | 60.00 (n=50) | 40.00 | 61.22 (n=49) | 38.78 | PASS |

## 4. Paired differences (seed 0, paired intersection)

Percentage points with 95 % paired-bootstrap CIs (`src/stats.paired_bootstrap`, 10000 resamples over questions, seed 4207, the same resample indices for all three contrasts). hard_drop = acc(oracle) - acc(hard); random_drop = acc(oracle) - acc(random); gap = acc(random) - acc(hard). These pilot figures are descriptive; they are not a test of H1.

| Model | Metric | n | hard_drop [95% CI] | random_drop [95% CI] | gap [95% CI] |
|---|---|---|---|---|---|
| Qwen3-8B | EM | 49 | +4.08 [-6.12, +14.29] | -6.12 [-16.33, +4.08] | +10.20 [+0.00, +22.45] |
| Qwen3-8B | F1 | 49 | +8.40 [+0.26, +16.67] | -0.29 [-7.54, +6.14] | +8.69 [+1.95, +16.63] |
| Qwen3-4B | EM | 49 | +16.33 [+4.08, +28.57] | +14.29 [+6.12, +24.49] | +2.04 [-4.08, +10.20] |
| Qwen3-4B | F1 | 49 | +13.80 [+5.89, +22.86] | +11.31 [+4.64, +18.78] | +2.49 [-3.99, +9.39] |
| Llama-3.1-8B-Instruct | EM | 49 | +8.16 [-8.16, +24.49] | +4.08 [-6.12, +16.33] | +4.08 [-6.12, +14.29] |
| Llama-3.1-8B-Instruct | F1 | 49 | +8.41 [+0.04, +17.04] | +5.16 [-1.90, +12.56] | +3.25 [-4.72, +11.40] |

## 5. Throughput diagnostics

Tripwire: the overall per-model **mean** seconds per generation must stay <= 3.0 s (total wall-clock time scales with the mean). `seconds` is the backend generation time per row.

| Model | Unit | n | s/gen mean | s/gen median | s/gen p90 | tok/s mean | tok/s median | tok/s p90 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | closed_book | 50 | 0.344 | 0.307 | 0.478 | 19.35 | 16.44 | 31.46 |
| Qwen3-8B | oracle (n=0) | 50 | 0.809 | 0.788 | 1.051 | 9.20 | 7.61 | 17.52 |
| Qwen3-8B | hard_negative n=8 seed=0 | 49 | 4.593 | 4.594 | 5.417 | 1.92 | 1.41 | 4.51 |
| Qwen3-8B | random_length_matched n=8 seed=0 | 49 | 4.452 | 4.402 | 5.240 | 1.74 | 1.14 | 3.72 |
| Qwen3-8B | *overall* | 198 | 2.529 | 1.177 | 4.967 | 8.12 | 4.26 | 22.27 |
| Qwen3-4B | closed_book | 50 | 0.221 | 0.183 | 0.357 | 39.84 | 34.51 | 62.46 |
| Qwen3-4B | oracle (n=0) | 50 | 0.457 | 0.456 | 0.600 | 18.14 | 15.25 | 31.95 |
| Qwen3-4B | hard_negative n=8 seed=0 | 49 | 2.523 | 2.523 | 2.999 | 3.78 | 2.77 | 8.24 |
| Qwen3-4B | random_length_matched n=8 seed=0 | 49 | 2.616 | 2.624 | 3.255 | 3.69 | 2.65 | 7.62 |
| Qwen3-4B | *overall* | 198 | 1.443 | 0.714 | 2.841 | 16.49 | 9.46 | 49.14 |
| Llama-3.1-8B-Instruct | closed_book | 50 | 0.366 | 0.330 | 0.487 | 18.00 | 16.41 | 28.27 |
| Llama-3.1-8B-Instruct | oracle (n=0) | 50 | 0.859 | 0.843 | 1.083 | 7.83 | 7.32 | 13.19 |
| Llama-3.1-8B-Instruct | hard_negative n=8 seed=0 | 49 | 4.603 | 4.556 | 5.534 | 1.97 | 1.61 | 3.72 |
| Llama-3.1-8B-Instruct | random_length_matched n=8 seed=0 | 49 | 4.385 | 4.335 | 5.383 | 2.00 | 1.58 | 3.79 |
| Llama-3.1-8B-Instruct | *overall* | 198 | 2.534 | 1.302 | 4.972 | 7.51 | 4.23 | 19.21 |

| Model | Overall mean s/gen | Tripwire (<= 3.0 s/gen) |
|---|---|---|
| Qwen3-8B | 2.529 | OK (below tripwire) |
| Qwen3-4B | 1.443 | OK (below tripwire) |
| Llama-3.1-8B-Instruct | 2.534 | OK (below tripwire) |

## 6. Statistical power & N-extension rule

> [!IMPORTANT]
> Pilot power is estimated from a **single seed (seed 0)** of paired per-question EM outcomes. Per-question outcomes are therefore binary (0/1) rather than averages over 3 seeds, so their variance is higher than in the main study. The estimates are **conservative** relative to the 3-seed main analysis.

Method: `src/stats.estimate_power`. The pilot paired differences are centered and shifted to the pre-registered targets (5-pt hard drop, 3-pt gap). Power is the share of 1000 simulations whose 95 % bootstrap CI (2000 resamples) has a lower bound > 0. Simulation seed 4207. Rule: extend the primary comparison to N=800 iff gap power at N=500 < 0.80.

| Model | n pilot (paired) | SD hard diff | SD gap diff | Power hard N=500 | Power gap N=500 | Power hard N=800 | Power gap N=800 | extend-to-800 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (primary, decisive) | 49 | 0.3511 | 0.4206 | 0.888 | 0.365 | 0.985 | 0.512 | YES |
| Qwen3-4B (informational) | 49 | 0.4256 | 0.2491 | 0.771 | 0.791 | 0.938 | 0.941 | YES |
| Llama-3.1-8B-Instruct (informational) | 49 | 0.5714 | 0.4062 | 0.533 | 0.384 | 0.723 | 0.587 | YES |

**Decision (Qwen3-8B):** gap power at N=500 = 0.365 < 0.80, so the pre-registered rule says **extend the primary comparison to N=800**. This needs a human decision before the main runs.

## 7. Data integrity & provenance

All rows were checked against the current hashes: config_hash `bfa40e01e7aa251f537f9065ea090a86e10c389b3254dae1696428bfce9528b4`, prereg_hash `6b89f8cdf7559b7192874993fd70a0918124a243030e9d16c0d95a4aadb0d242`, manifest_hash `98dab46790100b14c6dc00234c10b85e54df702c951bcedc6b5ab7a3a3c9dcb5`. Stub rows are rejected. Model, condition, distractor_type, n, seed and split were verified per row.

Gold answers: `data/sampling/pilot.json` (sha256 `646f6bdb4a83ef83e4745b24c88b9a09a0e5f0316d235768e7c382d818502997`).

| Model | Unit | File | Rows | Missing pilot qids (P2c exclusions) | sha256 |
|---|---|---|---|---|---|
| Qwen3-8B | closed_book | `results/pilot/Qwen3-8B__closed_book__none__n0__seednone__pilot.jsonl` | 50 | none | `a1380a0a8a0aada5884fa0b09d0baadef9eb35b65ee6afbb7487ae27453e0ba3` |
| Qwen3-8B | oracle (n=0) | `results/pilot/Qwen3-8B__oracle__none__n0__seednone__pilot.jsonl` | 50 | none | `0a3d74709a5eb62b72d1f25b9e94cb4e822fbdb5f375d2bed6cee4f1f1eae12c` |
| Qwen3-8B | hard_negative n=8 seed=0 | `results/pilot/Qwen3-8B__rag__hard_negative__n8__seed0__pilot.jsonl` | 49 | `57269cc3dd62a815002e8b15` | `202fc68c3a96d249c1327836cc6db73c85ff1bcb57edf82d872f7434c7ab3361` |
| Qwen3-8B | random_length_matched n=8 seed=0 | `results/pilot/Qwen3-8B__rag__random_length_matched__n8__seed0__pilot.jsonl` | 49 | `57269cc3dd62a815002e8b15` | `289cb65d94d2fce85721f31d43bdd7a953138685dca29e7b8cbc2011a575da32` |
| Qwen3-4B | closed_book | `results/pilot/Qwen3-4B__closed_book__none__n0__seednone__pilot.jsonl` | 50 | none | `ec3379fdc959c913c445b2290246d3e737e78d96e32d27a74e5161c4f2896d20` |
| Qwen3-4B | oracle (n=0) | `results/pilot/Qwen3-4B__oracle__none__n0__seednone__pilot.jsonl` | 50 | none | `5bc1e2981143d552188e983e67b06b6f45bbcf66a21e1d99d7806f297dc94514` |
| Qwen3-4B | hard_negative n=8 seed=0 | `results/pilot/Qwen3-4B__rag__hard_negative__n8__seed0__pilot.jsonl` | 49 | `57269cc3dd62a815002e8b15` | `ddd6e1ced258945cb604b4a67c2e717ea079f585b9f33e2247a4d8168f335f8c` |
| Qwen3-4B | random_length_matched n=8 seed=0 | `results/pilot/Qwen3-4B__rag__random_length_matched__n8__seed0__pilot.jsonl` | 49 | `57269cc3dd62a815002e8b15` | `96a3ea3bba3ac83ceb5941a0d3e4488edcf55995ef10f8bdff5d18150a0a8827` |
| Llama-3.1-8B-Instruct | closed_book | `results/pilot/Llama-3.1-8B-Instruct__closed_book__none__n0__seednone__pilot.jsonl` | 50 | none | `4298e63969d149dde0b07b40828414436d0e3d0c17e5c7e08ff40017ad21c44c` |
| Llama-3.1-8B-Instruct | oracle (n=0) | `results/pilot/Llama-3.1-8B-Instruct__oracle__none__n0__seednone__pilot.jsonl` | 50 | none | `797b6bb4375d0484489b8c01be847df8a5a6be8f45cc62e4a8503a914e5ab886` |
| Llama-3.1-8B-Instruct | hard_negative n=8 seed=0 | `results/pilot/Llama-3.1-8B-Instruct__rag__hard_negative__n8__seed0__pilot.jsonl` | 49 | `57269cc3dd62a815002e8b15` | `74e91bb783a815be34a0de3dc121ac72f668c070b0e494382a124c76e95c16a3` |
| Llama-3.1-8B-Instruct | random_length_matched n=8 seed=0 | `results/pilot/Llama-3.1-8B-Instruct__rag__random_length_matched__n8__seed0__pilot.jsonl` | 49 | `57269cc3dd62a815002e8b15` | `adb7ab967e434ad421add6780ad7f294aca188387f6e3984249e06dc4446037a` |

Generated by `scripts/evaluate_pilot.py`. Every number above is computed from the files listed in this section.
