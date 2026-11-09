# Distractor & Prompt-Input Build Report

## 1. Configuration & Exclusion Summary
- **Master seed**: `4207`
- **Draw seeds**: `[0, 1, 2]`
- **Candidate retrieval depth (`top_k`)**: `100` non-gold dense candidates per question
- **Hard-negative pool size (`k`)**: `20` (first `20` non-rejected candidates in rank order)
- **Leak filter (A3)**: Contiguous token sub-sequence matching on SQuAD-normalized whitespace-split tokens
- **Length-match rule (A5)**: Progressive window `[5, 10, 20]`% within hard limit `+/- 20%` (primary: `Qwen3-8B` tokenizer; secondary check: `Llama-3.1-8B-Instruct` tokenizer)
- **Total sampled questions processed**: `590` (`main=500`, `pilot=50`, `reserve=40`)

### Exclusions by Cause
| Exclusion Reason | Excluded Questions | Excluded `(question, seed)` Draws | Handling |
| --- | --- | --- | --- |
| **Degenerate gold answer** (any gold answer normalizes to empty token list `[]`) | `0` | `0` | Excluded from all outputs; to be replaced from reserve at freeze |
| **Insufficient hard pool (`< 20` non-rejected after depth `100`)** | `0` | `0` | Excluded from all outputs; recorded below |
| **Failed random length-match slot (no eligible passage within `+/- 20%`)** | `13` unique questions affected | `23` draws | Specific `(question, seed)` excluded; recorded below |

## 2. Hard-Candidate & Gold Similarity Score Diagnostics
Inner-product similarity scores (`sentence-transformers/all-MiniLM-L6-v2` L2-normalized embeddings) across non-excluded questions:

| Score Metric (per question) | N Questions | Min | Median | Mean | Max | Questions `< 0.30` (Count / Fraction) |
| --- | --- | --- | --- | --- | --- | --- |
| **Gold paragraph score** | 590 | 0.0059 | 0.5853 | 0.5619 | 0.8400 | 30 / 590 (5.08%) |
| **Mean hard-candidate score (20-candidate pool)** | 590 | 0.1810 | 0.3707 | 0.3784 | 0.6280 | 136 / 590 (23.05%) |
| **Mean hard-candidate score (sampled 8 distractors)** | 588 | 0.1778 | 0.3687 | 0.3773 | 0.6352 | 136 / 588 (23.13%) |

## 3. Progressive Length-Matching Window Usage & Token-Length Statistics
### Counts by Progressive Window Used (`+/- 5%`, `+/- 10%`, `+/- 20%`)
| Progressive Window Tier | Slot Count | Percentage of Matched Slots |
| --- | --- | --- |
| `+/- 5%` | 13902 | 99.47% |
| `+/- 10%` | 44 | 0.31% |
| `+/- 20%` | 30 | 0.21% |
| **Total matched slots** | **13976** | **100.00%** |

### Token-Length Differences (`random - hard`) After Progressive Window Matching
Evaluated across `13976` matched `(hard, random)` distractor pairs.

| Metric | Primary: Qwen3-8B Tokenizer | Secondary Check: Llama-3.1-8B Tokenizer |
| --- | --- | --- |
| Mean hard-negative token length | 171.4415 | 166.9375 |
| Mean random-control token length | 171.1172 | 166.3311 |
| **Mean signed difference (`random - hard`)** | **-0.3243** | **-0.6065** |
| **Mean absolute difference (`|random - hard|`)** | **4.3952** | **7.1689** |
| SD of signed difference (`random - hard`) | 6.9147 | 11.3013 |
| Median signed difference (`random - hard`) | +0.0 | +0.0 |
| Max absolute difference (`|random - hard|`) | 124 | 144 |
| Mean relative difference (`|random - hard| / hard`) | 2.50% | 4.12% |
| Max relative difference (`|random - hard| / hard`) | 19.81% | 33.49% |
| Mean signed diff (`main` split) | -0.3896 | -0.6676 |
| Mean signed diff (`pilot` split) | -0.2132 | -0.7808 |
| Mean signed diff (`reserve` split) | +0.3519 | +0.3687 |

## 4. Answer-Leak Rejection Counts (Token-Sequence Filter at Depth 100)
| Metric | Value |
| --- | --- |
| Non-degenerate questions evaluated | 590 / 590 |
| Questions with >= 1 rejected candidate in top-100 | 227 / 590 |
| Total top-100 hard candidates rejected | 1259 / 59000 |
| Top-100 rejections per question (mean / median / min / max) | 2.1339 / 0.0 / 0 / 38 |
| Rejections encountered before reaching k=20 hard pool (sum / max) | 770 / 35 |
| Non-gold-article dev pool rejections per question (mean / max) | 4.0610 / 301 |

## 5. Excluded Questions & Failed Slots Detail
### 5a. Degenerate Questions (Empty Normalized Gold Answer)
None (0 degenerate questions).

### 5b. Questions with Fewer Than 20 Non-Rejected Candidates After Depth 100
None (0 questions had fewer than 20 non-rejected candidates after depth 100).

### 5c. Failed Random Control Slots (No Eligible Passage Within `+/- 20%`)
| Question ID | Split | Seed | Slot | Hard Paragraph ID | Target Qwen3 Length |
| --- | --- | --- | --- | --- | --- |
| `5705e26d75f01819005e76d6` | `main` | 0 | 0 | `[2, 37]` | 43 |
| `5705e26d75f01819005e76d6` | `main` | 1 | 4 | `[2, 33]` | 43 |
| `5705e26d75f01819005e76d6` | `main` | 2 | 1 | `[2, 37]` | 43 |
| `5705e3f252bb89140068966e` | `main` | 0 | 1 | `[2, 33]` | 43 |
| `5705e3f252bb89140068966e` | `main` | 2 | 0 | `[2, 33]` | 43 |
| `5705e63175f01819005e7720` | `main` | 0 | 2 | `[2, 3]` | 42 |
| `5705fd8475f01819005e7844` | `main` | 2 | 6 | `[2, 3]` | 42 |
| `5706074552bb8914006897d8` | `main` | 0 | 4 | `[2, 33]` | 43 |
| `5706074552bb8914006897d8` | `main` | 1 | 4 | `[2, 33]` | 43 |
| `570607f575f01819005e78b7` | `main` | 0 | 7 | `[2, 33]` | 43 |
| `5706094b52bb8914006897df` | `main` | 0 | 2 | `[2, 33]` | 43 |
| `5706094b52bb8914006897df` | `main` | 1 | 1 | `[2, 37]` | 43 |
| `5706094b52bb8914006897df` | `main` | 2 | 7 | `[2, 33]` | 43 |
| `57060a6e52bb8914006897f8` | `main` | 1 | 6 | `[2, 33]` | 43 |
| `57060a6e52bb8914006897f8` | `main` | 2 | 3 | `[2, 33]` | 43 |
| `5706155352bb891400689895` | `main` | 1 | 6 | `[2, 37]` | 43 |
| `5706155352bb891400689895` | `main` | 2 | 3 | `[2, 37]` | 43 |
| `5726a1e5dd62a815002e8b86` | `main` | 0 | 7 | `[9, 31]` | 642 |
| `5705f09e75f01819005e77a5` | `pilot` | 1 | 6 | `[2, 3]` | 42 |
| `5705f09e75f01819005e77a5` | `pilot` | 2 | 6 | `[2, 3]` | 42 |
| `57269cc3dd62a815002e8b15` | `pilot` | 0 | 2 | `[9, 38]` | 788 |
| `57269cc3dd62a815002e8b15` | `pilot` | 2 | 5 | `[9, 38]` | 788 |
| `5705eb3375f01819005e7767` | `reserve` | 0 | 1 | `[2, 33]` | 43 |

## 6. Generated Output Files
| Split & Seed | Path | Row Count | SHA-256 |
| --- | --- | --- | --- |
| `main_seed0` | `data/inputs/main_seed0.jsonl` | 493 | `352fdb9dacd30fcc1685506dadcee50158d010b44d8b9a83deaff77bd5051c53` |
| `main_seed1` | `data/inputs/main_seed1.jsonl` | 495 | `3aa408a563e1a6fb1484be30d9adee80f5c86e4d95fac3a60f56762d6f43fcc1` |
| `main_seed2` | `data/inputs/main_seed2.jsonl` | 494 | `cc732eeaa6a29ea45f40ce093c12021e4e585a7363e3290e244dab3c971fa811` |
| `pilot_seed0` | `data/inputs/pilot_seed0.jsonl` | 49 | `15f6f90ac100f5e4da74e2add292ebf26b166f91033e92fc31448db5597cf286` |
| `pilot_seed1` | `data/inputs/pilot_seed1.jsonl` | 49 | `6aaf30998af0ce2aac8cb8c8be98f84bf5994cee8c4cd2bd2602d8bc5a9138bc` |
| `pilot_seed2` | `data/inputs/pilot_seed2.jsonl` | 48 | `71bcc2511e3a4f8ae32ed69742f808af88a2558320e8af524f85da2407e51025` |
| `reserve_seed0` | `data/inputs/reserve_seed0.jsonl` | 39 | `bf981d04153cdba206fc24a3be589432eedbcbe0b56bfc1e3b83c6485cd93e16` |
| `reserve_seed1` | `data/inputs/reserve_seed1.jsonl` | 40 | `1870b265bf1a38426d935b21aa63347b775b5014a246694c45cf53498287aa72` |
| `reserve_seed2` | `data/inputs/reserve_seed2.jsonl` | 40 | `bc9c1a869fda015a77a15d84f5cb18f1f1692b75b5be67c48f8296d6b3122bd2` |
