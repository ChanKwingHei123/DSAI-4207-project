# Pre-Outcome Design Clarifications & Construction Amendments

All clarifications below were specified and locked **prior to any LLM inference or outcome measurement** (`results/*.jsonl` empty) while constructing the distractor sets in Phase 2c (`p2c/distractors-inputs`). Fixed parameters in `configs/prereg.yaml` (`hard_negative_pool_k = 20`, `nested_prefixes = true`, `length_match_tolerance = 0.20`, `master_seed = 4207`, `draw_seeds = [0, 1, 2]`) remain unchanged.

---

## D1: Hard-Negative Pool ($k = 20$) and Nested Prefixes ($n \in \{1, 2, 4, 8\}$)
- **Pool construction**: For each question, dense-retrieved non-gold paragraphs are scanned in descending similarity rank order, rejecting any passage that fails the answer-leak filter. The first $k = 20$ (`design.hard_negative_pool_k`) non-rejected passages form the question's hard-negative pool (`hard_pool_paragraph_ids`).
- **Seed draws and nested prefixes**: For each `(question_id, draw_seed)` with `draw_seed` $\in \{0, 1, 2\}$, an isolated NumPy Generator (`np.random.default_rng([master_seed, draw_seed, uint64_sha256(question_id)])`) samples an ordered list of 8 hard negatives uniformly without replacement from the 20-candidate pool (`hard_paragraph_ids`), and a corresponding ordered list of 8 length-matched random passages (`random_paragraph_ids`). For each dose level $n \in \{1, 2, 4, 8\}$, the distractors are the prefix of length $n$ (`list[:1]`, `list[:2]`, `list[:4]`, `list[:8]`), ensuring strict nesting across $n$ within every `(question_id, draw_seed)`.
- **Gold position**: A single uniform fraction $p \in [0, 1)$ is drawn per `(question_id, draw_seed)` from the same RNG stream and shared across all conditions; for dose $n$, the gold passage is inserted at 0-based index $\text{round}(p \cdot n)$ among the $n + 1$ context slots.

---

## A3: Contiguous Token-Sequence Answer-Leak Filter
- **Motivation**: Under raw character-substring matching, single- or two-character normalized gold answers (such as `"p"`, `"k"`, `"qu"`, `"1"`) match inside unrelated words (e.g., `"p"` matching `"people"`), rejecting up to 30/30 candidates.
- **Rule**: Both the candidate passage and every gold answer are normalized with the official SQuAD v2.0 normalization (`lowercase`, strip ASCII punctuation, remove articles `a`/`an`/`the`, collapse whitespace) and split on whitespace into token lists.
  1. **Degenerate answer check**: If any gold answer of a question normalizes to an empty token list (`[]`), the question is marked `"degenerate"`, excluded from all outputs, and recorded in `data/inputs/build_report.md` (to be replaced from the reserve split at input freeze).
  2. **Contiguous token sub-sequence check**: A candidate passage (hard negative or random control) is rejected if and only if the normalized token list of **any** gold answer appears as a contiguous token sub-sequence of the passage's normalized token list.

---

## A4: Dense Candidate Retrieval Depth (`top_k = 100`)
- **Motivation**: For questions whose gold answer is a central entity or topic noun of the source article (e.g., `"california"`, `"steam"`, `"islam"`, `"oxygen"`, `"court of justice"`), more than 10 of the top-30 dense-retrieved paragraphs legitimately mention that term and are rejected by the leak filter, leaving fewer than $k = 20$ non-leaking candidates within the top 30.
- **Rule**: The FAISS retrieval stage (`scripts/search_hard_candidates.py`) writes the top 100 non-gold candidates per question (`top_k = 100`) to `data/candidates/hard_candidates.jsonl` (along with the question's `gold_score`). The hard pool remains the first $k = 20$ non-rejected candidates in rank order (`design.hard_negative_pool_k = 20`). Any question with fewer than 20 non-rejected candidates after depth 100 is excluded and listed in `data/inputs/build_report.md`. (Empirically, 0 of the 590 questions have fewer than 20 non-rejected candidates at depth 100.)

---

## A5: Progressive Length-Matching Window Inside the $\pm 20\%$ Hard Limit
- **Motivation**: Sampling uniformly across the entire $\pm 20\%$ token-length band creates a slight downward length drift (`mean(random - hard) = -4.46` Qwen3 tokens, `mean(|random - hard|) = 17.41` tokens) because paragraph lengths in SQuAD 2.0 dev are right-skewed (more shorter paragraphs than longer ones inside any symmetric $\pm 20\%$ interval).
- **Rule**: While preserving $\pm 20\%$ (`design.length_match_tolerance = 0.20`) as the strict hard limit, each random control slot $i \in \{0, \dots, 7\}$ is selected using a progressive window over the eligible (non-gold-article, non-leaking, unused) passages:
  1. Check passages within $\pm 5\%$ of the $i$-th hard negative's Qwen3 token length ($19 L_{\text{hard}} \le 20 L_{\text{cand}} \le 21 L_{\text{hard}}$).
  2. If none exist, check within $\pm 10\%$ ($9 L_{\text{hard}} \le 10 L_{\text{cand}} \le 11 L_{\text{hard}}$).
  3. If none exist, check within $\pm 20\%$ ($4 L_{\text{hard}} \le 5 L_{\text{cand}} \le 6 L_{\text{hard}}$).
  4. Sample one passage uniformly (`rng.integers`) from the first non-empty window tier and record the window used (`5`, `10`, or `20`) in `random_length_match_windows[i]`.
  5. If even the $\pm 20\%$ window has no eligible passage, do not relax the tolerance, substitute, or pad; record `(question, seed, slot)` in `data/inputs/build_report.md` and exclude that `(question, seed)` draw from the output.
