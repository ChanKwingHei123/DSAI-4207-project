# Schema Specification: `data/inputs/{main,pilot,reserve}_seed{0,1,2}.jsonl`

Each file in `data/inputs/{main,pilot,reserve}_seed{0,1,2}.jsonl` is a UTF-8 JSON Lines (`.jsonl`) file containing one JSON object per question in the corresponding split (`main`: 500 rows, `pilot`: 50 rows, `reserve`: 40 rows) for a specific distractor draw seed (`0`, `1`, or `2`).

---

## Top-Level Fields

| Field Name | JSON Type | Meaning |
| --- | --- | --- |
| `question_id` | `string` | Official SQuAD 2.0 question identifier (24-character hex string). |
| `split` | `string` | Dataset split name: `"main"`, `"pilot"`, or `"reserve"`. |
| `seed` | `integer` | Draw seed from `configs/prereg.yaml` (`rng.draw_seeds`: `0`, `1`, or `2`). |
| `article_index` | `integer` | 0-indexed article position (`0..34`) in `data/raw/dev-v2.0.json` for the question's gold passage. |
| `article_title` | `string` | SQuAD 2.0 Wikipedia article title for the question's gold passage. |
| `question_text` | `string` | Raw question string from SQuAD 2.0. |
| `gold_answers` | `array<string>` | All raw human-annotated gold answer strings for this question from `dev-v2.0.json`. |
| `gold_paragraph_id` | `array<integer>` (length 2) | Canonical paragraph identifier `[article_index, paragraph_index]` of the question's source (gold) paragraph in `data/raw/dev-v2.0.json`. |
| `hard_pool_paragraph_ids` | `array<array<integer>>` (length 20) | Ordered list of the first `k = 20` (`design.hard_negative_pool_k`) non-leaking dense-retrieved candidate paragraph IDs `[article_index, paragraph_index]` in descending similarity rank order. |
| `hard_paragraph_ids` | `array<array<integer>>` (length 8) | Ordered list of 8 hard-negative distractor paragraph IDs `[article_index, paragraph_index]` sampled uniformly without replacement from `hard_pool_paragraph_ids`. For condition $n \in \{1, 2, 4, 8\}$, the hard-negative distractors are the prefix `hard_paragraph_ids[:n]`. |
| `hard_candidate_scores` | `array<number>` (length 8) | Dense inner-product similarity scores (from `sentence-transformers/all-MiniLM-L6-v2`) corresponding slot-by-slot to `hard_paragraph_ids`. |
| `random_paragraph_ids` | `array<array<integer>>` (length 8) | Ordered list of 8 length-matched random distractor paragraph IDs `[article_index, paragraph_index]` sampled without replacement from articles other than `article_index` (`pid[0] != gold_paragraph_id[0]`), free of answer leaks, and within $\pm 20\%$ Qwen3 token length of the corresponding slot in `hard_paragraph_ids`. For condition $n \in \{1, 2, 4, 8\}$, the random-control distractors are the prefix `random_paragraph_ids[:n]`. |
| `p` | `number` (`float` in `[0.0, 1.0)`) | Uniform random fraction drawn per `(question_id, seed)` controlling where the gold paragraph is inserted among the $n + 1$ context slots across all conditions for this `(question_id, seed)`. |
| `gold_positions` | `object` | Mapping from stringified $n \in \{\text{"1"}, \text{"2"}, \text{"4"}, \text{"8"}\}$ to the 0-indexed insertion slot `round(p * n)` (integer in `0..n`) where `gold_paragraph_id` is inserted among the $n + 1$ context passages. |
| `token_lengths` | `object` | Token counts (`add_special_tokens=False`) under the primary (`qwen3`) and secondary-check (`llama3`) tokenizers (nested fields specified below). |
| `rejection_counts` | `object` | Per-question answer-leak filter diagnostic counts (nested fields specified below). |

---

## Nested Object: `gold_positions`

| Field Name | JSON Type | Meaning |
| --- | --- | --- |
| `"1"` | `integer` (`0..1`) | 0-indexed slot `round(p * 1)` where `gold_paragraph_id` is placed in the 2-passage context ($n = 1$). |
| `"2"` | `integer` (`0..2`) | 0-indexed slot `round(p * 2)` where `gold_paragraph_id` is placed in the 3-passage context ($n = 2$). |
| `"4"` | `integer` (`0..4`) | 0-indexed slot `round(p * 4)` where `gold_paragraph_id` is placed in the 5-passage context ($n = 4$). |
| `"8"` | `integer` (`0..8`) | 0-indexed slot `round(p * 8)` where `gold_paragraph_id` is placed in the 9-passage context ($n = 8$). |

---

## Nested Object: `token_lengths`

| Field Name | JSON Type | Meaning |
| --- | --- | --- |
| `token_lengths.qwen3` | `object` | Token lengths measured with `models/hf/Qwen__Qwen3-8B` (`add_special_tokens=False`), the primary length-matching tokenizer. |
| `token_lengths.qwen3.gold` | `integer` | Qwen3 token length of the gold paragraph text. |
| `token_lengths.qwen3.hard` | `array<integer>` (length 8) | Qwen3 token lengths of the 8 passages in `hard_paragraph_ids` in slot order `0..7`. |
| `token_lengths.qwen3.random` | `array<integer>` (length 8) | Qwen3 token lengths of the 8 passages in `random_paragraph_ids` in slot order `0..7`. Each slot $i$ satisfies $0.80 \cdot \text{hard}[i] \le \text{random}[i] \le 1.20 \cdot \text{hard}[i]$. |
| `token_lengths.llama3` | `object` | Token lengths measured with `models/hf/unsloth__Meta-Llama-3.1-8B-Instruct` (`add_special_tokens=False`), the secondary check tokenizer. |
| `token_lengths.llama3.gold` | `integer` | Llama-3.1 token length of the gold paragraph text. |
| `token_lengths.llama3.hard` | `array<integer>` (length 8) | Llama-3.1 token lengths of the 8 passages in `hard_paragraph_ids` in slot order `0..7`. |
| `token_lengths.llama3.random` | `array<integer>` (length 8) | Llama-3.1 token lengths of the 8 passages in `random_paragraph_ids` in slot order `0..7`. |

---

## Nested Object: `rejection_counts`

| Field Name | JSON Type | Meaning |
| --- | --- | --- |
| `rejection_counts.hard_top30_leak_rejected` | `integer` (`0..30`) | Number of passages among the question's top-30 dense candidates rejected by the SQuAD-normalized gold-answer substring filter. |
| `rejection_counts.hard_before_pool_k_leak_rejected` | `integer` (`0..30`) | Number of top-30 candidates rejected by the leak filter prior to accumulating the first $k = 20$ valid candidates for `hard_pool_paragraph_ids`. |
| `rejection_counts.hard_top30_non_rejected` | `integer` (`0..30`) | Total number of non-leaking candidates remaining out of the top-30 dense candidates (`30 - hard_top30_leak_rejected`; must be $\ge 20$). |
| `rejection_counts.random_pool_non_gold_article_total` | `integer` | Total number of paragraphs in the 1,204-paragraph dev pool belonging to articles other than the gold article (`article_index != gold_paragraph_id[0]`). |
| `rejection_counts.random_pool_leak_rejected` | `integer` | Number of paragraphs among `random_pool_non_gold_article_total` rejected because they contain a SQuAD-normalized gold answer substring. |

---

## Reconstructing Prompt Contexts from a Row

For any distractor condition (`distractor_type` $\in \{\text{"hard\_negative"}, \text{"random\_length\_matched"}\}$ and $n \in \{1, 2, 4, 8\}$):
1. Select the full 8-item distractor list (`hard_paragraph_ids` or `random_paragraph_ids`).
2. Take the nested prefix of length $n$: `prefix = distractor_list[:n]`.
3. Insert `gold_paragraph_id` at 0-based index `idx = round(p * n)` (`gold_positions[str(n)]`):
   `context_pids = prefix[:idx] + [gold_paragraph_id] + prefix[idx:]`.
4. Render the resulting $n + 1$ passages in order as `[Document 1] ... [Document n+1]` using `configs/prompt_v1.yaml`.
