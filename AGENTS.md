# AGENTS.md: DSAI4207 RAG irrelevant-context study

## Mission
Controlled dose-response study: how the number and type (random vs topically related "hard negative") of irrelevant passages in a RAG prompt affect answer accuracy of small open-weight LLMs when the gold passage is present.
Pre-registered H1 (primary model Qwen3-8B): adding 8 hard negatives to gold-only context lowers accuracy by >= 5 points AND by >= 3 points more than adding 8 length-matched random passages; both 95% CIs exclude 0.

## Fixed design (never change without explicit human instruction)
- Data: SQuAD 2.0 dev, answerable only. 500 questions stratified by article, plus a disjoint 50-question pilot set and a 40-question reserve.
- Models: Qwen3-4B, Qwen3-8B (non-thinking), Llama-3.1-8B-Instruct. Same runtime, same 4-bit quantization. Reported runs happen only on the lead's MacBook.
- Conditions: closed-book; oracle (n=0); hard negatives n in {1,2,4,8}; random length-matched n in {1,2,4,8} (+/-20% tokens, rank-matched). Nested prefixes. 3 seeds = 3 independent distractor draws. Gold position randomized and logged.
- Decoding: greedy, max 32 new tokens, one standard prompt template.
- Metrics: primary = SQuAD-normalized exact match against any gold answer; secondary = token F1.
- Statistics: per-question accuracy averaged over seeds; paired bootstrap over questions, 10,000 resamples, 95% CI. H1 is the only primary test; all else exploratory with Holm correction.
- Length matching is measured with the Qwen3 tokenizer (secondary check: Llama-3.1 tokenizer).

## Hard rules
1. configs/prereg.yaml and data/frozen/ are read-only. Tests assert their hashes. Never edit them.
2. results/ is append-only. Never delete or rewrite result files. Every row carries config_hash, prereg_hash, manifest_hash.
3. Every reported number must be read from a file in results/. Never invent, estimate or round from memory.
4. No network use unless the task explicitly says so. Never print or commit secrets, student IDs or emails.
5. Write tests first or alongside. Run them. Show real output before claiming done.
6. If the spec is ambiguous or conflicts with what you find, STOP and ask. Do not guess.
7. Ask before any command expected to take longer than 10 minutes; print the estimate first.
8. Commit small and often: "<module>: <what>". Do not put test claims in commit messages; test results belong in the REVIEW PACKET.
9. Do not modify files outside the task's listed scope.
10. Run every project command inside the conda environment `rag`. Never use the system or Homebrew Python. Never pip install outside that environment.
11. Never set KMP_DUPLICATE_LIB_OK. If an OpenMP library conflict occurs, STOP and report it. Compute embeddings and FAISS search in separate processes (save embeddings to disk in between).

## Layout
README.md, requirements.txt, AGENTS.md, Makefile, configs/, src/, scripts/, tests/, data/ (frozen/ is read-only), results/, docs/ (reference/ is read-only).

## Task protocol
1. Plan first: list files to create or change, the tests, and risks. Wait for approval.
2. Implement. 3. Run tests and show output. 4. Commit.
5. End with a REVIEW PACKET: files changed, test results, commands run, assumptions made, open questions.