Experimental Setup

1. Evaluation Dataset and Frozen Splits
We evaluate our models using a strictly controlled subset of the official SQuAD 2.0 development set. To ensure a clean signal for RAG reading behavior, we isolate answerable questions and discard unanswerable queries. This base frame is shuffled using a deterministic random seed of 42. From this split, we lock down two disjoint sets verified by SHA-256 cryptographic hashes:
• Pilot Set (\(N=50\)): Used for runtime throughput validation and format acceptance checks.
• Main Evaluation Grid (\(N=500\)): Used for the comprehensive multi-variable evaluation sweep.

2. Distractor Selection and Context Controls
For each question, a pool of distractor contexts is constructed to analyze model degradation across variable volumetric doses \(n \in \{0, 1, 2, 4, 8\}\). We test two distinct context topologies:
• Topically Related (Hard) Distractors: Formed by extracting the top-16 nearest-neighbor text fragments from the dev-set paragraph pool using a dense index built with all-MiniLM-L6-v2 embeddings and a FAISS backend. The gold paragraph containing the actual answer string is explicitly excluded from this candidate pool.
• Disjoint (Random) Distractors: Formed by sampling length-matched fragments from entirely non-gold articles. To keep the structural and attention mechanics uniform, random passages are matched rank-by-rank to fall within a tight \(\pm20\%\) token length bound relative to their hard-negative counterparts.<br>
2.1 Compliance and Leak Filters
To preserve the integrity of our primary hypotheses, all selected distractors pass through a normalized answer-leak filter. If a sampled passage contains a substring matching the normalized target gold answer, it is culled automatically to avoid accidental answer exposure outside the gold context. For every question, the final context is assembled across \(3\) independent Monte Carlo seeds, logging the exact permutation and positioning of the gold passage relative to the distractors.

3. Hardware Layer and Quantization Baseline
To eliminate cross-backend variance and kernel execution differences, all production numbers are extracted from a single local hardware runtime environment:
• Hardware: Apple MacBook Pro featuring an Apple M4 Pro chip with 24 GB of unified memory.
• Quantization: All evaluation models are run using a pinned, identical 4-bit quantization layout.
• Decoding Strategy: Models are evaluated using greedy decoding with a max generation budget constrained to \(32\) tokens, passing through regex token parsers to standardize accuracy metrics.
