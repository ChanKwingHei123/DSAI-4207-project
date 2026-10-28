# Does Irrelevant Context Hurt Small RAG Readers? 
### A Controlled Dose-Response Study of Random vs. Topically Related Passages

This repository forms the complete data apparatus and evaluation architecture for the **DSAI 4207 (HKPolyU)** project. It investigates the quantitative sensitivity of small RAG readers under varying volumetric doses of topically related (hard) vs. disjoint (random) distractor contexts.

---

## 👥 Team & Roles
*   **Muhammad (Lead)**: Experimental design, annotation rubrics, MacBook inference execution, and quantitative failure analysis sections.
*   **Kwing Hei**: Data apparatus engineering, FAISS indexing, random/hard distractor samplers, answer-leak isolation filter, unified resumable harness, and non-parametric paired bootstrap framework.

---

## 🛠️ Repository Structure
```text
.
├── configs/                  # Quantization manifests and Mac runtime specs
├── data/
│   ├── dense_index_db/       # Workspace for frozen inputs and manifest ledger
│   └── audit/                # Double-blind ambiguity annotations and Kappa reports
├── report/                   # Report drafts, LaTeX/Markdown sources, and diagrams
├── results/                  # Inference outputs (JSONL) and summary matrices
├── src/
│   ├── retrieval/            # FAISS building and hard-negative query scripts
│   ├── distractors/          # Length-matched sampler and answer leak filters
│   ├── harness/              # Resumable production harness engine
│   ├── metrics/              # SQuAD evaluation, Bootstrap CI, and Holm correction
│   └── analysis/             # Dose-response curve plotting pipelines
└── requirements.txt          # Frozen environment specs
```

---

## 🚀 Setup & Installation

1. **Clone the repository cleanly:**
   ```bash
   git clone <repository-url>
   cd rag-dose-response
   ```

2. **Install pinned environmental dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📋 Core Execution Pipeline

### Phase 1: Input Fetch & Deterministic Data Split (Gate G2 Freeze)
Downloads the official SQuAD 2.0 development set, isolates answerable items, shuffles using random seed `42`, and locks down the 50-item pilot split and 500-item main evaluation grid with SHA-256 hashes.
```bash
python scripts/generate_splits.py
```

### Phase 2: Paragraph Footprints & FAISS Dense Indexing
Constructs the vector search array over the text fragments using the sentence encoder:
```bash
python src/retrieval/build_index.py
```

### Phase 3: The Resumable Grid Evaluation Sweep
Executes the comprehensive batch matrix mapping across **10 conditions** and **3 experimental seeds** (39,000 total inference generations). The execution tracks line-by-line using content-hashed `UUID5` tokens. If an interruption occurs on the MacBook Pro M4 Pro, simply rerun the script; it will detect completed blocks from `pipeline_run_logs.jsonl` and skip them to prevent compute waste.
```bash
python src/harness/runner.py
```

### Phase 4: Non-Parametric Bootstrap & Holm-Bonferroni Sign-Off
Computes exact academic SQuAD EM/F1 metric configurations. Evaluates paired distributions under 10,000 bootstrap resamples and applies multi-hypothesis step-down corrections to handle familywise error bounds before outputting publication-grade curves.
```bash
python src/metrics/bootstrap_stats.py
```

---

## 🔒 Rigorous Verification Constraints
*   **Rule R1 (Pre-registration Freeze)**: All analytical hypothesis pipelines must be committed prior to main grid inference tracking.
*   **Rule R2 (Leak & Ambiguity Pruning)**: All distractors pass a normalized substring compliance check. Flagged items containing targets are automatically culled.
*   **Rule R3 (Single Mac Runtime)**: All production numbers are extracted from the pinned 24GB M4 Pro setup to eliminate cross-backend variance.
