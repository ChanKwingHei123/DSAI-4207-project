# Project Environment Guide (`rag`)

## Rule (AGENTS.md Hard Rule 10)
**Every project command must run inside the dedicated `rag` conda environment.**
Never use the system Python (`/usr/bin/python3`) or Homebrew Python (`/opt/homebrew/bin/python3`). Never run `pip install` outside the `rag` environment.

---

## 1. Create / Install the Environment

### A. On the Lead's MacBook (Apple Silicon `arm64` macOS)
Create or activate the `rag` environment (`python=3.12`), install conda-forge packages first, and install both core and Mac-only (`mlx`, `mlx-lm`) pinned requirements:
```bash
conda create -n rag python=3.12 -y
conda activate rag
conda install -c conda-forge numpy=2.5.4 pandas=3.1.1 scipy=1.18.1 scikit-learn=1.9.1 pyyaml=6.0.3 tqdm=4.70.1 pytest=9.2.0 matplotlib=3.10.9 faiss-cpu=1.14.3 -y
python -m pip install -r requirements.txt -r requirements-mac.txt
```

### B. On Any Other Machine (Cross-Platform Core Only)
Use `environment.yml` or `requirements.txt` only (excludes Mac-only `mlx` and `mlx-lm`):
```bash
# Option 1: via environment.yml
conda env create -f environment.yml
conda activate rag

# Option 2: via requirements.txt inside an activated Python 3.12 `rag` conda environment
conda create -n rag python=3.12 -y
conda activate rag
python -m pip install -r requirements.txt
```

---

## 2. Activate the Environment (Run in Every Terminal)

```bash
conda activate rag
```

*(For non-interactive single commands, you may also use `/Users/muhammadhassan/miniforge3/bin/conda run -n rag <command>`.)*

---

## 3. Verify the Environment

Run the repository verification script and dependency health check inside `rag`:
```bash
conda activate rag
python scripts/verify_env.py
python -m pip check
```
`scripts/verify_env.py` verifies:
1. Python minor version is `3.12`.
2. Every pinned package in `requirements.txt` matches its exact installed version (plus `requirements-mac.txt` when running on `arm64` macOS).
3. `mx.default_device()` is the GPU (`Device(gpu, 0)`) when `mlx` is present on `arm64` macOS.
4. A one-sentence embedding from `sentence-transformers/all-MiniLM-L6-v2` has shape `(1, 384)` and L2 norm `1.0`.

## 4. OpenMP Guard (AGENTS.md Hard Rule 11)
Import omp_guard before torch, transformers or sentence_transformers.
(In `scripts/` or `src/`, ensure the repository root is on `sys.path` and run `import src.omp_guard` before importing `torch`, `transformers`, or `sentence_transformers`; never set `KMP_DUPLICATE_LIB_OK`.)
