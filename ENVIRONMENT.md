# Project Environment Guide (`rag`)

## Rule (AGENTS.md Hard Rule 10)
**Every project command must run inside the dedicated `rag` conda environment.**
Never use the system Python (`/usr/bin/python3`) or Homebrew Python (`/opt/homebrew/bin/python3`). Never run `pip install` outside the `rag` environment.

---

## 1. Create the Environment

### From `environment.yml` + Mac-only extras (Apple Silicon `arm64`)
```bash
conda env create -f environment.yml
conda activate rag
python -m pip install -r requirements-mac.txt
```

### Manual Creation (Conda-Forge First, Pip Last)
```bash
conda create -n rag python=3.12 -y
conda activate rag
conda install -c conda-forge numpy pandas scipy scikit-learn pyyaml tqdm pytest matplotlib faiss-cpu -y
python -m pip install torch transformers sentence-transformers mlx mlx-lm
```

---

## 2. Activate the Environment (Run in Every Terminal)

```bash
conda activate rag
```

*(For non-interactive single commands, you may also use `/Users/muhammadhassan/miniforge3/bin/conda run -n rag <command>`.)*

---

## 3. Verify the Environment

```bash
conda activate rag
python -c "
import sys, platform, importlib.metadata
import mlx.core as mx
from sentence_transformers import SentenceTransformer

print('Python:', sys.version.split()[0], '| Arch:', platform.machine())
pkgs = [
    'numpy', 'pandas', 'scipy', 'scikit-learn', 'pyyaml',
    'tqdm', 'pytest', 'matplotlib', 'faiss-cpu',
    'torch', 'transformers', 'sentence-transformers',
    'mlx', 'mlx-lm',
]
for p in pkgs:
    print(f'{p}=={importlib.metadata.version(p)}')

print('MLX default device:', mx.default_device())
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
emb = model.encode(['Controlled dose-response study of RAG readers.'])
print('Embedding shape:', emb.shape)
"
```
