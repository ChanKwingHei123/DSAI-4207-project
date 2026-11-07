"""Centralised OpenMP import-order guard for macOS arm64 (`rag` conda environment).

Why this module exists:
Two distinct copies of `libomp.dylib` are present in the `rag` environment:
1. Conda-forge's `$CONDA_PREFIX/lib/libomp.dylib`, dynamically linked by `scipy`,
   `scikit-learn`, and `faiss-cpu`.
2. PyTorch's pip wheel copy (`site-packages/torch/lib/libomp.dylib`), dynamically
   linked by `torch`.

If `torch` is imported before `sklearn`, PyTorch's bundled `libomp.dylib`
initialises first. When `sentence_transformers` (or user code) subsequently imports
`sklearn` (which loads `sklearn.utils._openmp_helpers` linked against conda-forge's
`libomp.dylib`), LLVM's OpenMP runtime detects a duplicate initialisation and aborts
the process with:
  `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.`

Empirical determination:
Testing prefixes of `[numpy, scipy, scipy.sparse, sklearn, sklearn.utils]` in fresh
interpreters showed that `numpy`, `scipy`, and `scipy.sparse` alone do not initialise
`libomp` at import time (`rc = -6`, `OMP: Error #15`), whereas importing `sklearn`
(and `sklearn.utils`) triggers `sklearn.utils._openmp_helpers` and initialises the
conda-forge `libomp.dylib` before `torch` is loaded (`rc = 0`).

Rule (AGENTS.md Hard Rule 11):
- Any process that imports `torch`, `transformers`, or `sentence_transformers` MUST
  import `omp_guard` (`import src.omp_guard` or `import omp_guard`) BEFORE importing
  any of those packages.
- Never set `KMP_DUPLICATE_LIB_OK`.
- Keep embedding computation and FAISS search in separate OS processes.

Importing without installing the project:
- Scripts in `scripts/` prepend `REPO_ROOT = Path(__file__).resolve().parent.parent`
  to `sys.path` and run `import src.omp_guard  # noqa: F401` before any ML imports.
- Modules in `src/` ensure `REPO_ROOT` is on `sys.path` and run
  `import src.omp_guard  # noqa: F401`.
"""

import numpy
import scipy
import scipy.sparse
import sklearn
import sklearn.utils
