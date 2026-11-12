"""Inference backends for the DSAI4207 RAG irrelevant-context study."""

from __future__ import annotations

import src.omp_guard  # noqa: F401
from src.backends.mlx_backend import MLXBackend, unload_active_mlx_model

__all__ = ["MLXBackend", "unload_active_mlx_model"]
