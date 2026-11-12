"""Production MLX inference backend (P4b) for the DSAI4207 RAG study.

Reuses the validated loading, chat-template, and deterministic greedy decoding logic
from `scripts/validate_models.py` (P4a) behind the `src.harness.Backend` interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import os
from pathlib import Path
import sys
import time
from typing import Any

# Enforce offline execution before any HF / MLX imports (Hard Rule 4)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hard Rule 11: Import omp_guard before torch, transformers, sentence_transformers, or mlx_lm
import src.omp_guard  # noqa: F401, E402
from src.harness import (  # noqa: E402
    DEFAULT_RUNTIME_PATH,
    Backend,
    load_runtime_config,
)

SUPPORTED_MLX_MODELS = ("Qwen3-8B", "Qwen3-4B", "Llama-3.1-8B-Instruct")

# Singleton slot holding at most one resident MLX model + tokenizer at a time (<= 6 GB RAM).
_ACTIVE_MODEL_STATE: dict[str, Any] = {
    "model_name": None,
    "mlx_dir": None,
    "model": None,
    "tokenizer": None,
}


def _clear_mlx_cache() -> None:
    """Release unused Metal buffer cache if MLX is imported."""
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def unload_active_mlx_model() -> None:
    """Unload any currently resident MLX model and free Metal memory cache."""
    had_model = _ACTIVE_MODEL_STATE["model"] is not None or _ACTIVE_MODEL_STATE["tokenizer"] is not None
    _ACTIVE_MODEL_STATE["model_name"] = None
    _ACTIVE_MODEL_STATE["mlx_dir"] = None
    _ACTIVE_MODEL_STATE["model"] = None
    _ACTIVE_MODEL_STATE["tokenizer"] = None
    if had_model:
        gc.collect()
        _clear_mlx_cache()


def get_active_mlx_model_name() -> str | None:
    """Return the name of the currently loaded MLX model, or None if no model is resident."""
    name = _ACTIVE_MODEL_STATE.get("model_name")
    return str(name) if name is not None else None


class MLXBackend(Backend):
    """Production MLX backend resolving model paths and metadata strictly from `configs/runtime.yaml`."""

    def __init__(
        self,
        model_name: str = "Qwen3-8B",
        *,
        repo_root: Path = REPO_ROOT,
        runtime_path: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runtime_path = (
            Path(runtime_path).resolve()
            if runtime_path is not None
            else (self.repo_root / "configs" / "runtime.yaml").resolve()
        )
        if not self.runtime_path.exists() and DEFAULT_RUNTIME_PATH.exists():
            self.runtime_path = DEFAULT_RUNTIME_PATH.resolve()

        runtime_cfg = load_runtime_config(self.runtime_path)
        models_cfg = runtime_cfg.get("models", {})
        if model_name not in SUPPORTED_MLX_MODELS or model_name not in models_cfg:
            raise ValueError(
                f"Unsupported model {model_name!r}; expected one of {SUPPORTED_MLX_MODELS} "
                f"configured in {self.runtime_path}."
            )

        model_info = models_cfg[model_name]
        mlx_rel_path = str(model_info["mlx_path"])
        candidate_dir = (self.repo_root / mlx_rel_path).resolve()
        if not candidate_dir.exists() and (REPO_ROOT / mlx_rel_path).exists():
            candidate_dir = (REPO_ROOT / mlx_rel_path).resolve()

        self.model_name = model_name
        self.backend_name = "mlx"
        self.is_stub = False
        self.hf_repo_id = str(model_info["hf_repo_id"])
        self.commit_sha = str(model_info["commit_sha"])
        self.mlx_rel_path = mlx_rel_path
        self.mlx_dir = candidate_dir
        self.model_id = f"{self.hf_repo_id}@{self.commit_sha}"
        self.enable_thinking: bool | None = (
            False if self.model_name.startswith("Qwen3") else None
        )

        # Last-call generation timing metrics for `run_unit`
        self.last_generation_seconds: float | None = None
        self.last_generated_tokens: int | None = None
        self.last_tokens_per_second: float | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load `(model, tokenizer)` on demand, evicting any other resident model first."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if not self.mlx_dir.exists():
            raise FileNotFoundError(
                f"Missing MLX model directory for {self.model_name}: {self.mlx_dir}"
            )

        target_dir_str = str(self.mlx_dir)
        if (
            _ACTIVE_MODEL_STATE["mlx_dir"] == target_dir_str
            and _ACTIVE_MODEL_STATE["model"] is not None
            and _ACTIVE_MODEL_STATE["tokenizer"] is not None
        ):
            return _ACTIVE_MODEL_STATE["model"], _ACTIVE_MODEL_STATE["tokenizer"]

        # Unload any previously resident model so peak memory stays <= 6 GB
        unload_active_mlx_model()

        from mlx_lm import load

        model, tokenizer = load(target_dir_str)
        _ACTIVE_MODEL_STATE["model_name"] = self.model_name
        _ACTIVE_MODEL_STATE["mlx_dir"] = target_dir_str
        _ACTIVE_MODEL_STATE["model"] = model
        _ACTIVE_MODEL_STATE["tokenizer"] = tokenizer
        return model, tokenizer

    def unload(self) -> None:
        """Unload this backend's model if it is currently resident."""
        if _ACTIVE_MODEL_STATE.get("mlx_dir") == str(self.mlx_dir):
            unload_active_mlx_model()

    def close(self) -> None:
        """Alias for `unload()`."""
        self.unload()

    def _count_tokens(self, text: str, tokenizer: Any) -> int:
        """Count tokens in `text` using the model tokenizer (without special tokens)."""
        if not text:
            return 0
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(text)
        return max(1, len(token_ids))

    def format_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Apply the model tokenizer's chat template with `add_generation_prompt=True`.

        For Qwen3 models, passes `enable_thinking=False`. Rejects any custom system prompt.
        """
        if not messages:
            raise ValueError("messages must not be empty.")
        for msg in messages:
            if msg.get("role") == "system":
                raise ValueError("System prompt is forbidden by study design.")
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError(
                f"Expected single user message, got roles {[m.get('role') for m in messages]!r}"
            )

        _, tokenizer = self._ensure_loaded()
        chat_messages = [{"role": "user", "content": str(messages[0]["content"])}]
        chat_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.model_name.startswith("Qwen3"):
            chat_kwargs["enable_thinking"] = False

        rendered = tokenizer.apply_chat_template(chat_messages, **chat_kwargs)
        return str(rendered)

    def generate(
        self,
        prompt_texts: Sequence[str],
        max_new_tokens: int = 32,
        greedy: bool = True,
    ) -> list[str]:
        """Run deterministic greedy decoding (`temp=0.0`) up to `max_new_tokens` via `mlx_lm`."""
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if not greedy:
            raise ValueError("Study design requires greedy=True decoding.")

        model, tokenizer = self._ensure_loaded()

        import mlx.core as mx
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.0)
        outputs: list[str] = []
        total_gen_seconds = 0.0
        total_gen_tokens = 0

        for prompt_text in prompt_texts:
            t0 = time.perf_counter()
            out = mlx_generate(
                model,
                tokenizer,
                prompt=str(prompt_text),
                max_tokens=int(max_new_tokens),
                sampler=sampler,
                verbose=False,
            )
            mx.eval()
            elapsed = max(1e-9, time.perf_counter() - t0)
            out_str = str(out)
            outputs.append(out_str)
            total_gen_seconds += elapsed
            total_gen_tokens += self._count_tokens(out_str, tokenizer)

        self.last_generation_seconds = max(1e-9, total_gen_seconds)
        self.last_generated_tokens = max(1, total_gen_tokens) if outputs else 0
        self.last_tokens_per_second = (
            float(self.last_generated_tokens / self.last_generation_seconds)
            if outputs
            else 0.0
        )
        return outputs
