"""Inference harness for the DSAI4207 RAG irrelevant-context study (P3).

Implements:
1. Prompt renderer reading `configs/prompt_v1.yaml` (never editing template strings) for
   closed-book and RAG conditions (`n=0` oracle and `n in {1, 2, 4, 8}` distractors),
   producing `[{"role": "user", "content": text}]` with no system prompt.
2. Backend interface (`format_prompt`, `generate`) and deterministic `StubBackend` for tests.
3. Answer extractor loading `configs/extraction_v1.yaml`.
4. Run unit execution `(model, condition, distractor_type, n, seed, split)`, append-only
   JSONL writing, resume by `qid`, `LEDGER.csv` append, and strict output-directory guardrails
   ensuring stub and dry-run outputs are never written under `results/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.omp_guard  # noqa: F401, E402

logger = logging.getLogger(__name__)

DEFAULT_PREREG_PATH = REPO_ROOT / "configs" / "prereg.yaml"
DEFAULT_PROMPT_PATH = REPO_ROOT / "configs" / "prompt_v1.yaml"
DEFAULT_EXTRACTION_PATH = REPO_ROOT / "configs" / "extraction_v1.yaml"
DEFAULT_RUNTIME_PATH = REPO_ROOT / "configs" / "runtime.yaml"
DEFAULT_RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "dev-v2.0.json"
DEFAULT_INPUTS_DIR = REPO_ROOT / "data" / "inputs"
DEFAULT_SAMPLING_DIR = REPO_ROOT / "data" / "sampling"
DEFAULT_FROZEN_DIR = REPO_ROOT / "data" / "frozen"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_STUB_SCRATCH_DIR = REPO_ROOT / "scratch" / "stub"
DEFAULT_DRYRUN_SCRATCH_DIR = REPO_ROOT / "scratch" / "dryrun"

VALID_SPLITS = ("main", "pilot", "reserve")
VALID_CONDITIONS = ("closed_book", "oracle", "rag")
VALID_DISTRACTOR_TYPES = ("hard_negative", "random_length_matched")

DOCUMENTED_INPUT_ROW_FIELDS = frozenset(
    {
        "question_id",
        "split",
        "seed",
        "article_index",
        "article_title",
        "question_text",
        "gold_answers",
        "gold_paragraph_id",
        "gold_score",
        "hard_pool_paragraph_ids",
        "hard_paragraph_ids",
        "hard_candidate_scores",
        "random_paragraph_ids",
        "random_length_match_windows",
        "p",
        "gold_positions",
        "token_lengths",
        "rejection_counts",
    }
)


def sha256_bytes(data: bytes) -> str:
    """Compute hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Compute hex SHA-256 of a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    return sha256_bytes(path.read_bytes())


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML file and validate that the root is a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_prompt_config(path: Path = DEFAULT_PROMPT_PATH) -> dict[str, Any]:
    """Load `configs/prompt_v1.yaml` without altering template strings."""
    return load_yaml_dict(path)


def load_extraction_config(path: Path = DEFAULT_EXTRACTION_PATH) -> dict[str, Any]:
    """Load `configs/extraction_v1.yaml`."""
    return load_yaml_dict(path)


def load_prereg_config(path: Path = DEFAULT_PREREG_PATH) -> dict[str, Any]:
    """Load `configs/prereg.yaml`."""
    return load_yaml_dict(path)


def load_runtime_config(path: Path = DEFAULT_RUNTIME_PATH) -> dict[str, Any]:
    """Load `configs/runtime.yaml`."""
    return load_yaml_dict(path)


def compute_study_hashes(
    repo_root: Path = REPO_ROOT,
    *,
    prereg_path: Path | None = None,
    prompt_path: Path | None = None,
    extraction_path: Path | None = None,
    runtime_path: Path | None = None,
    split: str | None = None,
) -> dict[str, str]:
    """Compute `config_hash`, `prereg_hash`, and `manifest_hash` for result rows.

    For `split == "pilot"` (or `"reserve"`), uses `data/sampling/manifest.json` when
    present. For `split == "main"` (or when `split` is omitted), prefers
    `data/frozen/manifest.json` when present and falls back to `data/sampling/manifest.json`.
    """
    p_prereg = prereg_path or (repo_root / "configs" / "prereg.yaml")
    p_prompt = prompt_path or (repo_root / "configs" / "prompt_v1.yaml")
    p_extract = extraction_path or (repo_root / "configs" / "extraction_v1.yaml")
    p_runtime = runtime_path or (repo_root / "configs" / "runtime.yaml")

    prereg_hash = sha256_file(p_prereg)

    config_parts: list[str] = [f"prompt_v1:{sha256_file(p_prompt)}"]
    if p_extract.exists():
        config_parts.append(f"extraction_v1:{sha256_file(p_extract)}")
    if p_runtime.exists():
        config_parts.append(f"runtime:{sha256_file(p_runtime)}")
    config_hash = sha256_text("|".join(config_parts))

    frozen_manifest = repo_root / "data" / "frozen" / "manifest.json"
    sampling_manifest = repo_root / "data" / "sampling" / "manifest.json"
    if split in ("pilot", "reserve"):
        if sampling_manifest.exists():
            manifest_hash = sha256_file(sampling_manifest)
        elif frozen_manifest.exists():
            manifest_hash = sha256_file(frozen_manifest)
        else:
            raise FileNotFoundError(
                f"Neither {sampling_manifest} nor {frozen_manifest} exists to compute manifest_hash."
            )
    else:
        if frozen_manifest.exists():
            manifest_hash = sha256_file(frozen_manifest)
        elif sampling_manifest.exists():
            manifest_hash = sha256_file(sampling_manifest)
        else:
            raise FileNotFoundError(
                f"Neither {frozen_manifest} nor {sampling_manifest} exists to compute manifest_hash."
            )

    return {
        "config_hash": config_hash,
        "prereg_hash": prereg_hash,
        "manifest_hash": manifest_hash,
    }


# ---------------------------------------------------------------------------
# 1. Prompt Renderer
# ---------------------------------------------------------------------------


def render_closed_book_text(
    question: str,
    prompt_cfg: Mapping[str, Any] | None = None,
) -> str:
    """Render the closed-book prompt body using `configs/prompt_v1.yaml`."""
    cfg = prompt_cfg if prompt_cfg is not None else load_prompt_config()
    template = cfg["closed_book"]
    return template.format(q=question)


def render_rag_text(
    passages: Sequence[str],
    question: str,
    prompt_cfg: Mapping[str, Any] | None = None,
) -> str:
    """Render the RAG prompt body using `header`, `passage_line` (i from 1), and `footer`."""
    if not passages:
        raise ValueError("RAG prompt rendering requires at least one passage.")
    cfg = prompt_cfg if prompt_cfg is not None else load_prompt_config()
    rag_cfg = cfg["rag"]
    header = rag_cfg["header"]
    passage_line = rag_cfg["passage_line"]
    footer = rag_cfg["footer"]

    rendered_passages = "".join(
        passage_line.format(i=idx, text=text)
        for idx, text in enumerate(passages, start=1)
    )
    return header + rendered_passages + footer.format(q=question)


def build_messages(user_text: str) -> list[dict[str, str]]:
    """Wrap rendered prompt text into a single-turn user message list (no system prompt)."""
    return [{"role": "user", "content": user_text}]


def load_paragraph_lookup(raw_path: Path = DEFAULT_RAW_DATA_PATH) -> dict[tuple[int, int], str]:
    """Load `(article_index, paragraph_index) -> context` mapping from SQuAD 2.0 dev JSON."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing SQuAD 2.0 raw file: {raw_path}")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    articles = payload.get("data", [])
    lookup: dict[tuple[int, int], str] = {}
    for art_idx, article in enumerate(articles):
        for para_idx, para in enumerate(article["paragraphs"]):
            lookup[(int(art_idx), int(para_idx))] = str(para["context"])
    return lookup


def validate_documented_input_row(row: Mapping[str, Any]) -> None:
    """Verify that an input row contains required documented fields from `docs/schema_inputs.md`."""
    required = {
        "question_id",
        "split",
        "question_text",
        "gold_paragraph_id",
    }
    missing = required - set(row.keys())
    if missing:
        raise KeyError(f"Input row missing required documented fields: {sorted(missing)}")


def assemble_rag_passage_ids(
    input_row: Mapping[str, Any],
    condition: str,
    distractor_type: str | None,
    n: int,
) -> list[tuple[int, int]]:
    """Construct the ordered list of `(article_index, paragraph_index)` for a RAG prompt.

    Uses only documented fields from `docs/schema_inputs.md`:
    - `gold_paragraph_id`
    - `hard_paragraph_ids` or `random_paragraph_ids` (nested prefix `[:n]`)
    - `gold_positions[str(n)]` (0-based insertion index `0..n`)
    """
    validate_documented_input_row(input_row)
    gold_raw = input_row["gold_paragraph_id"]
    gold_pid = (int(gold_raw[0]), int(gold_raw[1]))

    if condition == "oracle" or n == 0:
        if n != 0:
            raise ValueError(f"Oracle condition requires n=0, got n={n}")
        return [gold_pid]

    if condition != "rag":
        raise ValueError(f"Unsupported RAG condition {condition!r}")
    if n <= 0:
        raise ValueError(f"Distractor condition requires n > 0, got n={n}")

    if distractor_type == "hard_negative":
        raw_distractors = input_row["hard_paragraph_ids"]
    elif distractor_type == "random_length_matched":
        raw_distractors = input_row["random_paragraph_ids"]
    else:
        raise ValueError(f"Unsupported distractor_type {distractor_type!r}")

    if n > len(raw_distractors):
        raise ValueError(
            f"Requested n={n} distractors, but row only has {len(raw_distractors)}."
        )

    prefix: list[tuple[int, int]] = [
        (int(pid[0]), int(pid[1])) for pid in raw_distractors[:n]
    ]
    gold_positions = input_row["gold_positions"]
    if str(n) not in gold_positions:
        raise KeyError(f"gold_positions missing key {str(n)!r} in row {input_row['question_id']}")
    gold_pos = int(gold_positions[str(n)])
    if not (0 <= gold_pos <= n):
        raise ValueError(f"Invalid gold position {gold_pos} for n={n}")

    return prefix[:gold_pos] + [gold_pid] + prefix[gold_pos:]


def render_unit_messages(
    input_row: Mapping[str, Any],
    condition: str,
    distractor_type: str | None,
    n: int,
    paragraph_lookup: Mapping[tuple[int, int], str] | None = None,
    prompt_cfg: Mapping[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Render `(user_text, messages)` for any condition (`closed_book`, `oracle`, or `rag`)."""
    validate_documented_input_row(input_row)
    question = str(input_row["question_text"])

    if condition == "closed_book":
        user_text = render_closed_book_text(question, prompt_cfg=prompt_cfg)
        return user_text, build_messages(user_text)

    if paragraph_lookup is None:
        raise ValueError("paragraph_lookup is required for oracle and rag conditions.")

    pids = assemble_rag_passage_ids(
        input_row=input_row,
        condition=condition,
        distractor_type=distractor_type,
        n=n,
    )
    passages: list[str] = []
    for pid in pids:
        if pid not in paragraph_lookup:
            raise KeyError(f"Paragraph ID {pid} not found in paragraph_lookup.")
        passages.append(paragraph_lookup[pid])

    user_text = render_rag_text(passages, question, prompt_cfg=prompt_cfg)
    return user_text, build_messages(user_text)


# ---------------------------------------------------------------------------
# 2. Backend Interface and Deterministic StubBackend
# ---------------------------------------------------------------------------


class Backend(ABC):
    """Abstract inference backend interface."""

    model_name: str
    backend_name: str
    is_stub: bool
    model_id: str

    @abstractmethod
    def format_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Apply the model's chat template to `messages` (for Qwen3 with thinking disabled)."""

    @abstractmethod
    def generate(
        self,
        prompt_texts: Sequence[str],
        max_new_tokens: int,
        greedy: bool = True,
    ) -> list[str]:
        """Generate one raw output string per formatted prompt in `prompt_texts`."""


class StubBackend(Backend):
    """Deterministic stub backend for tests and dry-runs ONLY.

    Applies a deterministic fake chat template (recording `enable_thinking=False` for Qwen3
    and verifying no system prompt is present) and produces deterministic outputs that are a
    pure function of the formatted prompt's SHA-256 hash.
    """

    def __init__(self, model_name: str = "Qwen3-8B") -> None:
        self.model_name = model_name
        self.backend_name = "stub"
        self.is_stub = True
        self.model_id = f"stub/{model_name}"
        self.enable_thinking = False if model_name.startswith("Qwen3") else None

    def format_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        if not messages:
            raise ValueError("messages must not be empty.")
        for msg in messages:
            if msg.get("role") == "system":
                raise ValueError("System prompt is forbidden by study design.")
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError(
                f"Expected single user message, got roles {[m.get('role') for m in messages]!r}"
            )
        content = str(messages[0]["content"])
        if self.model_name.startswith("Qwen3"):
            return (
                f"<|stub_chat|model={self.model_name}|enable_thinking=False|>\n"
                f"<|im_start|>user\n{content}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        return (
            f"<|stub_chat|model={self.model_name}|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def generate(
        self,
        prompt_texts: Sequence[str],
        max_new_tokens: int,
        greedy: bool = True,
    ) -> list[str]:
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if not greedy:
            raise ValueError("Study design requires greedy=True decoding.")
        outputs: list[str] = []
        for prompt_text in prompt_texts:
            digest = sha256_text(prompt_text)
            outputs.append(f"Answer: stub_{digest[:12]}")
        return outputs


# ---------------------------------------------------------------------------
# 3. Answer Extraction (`configs/extraction_v1.yaml`)
# ---------------------------------------------------------------------------


def extract_answer(
    raw_output: str,
    extraction_cfg: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    """Extract answer and validity flag according to `configs/extraction_v1.yaml`.

    Rules:
    1. Take the first non-empty line of `raw_output`.
    2. Strip a leading label `"Answer:"` or `"A:"` (case-insensitive).
    3. Answer is `valid` iff non-empty, at most 10 words, and not starting
       (case-insensitive) with any of the configured forbidden prefixes.
    """
    cfg = extraction_cfg if extraction_cfg is not None else load_extraction_config()

    first_line = ""
    if cfg.get("first_non_empty_line", True):
        for line in raw_output.splitlines():
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
    else:
        first_line = raw_output.strip()

    candidate = first_line
    labels: Sequence[str] = cfg.get("strip_leading_labels", ["Answer:", "A:"])
    case_insensitive = bool(cfg.get("label_case_insensitive", True))

    for label in labels:
        if case_insensitive:
            if candidate.lower().startswith(label.lower()):
                candidate = candidate[len(label) :].strip()
                break
        else:
            if candidate.startswith(label):
                candidate = candidate[len(label) :].strip()
                break

    max_words = int(cfg.get("max_words", 10))
    require_non_empty = bool(cfg.get("require_non_empty", True))
    invalid_prefixes: Sequence[str] = cfg.get(
        "invalid_starting_phrases_case_insensitive", []
    )

    if require_non_empty and not candidate:
        return candidate, False

    words = candidate.split()
    if len(words) > max_words:
        return candidate, False

    lowered = candidate.lower()
    for prefix in invalid_prefixes:
        if lowered.startswith(prefix.lower()):
            return candidate, False

    return candidate, True


# ---------------------------------------------------------------------------
# 4. Run Unit Specification, Path Guardrails, Resume, and Ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunUnit:
    """Canonical run unit key `(model, condition, distractor_type, n, seed, split)`."""

    model: str
    condition: str
    distractor_type: str | None
    n: int
    seed: int | None
    split: str

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {self.split!r}; expected one of {VALID_SPLITS}")
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid condition {self.condition!r}; expected one of {VALID_CONDITIONS}"
            )
        if self.condition in ("closed_book", "oracle"):
            if self.distractor_type is not None:
                raise ValueError(
                    f"Condition {self.condition!r} is seed/type-independent; "
                    f"expected distractor_type=None, got {self.distractor_type!r}"
                )
            if self.n != 0:
                raise ValueError(f"Condition {self.condition!r} requires n=0, got n={self.n}")
            if self.seed is not None:
                raise ValueError(
                    f"Condition {self.condition!r} is seed-independent; "
                    f"expected seed=None, got {self.seed!r}"
                )
        else:
            if self.distractor_type not in VALID_DISTRACTOR_TYPES:
                raise ValueError(
                    f"Condition 'rag' requires distractor_type in {VALID_DISTRACTOR_TYPES}, "
                    f"got {self.distractor_type!r}"
                )
            if self.n not in (1, 2, 4, 8):
                raise ValueError(f"Condition 'rag' requires n in (1, 2, 4, 8), got n={self.n}")
            if self.seed not in (0, 1, 2):
                raise ValueError(
                    f"Condition 'rag' requires seed in (0, 1, 2), got {self.seed!r}"
                )

    @property
    def is_seed_independent(self) -> bool:
        return self.condition in ("closed_book", "oracle")

    @property
    def unit_id(self) -> str:
        dtype_str = self.distractor_type if self.distractor_type is not None else "none"
        seed_str = str(self.seed) if self.seed is not None else "none"
        return (
            f"{self.model}__{self.condition}__{dtype_str}"
            f"__n{self.n}__seed{seed_str}__{self.split}"
        )

    @property
    def filename(self) -> str:
        return f"{self.unit_id}.jsonl"

    def output_path(self, base_results_dir: Path) -> Path:
        """Return the output JSONL file path (`pilot` units go under `<base_results_dir>/pilot/`)."""
        if self.split == "pilot":
            return base_results_dir / "pilot" / self.filename
        return base_results_dir / self.filename


def _normalize_condition_and_type(
    cond_raw: str,
    dtype_raw: str | None,
) -> tuple[str, str | None]:
    c = cond_raw.strip().lower().replace("-", "_")
    d = dtype_raw.strip().lower().replace("-", "_") if dtype_raw is not None else None

    if c in ("closed_book", "closedbook", "cb"):
        return "closed_book", None
    if c == "oracle":
        return "oracle", None

    type_aliases = {
        "hard": "hard_negative",
        "hard_negative": "hard_negative",
        "random": "random_length_matched",
        "random_length_matched": "random_length_matched",
    }
    if c in type_aliases and (d is None or d in ("none", "null", "") or d in type_aliases):
        return "rag", type_aliases[c]
    if c in ("rag", "distractor"):
        if d in type_aliases:
            return "rag", type_aliases[d]
        raise ValueError(f"Invalid distractor type {dtype_raw!r} for RAG unit.")
    raise ValueError(f"Unrecognized condition {cond_raw!r}")


def parse_run_unit(spec: str) -> RunUnit:
    """Parse a run unit string into a canonical `RunUnit`.

    Supported formats:
    1. 6-part double-underscore: `<model>__<condition>__<type>__n<N>__seed<S>__<split>`
       e.g. `Qwen3-8B__oracle__none__n0__seednone__pilot`,
            `Qwen3-8B__rag__hard_negative__n2__seed0__pilot`
    2. 5-part double-underscore: `<model>__<condition_or_type>__n<N>__seed<S>__<split>`
       e.g. `Qwen3-8B__hard_negative__n2__seed0__pilot`
    3. 3-part double-underscore (for closed_book / oracle): `<model>__<condition>__<split>`
       e.g. `Qwen3-8B__closed_book__pilot`, `Qwen3-8B__oracle__pilot`
    4. Colon-separated shorthand with 3, 5, or 6 parts.
    """
    cleaned = spec.strip()
    if cleaned.endswith(".jsonl"):
        cleaned = cleaned[: -len(".jsonl")]

    parts = cleaned.split("__") if "__" in cleaned else cleaned.split(":")
    parts = [p.strip() for p in parts if p.strip()]

    def _parse_n(token: str) -> int:
        t = token.lower()
        if t.startswith("n"):
            t = t[1:]
        if t in ("none", "null", ""):
            return 0
        return int(t)

    def _parse_seed(token: str) -> int | None:
        t = token.lower()
        if t.startswith("seed"):
            t = t[4:]
        if t in ("none", "null", ""):
            return None
        return int(t)

    if len(parts) == 6:
        model, cond_raw, dtype_raw, n_raw, seed_raw, split = parts
        condition, distractor_type = _normalize_condition_and_type(cond_raw, dtype_raw)
        n = _parse_n(n_raw)
        seed = _parse_seed(seed_raw)
        return RunUnit(
            model=model,
            condition=condition,
            distractor_type=distractor_type,
            n=n,
            seed=seed,
            split=split,
        )
    if len(parts) == 5:
        model, cond_or_type, n_raw, seed_raw, split = parts
        condition, distractor_type = _normalize_condition_and_type(cond_or_type, None)
        n = _parse_n(n_raw)
        seed = _parse_seed(seed_raw)
        return RunUnit(
            model=model,
            condition=condition,
            distractor_type=distractor_type,
            n=n,
            seed=seed,
            split=split,
        )
    if len(parts) == 3:
        model, cond_raw, split = parts
        condition, distractor_type = _normalize_condition_and_type(cond_raw, None)
        return RunUnit(
            model=model,
            condition=condition,
            distractor_type=distractor_type,
            n=0,
            seed=None,
            split=split,
        )

    raise ValueError(
        f"Cannot parse run unit specification {spec!r}. Expected "
        "'<model>__<condition>__<type>__n<N>__seed<S>__<split>'."
    )


def parse_units_list(raw_specs: Sequence[str] | str) -> list[RunUnit]:
    """Parse space- or comma-separated unit specifications into a list of `RunUnit`s."""
    if isinstance(raw_specs, str):
        tokens = [t.strip() for t in raw_specs.split(",") if t.strip()]
    else:
        tokens = []
        for item in raw_specs:
            for sub in item.split(","):
                if sub.strip():
                    tokens.append(sub.strip())
    if not tokens:
        raise ValueError("No run units specified.")
    return [parse_run_unit(tok) for tok in tokens]


def is_path_under_results(path: Path, repo_root: Path = REPO_ROOT) -> bool:
    """Return True if `path` resolves to `results/` (under `repo_root` or `REPO_ROOT`) or inside it."""
    resolved = (path if path.is_absolute() else (repo_root / path)).resolve()
    for root in (repo_root, REPO_ROOT):
        results_root = (root / "results").resolve()
        if resolved == results_root or results_root in resolved.parents:
            return True
    return False


def resolve_results_dir(
    requested_results_dir: Path | str | None = None,
    *,
    backend_name: str,
    is_stub: bool,
    dry_run: bool = False,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Resolve and validate the output directory according to Hard Rule 2 guardrails.

    Rules:
    - Stub (`backend == 'stub'` or `is_stub == True`) and `--dry-run` outputs must NEVER
      be written under `results/`. Any explicit `requested_results_dir` under `results/`
      is refused with a `ValueError`.
    - When `requested_results_dir` is omitted (`None`):
      - `dry_run=True` -> `repo_root / 'scratch' / 'dryrun'`
      - `is_stub=True` or `backend_name == 'stub'` -> `repo_root / 'scratch' / 'stub'`
      - official run -> `repo_root / 'results'`
    """
    non_official = bool(is_stub or backend_name.strip().lower() == "stub" or dry_run)

    if requested_results_dir is not None:
        req_path = Path(requested_results_dir)
        resolved = (req_path if req_path.is_absolute() else (repo_root / req_path)).resolve()
        if non_official and is_path_under_results(resolved, repo_root=repo_root):
            raise ValueError(
                f"Refusing to write stub or dry-run outputs under results/ "
                f"(requested={requested_results_dir!r}, backend={backend_name!r}, "
                f"is_stub={is_stub}, dry_run={dry_run}). Only official runs may write to results/."
            )
        return resolved

    if dry_run:
        return (repo_root / "scratch" / "dryrun").resolve()
    if non_official:
        return (repo_root / "scratch" / "stub").resolve()
    return (repo_root / "results").resolve()


def load_split_seed_rows(
    inputs_dir: Path,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Load rows from `data/inputs/<split>_seed<S>.jsonl`."""
    path = inputs_dir / f"{split}_seed{seed}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        validate_documented_input_row(row)
        rows.append(row)
    return rows


def load_seed_independent_rows(
    inputs_dir: Path,
    split: str,
    draw_seeds: Sequence[int] = (0, 1, 2),
) -> list[dict[str, Any]]:
    """Load unique questions appearing in at least one seed file of `split`.

    Preserves canonical question order across the seed files so each question is run
    at most once with `seed = null`.
    """
    by_qid: dict[str, dict[str, Any]] = {}
    seed_orders: list[list[str]] = []

    for s in draw_seeds:
        seed_path = inputs_dir / f"{split}_seed{s}.jsonl"
        if not seed_path.exists():
            continue
        qids_this_seed: list[str] = []
        for row in load_split_seed_rows(inputs_dir, split, s):
            qid = str(row["question_id"])
            qids_this_seed.append(qid)
            if qid not in by_qid:
                by_qid[qid] = row
        seed_orders.append(qids_this_seed)

    if not by_qid:
        raise FileNotFoundError(
            f"No input seed files found for split={split!r} in {inputs_dir}"
        )

    # Reconstruct canonical order: if a question in seed 1+ was absent from seed 0,
    # place it immediately after its predecessor in that seed file if possible.
    ordered_qids: list[str] = []
    seen: set[str] = set()
    if seed_orders:
        for qid in seed_orders[0]:
            ordered_qids.append(qid)
            seen.add(qid)
        for qids_list in seed_orders[1:]:
            for idx, qid in enumerate(qids_list):
                if qid in seen:
                    continue
                if idx > 0 and qids_list[idx - 1] in seen:
                    pred_pos = ordered_qids.index(qids_list[idx - 1])
                    ordered_qids.insert(pred_pos + 1, qid)
                else:
                    ordered_qids.append(qid)
                seen.add(qid)

    return [by_qid[qid] for qid in ordered_qids]


def read_existing_qids(unit_file: Path) -> set[str]:
    """Read already-recorded `qid` values from an existing unit JSONL file for resume."""
    if not unit_file.exists():
        return set()
    existing: set[str] = set()
    for line_no, line in enumerate(unit_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        qid = record.get("qid")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"Malformed row missing 'qid' in {unit_file}:{line_no}")
        existing.add(qid)
    return existing


def append_ledger_entry(
    ledger_path: Path,
    unit_file: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Append `(file, line_count, sha256)` for `unit_file` to `LEDGER.csv`."""
    if not unit_file.exists():
        raise FileNotFoundError(f"Cannot record non-existent result file in ledger: {unit_file}")

    raw_bytes = unit_file.read_bytes()
    line_count = len(raw_bytes.splitlines(keepends=True))
    file_sha256 = sha256_bytes(raw_bytes)

    try:
        rel_path = unit_file.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel_path = unit_file.name

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ledger_path.exists() or ledger_path.stat().st_size == 0

    with ledger_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["file", "line_count", "sha256"])
        writer.writerow([rel_path, line_count, file_sha256])

    return {
        "file": rel_path,
        "line_count": line_count,
        "sha256": file_sha256,
    }


def resolve_model_id(
    model_name: str,
    backend: Backend,
    runtime_cfg: Mapping[str, Any] | None = None,
) -> str:
    """Resolve `model_id` for a result row."""
    if backend.is_stub:
        return backend.model_id
    if getattr(backend, "model_id", None):
        return str(backend.model_id)
    if runtime_cfg is not None and "models" in runtime_cfg and model_name in runtime_cfg["models"]:
        info = runtime_cfg["models"][model_name]
        return f"{info['hf_repo_id']}@{info['commit_sha']}"
    return model_name


def resolve_split_input_sources(
    repo_root: Path,
    split: str,
    inputs_dir: Path | None = None,
) -> tuple[Path, Path | None, bool]:
    """Resolve `(inputs_dir, split_questions_json, is_frozen_main)` for `split`.

    When `inputs_dir` is omitted, `split == "main"` reads from `data/frozen/` if
    `data/frozen/manifest.json` exists, falling back to `data/inputs/` and
    `data/sampling/main.json` otherwise. `split == "pilot"` always reads from
    `data/inputs/` and `data/sampling/pilot.json`.
    """
    if inputs_dir is not None:
        return inputs_dir, None, False

    if split == "main" and (repo_root / "data" / "frozen" / "manifest.json").exists():
        frozen_dir = repo_root / "data" / "frozen"
        return frozen_dir, frozen_dir / "main.json", True

    sampling_file = repo_root / "data" / "sampling" / f"{split}.json"
    return (
        repo_root / "data" / "inputs",
        sampling_file if sampling_file.exists() else None,
        False,
    )


def run_unit(
    unit: RunUnit,
    backend: Backend,
    *,
    max_questions: int | None = None,
    results_dir: Path | str | None = None,
    dry_run: bool = False,
    repo_root: Path = REPO_ROOT,
    inputs_dir: Path | None = None,
    raw_data_path: Path | None = None,
    prereg_path: Path | None = None,
    prompt_path: Path | None = None,
    extraction_path: Path | None = None,
    runtime_path: Path | None = None,
    paragraph_lookup: Mapping[tuple[int, int], str] | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a single `RunUnit`, writing append-only JSONL rows and updating LEDGER when official."""
    if backend.model_name != unit.model:
        raise ValueError(
            f"Backend model_name={backend.model_name!r} does not match unit.model={unit.model!r}"
        )

    out_base_dir = resolve_results_dir(
        results_dir,
        backend_name=backend.backend_name,
        is_stub=backend.is_stub,
        dry_run=dry_run,
        repo_root=repo_root,
    )

    # Extra defense-in-depth: never allow stub or dry-run to write to repo_root/results
    if (backend.is_stub or backend.backend_name == "stub" or dry_run) and is_path_under_results(
        out_base_dir, repo_root=repo_root
    ):
        raise ValueError("Stub or dry-run output under results/ is strictly forbidden.")

    p_inputs, split_questions_file, using_frozen_main = resolve_split_input_sources(
        repo_root=repo_root,
        split=unit.split,
        inputs_dir=inputs_dir,
    )
    p_raw = raw_data_path or (repo_root / "data" / "raw" / "dev-v2.0.json")
    p_prereg = prereg_path or (repo_root / "configs" / "prereg.yaml")
    p_prompt = prompt_path or (repo_root / "configs" / "prompt_v1.yaml")
    p_extract = extraction_path or (repo_root / "configs" / "extraction_v1.yaml")
    p_runtime = runtime_path or (repo_root / "configs" / "runtime.yaml")

    prereg_cfg = load_prereg_config(p_prereg)
    prompt_cfg = load_prompt_config(p_prompt)
    extraction_cfg = load_extraction_config(p_extract)
    runtime_cfg = load_runtime_config(p_runtime) if p_runtime.exists() else {}

    hashes = compute_study_hashes(
        repo_root=repo_root,
        prereg_path=p_prereg,
        prompt_path=p_prompt,
        extraction_path=p_extract,
        runtime_path=p_runtime,
        split=unit.split,
    )

    max_new_tokens = int(prereg_cfg["decoding"]["max_new_tokens"])
    greedy = bool(prereg_cfg["decoding"]["greedy"])
    draw_seeds = [int(s) for s in prereg_cfg["rng"]["draw_seeds"]]

    if unit.condition != "closed_book" and paragraph_lookup is None:
        paragraph_lookup = load_paragraph_lookup(p_raw)

    if using_frozen_main:
        assert split_questions_file is not None
        if not split_questions_file.exists():
            raise FileNotFoundError(
                f"Frozen manifest exists, but {split_questions_file} is missing."
            )
        frozen_questions = json.loads(split_questions_file.read_text(encoding="utf-8"))
        frozen_qids = [str(item["question_id"]) for item in frozen_questions]
        if unit.is_seed_independent:
            candidate_rows = load_seed_independent_rows(
                p_inputs, unit.split, draw_seeds=draw_seeds
            )
        else:
            assert unit.seed is not None
            candidate_rows = load_split_seed_rows(p_inputs, unit.split, unit.seed)
        row_qids = [str(r["question_id"]) for r in candidate_rows]
        if row_qids != frozen_qids:
            raise ValueError(
                f"Frozen main inputs mismatch between {split_questions_file} "
                f"(n={len(frozen_qids)}) and {p_inputs} (n={len(row_qids)})"
            )
        skipped_missing_seed_rows = 0
    else:
        union_rows = load_seed_independent_rows(p_inputs, unit.split, draw_seeds=draw_seeds)
        union_count = len(union_rows)

        if inputs_dir is None and split_questions_file is not None:
            expected_split_total = len(
                json.loads(split_questions_file.read_text(encoding="utf-8"))
            )
        else:
            expected_split_total = union_count

        if unit.is_seed_independent:
            candidate_rows = union_rows
            skipped_missing_seed_rows = 0
        else:
            assert unit.seed is not None
            candidate_rows = load_split_seed_rows(p_inputs, unit.split, unit.seed)
            skipped_missing_seed_rows = max(0, expected_split_total - len(candidate_rows))
            if skipped_missing_seed_rows > 0:
                logger.info(
                    "Unit %s: skipped %d (question, seed=%d) pairs excluded by P2c (present=%d, split_total=%d).",
                    unit.unit_id,
                    skipped_missing_seed_rows,
                    unit.seed,
                    len(candidate_rows),
                    expected_split_total,
                )

    if max_questions is not None:
        if max_questions < 0:
            raise ValueError(f"max_questions must be non-negative, got {max_questions}")
        target_rows = candidate_rows[:max_questions]
    else:
        target_rows = candidate_rows

    unit_file = unit.output_path(out_base_dir)
    unit_file.parent.mkdir(parents=True, exist_ok=True)
    existing_qids = read_existing_qids(unit_file)

    rows_to_run = [r for r in target_rows if str(r["question_id"]) not in existing_qids]
    skipped_resume_count = len(target_rows) - len(rows_to_run)

    model_id = resolve_model_id(unit.model, backend, runtime_cfg=runtime_cfg)
    written_rows: list[dict[str, Any]] = []

    if rows_to_run:
        with unit_file.open("a", encoding="utf-8") as f:
            for input_row in rows_to_run:
                qid = str(input_row["question_id"])
                _, messages = render_unit_messages(
                    input_row=input_row,
                    condition=unit.condition,
                    distractor_type=unit.distractor_type,
                    n=unit.n,
                    paragraph_lookup=paragraph_lookup,
                    prompt_cfg=prompt_cfg,
                )
                formatted_prompt = backend.format_prompt(messages)
                prompt_sha = sha256_text(formatted_prompt)

                t0 = time.perf_counter()
                raw_outputs = backend.generate(
                    [formatted_prompt],
                    max_new_tokens=max_new_tokens,
                    greedy=greedy,
                )
                wall_elapsed = max(1e-9, time.perf_counter() - t0)
                if len(raw_outputs) != 1:
                    raise RuntimeError(
                        f"Backend.generate expected 1 output, got {len(raw_outputs)}"
                    )
                raw_output = str(raw_outputs[0])
                extracted_ans, is_valid = extract_answer(
                    raw_output, extraction_cfg=extraction_cfg
                )

                backend_gen_sec = getattr(backend, "last_generation_seconds", None)
                elapsed = (
                    max(1e-9, float(backend_gen_sec))
                    if isinstance(backend_gen_sec, (int, float)) and backend_gen_sec > 0
                    else wall_elapsed
                )
                backend_gen_toks = getattr(backend, "last_generated_tokens", None)
                num_tokens = (
                    max(1, int(backend_gen_toks))
                    if isinstance(backend_gen_toks, int) and backend_gen_toks > 0
                    else max(1, len(raw_output.split()))
                )
                tps = float(num_tokens / elapsed)

                result_row: dict[str, Any] = {
                    "qid": qid,
                    "model": unit.model,
                    "condition": unit.condition,
                    "distractor_type": unit.distractor_type,
                    "n": int(unit.n),
                    "seed": unit.seed,
                    "split": unit.split,
                    "backend": backend.backend_name,
                    "is_stub": bool(backend.is_stub),
                    "model_id": model_id,
                    "prompt_sha256": prompt_sha,
                    "raw_output": raw_output,
                    "extracted_answer": extracted_ans,
                    "valid": bool(is_valid),
                    "seconds": round(float(elapsed), 6),
                    "tokens_per_second": round(tps, 4),
                    "config_hash": hashes["config_hash"],
                    "prereg_hash": hashes["prereg_hash"],
                    "manifest_hash": hashes["manifest_hash"],
                }
                f.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                written_rows.append(result_row)

    is_official = (
        not backend.is_stub
        and backend.backend_name.strip().lower() != "stub"
        and not dry_run
    )
    ledger_entry: dict[str, Any] | None = None
    if is_official and is_path_under_results(out_base_dir, repo_root=repo_root):
        target_ledger = ledger_path or (repo_root / "results" / "LEDGER.csv")
        if unit_file.exists():
            ledger_entry = append_ledger_entry(
                ledger_path=target_ledger,
                unit_file=unit_file,
                repo_root=repo_root,
            )
    elif ledger_path is not None and is_official and unit_file.exists():
        ledger_entry = append_ledger_entry(
            ledger_path=ledger_path,
            unit_file=unit_file,
            repo_root=repo_root,
        )

    return {
        "unit": unit,
        "unit_id": unit.unit_id,
        "output_path": unit_file,
        "written_count": len(written_rows),
        "skipped_resume_count": skipped_resume_count,
        "skipped_p2c_count": skipped_missing_seed_rows,
        "total_target_count": len(target_rows),
        "ledger_entry": ledger_entry,
        "written_rows": written_rows,
    }


def load_result_rows(
    path_or_dir: Path,
    *,
    allow_stub: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    """Analysis loader for result JSONL files.

    Rejects any row with `is_stub == True` (or `backend == 'stub'`) when loading from
    `results/` or when `allow_stub=False`.
    """
    if not path_or_dir.exists():
        raise FileNotFoundError(f"Results path does not exist: {path_or_dir}")

    files = (
        sorted(path_or_dir.rglob("*.jsonl"))
        if path_or_dir.is_dir()
        else [path_or_dir]
    )

    rows: list[dict[str, Any]] = []
    for jsonl_file in files:
        under_results = is_path_under_results(jsonl_file, repo_root=repo_root)
        for line_no, raw_line in enumerate(
            jsonl_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_file}:{line_no}: expected JSON object")
            if (under_results or not allow_stub) and (
                record.get("is_stub") is True
                or str(record.get("backend", "")).strip().lower() == "stub"
            ):
                raise ValueError(
                    f"{jsonl_file}:{line_no}: stub row (is_stub=True) is forbidden in analysis/results."
                )
            rows.append(record)
    return rows


def __getattr__(name: str) -> Any:
    if name == "MLXBackend":
        from src.backends.mlx_backend import MLXBackend

        return MLXBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

