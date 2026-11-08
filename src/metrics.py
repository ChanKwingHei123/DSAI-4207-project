"""Official SQuAD answer normalization, exact match, and token F1 metrics.

Implements Item 1 of P5a (metrics and statistics core):
- Official SQuAD v1.1/v2.0 normalization pipeline:
  lowercase -> remove punctuation -> remove articles (a, an, the) -> collapse whitespace.
- exact_match against ANY gold answer (max over gold answers).
- token_f1 (max over gold answers, where per-pair F1 = 2PR / (P + R) over token multisets).
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import re
import string

_ARTICLE_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION_SET = frozenset(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, English articles, and extra whitespace.

    Follows the official SQuAD evaluation script pipeline order:
    white_space_fix(remove_articles(remove_punc(lower(s)))).
    """

    def remove_articles(text: str) -> str:
        return re.sub(_ARTICLE_REGEX, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in _PUNCTUATION_SET)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list[str]:
    """Return normalized whitespace-delimited tokens for string `s`."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute exact match (0 or 1) between a single gold answer and prediction."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute token-level F1 = 2PR / (P + R) between a single gold answer and prediction."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(int(gold_toks == pred_toks))
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2.0 * precision * recall) / (precision + recall)


def _validate_gold_answers(gold_answers: Sequence[str]) -> None:
    if isinstance(gold_answers, str):
        raise TypeError(
            "gold_answers must be a sequence of answer strings, not a single str."
        )
    if len(gold_answers) == 0:
        raise ValueError("gold_answers must contain at least one gold answer.")


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    """Compute SQuAD-normalized exact match against ANY gold answer (max over gold answers)."""
    _validate_gold_answers(gold_answers)
    return float(max(compute_exact(gold, prediction) for gold in gold_answers))


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Compute SQuAD-normalized token F1 (max over gold answers)."""
    _validate_gold_answers(gold_answers)
    return float(max(compute_f1(gold, prediction) for gold in gold_answers))
