"""Deterministic question keys for within- and cross-split deduplication."""
from .common import normalize_text


def question_key(question: str) -> str:
    return normalize_text(question)
