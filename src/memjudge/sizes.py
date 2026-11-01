"""The two result sizes every run scores: Top 10 and Top 30 (ADR 0018)."""

from __future__ import annotations

from typing import Literal

TOP_K: tuple[int, ...] = (10, 30)
"""Most items a connector may return per question, one retrieval each."""

SizeKey = Literal["top10", "top30"]


def size_key(top_k: int) -> SizeKey:
    """``top10`` or ``top30``: the size's key in results and file names."""
    return "top10" if top_k == 10 else "top30"


def questions_file(top_k: int) -> str:
    """The size's question records, e.g. ``questions-top10.jsonl``."""
    return f"questions-{size_key(top_k)}.jsonl"
