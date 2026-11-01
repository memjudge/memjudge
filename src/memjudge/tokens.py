"""Harness-side token counting with one fixed tokenizer for every system (ADR 0002)."""

from __future__ import annotations

from functools import cache
from typing import Any

TOKENIZER = "o200k_base"


@cache
def _encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding(TOKENIZER)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def truncate(text: str, max_tokens: int) -> tuple[str, int, bool]:
    """Cut ``text`` to at most ``max_tokens``. Returns (text, original_tokens, truncated)."""
    tokens = _encoding().encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text, len(tokens), False
    return _encoding().decode(tokens[:max_tokens]), len(tokens), True
