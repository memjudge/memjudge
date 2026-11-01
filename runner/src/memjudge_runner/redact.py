"""Keep users' API keys out of logs and published records.

Vendor SDKs can echo request data in exception text and debug output. Jobs pass every
string they write (stdout, stderr, logging) through ``redact``, and finish redacts the
retrieval records before they are judged and published.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from typing import TextIO

REDACTED = "[redacted]"
MIN_SECRET_LENGTH = 8
"""Shorter values would redact ordinary words; real API keys are much longer."""


def secret_values(values: Iterable[str]) -> tuple[str, ...]:
    """Longest first, so a key that contains another is replaced whole."""
    return tuple(sorted({v for v in values if len(v) >= MIN_SECRET_LENGTH}, key=len, reverse=True))


def redact(text: str, secrets: Iterable[str]) -> str:
    for value in secret_values(secrets):
        text = text.replace(value, REDACTED)
    return text


class RedactingStream:
    """A text stream that redacts secrets before writing to the wrapped stream."""

    def __init__(self, inner: TextIO, secrets: Iterable[str]) -> None:
        self._inner = inner
        self._secrets = secret_values(secrets)

    def write(self, text: str) -> int:
        self._inner.write(redact(text, self._secrets))
        return len(text)

    def flush(self) -> None:
        self._inner.flush()

    def isatty(self) -> bool:
        return False


def shown(record: logging.LogRecord) -> bool:
    """What the run's log shows (ADR 0016): MemJudge's own lines, SDK retries (the
    ``<sdk>._base_client`` loggers) and every library's warnings and errors. Libraries'
    lines per HTTP request or per stored item would bury the rest."""
    return (
        record.levelno >= logging.WARNING
        or record.name.split(".")[0] in ("memjudge", "memjudge_runner")
        or record.name.endswith("._base_client")
    )


def install(secrets: Iterable[str]) -> None:
    """Redact ``secrets`` from everything this process prints or logs from now on."""
    values = secret_values(secrets)
    sys.stdout = RedactingStream(sys.stdout, values)
    sys.stderr = RedactingStream(sys.stderr, values)
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(shown)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
