"""The run's log, for its owner on the /run page (ADR 0016).

The lines are what the run's job executions wrote, read back from Cloud Logging. The jobs
remove the users' keys from everything they write (redact.py), so these lines hold none.
Reads are cached per run: Cloud Logging allows 60 log reads a minute per project.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel

LOG_LINES = 200
"""The newest lines shown."""
LOG_CACHE_SECONDS = 15.0
EXECUTION_LABEL = 'labels."run.googleapis.com/execution_name"'


class LogLine(BaseModel):
    at: datetime
    job: Literal["retrieve", "finish"]
    text: str


class Logs(Protocol):
    def read(self, executions: list[str], since: datetime, limit: int) -> list[LogLine]:
        """The newest ``limit`` lines the executions wrote since ``since``, oldest first."""


def log_filter(executions: list[str], since: datetime) -> str:
    """Cloud Logging filter for the executions' lines. Executions are full resource names
    (``.../jobs/<job>/executions/<name>``); log entries carry the last part as a label."""
    names = " OR ".join(f'{EXECUTION_LABEL}="{e.rsplit("/", 1)[-1]}"' for e in executions)
    return f'resource.type="cloud_run_job" AND ({names}) AND timestamp>="{since.isoformat()}"'


def log_line(entry: dict[str, Any]) -> LogLine | None:
    """A Cloud Logging entry as a line; None if it carries no text."""
    payload = entry.get("jsonPayload") or {}
    text = entry.get("textPayload") or payload.get("message") or ""
    if not text.strip():
        return None
    job = entry.get("resource", {}).get("labels", {}).get("job_name", "")
    return LogLine(
        at=entry["timestamp"],
        job="finish" if job.startswith("finish-") else "retrieve",
        text=text.rstrip(),
    )


class LogCache:
    """Each run's lines, read at most once per ``seconds``. When a read fails, the last
    lines read are served instead."""

    def __init__(
        self,
        logs: Logs,
        seconds: float = LOG_CACHE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logs = logs
        self._seconds = seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: dict[str, tuple[float, list[LogLine]]] = {}

    def lines(
        self, run_id: str, executions: list[str], since: datetime
    ) -> tuple[list[LogLine], bool]:
        """The run's newest lines, and whether they could be read now."""
        with self._lock:
            cached = self._cached.get(run_id)
        if cached and self._clock() - cached[0] < self._seconds:
            return cached[1], True
        if not executions:
            return [], True
        try:
            lines = self._logs.read(executions, since, LOG_LINES)
        except Exception:
            return (cached[1] if cached else []), False
        with self._lock:
            now = self._clock()
            self._cached = {k: v for k, v in self._cached.items() if now - v[0] < self._seconds}
            self._cached[run_id] = (now, lines)
        return lines, True
