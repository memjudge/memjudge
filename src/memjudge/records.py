"""Per-conversation and per-question raw records, written as JSON lines during a run.

These raw records are the run's artifacts: metrics are recomputed from them, and
``memjudge rejudge`` re-scores the stored contexts without re-running memory systems.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict


class UsageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    credits: float | None = None
    cost_usd: float | None = None


class BatchRecord(BaseModel):
    """The run's batch ingestion (``BaseConnector.ingest_batch``), when used.

    It replaces the per-session ingestion. ``seconds`` adds up every attempt of a
    resumed run.
    """

    model_config = ConfigDict(extra="forbid")

    seconds: float
    usage: UsageRecord


class ConversationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    sessions: int
    turns: int
    ingest_seconds: float
    session_seconds: list[float]
    ingest_calls: int
    ingest_errors: int
    usage: UsageRecord
    error: str | None = None
    namespace: str | None = None
    """Where the data was stored in the memory system."""
    state: dict[str, Any] | None = None
    """Connector state needed to find that data again (see BaseConnector.save_state)."""
    reused_from: str | None = None
    """Run id whose ingestion this record was copied from (--reuse-ingestion)."""


class QuestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    conversation_id: str
    category: str
    kind: str
    # the question itself, so published runs can be read without the dataset
    question: str = ""
    gold_answer: str = ""
    asked_at: datetime | None = None
    # retrieval
    context: str = ""
    context_tokens: int = 0
    raw_context_tokens: int = 0
    truncated: bool = False
    retrieval_seconds: float | None = None
    items: int = 0
    retrieved_turn_ids: list[str] = []
    evidence_recall: float | None = None
    retrieval_usage: UsageRecord = UsageRecord()
    retrieval_error: str | None = None
    # judge
    correct: bool | None = None
    reason: str = ""
    stale: bool | None = None
    gold_suspect: bool = False
    judge_raw: str = ""
    judge_input_tokens: int | None = None
    judge_output_tokens: int | None = None
    judge_cost_usd: float | None = None
    judge_served_by: str | None = None
    judge_attempts: int = 0
    judge_error: str | None = None

    @property
    def failed(self) -> bool:
        return self.retrieval_error is not None or self.judge_error is not None


M = TypeVar("M", bound=BaseModel)


class JsonlWriter:
    """Thread-safe append-only JSON lines file, flushed per record so runs can resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: BaseModel) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    if not path.exists():
        return []
    out: list[M] = []
    # "\n" only: splitlines() also breaks on U+2028 and others, which JSON leaves unescaped
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            out.append(model.model_validate(json.loads(line)))
    return out


def latest_by(records: list[M], key: str) -> dict[str, M]:
    """Last record wins (a resumed run may re-write a key)."""
    out: dict[str, M] = {}
    for r in records:
        value: Any = getattr(r, key)
        out[str(value)] = r
    return out
