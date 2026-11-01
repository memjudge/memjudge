"""Canonical dataset format.

Every source dataset is converted by an adapter into this one shape:
conversations (sessions -> turns) and questions that point at a conversation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_SCHEMA_VERSION = "1"


class QuestionKind(StrEnum):
    """How a question is judged. See docs/scoring.md."""

    STANDARD = "standard"
    ABSTENTION = "abstention"
    KNOWLEDGE_UPDATE = "knowledge_update"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Turn(_Model):
    id: str
    speaker: str
    text: str
    timestamp: datetime | None = None


class Session(_Model):
    id: str
    started_at: datetime | None = None
    turns: tuple[Turn, ...]


class Conversation(_Model):
    id: str
    sessions: tuple[Session, ...]

    def turn_count(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


class Question(_Model):
    id: str
    conversation_id: str
    question: str
    gold_answer: str
    category: str
    kind: QuestionKind = QuestionKind.STANDARD
    asked_at: datetime | None = None
    evidence_turn_ids: tuple[str, ...] = ()
    distractor_answer: str | None = Field(
        default=None,
        description="Abstention only: a plausible answer the context must not support.",
    )


class Dataset(_Model):
    schema_version: str = DATASET_SCHEMA_VERSION
    name: str
    version: str
    source: str = Field(description="Upstream URL the data was converted from.")
    license: str
    conversations: tuple[Conversation, ...]
    questions: tuple[Question, ...]

    @model_validator(mode="after")
    def _check_references(self) -> Dataset:
        conv_ids = [c.id for c in self.conversations]
        if len(set(conv_ids)) != len(conv_ids):
            raise ValueError("duplicate conversation ids")
        q_ids = [q.id for q in self.questions]
        if len(set(q_ids)) != len(q_ids):
            raise ValueError("duplicate question ids")
        turn_ids: dict[str, set[str]] = {}
        for conv in self.conversations:
            ids = [t.id for s in conv.sessions for t in s.turns]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate turn ids in conversation {conv.id}")
            turn_ids[conv.id] = set(ids)
        for q in self.questions:
            if q.conversation_id not in turn_ids:
                raise ValueError(f"question {q.id} references unknown conversation")
            missing = set(q.evidence_turn_ids) - turn_ids[q.conversation_id]
            if missing:
                raise ValueError(f"question {q.id} references unknown turns {sorted(missing)}")
        return self

    def conversation(self, conversation_id: str) -> Conversation:
        for c in self.conversations:
            if c.id == conversation_id:
                return c
        raise KeyError(conversation_id)

    def turn_texts(self, conversation_id: str) -> dict[str, str]:
        conv = self.conversation(conversation_id)
        return {t.id: t.text for s in conv.sessions for t in s.turns}
