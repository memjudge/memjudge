"""Check and rebuild the retrieve job's output before it is judged and published.

Retrieve runs connector code, so its files are untrusted. Finish keeps only what the
connector legitimately controls (returned context, timings, usage, errors, its storage
namespace and state) and recomputes everything else from the dataset: question text and
gold answers, token counts, evidence recall. Judge fields are dropped. Every key value
the user gave is redacted first.
"""

from __future__ import annotations

from memjudge.datasets.loader import LoadedDataset
from memjudge.harness import question_fields
from memjudge.records import ConversationRecord, QuestionRecord
from memjudge.tokens import truncate

from memjudge_runner.redact import redact

MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_ERROR_CHARS = 300


class InvalidOutputError(ValueError):
    """The retrieval output is malformed or incomplete."""


def _lines(data: bytes, name: str, secrets: list[str]) -> list[str]:
    if len(data) > MAX_FILE_BYTES:
        raise InvalidOutputError(f"{name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidOutputError(f"{name} is not UTF-8") from exc
    # "\n" only: splitlines() also breaks on U+2028 and others, which JSON leaves unescaped
    return [line for line in redact(text, secrets).split("\n") if line.strip()]


def _short(error: str | None) -> str | None:
    return error[:MAX_ERROR_CHARS] if error else None


def clean_questions(
    data: bytes, dataset: LoadedDataset, budget: int, secrets: list[str], name: str
) -> list[QuestionRecord]:
    """One clean record per dataset question, in dataset order. ``name`` is the file's name,
    for errors (one file per result size, ADR 0018)."""
    questions = {q.id: q for q in dataset.dataset.questions}
    turns = {
        c.id: {t.id for s in c.sessions for t in s.turns} for c in dataset.dataset.conversations
    }
    latest: dict[str, QuestionRecord] = {}
    for n, line in enumerate(_lines(data, name, secrets), 1):
        try:
            rec = QuestionRecord.model_validate_json(line)
        except ValueError as exc:
            raise InvalidOutputError(f"{name} line {n} is not a valid record") from exc
        if rec.question_id not in questions:
            raise InvalidOutputError(f"{name} line {n} has an unknown question id")
        latest[rec.question_id] = rec
    missing = len(questions) - len(latest)
    if missing:
        raise InvalidOutputError(f"{missing} questions have no retrieval record in {name}")

    out = []
    for q in dataset.dataset.questions:
        rec = latest[q.id]
        if rec.retrieval_error:
            out.append(
                QuestionRecord(
                    **question_fields(q),
                    retrieval_seconds=rec.retrieval_seconds,
                    retrieval_usage=rec.retrieval_usage,
                    retrieval_error=_short(rec.retrieval_error),
                )
            )
            continue
        context, tokens, cut = truncate(rec.context, budget)
        raw = max(rec.raw_context_tokens, tokens)
        returned = sorted(set(rec.retrieved_turn_ids) & turns[q.conversation_id])
        recall = None
        if q.evidence_turn_ids and returned:
            recall = sum(1 for t in q.evidence_turn_ids if t in returned) / len(q.evidence_turn_ids)
        out.append(
            QuestionRecord(
                **question_fields(q),
                context=context,
                context_tokens=min(tokens, budget),
                raw_context_tokens=raw,
                truncated=cut or raw > budget,
                retrieval_seconds=(
                    max(0.0, rec.retrieval_seconds) if rec.retrieval_seconds is not None else None
                ),
                items=max(0, rec.items),
                retrieved_turn_ids=returned,
                evidence_recall=recall,
                retrieval_usage=rec.retrieval_usage,
            )
        )
    return out


def clean_conversations(
    data: bytes, dataset: LoadedDataset, secrets: list[str]
) -> list[ConversationRecord]:
    """One clean record per dataset conversation, in dataset order."""
    convs = {c.id: c for c in dataset.dataset.conversations}
    latest: dict[str, ConversationRecord] = {}
    for n, line in enumerate(_lines(data, "conversations.jsonl", secrets), 1):
        try:
            rec = ConversationRecord.model_validate_json(line)
        except ValueError as exc:
            raise InvalidOutputError(f"conversations.jsonl line {n} is not a valid record") from exc
        if rec.conversation_id not in convs:
            raise InvalidOutputError(f"conversations.jsonl line {n} has an unknown conversation id")
        latest[rec.conversation_id] = rec
    missing = len(convs) - len(latest)
    if missing:
        raise InvalidOutputError(f"{missing} conversations have no ingestion record")
    return [
        latest[c.id].model_copy(
            update={
                "sessions": len(c.sessions),
                "turns": c.turn_count(),
                "error": _short(latest[c.id].error),
                "reused_from": None,
            }
        )
        for c in dataset.dataset.conversations
    ]
