"""Finish must not trust what the retrieve job (connector code) wrote."""

from __future__ import annotations

import pytest
from fakes import TINY
from memjudge.datasets.loader import LoadedDataset, load_dataset
from memjudge.harness import question_fields
from memjudge.records import ConversationRecord, QuestionRecord, UsageRecord

from memjudge_runner.validate import InvalidOutputError, clean_conversations, clean_questions

BUDGET = 50


@pytest.fixture
def tiny() -> LoadedDataset:
    return load_dataset(str(TINY))


def _jsonl(records: list[QuestionRecord] | list[ConversationRecord]) -> bytes:
    return "".join(r.model_dump_json() + "\n" for r in records).encode()


def _records(tiny: LoadedDataset, **update: object) -> list[QuestionRecord]:
    return [
        QuestionRecord(**question_fields(q), context="some context").model_copy(update=update)
        for q in tiny.dataset.questions
    ]


def test_connector_cannot_set_verdicts_or_question_text(tiny: LoadedDataset) -> None:
    forged = _records(
        tiny, correct=True, reason="trust me", question="easier?", gold_answer="anything"
    )
    clean = clean_questions(_jsonl(forged), tiny, BUDGET, [], "questions-top30.jsonl")
    for rec, q in zip(clean, tiny.dataset.questions, strict=True):
        assert rec.correct is None and rec.reason == ""
        assert rec.question == q.question and rec.gold_answer == q.gold_answer


def test_token_counts_and_recall_are_recomputed(tiny: LoadedDataset) -> None:
    q = next(q for q in tiny.dataset.questions if q.evidence_turn_ids)
    forged = _records(tiny)
    i = tiny.dataset.questions.index(q)
    forged[i] = forged[i].model_copy(
        update={
            "context": "word " * 200,
            "context_tokens": 1,
            "evidence_recall": 1.0,
            "retrieved_turn_ids": ["not-a-turn"],
        }
    )
    rec = clean_questions(_jsonl(forged), tiny, BUDGET, [], "questions-top30.jsonl")[i]
    assert rec.context_tokens == BUDGET and rec.truncated
    assert rec.retrieved_turn_ids == [] and rec.evidence_recall is None


def test_keys_are_redacted_everywhere(tiny: LoadedDataset) -> None:
    key = "sk-vendor-key-abcdef"
    forged = _records(tiny, context=f"leaked {key}", retrieval_error=None)
    convs = [
        ConversationRecord(
            conversation_id=c.id,
            sessions=0,
            turns=0,
            ingest_seconds=1.0,
            session_seconds=[],
            ingest_calls=1,
            ingest_errors=1,
            usage=UsageRecord(),
            error=f"401 for {key}",
            state={"token": key},
        )
        for c in tiny.dataset.conversations
    ]
    qs = clean_questions(_jsonl(forged), tiny, BUDGET, [key], "questions-top30.jsonl")
    cs = clean_conversations(_jsonl(convs), tiny, [key])
    dumped = "".join(r.model_dump_json() for r in [*qs, *cs])
    assert key not in dumped and "[redacted]" in dumped
    assert cs[0].sessions == len(tiny.dataset.conversations[0].sessions)  # from the dataset


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda recs: recs[1:], "1 questions have no retrieval record"),
        (lambda recs: [*recs, recs[0].model_copy(update={"question_id": "x"})], "unknown question"),
    ],
)
def test_incomplete_or_foreign_records_are_refused(
    tiny: LoadedDataset, mutate: object, message: str
) -> None:
    records = mutate(_records(tiny))  # type: ignore[operator]
    with pytest.raises(InvalidOutputError, match=message):
        clean_questions(_jsonl(records), tiny, BUDGET, [], "questions-top30.jsonl")


def test_unknown_fields_are_refused(tiny: LoadedDataset) -> None:
    line = _records(tiny)[0].model_dump_json()[:-1] + ', "official": true}\n'
    with pytest.raises(InvalidOutputError, match="not a valid record"):
        clean_questions(line.encode(), tiny, BUDGET, [], "questions-top30.jsonl")
