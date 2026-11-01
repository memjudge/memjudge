from __future__ import annotations

import pytest

from memjudge.metrics import (
    Ingestion,
    category_metrics,
    cluster_bootstrap_ci,
    compute_metrics,
    percentile,
)
from memjudge.records import BatchRecord, ConversationRecord, QuestionRecord, UsageRecord


def q(
    qid: str, conv: str = "c1", correct: bool | None = True, tokens: int = 1000, **kw: object
) -> QuestionRecord:
    base: dict[str, object] = {
        "question_id": qid,
        "conversation_id": conv,
        "category": "cat",
        "kind": "standard",
        "correct": correct,
        "context_tokens": tokens,
        "retrieval_seconds": 0.5,
        "judge_attempts": 1,
        "judge_input_tokens": 1000,
        "judge_output_tokens": 100,
    }
    base.update(kw)
    return QuestionRecord(**base)  # type: ignore[arg-type]


def conv(cid: str = "c1", **kw: object) -> ConversationRecord:
    base: dict[str, object] = {
        "conversation_id": cid,
        "sessions": 2,
        "turns": 10,
        "ingest_seconds": 5.0,
        "session_seconds": [2.0, 3.0],
        "ingest_calls": 2,
        "ingest_errors": 0,
        "usage": UsageRecord(api_calls=2),
    }
    base.update(kw)
    return ConversationRecord(**base)  # type: ignore[arg-type]


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None


def test_category_metrics_hand_computed() -> None:
    records = [
        q("a", correct=True, tokens=500),
        q("b", correct=False, tokens=1500),
        q("c", kind="abstention", correct=True, tokens=0),
        q("d", kind="knowledge_update", correct=True, stale=True, tokens=2000, truncated=True),
        q("e", correct=None, judge_error="boom", tokens=1000, evidence_recall=0.5),
    ]
    m = category_metrics(records)
    assert m.questions == 5 and m.correct == 3
    assert m.accuracy == pytest.approx(0.6)
    assert m.context_tokens_mean == pytest.approx(1000)
    assert m.efficiency == pytest.approx(60.0)  # 60% per 1k tokens
    assert m.abstention_accuracy.n == 1 and m.abstention_accuracy.value == 1.0
    assert m.stale_fact_rate.n == 1 and m.stale_fact_rate.value == 1.0
    assert m.evidence_recall.value == 0.5 and m.evidence_recall.n == 1
    assert m.truncation_rate == pytest.approx(0.2)
    assert m.errors == 1


def test_cluster_bootstrap_ci() -> None:
    single = [q("a")]
    assert cluster_bootstrap_ci(single) == (1.0, 1.0)  # nothing to resample
    few = [q(f"{i}", conv=f"c{i % 2}", correct=i % 2 == 0) for i in range(40)]
    lo_few, hi_few = cluster_bootstrap_ci(few)
    assert lo_few < 0.5 < hi_few  # <5 conversations: question-level, not degenerate
    many = [q(f"{i}", conv=f"c{i % 5}", correct=i % 3 != 0) for i in range(60)]
    lo, hi = cluster_bootstrap_ci(many)
    acc = sum(r.correct is True for r in many) / 60
    assert lo <= acc <= hi
    assert cluster_bootstrap_ci(many) == (lo, hi)  # deterministic


def test_cost_prefers_api_then_price_table_then_not_reported() -> None:
    def price(i: int, o: int) -> float:
        return (i * 2 + o * 10) / 1_000_000

    api = compute_metrics(
        [q("a", judge_cost_usd=0.01), q("b", judge_cost_usd=0.02)], [conv()], price
    )
    assert api.cost.judge_usd == pytest.approx(0.03) and api.cost.judge_cost_source == "api"
    # memory system reported nothing -> total is not reported, never guessed
    assert api.cost.ingestion_usd is None and api.cost.total_usd is None

    table = compute_metrics([q("a"), q("b")], [conv(usage=UsageRecord(cost_usd=1.0))], price)
    assert table.cost.judge_cost_source == "price-table"
    assert table.cost.judge_usd == pytest.approx(2 * (1000 * 2 + 100 * 10) / 1e6)

    unknown = compute_metrics([q("a")], [conv()], lambda i, o: None)
    assert unknown.cost.judge_cost_source == "not reported" and unknown.cost.judge_usd is None


def test_totals_error_rate_and_ingestion() -> None:
    m = compute_metrics(
        [
            q("a", retrieval_usage=UsageRecord(cost_usd=0.5)),
            q("b", retrieval_error="Timeout", judge_attempts=0, correct=None),
        ],
        [conv(usage=UsageRecord(api_calls=2, cost_usd=1.5), ingest_errors=1, ingest_calls=3)],
        lambda i, o: 0.0,
    )
    # calls: 3 ingest + 2 retrieve + 1 judge = 6; failures: 1 ingest + 1 retrieve
    assert m.error_rate.n == 6 and m.error_rate.value == pytest.approx(2 / 6)
    assert m.cost.total_usd == pytest.approx(2.0)
    assert m.cost.per_1k_questions_usd == pytest.approx(1000.0)
    assert m.ingestion.per_turn_s == pytest.approx(0.5)
    assert m.ingestion.per_session_p50_s == pytest.approx(2.5)
    assert m.overall.accuracy == pytest.approx(0.5)  # failed questions count as not correct


def test_ingestion_mode_says_what_total_s_measures() -> None:
    per_session = compute_metrics([q("a")], [conv(), conv("c2")], lambda i, o: 0.0)
    assert per_session.ingestion.mode == "per-session"
    assert per_session.ingestion.total_s == pytest.approx(10.0)  # each conversation's, summed

    batched = conv(ingest_seconds=0.0, session_seconds=[], ingest_calls=0)
    batch = compute_metrics(
        [q("a")], [batched], lambda i, o: 0.0, BatchRecord(seconds=7.0, usage=UsageRecord())
    )
    assert batch.ingestion.mode == "batch"
    assert batch.ingestion.total_s == pytest.approx(7.0)  # the upload's wall-clock time
    assert batch.ingestion.per_session_p50_s is None

    # Results from before ADR 0017 have no mode: they are per-session.
    older = per_session.ingestion.model_dump(exclude={"mode"})
    assert Ingestion.model_validate(older).mode == "per-session"
