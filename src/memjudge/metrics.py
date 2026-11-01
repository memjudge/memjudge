"""Compute every published metric from raw records (spec §6, docs/scoring.md)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from memjudge.records import BatchRecord, ConversationRecord, QuestionRecord
from memjudge.schema import QuestionKind

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
MIN_CLUSTERS = 5


class Rate(BaseModel):
    """A proportion with its sample size; ``value`` is None when ``n`` is 0."""

    model_config = ConfigDict(extra="forbid")

    n: int
    value: float | None


class CategoryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: int
    correct: int
    accuracy: float
    accuracy_ci95: tuple[float, float]
    context_tokens_mean: float
    context_tokens_p95: float
    efficiency: float | None
    """Accuracy (%) per 1k context tokens."""
    retrieval_latency_p50_s: float | None
    retrieval_latency_p95_s: float | None
    abstention_accuracy: Rate
    stale_fact_rate: Rate
    evidence_recall: Rate
    truncation_rate: float
    flagged: int
    errors: int


class Cost(BaseModel):
    """USD. ``None`` means not reported by the system (never estimated)."""

    model_config = ConfigDict(extra="forbid")

    ingestion_usd: float | None
    retrieval_usd: float | None
    judge_usd: float | None
    total_usd: float | None
    per_1k_questions_usd: float | None
    ingestion_credits: float | None
    retrieval_credits: float | None
    judge_cost_source: str
    """'api' (reported by the route), 'price-table', or 'not reported'."""


class Ingestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversations: int
    sessions: int
    turns: int
    total_s: float
    """'per-session': each conversation's time until searchable, summed. 'batch': wall-clock
    time of the one upload until all of it is searchable. Not comparable (ADR 0017)."""
    per_turn_s: float | None
    per_session_p50_s: float | None
    per_session_p95_s: float | None
    api_calls: int
    input_tokens: int | None
    output_tokens: int | None
    mode: Literal["per-session", "batch"] = "per-session"
    """How the run ingested (ADR 0010); results without it are per-session."""


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: CategoryMetrics
    by_category: dict[str, CategoryMetrics]
    ingestion: Ingestion
    cost: Cost
    error_rate: Rate
    """Failed or timed-out calls (ingest + retrieve + judge) over all calls."""
    flagged_question_ids: list[str]


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def cluster_bootstrap_ci(
    records: Sequence[QuestionRecord],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI of accuracy, resampling whole conversations.

    Questions about the same conversation are correlated; resampling conversations
    rather than questions gives an honest (wider) interval. With fewer than
    ``MIN_CLUSTERS`` conversations a cluster bootstrap is degenerate, so questions are
    resampled instead (see docs/scoring.md).
    """
    if not records:
        return (0.0, 0.0)
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        g = groups[r.conversation_id]
        g[0] += int(r.correct is True)
        g[1] += 1
    clusters = list(groups.values())
    if len(clusters) < MIN_CLUSTERS:
        clusters = [[int(r.correct is True), 1] for r in records]
    if len(clusters) == 1:
        acc = clusters[0][0] / clusters[0][1]
        return (acc, acc)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(resamples):
        correct = total = 0
        for _ in range(len(clusters)):
            c, t = clusters[rng.randrange(len(clusters))]
            correct += c
            total += t
        stats.append(correct / total)
    low, high = percentile(stats, 2.5), percentile(stats, 97.5)
    assert low is not None and high is not None
    return (low, high)


def _rate(items: Iterable[bool]) -> Rate:
    values = list(items)
    return Rate(n=len(values), value=(sum(values) / len(values)) if values else None)


def category_metrics(records: Sequence[QuestionRecord]) -> CategoryMetrics:
    n = len(records)
    correct = sum(1 for r in records if r.correct is True)
    accuracy = correct / n if n else 0.0
    tokens = [float(r.context_tokens) for r in records]
    mean_tokens = sum(tokens) / n if n else 0.0
    latencies = [r.retrieval_seconds for r in records if r.retrieval_seconds is not None]
    abstention = [r for r in records if r.kind == QuestionKind.ABSTENTION.value]
    updates = [r for r in records if r.kind == QuestionKind.KNOWLEDGE_UPDATE.value]
    recalls = [r.evidence_recall for r in records if r.evidence_recall is not None]
    return CategoryMetrics(
        questions=n,
        correct=correct,
        accuracy=accuracy,
        accuracy_ci95=cluster_bootstrap_ci(records),
        context_tokens_mean=mean_tokens,
        context_tokens_p95=percentile(tokens, 95) or 0.0,
        efficiency=(accuracy * 100) / (mean_tokens / 1000) if mean_tokens else None,
        retrieval_latency_p50_s=percentile(latencies, 50),
        retrieval_latency_p95_s=percentile(latencies, 95),
        abstention_accuracy=_rate(r.correct is True for r in abstention),
        stale_fact_rate=_rate(
            r.stale is True for r in updates if r.retrieval_error is None and r.correct is not None
        ),
        evidence_recall=Rate(
            n=len(recalls), value=sum(recalls) / len(recalls) if recalls else None
        ),
        truncation_rate=sum(r.truncated for r in records) / n if n else 0.0,
        flagged=sum(r.gold_suspect for r in records),
        errors=sum(r.failed for r in records),
    )


def _sum_optional(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def compute_metrics(
    questions: Sequence[QuestionRecord],
    conversations: Sequence[ConversationRecord],
    judge_price: Callable[[int, int], float | None],
    batch: BatchRecord | None = None,
) -> Metrics:
    by_cat: dict[str, list[QuestionRecord]] = defaultdict(list)
    for r in questions:
        by_cat[r.category].append(r)

    session_times = [t for c in conversations for t in c.session_seconds]
    turns = sum(c.turns for c in conversations)
    # Batch ingestion (ADR 0010) is the run's ingestion; its time and usage are in
    # ingest_batch.json instead of the conversation records.
    ingest_usage = [c.usage for c in conversations] + ([batch.usage] if batch else [])
    ingest_total = sum(c.ingest_seconds for c in conversations) + (batch.seconds if batch else 0)
    ingestion = Ingestion(
        conversations=len(conversations),
        sessions=sum(c.sessions for c in conversations),
        turns=turns,
        total_s=ingest_total,
        per_turn_s=ingest_total / turns if turns else None,
        per_session_p50_s=percentile(session_times, 50),
        per_session_p95_s=percentile(session_times, 95),
        api_calls=sum(u.api_calls for u in ingest_usage),
        input_tokens=_int(_sum_optional(u.input_tokens for u in ingest_usage)),
        output_tokens=_int(_sum_optional(u.output_tokens for u in ingest_usage)),
        mode="batch" if batch else "per-session",
    )

    # Judge cost: reported by the route if every call reported it; else price table.
    judged = [r for r in questions if r.judge_attempts > 0]
    reported = [r.judge_cost_usd for r in judged]
    if judged and all(v is not None for v in reported):
        judge_usd: float | None = sum(v for v in reported if v is not None)
        source = "api"
    else:
        in_tok = sum(r.judge_input_tokens or 0 for r in judged)
        out_tok = sum(r.judge_output_tokens or 0 for r in judged)
        judge_usd = judge_price(in_tok, out_tok) if judged else 0.0
        source = "price-table" if judge_usd is not None else "not reported"

    ingestion_usd = _sum_optional(u.cost_usd for u in ingest_usage)
    retrieval_usd = _sum_optional(r.retrieval_usage.cost_usd for r in questions)
    parts = [ingestion_usd, retrieval_usd, judge_usd]
    total = sum(p for p in parts if p is not None) if all(p is not None for p in parts) else None
    cost = Cost(
        ingestion_usd=ingestion_usd,
        retrieval_usd=retrieval_usd,
        judge_usd=judge_usd,
        total_usd=total,
        per_1k_questions_usd=total * 1000 / len(questions)
        if total is not None and questions
        else None,
        ingestion_credits=_sum_optional(u.credits for u in ingest_usage),
        retrieval_credits=_sum_optional(r.retrieval_usage.credits for r in questions),
        judge_cost_source=source,
    )

    calls = sum(c.ingest_calls for c in conversations) + len(questions) + len(judged)
    failures = (
        sum(c.ingest_errors for c in conversations)
        + sum(r.retrieval_error is not None for r in questions)
        + sum(r.judge_error is not None for r in questions)
    )
    return Metrics(
        overall=category_metrics(questions),
        by_category={k: category_metrics(v) for k, v in sorted(by_cat.items())},
        ingestion=ingestion,
        cost=cost,
        error_rate=Rate(n=calls, value=failures / calls if calls else None),
        flagged_question_ids=sorted(r.question_id for r in questions if r.gold_suspect),
    )


def _int(value: float | None) -> int | None:
    return None if value is None else int(value)
