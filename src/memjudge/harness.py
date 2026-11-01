"""Run a benchmark: reset -> ingest -> retrieve -> judge, then compute metrics.

Conversations run in parallel (``concurrency``); judge calls run in their own pool.
Every finished conversation and question is appended to JSON lines immediately, so
an interrupted run can ``resume`` and skip completed conversations.

When the memory system fails (an ingestion fails, or ``MAX_FAILURES_IN_ROW`` retrievals
fail in a row) the run stops with ``RunStoppedError`` instead of recording errors for the
rest of the dataset. Conversations still in progress are not marked complete, so a resume
wipes and redoes them.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memjudge import __version__
from memjudge.canonical import file_hash
from memjudge.connectors.base import (
    BaseConnector,
    Retrieval,
    Usage,
    is_custom_endpoint,
    public_url,
)
from memjudge.connectors.loader import ConnectorRef, LoadedConnector
from memjudge.datasets.loader import LoadedDataset
from memjudge.judge.judge import Completer, Prompt, judge
from memjudge.metrics import compute_metrics
from memjudge.records import (
    BatchRecord,
    ConversationRecord,
    JsonlWriter,
    QuestionRecord,
    UsageRecord,
    latest_by,
    read_jsonl,
)
from memjudge.result import (
    Affiliation,
    Artifact,
    ConnectorInfo,
    DatasetInfo,
    JudgeInfo,
    RejudgedFrom,
    ReusedIngestionInfo,
    RunConfigInfo,
    RunEvent,
    RunResult,
    ToolInfo,
)
from memjudge.schema import Conversation, Question
from memjudge.sizes import TOP_K, questions_file, size_key
from memjudge.tokens import TOKENIZER, truncate

log = logging.getLogger(__name__)

CONVERSATIONS_FILE = "conversations.jsonl"
BATCH_FILE = "ingest_batch.json"
RESULT_FILE = "result.json"
MAX_FAILURES_IN_ROW = 5
"""Retrieval failures in a row that mean the memory system is down: the run stops."""


class RunStoppedError(Exception):
    """The memory system failed, so the run stopped early. Completed conversations are
    saved; resuming redoes the rest."""


@dataclass
class RunOptions:
    budget_tokens: int = 8192
    concurrency: int = 4
    judge_concurrency: int = 8
    cleanup: bool = True
    affiliation: Affiliation = field(default_factory=lambda: Affiliation(affiliated=None))
    resumes: int = 0
    """Times the run was resumed after it stopped; recorded in the result."""
    history: list[RunEvent] = field(default_factory=list)
    """A hosted run's stops and resumes; recorded in the result."""


@dataclass(frozen=True)
class ReusedIngestion:
    """Ingestion kept by an earlier run (``--keep-data``), reused to rerun only retrieval."""

    run_id: str
    connector_code_hash: str
    records: dict[str, ConversationRecord]
    """That run's conversation records, by conversation id."""
    batch: BatchRecord | None = None
    """That run's batch ingestion, if it used one (ADR 0010)."""

    @property
    def info(self) -> ReusedIngestionInfo:
        return ReusedIngestionInfo(run_id=self.run_id, connector_code_hash=self.connector_code_hash)


@dataclass
class JudgeSetup:
    client: Completer
    prompt: Prompt
    info: Callable[[list[str]], JudgeInfo]
    """Builds JudgeInfo once the served-by models are known."""
    price: Callable[[int, int], float | None]


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def namespace_for(run_id: str, conversation_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", conversation_id)
    return f"mj-{run_id[-6:]}-{safe}"[:120]


def _usage(u: Usage) -> UsageRecord:
    return UsageRecord(
        api_calls=u.api_calls,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        credits=u.credits,
        cost_usd=u.cost_usd,
    )


def _error(exc: BaseException) -> str:
    # Exception text from SDKs can echo request data; keep it short and never include headers.
    return f"{exc.__class__.__name__}: {str(exc)[:300]}"


def evidence_recall(question: Question, retrieval: Retrieval) -> float | None:
    """Share of evidence turns the connector says it returned (needs ``source_turn_ids``)."""
    if not question.evidence_turn_ids:
        return None
    returned = {tid for item in retrieval.items for tid in item.source_turn_ids}
    if not returned:
        return None  # system does not expose provenance: not reported
    hits = sum(1 for tid in question.evidence_turn_ids if tid in returned)
    return hits / len(question.evidence_turn_ids)


class Runner:
    def __init__(
        self,
        dataset: LoadedDataset,
        connector: LoadedConnector,
        connector_instance: BaseConnector,
        judge_setup: JudgeSetup | None,
        out_dir: Path,
        options: RunOptions,
        run_id: str | None = None,
        reuse: ReusedIngestion | None = None,
    ) -> None:
        """``judge_setup=None`` runs retrieval only: use ``execute`` and judge the records later."""
        self.data = dataset
        self.reuse = reuse
        self.connector = connector
        self.instance = connector_instance
        self.judge = judge_setup
        self.out_dir = out_dir
        self.options = options
        self.run_id = run_id or new_run_id()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._questions = {k: JsonlWriter(out_dir / questions_file(k)) for k in TOP_K}
        self._conversations = JsonlWriter(out_dir / CONVERSATIONS_FILE)
        self._judge_pool = ThreadPoolExecutor(options.judge_concurrency, "judge")
        self._batched = False
        """The run was ingested in one batch, so conversations skip per-session ingestion."""
        self._stop = threading.Event()
        self._stop_reason: str | None = None
        self._failures_in_row = 0
        self._lock = threading.Lock()

    # --- public -------------------------------------------------------------

    def run(
        self,
        progress: Callable[[int, int], None] | None = None,
        started: datetime | None = None,
        ingest_progress: Callable[[int, int], None] | None = None,
        batch_progress: Callable[[int, int], None] | None = None,
    ) -> RunResult:
        """``started`` is passed on resume so the original start time is kept."""
        started = started or datetime.now(UTC)
        self.execute(progress, ingest_progress, batch_progress)
        return self.finalize(started)

    def execute(
        self,
        progress: Callable[[int, int], None] | None = None,
        ingest_progress: Callable[[int, int], None] | None = None,
        batch_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Ingest, retrieve and (with a judge) judge every question; write the records.

        ``ingest_progress(done, total)`` counts ingested sessions; it is not called when
        reusing an earlier run's ingestion. ``batch_progress(done, total)`` counts turns
        while a batch runs (ADR 0010); sessions are counted when it finishes. Raises
        ``RunStoppedError`` if the memory system fails.
        """
        ds = self.data.dataset
        done = self._completed_conversations()
        todo = [c for c in ds.conversations if c.id not in done]
        if done:
            log.info("resuming: %d conversations already complete", len(done))
        questions_by_conv: dict[str, list[Question]] = {}
        for q in ds.questions:
            questions_by_conv.setdefault(q.conversation_id, []).append(q)

        # one record per question and size
        total_q = len(ds.questions) * len(TOP_K)
        finished = [sum(len(questions_by_conv.get(c, [])) for c in done) * len(TOP_K)]
        if progress:
            progress(finished[0], total_q)

        def on_question() -> None:
            finished[0] += 1
            if progress:
                progress(finished[0], total_q)

        if self.reuse:
            ingest_progress = None
            if self.reuse.batch:  # the earlier run's batch is this run's ingestion
                (self.out_dir / BATCH_FILE).write_text(
                    self.reuse.batch.model_dump_json(indent=2), encoding="utf-8"
                )
        total_s = sum(len(c.sessions) for c in ds.conversations)
        ingested = [sum(len(c.sessions) for c in ds.conversations if c.id in done)]
        if ingest_progress:
            ingest_progress(ingested[0], total_s)

        def on_sessions(n: int) -> None:
            ingested[0] += n
            if ingest_progress:
                ingest_progress(ingested[0], total_s)

        pool = ThreadPoolExecutor(self.options.concurrency, "conv")
        interrupted = False
        try:
            if not self.reuse and todo:
                self._ingest_batch(todo, on_sessions, batch_progress)
            futures = [
                pool.submit(
                    self._run_conversation,
                    c,
                    questions_by_conv.get(c.id, []),
                    on_question,
                    on_sessions,
                )
                for c in todo
            ]
            _wait_all(futures)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            # On Ctrl+C, stop waiting for work in flight; finished records are saved.
            pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
            self._judge_pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
            self.instance.close()
        if self._stop_reason:
            raise RunStoppedError(self._stop_reason)

    def finalize(self, started: datetime) -> RunResult:
        if self.judge is None:
            raise ValueError("a retrieval-only run has no judge: judge its records first")
        return build_result(
            self.out_dir,
            run_id=self.run_id,
            started=started,
            dataset=self.data,
            connector=self.connector,
            settings=dict(self.instance.settings),
            judge_setup=self.judge,
            options=self.options,
            endpoint=self.instance.endpoint,
            reused=self.reuse.info if self.reuse else None,
        )

    # --- internals ----------------------------------------------------------

    def _ingest_batch(
        self,
        todo: list[Conversation],
        on_sessions: Callable[[int], None],
        batch_progress: Callable[[int, int], None] | None,
    ) -> None:
        """Ingest the whole run in one batch, if the connector does that (ADR 0010)."""
        t0 = time.perf_counter()
        # A failure stops the run: asking for a batch and getting per-session
        # ingestion instead would measure something else, at another cost.
        usage = self.instance.ingest_batch(
            {namespace_for(self.run_id, c.id): c for c in todo},
            self.out_dir,
            progress=batch_progress,
        )
        if usage is None:
            return
        self._batched = True
        on_sessions(sum(len(c.sessions) for c in todo))
        record = BatchRecord(seconds=time.perf_counter() - t0, usage=_usage(usage))
        log.info("ingested %d conversations in one batch in %.0f s", len(todo), record.seconds)
        path = self.out_dir / BATCH_FILE
        if path.exists():  # resumed: keep the earlier attempts' time and calls
            old = BatchRecord.model_validate_json(path.read_text(encoding="utf-8"))
            record.seconds += old.seconds
            record.usage.api_calls += old.usage.api_calls
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _completed_conversations(self) -> set[str]:
        convs = latest_by(
            read_jsonl(self.out_dir / CONVERSATIONS_FILE, ConversationRecord), "conversation_id"
        )
        answered = [
            set(
                latest_by(
                    read_jsonl(self.out_dir / questions_file(k), QuestionRecord), "question_id"
                )
            )
            for k in TOP_K
        ]
        expected: dict[str, set[str]] = {}
        for q in self.data.dataset.questions:
            expected.setdefault(q.conversation_id, set()).add(q.id)
        return {cid for cid in convs if all(expected.get(cid, set()) <= a for a in answered)}

    def _run_conversation(
        self,
        conv: Conversation,
        questions: list[Question],
        on_question: Callable[[], None],
        on_sessions: Callable[[int], None],
    ) -> None:
        if self._stop.is_set():
            return
        if self.reuse:
            ns, record = self._restore(conv)
        elif self._batched:
            ns = namespace_for(self.run_id, conv.id)
            record = self._batched_record(ns, conv)
        else:
            ns = namespace_for(self.run_id, conv.id)
            ingested = self._ingest(ns, conv, on_sessions)
            if ingested is None:
                return
            record = ingested
        judged: list[Future[None]] = []
        for i, q in enumerate(questions):
            # Alternate which size goes first, so neither always gets the cold call (ADR 0018).
            for top_k in TOP_K if i % 2 == 0 else TOP_K[::-1]:
                if self._stop.is_set():
                    return
                if record.error:
                    rec = QuestionRecord(
                        **question_fields(q), retrieval_error=f"ingestion failed: {record.error}"
                    )
                    self._questions[top_k].write(rec)
                    on_question()
                    continue
                rec = self._retrieve(ns, q, top_k)
                self._count_failure(rec.retrieval_error)
                if rec.retrieval_error:
                    log.warning(
                        "retrieve %s (%s) failed: %s", q.id, size_key(top_k), rec.retrieval_error
                    )
                if rec.retrieval_error or self.judge is None:
                    self._questions[top_k].write(rec)
                    on_question()
                else:
                    judged.append(self._judge_pool.submit(self._judge, q, rec, top_k, on_question))
        for f in judged:
            f.result()
        if self._stop.is_set():
            return  # possibly hit by the failure that stopped the run: redo it on resume
        # Written last: a conversation counts as complete only after all its questions.
        self._conversations.write(record)
        log.info("conversation %s complete: %d questions", conv.id, len(questions))
        # Reused data belongs to the earlier run; only `memjudge cleanup` on that run deletes it.
        if self.options.cleanup and not self.reuse:
            try:
                self.instance.reset(ns)
            except Exception as exc:  # cleanup failure must not fail the run
                log.warning("cleanup of %s failed: %s", ns, _error(exc))

    def _halt(self, reason: str) -> None:
        """Stop the run; the first reason is the one reported."""
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason
        self._stop.set()

    def _count_failure(self, error: str | None) -> None:
        with self._lock:
            self._failures_in_row = self._failures_in_row + 1 if error else 0
            n = self._failures_in_row
        if n >= MAX_FAILURES_IN_ROW:
            self._halt(f"{n} retrievals failed in a row. Last error: {error}")

    def _ingest(
        self, ns: str, conv: Conversation, on_sessions: Callable[[int], None]
    ) -> ConversationRecord | None:
        """Ingest one conversation; None if the run stopped (this failure or another)."""
        usage = Usage()
        session_seconds: list[float] = []
        start = time.perf_counter()
        log.info("ingesting %s: %d sessions", conv.id, len(conv.sessions))
        try:
            self.instance.reset(ns)
            for session in conv.sessions:
                if self._stop.is_set():
                    return None
                t0 = time.perf_counter()
                usage = usage + self.instance.ingest(ns, session)
                session_seconds.append(time.perf_counter() - t0)
                on_sessions(1)
        except Exception as exc:
            error = _error(exc)
            log.warning("ingest %s failed: %s", conv.id, error)
            self._halt(f"Ingestion failed for conversation {conv.id}. {error}")
            return None
        try:
            state = self.instance.save_state(ns)
        except Exception as exc:
            state = None
            log.warning("could not save connector state for %s: %s", ns, _error(exc))
        seconds = time.perf_counter() - start
        log.info("ingested %s in %.0f s", conv.id, seconds)
        return ConversationRecord(
            conversation_id=conv.id,
            sessions=len(conv.sessions),
            turns=conv.turn_count(),
            ingest_seconds=seconds,
            session_seconds=session_seconds,
            ingest_calls=len(session_seconds),
            ingest_errors=0,
            usage=_usage(usage),
            namespace=ns,
            state=state,
        )

    def _batched_record(self, ns: str, conv: Conversation) -> ConversationRecord:
        """The record of a conversation the batch ingested: its time and usage are in
        ingest_batch.json, for the whole run."""
        try:
            state = self.instance.save_state(ns)
        except Exception as exc:
            state = None
            log.warning("could not save connector state for %s: %s", ns, _error(exc))
        return ConversationRecord(
            conversation_id=conv.id,
            sessions=len(conv.sessions),
            turns=conv.turn_count(),
            ingest_seconds=0.0,
            session_seconds=[],
            ingest_calls=0,
            ingest_errors=0,
            usage=UsageRecord(),
            namespace=ns,
            state=state,
        )

    def _restore(self, conv: Conversation) -> tuple[str, ConversationRecord]:
        """Point the connector at an earlier run's stored data instead of ingesting."""
        assert self.reuse is not None
        old = self.reuse.records[conv.id]
        ns = old.namespace or namespace_for(self.reuse.run_id, conv.id)
        record = old.model_copy(update={"reused_from": self.reuse.run_id})
        try:
            self.instance.restore_state(ns, old.state)
        except Exception as exc:
            record = record.model_copy(update={"error": f"restore failed: {_error(exc)}"})
        return ns, record

    def _retrieve(self, ns: str, q: Question, top_k: int) -> QuestionRecord:
        base = question_fields(q)
        t0 = time.perf_counter()
        try:
            retrieval = self.instance.retrieve(
                ns, q.question, q.asked_at, self.options.budget_tokens, top_k
            )
        except Exception as exc:
            return QuestionRecord(
                **base, retrieval_seconds=time.perf_counter() - t0, retrieval_error=_error(exc)
            )
        seconds = time.perf_counter() - t0
        context, raw_tokens, truncated = truncate(retrieval.context, self.options.budget_tokens)
        return QuestionRecord(
            **base,
            context=context,
            context_tokens=min(raw_tokens, self.options.budget_tokens),
            raw_context_tokens=raw_tokens,
            truncated=truncated,
            retrieval_seconds=seconds,
            items=len(retrieval.items),
            retrieved_turn_ids=sorted({t for i in retrieval.items for t in i.source_turn_ids}),
            evidence_recall=evidence_recall(q, retrieval),
            retrieval_usage=_usage(retrieval.usage),
        )

    def _judge(
        self, q: Question, rec: QuestionRecord, top_k: int, on_question: Callable[[], None]
    ) -> None:
        assert self.judge is not None
        try:
            verdict = judge(self.judge.client, self.judge.prompt, q, rec.context)
        except Exception as exc:  # judge-side failure: record it, never abort the run
            log.warning("judge %s (%s) failed: %s", q.id, size_key(top_k), _error(exc))
            self._questions[top_k].write(rec.model_copy(update={"judge_error": _error(exc)}))
        else:
            self._questions[top_k].write(apply_verdict(rec, verdict))
        on_question()


def _wait_all(futures: list[Future[None]]) -> None:
    """Wait for every future, raising the first error. Waits in short steps: on
    Windows a plain wait cannot be interrupted, so Ctrl+C would go unnoticed."""
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=0.5)
        for f in done:
            f.result()


def question_fields(q: Question) -> dict[str, Any]:
    """The fields that identify a question in its record, with its text and gold answer."""
    return {
        "question_id": q.id,
        "conversation_id": q.conversation_id,
        "category": q.category,
        "kind": q.kind.value,
        "question": q.question,
        "gold_answer": q.gold_answer,
        "asked_at": q.asked_at,
    }


def apply_verdict(rec: QuestionRecord, verdict: Any) -> QuestionRecord:
    return rec.model_copy(
        update={
            "correct": verdict.correct,
            "reason": verdict.reason,
            "stale": verdict.stale,
            "gold_suspect": verdict.gold_suspect,
            "judge_raw": verdict.raw_text,
            "judge_input_tokens": verdict.input_tokens,
            "judge_output_tokens": verdict.output_tokens,
            "judge_cost_usd": verdict.cost_usd,
            "judge_served_by": verdict.served_by,
            "judge_attempts": verdict.attempts,
            "judge_error": verdict.error,
        }
    )


_NO_SPEND: dict[str, Any] = {
    "judge_input_tokens": None,
    "judge_output_tokens": None,
    "judge_cost_usd": None,
    "judge_attempts": 0,
}
"""Judge fields of an attempt that raised before any call completed."""


def add_judge_spend(rec: QuestionRecord, earlier: QuestionRecord) -> QuestionRecord:
    """``rec`` plus the judge calls of an earlier failed attempt on the same question,
    which were billed too."""

    def add(a: float | None, b: float | None) -> float | None:
        return b if a is None else a if b is None else a + b

    tokens_in = add(earlier.judge_input_tokens, rec.judge_input_tokens)
    tokens_out = add(earlier.judge_output_tokens, rec.judge_output_tokens)
    return rec.model_copy(
        update={
            "judge_input_tokens": None if tokens_in is None else int(tokens_in),
            "judge_output_tokens": None if tokens_out is None else int(tokens_out),
            "judge_cost_usd": add(earlier.judge_cost_usd, rec.judge_cost_usd),
            "judge_attempts": earlier.judge_attempts + rec.judge_attempts,
        }
    )


def judge_records(
    records: Iterable[QuestionRecord],
    questions: Mapping[str, Question],
    setup: JudgeSetup,
    concurrency: int = 8,
    retry_rounds: int = 0,
    progress: Callable[[int, int], None] | None = None,
    on_record: Callable[[QuestionRecord], None] | None = None,
) -> list[QuestionRecord]:
    """Judge stored contexts, in order, with question text refilled from the dataset.

    Records with a retrieval error pass through unjudged. Records the judge still failed
    on are judged again, up to ``retry_rounds`` more times; a retried record keeps the
    spend of its failed attempts. ``progress(done, total)`` is called after each record of
    the first round, ``on_record(record)`` after each record of every round.
    """

    def one(rec: QuestionRecord) -> QuestionRecord:
        q = questions[rec.question_id]
        rec = rec.model_copy(update=question_fields(q))
        if rec.retrieval_error:
            return rec
        try:
            return apply_verdict(rec, judge(setup.client, setup.prompt, q, rec.context))
        except Exception as exc:  # judge-side failure: record it, never abort the run
            log.warning("judge %s failed: %s", q.id, _error(exc))
            return rec.model_copy(update={"judge_error": _error(exc), **_NO_SPEND})

    out = list(records)
    todo = list(range(len(out)))
    with ThreadPoolExecutor(concurrency, "judge") as pool:
        for round_ in range(retry_rounds + 1):
            for n, (i, rec) in enumerate(
                zip(todo, pool.map(one, [out[i] for i in todo]), strict=True), 1
            ):
                out[i] = add_judge_spend(rec, out[i]) if round_ else rec
                if on_record:
                    on_record(out[i])
                if progress and round_ == 0:
                    progress(n, len(out))
            todo = [i for i in todo if out[i].judge_error and not out[i].retrieval_error]
            if not todo:
                break
    return out


def public_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Connector settings as published: a ``base_url`` loses any credentials or query."""
    out = dict(settings)
    if isinstance(out.get("base_url"), str):
        out["base_url"] = public_url(out["base_url"])
    return out


def git_commit() -> str | None:
    """The commit of a MemJudge checkout, or None for an installed wheel."""
    root = Path(__file__).resolve().parents[2]  # <checkout>/src/memjudge/harness.py
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = out.stdout.splitlines()
    # An installed wheel sits in site-packages: whatever repo contains the venv is not ours.
    if len(lines) != 2 or Path(lines[0]).resolve() != root:
        return None
    return lines[1].strip() or None


def build_result(
    out_dir: Path,
    *,
    run_id: str,
    started: datetime,
    dataset: LoadedDataset,
    connector: ConnectorRef | None,
    settings: dict[str, Any] | None,
    judge_setup: JudgeSetup,
    options: RunOptions,
    endpoint: str | None = None,
    reused: ReusedIngestionInfo | None = None,
    mode: str = "local",
    tool: ToolInfo | None = None,
    rejudged_from: RejudgedFrom | None = None,
    connector_info: ConnectorInfo | None = None,
) -> RunResult:
    """Recompute metrics from the raw records on disk and write ``result.json``.

    ``tool`` defaults to this package's version and checkout commit; the hosted runner
    passes its own (commit and image digest are baked into its image). A re-judge passes the
    earlier run's ``connector_info`` (the code that retrieved the contexts) instead of a
    ``connector`` and ``settings``.
    """
    from memjudge.pricing.cost import load_price_table

    question_ids = {q.id for q in dataset.dataset.questions}
    questions = {
        k: [
            r
            for r in latest_by(
                read_jsonl(out_dir / questions_file(k), QuestionRecord), "question_id"
            ).values()
            if r.question_id in question_ids
        ]
        for k in TOP_K
    }
    conversations = list(
        latest_by(
            read_jsonl(out_dir / CONVERSATIONS_FILE, ConversationRecord), "conversation_id"
        ).values()
    )
    batch_path = out_dir / BATCH_FILE
    batch = (
        BatchRecord.model_validate_json(batch_path.read_text(encoding="utf-8"))
        if batch_path.exists()
        else None
    )
    metrics = {
        size_key(k): compute_metrics(records, conversations, judge_setup.price, batch)
        for k, records in questions.items()
    }
    served_by = sorted(
        {r.judge_served_by for records in questions.values() for r in records if r.judge_served_by}
    )
    ds = dataset.dataset
    if connector_info is None:
        assert connector is not None and settings is not None
        connector_info = ConnectorInfo(
            id=connector.spec.id,
            code_hash=connector.code_hash,
            target=connector.spec.target,
            label=connector.spec.label,
            sdk=connector.spec.sdk,
            settings=public_settings(settings),
            models=connector.spec.model_values(settings),
            endpoint=public_url(endpoint) if endpoint else None,
            custom_endpoint=is_custom_endpoint(endpoint, connector.spec.allowed_domains),
        )
    result = RunResult(
        run_id=run_id,
        mode=mode,  # type: ignore[arg-type]
        started_at=started,
        finished_at=datetime.now(UTC),
        tool=tool or ToolInfo(version=__version__, commit=git_commit()),
        dataset=DatasetInfo(
            name=ds.name,
            version=ds.version,
            hash=dataset.hash,
            curated=dataset.curated,
            license=ds.license,
            upstream_sha256=dataset.upstream_sha256,
            errata_applied=dataset.errata_applied,
            selection=dataset.selection,
            conversation_ids=[c.id for c in ds.conversations] if dataset.selection else None,
            questions=len(question_ids),
        ),
        connector=connector_info,
        judge=judge_setup.info(served_by),
        config=RunConfigInfo(
            budget_tokens=options.budget_tokens,
            tokenizer=TOKENIZER,
            concurrency=options.concurrency,
            judge_concurrency=options.judge_concurrency,
        ),
        metrics=metrics,
        price_table_version=load_price_table().version,
        artifacts=[
            Artifact(name=name, sha256=file_hash(out_dir / name))
            for name in (CONVERSATIONS_FILE, *map(questions_file, TOP_K), BATCH_FILE)
            if (out_dir / name).exists()
        ],
        affiliation=options.affiliation,
        reused_ingestion=reused,
        rejudged_from=rejudged_from,
        resumes=options.resumes,
        history=options.history,
    )
    (out_dir / RESULT_FILE).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
