"""Run state for the /run form, and the abuse limits.

A run moves ``starting -> retrieving -> judging -> publishing -> done``. From an active
state it can go to ``stopped`` (the memory system or judge failed, the job crashed, or
the owner pressed Stop) or to ``failed`` (final). A stopped run keeps its progress; its
owner can resume it within ``RESUME_HOURS``, at most ``MAX_RESUMES`` times, and it goes
back to the phase it stopped in. Each user has a counter per UTC day; stops and failures
caused by the runner itself (``counts=False``) are refunded, and resuming is free.

A re-judge (ADR 0015) is a run that judges a published run's stored contexts again for the
current season: it starts in ``judging``. A lock per published run and season allows one at
a time and one success per season; a stopped or failed re-judge frees it, and is not
resumed (anyone may start it again).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from memjudge.result import RunEvent, StopCause
from pydantic import BaseModel, ConfigDict

from memjudge_runner.config import MAX_RESUMES, RESUME_HOURS, RUNS_LISTED, Limits

State = Literal["starting", "retrieving", "judging", "publishing", "done", "stopped", "failed"]
ACTIVE_STATES: frozenset[str] = frozenset({"starting", "retrieving", "judging", "publishing"})
Phase = Literal["retrieve", "finish"]
Ending = Literal["stopped", "failed"]
IngestUnit = Literal["sessions", "turns"]
RejudgeState = Literal["active", "done", "failed"]
REJUDGE_ACTIVE = "This run is being re-judged right now."
REJUDGE_DONE = "This run was already re-judged in the current season."


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    user: str
    """HMAC of the GitHub user id; the id itself is never stored."""
    day: str
    """UTC day the run was started (YYYYMMDD), for the daily limit."""
    dataset: str
    connector: str
    preset: str | None = None
    """The connector preset the run uses; None when the connector has none."""
    judge_route: str
    affiliation: dict[str, Any]
    terms: str | None = None
    """Date of the Terms of Use agreed to at the start (None on older records)."""
    state: State = "starting"
    done: int = 0
    total: int = 0
    """Questions answered while retrieving, then questions judged."""
    ingested: int = 0
    ingest_total: int = 0
    ingest_unit: IngestUnit = "sessions"
    """Turns while a batch ingestion runs (ADR 0010), else sessions."""
    error: str | None = None
    counts: bool = True
    """Counts against the daily limit; False once refunded."""
    created_at: datetime
    updated_at: datetime
    execution: str | None = None
    """The Cloud Run execution currently working on the run."""
    executions: list[str] = []
    """Every execution that worked on the run, oldest first: the owner's log reads them."""
    job_started: bool = False
    """Whether the current execution has reported in, past its container start and set-up
    (the form shows "Starting the job" until then)."""
    run_token_hash: str
    judge_object: str | None = None
    endpoint: str | None = None
    result_url: str | None = None
    version: str
    """Release whose jobs run this run, on every attempt."""
    dataset_hash: str
    budget_tokens: int
    key_names: list[str]
    """Connector keys the run needs, asked for again on resume."""
    resumes: int = 0
    history: list[RunEvent] = []
    """Stops (with their cause) and resumes, oldest first; published with the result."""
    stopped_at: datetime | None = None
    resume_from: Phase | None = None
    """The job a resume starts: retrieve (unfinished conversations) or finish (judging)."""
    rejudge_of: str | None = None
    """For a re-judge: the published run whose stored contexts this run judges again."""
    rejudge_season: str | None = None
    """For a re-judge: the season it judges into (the current season when it started)."""

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def all_executions(self) -> list[str]:
        """``executions``, or the current one on records from before they were kept."""
        return self.executions or ([self.execution] if self.execution else [])

    @property
    def rejudge_lock(self) -> str | None:
        """The lock document of a re-judge: one per published run and season."""
        return f"{self.rejudge_of}_{self.rejudge_season}" if self.rejudge_of else None

    def resume_until(self) -> datetime | None:
        """Until when the run can be resumed; None if it cannot be."""
        if resume_error(self) or self.stopped_at is None:
            return None
        return self.stopped_at + timedelta(hours=RESUME_HOURS)

    def public(self) -> dict[str, Any]:
        """What the run's owner sees on the form."""
        until = self.resume_until()
        return {
            **self.model_dump(
                mode="json",
                include={
                    "run_id",
                    "dataset",
                    "connector",
                    "preset",
                    "judge_route",
                    "state",
                    "done",
                    "total",
                    "ingested",
                    "ingest_total",
                    "ingest_unit",
                    "error",
                    "result_url",
                    "created_at",
                    "updated_at",
                    "resumes",
                    "resume_from",
                    "key_names",
                    "rejudge_of",
                    "rejudge_season",
                    "job_started",
                },
            ),
            "resume_until": until.isoformat() if until else None,
        }


def limit_error(user: str, active: list[RunRecord], today: int, limits: Limits) -> str | None:
    """Why ``user`` may not start a run now, or None."""
    if sum(1 for r in active if r.user == user) >= limits.user_active:
        return "You already have a run in progress. Wait for it to finish."
    if today >= limits.user_daily:
        return (
            f"You have reached today's limit of {limits.user_daily} runs. Try again tomorrow (UTC)."
        )
    if len(active) >= limits.global_active:
        return "The runner is busy with other runs. Try again in a while."
    return None


def rejudge_error(lock: dict[str, Any] | None) -> str | None:
    """Why a re-judge cannot take ``lock`` (the lock document, if any), or None."""
    if lock and lock.get("state") == "active":
        return REJUDGE_ACTIVE
    if lock and lock.get("state") == "done":
        return REJUDGE_DONE
    return None


def resume_error(run: RunRecord) -> str | None:
    """Why ``run`` cannot be resumed, or None."""
    if run.rejudge_of:
        return "A re-judge cannot be resumed. Start it again from the run's page."
    if run.state != "stopped":
        return "Only a stopped run can be resumed."
    if run.resumes >= MAX_RESUMES:
        return f"This run was already resumed {MAX_RESUMES} times. Start a new run."
    if run.stopped_at and now() - run.stopped_at > timedelta(hours=RESUME_HOURS):
        return f"A run can be resumed for {RESUME_HOURS} hours after it stops. Start a new run."
    return None


def lock_doc(run: RunRecord, state: RejudgeState) -> dict[str, str]:
    assert run.rejudge_of and run.rejudge_season
    return {
        "original": run.rejudge_of,
        "season": run.rejudge_season,
        "run_id": run.run_id,
        "state": state,
    }


def locks_by_original(locks: Iterable[dict[str, Any]], season: str) -> dict[str, dict[str, str]]:
    return {
        d["original"]: {"state": d["state"], "run_id": d["run_id"]}
        for d in locks
        if d.get("season") == season
    }


def newest(runs: Iterable[RunRecord]) -> list[RunRecord]:
    """A user's runs as the form lists them: newest first, at most ``RUNS_LISTED``."""
    return sorted(runs, key=lambda r: r.created_at, reverse=True)[:RUNS_LISTED]


def now() -> datetime:
    return datetime.now(UTC)


def with_execution(run: RunRecord, execution: str) -> dict[str, Any]:
    """The fields that record a job execution starting work on the run."""
    return {
        "execution": execution,
        "executions": [*run.all_executions, execution],
        "job_started": False,
    }


def ended(
    run: RunRecord, state: Ending, error: str, counts: bool, cause: StopCause | None = None
) -> dict[str, Any]:
    """The fields that end an active run's attempt. A stop remembers which job resumes it,
    and why it stopped."""
    fields: dict[str, Any] = {
        "state": state,
        "error": error,
        "counts": run.counts and counts,
        "updated_at": now(),
    }
    if state == "stopped":
        fields["stopped_at"] = now()
        fields["resume_from"] = "finish" if run.state in ("judging", "publishing") else "retrieve"
        stop = RunEvent(at=fields["stopped_at"], event="stopped", cause=cause)
        fields["history"] = [*run.history, stop]
    return fields


def resumed(run: RunRecord, fields: dict[str, Any]) -> dict[str, Any]:
    """The fields that put a stopped run back to work in the phase it stopped in."""
    at = now()
    return {
        **fields,
        "state": "judging" if run.resume_from == "finish" else "retrieving",
        "resumes": run.resumes + 1,
        "history": [*run.history, RunEvent(at=at, event="resumed")],
        "error": None,
        "stopped_at": None,
        "updated_at": at,
    }


class RunStore(Protocol):
    def create(self, record: RunRecord, limits: Limits) -> str | None:
        """Store ``record`` if the limits allow it (atomically); else return why not. A
        re-judge also takes its lock in the same step (``REJUDGE_ACTIVE``/``REJUDGE_DONE``
        when it is held)."""

    def set_rejudge(self, run: RunRecord, state: RejudgeState) -> None:
        """Set a re-judge's lock, if this run still holds it."""

    def rejudges(self, season: str) -> dict[str, dict[str, str]]:
        """``{published run id: {"state", "run_id"}}`` for the re-judges into ``season``."""

    def get(self, run_id: str) -> RunRecord | None: ...

    def update(self, run_id: str, **fields: Any) -> None: ...

    def transition(self, run_id: str, expect: State, **fields: Any) -> bool:
        """Apply ``fields`` only if the run is still ``expect`` (atomically). Returns whether
        it was, so two callers racing to move a run on cannot both win."""

    def active(self) -> list[RunRecord]: ...

    def for_user(self, user: str) -> list[RunRecord]:
        """``user``'s runs, newest first (``newest``)."""

    def fail(self, run_id: str, error: str, *, counts: bool, execution: str | None = None) -> bool:
        """Mark an active run failed (only if ``execution`` is still its execution, when
        given). ``counts=False`` refunds the run. Returns whether the run was changed."""

    def stop(
        self,
        run_id: str,
        error: str,
        *,
        counts: bool,
        cause: StopCause,
        execution: str | None = None,
    ) -> bool:
        """Like ``fail``, but the run is stopped: its owner can resume it. ``cause`` goes in
        the run's history."""

    def resume(self, run_id: str, limits: Limits, **fields: Any) -> str | None:
        """Put a stopped run back to work with ``fields`` if it is resumable and the
        active-run limits allow it (atomically); else return why not."""


class MemoryRuns:
    """In-process RunStore for tests and local development."""

    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.quota: dict[str, int] = {}
        self.locks: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def create(self, record: RunRecord, limits: Limits) -> str | None:
        with self._lock:
            active = [r for r in self.runs.values() if r.active]
            key = f"{record.user}_{record.day}"
            lock = record.rejudge_lock
            error = (rejudge_error(self.locks.get(lock)) if lock else None) or limit_error(
                record.user, active, self.quota.get(key, 0), limits
            )
            if error:
                return error
            self.quota[key] = self.quota.get(key, 0) + 1
            self.runs[record.run_id] = record
            if lock:
                self.locks[lock] = lock_doc(record, "active")
            return None

    def set_rejudge(self, run: RunRecord, state: RejudgeState) -> None:
        with self._lock:
            lock = run.rejudge_lock
            if lock and self.locks.get(lock, {}).get("run_id") == run.run_id:
                self.locks[lock] = lock_doc(run, state)

    def rejudges(self, season: str) -> dict[str, dict[str, str]]:
        with self._lock:
            return locks_by_original(self.locks.values(), season)

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self.runs.get(run_id)

    def update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            run = self.runs[run_id]
            self.runs[run_id] = run.model_copy(update={**fields, "updated_at": now()})

    def transition(self, run_id: str, expect: State, **fields: Any) -> bool:
        with self._lock:
            run = self.runs.get(run_id)
            if run is None or run.state != expect:
                return False
            self.runs[run_id] = run.model_copy(update={**fields, "updated_at": now()})
            return True

    def active(self) -> list[RunRecord]:
        with self._lock:
            return [r for r in self.runs.values() if r.active]

    def for_user(self, user: str) -> list[RunRecord]:
        with self._lock:
            return newest(r for r in self.runs.values() if r.user == user)

    def fail(self, run_id: str, error: str, *, counts: bool, execution: str | None = None) -> bool:
        return self._end(run_id, "failed", error, counts, execution, None)

    def stop(
        self,
        run_id: str,
        error: str,
        *,
        counts: bool,
        cause: StopCause,
        execution: str | None = None,
    ) -> bool:
        return self._end(run_id, "stopped", error, counts, execution, cause)

    def _end(
        self,
        run_id: str,
        state: Ending,
        error: str,
        counts: bool,
        execution: str | None,
        cause: StopCause | None,
    ) -> bool:
        with self._lock:
            run = self.runs.get(run_id)
            if run is None or not run.active:
                return False
            if execution is not None and run.execution != execution:
                return False
            if not counts and run.counts:
                key = f"{run.user}_{run.day}"
                self.quota[key] = max(0, self.quota.get(key, 0) - 1)
            self.runs[run_id] = run.model_copy(update=ended(run, state, error, counts, cause))
            lock = run.rejudge_lock
            if lock and self.locks.get(lock, {}).get("run_id") == run.run_id:
                self.locks[lock] = lock_doc(run, "failed")
            return True

    def resume(self, run_id: str, limits: Limits, **fields: Any) -> str | None:
        with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                return "No such run."
            active = [r for r in self.runs.values() if r.active]
            error = resume_error(run) or limit_error(run.user, active, 0, limits)
            if error:
                return error
            self.runs[run_id] = run.model_copy(update=resumed(run, fields))
            return None
