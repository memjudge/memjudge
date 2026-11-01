from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from memjudge import harness
from memjudge.canonical import file_hash
from memjudge.connectors.base import BaseConnector, Retrieval, Usage
from memjudge.connectors.loader import connector_ref, load_connector
from memjudge.datasets.loader import load_dataset
from memjudge.harness import (
    CONVERSATIONS_FILE,
    RESULT_FILE,
    ReusedIngestion,
    Runner,
    RunOptions,
    git_commit,
)
from memjudge.judge.client import Completion, JudgeError
from memjudge.records import (
    BatchRecord,
    ConversationRecord,
    QuestionRecord,
    UsageRecord,
    read_jsonl,
)
from memjudge.result import RunResult
from memjudge.schema import Session
from memjudge.sizes import TOP_K, questions_file
from tests.conftest import FakeCompleter, fake_judge_setup


def _run(
    tiny_path: Path,
    out: Path,
    instance: BaseConnector | None = None,
    completer: Any | None = None,
    **opts: Any,
) -> RunResult:
    loaded = load_dataset(str(tiny_path))
    conn = load_connector("baselines/bm25")
    runner = Runner(
        loaded,
        conn,
        instance or conn.instantiate({}),
        fake_judge_setup(completer),
        out,
        RunOptions(**opts),
        run_id="20240101T000000Z-abcdef",
    )
    return runner.run()


def test_end_to_end_bm25(tiny_path: Path, tmp_path: Path) -> None:
    result = _run(tiny_path, tmp_path, budget_tokens=4096, concurrency=2)
    assert (tmp_path / RESULT_FILE).exists()
    assert RunResult.model_validate_json((tmp_path / RESULT_FILE).read_text("utf-8")) == result
    m = result.metrics["top30"]
    assert m.overall.questions == 5
    # BM25 finds Biscuit, Bristol and genmaicha; marathon question returns both May and
    # October turns (stale); abstention context mentions Biscuit (Ana's dog) -> incorrect.
    assert m.overall.correct >= 3
    assert m.overall.evidence_recall.n == 4  # BM25 reports source turn ids
    assert m.by_category["knowledge-update"].stale_fact_rate.n == 2
    assert m.ingestion.conversations == 2 and m.ingestion.turns == 6
    assert m.error_rate.value == 0
    assert result.dataset.hash == load_dataset(str(tiny_path)).hash
    assert result.connector.code_hash.startswith("sha256:")
    assert result.judge.served_by == ["fake-model"] and not result.judge.official
    assert {a.name for a in result.artifacts} == {
        CONVERSATIONS_FILE,
        "questions-top10.jsonl",
        "questions-top30.jsonl",
    }
    for a in result.artifacts:
        assert a.sha256 == file_hash(tmp_path / a.name)
    top10 = result.metrics["top10"]
    assert top10.overall.questions == 5 and top10.ingestion == m.ingestion  # shared ingestion


def test_each_size_gets_its_own_retrieval(tiny_path: Path, tmp_path: Path) -> None:
    seen: list[tuple[str, int]] = []

    class Sized(BaseConnector):
        def reset(self, namespace: str) -> None: ...
        def ingest(self, namespace: str, session: Session) -> Usage:
            return Usage()

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            seen.append((query, top_k))
            return Retrieval(context=f"{top_k} items")

    _run(tiny_path, tmp_path, instance=Sized({}, {}), concurrency=1)
    for top_k in TOP_K:
        rows = read_jsonl(tmp_path / questions_file(top_k), QuestionRecord)
        assert len(rows) == 5 and {r.context for r in rows} == {f"{top_k} items"}
    # Every question is asked at both sizes, and the first size alternates (ADR 0018).
    loaded = load_dataset(str(tiny_path))
    first = [seen[i][1] for i in range(0, len(seen), 2)]
    assert len(seen) == 2 * len(loaded.dataset.questions)
    assert first[:2] == [10, 30]
    assert all({seen[i][1], seen[i + 1][1]} == {10, 30} for i in range(0, len(seen), 2))


def test_resume_needs_both_sizes(tiny_path: Path, tmp_path: Path) -> None:
    _run(tiny_path, tmp_path)
    # c2's Top 10 records are lost: the conversation is not complete and is redone.
    path = tmp_path / questions_file(10)
    kept = [r for r in read_jsonl(path, QuestionRecord) if r.conversation_id != "c2"]
    path.write_text("".join(r.model_dump_json() + "\n" for r in kept), encoding="utf-8")
    loaded = load_dataset(str(tiny_path))
    runner = Runner(
        loaded,
        load_connector("baselines/bm25"),
        load_connector("baselines/bm25").instantiate({}),
        fake_judge_setup(),
        tmp_path,
        RunOptions(),
    )
    assert runner._completed_conversations() == {"c1"}


def test_budget_is_enforced(tiny_path: Path, tmp_path: Path) -> None:
    class Verbose(BaseConnector):
        def reset(self, namespace: str) -> None: ...
        def ingest(self, namespace: str, session: Session) -> Usage:
            return Usage()

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            return Retrieval(context="Biscuit " * 500)

    result = _run(tiny_path, tmp_path, instance=Verbose({}, {}), budget_tokens=50)
    rows = read_jsonl(tmp_path / questions_file(30), QuestionRecord)
    assert all(r.context_tokens <= 50 and r.truncated for r in rows)
    assert rows[0].raw_context_tokens > 50
    assert result.metrics["top30"].overall.truncation_rate == 1.0


def test_ingest_failure_stops_the_run_and_resume_redoes_it(tiny_path: Path, tmp_path: Path) -> None:
    down = [True]  # the vendor is down for c2 on the first attempt only
    calls: list[str] = []

    class Flaky(BaseConnector):
        def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
            super().__init__(keys, settings)
            self.inner = load_connector("baselines/bm25").instantiate({})

        def reset(self, namespace: str) -> None:
            self.inner.reset(namespace)

        def ingest(self, namespace: str, session: Session) -> Usage:
            calls.append(namespace)
            if down[0] and namespace.endswith("-c2"):
                raise TimeoutError("vendor timed out")
            return self.inner.ingest(namespace, session)

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            return self.inner.retrieve(namespace, query, asked_at, budget, top_k)

    with pytest.raises(harness.RunStoppedError, match=r"conversation c2. TimeoutError: vendor"):
        _run(tiny_path, tmp_path, instance=Flaky({}, {}), concurrency=1)
    convs = read_jsonl(tmp_path / CONVERSATIONS_FILE, ConversationRecord)
    assert [c.conversation_id for c in convs] == ["c1"]  # c2 is not complete

    down[0] = False
    calls.clear()
    result = _run(tiny_path, tmp_path, instance=Flaky({}, {}), resumes=1)
    assert calls and all(ns.endswith("-c2") for ns in calls)  # only c2 is redone
    assert (
        result.metrics["top30"].overall.questions == 5
        and result.metrics["top30"].overall.errors == 0
    )
    assert result.resumes == 1


@pytest.mark.parametrize("failures", [1, harness.MAX_FAILURES_IN_ROW])
def test_retrieval_failures_in_a_row_stop_the_run(
    tiny_path: Path, tmp_path: Path, failures: int
) -> None:
    calls = [0]

    class Flaky(BaseConnector):
        def reset(self, namespace: str) -> None: ...
        def ingest(self, namespace: str, session: Session) -> Usage:
            return Usage()

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            calls[0] += 1
            if calls[0] <= failures:
                raise ConnectionError("vendor down")
            return Retrieval(context="")

    if failures < harness.MAX_FAILURES_IN_ROW:  # one failure is recorded; the run goes on
        result = _run(tiny_path, tmp_path, instance=Flaky({}, {}), concurrency=1)
        # the first call is the first question's Top 10
        assert result.metrics["top10"].overall.errors == 1
        assert result.metrics["top30"].overall.errors == 0
        return
    with pytest.raises(harness.RunStoppedError, match="5 retrievals failed in a row"):
        _run(tiny_path, tmp_path, instance=Flaky({}, {}), concurrency=1)
    convs = read_jsonl(tmp_path / CONVERSATIONS_FILE, ConversationRecord)
    assert "c2" not in {c.conversation_id for c in convs}  # in progress at the stop


def test_ingest_progress_counts_sessions(tiny_path: Path, tmp_path: Path) -> None:
    class Quiet(BaseConnector):
        def reset(self, namespace: str) -> None: ...
        def ingest(self, namespace: str, session: Session) -> Usage:
            return Usage()

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            return Retrieval(context="")

    loaded = load_dataset(str(tiny_path))
    total = sum(len(c.sessions) for c in loaded.dataset.conversations)
    conn = load_connector("baselines/bm25")
    seen: list[tuple[int, int]] = []
    Runner(loaded, conn, Quiet({}, {}), None, tmp_path, RunOptions()).execute(
        ingest_progress=lambda d, t: seen.append((d, t))
    )
    assert seen[0] == (0, total) and seen[-1] == (total, total)


def test_judge_failure_is_recorded_not_raised(tiny_path: Path, tmp_path: Path) -> None:
    class Failing:
        def complete(self, system: str, user: str) -> Completion:
            raise RuntimeError("boom")  # not a JudgeError: a client bug or unexpected reply

    result = _run(tiny_path, tmp_path, completer=Failing())
    assert (
        result.metrics["top30"].overall.errors == 5
        and result.metrics["top30"].overall.accuracy == 0
    )
    for top_k in TOP_K:
        rows = read_jsonl(tmp_path / questions_file(top_k), QuestionRecord)
        assert len(rows) == 5
        assert all(
            r.correct is None and "RuntimeError: boom" in (r.judge_error or "") for r in rows
        )


def test_git_commit_ignores_foreign_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = Path(harness.__file__).resolve().parents[2]

    def fake_run(*args: Any, toplevel: Path, **kwargs: Any) -> Any:
        return type("Out", (), {"stdout": f"{toplevel}\nabc123\n"})()

    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: fake_run(toplevel=checkout))
    assert git_commit() == "abc123"
    # Installed in a venv inside someone else's repository: that commit is not ours.
    monkeypatch.setattr(
        harness.subprocess, "run", lambda *a, **k: fake_run(toplevel=checkout.parent)
    )
    assert git_commit() is None


def test_resume_skips_completed_conversations(tiny_path: Path, tmp_path: Path) -> None:
    _run(tiny_path, tmp_path)
    # Simulate a crash before conversation c2 completed.
    lines = (tmp_path / CONVERSATIONS_FILE).read_text("utf-8").splitlines()
    kept = [line for line in lines if json.loads(line)["conversation_id"] != "c2"]
    (tmp_path / CONVERSATIONS_FILE).write_text("\n".join(kept) + "\n", encoding="utf-8")

    calls: list[str] = []

    class Counting(BaseConnector):
        def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
            super().__init__(keys, settings)
            self.inner = load_connector("baselines/bm25").instantiate({})

        def reset(self, namespace: str) -> None:
            self.inner.reset(namespace)

        def ingest(self, namespace: str, session: Session) -> Usage:
            calls.append(namespace)
            return self.inner.ingest(namespace, session)

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            return self.inner.retrieve(namespace, query, asked_at, budget, top_k)

    result = _run(tiny_path, tmp_path, instance=Counting({}, {}))
    assert calls and all(ns.endswith("-c2") for ns in calls)
    assert result.metrics["top30"].overall.questions == 5  # re-written records are de-duplicated


def test_fake_judge_counts_calls(tiny_path: Path, tmp_path: Path) -> None:
    fake = FakeCompleter()
    loaded = load_dataset(str(tiny_path))
    conn = load_connector("baselines/bm25")
    Runner(loaded, conn, conn.instantiate({}), fake_judge_setup(fake), tmp_path, RunOptions()).run()
    assert fake.calls == 10  # 5 questions, 2 sizes


def test_retrieval_only_then_judge_matches_full_run(tiny_path: Path, tmp_path: Path) -> None:
    full = _run(tiny_path, tmp_path / "full")

    loaded = load_dataset(str(tiny_path))
    conn = load_connector("baselines/bm25")
    out = tmp_path / "split"
    runner = Runner(
        loaded,
        conn,
        conn.instantiate({}),
        None,
        out,
        RunOptions(),
        run_id="20240101T000000Z-abcdef",
    )
    runner.execute()
    by_id = {q.id: q for q in loaded.dataset.questions}
    for top_k in TOP_K:
        retrieved = read_jsonl(out / questions_file(top_k), QuestionRecord)
        assert len(retrieved) == 5 and all(r.correct is None for r in retrieved)
        # The hosted runner judges in another process, knowing the connector only by reference.
        judged = harness.judge_records(retrieved, by_id, fake_judge_setup())
        (out / questions_file(top_k)).write_text(
            "".join(r.model_dump_json() + "\n" for r in judged), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="no judge"):
        runner.finalize(datetime.now())

    tool = harness.ToolInfo(version="9.9.9", commit="abc123", image_digest="sha256:feed")
    result = harness.build_result(
        out,
        run_id="20240101T000000Z-abcdef",
        started=datetime.now(),
        dataset=loaded,
        connector=connector_ref("baselines/bm25"),
        settings=conn.settings(),
        judge_setup=fake_judge_setup(),
        options=RunOptions(),
        mode="hosted",
        tool=tool,
    )
    assert result.mode == "hosted" and result.tool == tool
    assert result.connector.code_hash == full.connector.code_hash
    assert result.metrics["top30"].overall.correct == full.metrics["top30"].overall.correct


def test_judge_records_retries_failed_judgements(tiny_path: Path) -> None:
    class Flaky(FakeCompleter):
        def complete(self, system: str, user: str) -> Completion:
            if self.calls < 2:
                self.calls += 1
                raise JudgeError("overloaded")
            return super().complete(system, user)

    loaded = load_dataset(str(tiny_path))
    q = loaded.dataset.questions[0]
    rec = QuestionRecord(**harness.question_fields(q), context="some context")
    by_id = {q.id: q}

    once = harness.judge_records([rec], by_id, fake_judge_setup(Flaky()))
    assert once[0].judge_error and once[0].correct is None

    seen: list[tuple[int, int]] = []
    retried = harness.judge_records(
        [rec],
        by_id,
        fake_judge_setup(Flaky()),
        retry_rounds=2,
        progress=lambda d, t: seen.append((d, t)),
    )
    assert retried[0].judge_error is None and retried[0].correct is not None
    assert seen == [(1, 1)]  # progress counts the first round only


def test_retried_judgement_keeps_the_spend_of_failed_attempts(tiny_path: Path) -> None:
    class Garbled(FakeCompleter):
        """The first two replies (one judge call's parse attempts) are not JSON."""

        def complete(self, system: str, user: str) -> Completion:
            if self.calls < 2:
                self.calls += 1
                return Completion(text="not json", input_tokens=100, output_tokens=10)
            return super().complete(system, user)

    loaded = load_dataset(str(tiny_path))
    q = loaded.dataset.questions[0]
    rec = QuestionRecord(**harness.question_fields(q), context="some context")
    completer = Garbled()
    seen: list[QuestionRecord] = []
    out = harness.judge_records(
        [rec], {q.id: q}, fake_judge_setup(completer), retry_rounds=1, on_record=seen.append
    )
    assert out[0].judge_error is None and completer.calls == 3
    assert out[0].judge_input_tokens == 300 and out[0].judge_output_tokens == 30
    assert out[0].judge_attempts == 3
    assert [r.judge_input_tokens for r in seen] == [200, 300]  # every round is reported


def test_connector_ref_does_not_import_code() -> None:
    ref = connector_ref("baselines/bm25")
    assert not hasattr(ref, "cls")
    assert ref.code_hash == load_connector("baselines/bm25").code_hash


def test_batch_ingestion_replaces_per_session_ingest(tiny_path: Path, tmp_path: Path) -> None:
    calls: list[str] = []

    class Batching(BaseConnector):
        def ingest_batch(
            self,
            conversations: Mapping[str, Any],
            state_dir: Path,
            progress: Callable[[int, int], None] | None = None,
        ) -> Usage | None:
            calls.append(f"batch:{len(conversations)}")
            assert state_dir == tmp_path
            return Usage(api_calls=3, input_tokens=1000, output_tokens=100)

        def reset(self, namespace: str) -> None:
            calls.append("reset")

        def ingest(self, namespace: str, session: Session) -> Usage:
            raise AssertionError("a batched run must not ingest session by session")

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            return Retrieval(context="")

    result = _run(tiny_path, tmp_path, instance=Batching({}, {}))
    assert calls[0] == "batch:2"
    assert calls.count("reset") == 2  # each conversation is still cleaned up after judging
    ing = result.metrics["top30"].ingestion
    batch = json.loads((tmp_path / "ingest_batch.json").read_text("utf-8"))
    assert ing.total_s >= batch["seconds"] and batch["usage"]["api_calls"] == 3
    assert ing.api_calls == 3 and ing.input_tokens == 1000 and ing.output_tokens == 100
    assert ing.mode == "batch"
    assert "ingest_batch.json" in {a.name for a in result.artifacts}


def test_reusing_a_batch_run_keeps_its_ingestion(tiny_path: Path, tmp_path: Path) -> None:
    loaded = load_dataset(str(tiny_path))
    records = {
        c.id: ConversationRecord(
            conversation_id=c.id,
            sessions=len(c.sessions),
            turns=c.turn_count(),
            ingest_seconds=0.0,
            session_seconds=[],
            ingest_calls=0,
            ingest_errors=0,
            usage=UsageRecord(),
            namespace=f"old-{c.id}",
        )
        for c in loaded.dataset.conversations
    }
    batch = BatchRecord(seconds=42.0, usage=UsageRecord(api_calls=3))
    conn = load_connector("baselines/bm25")
    result = Runner(
        loaded,
        conn,
        conn.instantiate({}),
        fake_judge_setup(),
        tmp_path,
        RunOptions(),
        run_id="20240101T000000Z-abcdef",
        reuse=ReusedIngestion("20230101T000000Z-old000", "sha256:old", records, batch),
    ).run()
    ing = result.metrics["top30"].ingestion
    assert ing.mode == "batch" and ing.total_s == 42.0 and ing.api_calls == 3


def test_batch_progress_reaches_the_caller(tiny_path: Path, tmp_path: Path) -> None:
    class Batching(BaseConnector):
        def ingest_batch(
            self,
            conversations: Mapping[str, Any],
            state_dir: Path,
            progress: Callable[[int, int], None] | None = None,
        ) -> Usage | None:
            assert progress is not None
            progress(1, 4)
            progress(4, 4)
            return Usage(api_calls=1)

        def reset(self, namespace: str) -> None: ...

        def ingest(self, namespace: str, session: Session) -> Usage:
            raise AssertionError("a batched run must not ingest session by session")

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            return Retrieval(context="")

    loaded = load_dataset(str(tiny_path))
    conn = load_connector("baselines/bm25")
    runner = Runner(
        loaded,
        conn,
        Batching({}, {}),
        fake_judge_setup(None),
        tmp_path,
        RunOptions(),
        run_id="20240101T000000Z-abcdef",
    )
    turns: list[tuple[int, int]] = []
    sessions: list[tuple[int, int]] = []
    runner.run(
        ingest_progress=lambda d, t: sessions.append((d, t)),
        batch_progress=lambda d, t: turns.append((d, t)),
    )
    assert turns == [(1, 4), (4, 4)]
    assert sessions[-1] == (3, 3)  # tiny.json has 3 sessions, counted when the batch ends


def test_failed_batch_stops_the_run(tiny_path: Path, tmp_path: Path) -> None:
    class Broken(BaseConnector):
        def ingest_batch(
            self,
            conversations: Mapping[str, Any],
            state_dir: Path,
            progress: Callable[[int, int], None] | None = None,
        ) -> Usage | None:
            raise RuntimeError("batch failed")

        def reset(self, namespace: str) -> None: ...

        def ingest(self, namespace: str, session: Session) -> Usage:
            raise AssertionError("a failed batch must not fall back to per-session ingest")

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            raise AssertionError("not reached")

    with pytest.raises(RuntimeError, match="batch failed"):
        _run(tiny_path, tmp_path, instance=Broken({}, {}))
    assert not (tmp_path / "ingest_batch.json").exists()


def test_wait_all_raises_the_first_error() -> None:
    from concurrent.futures import ThreadPoolExecutor

    def boom() -> None:
        raise ValueError("conversation failed")

    with ThreadPoolExecutor(2) as pool, pytest.raises(ValueError, match="conversation failed"):
        harness._wait_all([pool.submit(lambda: None), pool.submit(boom)])


def test_without_batch_nothing_changes(tiny_path: Path, tmp_path: Path) -> None:
    result = _run(tiny_path, tmp_path)
    assert not (tmp_path / "ingest_batch.json").exists()
    assert "ingest_batch.json" not in {a.name for a in result.artifacts}
    assert result.metrics["top30"].ingestion.mode == "per-session"


def test_read_jsonl_keeps_unicode_line_separators(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class Row(BaseModel):
        text: str

    texts = ["first\u2028still first", "second\u0085\x0cstill second"]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(Row(text=t).model_dump_json() + "\n" for t in texts), "utf-8")
    assert [r.text for r in read_jsonl(path, Row)] == texts
