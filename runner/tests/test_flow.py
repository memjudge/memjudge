"""A whole hosted run with fakes: form request -> retrieve job -> finish job -> published."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import GOOD_JUDGE_KEY, run_request, session
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fakes import TINY, FakePublisher, GoldCompleter, official_judge
from fastapi.testclient import TestClient
from memjudge.canonical import sha256_hex
from memjudge.connectors.base import BaseConnector, ConnectorSpec, Retrieval, Usage
from memjudge.connectors.loader import ConnectorRef, connector_ref, load_connector, load_spec
from memjudge.datasets.loader import load_dataset
from memjudge.harness import Runner
from memjudge.judge.client import Completion, JudgeError
from memjudge.judge.judge import Completer
from memjudge.schema import Session
from memjudge.signing import key_id, verify_envelope

from memjudge_runner.api import Services
from memjudge_runner.config import finish_job_name, job_name
from memjudge_runner.finish import FinishServices, run_finish
from memjudge_runner.retrieve import run_retrieve


def _start(client: TestClient, services: Services) -> tuple[str, str]:
    resp = client.post("/runs", json=run_request(), headers=session())
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    job, args = services.jobs.started[-1]  # type: ignore[attr-defined]
    assert job == job_name("baselines/bm25", services.settings.version)
    assert args[:2] == ["retrieve", run_id] and len(args) == 3  # no keys in job arguments
    return run_id, args[2]


def _finish_services(
    services: Services,
    publisher: FakePublisher,
    fail: bool = False,
    completer: Completer | None = None,
) -> FinishServices:
    return FinishServices(
        settings=services.settings,
        store=services.store,
        cipher=services.cipher,
        blobs=services.blobs,
        options=services.options,
        publisher=publisher,
        judge_setup=lambda route, key: official_judge(completer or GoldCompleter(fail=fail)),
    )


def _resume(client: TestClient, services: Services, run_id: str) -> list[str]:
    body = {"connector_keys": {}, "judge_key": GOOD_JUDGE_KEY}
    resp = client.post(f"/runs/{run_id}/resume", json=body, headers=session())
    assert resp.status_code == 200, resp.text
    _, args = services.jobs.started[-1]  # type: ignore[attr-defined]
    return args


def _retrieve(client: TestClient, services: Services, run_id: str, bundle: str) -> int:
    s = services
    return run_retrieve(run_id, bundle, s.settings, s.cipher, s.blobs, client)


def test_hosted_run_end_to_end(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id, bundle = _start(client, services)
    blobs = services.blobs

    caplog.set_level(logging.INFO)
    assert run_retrieve(run_id, bundle, services.settings, services.cipher, blobs, client) == 0
    logged = [r.getMessage() for r in caplog.records if r.name.startswith("memjudge")]
    assert logged[:3] == [
        f"retrieve job started for run {run_id}",
        f"loading the dataset ({TINY})",
        "starting the connector (baselines/bm25)",
    ]
    assert logged[-1] == "retrieval complete: judging starts in its own job"
    assert ("secrets", bundle) not in blobs.objects  # type: ignore[attr-defined]  # read once
    assert services.jobs.started[-1][0] == finish_job_name(services.settings.version)  # type: ignore[attr-defined]
    retrieved = client.get(f"/runs/{run_id}", headers=session()).json()
    assert retrieved["state"] == "judging"
    sessions = sum(len(c.sessions) for c in load_dataset(str(TINY)).dataset.conversations)
    assert (retrieved["ingested"], retrieved["ingest_total"]) == (sessions, sessions)
    assert retrieved["ingest_unit"] == "sessions"

    publisher = FakePublisher()
    assert run_finish(run_id, _finish_services(services, publisher)) == 0

    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert status["state"] == "done" and status["job_started"] is True  # finish reported in
    assert status["result_url"] == f"https://memjudge.test/runs/{run_id}/"

    envelope = json.loads(publisher.commits[f"runs/{run_id}.json"])
    pub = signing_key.public_key()
    assert verify_envelope(envelope, {key_id(pub): pub}) == key_id(pub)
    payload = envelope["payload"]
    assert payload["mode"] == "hosted" and payload["run_id"] == run_id
    assert payload["tool"]["image_digest"] == "sha256:feed"
    assert payload["tool"]["commit"] == "abc123"
    assert payload["judge"]["official"] is True
    assert payload["affiliation"] == {"affiliated": False, "system": None}
    for art in payload["artifacts"]:
        assert art["url"] == f"https://artifacts.memjudge.test/runs/{run_id}/{art['name']}"
        data = publisher.artifacts[f"runs/{run_id}/{art['name']}"]
        assert art["sha256"] == "sha256:" + sha256_hex(data)
    for size in ("top10", "top30"):  # one score per result size (ADR 0018)
        assert payload["metrics"][size]["overall"]["questions"] == 5
    assert payload["resumes"] == 0 and payload["history"] == []
    run = services.store.get(run_id)
    assert run is not None and len(run.executions) == 2  # retrieve, then finish
    assert GOOD_JUDGE_KEY not in json.dumps(envelope)
    assert all(GOOD_JUDGE_KEY.encode() not in d for d in publisher.artifacts.values())


def test_connector_max_concurrency_caps_the_hosted_run(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capped_spec(cid: str) -> ConnectorSpec:
        return load_spec(cid).model_copy(update={"max_concurrency": 2})

    def capped_ref(cid: str) -> ConnectorRef:
        ref = connector_ref(cid)
        return replace(ref, spec=ref.spec.model_copy(update={"max_concurrency": 2}))

    used: list[int] = []

    class Recording(Runner):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            used.append(self.options.concurrency)

    monkeypatch.setattr("memjudge_runner.retrieve.load_spec", capped_spec)
    monkeypatch.setattr("memjudge_runner.retrieve.Runner", Recording)
    monkeypatch.setattr("memjudge_runner.finish.connector_ref", capped_ref)
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 0
    assert used == [2]
    publisher = FakePublisher()
    assert run_finish(run_id, _finish_services(services, publisher)) == 0
    payload = json.loads(publisher.commits[f"runs/{run_id}.json"])["payload"]
    assert payload["config"]["concurrency"] == 2


class OutOfCredit:
    """Judges like GoldCompleter, but fails on the questions in ``failing``; logs the rest."""

    def __init__(self, failing: list[str]) -> None:
        self.failing = failing
        self.seen: list[str] = []
        self._gold = GoldCompleter()

    def complete(self, system: str, user: str) -> Completion:
        if any(text in user for text in self.failing):
            raise JudgeError("credit balance too low")
        self.seen.append(user)
        return self._gold.complete(system, user)


def test_judge_failure_stops_and_resume_judges_only_the_rest(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 0
    texts = [q.question for q in load_dataset(str(TINY)).dataset.questions]
    publisher = FakePublisher()
    first = OutOfCredit(failing=texts[2:])
    assert run_finish(run_id, _finish_services(services, publisher, completer=first)) == 1
    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert status["state"] == "stopped" and status["resume_from"] == "finish"
    assert "judge failed on 6 of 10" in status["error"]  # 3 questions, 2 sizes
    assert not publisher.commits and not publisher.artifacts

    assert _resume(client, services, run_id) == ["finish", run_id]
    second = OutOfCredit(failing=[])
    assert run_finish(run_id, _finish_services(services, publisher, completer=second)) == 0
    assert len(second.seen) == 6 and all(any(t in u for t in texts[2:]) for u in second.seen)
    payload = json.loads(publisher.commits[f"runs/{run_id}.json"])["payload"]
    assert payload["resumes"] == 1 and payload["metrics"]["top30"]["overall"]["questions"] == 5
    assert [(e["event"], e["cause"]) for e in payload["history"]] == [
        ("stopped", "judge"),
        ("resumed", None),
    ]


class Crash(BaseException):
    """The finish job dies (killed, out of memory): nothing inside the job handles it."""


class Counting:
    """Judges like GoldCompleter and counts its calls. Replies to the questions in
    ``garbled`` with text that is no verdict (billed all the same); dies after
    ``crash_after`` calls."""

    def __init__(self, garbled: list[str] | None = None, crash_after: int | None = None) -> None:
        self.garbled = garbled or []
        self.crash_after = crash_after
        self.calls = 0
        self._gold = GoldCompleter()

    def complete(self, system: str, user: str) -> Completion:
        if self.crash_after is not None and self.calls >= self.crash_after:
            raise Crash
        self.calls += 1
        if any(text in user for text in self.garbled):
            return Completion(text="not json", input_tokens=100, output_tokens=10)
        return self._gold.complete(system, user)


def test_a_run_stopped_while_judging_is_not_published(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 0

    class StoppedMeanwhile(GoldCompleter):
        """The owner presses Stop while judging, and the cancel never reaches the job."""

        def complete(self, system: str, user: str) -> Completion:
            services.store.stop(run_id, "Stopped by you.", counts=True, cause="submitter")
            return super().complete(system, user)

    publisher = FakePublisher()
    finish = _finish_services(services, publisher, completer=StoppedMeanwhile())
    assert run_finish(run_id, finish) == 1
    assert publisher.commits == {} and publisher.artifacts == {}
    run = services.store.get(run_id)
    assert run is not None and run.state == "stopped" and run.error == "Stopped by you."


def _judge_input_tokens(publisher: FakePublisher, run_id: str) -> int:
    total = 0
    for size in ("top10", "top30"):
        data = publisher.artifacts[f"runs/{run_id}/questions-{size}.jsonl"]
        total += sum(json.loads(line)["judge_input_tokens"] for line in data.splitlines())
    return total


def test_crash_while_judging_keeps_what_was_judged(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("memjudge_runner.finish.JUDGE_CONCURRENCY", 1)
    monkeypatch.setattr("memjudge_runner.finish.JUDGED_SAVE_SECONDS", 0.0)
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 0
    publisher = FakePublisher()
    first = Counting(crash_after=2)
    with pytest.raises(Crash):
        run_finish(run_id, _finish_services(services, publisher, completer=first))
    services.jobs.ended.add(services.store.get(run_id).execution)  # type: ignore[attr-defined, union-attr]
    assert client.get(f"/runs/{run_id}", headers=session()).json()["state"] == "stopped"

    assert _resume(client, services, run_id) == ["finish", run_id]
    second = Counting()
    assert run_finish(run_id, _finish_services(services, publisher, completer=second)) == 0
    # The two judged answers are not redone: 3 Top 10 and 5 Top 30 answers are left.
    assert (first.calls, second.calls) == (2, 8)
    assert _judge_input_tokens(publisher, run_id) == 100 * (first.calls + second.calls)


def test_resumed_judging_keeps_the_spend_of_failed_attempts(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("memjudge_runner.finish.JUDGE_CONCURRENCY", 1)  # exact call counts
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 0
    texts = [q.question for q in load_dataset(str(TINY)).dataset.questions]
    publisher = FakePublisher()
    first = Counting(garbled=texts[2:])
    assert run_finish(run_id, _finish_services(services, publisher, completer=first)) == 1
    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert "judge failed on 6 of 10" in status["error"]  # 3 questions, 2 sizes

    assert _resume(client, services, run_id) == ["finish", run_id]
    second = Counting()
    assert run_finish(run_id, _finish_services(services, publisher, completer=second)) == 0
    # Per size: 2 good calls, then 3 questions x 2 parse attempts x 3 rounds; 3 calls each
    # on resume.
    assert (first.calls, second.calls) == (40, 6)
    assert _judge_input_tokens(publisher, run_id) == 100 * (first.calls + second.calls)


def test_stopped_retrieval_resumes_where_it_stopped(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    down = [True]  # the memory system fails on c2 during the first attempt
    calls: list[str] = []

    class Flaky(BaseConnector):
        def __init__(self) -> None:
            super().__init__({}, {})
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

    class Loaded:
        def instantiate(self, keys: object, overrides: object = None) -> Flaky:
            return Flaky()

    monkeypatch.setattr("memjudge_runner.retrieve.load_connector", lambda cid: Loaded())
    monkeypatch.setattr("memjudge_runner.retrieve.CONCURRENCY", 1)  # c1 completes, then c2
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 1
    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert status["state"] == "stopped" and status["resume_from"] == "retrieve"
    assert "Ingestion failed for conversation c2" in status["error"]
    saved = services.blobs.objects["staging", f"{run_id}/conversations.jsonl"]  # type: ignore[attr-defined]
    assert b'"c1"' in saved and b'"c2"' not in saved

    down[0] = False
    calls.clear()
    args = _resume(client, services, run_id)
    assert _retrieve(client, services, run_id, args[2]) == 0
    assert calls and all(ns.endswith("-c2") for ns in calls)  # c1 is not run again
    publisher = FakePublisher()
    assert run_finish(run_id, _finish_services(services, publisher)) == 0
    payload = json.loads(publisher.commits[f"runs/{run_id}.json"])["payload"]
    assert payload["resumes"] == 1 and payload["metrics"]["top30"]["overall"]["questions"] == 5


def test_context_with_unicode_line_separators_is_judged(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset text can hold U+2028, which JSON leaves raw: it must not split a record."""

    class Separators(BaseConnector):
        def __init__(self) -> None:
            super().__init__({}, {})
            self.inner = load_connector("baselines/bm25").instantiate({})

        def reset(self, namespace: str) -> None:
            self.inner.reset(namespace)

        def ingest(self, namespace: str, session: Session) -> Usage:
            return self.inner.ingest(namespace, session)

        def retrieve(
            self,
            namespace: str,
            query: str,
            asked_at: datetime | None,
            budget: int,
            top_k: int,
        ) -> Retrieval:
            found = self.inner.retrieve(namespace, query, asked_at, budget, top_k)
            return Retrieval("one\u2028two\u2029three\n" + found.context, found.items, found.usage)

    class Loaded:
        def instantiate(self, keys: object, overrides: object = None) -> Separators:
            return Separators()

    monkeypatch.setattr("memjudge_runner.retrieve.load_connector", lambda cid: Loaded())
    run_id, bundle = _start(client, services)
    assert _retrieve(client, services, run_id, bundle) == 0
    publisher = FakePublisher()
    assert run_finish(run_id, _finish_services(services, publisher)) == 0
    assert client.get(f"/runs/{run_id}", headers=session()).json()["state"] == "done"


def test_crashed_run_resumes_from_scratch_without_saved_progress(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    run_id, _ = _start(client, services)
    services.jobs.ended.add(services.store.get(run_id).execution)  # type: ignore[attr-defined, union-attr]
    assert client.get(f"/runs/{run_id}", headers=session()).json()["state"] == "stopped"
    args = _resume(client, services, run_id)
    assert _retrieve(client, services, run_id, args[2]) == 0  # nothing saved: starts fresh
    publisher = FakePublisher()
    assert run_finish(run_id, _finish_services(services, publisher)) == 0
    assert json.loads(publisher.commits[f"runs/{run_id}.json"])["payload"]["resumes"] == 1


def test_tampered_retrieval_output_is_refused(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    run_id, bundle = _start(client, services)
    run_retrieve(run_id, bundle, services.settings, services.cipher, services.blobs, client)
    blobs = services.blobs.objects  # type: ignore[attr-defined]
    lines = blobs["staging", f"{run_id}/questions-top10.jsonl"].splitlines()
    blobs["staging", f"{run_id}/questions-top10.jsonl"] = b"\n".join(lines[1:])  # drop one
    publisher = FakePublisher()
    assert run_finish(run_id, _finish_services(services, publisher)) == 1
    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert (
        status["state"] == "failed"
        and "1 questions have no retrieval record in questions-top10" in status["error"]
    )
    assert not publisher.commits


def test_retrieve_failure_is_reported_without_keys(
    client: TestClient, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "letta-key-1234567890"
    resp = client.post(
        "/runs",
        json=run_request(connector="letta/cloud", connector_keys={"LETTA_API_KEY": key}),
        headers=session(),
    )
    run_id = resp.json()["run_id"]
    bundle = services.jobs.started[-1][1][2]  # type: ignore[attr-defined]

    def broken(connector_id: str) -> None:
        raise RuntimeError(f"401 Unauthorized for api_key={key}")  # SDKs do echo keys

    monkeypatch.setattr("memjudge_runner.retrieve.wait_for_hosts", lambda *a, **k: None)
    monkeypatch.setattr("memjudge_runner.retrieve.load_connector", broken)
    code = run_retrieve(run_id, bundle, services.settings, services.cipher, services.blobs, client)
    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert code == 1 and status["state"] == "stopped"
    assert (
        status["error"]
        == "Retrieval stopped: RuntimeError: 401 Unauthorized for api_key=[redacted]"
    )
    assert [e.cause for e in services.store.get(run_id).history] == ["retrieval"]  # type: ignore[union-attr]
