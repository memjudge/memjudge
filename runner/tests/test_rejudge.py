"""Re-judging a published run into the current season (ADR 0015), with fakes."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from conftest import GOOD_JUDGE_KEY, run_request, session
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fakes import FakePublisher, GoldCompleter, official_judge
from fastapi.testclient import TestClient
from memjudge.judge import setup as setup_mod
from memjudge.judge.setup import pinned_judge, seasons
from memjudge.signing import sign_payload

from memjudge_runner.api import Services
from memjudge_runner.config import finish_job_name
from memjudge_runner.finish import FinishServices, run_finish
from memjudge_runner.publish import envelope_text
from memjudge_runner.retrieve import run_retrieve
from memjudge_runner.runs import REJUDGE_ACTIVE, REJUDGE_DONE

OLD_SEASON = "0-old"
ORIGINAL_HISTORY = [
    {"at": "2026-10-01T10:00:00Z", "event": "stopped", "cause": "judge"},
    {"at": "2026-10-01T11:00:00Z", "event": "resumed", "cause": None},
]


@pytest.fixture(autouse=True)
def _older_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """An earlier season to re-judge from, before the real ones."""
    chain = (dataclasses.replace(seasons()[0], id=OLD_SEASON, started="2026-01-01"), *seasons())
    monkeypatch.setattr(setup_mod, "seasons", lambda: chain)


def _finish(services: Services, publisher: FakePublisher) -> FinishServices:
    return FinishServices(
        settings=services.settings,
        store=services.store,
        cipher=services.cipher,
        blobs=services.blobs,
        options=services.options,
        publisher=publisher,
        judge_setup=lambda route, key: official_judge(GoldCompleter()),
    )


def _published(
    client: TestClient,
    services: Services,
    key: Ed25519PrivateKey,
    publisher: FakePublisher,
    season: str = OLD_SEASON,
) -> dict[str, Any]:
    """A hosted run published under ``season``; returns its signed envelope."""
    resp = client.post("/runs", json=run_request(), headers=session("9"))
    run_id = resp.json()["run_id"]
    bundle = services.jobs.started[-1][1][2]  # type: ignore[attr-defined]
    s = services
    assert run_retrieve(run_id, bundle, s.settings, s.cipher, s.blobs, client) == 0
    assert run_finish(run_id, _finish(services, publisher)) == 0
    path = f"runs/{run_id}.json"
    payload = json.loads(publisher.commits[path])["payload"]
    payload["judge"]["season"] = season  # the fake judge has no real season
    payload["resumes"], payload["history"] = 1, ORIGINAL_HISTORY
    envelope = sign_payload(payload, key)
    publisher.commits[path] = envelope_text(envelope)
    return envelope


def _rejudge(client: TestClient, envelope: dict[str, Any], user: str = "1001") -> Any:
    body = {
        "envelope": envelope,
        "judge_route": "anthropic",
        "judge_key": GOOD_JUDGE_KEY,
        "terms": "2026-10-17",
    }
    run_id = envelope["payload"]["run_id"]
    return client.post(f"/runs/{run_id}/rejudge", json=body, headers=session(user))


def test_rejudge_end_to_end(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    publisher = FakePublisher()
    original = _published(client, services, signing_key, publisher)
    orig_id = original["payload"]["run_id"]

    resp = _rejudge(client, original)
    assert resp.status_code == 200, resp.text
    new_id = resp.json()["run_id"]
    assert services.jobs.started[-1] == (  # type: ignore[attr-defined]
        finish_job_name(services.settings.version),
        ["finish", new_id],
    )
    season = pinned_judge().season
    rejudges = client.get("/rejudges").json()
    assert rejudges == {"season": season, "runs": {orig_id: {"state": "active", "run_id": new_id}}}
    assert _rejudge(client, original, user="2002").json()["detail"] == REJUDGE_ACTIVE

    assert run_finish(new_id, _finish(services, publisher)) == 0
    status = client.get(f"/runs/{new_id}", headers=session()).json()
    assert status["state"] == "done" and status["rejudge_of"] == orig_id
    payload = json.loads(publisher.commits[f"runs/{new_id}.json"])["payload"]
    assert payload["rejudged_from"] == {"run_id": orig_id, "season": OLD_SEASON}
    assert payload["connector"] == original["payload"]["connector"]
    assert payload["affiliation"] == original["payload"]["affiliation"]
    assert (payload["resumes"], payload["history"]) == (1, ORIGINAL_HISTORY)
    for size in ("top10", "top30"):
        assert payload["metrics"][size]["overall"]["questions"] == 5
    assert {
        f"runs/{new_id}/questions-top10.jsonl",
        f"runs/{new_id}/questions-top30.jsonl",
        f"runs/{new_id}/conversations.jsonl",
    } <= set(publisher.artifacts)
    assert client.get("/rejudges").json()["runs"][orig_id]["state"] == "done"
    resp = _rejudge(client, original, user="2002")
    assert (resp.status_code, resp.json()["detail"]) == (409, REJUDGE_DONE)


def test_finish_uses_the_published_original_not_the_copy_sent(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    publisher = FakePublisher()
    original = _published(client, services, signing_key, publisher)
    sent = json.loads(json.dumps(original))
    sent["payload"]["affiliation"] = {"affiliated": True, "system": "Recalld"}
    new_id = _rejudge(client, sent).json()["run_id"]
    assert run_finish(new_id, _finish(services, publisher)) == 0
    payload = json.loads(publisher.commits[f"runs/{new_id}.json"])["payload"]
    assert payload["affiliation"] == original["payload"]["affiliation"]


def test_tampered_published_file_fails_and_frees_the_lock(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    publisher = FakePublisher()
    original = _published(client, services, signing_key, publisher)
    orig_id = original["payload"]["run_id"]
    publisher.artifacts[f"runs/{orig_id}/questions-top10.jsonl"] += b"\n"
    new_id = _rejudge(client, original).json()["run_id"]
    assert run_finish(new_id, _finish(services, publisher)) == 1
    status = client.get(f"/runs/{new_id}", headers=session()).json()
    assert status["state"] == "failed" and "does not match its hash" in status["error"]
    assert not any(k.startswith(f"runs/{new_id}") for k in publisher.commits)
    assert client.get("/rejudges").json()["runs"][orig_id]["state"] == "failed"
    assert _rejudge(client, original, user="2002").status_code == 200  # free again


def test_a_stopped_rejudge_frees_the_lock_and_cannot_be_resumed(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    publisher = FakePublisher()
    original = _published(client, services, signing_key, publisher)
    new_id = _rejudge(client, original).json()["run_id"]
    assert client.post(f"/runs/{new_id}/stop", headers=session()).json()["state"] == "stopped"
    body = {"connector_keys": {}, "judge_key": GOOD_JUDGE_KEY}
    resp = client.post(f"/runs/{new_id}/resume", json=body, headers=session())
    assert resp.status_code == 409 and "cannot be resumed" in resp.json()["detail"]
    assert _rejudge(client, original, user="2002").status_code == 200


def test_a_run_already_in_the_current_season_is_refused(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    publisher = FakePublisher()
    current = _published(client, services, signing_key, publisher, season=pinned_judge().season)
    resp = _rejudge(client, current)
    assert resp.status_code == 409 and "already in the current season" in resp.json()["detail"]


def test_a_rerun_season_in_between_needs_a_new_run(
    client: TestClient,
    services: Services,
    signing_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = FakePublisher()
    original = _published(client, services, signing_key, publisher)
    chain = (*setup_mod.seasons(), dataclasses.replace(seasons()[-1], id="next", upgrade="re-run"))
    monkeypatch.setattr(setup_mod, "seasons", lambda: chain)
    resp = _rejudge(client, original)
    assert resp.status_code == 409 and "needs a new run" in resp.json()["detail"]


def test_the_result_must_be_for_the_run_in_the_url(
    client: TestClient, services: Services, signing_key: Ed25519PrivateKey
) -> None:
    publisher = FakePublisher()
    original = _published(client, services, signing_key, publisher)
    body = {"envelope": original, "judge_route": "anthropic", "judge_key": GOOD_JUDGE_KEY}
    resp = client.post(
        "/runs/20260101T000000Z-aaaaaa/rejudge",
        json={**body, "terms": "2026-10-17"},
        headers=session(),
    )
    assert resp.status_code == 422
    resp = client.post(
        "/runs/x/rejudge",
        json={**body, "envelope": {"payload": {}}, "terms": "2026-10-17"},
        headers=session(),
    )
    assert resp.status_code == 422 and resp.json()["detail"] == "That is not a MemJudge result."
