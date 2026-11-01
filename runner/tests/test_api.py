from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import GOOD_JUDGE_KEY, SESSION_SECRET, run_request, session
from fastapi.testclient import TestClient
from memjudge.sizes import TOP_K

from memjudge_runner import auth
from memjudge_runner.api import LOG_UNREADABLE, Services, create_app, token_hash
from memjudge_runner.config import (
    AFFILIATED_SYSTEMS,
    MAX_RESUMES,
    RESUME_HOURS,
    finish_job_name,
    job_name,
)
from memjudge_runner.logs import LogLine
from memjudge_runner.runs import now


def _start(client: TestClient, user: str = "1001", **overrides: object) -> httpx.Response:
    return client.post("/runs", json=run_request(**overrides), headers=session(user))


def _resume(
    client: TestClient, run_id: str, user: str = "1001", **overrides: object
) -> httpx.Response:
    body: dict[str, object] = {"connector_keys": {}, "judge_key": GOOD_JUDGE_KEY}
    body.update(overrides)
    return client.post(f"/runs/{run_id}/resume", json=body, headers=session(user))


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_options_are_public(client: TestClient, services: Services) -> None:
    resp = client.get("/options")
    assert resp.status_code == 200
    assert [c["id"] for c in resp.json()["connectors"]] == [
        "baselines/bm25",
        "letta/cloud",
        "recalld/cloud",
    ]
    runner = resp.json()["runner"]
    assert runner == {
        "version": services.settings.version,
        "commit": "abc123",
        "image_digest": "sha256:feed",
    }


def test_starting_a_run_needs_login(client: TestClient) -> None:
    assert client.post("/runs", json=run_request()).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert client.post("/runs", json=run_request(), headers=bad).status_code == 401


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dataset": "other"}, "from the list"),
        ({"connector": "letta/cloud"}, "needs: LETTA_API_KEY"),
        (
            {"connector": "letta/cloud", "connector_keys": {"LETTA_API_KEY": "  "}},
            "needs: LETTA_API_KEY",
        ),
        ({"connector_keys": {"EXTRA": "x"}}, "needs: no keys"),
        ({"affiliation": {"affiliated": True, "system": "Acme"}}, "from the list"),
        ({"affiliation": {"affiliated": False, "system": "Zep"}}, "from the list"),
        ({"judge_key": "sk-ant-wrong"}, "rejected by the provider"),
    ],
)
def test_invalid_requests_are_refused(
    client: TestClient, services: Services, overrides: dict[str, object], message: str
) -> None:
    resp = _start(client, **overrides)
    assert resp.status_code == 422 and message in resp.json()["detail"]
    assert not services.jobs.started  # type: ignore[attr-defined]


def test_presets(client: TestClient, services: Services) -> None:
    keys = {"RECALLD_API_KEY": "rcd_us_key_1234567890"}
    for preset in (None, "nope"):  # a connector with presets needs one of them
        resp = _start(client, connector="recalld/cloud", connector_keys=keys, preset=preset)
        assert resp.status_code == 422 and "preset" in resp.json()["detail"]
    assert _start(client, preset="recall").status_code == 422  # bm25 has none
    resp = _start(client, connector="recalld/cloud", connector_keys=keys, preset="search")
    record = services.store.get(resp.json()["run_id"])
    assert record is not None and record.preset == "search"
    assert record.public()["preset"] == "search"


def test_terms_and_affiliation_are_recorded(client: TestClient, services: Services) -> None:
    assert _start(client, terms=None).status_code == 422  # agreeing to the terms is required
    named = _start(client, affiliation={"affiliated": True, "system": "Recalld"})
    record = services.store.get(named.json()["run_id"])
    assert record is not None and record.terms == "2026-10-17"
    assert record.affiliation == {"affiliated": True, "system": "Recalld"}
    other = _start(client, user="2", affiliation={"affiliated": True})  # another system
    assert other.status_code == 200
    assert client.get("/options").json()["affiliations"] == list(AFFILIATED_SYSTEMS)


def test_users_see_their_own_runs_newest_first(client: TestClient, services: Services) -> None:
    assert client.get("/runs").status_code == 401
    first = _start(client).json()["run_id"]
    services.store.fail(first, "the vendor key was wrong", counts=True)
    second = _start(client).json()["run_id"]
    _start(client, user="2")
    runs = client.get("/runs", headers=session()).json()["runs"]
    assert [r["run_id"] for r in runs] == [second, first]
    assert runs[0]["state"] == "retrieving" and runs[0]["judge_route"] == "anthropic"
    assert runs[1]["state"] == "failed" and "user" not in runs[1]


def test_validation_errors_do_not_echo_keys(client: TestClient) -> None:
    body = run_request(judge_key="sk-ant-secret-value-123", affiliation={"affiliated": "maybe"})
    resp = client.post("/runs", json=body, headers=session())
    assert resp.status_code == 422 and "sk-ant-secret-value-123" not in resp.text


def test_keys_are_sealed_not_stored(client: TestClient, services: Services) -> None:
    resp = _start(
        client, connector="letta/cloud", connector_keys={"LETTA_API_KEY": "letta-key-1234567890"}
    )
    run_id = resp.json()["run_id"]
    record = services.store.get(run_id)
    assert record is not None and record.state == "retrieving"
    stored = record.model_dump_json() + json.dumps(
        {f"{b}/{n}": v.decode() for (b, n), v in services.blobs.objects.items()}  # type: ignore[attr-defined]
    )
    for secret in ("letta-key-1234567890", GOOD_JUDGE_KEY):
        assert secret not in stored  # only ciphertext (the fake base64-encodes it)
    job, args = services.jobs.started[0]  # type: ignore[attr-defined]
    assert job == job_name("letta/cloud", services.settings.version)
    assert args[:2] == ["retrieve", run_id]


def test_limits(client: TestClient, services: Services) -> None:
    assert _start(client, user="1").status_code == 200
    second = _start(client, user="1")
    assert second.status_code == 429 and "already have a run" in second.json()["detail"]
    assert _start(client, user="2").status_code == 200
    assert _start(client, user="3").status_code == 200
    busy = _start(client, user="4")
    assert busy.status_code == 429 and "busy" in busy.json()["detail"]


def test_daily_limit_and_refund_of_crashed_runs(client: TestClient, services: Services) -> None:
    store, jobs = services.store, services.jobs
    for _ in range(2):  # two runs that fail through the user's fault: they count
        run_id = _start(client).json()["run_id"]
        store.fail(run_id, "the vendor key was wrong", counts=True)

    # The third run's job dies without reporting: the run is stopped (resumable) and refunded.
    crashed = _start(client).json()["run_id"]
    jobs.ended.add(store.get(crashed).execution)  # type: ignore[attr-defined, union-attr]
    status = client.get(f"/runs/{crashed}", headers=session()).json()
    assert status["state"] == "stopped" and "does not count" in status["error"]
    assert status["resume_until"] and status["resume_from"] == "retrieve"
    assert [e.cause for e in store.get(crashed).history] == ["crash"]  # type: ignore[union-attr]

    third = _start(client)
    assert third.status_code == 200
    store.fail(third.json()["run_id"], "", counts=True)
    refused = _start(client)
    assert refused.status_code == 429 and "today's limit of 3" in refused.json()["detail"]


def test_runs_are_visible_to_their_owner_only(client: TestClient) -> None:
    run_id = _start(client, user="1").json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers=session("1")).status_code == 200
    assert client.get(f"/runs/{run_id}", headers=session("2")).status_code == 404


def test_internal_endpoints_need_the_run_token(client: TestClient) -> None:
    run_id = _start(client).json()["run_id"]
    for path in ("progress", "uploads", "downloads", "retrieved", "stopped"):
        resp = client.post(
            f"/internal/runs/{run_id}/{path}",
            json={"done": 1, "total": 1, "message": "x"},
            headers={"Authorization": "Bearer guessed"},
        )
        assert resp.status_code == 403


def _finishes(services: Services) -> int:
    finish = finish_job_name(services.settings.version)
    return sum(1 for job, _ in services.jobs.started if job == finish)  # type: ignore[attr-defined]


def test_retrieved_starts_one_finish(client: TestClient, services: Services) -> None:
    run_id = _start(client).json()["run_id"]
    services.store.update(run_id, run_token_hash=token_hash("job-token"))
    job = {"Authorization": "Bearer job-token"}
    first = client.post(f"/internal/runs/{run_id}/retrieved", json={}, headers=job)
    again = client.post(f"/internal/runs/{run_id}/retrieved", json={}, headers=job)
    assert first.status_code == 200 and again.status_code == 403  # the token is spent
    assert _finishes(services) == 1


def test_the_job_reports_in_with_its_first_progress(client: TestClient, services: Services) -> None:
    run_id = _start(client).json()["run_id"]

    def status() -> dict[str, object]:
        return client.get(f"/runs/{run_id}", headers=session()).json()  # type: ignore[no-any-return]

    assert status()["job_started"] is False  # the container is still starting
    services.store.update(run_id, run_token_hash=token_hash("job-token"))
    job = {"Authorization": "Bearer job-token"}
    assert client.post(f"/internal/runs/{run_id}/progress", json={}, headers=job).status_code == 200
    assert status()["job_started"] is True
    client.post(f"/internal/runs/{run_id}/retrieved", json={}, headers=job)
    assert status()["state"] == "judging" and status()["job_started"] is False  # a new job


def test_search_progress_counts_both_sizes(client: TestClient, services: Services) -> None:
    run_id = _start(client).json()["run_id"]
    questions = services.options.datasets[0].questions
    services.store.update(run_id, run_token_hash=token_hash("job-token"))
    job = {"Authorization": "Bearer job-token"}
    done = questions + 1  # past the question count, into the second size
    client.post(f"/internal/runs/{run_id}/progress", json={"done": done}, headers=job)
    status = client.get(f"/runs/{run_id}", headers=session()).json()
    assert (status["done"], status["total"]) == (done, questions * len(TOP_K))


def test_retrieved_after_a_stop_starts_no_finish(client: TestClient, services: Services) -> None:
    run_id = _start(client).json()["run_id"]
    services.store.update(run_id, run_token_hash=token_hash("job-token"))
    seen = services.store.get(run_id)  # what running_job read just before the owner's Stop
    assert client.post(f"/runs/{run_id}/stop", headers=session()).status_code == 200
    services.store.get = lambda _: seen  # type: ignore[method-assign]
    late = client.post(
        f"/internal/runs/{run_id}/retrieved", json={}, headers={"Authorization": "Bearer job-token"}
    )
    assert late.status_code == 409
    assert services.store.runs[run_id].state == "stopped"  # type: ignore[attr-defined]
    assert _finishes(services) == 0


def test_stop_and_resume(client: TestClient, services: Services) -> None:
    store, jobs = services.store, services.jobs
    run_id = _start(client).json()["run_id"]
    execution = store.get(run_id).execution  # type: ignore[union-attr]
    old_token_hash = store.get(run_id).run_token_hash  # type: ignore[union-attr]

    stopped = client.post(f"/runs/{run_id}/stop", headers=session()).json()
    assert stopped["state"] == "stopped" and stopped["error"] == "Stopped by you."
    assert execution in jobs.ended  # type: ignore[attr-defined]  # the job was cancelled
    assert client.post(f"/runs/{run_id}/stop", headers=session()).status_code == 409

    assert _resume(client, run_id, user="2").status_code == 404
    assert _resume(client, run_id, judge_key="sk-ant-wrong").status_code == 422
    assert _resume(client, run_id, connector_keys={"EXTRA": "x"}).status_code == 422
    assert _resume(client, run_id).status_code == 200
    run = store.get(run_id)
    assert run is not None and run.state == "retrieving" and run.resumes == 1
    assert run.run_token_hash != old_token_hash  # each attempt has its own token
    job, args = jobs.started[-1]  # type: ignore[attr-defined]
    assert job == job_name("baselines/bm25", run.version) and args[:2] == ["retrieve", run_id]
    assert run.executions == [execution, run.execution]  # both attempts, for the log
    assert [(e.event, e.cause) for e in run.history] == [
        ("stopped", "submitter"),
        ("resumed", None),
    ]
    assert _resume(client, run_id).status_code == 409  # not stopped any more


def test_resume_limits(client: TestClient, services: Services) -> None:
    store = services.store
    run_id = _start(client).json()["run_id"]
    store.stop(run_id, "vendor down", counts=True, cause="retrieval")

    other = _start(client).json()["run_id"]  # a stopped run holds no slot
    busy = _resume(client, run_id)
    assert busy.status_code == 429 and "already have a run" in busy.json()["detail"]
    store.fail(other, "", counts=True)

    store.update(run_id, stopped_at=now() - timedelta(hours=RESUME_HOURS, minutes=1))
    late = _resume(client, run_id)
    assert late.status_code == 409 and f"{RESUME_HOURS} hours" in late.json()["detail"]
    assert client.get(f"/runs/{run_id}", headers=session()).json()["resume_until"] is None

    store.update(run_id, stopped_at=now(), resumes=MAX_RESUMES)
    capped = _resume(client, run_id)
    assert capped.status_code == 409 and f"{MAX_RESUMES} times" in capped.json()["detail"]


def test_resume_uses_the_runs_own_version(client: TestClient, services: Services) -> None:
    run_id = _start(client).json()["run_id"]
    started_on = services.store.get(run_id).version  # type: ignore[union-attr]
    services.store.update(run_id, state="judging")  # retrieval done
    services.store.stop(run_id, "judge out of credit", counts=True, cause="judge")

    services.settings = replace(services.settings, version="9.9.9")  # a release in between
    app = create_app(services)
    resumed = TestClient(app, base_url="https://testserver")
    assert _resume(resumed, run_id).status_code == 200
    job, args = services.jobs.started[-1]  # type: ignore[attr-defined]
    assert job == finish_job_name(started_on) and args == ["finish", run_id]
    _start(resumed, user="2")
    assert services.jobs.started[-1][0] == job_name("baselines/bm25", "9.9.9")  # type: ignore[attr-defined]


def test_resume_launch_failure_gives_the_resume_back(
    client: TestClient, services: Services
) -> None:
    run_id = _start(client).json()["run_id"]
    services.store.stop(run_id, "vendor down", counts=True, cause="retrieval")

    def broken(job: str, args: list[str]) -> str:
        raise RuntimeError("quota exceeded")

    services.jobs.start = broken  # type: ignore[method-assign]
    assert _resume(client, run_id).status_code == 503
    run = services.store.get(run_id)
    assert run is not None and run.state == "stopped" and run.resumes == 0
    assert [(e.event, e.cause) for e in run.history] == [("stopped", "retrieval")]


def test_run_log_is_for_its_owner_and_covers_every_attempt(
    client: TestClient, services: Services
) -> None:
    store, logs = services.store, services.logs
    run_id = _start(client).json()["run_id"]
    first = store.get(run_id).execution  # type: ignore[union-attr]
    store.stop(run_id, "vendor down", counts=True, cause="retrieval")
    assert _resume(client, run_id).status_code == 200
    second = store.get(run_id).execution  # type: ignore[union-attr]
    at = now()
    logs.lines = {  # type: ignore[attr-defined]
        first: [LogLine(at=at, job="retrieve", text="ERROR retrieve stopped: vendor down")],
        second: [LogLine(at=at + timedelta(seconds=1), job="retrieve", text="INFO resuming")],
        "executions/someone-elses": [LogLine(at=at, job="retrieve", text="not this run")],
    }

    assert client.get(f"/runs/{run_id}/log").status_code == 401
    assert client.get(f"/runs/{run_id}/log", headers=session("2")).status_code == 404
    body = client.get(f"/runs/{run_id}/log", headers=session()).json()
    assert [line["text"] for line in body["lines"]] == [
        "ERROR retrieve stopped: vendor down",
        "INFO resuming",
    ]
    assert body["error"] is None


def test_run_log_of_a_run_started_before_executions_were_kept(
    client: TestClient, services: Services
) -> None:
    store = services.store
    run_id = _start(client).json()["run_id"]
    execution = store.get(run_id).execution  # type: ignore[union-attr]
    store.update(run_id, executions=[])  # as such records were
    services.logs.lines = {execution: [LogLine(at=now(), job="retrieve", text="INFO x")]}  # type: ignore[attr-defined, dict-item]
    body = client.get(f"/runs/{run_id}/log", headers=session()).json()
    assert [line["text"] for line in body["lines"]] == ["INFO x"]
    store.update(run_id, run_token_hash=token_hash("job-token"))
    client.post(
        f"/internal/runs/{run_id}/retrieved", json={}, headers={"Authorization": "Bearer job-token"}
    )
    run = store.get(run_id)
    assert run is not None and run.executions == [execution, run.execution]  # kept from now on


def test_run_log_read_failure_keeps_the_page_working(
    client: TestClient, services: Services
) -> None:
    run_id = _start(client).json()["run_id"]
    services.logs.fail = True  # type: ignore[attr-defined]
    body = client.get(f"/runs/{run_id}/log", headers=session()).json()
    assert body == {"lines": [], "error": LOG_UNREADABLE}


def test_launch_failure_is_refunded(client: TestClient, services: Services) -> None:
    def broken(job: str, args: list[str]) -> str:
        raise RuntimeError("quota exceeded")

    services.jobs.start = broken  # type: ignore[method-assign]
    resp = _start(client)
    assert resp.status_code == 503
    run = next(iter(services.store.runs.values()))  # type: ignore[attr-defined]
    assert run.state == "failed" and not run.counts
    assert services.store.quota[f"{run.user}_{run.day}"] == 0  # type: ignore[attr-defined]


LOGIN_ID = "e6a1c1d4-3b0f-4c7e-9d2a-5f8b7c6d1e2f"  # the site's id for this login


def test_github_login(client: TestClient, services: Services) -> None:
    login = client.get("/auth/login", params={"login_id": LOGIN_ID}, follow_redirects=False)
    assert login.status_code == 307
    target = urlsplit(login.headers["location"])
    assert target.netloc == "github.com"
    assert parse_qs(target.query)["prompt"] == ["select_account"]  # lets a visitor switch
    state = parse_qs(target.query)["state"][0]

    def github(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/oauth/access_token":
            return httpx.Response(200, json={"access_token": "gho_x"})
        return httpx.Response(200, json={"id": 4242, "login": "someone"})

    services.http = httpx.Client(transport=httpx.MockTransport(github))
    back = client.get(f"/auth/callback?code=c&state={state}", follow_redirects=False)
    assert back.headers["location"].startswith("https://memjudge.test/run#token=")
    fragment = parse_qs(urlsplit(back.headers["location"]).fragment)
    assert auth.session_user(SESSION_SECRET, fragment["token"][0]) == "4242"
    assert fragment["login_id"] == [LOGIN_ID]  # the site takes the token only with its own id


def test_login_needs_the_sites_login_id(client: TestClient) -> None:
    assert client.get("/auth/login", follow_redirects=False).status_code == 422
    bad = client.get("/auth/login", params={"login_id": "x#y"}, follow_redirects=False)
    assert bad.status_code == 422


def test_login_state_must_match_the_browser(client: TestClient) -> None:
    state = auth.issue_state(SESSION_SECRET, "someone-elses-nonce", LOGIN_ID)
    back = client.get(f"/auth/callback?code=c&state={state}", follow_redirects=False)
    assert back.headers["location"].endswith("#error=login-expired")
