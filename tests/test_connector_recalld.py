"""Recalld connector contract test against a mocked REST API (offline)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import respx

from memjudge.connectors.loader import load_connector
from memjudge.schema import Conversation, Session, Turn

BASE = "https://eu.recalld.ai"
KEY = {"RECALLD_API_KEY": "rcd_eu_test"}
USED = {
    "x-recalld-models": "decompose=openai-gpt-6-luna, evaluate=openai-gpt-6-luna, "
    "select_facts=openai-gpt-6-luna, embedding=gemini-embedding-002-vertex"
}


def _session() -> Session:
    t = datetime(2024, 3, 1, 10, 0)
    return Session(
        id="s1",
        started_at=t,
        turns=(
            Turn(id="1", speaker="Ana", text="My dog is called Biscuit.", timestamp=t),
            Turn(id="2", speaker="assistant", text="Lovely name!", timestamp=t),
        ),
    )


@respx.mock
def test_recalld_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    create = respx.post(f"{BASE}/v1/agents").mock(
        return_value=httpx.Response(201, json={"id": "a1"})
    )
    respx.post(f"{BASE}/v1/agents/a1/threads").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    ingest = respx.post(f"{BASE}/v1/agents/a1/memory").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(
                200,
                json={"usage": {"input_tokens": 900, "output_tokens": 50}, "credits_charged": 3},
                headers=USED,
            ),
            httpx.Response(200, json={}, headers=USED),
            httpx.Response(200, json={}, headers=USED),
            httpx.Response(200, json={}, headers=USED),
        ]
    )
    recall = respx.post(f"{BASE}/v1/agents/a1/memory/recall").mock(
        return_value=httpx.Response(
            200,
            json={
                "facts": [
                    {"text": "Ana's dog is called Biscuit", "event_date": "2024-03-01T10:00:00Z"},
                    {"text": "Ana likes tea"},
                ],
                "usage": {"input_tokens": 120, "output_tokens": 8},
            },
            headers=USED,
        )
    )
    delete = respx.delete(f"{BASE}/v1/agents/a1").mock(return_value=httpx.Response(200, json={}))

    c = load_connector("recalld/cloud").instantiate(KEY)
    c.reset("ns")  # nothing exists yet: no API call, no empty agent created
    assert not create.called and not delete.called
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""

    usage = c.ingest("ns", _session())
    assert create.calls[0].request.headers["authorization"] == "Bearer rcd_eu_test"
    assert json.loads(create.calls[0].request.content) == {  # the account default never applies
        "name": "ns",
        "llm_models": {
            "decompose": "openai-gpt-6-luna",
            "evaluate": "openai-gpt-6-luna",
            "select_facts": "openai-gpt-6-luna",
        },
        "embedding_model": "gemini-embedding-002-vertex",
    }
    first, second = (json.loads(call.request.content) for call in ingest.calls[1:3])
    assert first["thread_id"] == "t1"
    assert first["inputs"] == [  # one message per call, in order
        {
            "kind": "USER",
            "content": "My dog is called Biscuit.",
            "content_type": "text/plain",
            "author": "Ana",
            "event_date": "2024-03-01T10:00:00Z",
        }
    ]
    assert [i["kind"] for i in second["inputs"]] == ["AGENT"]
    assert ingest.call_count == 3  # retried the 503, then one call per turn
    assert usage.api_calls == 4 and usage.input_tokens == 900 and usage.credits == 3

    r = c.retrieve("ns", "What is Ana's dog called?", datetime(2024, 6, 1), 1000, 30)
    body = json.loads(recall.calls[0].request.content)
    assert body == {
        "thread_id": "t1",
        "query": "What is Ana's dog called?",
        "limit": 30,
        "event_date": "2024-06-01T00:00:00Z",
    }
    assert r.context == "- Ana's dog is called Biscuit (date: 2024-03-01)\n- Ana likes tea"
    assert r.usage.input_tokens == 120 and r.usage.credits is None

    c.ingest("ns", _session())
    assert create.call_count == 1  # agent reused within a namespace
    c.reset("ns")
    assert delete.called
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""


@respx.mock
def test_recalld_search_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    respx.post("https://us.recalld.ai/v1/agents").mock(
        return_value=httpx.Response(401, text="invalid key")
    )
    c = load_connector("recalld/cloud").instantiate(
        {"RECALLD_API_KEY": "rcd_us_secret"}, {"retrieval": "search"}
    )
    with pytest.raises(RuntimeError) as exc:
        c.ingest("ns", _session())
    assert "401" in str(exc.value) and "rcd_us_secret" not in str(exc.value)

    with pytest.raises(ValueError, match="region"):
        load_connector("recalld/cloud").instantiate({"RECALLD_API_KEY": "unknown"})
    with pytest.raises(ValueError, match="retrieval"):
        load_connector("recalld/cloud").instantiate(KEY, {"retrieval": "answer"})


@respx.mock
def test_recalld_reset_tolerates_missing_agent() -> None:
    delete = respx.delete(f"{BASE}/v1/agents/a9").mock(return_value=httpx.Response(404))
    c = load_connector("recalld/cloud").instantiate(KEY)
    c.restore_state("ns", {"agent_id": "a9", "thread_id": "t9"})
    c.reset("ns")  # already deleted server-side: reset must still succeed
    assert delete.called
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""


@respx.mock
def test_recalld_base_url_override() -> None:
    respx.post("http://localhost:8080/v1/agents").mock(
        return_value=httpx.Response(201, json={"id": "a1"})
    )
    respx.post("http://localhost:8080/v1/agents/a1/threads").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    respx.post("http://localhost:8080/v1/agents/a1/memory").mock(
        return_value=httpx.Response(200, json={}, headers=USED)
    )
    c = load_connector("recalld/cloud").instantiate(KEY, {"base_url": "http://localhost:8080/"})
    assert c.endpoint == "http://localhost:8080"
    c.ingest("ns", _session())  # goes to localhost, not eu.recalld.ai
    assert load_connector("recalld/cloud").instantiate(KEY).endpoint == BASE


@pytest.mark.skipif(not os.environ.get("RECALLD_API_KEY"), reason="RECALLD_API_KEY not set")
def test_live_recalld_smoke() -> None:  # pragma: no cover - needs network and a key
    c = load_connector("recalld/cloud").instantiate(dict(os.environ))
    try:
        c.ingest("memjudge-smoke", _session())
        context = c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
        assert "Biscuit" in context
    finally:
        c.reset("memjudge-smoke")


@respx.mock
@pytest.mark.parametrize(
    "headers",
    [
        {
            "x-recalld-models": USED["x-recalld-models"].replace(
                "select_facts=openai", "select_facts=x"
            )
        },
        {},  # a gateway that does not report the models
    ],
)
def test_recalld_stops_when_other_models_ran(headers: dict[str, str]) -> None:
    respx.post(f"{BASE}/v1/agents").mock(return_value=httpx.Response(201, json={"id": "a1"}))
    respx.post(f"{BASE}/v1/agents/a1/threads").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    ingest = respx.post(f"{BASE}/v1/agents/a1/memory").mock(
        return_value=httpx.Response(200, json={}, headers=headers)
    )
    c = load_connector("recalld/cloud").instantiate(KEY)
    with pytest.raises(RuntimeError, match="other models than requested"):
        c.ingest("ns", _session())
    assert ingest.call_count == 1  # not retried


def _conversation() -> Conversation:
    return Conversation(id="c1", sessions=(_session(),))


@respx.mock
def test_recalld_batch_uploads_once_and_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    create = respx.post(f"{BASE}/v1/agents").mock(
        return_value=httpx.Response(201, json={"id": "a1"})
    )
    respx.post(f"{BASE}/v1/agents/a1/threads").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    upload = respx.post(f"{BASE}/v1/memory/batches").mock(
        return_value=httpx.Response(201, json={"id": "b1", "status": "in_progress"})
    )
    status = respx.get(f"{BASE}/v1/memory/batches/b1").mock(
        side_effect=[
            httpx.Response(200, json={"status": "in_progress", "counts": {"total": 2}}),
            httpx.Response(
                200, json={"status": "in_progress", "counts": {"total": 2, "completed": 1}}
            ),
            httpx.Response(
                200,
                json={
                    "status": "completed",
                    "counts": {"total": 2, "completed": 2, "failed": 0},
                    "usage": {"input_tokens": 5000, "output_tokens": 400},
                },
            ),
        ]
    )
    ingest = respx.post(f"{BASE}/v1/agents/a1/memory").mock(
        return_value=httpx.Response(200, json={})
    )

    c = load_connector("recalld/cloud").instantiate(KEY, {"batch": True, "allow_non_eu": True})
    seen: list[tuple[int, int]] = []
    usage = c.ingest_batch({"ns": _conversation()}, tmp_path, lambda d, t: seen.append((d, t)))
    assert usage is not None
    assert seen == [(0, 2), (1, 2), (2, 2)]  # turns processed, from each status poll
    assert usage.input_tokens == 5000 and usage.output_tokens == 400
    assert usage.api_calls == 2 + 1  # agent + thread, then the one upload
    assert upload.call_count == 1 and status.call_count == 3
    request = upload.calls[0].request
    assert request.url.params["allow_non_eu"] == "true"
    assert request.headers["content-type"] == "application/jsonl"
    lines = [json.loads(line) for line in request.content.decode().splitlines()]
    # Each line is the body ingest would send for that turn, plus its agent, in order.
    assert lines == [
        {"agent_id": "a1", "thread_id": "t1", "inputs": [i]}
        for i in [
            {
                "kind": "USER",
                "content": "My dog is called Biscuit.",
                "content_type": "text/plain",
                "author": "Ana",
                "event_date": "2024-03-01T10:00:00Z",
            },
            {
                "kind": "AGENT",
                "content": "Lovely name!",
                "content_type": "text/plain",
                "author": "assistant",
                "event_date": "2024-03-01T10:00:00Z",
            },
        ]
    ]
    assert not ingest.called and create.call_count == 1
    assert c.save_state("ns") == {"agent_id": "a1", "thread_id": "t1"}
    state = json.loads((tmp_path / "recalld-batch.json").read_text("utf-8"))
    assert state == {"agents": {"ns": ["a1", "t1"]}, "batch_id": "b1"}


@respx.mock
def test_recalld_batch_upload_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    respx.post(f"{BASE}/v1/agents").mock(return_value=httpx.Response(201, json={"id": "a1"}))
    respx.post(f"{BASE}/v1/agents/a1/threads").mock(
        return_value=httpx.Response(201, json={"id": "t1"})
    )
    upload = respx.post(f"{BASE}/v1/memory/batches").mock(return_value=httpx.Response(502))
    delete = respx.delete(f"{BASE}/v1/agents/a1").mock(return_value=httpx.Response(200, json={}))
    c = load_connector("recalld/cloud").instantiate(KEY, {"batch": True})
    with pytest.raises(RuntimeError):
        c.ingest_batch({"ns": _conversation()}, tmp_path)
    assert upload.call_count == 1  # it may have been stored; a retry could store it twice
    assert delete.called  # the agents it made are removed
    assert not (tmp_path / "recalld-batch.json").exists()


@respx.mock
def test_recalld_batch_resume_waits_for_the_same_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    (tmp_path / "recalld-batch.json").write_text(
        json.dumps({"agents": {"ns": ["a1", "t1"]}, "batch_id": "b1"}), "utf-8"
    )
    create = respx.post(f"{BASE}/v1/agents")
    upload = respx.post(f"{BASE}/v1/memory/batches")
    respx.get(f"{BASE}/v1/memory/batches/b1").mock(
        return_value=httpx.Response(
            200, json={"status": "completed", "counts": {"total": 2, "completed": 2}}
        )
    )
    c = load_connector("recalld/cloud").instantiate(KEY, {"batch": True})
    usage = c.ingest_batch({"ns": _conversation()}, tmp_path)
    assert usage is not None and usage.api_calls == 0
    assert not create.called and not upload.called
    assert c.save_state("ns") == {"agent_id": "a1", "thread_id": "t1"}


@respx.mock
def test_recalld_batch_that_fails_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    (tmp_path / "recalld-batch.json").write_text(
        json.dumps({"agents": {"ns": ["a1", "t1"]}, "batch_id": "b1"}), "utf-8"
    )
    respx.get(f"{BASE}/v1/memory/batches/b1").mock(
        return_value=httpx.Response(
            200, json={"status": "failed", "error": "no memory addition could be stored"}
        )
    )
    delete = respx.delete(f"{BASE}/v1/agents/a1").mock(return_value=httpx.Response(200, json={}))
    c = load_connector("recalld/cloud").instantiate(KEY, {"batch": True})
    with pytest.raises(RuntimeError, match="failed"):
        c.ingest_batch({"ns": _conversation()}, tmp_path)
    assert delete.called
    assert not (tmp_path / "recalld-batch.json").exists()


@respx.mock
def test_recalld_batch_cancelled_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Cancelled elsewhere (e.g. in the dashboard): the wait ends and the run stops.
    monkeypatch.setattr("time.sleep", lambda s: None)
    (tmp_path / "recalld-batch.json").write_text(
        json.dumps({"agents": {"ns": ["a1", "t1"]}, "batch_id": "b1"}), "utf-8"
    )
    status = respx.get(f"{BASE}/v1/memory/batches/b1").mock(
        side_effect=[
            httpx.Response(200, json={"status": "in_progress", "counts": {"total": 2}}),
            httpx.Response(200, json={"status": "cancelling", "counts": {"total": 2}}),
            httpx.Response(
                200, json={"status": "cancelled", "counts": {"total": 2, "completed": 1}}
            ),
        ]
    )
    delete = respx.delete(f"{BASE}/v1/agents/a1").mock(return_value=httpx.Response(200, json={}))
    c = load_connector("recalld/cloud").instantiate(KEY, {"batch": True})
    with pytest.raises(RuntimeError, match="b1 was cancelled"):
        c.ingest_batch({"ns": _conversation()}, tmp_path)
    assert status.call_count == 3
    assert delete.called
    assert not (tmp_path / "recalld-batch.json").exists()


def test_recalld_batch_off_by_default(tmp_path: Path) -> None:
    c = load_connector("recalld/cloud").instantiate(KEY)
    assert c.ingest_batch({"ns": _conversation()}, tmp_path) is None
