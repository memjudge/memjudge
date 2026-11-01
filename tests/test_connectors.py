"""Connector contract tests. Vendor SDKs are replaced with in-memory fakes, so these run
offline; live smoke tests run only when the vendor key is set (see test_live_*)."""

from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from memjudge.connectors.base import ConnectorSpec
from memjudge.connectors.loader import list_connectors, load_connector, load_spec
from memjudge.datasets.loader import load_dataset
from memjudge.schema import Session, Turn


def _session(sid: str = "s1") -> Session:
    t = datetime(2024, 3, 1, 10, 0)
    return Session(
        id=sid,
        started_at=t,
        turns=(
            Turn(id="1", speaker="Ana", text="My dog is called Biscuit.", timestamp=t),
            Turn(id="2", speaker="assistant", text="Lovely name!", timestamp=t),
        ),
    )


def test_all_connectors_have_valid_specs() -> None:
    ids = list_connectors()
    assert {"baselines/bm25", "mem0/platform", "zep/cloud"} <= set(ids)
    for cid in ids:
        spec = load_spec(cid)  # no SDK import needed to validate a spec
        assert spec.id == cid and spec.author
        assert all(k.isupper() for k in spec.required_keys)
        if spec.required_keys:
            assert spec.allowed_domains, f"{cid} calls an API but declares no allowed domains"
            assert spec.models, f"{cid} runs a hosted system but names no models (ADR 0014)"


def test_presets_override_known_settings_only() -> None:
    spec = load_spec("recalld/cloud")
    assert spec.preset("search") == {"retrieval": "search"}
    assert spec.preset(None) == {}
    with pytest.raises(ValueError, match="no preset"):
        spec.preset("nope")
    data = spec.model_dump()
    with pytest.raises(ValueError, match="unknown settings"):
        ConnectorSpec.model_validate({**data, "presets": {"x": {"typo": 1}}})
    with pytest.raises(ValueError, match="base_url"):
        ConnectorSpec.model_validate({**data, "presets": {"x": {"base_url": "http://x"}}})


def test_max_concurrency_caps_the_requested_concurrency() -> None:
    assert load_spec("letta/cloud").concurrency(4) == 3
    assert load_spec("letta/cloud").concurrency(2) == 2
    assert load_spec("baselines/bm25").concurrency(4) == 4  # no cap
    data = load_spec("letta/cloud").model_dump()
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        ConnectorSpec.model_validate({**data, "max_concurrency": 0})


def test_models_name_known_settings() -> None:
    spec = load_spec("recalld/cloud")
    assert spec.model_values({**spec.settings, "select_model": "llm-b"})["select_model"] == "llm-b"
    assert load_spec("zep/cloud").model_values({}) == "vendor"
    assert load_spec("baselines/bm25").model_values({}) == {}
    with pytest.raises(ValueError, match="unknown settings"):
        ConnectorSpec.model_validate({**spec.model_dump(), "models": ["typo_model"]})


def test_missing_keys_are_reported() -> None:
    with pytest.raises(KeyError, match="RECALLD_API_KEY"):
        load_connector("recalld/cloud").instantiate({})


def test_bm25_contract(tiny_path: Path) -> None:
    bm25 = load_connector("baselines/bm25").instantiate({})
    ds = load_dataset(str(tiny_path)).dataset
    bm25.reset("ns")
    for s in ds.conversation("c1").sessions:
        bm25.ingest("ns", s)
    r = bm25.retrieve("ns", "What is the name of Ana's dog?", None, 1000, 30)
    assert "Biscuit" in r.items[0].text and r.items[0].source_turn_ids == ("c1:1",)
    assert bm25.retrieve("ns", "dog", None, 5, 30).context == ""  # budget too small for any turn
    bm25.reset("ns")
    assert bm25.retrieve("ns", "dog", None, 1000, 30).context == ""
    assert bm25.retrieve("never-ingested", "dog", None, 1000, 30).context == ""


class _Resp:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]:
        return self._data


class FakeMem0:
    instances: ClassVar[list[FakeMem0]] = []

    def __init__(self, api_key: str, host: str | None = None) -> None:
        self.api_key = api_key
        self.host = host or "https://api.mem0.ai"
        self.adds: list[dict[str, Any]] = []
        self.polls = 0
        self.client = SimpleNamespace(get=self._get, close=lambda: None)
        FakeMem0.instances.append(self)

    def _get(self, url: str) -> _Resp:
        self.polls += 1
        return _Resp({"status": "SUCCEEDED" if self.polls > 1 else "PENDING"})

    def add(self, messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        self.adds.append({"messages": messages, **kw})
        return {"status": "PENDING", "event_id": f"evt-{len(self.adds)}"}

    def search(self, query: str, **kw: Any) -> dict[str, Any]:
        assert kw["filters"] == {"user_id": "ns"}
        return {
            "results": [
                {
                    "memory": "Ana's dog is Biscuit",
                    "score": 0.9,
                    "created_at": "2024-03-01T10:00:00Z",
                }
            ]
        }

    def delete_users(self, user_id: str) -> None:
        raise ValueError("Entity not found")


def test_mem0_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    mem0 = pytest.importorskip("mem0")

    monkeypatch.setattr(mem0, "MemoryClient", FakeMem0)
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = load_connector("mem0/platform").instantiate({"MEM0_API_KEY": "k"}, {"poll_interval_s": 0})
    fake = FakeMem0.instances[-1]
    c.reset("ns")  # not-found is fine
    usage = c.ingest("ns", _session())
    add = fake.adds[0]
    assert add["user_id"] == "ns" and add["timestamp"] == int(datetime(2024, 3, 1, 10).timestamp())
    # one message per call, in order
    assert [a["messages"] for a in fake.adds] == [
        [{"role": "user", "content": "Ana: My dog is called Biscuit."}],
        [{"role": "assistant", "content": "Lovely name!"}],
    ]
    assert usage.api_calls == 2 + fake.polls  # two adds + polling until each SUCCEEDED
    r = c.retrieve("ns", "dog?", None, 1000, 30)
    assert r.context == "- Ana's dog is Biscuit (recorded 2024-03-01)"
    assert os.environ.get("MEM0_TELEMETRY") == "False"
    assert c.endpoint == "https://api.mem0.ai"
    local = load_connector("mem0/platform").instantiate(
        {"MEM0_API_KEY": "k"}, {"base_url": "http://localhost:8888"}
    )
    assert FakeMem0.instances[-1].host == "http://localhost:8888"
    assert local.endpoint == "http://localhost:8888"


class FakeMem0Oss:
    instances: ClassVar[list[FakeMem0Oss]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.adds: list[dict[str, Any]] = []
        self.closed = False
        self.vector_store = SimpleNamespace(client=SimpleNamespace(close=lambda: None))
        FakeMem0Oss.instances.append(self)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FakeMem0Oss:
        return cls(config)

    def add(self, messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        assert "timestamp" not in kw  # platform-only in Mem0 OSS
        self.adds.append({"messages": messages, **kw})
        return {"results": []}

    def search(self, query: str, **kw: Any) -> dict[str, Any]:
        assert kw["filters"] == {"user_id": "ns"} and kw["top_k"] == 30
        return {
            "results": [
                {
                    "memory": "Ana's dog is Biscuit",
                    "score": 0.9,
                    "created_at": "2026-10-19T09:00:00Z",  # the run date, not shown
                    "metadata": {"session_id": "s1", "session_date": "2024-03-01"},
                }
            ]
        }

    def close(self) -> None:
        self.closed = True


def _fake_mem0_oss_setup(monkeypatch: pytest.MonkeyPatch, *, spacy_model: bool = True) -> None:
    fastembed = pytest.importorskip("fastembed")
    mem0 = pytest.importorskip("mem0")
    spacy = pytest.importorskip("spacy")

    monkeypatch.setattr(mem0, "Memory", FakeMem0Oss)
    monkeypatch.setattr(spacy.util, "is_package", lambda name: spacy_model)
    monkeypatch.setattr(fastembed, "SparseTextEmbedding", lambda model_name: None)


def test_mem0_oss_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_mem0_oss_setup(monkeypatch)
    c = load_connector("mem0/oss").instantiate({"MEM0_OSS_OPENAI_API_KEY": "k"})
    assert c.endpoint is None  # in-process
    c.reset("ns")  # nothing stored yet is fine
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""
    usage = c.ingest("ns", _session())
    fake = FakeMem0Oss.instances[-1]
    assert fake.config["llm"]["config"] == {
        "api_key": "k",
        "model": "gpt-6-luna",
        "is_reasoning_model": True,
    }
    assert fake.config["vector_store"]["provider"] == "qdrant"
    assert [a["messages"] for a in fake.adds] == [
        [{"role": "user", "content": "Ana: My dog is called Biscuit."}],
        [{"role": "assistant", "content": "Lovely name!"}],
    ]
    assert fake.adds[0]["user_id"] == "ns" and usage.api_calls == 2
    # the conversation date reaches Mem0's extraction prompt and is stored with the memory
    assert all("took place on 2024-03-01" in a["prompt"] for a in fake.adds)
    assert fake.adds[0]["metadata"] == {"session_id": "s1", "session_date": "2024-03-01"}
    assert (
        c.retrieve("ns", "dog?", None, 1000, 30).context
        == "- Ana's dog is Biscuit (said 2024-03-01)"
    )
    c.reset("ns")
    assert fake.closed
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""
    c.close()


def test_mem0_oss_refuses_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_mem0_oss_setup(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "judge-key")
    with pytest.raises(RuntimeError, match="OpenRouter"):
        load_connector("mem0/oss").instantiate({"MEM0_OSS_OPENAI_API_KEY": "k"})


def test_mem0_oss_refuses_reduced_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_mem0_oss_setup(monkeypatch, spacy_model=False)
    with pytest.raises(ImportError, match="en_core_web_sm"):
        load_connector("mem0/oss").instantiate({"MEM0_OSS_OPENAI_API_KEY": "k"})


class FakeSupermemory:
    instances: ClassVar[list[FakeSupermemory]] = []

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.base_url = base_url
        self.adds: list[dict[str, Any]] = []
        self.polls = 0
        sm = self

        class Namespaces:
            def delete(self, namespace: str) -> None:
                from supermemory.errors import NotFoundError

                raise NotFoundError(body=None)  # type: ignore[arg-type]

        class Documents:
            def get(self, namespace: str, doc_id: str) -> Any:
                # Indexed on the first poll; memories built on the second.
                sm.polls += 1
                dreaming = "done" if sm.polls % 2 == 0 else "dreaming"
                return SimpleNamespace(
                    system=SimpleNamespace(status="done", dreamingStatus=dreaming)
                )

        self.namespaces, self.documents = Namespaces(), Documents()
        FakeSupermemory.instances.append(self)

    def add(self, namespace: str, **kw: Any) -> Any:
        self.adds.append({"namespace": namespace, **kw})
        return SimpleNamespace(id=f"doc-{len(self.adds)}", status="queued")

    def search(self, namespace: str, **kw: Any) -> Any:
        assert namespace == "ns" and kw["limit"] == 30
        return SimpleNamespace(
            results=[
                SimpleNamespace(memory="Ana's dog is Biscuit", chunk=None, similarity=0.9),
                SimpleNamespace(
                    memory=None, chunk="Ana: My dog is called Biscuit.", similarity=0.7
                ),
            ]
        )


def test_supermemory_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    supermemory = pytest.importorskip("supermemory")

    monkeypatch.setattr(supermemory, "Supermemory", FakeSupermemory)
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = load_connector("supermemory/cloud").instantiate(
        {"SUPERMEMORY_API_KEY": "k"}, {"poll_interval_s": 0}
    )
    fake = FakeSupermemory.instances[-1]
    c.reset("ns")  # not-found is fine
    usage = c.ingest("ns", _session())
    assert [a["content"] for a in fake.adds] == [
        "Ana: My dog is called Biscuit.\nassistant: Lovely name!"
    ]
    add = fake.adds[0]
    assert add["namespace"] == "ns" and add["date"] == "2024-03-01T10:00:00Z"
    assert add["dreaming"] == "instant" and add["metadata"] == {"session_id": "s1"}
    # One add, then polls until indexed and memories built, not just indexed.
    assert usage.api_calls == 1 + fake.polls == 3
    r = c.retrieve("ns", "dog?", None, 1000, 30)
    assert r.context == "- Ana's dog is Biscuit\n- Ana: My dog is called Biscuit."
    assert c.endpoint == "https://api.supermemory.ai"
    local = load_connector("supermemory/cloud").instantiate(
        {"SUPERMEMORY_API_KEY": "k"}, {"base_url": "http://localhost:8787"}
    )
    assert FakeSupermemory.instances[-1].base_url == local.endpoint == "http://localhost:8787"


def test_supermemory_ingest_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    supermemory = pytest.importorskip("supermemory")

    monkeypatch.setattr(supermemory, "Supermemory", FakeSupermemory)
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = load_connector("supermemory/cloud").instantiate(
        {"SUPERMEMORY_API_KEY": "k"}, {"poll_interval_s": 0, "ingest_timeout_s": -1}
    )
    with pytest.raises(RuntimeError, match=r"still queued \(memories: None\)"):
        c.ingest("ns", _session())


class FakeLetta:
    instances: ClassVar[list[FakeLetta]] = []
    registry: ClassVar[dict[str, str]] = {}  # server-side agents: id -> name, shared by clients
    counter: ClassVar[int] = 0
    served_model: ClassVar[str | None] = None  # set to make the server ignore the asked model
    own_models: ClassVar[list[Any]] = []  # models of the account's own provider keys (byok)
    agent_limit: ClassVar[int | None] = None  # the plan's agent limit, if any

    def __init__(self, api_key: str, base_url: str) -> None:
        self.base_url = base_url
        self.created: list[dict[str, Any]] = []
        self.sent: list[list[dict[str, str]]] = []
        self.deleted: list[str] = []
        self.options: dict[str, Any] = {}
        # What the next sends do: (error raised or None, whether Letta stored the turn first,
        # the status its step ends with). Unscripted sends succeed.
        self.sends: list[tuple[Exception | None, bool, str]] = []
        self.stored: list[Any] = []  # stored user messages, newest first
        self.runs_status: dict[str, str] = {}
        self.busy_polls = 0  # how many checks still find a step running
        self.polls = 0
        lt = self

        class Messages:
            def create(self, agent_id: str, messages: list[dict[str, str]]) -> Any:
                lt.sent.append(messages)
                error, stored, status = lt.sends.pop(0) if lt.sends else (None, True, "completed")
                run_id = f"run-{len(lt.sent)}"
                if stored:
                    lt.runs_status[run_id] = status
                    lt.stored.insert(0, SimpleNamespace(otid=messages[0]["otid"], run_id=run_id))
                if error is not None:
                    raise error
                return SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5)
                )

            def list(self, agent_id: str, **kw: Any) -> Any:
                assert kw == {
                    "limit": 20,
                    "order": "desc",
                    "include_return_message_types": ["user_message"],
                }
                return SimpleNamespace(items=lt.stored[: kw["limit"]])

        class Blocks:
            def list(self, agent_id: str) -> list[Any]:
                return [
                    SimpleNamespace(label="human", value="Ana has a dog named Biscuit."),
                    SimpleNamespace(label="empty", value="  "),
                ]

        # 101 archival passages, newest first, so retrieval has to fetch two pages.
        archive = [
            SimpleNamespace(id=f"p{i}", text=f"fact {i}", created_at=datetime(2026, 10, 6))
            for i in range(101)
        ]

        class Passages:
            def list(self, agent_id: str, **kw: Any) -> list[Any]:
                from letta_client import Omit

                assert kw["ascending"] is False
                after = kw["after"]
                start = 0 if isinstance(after, Omit) else [p.id for p in archive].index(after) + 1
                return archive[start : start + kw["limit"]]

        class Search:
            def search(self, **kw: Any) -> list[Any]:
                assert kw == {"query": "dog?", "agent_id": "agent-1", "limit": 30}
                when = datetime(2026, 10, 6)
                return [
                    SimpleNamespace(
                        message_type="user_message", content="Ana: My dog...", created_at=when
                    ),
                    SimpleNamespace(
                        message_type="reasoning_message", content="hmm", created_at=when
                    ),
                ]

        class Agents:
            messages, blocks, passages = Messages(), Blocks(), Passages()

            def create(self, **kw: Any) -> Any:
                limit = FakeLetta.agent_limit
                if limit is not None and len(FakeLetta.registry) >= limit:
                    import httpx
                    from letta_client import APIStatusError

                    request = httpx.Request("POST", "https://api.letta.com/v1/agents/")
                    raise APIStatusError(
                        "limit",
                        response=httpx.Response(402, request=request),
                        body={"error": "You have reached your limit for agents", "limit": limit},
                    )
                lt.created.append(kw)
                FakeLetta.counter += 1
                agent_id = f"agent-{FakeLetta.counter}"
                FakeLetta.registry[agent_id] = kw["name"]
                model = FakeLetta.served_model or kw["model"]
                return SimpleNamespace(id=agent_id, model=model, embedding=kw["embedding"])

            def list(self, **kw: Any) -> list[Any]:
                return [
                    SimpleNamespace(id=i, name=n)
                    for i, n in FakeLetta.registry.items()
                    if "name" not in kw or n == kw["name"]
                ]

            def delete(self, agent_id: str) -> None:
                lt.deleted.append(agent_id)
                FakeLetta.registry.pop(agent_id, None)

        class Models:
            def list(self, **kw: Any) -> list[Any]:
                assert kw == {"provider_category": ["byok"]}
                return FakeLetta.own_models

        class RunUsage:
            def retrieve(self, run_id: str) -> Any:
                return SimpleNamespace(prompt_tokens=70, completion_tokens=3)

        class Runs:
            usage = RunUsage()

            def list(self, **kw: Any) -> Any:
                assert kw["active"] is True and kw["limit"] == 1
                lt.polls += 1
                busy, lt.busy_polls = lt.busy_polls > 0, max(lt.busy_polls - 1, 0)
                return SimpleNamespace(items=[SimpleNamespace(id="run-x")] if busy else [])

            def retrieve(self, run_id: str) -> Any:
                status = lt.runs_status[run_id]
                stop = "end_turn" if status == "completed" else "error"
                return SimpleNamespace(status=status, stop_reason=stop)

        self.agents, self.messages, self.models = Agents(), Search(), Models()
        self.runs = Runs()
        self.close = lambda: None
        FakeLetta.instances.append(self)

    def with_options(self, **kw: Any) -> FakeLetta:
        self.options = kw
        return self


def test_letta_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    FakeLetta.registry.clear()
    FakeLetta.counter = 0
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    fake = FakeLetta.instances[-1]
    c.reset("ns")  # nothing created yet is fine
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""
    usage = c.ingest("ns", _session())
    assert fake.created[0]["name"] == "ns" and fake.created[0]["model"] == "openai/gpt-6-luna"
    assert fake.created[0]["embedding"] == "openai/text-embedding-3-small"
    otids = [m[0].pop("otid") for m in fake.sent]
    assert fake.sent == [
        [{"role": "user", "content": "Ana: My dog is called Biscuit."}],
        [{"role": "assistant", "content": "Lovely name!"}],
    ]
    assert len(set(otids)) == 2 and all(o.startswith("mj-") for o in otids)
    assert fake.options == {"max_retries": 0, "timeout": 600.0}  # turns are never re-sent blindly
    # model list + agent + two messages
    assert usage.api_calls == 4 and usage.input_tokens == 200 and usage.output_tokens == 10
    c.ingest("ns", _session("s2"))
    assert len(fake.created) == 1  # one agent per namespace
    r = c.retrieve("ns", "dog?", None, 1000, 30)
    assert r.context.startswith(
        "MEMORY BLOCKS:\n[human]\nAna has a dog named Biscuit.\n\n"
        "CONVERSATION SEARCH:\n- [2026-10-06] Ana: My dog...\n\n"  # reasoning skipped
        "ARCHIVAL MEMORY (newest first):\n- fact 0 (saved 2026-10-06)\n"
    )
    assert r.context.endswith("- fact 100 (saved 2026-10-06)") and len(r.items) == 103
    assert r.usage.api_calls == 4  # blocks + search + two archival pages
    state = c.save_state("ns")
    assert state == {"agent_id": "agent-1"}
    other = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    other.restore_state("ns", state)
    other.reset("ns")
    assert FakeLetta.instances[-1].deleted == ["agent-1"]  # restored id, deleted once
    # An agent left by an interrupted run (another process, no state saved) is found by name.
    load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"}).ingest("ns", _session())
    assert FakeLetta.registry == {"agent-2": "ns"}
    fresh = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    fresh.reset("ns")
    assert FakeLetta.instances[-1].deleted == ["agent-2"] and FakeLetta.registry == {}
    assert c.endpoint == fake.base_url == "https://api.letta.com"
    local = load_connector("letta/cloud").instantiate(
        {"LETTA_API_KEY": "k"}, {"base_url": "http://localhost:8283"}
    )
    assert FakeLetta.instances[-1].base_url == local.endpoint == "http://localhost:8283"


def test_letta_uses_the_accounts_own_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    own = [
        SimpleNamespace(handle="open-ai/gpt-5", provider_type="openai"),
        SimpleNamespace(handle="open-ai/gpt-6-luna", provider_type="openai"),
        SimpleNamespace(handle="my-router/gpt-6-luna", provider_type="openrouter"),
    ]
    monkeypatch.setattr(FakeLetta, "own_models", own)
    FakeLetta.registry.clear()
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    assert c.ingest("a", _session()).api_calls == 4  # model list + agent + two messages
    assert c.ingest("b", _session()).api_calls == 3  # the handle is looked up once
    fake = FakeLetta.instances[-1]
    # The same model through the account's OpenAI key, whatever the key is named.
    assert [a["model"] for a in fake.created] == ["open-ai/gpt-6-luna"] * 2
    assert fake.created[0]["embedding"] == "openai/text-embedding-3-small"


def test_letta_explains_the_agent_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    monkeypatch.setattr(FakeLetta, "agent_limit", 2)
    FakeLetta.registry.clear()
    left = {"agent-old": "mj-f99850-conv-26", "agent-own": "support-bot"}
    FakeLetta.registry.update(left)
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    with pytest.raises(RuntimeError) as err:
        c.ingest("mj-abc123-conv-30", _session())
    assert "Letta's plan allows 2 agents" in str(err.value)
    assert "MemJudge agents in the account: mj-f99850-conv-26." in str(err.value)
    assert "support-bot" not in str(err.value)
    assert FakeLetta.registry == left  # nothing deleted
    FakeLetta.registry.clear()


def test_letta_stops_when_the_agent_runs_another_model(monkeypatch: pytest.MonkeyPatch) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    monkeypatch.setattr(FakeLetta, "served_model", "openai/gpt-5")
    FakeLetta.registry.clear()
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="other models than requested"):
        c.ingest("ns", _session())
    assert FakeLetta.registry == {}  # the wrong agent is deleted
    assert FakeLetta.instances[-1].sent == []  # nothing ingested into it


def _letta_errors() -> tuple[Any, Any]:
    """Letta's errors: an HTTP status error, and a call with no (complete) response."""
    import httpx
    import letta_client

    request = httpx.Request("POST", "https://api.letta.com/v1/agents/agent-1/messages")

    def status(code: int) -> Exception:
        error = letta_client.APIStatusError
        if code == 409:
            error = letta_client.ConflictError
        elif code >= 500:
            error = letta_client.InternalServerError
        return error(str(code), response=httpx.Response(code, request=request), body=None)

    return status, letta_client.APIConnectionError(request=request)


def test_letta_stores_each_turn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    monkeypatch.setattr("time.sleep", lambda s: None)
    status, broken = _letta_errors()
    FakeLetta.registry.clear()
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    c.ingest("ns", _session())  # creates the agent
    fake = FakeLetta.instances[-1]
    fake.sent.clear()
    fake.stored.clear()
    # Turn 1: Letta stored it and finished, but the response broke; the step was still
    # running at the first check. Turn 2: a 500 before Letta stored it.
    fake.sends = [(broken, True, "completed"), (status(500), False, "")]
    fake.busy_polls = 1
    usage = c.ingest("ns", _session())
    assert [m[0]["content"] for m in fake.sent] == [
        "Ana: My dog is called Biscuit.",  # not sent again
        "Lovely name!",
        "Lovely name!",  # sent again, with the same id
    ]
    assert fake.sent[1][0]["otid"] == fake.sent[2][0]["otid"] != fake.sent[0][0]["otid"]
    assert len(fake.stored) == 2  # each turn stored once
    assert fake.polls == 3  # turn 1: busy, then idle; turn 2: idle
    # Turn 1's usage comes from its step; turn 2's from the second send.
    assert usage.input_tokens == 70 + 100 and usage.output_tokens == 3 + 5
    # Turn 1: send, two checks, lookup, step status and usage. Turn 2: send, check, lookup,
    # send again.
    assert usage.api_calls == 6 + 4
    # A 409 means another step is still running and the turn was refused: wait for that
    # step, find the turn missing and send it again.
    fake.sent.clear()
    fake.sends = [(status(409), False, "")]
    fake.busy_polls = 1
    c.ingest("ns", _session())
    assert [m[0]["content"] for m in fake.sent] == [
        "Ana: My dog is called Biscuit.",
        "Ana: My dog is called Biscuit.",
        "Lovely name!",
    ]
    assert len(fake.stored) == 4


def test_letta_stops_when_a_stored_turn_was_not_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    monkeypatch.setattr("time.sleep", lambda s: None)
    status, _ = _letta_errors()
    FakeLetta.registry.clear()
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    fake = FakeLetta.instances[-1]
    fake.sends = [(status(503), True, "failed")]
    with pytest.raises(RuntimeError, match="Letta stored a turn but its step ended failed"):
        c.ingest("ns", _session())
    assert len(fake.sent) == 1  # not sent again: that would store it twice


def test_letta_stops_after_three_failed_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    letta_client = pytest.importorskip("letta_client")

    monkeypatch.setattr(letta_client, "Letta", FakeLetta)
    monkeypatch.setattr("time.sleep", lambda s: None)
    status, _ = _letta_errors()
    FakeLetta.registry.clear()
    c = load_connector("letta/cloud").instantiate({"LETTA_API_KEY": "k"})
    fake = FakeLetta.instances[-1]
    fake.sends = [(status(503), False, "")] * 3
    with pytest.raises(letta_client.InternalServerError):
        c.ingest("ns", _session())
    assert len(fake.sent) == 3 and len({m[0]["otid"] for m in fake.sent}) == 1
    # Errors that a new try cannot fix stop at once, without looking anything up.
    fake.sent.clear()
    fake.polls = 0
    fake.sends = [(status(400), False, "")]
    with pytest.raises(letta_client.APIStatusError):
        c.ingest("ns", _session())
    assert len(fake.sent) == 1 and fake.polls == 0


class FakeZep:
    instances: ClassVar[list[FakeZep]] = []

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.base_url = base_url
        self.messages: list[Any] = []
        self.users: set[str] = set()
        self.adds = 0
        self.polls = 0
        self.limits: list[int] = []
        z = self

        class User:
            def get(self, user_id: str) -> Any:
                from zep_cloud.errors import NotFoundError

                if user_id not in z.users:
                    raise NotFoundError(body="missing")
                return SimpleNamespace(user_id=user_id)

            def delete(self, user_id: str) -> None:
                from zep_cloud.errors import NotFoundError

                if user_id not in z.users:
                    raise NotFoundError(body="missing")
                z.users.discard(user_id)

            def add(self, user_id: str) -> None:
                z.users.add(user_id)
                z.adds += 1

        class Thread:
            def create(self, thread_id: str, user_id: str) -> None: ...

            def add_messages(self, thread_id: str, messages: list[Any]) -> None:
                z.messages.extend(messages)

            def get_episodes(self, thread_id: str) -> Any:
                z.polls += 1
                done = z.polls > 1
                return SimpleNamespace(
                    episodes=[SimpleNamespace(processed=done) for _ in z.messages]
                )

        class Graph:
            def search(self, **kw: Any) -> Any:
                assert len(kw["query"]) <= 400
                z.limits.append(kw["limit"])
                if kw["scope"] == "edges":
                    edge = SimpleNamespace(
                        fact="Ana has a dog named Biscuit",
                        valid_at="2024-03-01T10:00:00Z",
                        invalid_at=None,
                        score=0.8,
                    )
                    return SimpleNamespace(edges=[edge], nodes=None)
                node = SimpleNamespace(name="Biscuit", summary="Ana's beagle", score=0.7)
                return SimpleNamespace(edges=None, nodes=[node])

        self.user, self.thread, self.graph = User(), Thread(), Graph()
        FakeZep.instances.append(self)


def test_zep_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    zep_client = pytest.importorskip("zep_cloud.client")

    monkeypatch.setattr(zep_client, "Zep", FakeZep)
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = load_connector("zep/cloud").instantiate({"ZEP_API_KEY": "k"}, {"max_message_chars": 10})
    fake = FakeZep.instances[-1]
    c.reset("ns")  # nothing to delete yet, and no user is created
    assert "ns" not in fake.users
    c.ingest("ns", _session())
    assert "ns" in fake.users  # created on first ingest
    # "My dog is called Biscuit." (25 chars) splits into 3 messages; assistant turn into 2.
    assert len(fake.messages) == 5
    assert fake.messages[0].name == "Ana" and fake.messages[0].role == "user"
    assert fake.messages[0].created_at == "2024-03-01T10:00:00Z"
    c.ingest("ns", _session("s2"))
    assert fake.adds == 1  # one user per namespace, reused across sessions
    r = c.retrieve("ns", "x" * 1000, None, 1000, 30)
    assert "- Ana has a dog named Biscuit (2024-03-01 – present)" in r.context
    assert "- Biscuit: Ana's beagle" in r.context
    c.retrieve("ns", "dog?", None, 1000, 10)
    assert fake.limits == [20, 10, 7, 3]  # facts + entities: 30 and 10 items (ADR 0018)
    c.reset("ns")
    assert fake.users == set()  # cleanup leaves no empty user behind
    assert c.endpoint == fake.base_url == "https://api.getzep.com/api/v2"
    local = load_connector("zep/cloud").instantiate(
        {"ZEP_API_KEY": "k"}, {"base_url": "http://localhost:8000/api/v2"}
    )
    assert FakeZep.instances[-1].base_url == local.endpoint == "http://localhost:8000/api/v2"


class FakeFalkor:
    instances: ClassVar[list[FakeFalkor]] = []

    def __init__(self, host: str, port: int) -> None:
        self.graphs: set[str] = set()
        self.deleted: list[str] = []
        f = self

        class Graph:
            def __init__(self, name: str) -> None:
                self.name = name

            async def delete(self) -> None:
                f.graphs.discard(self.name)
                f.deleted.append(self.name)

        self._graph = Graph
        FakeFalkor.instances.append(self)

    async def list_graphs(self) -> list[str]:
        return sorted(self.graphs)

    def select_graph(self, name: str) -> Any:
        return self._graph(name)


class FakeGraphiti:
    instances: ClassVar[list[FakeGraphiti]] = []

    def __init__(self, graph_driver: Any, **clients: Any) -> None:
        self.clients = clients
        self.episodes: list[dict[str, Any]] = []
        self.searches: list[Any] = []
        falkor = graph_driver.client

        class Clone:
            def __init__(self, database: str) -> None:
                self.database = database

            async def build_indices_and_constraints(self) -> None:
                falkor.graphs.add(self.database)

        self.driver = SimpleNamespace(clone=lambda database: Clone(database))
        FakeGraphiti.instances.append(self)

    async def add_episode(self, **kw: Any) -> None:
        self.episodes.append(kw)

    async def search_(self, query: str, config: Any, group_ids: list[str]) -> Any:
        self.searches.append(config)
        if config.edge_config is not None:
            edge = SimpleNamespace(
                fact="Ana has a dog named Biscuit", valid_at=datetime(2024, 3, 1), invalid_at=None
            )
            return SimpleNamespace(edges=[edge], edge_reranker_scores=[0.8])
        node = SimpleNamespace(name="Biscuit", summary="Ana's dog")
        return SimpleNamespace(nodes=[node], node_reranker_scores=[0.7])

    async def close(self) -> None: ...


def test_graphiti_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    graphiti_core = pytest.importorskip("graphiti_core")
    falkordb_asyncio = pytest.importorskip("falkordb.asyncio")
    recipes = pytest.importorskip("graphiti_core.search.search_config_recipes")

    monkeypatch.setattr(graphiti_core, "Graphiti", FakeGraphiti)
    monkeypatch.setattr(falkordb_asyncio, "FalkorDB", FakeFalkor)
    c = load_connector("zep/graphiti").instantiate({"GRAPHITI_OPENAI_API_KEY": "k"})
    fake, falkor = FakeGraphiti.instances[-1], FakeFalkor.instances[-1]
    assert c.endpoint is None  # in-process
    assert os.environ.get("GRAPHITI_TELEMETRY_ENABLED") == "false"
    assert fake.clients["llm_client"].model == "gpt-6-luna"
    c.reset("ns")  # no graph yet is fine
    assert falkor.deleted == []
    assert c.retrieve("ns", "dog?", None, 1000, 30).context == ""
    usage = c.ingest("ns", _session())
    assert falkor.graphs == {"ns"} and usage.api_calls == 2  # one episode per turn
    first = fake.episodes[0]
    assert first["episode_body"] == "Ana: My dog is called Biscuit." and first["group_id"] == "ns"
    assert first["reference_time"] == datetime(2024, 3, 1, 10, tzinfo=UTC)
    r = c.retrieve("ns", "dog?", None, 1000, 30)
    assert "- Ana has a dog named Biscuit (2024-03-01 – present)" in r.context
    assert "- Biscuit: Ana's dog" in r.context
    assert [s.limit for s in fake.searches] == [20, 10]  # 30 items in total
    c.retrieve("ns", "dog?", None, 1000, 10)
    assert [s.limit for s in fake.searches[2:]] == [7, 3]  # Top 10 (ADR 0018)
    assert recipes.EDGE_HYBRID_SEARCH_CROSS_ENCODER.limit == 10  # shared recipe untouched
    c.reset("ns")
    assert falkor.deleted == ["ns"] and c.retrieve("ns", "dog?", None, 1000, 30).context == ""
    c.close()


def test_offline_replay_contract(tmp_path: Path) -> None:
    results = [
        {"memory": f"fact {i}", "score": 1 - i / 10, "created_at": "2024-03-01T10:00:00"}
        for i in range(3)
    ]
    search = {
        "request": {"query": "Q?", "top_k": 3},
        "http_status": 200,
        "search_response": {"results": results},
    }
    path = tmp_path / "output.json"
    path.write_text(json.dumps({"conversations": [{"sample_id": "c 1", "searches": [search]}]}))
    loaded = load_connector("offline/replay")
    c = loaded.instantiate({}, {"path": str(path), "tool": "mem0"})
    c.reset("mj-abc123-c-1")
    c.ingest("mj-abc123-c-1", _session())
    r = c.retrieve("mj-abc123-c-1", "Q?", None, 1000, 2)
    assert r.context == "- fact 0 (recorded 2024-03-01)\n- fact 1 (recorded 2024-03-01)"
    assert [i.score for i in r.items] == [1.0, 0.9]
    with pytest.raises(KeyError):
        c.retrieve("mj-abc123-c-1", "other?", None, 1000, 2)
    with pytest.raises(ValueError, match="exceeds"):
        c.retrieve("mj-abc123-c-1", "Q?", None, 1000, 4)


@pytest.mark.skipif(not os.environ.get("MEM0_API_KEY"), reason="MEM0_API_KEY not set")
def test_live_mem0_smoke() -> None:  # pragma: no cover - needs network and a key
    c = load_connector("mem0/platform").instantiate(dict(os.environ))
    c.reset("memjudge-smoke")
    c.ingest("memjudge-smoke", _session())
    assert (
        "Biscuit"
        in c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
    )
    c.reset("memjudge-smoke")


@pytest.mark.skipif(
    not os.environ.get("MEM0_OSS_OPENAI_API_KEY"), reason="MEM0_OSS_OPENAI_API_KEY not set"
)
def test_live_mem0_oss_smoke() -> None:  # pragma: no cover - needs network and a key
    c = load_connector("mem0/oss").instantiate(dict(os.environ))
    try:
        c.ingest("memjudge-smoke", _session())
        context = c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
        assert "Biscuit" in context
    finally:
        c.close()


@pytest.mark.skipif(not os.environ.get("SUPERMEMORY_API_KEY"), reason="SUPERMEMORY_API_KEY not set")
def test_live_supermemory_smoke() -> None:  # pragma: no cover - needs network and a key
    c = load_connector("supermemory/cloud").instantiate(dict(os.environ))
    c.reset("memjudge-smoke")
    c.ingest("memjudge-smoke", _session())
    assert (
        "Biscuit"
        in c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
    )
    c.reset("memjudge-smoke")


@pytest.mark.skipif(not os.environ.get("LETTA_API_KEY"), reason="LETTA_API_KEY not set")
def test_live_letta_smoke() -> None:  # pragma: no cover - needs network and a key
    c = load_connector("letta/cloud").instantiate(dict(os.environ))
    try:
        c.ingest("memjudge-smoke", _session())
        context = c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
        assert "Biscuit" in context
    finally:
        c.reset("memjudge-smoke")


@pytest.mark.skipif(not os.environ.get("ZEP_API_KEY"), reason="ZEP_API_KEY not set")
def test_live_zep_smoke() -> None:  # pragma: no cover - needs network and a key
    c = load_connector("zep/cloud").instantiate(dict(os.environ))
    c.reset("memjudge-smoke")
    c.ingest("memjudge-smoke", _session())
    assert (
        "Biscuit"
        in c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
    )


def _falkordb_running() -> bool:
    try:
        socket.create_connection(("localhost", 6379), timeout=1).close()
    except OSError:
        return False
    return True


@pytest.mark.skipif(
    not os.environ.get("GRAPHITI_OPENAI_API_KEY") or not _falkordb_running(),
    reason="GRAPHITI_OPENAI_API_KEY not set or no FalkorDB on localhost:6379",
)
def test_live_graphiti_smoke() -> None:  # pragma: no cover - needs network, a key and FalkorDB
    c = load_connector("zep/graphiti").instantiate(dict(os.environ))
    try:
        c.reset("memjudge-smoke")
        c.ingest("memjudge-smoke", _session())
        context = c.retrieve("memjudge-smoke", "What is Ana's dog called?", None, 2000, 30).context
        assert "Biscuit" in context
        c.reset("memjudge-smoke")
    finally:
        c.close()
