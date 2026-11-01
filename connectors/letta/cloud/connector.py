"""Letta (hosted API, v1) via the official ``letta-client`` SDK.

Letta's memory is managed by an agent: it decides what to keep in its in-context memory
blocks and what to save to archival memory.

- One Letta agent per benchmark namespace, created with the ``model`` setting (Letta
  requires one), Letta's default tools and two neutral memory blocks (``human``,
  ``persona``).
- Each turn is sent to the agent as its own message. The call returns once the agent has
  finished its step, so whatever it chose to remember is stored when ``ingest`` returns.
  Letta messages carry no source date, so turns are stored as of the run date.
- Each turn is stored once. Letta may store a turn although the call failed (a 5xx, a
  timeout, or a response that broke after Letta finished), and a turn sent again is stored
  again, even with the same id. So a turn is never re-sent blindly: it carries its own id
  (``otid``), and after a failure the connector waits until the agent has no step running,
  looks the id up and sends the turn again only if Letta did not store it. A stored turn
  whose step failed stops the run (resuming redoes the conversation).
- Retrieval returns, in order: the agent's memory blocks; Letta's search over the
  conversation history ("recall memory") for the query; and everything the agent saved to
  archival memory, newest first (the harness's token budget cuts the end). Letta Cloud
  has no working semantic archival search: the SDK's search routes return 404 and the
  list filter only matches exact text. No answer is generated (judge-only scoring).
- Letta runs an LLM step per message and reports its token usage, which is passed through.
- The agent is created with the ``model`` and ``embedding`` settings, and the run stops if
  the created agent reports other models (ADR 0014).
- Own provider keys: when the account has its own key connected for the model's provider
  (e.g. an OpenAI key, listed as ``byok``), the same model is used through that key, under
  the name Letta gives it (``<key name>/gpt-6-luna``), so it is billed by the provider
  instead of from Letta credits. Without one, Letta's hosted model is used.
- When Letta refuses a new agent because the plan's agent limit is reached, the run stops
  with a message naming the MemJudge agents (``mj-...``) in the account. A stopped run
  keeps its agents so it can be resumed; they are never deleted for another run.
- ``base_url`` points the client at a self-hosted Letta server.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from letta_client import APIConnectionError, APIStatusError, Letta, NotFoundError, Omit, omit
from letta_client.types import CreateBlockParam, MessageCreateParam

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session

log = logging.getLogger(__name__)

_CLOUD = "https://api.letta.com"
_PAGE = 100
_SEND_TIMEOUT_S = 600.0  # a Letta step can be slow: waiting for it beats sending the turn again
_SEND_TRIES = 3
_STEP_WAIT_S = 600.0  # after a failed send, how long to wait for the agent's running step
_POLL_S = 5.0
_RECENT_TURNS = 20  # where a turn sent just now must be, among the agent's latest user messages
_RECALL_TYPES = {"user_message", "assistant_message"}
_BLOCKS: list[CreateBlockParam] = [
    {"label": "human", "value": "Nothing is known about the people in this conversation yet."},
    {"label": "persona", "value": "I am an assistant that remembers what people tell me."},
]


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        base_url = self.settings.get("base_url") or None
        self.client = Letta(api_key=self.keys["LETTA_API_KEY"], base_url=base_url or _CLOUD)
        # Sends turns: no automatic re-send, see ``_send``.
        self._sender = self.client.with_options(max_retries=0, timeout=_SEND_TIMEOUT_S)
        self.endpoint = str(base_url or _CLOUD).rstrip("/")
        self._lock = threading.Lock()
        self._agents: dict[str, str] = {}
        self._model: str | None = None  # the model handle, once resolved

    def reset(self, namespace: str) -> None:
        with self._lock:
            agent_id = self._agents.pop(namespace, None)
        ids = {agent_id} if agent_id else set()
        # Agents left by an interrupted run are found by name, not only by this process's map.
        ids.update(a.id for a in self.client.agents.list(name=namespace) if a.name == namespace)
        for agent_id in ids:
            with contextlib.suppress(NotFoundError):
                self.client.agents.delete(agent_id)

    def ingest(self, namespace: str, session: Session) -> Usage:
        usage = Usage()
        agent_id, created = self._agent(namespace)
        usage.api_calls += created
        for turn in session.turns:
            usage = usage + self._send(agent_id, _message(turn.speaker, turn.text))
        return usage

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        with self._lock:
            agent_id = self._agents.get(namespace)
        if agent_id is None:
            return Retrieval(context="", usage=Usage(api_calls=0))
        calls = 2
        blocks = [
            Item(text=f"[{b.label}]\n{b.value.strip()}")
            for b in self.client.agents.blocks.list(agent_id)
            if b.value.strip()
        ]
        recall = [
            Item(text=f"- [{m.created_at:%Y-%m-%d}] {_text(getattr(m, 'content', ''))}")
            for m in self.client.messages.search(query=query, agent_id=agent_id, limit=top_k)
            if getattr(m, "message_type", None) in _RECALL_TYPES
            and _text(getattr(m, "content", ""))
        ]
        archival: list[Item] = []
        after: str | Omit = omit
        while True:
            page = self.client.agents.passages.list(
                agent_id, ascending=False, limit=_PAGE, after=after
            )
            calls += 1
            for p in page:
                when = f" (saved {p.created_at:%Y-%m-%d})" if p.created_at else ""
                archival.append(Item(text=f"- {p.text}{when}"))
            last = page[-1].id if page else None
            if len(page) < _PAGE or last is None:
                break
            after = last
        sections = [
            ("MEMORY BLOCKS", blocks),
            ("CONVERSATION SEARCH", recall),
            ("ARCHIVAL MEMORY (newest first)", archival),
        ]
        context = "\n\n".join(
            f"{title}:\n" + "\n".join(i.text for i in items) for title, items in sections if items
        )
        return Retrieval(
            context=context, items=blocks + recall + archival, usage=Usage(api_calls=calls)
        )

    def save_state(self, namespace: str) -> dict[str, Any] | None:
        with self._lock:
            agent_id = self._agents.get(namespace)
        return {"agent_id": agent_id} if agent_id else None

    def restore_state(self, namespace: str, state: dict[str, Any] | None) -> None:
        if state and state.get("agent_id"):
            with self._lock:
                self._agents[namespace] = str(state["agent_id"])

    def close(self) -> None:
        self.client.close()

    def _send(self, agent_id: str, message: MessageCreateParam) -> Usage:
        """Send one turn so that Letta stores it once, and its usage."""
        otid = message["otid"] = f"mj-{uuid.uuid4()}"
        calls = 0
        for attempt in range(1, _SEND_TRIES + 1):
            calls += 1
            try:
                resp = self._sender.agents.messages.create(agent_id, messages=[message])
            except (APIConnectionError, APIStatusError) as exc:
                if not _may_be_temporary(exc) or attempt == _SEND_TRIES:
                    raise
                log.warning(
                    "sending a turn to Letta failed (%s); checking whether Letta stored it",
                    _describe(exc),
                )
                calls += self._wait_for_steps(agent_id)
                run_id = self._stored(agent_id, otid)
                calls += 1
                if run_id is None:
                    log.warning("Letta did not store the turn: sending it again")
                    continue
                run = self.client.runs.retrieve(run_id)
                if run.status != "completed":
                    raise RuntimeError(
                        f"Letta stored a turn but its step ended {run.status} "
                        f"({run.stop_reason}), so the agent may not have processed it "
                        f"(agent {agent_id}, run {run_id})"
                    ) from exc
                u = self.client.runs.usage.retrieve(run_id)
                log.warning("Letta had stored the turn: not sending it again")
                return Usage(
                    api_calls=calls + 2,
                    input_tokens=u.prompt_tokens or 0,
                    output_tokens=u.completion_tokens or 0,
                )
            return Usage(
                api_calls=calls,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
        raise AssertionError("unreachable")

    def _wait_for_steps(self, agent_id: str) -> int:
        """Wait until the agent has no step running; calls made."""
        deadline = time.monotonic() + _STEP_WAIT_S
        calls = 0
        while True:
            calls += 1
            if not self.client.runs.list(agent_id=agent_id, active=True, limit=1).items:
                return calls
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Letta's agent {agent_id} still has a step running after "
                    f"{_STEP_WAIT_S / 60:.0f} minutes"
                )
            time.sleep(_POLL_S)

    def _stored(self, agent_id: str, otid: str) -> str | None:
        """The run id of the stored turn with this id, or None if Letta did not store it."""
        # One page only: following the pages of this list repeats messages.
        page = self.client.agents.messages.list(
            agent_id,
            limit=_RECENT_TURNS,
            order="desc",
            include_return_message_types=["user_message"],
        )
        return next((m.run_id or "" for m in page.items if m.otid == otid), None)

    def _agent(self, namespace: str) -> tuple[str, int]:
        """The namespace's agent id, creating the agent on first use, and calls made."""
        with self._lock:
            agent_id = self._agents.get(namespace)
        if agent_id is not None:
            return agent_id, 0
        model, calls = self._model_handle()
        embedding = str(self.settings["embedding"])
        try:
            agent = self.client.agents.create(
                name=namespace, memory_blocks=_BLOCKS, model=model, embedding=embedding
            )
        except APIStatusError as exc:
            if exc.status_code == 402 and "agent" in str(exc.body).lower():
                raise RuntimeError(self._agent_limit(exc.body)) from exc
            raise
        used = (
            agent.model or getattr(agent.llm_config, "handle", None),
            agent.embedding or getattr(agent.embedding_config, "handle", None),
        )
        if used != (model, embedding):
            self.client.agents.delete(agent.id)
            raise RuntimeError(
                f"Letta created the agent with other models than requested: {used} "
                f"(expected {(model, embedding)})"
            )
        with self._lock:
            self._agents[namespace] = agent.id
        return agent.id, calls + 1

    def _agent_limit(self, body: object) -> str:
        """What to do when Letta's plan has no agent left."""
        limit = body.get("limit") if isinstance(body, dict) else None
        try:
            ours = sorted(
                a.name for a in self.client.agents.list(limit=100) if a.name.startswith("mj-")
            )
        except Exception:
            ours = []
        plan = f"Letta's plan allows {limit} agents" if limit else "Letta's agent limit is reached"
        if not ours:
            return f"{plan} and none is free. Delete agents in Letta or upgrade the plan."
        return (
            f"{plan} and none is free. MemJudge agents in the account: {', '.join(ours)}. "
            "A stopped run keeps its agents so it can be resumed: resume that run (it deletes "
            "them), or delete them in Letta."
        )

    def _model_handle(self) -> tuple[str, int]:
        """The ``model`` setting's handle, through the account's own key for its provider
        when one is connected (``openai/gpt-6-luna`` -> ``<key name>/gpt-6-luna``), and calls
        made."""
        with self._lock:
            if self._model is not None:
                return self._model, 0
        requested = str(self.settings["model"])
        provider, _, name = requested.partition("/")
        own = sorted(
            m.handle
            for m in self.client.models.list(provider_category=["byok"])
            if m.handle and m.provider_type == provider and m.handle.partition("/")[2] == name
        )
        handle = own[0] if own else requested
        with self._lock:
            self._model = handle
        return handle, 1


def _text(content: Any) -> str:
    """Message content is a string or a list of content parts."""
    if isinstance(content, str):
        return content.strip()
    return " ".join(str(getattr(p, "text", "")) for p in content).strip()


def _may_be_temporary(exc: APIConnectionError | APIStatusError) -> bool:
    """The failures the SDK would retry: no response, 408, 409, 429 and 5xx."""
    if isinstance(exc, APIConnectionError):
        return True
    return exc.status_code in (408, 409, 429) or exc.status_code >= 500


def _describe(exc: APIConnectionError | APIStatusError) -> str:
    if isinstance(exc, APIStatusError):
        return f"{type(exc).__name__} {exc.status_code}"
    return type(exc).__name__


def _message(speaker: str, text: str) -> MessageCreateParam:
    if speaker == "assistant":
        return {"role": "assistant", "content": text}
    if speaker == "user":
        return {"role": "user", "content": text}
    # Named speakers (e.g. LoCoMo): keep the name in the content so attribution survives.
    return {"role": "user", "content": f"{speaker}: {text}"}
