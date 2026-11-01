"""Zep Cloud (temporal knowledge graph) via the official ``zep-cloud`` SDK, v3 API.

- One Zep user per benchmark namespace, created on the first ingest and deleted on reset
  (so cleanup leaves no empty users behind); one thread per session.
- Turns are added with their timestamps (``created_at``) so Zep can build valid-time
  ranges for facts. Long turns are split to stay under Zep's message size limit.
- Each message is added as its own call (one message per call, as in every connector).
  Graph building is asynchronous; after each add ``ingest`` polls the thread's episodes
  until it is processed before sending the next, so ingestion time includes
  time-to-searchable and messages are stored in order.
- Retrieval searches the user graph for facts (edges) and entities (nodes) and
  formats them the way Zep's own context block does, with fact validity dates.
- ``base_url`` points the client at another host (e.g. a local or self-hosted server).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from zep_cloud.client import Zep
from zep_cloud.environment import ZepEnvironment
from zep_cloud.errors import NotFoundError
from zep_cloud.types import Message

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session

_ROLES = {"user", "assistant", "system"}
_MAX_QUERY_CHARS = 400  # Zep rejects longer search queries


class IngestTimeoutError(RuntimeError):
    pass


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        base_url = self.settings.get("base_url") or None
        self.endpoint = str(base_url or ZepEnvironment.DEFAULT.value).rstrip("/")
        self.client = Zep(api_key=self.keys["ZEP_API_KEY"], base_url=self.endpoint)
        self.reranker = str(self.settings.get("reranker", "rrf"))
        self.max_message_chars = int(self.settings.get("max_message_chars", 4000))
        self.poll_interval_s = float(self.settings.get("poll_interval_s", 2.0))
        self.ingest_timeout_s = float(self.settings.get("ingest_timeout_s", 1800))

    def reset(self, namespace: str) -> None:
        with contextlib.suppress(NotFoundError):
            self.client.user.delete(namespace)

    def ingest(self, namespace: str, session: Session) -> Usage:
        thread_id = f"{namespace}-{session.id}"[:200]
        calls = 1
        try:
            self.client.user.get(namespace)
        except NotFoundError:
            self.client.user.add(user_id=namespace)
            calls += 1
        self.client.thread.create(thread_id=thread_id, user_id=namespace)
        messages = [
            m
            for t in session.turns
            for m in self._messages(t.speaker, t.text, t.timestamp or session.started_at)
        ]
        for i, message in enumerate(messages, start=1):
            self.client.thread.add_messages(thread_id, messages=[message])
            calls += 1 + self._wait(thread_id, expected=i)
        return Usage(api_calls=calls)

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        q = query[:_MAX_QUERY_CHARS]
        facts = round(top_k * 2 / 3)  # 20 facts + 10 entities, or 7 + 3 (ADR 0018)
        edges = (
            self.client.graph.search(
                user_id=namespace,
                query=q,
                scope="edges",
                limit=facts,
                reranker=self.reranker,
            ).edges
            or []
        )
        nodes = (
            self.client.graph.search(
                user_id=namespace,
                query=q,
                scope="nodes",
                limit=top_k - facts,
                reranker=self.reranker,
            ).nodes
            or []
        )
        items: list[Item] = []
        fact_lines = []
        for e in edges:
            since, until = _date(e.valid_at) or "date unknown", _date(e.invalid_at) or "present"
            line = f"- {e.fact} ({since} – {until})"
            fact_lines.append(line)
            items.append(Item(text=line, score=e.score))
        node_lines = []
        for n in nodes:
            line = f"- {n.name}: {n.summary}"
            node_lines.append(line)
            items.append(Item(text=line, score=n.score))
        parts = []
        if fact_lines:
            parts.append("FACTS (valid from – until):\n" + "\n".join(fact_lines))
        if node_lines:
            parts.append("ENTITIES:\n" + "\n".join(node_lines))
        return Retrieval(context="\n\n".join(parts), items=items, usage=Usage(api_calls=2))

    def _messages(self, speaker: str, text: str, when: datetime | None) -> list[Message]:
        role = speaker if speaker in _ROLES else "user"
        name = None if speaker in _ROLES else speaker
        chunks = [
            text[i : i + self.max_message_chars]
            for i in range(0, len(text), self.max_message_chars)
        ] or [""]
        return [
            Message(
                role=role,
                name=name,
                content=chunk,
                created_at=when.isoformat() + "Z" if when else None,
            )
            for chunk in chunks
        ]

    def _wait(self, thread_id: str, expected: int) -> int:
        calls = 0
        deadline = time.monotonic() + self.ingest_timeout_s
        while True:
            episodes = self.client.thread.get_episodes(thread_id).episodes or []
            calls += 1
            if len(episodes) >= expected and all(e.processed for e in episodes):
                return calls
            if time.monotonic() > deadline:
                done = sum(bool(e.processed) for e in episodes)
                raise IngestTimeoutError(f"Zep processed {done}/{expected} episodes before timeout")
            time.sleep(self.poll_interval_s)


def _date(value: str | None) -> str | None:
    return value[:10] if value else None
