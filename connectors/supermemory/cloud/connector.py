"""Supermemory (hosted API, v5) via the official ``supermemory`` SDK.

- One Supermemory namespace per benchmark namespace.
- Each session is added as one document (one ``speaker: text`` line per turn), with the
  session's date as ``date``, as Supermemory's own benchmark harness does.
- Supermemory indexes a document (``status``), then builds its memories in a second step
  (``dreamingStatus``). ``ingest`` polls the document until both are ``done``, so ingestion
  time includes time-to-searchable (ADR 0007).
- ``dreaming: instant`` builds each document's memories right away (one extra billed
  operation per document). The default, ``dynamic``, waits to group documents and can take
  minutes per session, which a per-session wait turns into hours per run.
- Retrieval is ``search`` with Supermemory's default mode (hybrid: learned memories plus
  source chunks).
- ``base_url`` points the client at another host (e.g. a local or self-hosted server).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from supermemory import Supermemory
from supermemory.environment import SupermemoryEnvironment
from supermemory.errors import NotFoundError

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session


class IngestTimeoutError(RuntimeError):
    pass


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        base_url = self.settings.get("base_url") or None
        self.client = Supermemory(api_key=self.keys["SUPERMEMORY_API_KEY"], base_url=base_url)
        self.endpoint = str(base_url or SupermemoryEnvironment.DEFAULT.value).rstrip("/")
        self.dreaming = str(self.settings.get("dreaming", "instant"))
        self.poll_interval_s = float(self.settings.get("poll_interval_s", 1.0))
        self.ingest_timeout_s = float(self.settings.get("ingest_timeout_s", 3600))

    def reset(self, namespace: str) -> None:
        with contextlib.suppress(NotFoundError):  # nothing stored yet is fine
            self.client.namespaces.delete(namespace)

    def ingest(self, namespace: str, session: Session) -> Usage:
        if not session.turns:
            return Usage()
        when = session.started_at or session.turns[0].timestamp
        kwargs: dict[str, Any] = {"metadata": {"session_id": session.id}}
        if when is not None:
            kwargs["date"] = _iso(when)
        content = "\n".join(f"{t.speaker}: {t.text}" for t in session.turns)
        ref = self.client.add(namespace, content=content, dreaming=self.dreaming, **kwargs)
        return Usage(api_calls=1 + self._wait(namespace, ref.id, str(ref.status)))

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        resp = self.client.search(namespace, query=query, limit=top_k)
        items: list[Item] = []
        for r in resp.results:
            text = r.memory or r.chunk
            if text:
                items.append(Item(text=f"- {text}", score=r.similarity))
        return Retrieval(
            context="\n".join(i.text for i in items), items=items, usage=Usage(api_calls=1)
        )

    def _wait(self, namespace: str, doc_id: str, status: str) -> int:
        """Poll one document until it is indexed and its memories are built. Returns the
        number of polling calls."""
        calls = 0
        dreaming = None  # the add response does not report it
        deadline = time.monotonic() + self.ingest_timeout_s
        while status != "done" or dreaming != "done":
            if status == "failed":
                raise RuntimeError(f"Supermemory document {doc_id} failed")
            if time.monotonic() > deadline:
                raise IngestTimeoutError(
                    f"Supermemory document {doc_id} still {status} (memories: {dreaming})"
                )
            time.sleep(self.poll_interval_s)
            system = self.client.documents.get(namespace, doc_id).system
            status = str(system.status)
            # In the v5 API but not typed by SDK 5.0.0, which keeps unknown fields as-is.
            dreaming = getattr(system, "dreamingStatus", None)
            calls += 1
        return calls


def _iso(when: datetime) -> str:
    # Dataset timestamps are naive; treat them as UTC, as the other connectors do.
    return when.isoformat() + ("Z" if when.tzinfo is None else "")
