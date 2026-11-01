"""Mem0 Platform (hosted API, v3 memory algorithm) via the official ``mem0ai`` SDK.

- One Mem0 ``user_id`` per benchmark namespace.
- Each turn is added as its own call (one message per call, as in every connector)
  with the session start as ``timestamp``.
- Mem0 processes adds asynchronously; after each add ``ingest`` polls
  ``/v1/event/{event_id}/`` until it has SUCCEEDED before sending the next turn, so
  ingestion time includes time-to-searchable and turns are stored in order.
- Retrieval is ``search`` filtered by ``user_id``; memories are listed with their dates.
- ``base_url`` points the client at another host (e.g. a local or self-hosted server).
"""

from __future__ import annotations

import os

# Mem0's SDK sends PostHog telemetry by default. Hosted runs only allow declared domains,
# and a benchmark must not leak run metadata, so disable it before the SDK is imported.
os.environ.setdefault("MEM0_TELEMETRY", "False")

import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from mem0 import MemoryClient

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session

_ROLES = {"user", "assistant"}


class IngestTimeoutError(RuntimeError):
    pass


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        base_url = self.settings.get("base_url") or None
        self.client = MemoryClient(api_key=self.keys["MEM0_API_KEY"], host=base_url)
        self.endpoint = str(self.client.host).rstrip("/")
        self.rerank = bool(self.settings.get("rerank", False))
        self.poll_interval_s = float(self.settings.get("poll_interval_s", 1.0))
        self.ingest_timeout_s = float(self.settings.get("ingest_timeout_s", 900))

    def reset(self, namespace: str) -> None:
        try:
            self.client.delete_users(user_id=namespace)
        except Exception as exc:  # nothing stored yet is fine
            if "not found" not in str(exc).lower() and "404" not in str(exc):
                raise

    def ingest(self, namespace: str, session: Session) -> Usage:
        timestamp = int(session.started_at.timestamp()) if session.started_at else None
        kwargs: dict[str, Any] = {"user_id": namespace, "metadata": {"session_id": session.id}}
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        calls = 0
        for turn in session.turns:
            resp = self.client.add([_message(turn.speaker, turn.text)], **kwargs)
            calls += 1 + self._wait(_event_ids(resp))
        return Usage(api_calls=calls)

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        resp = self.client.search(
            query, filters={"user_id": namespace}, top_k=top_k, rerank=self.rerank
        )
        results = resp.get("results", []) if isinstance(resp, dict) else resp
        items: list[Item] = []
        for r in results:
            when = (r.get("created_at") or "")[:10]
            text = f"- {r['memory']}" + (f" (recorded {when})" if when else "")
            items.append(Item(text=text, score=r.get("score")))
        return Retrieval(
            context="\n".join(i.text for i in items), items=items, usage=Usage(api_calls=1)
        )

    def close(self) -> None:
        self.client.client.close()

    def _wait(self, events: list[str]) -> int:
        """Poll queued events until all succeed. Returns the number of polling calls."""
        pending = set(events)
        calls = 0
        deadline = time.monotonic() + self.ingest_timeout_s
        while pending:
            if time.monotonic() > deadline:
                raise IngestTimeoutError(f"{len(pending)} Mem0 events still pending after timeout")
            time.sleep(self.poll_interval_s)
            for event_id in list(pending):
                resp = self.client.client.get(f"/v1/event/{event_id}/")
                calls += 1
                resp.raise_for_status()
                status = str(resp.json().get("status", "")).upper()
                if status == "SUCCEEDED":
                    pending.discard(event_id)
                elif status == "FAILED":
                    raise RuntimeError(f"Mem0 event {event_id} failed")
        return calls


def _message(speaker: str, text: str) -> dict[str, str]:
    if speaker in _ROLES:
        return {"role": speaker, "content": text}
    # Named speakers (e.g. LoCoMo): keep the name in the content so attribution survives.
    return {"role": "user", "content": f"{speaker}: {text}"}


def _event_ids(resp: Any) -> list[str]:
    if isinstance(resp, dict):
        if resp.get("event_id"):
            return [str(resp["event_id"])]
        return [str(r["event_id"]) for r in resp.get("results", []) if r.get("event_id")]
    if isinstance(resp, list):
        return [str(r["event_id"]) for r in resp if isinstance(r, dict) and r.get("event_id")]
    return []
