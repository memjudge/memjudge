"""Recalld (hosted API) over its public REST API (https://recalld.ai/openapi.yaml).

- One Recalld agent, with one thread, per benchmark namespace. The agent is created on
  the first ingest and deleted on reset, so no empty agents are left behind. An agent
  created by an attempt that was interrupted before its conversation completed is not
  found again on ``--resume``; delete it in the Recalld console.
- Each turn is sent as its own synchronous ingest call (one message per call, as in
  every connector); the API returns once the write is complete, so ingestion time is
  time-to-searchable.
- Each turn becomes one input with its speaker as ``author`` and its time as
  ``event_date``. Assistant turns are ``AGENT`` inputs; every other speaker is ``USER``.
- Retrieval uses ``/memory/recall`` (full pipeline) or ``/memory/search`` (vector only),
  with the question time as the "as of" date.
- Every agent is created with explicit models (``decompose_model``, ``evaluate_model``,
  ``select_model``, ``embedding_model``), so the account's defaults never apply. Recalld
  reports the models each memory call ran with (``X-Recalld-Models``); any difference
  from the requested ones stops the run (ADR 0014). Batch runs are not checked per call.
- The API region comes from the key prefix (``rcd_eu_``, ``rcd_us_``); ``base_url``
  overrides it, e.g. for a local or staging server.
- ``batch: true`` (ADR 0010) uploads every turn of the run in one Recalld batch and
  waits until Recalld has stored them all. Recalld ingests them in order on the model
  provider's Flex tier, with LLM tokens at half price but more slowly. On the EU host,
  a model whose batch processing leaves the EU needs ``allow_non_eu: true``.

Disclosure: Recalld is built by the maintainer of MemJudge.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Conversation, Session, Turn

log = logging.getLogger(__name__)

_REGIONS = {"rcd_eu_": "https://eu.recalld.ai", "rcd_us_": "https://us.recalld.ai"}
_RETRYABLE = {429, 500, 502, 503, 504}
_BATCH_FILE = "recalld-batch.json"
_BATCH_DONE = {"completed", "failed", "cancelled"}
_MODELS_HEADER = "x-recalld-models"


class RecalldError(RuntimeError):
    pass


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        key = self.keys["RECALLD_API_KEY"]
        base_url = self.settings.get("base_url") or next(
            (url for prefix, url in _REGIONS.items() if key.startswith(prefix)), None
        )
        if not base_url:
            raise ValueError("cannot tell the Recalld region from the key; set base_url")
        self.endpoint = str(base_url).rstrip("/")
        self.retrieval = str(self.settings.get("retrieval", "recall"))
        if self.retrieval not in ("recall", "search"):
            raise ValueError("retrieval must be 'recall' or 'search'")
        self.models = {
            "decompose": str(self.settings["decompose_model"]),
            "evaluate": str(self.settings["evaluate_model"]),
            "select_facts": str(self.settings["select_model"]),
            "embedding": str(self.settings["embedding_model"]),
        }
        self.max_retries = int(self.settings.get("max_retries", 4))
        self.batch = bool(self.settings.get("batch", False))
        self.allow_non_eu = bool(self.settings.get("allow_non_eu", False))
        self.batch_poll_s = float(self.settings.get("batch_poll_s", 60))
        self._http = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {key}"},
            timeout=float(self.settings.get("timeout_s", 300)),
        )
        self._agents: dict[str, tuple[str, str]] = {}  # namespace -> (agent_id, thread_id)
        self._lock = threading.Lock()

    def reset(self, namespace: str) -> None:
        with self._lock:
            agent = self._agents.pop(namespace, None)
        if agent:
            self._request("DELETE", f"/v1/agents/{agent[0]}", not_found_ok=True)

    def ingest(self, namespace: str, session: Session) -> Usage:
        agent_id, thread_id, calls = self._agent(namespace)
        usage = Usage(api_calls=calls)
        for turn in session.turns:
            body = _add_body(thread_id, session, turn)
            resp = self._request("POST", f"/v1/agents/{agent_id}/memory", body, check_models=True)
            usage = usage + _usage(resp, 1)
        return usage

    def ingest_batch(
        self,
        conversations: Mapping[str, Conversation],
        state_dir: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> Usage | None:
        """Ingest the whole run in one Recalld batch (``batch: true``).

        The agents are created here and the batch id is kept in the run folder, so a
        resumed run waits for the same batch instead of uploading again. Status polls
        are not counted as API calls.
        """
        if not self.batch:
            return None
        path = state_dir / _BATCH_FILE
        state: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.exists() else {}
        with self._lock:
            for ns, (agent_id, thread_id) in state.get("agents", {}).items():
                self._agents.setdefault(ns, (str(agent_id), str(thread_id)))

        calls = 0
        batch_id = state.get("batch_id")
        if not batch_id:
            agents = state.setdefault("agents", {})
            for ns in conversations:
                agent_id, thread_id, n = self._agent(ns)
                calls += n
                agents[ns] = [agent_id, thread_id]
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            try:
                resp = self._request(
                    "POST",
                    "/v1/memory/batches",
                    content=self._batch_upload(conversations),
                    params={"allow_non_eu": "true"} if self.allow_non_eu else None,
                    retries=0,  # a retried upload could store every write twice
                )
            except RecalldError:
                self._drop_batch(conversations, path)
                raise
            batch_id = str(resp["id"])
            calls += 1
            state["batch_id"] = batch_id
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            log.info("batch %s started for %d conversations", batch_id, len(conversations))

        final = self._wait_for_batch(batch_id, progress)
        status = final.get("status")
        if status != "completed":
            self._drop_batch(conversations, path)
            if status == "cancelled":
                raise RecalldError(f"batch {batch_id} was cancelled")
            raise RecalldError(f"batch {batch_id} ended {status}: {final.get('error', '')}")
        failed = (final.get("counts") or {}).get("failed", 0)
        if failed:
            log.warning("batch %s: %d memory additions could not be stored", batch_id, failed)
        return _usage(final, calls)

    def _drop_batch(self, conversations: Mapping[str, Conversation], path: Path) -> None:
        """After a batch that cannot complete: delete the agents it made and its state."""
        for ns in conversations:
            try:
                self.reset(ns)
            except RecalldError as exc:
                log.warning("could not delete agent of %s: %s", ns, exc)
        path.unlink(missing_ok=True)

    def _batch_upload(self, conversations: Mapping[str, Conversation]) -> bytes:
        """Every turn of the run as one JSON line, in ingest order."""
        lines: list[str] = []
        for ns, conv in conversations.items():
            agent_id, thread_id = self._agents[ns]
            for session in conv.sessions:
                for turn in session.turns:
                    # The API rejects blank content; one such turn must not sink the batch.
                    if not turn.text.strip():
                        continue
                    lines.append(
                        json.dumps({"agent_id": agent_id, **_add_body(thread_id, session, turn)})
                    )
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _wait_for_batch(
        self, batch_id: str, progress: Callable[[int, int], None] | None = None
    ) -> dict[str, Any]:
        last = None
        while True:
            resp = self._request("GET", f"/v1/memory/batches/{batch_id}")
            status = resp.get("status")
            counts = resp.get("counts") or {}
            seen = (status, json.dumps(counts, sort_keys=True))
            if seen != last:
                log.info("batch %s: %s %s", batch_id, status, counts)
                last = seen
            if progress and counts.get("total"):
                progress(counts.get("completed", 0) + counts.get("failed", 0), counts["total"])
            if status in _BATCH_DONE:
                return resp
            time.sleep(self.batch_poll_s)

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        with self._lock:
            agent = self._agents.get(namespace)
        if not agent:
            return Retrieval(context="")
        agent_id, thread_id = agent
        body: dict[str, Any] = {"thread_id": thread_id, "query": query, "limit": top_k}
        if asked_at:
            body["event_date"] = _rfc3339(asked_at)
        resp = self._request(
            "POST", f"/v1/agents/{agent_id}/memory/{self.retrieval}", body, check_models=True
        )
        items: list[Item] = []
        for f in resp.get("facts") or []:
            date = (f.get("event_date") or "")[:10]
            items.append(Item(text=f"- {f.get('text', '')}" + (f" (date: {date})" if date else "")))
        return Retrieval(
            context="\n".join(i.text for i in items), items=items, usage=_usage(resp, 1)
        )

    def save_state(self, namespace: str) -> dict[str, Any] | None:
        with self._lock:
            agent = self._agents.get(namespace)
        return {"agent_id": agent[0], "thread_id": agent[1]} if agent else None

    def restore_state(self, namespace: str, state: dict[str, Any] | None) -> None:
        if state:
            with self._lock:
                self._agents[namespace] = (str(state["agent_id"]), str(state["thread_id"]))

    def close(self) -> None:
        self._http.close()

    def _agent(self, namespace: str) -> tuple[str, str, int]:
        """Return (agent_id, thread_id, api_calls_made), creating the agent if needed."""
        with self._lock:
            agent = self._agents.get(namespace)
        if agent:
            return (*agent, 0)
        llm = {k: v for k, v in self.models.items() if k != "embedding"}
        body = {
            "name": namespace[:120],
            "llm_models": llm,
            "embedding_model": self.models["embedding"],
        }
        agent_id = str(self._request("POST", "/v1/agents", body)["id"])
        thread_id = str(self._request("POST", f"/v1/agents/{agent_id}/threads", {})["id"])
        with self._lock:
            self._agents[namespace] = (agent_id, thread_id)
        return agent_id, thread_id, 2

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        not_found_ok: bool = False,
        params: dict[str, str] | None = None,
        content: bytes | None = None,
        retries: int | None = None,
        check_models: bool = False,
    ) -> dict[str, Any]:
        delay = 2.0
        headers = {"Content-Type": "application/jsonl"} if content is not None else None
        max_retries = self.max_retries if retries is None else retries
        for attempt in range(max_retries + 1):
            last = attempt == max_retries
            try:
                resp = self._http.request(
                    method, path, json=body, params=params, content=content, headers=headers
                )
            except httpx.TransportError as exc:
                if last:
                    raise RecalldError(f"{method} {path}: {exc.__class__.__name__}") from None
            else:
                if not_found_ok and resp.status_code == 404:
                    return {}
                if resp.is_success:
                    if check_models:
                        self._check_models(resp.headers.get(_MODELS_HEADER))
                    data = resp.json() if resp.content else {}
                    return data if isinstance(data, dict) else {}
                if resp.status_code not in _RETRYABLE or last:
                    raise RecalldError(f"{method} {path} -> {resp.status_code}: {resp.text[:200]}")
                retry_after = resp.headers.get("retry-after", "")
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RecalldError("unreachable")

    def _check_models(self, header: str | None) -> None:
        """Stop unless Recalld ran exactly the models this run asked for."""
        used = (
            dict(p.strip().split("=", 1) for p in header.split(",") if "=" in p) if header else {}
        )
        if used != self.models:
            raise RecalldError(
                f"Recalld ran other models than requested: {header or 'not reported'} "
                f"(expected {self.models})"
            )


def _add_body(thread_id: str, session: Session, turn: Turn) -> dict[str, Any]:
    """One turn as one memory write: the body ``ingest`` sends, and a batch line."""
    item: dict[str, Any] = {
        "kind": "AGENT" if turn.speaker == "assistant" else "USER",
        "content": turn.text,
        "content_type": "text/plain",
        "author": turn.speaker,
    }
    when = turn.timestamp or session.started_at
    if when:
        item["event_date"] = _rfc3339(when)
    return {"thread_id": thread_id, "inputs": [item]}


def _rfc3339(value: datetime) -> str:
    # Dataset timestamps are naive; they are sent as UTC.
    return value.isoformat() + ("Z" if value.tzinfo is None else "")


def _usage(resp: Mapping[str, Any], calls: int) -> Usage:
    """Token usage and credits, when the API returns them for this key."""
    raw = resp.get("usage")
    usage: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
    credits = resp.get("credits_charged", resp.get("credits"))
    return Usage(
        api_calls=calls,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        credits=float(credits) if isinstance(credits, int | float) else None,
    )
