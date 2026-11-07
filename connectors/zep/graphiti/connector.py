"""Zep Graphiti open source (``graphiti-core``), run in-process against a local FalkorDB.

- Start FalkorDB first, e.g. ``docker run -p 6379:6379 falkordb/falkordb``. The hosted
  runner starts its own copy inside the job, empty for every run.
- Each benchmark namespace is its own FalkorDB graph (Graphiti's ``group_id``), so
  namespaces stay isolated; ``reset`` deletes that graph.
- The LLM and embedder are the same as mem0/oss (gpt-6-luna, text-embedding-3-small), so
  the comparison is about the memory design, not the model. The reranker is Graphiti's
  default.
- Each turn is added as its own episode, as in every connector, with the turn's date as
  ``reference_time`` so Graphiti can build valid-time ranges for facts. ``add_episode``
  returns once the episode is processed, so data is searchable when ``ingest`` returns.
- Retrieval searches facts (edges, cross-encoder reranked) and entities (nodes, RRF) at the
  same time and formats them like the zep/cloud connector, with fact validity dates.
- graphiti-core is async; one private event loop in a background thread runs every call.
  Graphiti's OpenAI client does not report token usage, so only calls are counted.
"""

from __future__ import annotations

import os

# Graphiti sends PostHog telemetry by default. Hosted runs only allow declared domains,
# and a benchmark must not leak run metadata.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

import asyncio
import threading
from collections.abc import Coroutine, Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar

from falkordb.asyncio import FalkorDB
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session

T = TypeVar("T")
U = TypeVar("U")


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="graphiti", daemon=True).start()
        self._lock = threading.Lock()
        self._ready: set[str] = set()
        self._graphiti = self._connect()

    def reset(self, namespace: str) -> None:
        with self._lock:
            self._ready.discard(namespace)
        self._run(self._delete_graph(namespace))

    def ingest(self, namespace: str, session: Session) -> Usage:
        self._ensure_graph(namespace)
        calls = 0
        for turn in session.turns:
            when = turn.timestamp or session.started_at
            self._run(
                self._graphiti.add_episode(
                    name=f"{session.id}-{turn.id}",
                    episode_body=f"{turn.speaker}: {turn.text}",
                    source_description="conversation message",
                    reference_time=_utc(when) if when else datetime.now(UTC),
                    source=EpisodeType.message,
                    group_id=namespace,
                )
            )
            calls += 1
        return Usage(api_calls=calls)

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        with self._lock:
            if namespace not in self._ready:
                return Retrieval(context="", usage=Usage(api_calls=0))
        facts = round(top_k * 2 / 3)  # 20 facts + 10 entities, or 7 + 3 (ADR 0018)
        # Copies: search_ must never change Graphiti's shared recipe objects.
        edge_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(update={"limit": facts})
        node_config = NODE_HYBRID_SEARCH_RRF.model_copy(update={"limit": top_k - facts})
        # Both searches at once, as an app would send them, so latency is not their sum.
        edges, nodes = self._run(
            _both(
                self._graphiti.search_(query, edge_config, group_ids=[namespace]),
                self._graphiti.search_(query, node_config, group_ids=[namespace]),
            )
        )
        items: list[Item] = []
        fact_lines = []
        for i, e in enumerate(edges.edges):
            since, until = _date(e.valid_at) or "date unknown", _date(e.invalid_at) or "present"
            line = f"- {e.fact} ({since} – {until})"
            fact_lines.append(line)
            items.append(Item(text=line, score=_score(edges.edge_reranker_scores, i)))
        node_lines = []
        for i, n in enumerate(nodes.nodes):
            line = f"- {n.name}: {n.summary}"
            node_lines.append(line)
            items.append(Item(text=line, score=_score(nodes.node_reranker_scores, i)))
        parts = []
        if fact_lines:
            parts.append("FACTS (valid from – until):\n" + "\n".join(fact_lines))
        if node_lines:
            parts.append("ENTITIES:\n" + "\n".join(node_lines))
        return Retrieval(context="\n\n".join(parts), items=items, usage=Usage(api_calls=2))

    def close(self) -> None:
        self._run(self._graphiti.close())  # type: ignore[no-untyped-call]
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _ensure_graph(self, namespace: str) -> None:
        with self._lock:
            if namespace in self._ready:
                return
        driver = self._graphiti.driver.clone(database=namespace)
        self._run(driver.build_indices_and_constraints())
        with self._lock:
            self._ready.add(namespace)

    def _connect(self) -> Graphiti:
        host = str(self.settings.get("falkordb_host", "localhost"))
        port = int(self.settings.get("falkordb_port", 6379))
        try:
            self._falkor = FalkorDB(host=host, port=port)  # connects to check for a cluster
        except Exception as exc:
            raise ConnectionError(
                f"zep/graphiti: no FalkorDB at {host}:{port}. Start one with: "
                f"docker run -p {port}:6379 falkordb/falkordb"
            ) from exc
        key = self.keys["GRAPHITI_OPENAI_API_KEY"]
        llm = LLMConfig(
            api_key=key,
            model=self.settings.get("llm_model"),
            small_model=self.settings.get("small_model"),
        )
        embedder = OpenAIEmbedderConfig(api_key=key)
        if self.settings.get("embedder_model"):
            embedder.embedding_model = str(self.settings["embedder_model"])
        # Built outside the event loop: inside it, the driver would also start building
        # indexes in an unused default graph. Each namespace graph gets them in _ensure_graph.
        return Graphiti(
            graph_driver=FalkorDriver(falkor_db=self._falkor),
            llm_client=OpenAIClient(config=llm),
            embedder=OpenAIEmbedder(config=embedder),
            cross_encoder=OpenAIRerankerClient(config=LLMConfig(api_key=key)),
        )

    async def _delete_graph(self, namespace: str) -> None:
        if namespace in await self._falkor.list_graphs():
            await self._falkor.select_graph(namespace).delete()

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


async def _both(a: Coroutine[Any, Any, T], b: Coroutine[Any, Any, U]) -> tuple[T, U]:
    return await asyncio.gather(a, b)


def _utc(when: datetime) -> datetime:
    # Dataset dates are naive; treat them as UTC, as the zep/cloud connector does.
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when


def _date(value: datetime | None) -> str | None:
    return value.date().isoformat() if value else None


def _score(scores: list[float], i: int) -> float | None:
    return scores[i] if i < len(scores) else None
