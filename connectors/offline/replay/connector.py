"""Replay search results recorded from a memory system, without calling its API.

- ``path`` is a recorded run's ``output.json``: ``{"conversations": [{"sample_id",
  "searches": [{"request": {"query", "top_k"}, "search_response": ...}]}]}``.
- Nothing is ingested; ``reset`` and ``ingest`` do nothing.
- ``retrieve`` finds the recorded search for the query in the namespace's conversation and
  keeps its first ``top_k`` results, so one recording at a large top_k serves any smaller k.
- ``tool`` picks how a raw ``search_response`` becomes items. Each parser formats context
  exactly like that system's live connector, so replayed and live runs are comparable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session


def _mem0(response: Any) -> list[Item]:
    # Same text as connectors/mem0/platform.
    results = response.get("results", []) if isinstance(response, dict) else response
    items = []
    for r in results:
        when = (r.get("created_at") or "")[:10]
        text = f"- {r['memory']}" + (f" (recorded {when})" if when else "")
        items.append(Item(text=text, score=r.get("score")))
    return items


_PARSERS: dict[str, Callable[[Any], list[Item]]] = {"mem0": _mem0}


def _safe(conversation_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", conversation_id)


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        if not self.settings.get("path"):
            raise ValueError("set path to a recorded output.json")
        tool = self.settings.get("tool")
        if tool not in _PARSERS:
            raise ValueError(f"tool must be one of {sorted(_PARSERS)}")
        self.parse = _PARSERS[tool]
        data = json.loads(Path(self.settings["path"]).read_text(encoding="utf-8"))
        # sample_id -> query -> recorded search (first one wins for repeated questions)
        self.searches: dict[str, dict[str, dict[str, Any]]] = {}
        for conv in data["conversations"]:
            by_query = self.searches.setdefault(conv["sample_id"], {})
            for s in conv["searches"]:
                by_query.setdefault(s["request"]["query"], s)

    def reset(self, namespace: str) -> None:
        pass

    def ingest(self, namespace: str, session: Session) -> Usage:
        return Usage()

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        # The harness names namespaces "mj-<run>-<conversation id with unsafe chars as '-'>".
        sample = next((s for s in self.searches if namespace.endswith("-" + _safe(s))), None)
        if sample is None:
            raise KeyError(f"no recorded conversation for namespace {namespace!r}")
        search = self.searches[sample].get(query)
        if search is None:
            raise KeyError(f"no recorded search in {sample} for query {query!r}")
        if search.get("http_status", 200) != 200:
            raise RuntimeError(f"recorded search failed with HTTP {search['http_status']}")
        recorded_k = int(search["request"].get("top_k") or 0)
        if recorded_k and top_k > recorded_k:
            raise ValueError(f"top_k {top_k} exceeds the recorded top_k {recorded_k}")
        items = self.parse(search["search_response"])[:top_k]
        return Retrieval(context="\n".join(i.text for i in items), items=items)
