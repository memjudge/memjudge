"""BM25 baseline: indexes every turn in memory and returns the best-scoring turns.

A deliberately simple, dependency-free reference point. Any memory system should
beat it at the same token budget.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session
from memjudge.tokens import count_tokens

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her his how i in is it "
    "its me my of on or our she so that the their them they this to was we were what when where "
    "which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


@dataclass
class _Doc:
    text: str
    turn_id: str
    terms: Counter[str]
    length: int


@dataclass
class _Index:
    docs: list[_Doc] = field(default_factory=list)
    df: Counter[str] = field(default_factory=Counter)
    total_len: int = 0


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        self.k1 = float(self.settings.get("k1", 1.5))
        self.b = float(self.settings.get("b", 0.75))
        self._indexes: dict[str, _Index] = {}
        self._lock = threading.Lock()

    def reset(self, namespace: str) -> None:
        with self._lock:
            self._indexes[namespace] = _Index()

    def ingest(self, namespace: str, session: Session) -> Usage:
        with self._lock:
            index = self._indexes.setdefault(namespace, _Index())
        for turn in session.turns:
            when = turn.timestamp or session.started_at
            prefix = f"[{when:%Y-%m-%d %H:%M}] " if when else ""
            text = f"{prefix}{turn.speaker}: {turn.text}"
            terms = Counter(tokenize(turn.text + " " + turn.speaker))
            length = sum(terms.values())
            index.docs.append(_Doc(text=text, turn_id=turn.id, terms=terms, length=length))
            index.df.update(terms.keys())
            index.total_len += length
        return Usage()

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        index = self._indexes.get(namespace)
        if not index or not index.docs:
            return Retrieval(context="")
        n = len(index.docs)
        avg_len = index.total_len / n or 1.0
        q_terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(index.docs):
            score = 0.0
            for term in q_terms:
                tf = doc.terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - index.df[term] + 0.5) / (index.df[term] + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * doc.length / avg_len)
                score += idf * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))

        items: list[Item] = []
        used = 0
        for score, i in scored[:top_k]:
            doc = index.docs[i]
            cost = count_tokens(doc.text) + 1
            if used + cost > budget_tokens:
                break
            used += cost
            items.append(Item(text=doc.text, score=score, source_turn_ids=(doc.turn_id,)))
        return Retrieval(context="\n".join(item.text for item in items), items=items)
