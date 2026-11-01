"""Mem0 open source (``mem0.Memory``), run in-process with its documented full setup.

- Each benchmark namespace gets its own ``Memory``: a local Qdrant store and SQLite history
  in a temporary folder, so namespaces stay isolated and can run in parallel.
- The LLM (gpt-6-luna) is set with Mem0's documented ``is_reasoning_model`` flag:
  Mem0 2.2.1's name check misses gpt-6-luna and would send parameters the model rejects
  (temperature, ``max_tokens``). The embedder is Mem0's default, named explicitly.
- The key is passed in under its own name so this connector never receives the judge's
  key. Mem0 sends its LLM calls through OpenRouter whenever ``OPENROUTER_API_KEY`` is set,
  so ``__init__`` refuses to run then.
- Multi-signal retrieval (semantic + BM25 keyword + entity matching) needs spaCy's
  ``en_core_web_sm`` and ``fastembed``. Mem0 silently falls back to semantic-only search
  without them, so ``__init__`` checks both and fails instead.
- Each turn is added as its own call, as in every connector. Mem0 OSS rejects ``timestamp``
  (platform-only), so its extraction prompt would date every memory to the run date. The
  conversation date is passed instead through ``add(prompt=...)``, which Mem0 places in the
  prompt's Custom Instructions section, and stored in metadata so retrieval shows it.
- ``add`` and ``search`` are synchronous, so data is searchable when ``ingest`` returns.
  Mem0 OSS does not report token usage, so only calls are counted.
"""

from __future__ import annotations

import os

# Mem0 sends PostHog telemetry and fetches a remote notice config by default. Hosted runs
# only allow declared domains, and a benchmark must not leak run metadata.
os.environ.setdefault("MEM0_TELEMETRY", "False")

import shutil
import tempfile
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from mem0 import Memory

from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage
from memjudge.schema import Session

_ROLES = {"user", "assistant"}
_SPACY_MODEL = "en_core_web_sm"
_BM25_MODEL = "Qdrant/bm25"


class Connector(BaseConnector):
    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        if os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "mem0/oss: Mem0 would send its LLM calls through OpenRouter using "
                "OPENROUTER_API_KEY. Unset it (use another judge route) to run this connector."
            )
        _check_full_setup()
        self._root = Path(tempfile.mkdtemp(prefix="memjudge-mem0-oss-"))
        self._lock = threading.Lock()
        self._memories: dict[str, Memory] = {}

    def reset(self, namespace: str) -> None:
        with self._lock:
            memory = self._memories.pop(namespace, None)
        if memory is not None:
            _close(memory)
        shutil.rmtree(self._root / namespace, ignore_errors=True)

    def ingest(self, namespace: str, session: Session) -> Usage:
        memory = self._memory(namespace)
        calls = 0
        for turn in session.turns:
            when = turn.timestamp or session.started_at
            day = when.date().isoformat() if when else None
            metadata = {"session_id": session.id}
            if day:
                metadata["session_date"] = day
            memory.add(
                [_message(turn.speaker, turn.text)],
                user_id=namespace,
                metadata=metadata,
                prompt=_date_instruction(day) if day else None,
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
            memory = self._memories.get(namespace)
        if memory is None:
            return Retrieval(context="", usage=Usage(api_calls=0))
        resp = memory.search(query, top_k=top_k, filters={"user_id": namespace})
        results = resp.get("results", []) if isinstance(resp, dict) else resp
        items: list[Item] = []
        for r in results:
            when = (r.get("metadata") or {}).get("session_date")
            text = f"- {r['memory']}" + (f" (said {when})" if when else "")
            items.append(Item(text=text, score=r.get("score")))
        return Retrieval(
            context="\n".join(i.text for i in items), items=items, usage=Usage(api_calls=1)
        )

    def close(self) -> None:
        with self._lock:
            memories = list(self._memories.values())
            self._memories.clear()
        for memory in memories:
            _close(memory)
        shutil.rmtree(self._root, ignore_errors=True)

    def _memory(self, namespace: str) -> Memory:
        with self._lock:
            memory = self._memories.get(namespace)
            if memory is None:
                memory = Memory.from_config(self._config(self._root / namespace))
                self._memories[namespace] = memory
            return memory

    def _config(self, folder: Path) -> dict[str, Any]:
        key = self.keys["MEM0_OSS_OPENAI_API_KEY"]
        llm: dict[str, Any] = {"api_key": key}
        embedder: dict[str, Any] = {"api_key": key}
        qdrant: dict[str, Any] = {"path": str(folder / "qdrant"), "on_disk": True}
        if self.settings.get("llm_model"):
            llm["model"] = self.settings["llm_model"]
        if self.settings.get("llm_is_reasoning_model") is not None:
            llm["is_reasoning_model"] = bool(self.settings["llm_is_reasoning_model"])
        if self.settings.get("embedder_model"):
            embedder["model"] = self.settings["embedder_model"]
        if self.settings.get("embedding_dims"):
            embedder["embedding_dims"] = int(self.settings["embedding_dims"])
            qdrant["embedding_model_dims"] = int(self.settings["embedding_dims"])
        folder.mkdir(parents=True, exist_ok=True)
        return {
            "llm": {"provider": "openai", "config": llm},
            "embedder": {"provider": "openai", "config": embedder},
            "vector_store": {"provider": "qdrant", "config": qdrant},
            "history_db_path": str(folder / "history.db"),
        }


def _check_full_setup() -> None:
    """Fail when Mem0 would silently run without keyword or entity matching."""
    hint = 'Install the full setup with: pip install "memjudge[mem0-oss]"'
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(f"mem0/oss needs spaCy for entity matching. {hint}") from exc
    if not spacy.util.is_package(_SPACY_MODEL):
        raise ImportError(
            f"mem0/oss needs the spaCy model {_SPACY_MODEL}. "
            f"Install it with: python -m spacy download {_SPACY_MODEL}"
        )
    try:
        from fastembed import SparseTextEmbedding
    except ImportError as exc:
        raise ImportError(f"mem0/oss needs fastembed for BM25 keyword search. {hint}") from exc
    SparseTextEmbedding(model_name=_BM25_MODEL)  # downloads the model once if not cached


def _close(memory: Memory) -> None:
    memory.close()
    client = getattr(memory.vector_store, "client", None)
    if client is not None:
        client.close()


def _date_instruction(day: str) -> str:
    return (
        f"These messages are from a conversation that took place on {day}. Use {day} as the "
        "observation date instead of the Observation Date and Current Date above, and resolve "
        "relative dates against it."
    )


def _message(speaker: str, text: str) -> dict[str, str]:
    if speaker in _ROLES:
        return {"role": speaker, "content": text}
    # Named speakers (e.g. LoCoMo): keep the name in the content so attribution survives.
    return {"role": "user", "content": f"{speaker}: {text}"}
