"""The connector contract every memory system implements (ADR 0007).

A connector lives in ``connectors/<system>/<name>/`` with ``connector.py``
(defining a class named ``Connector``) and ``connector.yaml``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memjudge.schema import Conversation, Session


@dataclass
class Usage:
    """What the memory system reported for one call. ``None`` means not reported."""

    api_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    credits: float | None = None
    cost_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __add__(self, other: Usage) -> Usage:
        def add(a: float | None, b: float | None) -> Any:
            return None if a is None and b is None else (a or 0) + (b or 0)

        return Usage(
            api_calls=self.api_calls + other.api_calls,
            input_tokens=add(self.input_tokens, other.input_tokens),
            output_tokens=add(self.output_tokens, other.output_tokens),
            credits=add(self.credits, other.credits),
            cost_usd=add(self.cost_usd, other.cost_usd),
        )


@dataclass
class Item:
    """One retrieved memory. ``source_turn_ids`` enables exact evidence recall."""

    text: str
    score: float | None = None
    source_turn_ids: tuple[str, ...] = ()


@dataclass
class Retrieval:
    context: str
    """Exactly the text a downstream LLM would be given. This is what is judged and counted."""
    items: list[Item] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class ConnectorSpec(BaseModel):
    """Schema of ``connector.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="<system>/<name>, matches the folder path")
    target: str = Field(description="Memory system and product tested, e.g. 'Mem0 Platform'")
    author: str
    label: Literal["maintainer-written", "vendor-maintained"] = "maintainer-written"
    sdk: str | None = Field(default=None, description="Pinned SDK requirement, e.g. mem0ai==2.2.1")
    required_keys: list[str] = []
    allowed_domains: list[str] = []
    """Hosts the connector may call. An endpoint on any other host is a custom endpoint."""
    settings: dict[str, Any] = {}
    """Default settings; part of the published run config."""
    presets: dict[str, dict[str, Any]] = {}
    """Named setting overrides a hosted run picks from; the first is the form's default."""
    models: list[str] | Literal["vendor"] = []
    """The settings that name the models the system runs, or ``vendor`` when the vendor picks
    them and does not say which (ADR 0014). Empty for systems without a model."""
    max_concurrency: int | None = Field(default=None, ge=1)
    """Most conversations a run may process at once, when the vendor's free plan limits it
    (ADR 0013). Applies to every run of this connector; None = no cap."""

    @model_validator(mode="after")
    def _check_presets(self) -> ConnectorSpec:
        for name, overrides in self.presets.items():
            unknown = set(overrides) - set(self.settings)
            if unknown:
                raise ValueError(f"preset {name!r} sets unknown settings: {sorted(unknown)}")
            if "base_url" in overrides:
                raise ValueError(f"preset {name!r} cannot set base_url")
        if isinstance(self.models, list) and (unknown := set(self.models) - set(self.settings)):
            raise ValueError(f"models names unknown settings: {sorted(unknown)}")
        return self

    def preset(self, name: str | None) -> dict[str, Any]:
        """The overrides of preset ``name``; none for ``None``."""
        if name is None:
            return {}
        if name not in self.presets:
            raise ValueError(f"connector {self.id} has no preset {name!r}")
        return dict(self.presets[name])

    def model_values(self, settings: Mapping[str, Any]) -> dict[str, Any] | Literal["vendor"]:
        """The models a run with ``settings`` uses: setting name -> value, or ``vendor``."""
        if self.models == "vendor":
            return "vendor"
        return {name: settings.get(name) for name in self.models}

    def concurrency(self, requested: int) -> int:
        """``requested``, lowered to ``max_concurrency`` when the connector sets one."""
        return min(requested, self.max_concurrency or requested)


class BaseConnector(ABC):
    """Implement ``reset``, ``ingest`` and ``retrieve``. Methods may be called from threads,
    but never concurrently for the same namespace."""

    endpoint: str | None = None
    """Base URL of the API this instance calls; None for in-process systems (e.g. BM25).

    Connectors that call an API set it in ``__init__`` from the ``base_url`` setting, or
    from the vendor's default. It is recorded (without credentials) in every result.
    """

    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        self.keys = dict(keys)
        self.settings = dict(settings)

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Delete everything stored under ``namespace``. Must succeed if it does not exist."""

    @abstractmethod
    def ingest(self, namespace: str, session: Session) -> Usage:
        """Store one session. Must return only once the data is searchable."""

    @abstractmethod
    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget_tokens: int,
        top_k: int,
    ) -> Retrieval:
        """Return context for ``query``: at most ``top_k`` items (10 or 30, ADR 0018).
        ``budget_tokens`` is a hint; the harness enforces it."""

    def ingest_batch(
        self,
        conversations: Mapping[str, Conversation],
        state_dir: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> Usage | None:
        """Optional: ingest the whole run in one go (ADR 0010). Default: not supported.

        ``conversations`` maps each namespace the run needs to its conversation. A
        connector that supports it uploads everything at once (e.g. to a vendor's batch
        API) and returns only when all of it is searchable, with the usage; the harness
        then skips per-session ingestion. Raise when the batch cannot complete: the
        run stops. Return ``None`` when not enabled, to ingest session by session.
        ``state_dir`` is the run folder, for state that lets a resumed run wait for the
        same batch instead of uploading again. ``progress(done, total)``, when given, is
        called while the batch runs with the turns processed so far and the turns uploaded.
        """
        return None

    def save_state(self, namespace: str) -> dict[str, Any] | None:
        """What this connector needs to find ``namespace``'s stored data again, if anything.

        Used to rerun retrieval later without re-ingesting (``--reuse-ingestion``). Return
        JSON-serialisable data, never keys. Connectors whose data is found by the namespace
        alone (e.g. it is the vendor's user id) keep the default.
        """
        return None

    def restore_state(self, namespace: str, state: dict[str, Any] | None) -> None:  # noqa: B027
        """Load what ``save_state`` returned, so ``retrieve``/``reset`` work on old data."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release clients. Optional."""


def public_url(url: str) -> str:
    """``url`` without credentials, query or fragment, safe to publish in a result."""
    parts = urlsplit(url.strip())
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))


def is_custom_endpoint(url: str | None, allowed_domains: list[str]) -> bool:
    """True when ``url`` points anywhere other than one of the connector's declared hosts.

    Runs against a custom endpoint (localhost, staging, self-hosted) test a different
    deployment than the vendor's product; results mark them and are never published.
    """
    if url is None:
        return False
    host = (urlsplit(url).hostname or "").lower()
    return host not in {d.lower() for d in allowed_domains}
