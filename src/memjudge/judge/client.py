"""Minimal HTTP clients for the judge model, one per route (ADR 0001).

Routes: the provider's own API (``anthropic``, ``openai``, ``google``) or
``openrouter`` with the upstream provider pinned and fallback routing disabled.
Plain httpx keeps one code path per route and avoids three vendor SDKs.
Sampling parameters are left at provider defaults (several current models reject
them); reasoning effort is pinned instead.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

Route = Literal["anthropic", "openai", "google", "openrouter"]

KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class JudgeModel(BaseModel):
    """Everything that pins the judge. Recorded in every result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Route
    model: str
    upstream_provider: str | None = None
    """OpenRouter only: the single upstream provider allowed to serve the model."""
    effort: str | None = "low"
    max_output_tokens: int = 4000
    timeout_s: float = 600.0
    """Long enough for any judge call to finish: a call that times out on our side may still
    be billed, and its retry would then go uncounted."""

    @classmethod
    def parse(cls, spec: str, **overrides: Any) -> JudgeModel:
        """Parse ``route:model`` or ``openrouter:vendor/model@provider``."""
        route, sep, rest = spec.partition(":")
        if not sep or route not in KEY_ENV:
            raise ValueError(
                f"judge must look like '<route>:<model>' with route in {list(KEY_ENV)}"
            )
        upstream = None
        if route == "openrouter":
            rest, at, upstream_part = rest.partition("@")
            if not at:
                raise ValueError(
                    "openrouter judge must pin a provider: openrouter:<model>@<provider>"
                )
            upstream = upstream_part
        return cls(route=route, model=rest, upstream_provider=upstream, **overrides)  # type: ignore[arg-type]

    @property
    def spec(self) -> str:
        suffix = f"@{self.upstream_provider}" if self.upstream_provider else ""
        return f"{self.route}:{self.model}{suffix}"


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    """Only when the route reports it (OpenRouter)."""
    served_by: str | None = None
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)


class JudgeError(RuntimeError):
    pass


class JudgeClient:
    def __init__(self, judge: JudgeModel, api_key: str, http: httpx.Client | None = None) -> None:
        self.judge = judge
        self._key = api_key
        self._http = http or httpx.Client(timeout=judge.timeout_s)

    def complete(self, system: str, user: str, max_attempts: int = 6) -> Completion:
        url, headers, body = self._request(system, user)
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._http.post(url, headers=headers, json=body)
            except httpx.TransportError as exc:
                if attempt == max_attempts:
                    raise JudgeError(f"judge request failed: {exc.__class__.__name__}") from None
            else:
                if resp.status_code == 200:
                    try:
                        completion = self._parse(resp.json())
                    except Exception as exc:  # not JSON, or not the shape this route returns
                        raise JudgeError(
                            f"malformed judge response: {exc.__class__.__name__}: {str(exc)[:200]}"
                        ) from None
                    completion.attempts = attempt
                    return completion
                if resp.status_code not in RETRYABLE_STATUS or attempt == max_attempts:
                    # Never echo request headers: they hold the key.
                    raise JudgeError(f"judge HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60.0)
        raise JudgeError("unreachable")

    def close(self) -> None:
        self._http.close()

    # --- per-route request/response shapes ---------------------------------

    def _request(self, system: str, user: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        j = self.judge
        if j.route == "anthropic":
            body: dict[str, Any] = {
                "model": j.model,
                "max_tokens": j.max_output_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if j.effort:
                body["output_config"] = {"effort": j.effort}
            headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01"}
            return "https://api.anthropic.com/v1/messages", headers, body
        if j.route in ("openai", "openrouter"):
            body = {
                "model": j.model,
                "max_completion_tokens": j.max_output_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if j.route == "openai":
                if j.effort:
                    body["reasoning_effort"] = j.effort
                url = "https://api.openai.com/v1/chat/completions"
            else:
                body["provider"] = {"order": [j.upstream_provider], "allow_fallbacks": False}
                body["usage"] = {"include": True}
                if j.effort:
                    body["reasoning"] = {"effort": j.effort}
                url = "https://openrouter.ai/api/v1/chat/completions"
            return url, {"Authorization": f"Bearer {self._key}"}, body
        # google
        gen: dict[str, Any] = {"maxOutputTokens": j.max_output_tokens}
        if j.effort:
            gen["thinkingConfig"] = {"thinkingLevel": j.effort}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{j.model}:generateContent"
        return url, {"x-goog-api-key": self._key}, body

    def _parse(self, data: dict[str, Any]) -> Completion:
        route = self.judge.route
        if route == "anthropic":
            text = "".join(
                b.get("text", "") for b in data.get("content", []) if b["type"] == "text"
            )
            usage = data.get("usage", {})
            return Completion(
                text=text,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                served_by=data.get("model"),
                raw=data,
            )
        if route in ("openai", "openrouter"):
            choices = data.get("choices") or []
            text = (choices[0].get("message", {}).get("content") or "") if choices else ""
            usage = data.get("usage") or {}
            served = data.get("model")
            if route == "openrouter" and data.get("provider"):
                served = f"{served}@{data['provider']}"
            return Completion(
                text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                cost_usd=usage.get("cost") if route == "openrouter" else None,
                served_by=served,
                raw=data,
            )
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        meta = data.get("usageMetadata", {})
        output = (meta.get("candidatesTokenCount") or 0) + (meta.get("thoughtsTokenCount") or 0)
        return Completion(
            text=text,
            input_tokens=meta.get("promptTokenCount"),
            output_tokens=output or None,
            served_by=data.get("modelVersion"),
            raw=data,
        )
