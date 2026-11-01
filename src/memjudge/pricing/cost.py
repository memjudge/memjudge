"""Cost from API-reported values first, then the versioned price table, else unknown."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import yaml


@dataclass(frozen=True)
class PriceTable:
    version: str
    models: dict[str, dict[str, Any]]

    def token_cost(self, model_key: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.models.get(model_key)
        if price is None:
            return None
        return float(input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


@cache
def load_price_table(version: str = "v1") -> PriceTable:
    text = resources.files("memjudge.pricing").joinpath(f"{version}.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    return PriceTable(version=str(data["version"]), models=dict(data["models"]))


def judge_model_key(route: str, model: str, upstream_provider: str | None) -> str:
    """Price-table key: OpenRouter routes are priced like the upstream provider's own API."""
    if route == "openrouter" and upstream_provider:
        return f"{upstream_provider}:{model.split('/', 1)[-1].replace('.', '-')}"
    return f"{route}:{model}"
