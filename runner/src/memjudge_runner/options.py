"""What the /run form offers: datasets, connectors, judge routes and cost estimates.

Built once per image (``memjudge-runner build-options``), so the API never loads datasets
and the estimates match the pinned code exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from memjudge.connectors.base import ConnectorSpec
from memjudge.connectors.loader import list_connectors, load_spec
from memjudge.datasets.loader import load_dataset
from memjudge.judge.client import KEY_ENV
from memjudge.judge.setup import pinned_judge
from pydantic import BaseModel, ConfigDict

from memjudge_runner.config import (
    AFFILIATED_SYSTEMS,
    BUDGET_TOKENS,
    EXCLUDED_CONNECTORS,
    HOSTED_DATASETS,
    subnet_name,
)


class DatasetOption(BaseModel):
    name: str
    label: str
    hash: str
    license: str | None
    conversations: int
    questions: int


class ConnectorOption(BaseModel):
    id: str
    target: str
    label: str
    required_keys: list[str]
    allowed_domains: list[str]
    presets: list[str] = []
    """Names from ``connector.yaml``; a run must pick one when there are any."""
    models: dict[str, dict[str, Any]] | Literal["vendor"] = {}
    """The models a run uses with each preset (key ``""`` without presets), or ``vendor``."""


class RouteOption(BaseModel):
    route: str
    spec: str
    key_name: str


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_tokens: int
    judge_season: str
    datasets: list[DatasetOption]
    connectors: list[ConnectorOption]
    judge_routes: list[RouteOption]
    estimates: dict[str, dict[str, dict[str, Any]]]
    """``estimates[dataset][route]``: the output of ``memjudge estimate --json``."""
    affiliations: tuple[str, ...] = AFFILIATED_SYSTEMS

    def dataset(self, name: str) -> DatasetOption | None:
        return next((d for d in self.datasets if d.name == name), None)

    def connector(self, connector_id: str) -> ConnectorOption | None:
        return next((c for c in self.connectors if c.id == connector_id), None)

    def route(self, route: str) -> RouteOption | None:
        return next((r for r in self.judge_routes if r.route == route), None)


def _models(spec: ConnectorSpec) -> dict[str, dict[str, Any]] | Literal["vendor"]:
    if spec.models == "vendor":
        return "vendor"
    out: dict[str, dict[str, Any]] = {}
    for name in list(spec.presets) or [""]:
        settings = {**spec.settings, **spec.preset(name or None)}
        out[name] = {m: settings.get(m) for m in spec.models}
    return out


def is_hosted_connector(spec: ConnectorSpec) -> bool:
    """Hosted runs use each connector's declared defaults only: no custom endpoint."""
    return spec.id not in EXCLUDED_CONNECTORS and spec.settings.get("base_url") is None


def egress_profiles() -> dict[str, list[str]]:
    """Each hosted connector's subnet and the domains its ``connector.yaml`` declares:
    the only internet hosts its retrieve jobs may reach (see ``infra/setup.sh``)."""
    specs = [load_spec(cid) for cid in list_connectors()]
    return {subnet_name(s.id): list(s.allowed_domains) for s in specs if is_hosted_connector(s)}


def estimate(dataset: str, judge_spec: str, budget: int) -> dict[str, Any]:
    cmd = [sys.executable, "-c", "from memjudge.cli import main; main()", "estimate"]
    args = ["--dataset", dataset, "--judge", judge_spec, "--budget", str(budget), "--json"]
    out = subprocess.run(cmd + args, capture_output=True, text=True, check=True)
    result: dict[str, Any] = json.loads(out.stdout)
    return result


def build_options(datasets: tuple[str, ...] = HOSTED_DATASETS) -> Options:
    pinned = pinned_judge()
    routes = [
        RouteOption(route=r, spec=spec, key_name=KEY_ENV[r]) for r, spec in pinned.routes.items()
    ]
    ds_options = []
    for name in datasets:
        loaded = load_dataset(name)
        ds_options.append(
            DatasetOption(
                name=name,
                label=loaded.label,
                hash=loaded.hash,
                license=loaded.dataset.license,
                conversations=len(loaded.dataset.conversations),
                questions=len(loaded.dataset.questions),
            )
        )
    connectors = []
    for cid in list_connectors():
        spec = load_spec(cid)
        if is_hosted_connector(spec):
            connectors.append(
                ConnectorOption(
                    id=cid,
                    target=spec.target,
                    label=spec.label,
                    required_keys=spec.required_keys,
                    allowed_domains=spec.allowed_domains,
                    presets=list(spec.presets),
                    models=_models(spec),
                )
            )
    return Options(
        budget_tokens=BUDGET_TOKENS,
        judge_season=pinned.season,
        datasets=ds_options,
        connectors=connectors,
        judge_routes=routes,
        estimates={
            d: {r.route: estimate(d, r.spec, BUDGET_TOKENS) for r in routes} for d in datasets
        },
    )


def load_options(path: Path) -> Options:
    return Options.model_validate_json(path.read_text(encoding="utf-8"))
