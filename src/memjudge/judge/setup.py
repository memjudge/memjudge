"""Resolve which judge to use and whether it is the pinned (official) one.

The pinned judge is the current season's: the last entry of ``seasons.json``. A season is one
fixed set of judging rules; any change to them is a new season (docs/seasons.md).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal

from memjudge.harness import JudgeSetup
from memjudge.judge.client import KEY_ENV, JudgeClient, JudgeModel
from memjudge.judge.judge import load_prompt
from memjudge.pricing.cost import judge_model_key, load_price_table
from memjudge.result import JudgeInfo

Upgrade = Literal["re-judge", "re-run"]


@dataclass(frozen=True)
class PinnedJudge:
    season: str
    provisional: bool
    routes: dict[str, str]
    default_route: str
    effort: str | None
    max_output_tokens: int
    prompt_version: str
    prompt_hash: str


@dataclass(frozen=True)
class Season:
    id: str
    name: str
    started: str
    summary: str
    details: str | None
    upgrade: Upgrade
    """How runs judged in earlier seasons get into this one: their stored contexts can be
    judged again, or (when what the judge sees changed) they need a new run."""
    judge: PinnedJudge


@cache
def seasons() -> tuple[Season, ...]:
    """Every season, oldest first; the last one is the current season."""
    text = resources.files("memjudge.judge").joinpath("seasons.json").read_text(encoding="utf-8")
    out = []
    for s in json.loads(text):
        j = s["judge"]
        judge = PinnedJudge(
            season=str(s["id"]),
            provisional=bool(s["provisional"]),
            routes=dict(j["routes"]),
            default_route=str(j["default_route"]),
            effort=j.get("effort"),
            max_output_tokens=int(j["max_output_tokens"]),
            prompt_version=str(j["prompt_version"]),
            prompt_hash=str(j["prompt_hash"]),
        )
        out.append(
            Season(
                id=judge.season,
                name=str(s["name"]),
                started=str(s["started"]),
                summary=str(s["summary"]),
                details=s.get("details"),
                upgrade=s["upgrade"],
                judge=judge,
            )
        )
    return tuple(out)


def pinned_judge() -> PinnedJudge:
    return seasons()[-1].judge


def can_rejudge(season: str | None) -> bool:
    """Whether a run judged in ``season`` can be judged again into the current season: it is
    an earlier season and every season after it allows re-judging stored contexts."""
    ids = [s.id for s in seasons()]
    if season not in ids or season == ids[-1]:
        return False
    return all(s.upgrade == "re-judge" for s in seasons()[ids.index(season) + 1 :])


def resolve_judge(spec: str | None, env: Mapping[str, str]) -> JudgeModel:
    """``spec`` overrides the pinned judge. Without it, use the pinned judge on whichever
    route the user has a key for (direct provider first, then OpenRouter)."""
    pinned = pinned_judge()
    settings = {"effort": pinned.effort, "max_output_tokens": pinned.max_output_tokens}
    if spec:
        return JudgeModel.parse(spec, **settings)
    order = [pinned.default_route] + [r for r in pinned.routes if r != pinned.default_route]
    for route in order:
        if env.get(KEY_ENV[route]):
            return JudgeModel.parse(pinned.routes[route], **settings)
    wanted = " or ".join(KEY_ENV[r] for r in order)
    raise KeyError(f"no judge key found: set {wanted}, or pass --judge")


def is_official(judge: JudgeModel) -> bool:
    pinned = pinned_judge()
    return (
        judge.spec in pinned.routes.values()
        and judge.effort == pinned.effort
        and judge.max_output_tokens == pinned.max_output_tokens
        and load_prompt().hash == pinned.prompt_hash
    )


def make_judge_setup(judge: JudgeModel, env: Mapping[str, str]) -> JudgeSetup:
    key = env.get(KEY_ENV[judge.route])
    if not key:
        raise KeyError(f"judge route {judge.route} needs {KEY_ENV[judge.route]}")
    prompt = load_prompt()
    table = load_price_table()
    price_key = judge_model_key(judge.route, judge.model, judge.upstream_provider)
    official = is_official(judge)

    def info(served_by: list[str]) -> JudgeInfo:
        return JudgeInfo(
            route=judge.route,
            model=judge.model,
            upstream_provider=judge.upstream_provider,
            effort=judge.effort,
            max_output_tokens=judge.max_output_tokens,
            prompt_version=prompt.version,
            prompt_hash=prompt.hash,
            served_by=served_by,
            official=official,
            season=pinned_judge().season if official else None,
        )

    return JudgeSetup(
        client=JudgeClient(judge, key),
        prompt=prompt,
        info=info,
        price=lambda i, o: table.token_cost(price_key, i, o),
    )
