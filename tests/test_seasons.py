"""seasons.json: the judging rules of every season, and the current season's pinned judge."""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from importlib import resources

import pytest

from memjudge.judge import setup as setup_mod
from memjudge.judge.client import JudgeModel
from memjudge.judge.judge import PROMPT_VERSION, load_prompt
from memjudge.judge.setup import can_rejudge, is_official, pinned_judge, seasons

FIELDS = {"id", "name", "started", "provisional", "summary", "details", "upgrade", "judge"}
JUDGE_FIELDS = {
    "routes",
    "default_route",
    "effort",
    "max_output_tokens",
    "prompt_version",
    "prompt_hash",
}


def _raw() -> list[dict[str, object]]:
    text = resources.files("memjudge.judge").joinpath("seasons.json").read_text("utf-8")
    data: list[dict[str, object]] = json.loads(text)
    return data


def test_every_season_is_complete() -> None:
    for s in _raw():
        assert set(s) == FIELDS, s["id"]
        judge = s["judge"]
        assert isinstance(judge, dict) and set(judge) == JUDGE_FIELDS, s["id"]
        assert s["upgrade"] in ("re-judge", "re-run")
        assert judge["default_route"] in judge["routes"]
        assert s["summary"] and s["name"]


def test_ids_are_unique_and_dates_go_up() -> None:
    ids = [s.id for s in seasons()]
    assert len(ids) == len(set(ids))
    started = [date.fromisoformat(s.started) for s in seasons()]
    assert started == sorted(started)


def test_every_season_prompt_file_is_frozen() -> None:
    # A season's prompt file may never change: editing it would rewrite past results' rules.
    for s in seasons():
        assert load_prompt(s.judge.prompt_version).hash == s.judge.prompt_hash, s.id


def test_current_season_uses_the_current_prompt() -> None:
    pinned = pinned_judge()
    assert pinned.season == seasons()[-1].id
    assert pinned.prompt_version == PROMPT_VERSION
    assert pinned.prompt_hash == load_prompt().hash


def test_can_rejudge_from_earlier_seasons_only() -> None:
    current = seasons()[-1].id
    assert not can_rejudge(current)
    assert not can_rejudge(None)
    assert not can_rejudge("no-such-season")


def test_a_rerun_season_blocks_rejudging_across_it(monkeypatch: pytest.MonkeyPatch) -> None:
    base = seasons()[-1]
    chain = (
        dataclasses.replace(base, id="a"),
        dataclasses.replace(base, id="b", upgrade="re-run"),
        dataclasses.replace(base, id="c", upgrade="re-judge"),
    )
    monkeypatch.setattr(setup_mod, "seasons", lambda: chain)
    assert not can_rejudge("a")  # b changed what the judge sees: a's contexts are stale
    assert can_rejudge("b")
    assert not can_rejudge("c")


def test_official_needs_the_season_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = pinned_judge()
    judge = JudgeModel.parse(
        pinned.routes[pinned.default_route],
        effort=pinned.effort,
        max_output_tokens=pinned.max_output_tokens,
    )
    assert is_official(judge)
    other = dataclasses.replace(load_prompt(), hash="sha256:other")
    monkeypatch.setattr(setup_mod, "load_prompt", lambda: other)
    assert not is_official(judge)
