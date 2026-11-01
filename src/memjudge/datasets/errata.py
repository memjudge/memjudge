"""Errata overlays: versioned, reviewed corrections to upstream gold data (ADR 0004).

An overlay lives at ``errata/<dataset>-<version>.yaml``::

    dataset: locomo
    version: v1
    changes:
      - question_id: "conv-26:q12"
        field: gold_answer
        old: "7 May 2023"
        new: "8 May 2023"
        reason: "Session 1 is dated 8 May; 'yesterday' refers to 7 May"  # why
        source: https://github.com/...                                     # evidence

``old`` must match the current value, so an overlay can never silently apply
to the wrong data. Applied overlays change the dataset hash.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from memjudge.schema import Question, QuestionKind

PATCHABLE_FIELDS = {"gold_answer", "category", "kind", "evidence_turn_ids", "distractor_answer"}


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    field: str
    old: Any
    new: Any
    reason: str
    source: str | None = None


class Errata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str
    changes: list[Change] = []


def load_errata(name: str, version: str) -> Errata | None:
    path = resources.files("memjudge.datasets").joinpath("errata", f"{name}-{version}.yaml")
    if not path.is_file():
        return None
    return Errata.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def apply_errata(questions: list[Question], errata: Errata) -> list[Question]:
    by_id = {q.id: i for i, q in enumerate(questions)}
    out = list(questions)
    for change in errata.changes:
        if change.field not in PATCHABLE_FIELDS:
            raise ValueError(f"errata: field {change.field!r} cannot be patched")
        if change.question_id not in by_id:
            raise ValueError(f"errata: unknown question {change.question_id!r}")
        idx = by_id[change.question_id]
        current = out[idx]
        old = getattr(current, change.field)
        if isinstance(old, tuple):
            old = list(old)
        if isinstance(old, QuestionKind):
            old = old.value
        if old != change.old:
            raise ValueError(
                f"errata: {change.question_id}.{change.field} is {old!r}, expected {change.old!r}"
            )
        out[idx] = Question.model_validate({**current.model_dump(), change.field: change.new})
    return out
