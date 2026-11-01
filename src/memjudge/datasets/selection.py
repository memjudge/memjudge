"""Run part of a dataset: chosen conversations, categories, the N smallest, or a seeded sample.

Used to iterate cheaply (start with one small conversation, then widen) before a full run.
The selection is recorded in the result, and results stay tied to the full dataset hash,
so a partial run is never mistaken for a full one.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, model_validator

from memjudge.schema import Conversation, Dataset


class Selection(BaseModel):
    """Which part of a dataset to run. Filters apply in this order:

    1. ``conversation_ids`` keeps only those conversations;
    2. ``categories`` keeps only questions in those categories, and drops conversations
       left with no questions (so nothing is ingested for nothing);
    3. ``smallest`` keeps the N conversations with the fewest turns, or
       ``sample`` keeps N conversations chosen at random with ``seed``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    smallest: int | None = None
    sample: int | None = None
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> Selection:
        if self.smallest is not None and self.sample is not None:
            raise ValueError("use either smallest or sample, not both")
        for name in ("smallest", "sample"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        return self

    @property
    def is_full(self) -> bool:
        return not (self.conversation_ids or self.categories or self.smallest or self.sample)

    def describe(self) -> str:
        parts = []
        if self.conversation_ids:
            parts.append("conversations " + ", ".join(self.conversation_ids))
        if self.categories:
            parts.append("categories " + ", ".join(self.categories))
        if self.smallest:
            parts.append(f"{self.smallest} smallest conversations")
        if self.sample:
            parts.append(f"random {self.sample} conversations (seed {self.seed})")
        return "; ".join(parts) or "full dataset"


def select(dataset: Dataset, selection: Selection) -> Dataset:
    """Return the selected part of ``dataset``. Raises ValueError on unknown ids/categories."""
    if selection.is_full:
        return dataset
    convs = list(dataset.conversations)
    questions = list(dataset.questions)

    if selection.conversation_ids:
        known = {c.id for c in convs}
        unknown = [cid for cid in selection.conversation_ids if cid not in known]
        if unknown:
            raise ValueError(
                f"unknown conversation ids {unknown}; list them with `memjudge conversations`"
            )
        wanted = set(selection.conversation_ids)
        convs = [c for c in convs if c.id in wanted]

    if selection.categories:
        known_cats = {q.category for q in questions}
        unknown = [c for c in selection.categories if c not in known_cats]
        if unknown:
            raise ValueError(f"unknown categories {unknown}; available: {sorted(known_cats)}")
        wanted = set(selection.categories)
        questions = [q for q in questions if q.category in wanted]
        with_questions = {q.conversation_id for q in questions}
        convs = [c for c in convs if c.id in with_questions]

    if selection.smallest is not None:
        convs = sorted(convs, key=lambda c: (c.turn_count(), c.id))[: selection.smallest]
    elif selection.sample is not None and selection.sample < len(convs):
        chosen = set(random.Random(selection.seed).sample([c.id for c in convs], selection.sample))
        convs = [c for c in convs if c.id in chosen]

    kept = {c.id for c in convs}
    # Keep the dataset's original order for both conversations and questions.
    return dataset.model_copy(
        update={
            "conversations": tuple(c for c in dataset.conversations if c.id in kept),
            "questions": tuple(q for q in questions if q.conversation_id in kept),
        }
    )


def conversation_sizes(dataset: Dataset) -> list[tuple[Conversation, int, int]]:
    """(conversation, questions, approx_tokens) sorted smallest first.

    Tokens are approximated as characters / 4 so large datasets list instantly.
    """
    per_conv: dict[str, int] = {}
    for q in dataset.questions:
        per_conv[q.conversation_id] = per_conv.get(q.conversation_id, 0) + 1
    rows = [
        (
            c,
            per_conv.get(c.id, 0),
            sum(len(t.text) for s in c.sessions for t in s.turns) // 4,
        )
        for c in dataset.conversations
    ]
    return sorted(rows, key=lambda r: (r[0].turn_count(), r[0].id))
