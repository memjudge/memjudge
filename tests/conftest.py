from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from memjudge.harness import JudgeSetup
from memjudge.judge.client import Completion
from memjudge.judge.judge import load_prompt
from memjudge.result import JudgeInfo

FIXTURES = Path(__file__).parent / "fixtures"


class FakeCompleter:
    """Deterministic stand-in for the judge model.

    Standard: correct if the gold answer appears in the context. Abstention: correct if
    the distractor does not appear. Knowledge-update: stale if 'Leeds' or 'May' appear.
    """

    def __init__(self, reply: str | None = None, cost: float | None = None) -> None:
        self.reply = reply
        self.cost = cost
        self.calls = 0

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        if self.reply is not None:
            return Completion(
                text=self.reply, input_tokens=100, output_tokens=10, cost_usd=self.cost
            )
        gold = re.search(r"Gold answer: (.*)", user).group(1)  # type: ignore[union-attr]
        kind = re.search(r"Question type: (.*)", user).group(1)  # type: ignore[union-attr]
        context = user.split("<<<CONTEXT\n", 1)[1].rsplit("\nCONTEXT>>>", 1)[0]
        distractor = re.search(r"Distractor answer \(must NOT be supported\): (.*)", user)
        if kind == "ABSTENTION":
            ok = not (distractor and distractor.group(1).lower() in context.lower())
        else:
            ok = gold.lower() in context.lower()
        stale = None
        if kind == "KNOWLEDGE-UPDATE":
            stale = "leeds" in context.lower() or " may" in context.lower()
        reply = {
            "verdict": "correct" if ok else "incorrect",
            "reason": "fake",
            "stale": stale,
            "gold_suspect": False,
        }
        return Completion(
            text=json.dumps(reply),
            input_tokens=100,
            output_tokens=10,
            cost_usd=self.cost,
            served_by="fake-model",
        )


def fake_judge_setup(completer: Any | None = None) -> JudgeSetup:
    prompt = load_prompt()

    def info(served_by: list[str]) -> JudgeInfo:
        return JudgeInfo(
            route="anthropic",
            model="fake",
            upstream_provider=None,
            effort="low",
            max_output_tokens=100,
            prompt_version=prompt.version,
            prompt_hash=prompt.hash,
            served_by=served_by,
            official=False,
            season=None,
        )

    return JudgeSetup(
        client=completer or FakeCompleter(),
        prompt=prompt,
        info=info,
        price=lambda i, o: (i * 2 + o * 10) / 1_000_000,
    )


@pytest.fixture
def tiny_path() -> Path:
    return FIXTURES / "tiny.json"
