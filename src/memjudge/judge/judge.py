"""Build the judge prompt, call the judge and parse its verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Protocol

from memjudge.canonical import sha256_hex
from memjudge.judge.client import Completion, JudgeError
from memjudge.schema import Question, QuestionKind

PROMPT_VERSION = "v3"

_KIND_LABEL = {
    QuestionKind.STANDARD: "STANDARD",
    QuestionKind.ABSTENTION: "ABSTENTION",
    QuestionKind.KNOWLEDGE_UPDATE: "KNOWLEDGE-UPDATE",
}


class Completer(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user_template: str
    hash: str


@cache
def load_prompt(version: str = PROMPT_VERSION) -> Prompt:
    text = (
        resources.files("memjudge.judge")
        .joinpath("prompts", f"{version}.md")
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
    )
    body = text.split("[system]\n", 1)[1]
    system, user = body.split("[user]\n", 1)
    return Prompt(
        version=version,
        system=system.strip(),
        user_template=user.strip(),
        hash="sha256:" + sha256_hex(text.encode()),
    )


def render_user(prompt: Prompt, question: Question, context: str) -> str:
    distractor = (
        f"\nDistractor answer (must NOT be supported): {question.distractor_answer}"
        if question.distractor_answer
        else ""
    )
    values = {
        "{kind}": _KIND_LABEL[question.kind],
        "{asked_at}": question.asked_at.isoformat(sep=" ") if question.asked_at else "unknown",
        "{question}": question.question,
        "{gold_answer}": question.gold_answer,
        "{distractor}": distractor,
        "{context}": context if context.strip() else "(empty)",
    }
    out = prompt.user_template
    # Context goes last so placeholder-looking text inside it is never substituted.
    for key in ["{kind}", "{asked_at}", "{question}", "{gold_answer}", "{distractor}"]:
        out = out.replace(key, values[key])
    return out.replace("{context}", values["{context}"])


@dataclass
class Verdict:
    correct: bool | None
    """None when the judge failed; counted as an error, not as incorrect."""
    reason: str
    stale: bool | None = None
    gold_suspect: bool = False
    raw_text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    served_by: str | None = None
    attempts: int = 0
    error: str | None = None


def parse_verdict(text: str) -> dict[str, Any]:
    """The last valid verdict object in the reply: a judge that corrects itself writes its
    final verdict after the first one."""
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    error: ValueError | None = None
    i = text.find("{")
    while i != -1:
        try:
            data, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(data, dict):
            try:
                found = _checked(data)
            except ValueError as exc:
                error = exc
        i = text.find("{", end)
    if found is not None:
        return found
    raise error or ValueError("no JSON object in judge output")


def _checked(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("verdict") not in ("correct", "incorrect"):
        raise ValueError(f"bad verdict {data.get('verdict')!r}")
    if not isinstance(data.get("reason", ""), str):
        raise ValueError("reason must be a string")
    if data.get("stale") not in (True, False, None):
        raise ValueError("stale must be true/false/null")
    return data


def judge(
    client: Completer, prompt: Prompt, question: Question, context: str, max_parse_attempts: int = 2
) -> Verdict:
    user = render_user(prompt, question, context)
    attempts = 0
    last_error = ""
    totals: dict[str, float | None] = {"in": None, "out": None, "cost": None}
    last: Completion | None = None
    for _ in range(max_parse_attempts):
        try:
            completion = client.complete(prompt.system, user)
        except JudgeError as exc:
            # An earlier unparseable reply was still billed: keep its tokens.
            return Verdict(
                correct=None,
                reason="",
                raw_text=last.text if last else "",
                input_tokens=_int(totals["in"]),
                output_tokens=_int(totals["out"]),
                cost_usd=totals["cost"],
                error=str(exc),
                attempts=attempts + 1,
            )
        last = completion
        attempts += completion.attempts
        for key, value in (
            ("in", completion.input_tokens),
            ("out", completion.output_tokens),
            ("cost", completion.cost_usd),
        ):
            if value is not None:
                totals[key] = (totals[key] or 0) + value
        try:
            data = parse_verdict(completion.text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"unparseable judge output: {exc}"
            continue
        stale = data.get("stale")
        return Verdict(
            correct=data["verdict"] == "correct",
            reason=str(data.get("reason", "")),
            stale=stale if question.kind is QuestionKind.KNOWLEDGE_UPDATE else None,
            gold_suspect=bool(data.get("gold_suspect", False)),
            raw_text=completion.text,
            input_tokens=_int(totals["in"]),
            output_tokens=_int(totals["out"]),
            cost_usd=totals["cost"],
            served_by=completion.served_by,
            attempts=attempts,
        )
    return Verdict(
        correct=None,
        reason="",
        raw_text=last.text if last else "",
        input_tokens=_int(totals["in"]),
        output_tokens=_int(totals["out"]),
        cost_usd=totals["cost"],
        attempts=attempts,
        error=last_error,
    )


def _int(value: float | None) -> int | None:
    return None if value is None else int(value)
