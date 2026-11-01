"""Judge calibration: agreement with hand labels, and how often wrong context is accepted.

Labels file (JSON lines), one judged example per line::

    {"id": "c1", "question": "...", "gold_answer": "...", "kind": "standard",
     "context": "...", "label": "incorrect", "tags": ["plausible-wrong"]}

``kind`` is standard | abstention | knowledge_update; ``label`` is correct | incorrect.
Tag adversarial examples (on-topic but wrong value, vague, contradictory) so their
false-accept rate is reported separately: a lenient judge is the most common way
memory benchmarks overstate accuracy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from memjudge.judge.judge import Completer, Prompt, Verdict, judge
from memjudge.schema import Question, QuestionKind


class LabelledExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    gold_answer: str
    kind: QuestionKind = QuestionKind.STANDARD
    distractor_answer: str | None = None
    context: str
    label: bool
    tags: list[str] = []

    @classmethod
    def from_line(cls, line: str) -> LabelledExample:
        data = json.loads(line)
        label = data.pop("label")
        if label not in ("correct", "incorrect"):
            raise ValueError(f"label must be correct|incorrect, got {label!r}")
        return cls(label=label == "correct", **data)


class CalibrationReport(BaseModel):
    examples: int
    judged: int
    errors: int
    agreement: float | None
    false_accept_rate: float | None
    """Judge said correct, humans said incorrect."""
    false_reject_rate: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    """Judge cost for this set; None when neither the API nor the price table knows it."""
    by_tag: dict[str, dict[str, float | int | None]]
    disagreements: list[dict[str, str]]


def load_labels(path: Path) -> list[LabelledExample]:
    lines = path.read_text(encoding="utf-8").split("\n")  # not splitlines(): see read_jsonl
    return [LabelledExample.from_line(line) for line in lines if line.strip()]


def calibrate(
    examples: list[LabelledExample],
    client: Completer,
    prompt: Prompt,
    concurrency: int = 8,
    price: Callable[[int, int], float | None] | None = None,
) -> CalibrationReport:
    """``price`` maps (input_tokens, output_tokens) to USD when the API reports no cost."""

    def one(ex: LabelledExample) -> Verdict:
        q = Question(
            id=ex.id,
            conversation_id="calibration",
            question=ex.question,
            gold_answer=ex.gold_answer,
            category="calibration",
            kind=ex.kind,
            distractor_answer=ex.distractor_answer,
        )
        return judge(client, prompt, q, ex.context)

    with ThreadPoolExecutor(concurrency) as pool:
        full = list(pool.map(one, examples))
    verdicts = [v.correct for v in full]
    input_tokens = sum(v.input_tokens or 0 for v in full)
    output_tokens = sum(v.output_tokens or 0 for v in full)
    reported = [v.cost_usd for v in full]
    cost: float | None
    if full and all(c is not None for c in reported):
        cost = sum(c for c in reported if c is not None)
    else:
        cost = price(input_tokens, output_tokens) if price else None

    pairs = [(ex, v) for ex, v in zip(examples, verdicts, strict=True) if v is not None]

    def stats(items: list[tuple[LabelledExample, bool | None]]) -> dict[str, float | int | None]:
        judged = [(ex, v) for ex, v in items if v is not None]
        neg = [v for ex, v in judged if not ex.label]
        pos = [v for ex, v in judged if ex.label]
        return {
            "n": len(judged),
            "agreement": sum(ex.label == v for ex, v in judged) / len(judged) if judged else None,
            "false_accept_rate": sum(neg) / len(neg) if neg else None,
            "false_reject_rate": sum(not v for v in pos) / len(pos) if pos else None,
        }

    overall = stats(list(zip(examples, verdicts, strict=True)))
    tagged: dict[str, list[tuple[LabelledExample, bool | None]]] = defaultdict(list)
    for ex, v in zip(examples, verdicts, strict=True):
        for tag in ex.tags or ["untagged"]:
            tagged[tag].append((ex, v))
    return CalibrationReport(
        examples=len(examples),
        judged=len(pairs),
        errors=len(examples) - len(pairs),
        agreement=_f(overall["agreement"]),
        false_accept_rate=_f(overall["false_accept_rate"]),
        false_reject_rate=_f(overall["false_reject_rate"]),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        by_tag={tag: stats(items) for tag, items in sorted(tagged.items())},
        disagreements=[
            {
                "id": ex.id,
                "label": "correct" if ex.label else "incorrect",
                "judge": "correct" if v else "incorrect",
            }
            for ex, v in pairs
            if ex.label != v
        ],
    )


def _f(value: float | int | None) -> float | None:
    return None if value is None else float(value)
