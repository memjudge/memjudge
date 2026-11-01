"""Cross-judge check: re-judge a sample of a run with a second judge (ADR 0008).

The pinned judge may share a model family with one of the tested systems. Re-judging
a seeded sample with a judge from a different vendor shows, per connector, how much the
score depends on the judge. A system whose accuracy moves much more than the others'
under the second judge is being favoured (or penalised) by one of them.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ConfigDict

from memjudge.judge.judge import Completer, Prompt, judge
from memjudge.records import QuestionRecord
from memjudge.schema import Question


class CrossCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    connector: str
    original_judge: str
    second_judge: str
    sample: int
    judged: int
    """Sampled questions the second judge returned a verdict for."""
    disagreement_rate: float | None
    accuracy_original: float | None
    accuracy_second: float | None
    accuracy_delta: float | None
    """second - original, on the judged sample."""
    disagreements: list[str]


def sample_records(records: Sequence[QuestionRecord], n: int, seed: int) -> list[QuestionRecord]:
    """Seeded sample of questions that have an original verdict.

    The sample is drawn from sorted question ids, so the same seed is reproducible.
    """
    eligible = sorted(
        (r for r in records if r.correct is not None and r.retrieval_error is None),
        key=lambda r: r.question_id,
    )
    if n >= len(eligible):
        return eligible
    chosen = set(random.Random(seed).sample([r.question_id for r in eligible], n))
    return [r for r in eligible if r.question_id in chosen]


def cross_check(
    sample: Sequence[QuestionRecord],
    questions: Mapping[str, Question],
    client: Completer,
    prompt: Prompt,
    *,
    run_id: str,
    connector: str,
    original_judge: str,
    second_judge: str,
    concurrency: int = 8,
) -> CrossCheck:
    def one(rec: QuestionRecord) -> bool | None:
        return judge(client, prompt, questions[rec.question_id], rec.context).correct

    with ThreadPoolExecutor(concurrency) as pool:
        verdicts = list(pool.map(one, sample))
    pairs = [(r, v) for r, v in zip(sample, verdicts, strict=True) if v is not None]
    n = len(pairs)
    original = sum(r.correct is True for r, _ in pairs) / n if n else None
    second = sum(v for _, v in pairs) / n if n else None
    differing = sorted(r.question_id for r, v in pairs if r.correct != v)
    return CrossCheck(
        run_id=run_id,
        connector=connector,
        original_judge=original_judge,
        second_judge=second_judge,
        sample=len(sample),
        judged=n,
        disagreement_rate=len(differing) / n if n else None,
        accuracy_original=original,
        accuracy_second=second,
        accuracy_delta=second - original if original is not None and second is not None else None,
        disagreements=differing,
    )
