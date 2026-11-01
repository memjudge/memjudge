"""Render one or more runs into a single self-contained HTML report."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from jinja2 import Environment, select_autoescape

from memjudge import __version__
from memjudge.datasets.loader import LoadedDataset
from memjudge.metrics import Metrics, Rate
from memjudge.records import QuestionRecord
from memjudge.result import RunResult
from memjudge.sizes import size_key

ECHARTS_SRI = "sha384-Mx5lkUEQPM1pOJCwFtUICyX45KNojXbkWdYhkKUKsbv391mavbfoAmONbzkgYPzR"
CONTEXT_CHARS = 1500
NOT_REPORTED = "not reported"


def _pct(v: float | None) -> str:
    return NOT_REPORTED if v is None else f"{v * 100:.1f}%"


def _num(v: float | None) -> str:
    return NOT_REPORTED if v is None else f"{v:,.0f}"


def _fixed(v: float | None, digits: int = 2) -> str:
    return NOT_REPORTED if v is None else f"{v:.{digits}f}"


def _secs(v: float | None, digits: int = 2) -> str:
    if v is None:
        return NOT_REPORTED
    if 0 < v < 0.001:
        return f"{v * 1e6:.0f} µs"
    return f"{v * 1000:.0f} ms" if v < 0.1 else f"{v:.{digits}f} s"


def _rate(r: Rate) -> str:
    return "n/a" if r.value is None else f"{r.value * 100:.1f}%"


def run_labels(runs: list[RunResult]) -> list[str]:
    """Connector id, disambiguated by budget (or run id) when the same connector repeats."""
    labels = [r.connector.id for r in runs]
    for i, r in enumerate(runs):
        twins = [o for o in runs if o.connector.id == r.connector.id]
        if len(twins) > 1:
            budgets = {o.config.budget_tokens for o in twins}
            suffix = (
                f"@{r.config.budget_tokens}" if len(budgets) == len(twins) else f"#{r.run_id[-6:]}"
            )
            labels[i] = r.connector.id + suffix
    return labels


def render_report(
    runs: list[RunResult],
    questions: list[QuestionRecord],
    dataset: LoadedDataset | None,
    top_k: int,
    signature_status: str | None = None,
) -> str:
    """The runs at one result size (ADR 0018). ``questions`` are the first run's records of
    that size; ``dataset`` (optional) supplies questions text."""
    primary = runs[0]
    key = size_key(top_k)
    by_id = {q.id: q for q in dataset.dataset.questions} if dataset else {}
    q_rows: list[dict[str, Any]] = []
    for r in sorted(questions, key=lambda r: r.question_id):
        q = by_id.get(r.question_id)
        q_rows.append(
            {
                "id": r.question_id,
                "question": q.question if q else r.question_id,
                "gold": q.gold_answer if q else "",
                "category": r.category,
                "correct": r.correct,
                "error": r.retrieval_error or r.judge_error,
                "reason": r.reason,
                "stale": r.stale,
                "gold_suspect": r.gold_suspect,
                "tokens": r.context_tokens,
                "context": r.context[:CONTEXT_CHARS]
                + ("…" if len(r.context) > CONTEXT_CHARS else ""),
            }
        )
    labels = run_labels(runs)
    data = {
        "runs": [
            {"label": label, "metrics": r.metrics[key].model_dump(mode="json")}
            for label, r in zip(labels, runs, strict=True)
        ],
        "latencies": [r.retrieval_seconds for r in questions if r.retrieval_seconds is not None],
        "questions": q_rows,
    }
    # Escape every "<" so untrusted text (contexts, reasons) cannot break out of the script.
    data_json = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")

    env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
    env.globals.update(pct=_pct, num=_num, fixed=_fixed, secs=_secs, rate=_rate)
    package = resources.files("memjudge.report")
    template = env.from_string(package.joinpath("template.html.j2").read_text(encoding="utf-8"))
    return template.render(
        runs=[_RunView(label, r.metrics[key]) for label, r in zip(labels, runs, strict=True)],
        primary=primary,
        primary_metrics=primary.metrics[key],
        top_k=top_k,
        primary_label=labels[0],
        signature_status=signature_status,
        charts_js=package.joinpath("charts.js").read_text(encoding="utf-8"),
        data_json=data_json,
        echarts_sri=ECHARTS_SRI,
        context_chars=CONTEXT_CHARS,
        version=__version__,
    )


class _RunView:
    """Template view of a run: its display label and its metrics at the report's size."""

    def __init__(self, label: str, metrics: Metrics) -> None:
        self.label = label
        self.metrics = metrics
