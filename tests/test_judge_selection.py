"""Tools used to choose and audit the judge: multi-judge calibration and the cross-judge check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import memjudge.judge.setup as setup
from memjudge import cli
from memjudge.crosscheck import sample_records
from memjudge.records import QuestionRecord
from tests.conftest import FakeCompleter, fake_judge_setup

runner = CliRunner()

LABELS = [
    {"id": "a", "question": "Dog?", "gold_answer": "Biscuit", "context": "The dog is Biscuit.",
     "label": "correct"},
    {"id": "b", "question": "Dog?", "gold_answer": "Biscuit", "context": "The dog is Rex.",
     "label": "incorrect", "tags": ["plausible-wrong"]},
    {"id": "c", "question": "Dog?", "gold_answer": "Biscuit", "context": "Biscuit, maybe.",
     "label": "incorrect", "tags": ["vague"]},
]  # fmt: skip


def test_calibrate_compares_several_judges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "make_judge_setup", lambda judge, env: fake_judge_setup())
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in LABELS), encoding="utf-8")
    out = tmp_path / "cal.json"
    res = runner.invoke(
        cli.app,
        ["calibrate", str(labels), "--judge", "openai:a", "--judge", "google:b", "-o", str(out)],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(out.read_text("utf-8"))
    assert [j["judge"] for j in data["judges"]] == ["openai:a", "google:b"]
    first = data["judges"][0]
    # The fake judge accepts any context containing the gold string, so it wrongly accepts "c".
    assert first["agreement"] == pytest.approx(2 / 3)
    assert first["false_accept_rate"] == pytest.approx(0.5)
    assert first["by_tag"]["vague"]["false_accept_rate"] == 1.0
    assert first["by_tag"]["plausible-wrong"]["false_accept_rate"] == 0.0
    # 3 calls x (100 in, 10 out) at the fake price of $2 / $10 per million tokens.
    assert first["input_tokens"] == 300
    assert first["cost_usd"] == pytest.approx((300 * 2 + 30 * 10) / 1e6)


def test_sample_records_is_seeded_and_skips_unjudged() -> None:
    def rec(i: int, **kw: object) -> QuestionRecord:
        base: dict[str, object] = {
            "question_id": f"q{i:02d}", "conversation_id": "c", "category": "x",
            "kind": "standard", "correct": True,
        }  # fmt: skip
        return QuestionRecord(**{**base, **kw})  # type: ignore[arg-type]

    records = [rec(i) for i in range(20)]
    records += [rec(98, correct=None, judge_error="x"), rec(99, retrieval_error="x", correct=None)]
    a, b = sample_records(records, 5, seed=1), sample_records(records, 5, seed=1)
    assert [r.question_id for r in a] == [r.question_id for r in b] and len(a) == 5
    assert sample_records(records, 5, seed=2) != a
    assert len(sample_records(records, 500, seed=0)) == 20  # unjudged questions excluded


def test_crosscheck_reports_disagreement(
    tiny_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup, "make_judge_setup", lambda judge, env: fake_judge_setup())
    runs = tmp_path / "runs"
    run = runner.invoke(
        cli.app,
        ["run", "--dataset", str(tiny_path), "--connector", "baselines/bm25",
         "--judge", "openai:x", "--out", str(runs), "--no-report"],
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    (run_dir,) = runs.iterdir()

    out = tmp_path / "same.json"
    same = runner.invoke(
        cli.app,
        ["crosscheck", str(run_dir), "--judge", "google:y", "--sample", "3", "-o", str(out)],
    )
    assert same.exit_code == 0, same.output
    check = json.loads(out.read_text("utf-8"))["checks"][0]
    assert check["sample"] == 3 and check["second_judge"] == "google:y"
    assert check["disagreement_rate"] == 0 and check["accuracy_delta"] == 0

    strict = FakeCompleter(reply='{"verdict": "incorrect", "reason": "r", "stale": null}')
    monkeypatch.setattr(setup, "make_judge_setup", lambda judge, env: fake_judge_setup(strict))
    out2 = tmp_path / "strict.json"
    diff = runner.invoke(
        cli.app, ["crosscheck", str(run_dir), "--judge", "google:y", "-o", str(out2)]
    )
    assert diff.exit_code == 0, diff.output
    check = json.loads(out2.read_text("utf-8"))["checks"][0]
    assert check["judged"] == 5 and check["accuracy_second"] == 0
    assert check["accuracy_original"] > 0
    assert check["disagreement_rate"] == check["accuracy_original"]
    assert check["accuracy_delta"] == -check["accuracy_original"]
