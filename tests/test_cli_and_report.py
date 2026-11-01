from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from memjudge import cli
from memjudge.connectors.loader import load_connector
from memjudge.datasets.loader import load_dataset
from memjudge.harness import (
    BATCH_FILE,
    RESULT_FILE,
    Runner,
    RunOptions,
    RunStoppedError,
)
from memjudge.records import BatchRecord, QuestionRecord, UsageRecord, read_jsonl
from memjudge.report.render import render_report
from memjudge.result import RunResult
from memjudge.sizes import TOP_K, questions_file
from tests.conftest import fake_judge_setup

runner = CliRunner()


@pytest.fixture
def patched_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    import memjudge.judge.setup as setup

    monkeypatch.setattr(setup, "make_judge_setup", lambda judge, env: fake_judge_setup())


def _cli_run(tiny_path: Path, out: Path) -> Result:
    return runner.invoke(
        cli.app,
        [
            "run",
            "--dataset",
            str(tiny_path),
            "--connector",
            "baselines/bm25",
            "--judge",
            "anthropic:claude-sonnet-5-5",
            "--out",
            str(out),
            "--affiliated",
            "no",
            "--no-report",
        ],
    )


def test_stopped_run_resumes_and_counts_resumes(
    tiny_path: Path, tmp_path: Path, patched_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stopped(self: Runner, *args: object, **kwargs: object) -> RunResult:
        raise RunStoppedError("Ingestion failed for conversation c2. TimeoutError: vendor")

    with monkeypatch.context() as m:
        m.setattr(Runner, "run", stopped)
        res = _cli_run(tiny_path, tmp_path / "runs")
    assert res.exit_code == 1
    assert "memjudge run --resume" in res.output
    (run_dir,) = (tmp_path / "runs").iterdir()
    for n in (1, 2):
        res = runner.invoke(cli.app, ["run", "--resume", str(run_dir), "--no-report"])
        assert res.exit_code == 0, res.output
        result = RunResult.model_validate_json((run_dir / "result.json").read_text("utf-8"))
        assert result.resumes == n
        assert json.loads((run_dir / "run.json").read_text("utf-8"))["resumes"] == n


def test_connector_max_concurrency_caps_the_run(
    tiny_path: Path, tmp_path: Path, patched_judge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    bm25 = load_connector("baselines/bm25")
    capped = replace(bm25, spec=bm25.spec.model_copy(update={"max_concurrency": 2}))
    monkeypatch.setattr(cli, "load_connector", lambda cid: capped)
    res = runner.invoke(
        cli.app,
        [
            "run",
            "--dataset",
            str(tiny_path),
            "--connector",
            "baselines/bm25",
            "--judge",
            "anthropic:claude-sonnet-5-5",
            "--out",
            str(tmp_path / "runs"),
            "--concurrency",
            "8",
            "--no-report",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "at most 2 conversations at once" in res.output
    (run_dir,) = (tmp_path / "runs").iterdir()
    result = RunResult.model_validate_json((run_dir / "result.json").read_text("utf-8"))
    assert result.config.concurrency == 2
    assert json.loads((run_dir / "run.json").read_text("utf-8"))["options"]["concurrency"] == 2


def test_validate_dataset_prints_hash(tiny_path: Path) -> None:
    res = runner.invoke(cli.app, ["validate-dataset", str(tiny_path)])
    assert res.exit_code == 0, res.output
    assert load_dataset(str(tiny_path)).hash in res.output


def test_run_report_sign_verify(tiny_path: Path, tmp_path: Path, patched_judge: None) -> None:
    res = runner.invoke(
        cli.app,
        [
            "run",
            "--dataset",
            str(tiny_path),
            "--connector",
            "baselines/bm25",
            "--judge",
            "anthropic:claude-sonnet-5-5",
            "--out",
            str(tmp_path / "runs"),
            "--affiliated",
            "no",
        ],
    )
    assert res.exit_code == 0, res.output
    (run_dir,) = (tmp_path / "runs").iterdir()
    result = RunResult.model_validate_json((run_dir / "result.json").read_text("utf-8"))
    assert result.affiliation.affiliated is False and result.mode == "local"
    assert "MemJudge report · Top 10" in (run_dir / "report-top10.html").read_text("utf-8")
    assert "MemJudge report · Top 30" in (run_dir / "report-top30.html").read_text("utf-8")
    # Each question record carries its text, so a published run reads without the dataset.
    by_id = {q.id: q for q in load_dataset(str(tiny_path)).dataset.questions}
    for rec in read_jsonl(run_dir / questions_file(10), QuestionRecord):
        q = by_id[rec.question_id]
        assert (rec.question, rec.gold_answer, rec.asked_at) == (
            q.question,
            q.gold_answer,
            q.asked_at,
        )

    keys = tmp_path / "keys"
    assert runner.invoke(cli.app, ["keygen", "--out", str(keys)]).exit_code == 0
    (key,) = keys.glob("*.key")
    prefix = "https://artifacts.example.org/runs/"
    signing = runner.invoke(
        cli.app,
        ["sign", str(run_dir / "result.json"), "--key", str(key), "--artifact-url-prefix", prefix],
    )
    assert signing.exit_code == 0, signing.output
    signed = run_dir / "result.signed.json"
    artifacts = json.loads(signed.read_text("utf-8"))["payload"]["artifacts"]
    assert artifacts and all(a["url"] == f"{prefix}{result.run_id}/{a['name']}" for a in artifacts)
    ok = runner.invoke(
        cli.app, ["verify", str(signed), "--keys", str(keys), "--artifacts", str(run_dir)]
    )
    assert ok.exit_code == 0, ok.output

    (run_dir / questions_file(10)).write_text("tampered", encoding="utf-8")
    bad = runner.invoke(
        cli.app, ["verify", str(signed), "--keys", str(keys), "--artifacts", str(run_dir)]
    )
    assert bad.exit_code == 1

    envelope = json.loads(signed.read_text("utf-8"))
    envelope["payload"]["metrics"]["top30"]["overall"]["accuracy"] = 0.123
    signed.write_text(json.dumps(envelope), encoding="utf-8")
    assert runner.invoke(cli.app, ["verify", str(signed), "--keys", str(keys)]).exit_code == 1

    empty = tmp_path / "no-keys"
    empty.mkdir()
    none = runner.invoke(cli.app, ["verify", str(signed), "--keys", str(empty)])
    assert none.exit_code == 1 and "no trusted .pub keys" in none.output


def test_rejudge_keeps_contexts(tiny_path: Path, tmp_path: Path, patched_judge: None) -> None:
    out = tmp_path / "runs"
    runner.invoke(
        cli.app,
        [
            "run",
            "--dataset",
            str(tiny_path),
            "--connector",
            "baselines/bm25",
            "--judge",
            "openai:x",
            "--out",
            str(out),
            "--no-report",
            "--keep-data",
        ],
    )
    (run_dir,) = out.iterdir()
    # Simulate a run made before records stored the question text.
    for top_k in TOP_K:
        questions = run_dir / questions_file(top_k)
        old_style = [
            r.model_copy(update={"question": "", "gold_answer": "", "asked_at": None})
            for r in read_jsonl(questions, QuestionRecord)
        ]
        questions.write_text("".join(r.model_dump_json() + "\n" for r in old_style), "utf-8")
    rejudged = tmp_path / "rejudged"
    res = runner.invoke(
        cli.app, ["rejudge", str(run_dir), "--judge", "openai:y", "--out", str(rejudged)]
    )
    assert res.exit_code == 0, res.output
    by_id = {q.id: q for q in load_dataset(str(tiny_path)).dataset.questions}
    for top_k in TOP_K:
        questions = run_dir / questions_file(top_k)
        before = {r.question_id: r.context for r in read_jsonl(questions, QuestionRecord)}
        after_records = read_jsonl(rejudged / questions_file(top_k), QuestionRecord)
        assert before == {r.question_id: r.context for r in after_records}
        assert all(r.question == by_id[r.question_id].question for r in after_records)
    # The kept data belongs to the original run: a rejudged run can neither delete nor reuse it.
    meta = json.loads((rejudged / cli.RUN_META).read_text("utf-8"))
    assert (
        meta["rejudged_from"] == json.loads((run_dir / cli.RUN_META).read_text("utf-8"))["run_id"]
    )
    assert meta["keep_data"] is False and meta["reuse_ingestion_from"] is None


def test_rejudge_keeps_batch_ingestion(
    tiny_path: Path, tmp_path: Path, patched_judge: None
) -> None:
    out = tmp_path / "runs"
    args = ["--dataset", str(tiny_path), "--connector", "baselines/bm25", "--judge", "openai:x"]
    runner.invoke(cli.app, ["run", *args, "--out", str(out), "--no-report"])
    (run_dir,) = out.iterdir()
    # Stand in for a batch run (ADR 0010): its ingestion time is in ingest_batch.json.
    batch = BatchRecord(seconds=42.0, usage=UsageRecord(api_calls=3))
    (run_dir / BATCH_FILE).write_text(batch.model_dump_json(), encoding="utf-8")
    rejudged = tmp_path / "rejudged"
    res = runner.invoke(
        cli.app, ["rejudge", str(run_dir), "--judge", "openai:y", "--out", str(rejudged)]
    )
    assert res.exit_code == 0, res.output
    result = RunResult.model_validate_json((rejudged / RESULT_FILE).read_text("utf-8"))
    for m in result.metrics.values():
        assert m.ingestion.mode == "batch" and m.ingestion.total_s >= 42.0
    nothing = runner.invoke(cli.app, ["cleanup", str(rejudged)])
    assert nothing.exit_code == 0 and "Nothing to delete" in nothing.output


def test_run_json_does_not_leak_credentials(
    tiny_path: Path, tmp_path: Path, patched_judge: None
) -> None:
    res = runner.invoke(
        cli.app,
        [
            "run",
            "--dataset",
            str(tiny_path),
            "--connector",
            "baselines/bm25",
            "--judge",
            "openai:x",
            "--out",
            str(tmp_path),
            "--no-report",
            "--setting",
            "base_url=http://u:pw@localhost:9/x/?k=1",
        ],
    )
    assert res.exit_code == 0, res.output
    (run_dir,) = tmp_path.iterdir()
    text = (run_dir / cli.RUN_META).read_text("utf-8")
    assert "pw" not in text
    assert json.loads(text)["settings"]["base_url"] == "http://localhost:9/x"


def test_missing_key_is_a_clean_error(
    tiny_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("RECALLD_API_KEY", raising=False)
    argv = ["memjudge", "run", "--dataset", str(tiny_path), "--connector", "recalld/cloud"]
    monkeypatch.setattr(sys, "argv", [*argv, "--judge", "openai:x", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "RECALLD_API_KEY" in err and "Traceback" not in err


def test_report_escapes_untrusted_context(tiny_path: Path, tmp_path: Path) -> None:
    loaded = load_dataset(str(tiny_path))
    conn = load_connector("baselines/bm25")
    result = Runner(
        loaded, conn, conn.instantiate({}), fake_judge_setup(), tmp_path, RunOptions()
    ).run()
    records = read_jsonl(tmp_path / questions_file(30), QuestionRecord)
    records[0] = records[0].model_copy(update={"context": "</script><script>alert(1)</script>"})
    html = render_report([result, result], records, loaded, 30)
    assert "<script>alert(1)" not in html
    assert "Compare runs" in html  # two runs -> compare view
    assert "batch, wall-clock" not in html
    assert "all conversations added up" in html
    top30 = result.metrics["top30"]
    ingestion = top30.ingestion.model_copy(update={"mode": "batch"})
    batch = result.model_copy(
        update={
            "metrics": {
                **result.metrics,
                "top30": top30.model_copy(update={"ingestion": ingestion}),
            }
        }
    )
    assert "batch, wall-clock" in render_report([batch, result], records, loaded, 30)


def test_estimate_runs_offline(tiny_path: Path) -> None:
    res = runner.invoke(cli.app, ["estimate", "--dataset", str(tiny_path)])
    assert res.exit_code == 0, res.output
    assert "Judge cost upper bound" in res.output
    # tiny.json: 3 sessions, 6 turns, 5 questions
    assert "3 sessions, 6 write calls" in res.output and "10 searches" in res.output  # 2 sizes


def test_published_result_schema_is_current() -> None:
    schema_file = Path(__file__).parents[1] / "schemas" / "result.v1.json"
    assert json.loads(schema_file.read_text("utf-8")) == RunResult.model_json_schema()


def test_estimate_json(tiny_path: Path) -> None:
    res = runner.invoke(cli.app, ["estimate", "--dataset", str(tiny_path), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["sessions"] == 3 and data["turns"] == 6 and data["questions"] == 5
    assert data["judge_input_tokens"] > 5 * 8192
    assert 0 < data["judge_cost_usd_typical"] < data["judge_cost_usd_upper_bound"]
