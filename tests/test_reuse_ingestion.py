"""Keep data, rerun only retrieval (--reuse-ingestion), then clean up — end to end via the CLI.

Uses the Recalld connector against a mocked API, because Recalld needs saved state (agent
and thread ids) to find its data again, which is the hard case.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

import memjudge.judge.setup as setup
from memjudge import cli
from memjudge.records import ConversationRecord, read_jsonl
from memjudge.result import RunResult
from tests.conftest import fake_judge_setup

BASE = "https://eu.recalld.ai"
USED = {
    "x-recalld-models": "decompose=openai-gpt-6-luna, evaluate=openai-gpt-6-luna, "
    "select_facts=openai-gpt-6-luna, embedding=gemini-embedding-002-vertex"
}
runner = CliRunner()


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(setup, "make_judge_setup", lambda judge, env: fake_judge_setup())
    path = tmp_path / ".env"
    path.write_text("RECALLD_API_KEY=rcd_eu_test\n", encoding="utf-8")
    return path


def _mock_recalld() -> dict[str, respx.Route]:
    ids = itertools.count(1)

    def create_agent(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": f"a{next(ids)}"})

    return {
        "create": respx.post(f"{BASE}/v1/agents").mock(side_effect=create_agent),
        "thread": respx.post(url__regex=rf"{BASE}/v1/agents/a\d+/threads").mock(
            return_value=httpx.Response(201, json={"id": "t1"})
        ),
        "ingest": respx.post(url__regex=rf"{BASE}/v1/agents/a\d+/memory$").mock(
            return_value=httpx.Response(200, json={}, headers=USED)
        ),
        "recall": respx.post(url__regex=rf"{BASE}/v1/agents/a\d+/memory/recall").mock(
            return_value=httpx.Response(
                200, json={"facts": [{"text": "Ana's dog is Biscuit"}]}, headers=USED
            )
        ),
        "delete": respx.delete(url__regex=rf"{BASE}/v1/agents/a\d+").mock(
            return_value=httpx.Response(200, json={})
        ),
    }


def _run(tiny_path: Path, out: Path, env_file: Path, *extra: str) -> Path:
    res = runner.invoke(
        cli.app,
        [
            "run", "--dataset", str(tiny_path), "--connector", "recalld/cloud",
            "--judge", "openai:x", "--out", str(out), "--env-file", str(env_file),
            "--no-report", *extra,
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    (run_dir,) = [d for d in out.iterdir() if d.is_dir()]
    return run_dir


@respx.mock
def test_keep_reuse_cleanup(tiny_path: Path, tmp_path: Path, env_file: Path) -> None:
    api = _mock_recalld()

    first = _run(tiny_path, tmp_path / "first", env_file, "--keep-data")
    assert api["create"].call_count == 2 and api["ingest"].called
    assert not api["delete"].called  # --keep-data: nothing deleted at the end
    kept = read_jsonl(first / "conversations.jsonl", ConversationRecord)
    assert {r.state["agent_id"] for r in kept if r.state} == {"a1", "a2"}

    calls_before = (api["create"].call_count, api["ingest"].call_count)
    second = _run(tiny_path, tmp_path / "second", env_file, "--reuse-ingestion", str(first))
    assert (api["create"].call_count, api["ingest"].call_count) == calls_before  # no ingestion
    assert not api["delete"].called  # reused data belongs to the first run
    # Retrieval went to the agents the first run created.
    recalled = {c.request.url.path.split("/")[3] for c in api["recall"].calls}
    assert recalled == {"a1", "a2"}

    first_result = RunResult.model_validate_json((first / "result.json").read_text("utf-8"))
    result = RunResult.model_validate_json((second / "result.json").read_text("utf-8"))
    assert result.reused_ingestion is not None
    assert result.reused_ingestion.run_id == first_result.run_id
    first_ingestion = first_result.metrics["top30"].ingestion
    assert all(m.ingestion.total_s == first_ingestion.total_s for m in result.metrics.values())
    assert all(m.overall.questions == 5 for m in result.metrics.values())

    # The second run has nothing of its own to delete; the first run's cleanup deletes both.
    nothing = runner.invoke(cli.app, ["cleanup", str(second), "--env-file", str(env_file)])
    assert nothing.exit_code == 0 and "Nothing to delete" in nothing.output
    done = runner.invoke(cli.app, ["cleanup", str(first), "--env-file", str(env_file)])
    assert done.exit_code == 0, done.output
    assert {c.request.url.path for c in api["delete"].calls} == {"/v1/agents/a1", "/v1/agents/a2"}
    assert json.loads((first / "run.json").read_text("utf-8"))["data_deleted"] is True

    # Reusing deleted data is refused.
    again = runner.invoke(
        cli.app,
        [
            "run", "--dataset", str(tiny_path), "--connector", "recalld/cloud",
            "--judge", "openai:x", "--out", str(tmp_path / "third"),
            "--env-file", str(env_file), "--reuse-ingestion", str(first),
        ],
    )  # fmt: skip
    assert again.exit_code != 0 and "deleted" in again.output


@respx.mock
def test_reuse_is_refused_when_data_was_not_kept(
    tiny_path: Path, tmp_path: Path, env_file: Path
) -> None:
    api = _mock_recalld()
    first = _run(tiny_path, tmp_path / "first", env_file)  # default: data deleted at the end
    assert api["delete"].call_count == 2
    res = runner.invoke(
        cli.app,
        [
            "run", "--dataset", str(tiny_path), "--connector", "recalld/cloud",
            "--judge", "openai:x", "--out", str(tmp_path / "second"),
            "--env-file", str(env_file), "--reuse-ingestion", str(first),
        ],
    )  # fmt: skip
    # Typer colours option names on GitHub Actions, splitting "--keep-data" with codes.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert res.exit_code != 0 and "--keep-data" in plain


def test_reuse_is_refused_for_in_process_connectors(
    tiny_path: Path, tmp_path: Path, env_file: Path
) -> None:
    res = runner.invoke(
        cli.app,
        [
            "run", "--dataset", str(tiny_path), "--connector", "baselines/bm25",
            "--judge", "openai:x", "--out", str(tmp_path), "--reuse-ingestion", str(tmp_path),
        ],
    )  # fmt: skip
    assert res.exit_code != 0 and "in-process" in res.output
