from __future__ import annotations

import base64
import json
import logging
import re
import socket
import sys
import tomllib
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fakes import FakeBlobs, FakeCipher, FakeLogs
from memjudge.connectors.base import ConnectorSpec

from memjudge_runner import auth, redact
from memjudge_runner.bundle import open_once, seal
from memjudge_runner.config import finish_job_name, job_name, subnet_name
from memjudge_runner.deploy import deploy_jobs, stale_releases
from memjudge_runner.envcheck import canonical_name, core_env_problems, lock_pins
from memjudge_runner.logs import LogCache, LogLine, log_filter, log_line
from memjudge_runner.options import _models, egress_profiles, is_hosted_connector
from memjudge_runner.publish import commit_file, installation_token
from memjudge_runner.retrieve import ProgressReport, local_service, wait_for_hosts


def test_bundle_opens_once_for_its_own_run() -> None:
    cipher, blobs = FakeCipher(), FakeBlobs()
    name = seal(cipher, blobs, "secrets", "kms/a", "run-1", {"keys": {"K": "v"}})
    assert len(name) >= 40  # unguessable

    stolen = blobs.objects["secrets", name]
    with pytest.raises(PermissionError):  # ciphertext bound to its run
        cipher.decrypt("kms/a", stolen, b"run-2")
    with pytest.raises(PermissionError):  # and to its KMS key
        cipher.decrypt("kms/b", stolen, b"run-1")

    assert open_once(cipher, blobs, "secrets", "kms/a", "run-1", name) == {"keys": {"K": "v"}}
    with pytest.raises(KeyError):
        open_once(cipher, blobs, "secrets", "kms/a", "run-1", name)


def test_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    assert redact.redact("a sk-1234567890 b", ["sk-1234567890", "short"]) == "a [redacted] b"
    out = StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", StringIO())
    redact.install(["key-abcdefgh"])
    print("calling with key-abcdefgh")
    assert out.getvalue() == "calling with [redacted]\n"


def test_the_run_log_shows_memjudge_lines_sdk_retries_and_warnings() -> None:
    def line(name: str, level: int = logging.INFO) -> logging.LogRecord:
        return logging.LogRecord(name, level, "", 0, "x", None, None)

    assert redact.shown(line("memjudge.harness"))
    assert redact.shown(line("memjudge_runner.retrieve"))
    assert redact.shown(line("openai._base_client"))  # "Retrying request to ..."
    assert redact.shown(line("mem0.vector_stores.qdrant", logging.WARNING))
    assert not redact.shown(line("httpx"))  # a line per HTTP request
    assert not redact.shown(line("httpx2"))
    assert not redact.shown(line("mem0.vector_stores.qdrant"))  # a line per stored item
    assert not redact.shown(line("memjudgex"))


def test_log_filter_names_every_execution() -> None:
    executions = [
        "projects/p/locations/r/jobs/retrieve-bm25-0-1-0/executions/retrieve-bm25-0-1-0-ab1",
        "projects/p/locations/r/jobs/finish-0-1-0/executions/finish-0-1-0-cd2",
    ]
    label = 'labels."run.googleapis.com/execution_name"'
    assert log_filter(executions, datetime(2026, 10, 28, tzinfo=UTC)) == (
        f'resource.type="cloud_run_job" AND ({label}="retrieve-bm25-0-1-0-ab1" OR '
        f'{label}="finish-0-1-0-cd2") AND timestamp>="2026-10-28T00:00:00+00:00"'
    )


def test_log_lines_from_cloud_logging_entries() -> None:
    def entry(job: str, **payload: Any) -> dict[str, Any]:
        resource = {"labels": {"job_name": job}}
        return {"timestamp": "2026-10-28T21:23:54.257806Z", "resource": resource, **payload}

    line = log_line(entry("finish-0-1-0", textPayload="INFO memjudge: judging\n"))
    assert line is not None and (line.job, line.text) == ("finish", "INFO memjudge: judging")
    system = log_line(entry("retrieve-x-0-1-0", jsonPayload={"message": "Container exited."}))
    assert system is not None and (system.job, system.text) == ("retrieve", "Container exited.")
    assert log_line(entry("finish-0-1-0", textPayload="  ")) is None


def test_log_cache_reads_once_per_interval_and_keeps_lines_when_a_read_fails() -> None:
    logs, clock = FakeLogs(), [0.0]
    cache = LogCache(logs, seconds=15, clock=lambda: clock[0])
    since = datetime(2026, 10, 28, tzinfo=UTC)
    line = LogLine(at=since, job="retrieve", text="INFO ingesting conv-1: 3 sessions")
    logs.lines["executions/a"] = [line]

    assert cache.lines("run", ["executions/a"], since) == ([line], True)
    assert cache.lines("run", ["executions/a"], since) == ([line], True)
    assert logs.reads == 1
    clock[0], logs.fail = 16.0, True
    assert cache.lines("run", ["executions/a"], since) == ([line], False)
    assert cache.lines("other", ["executions/b"], since) == ([], False)
    assert cache.lines("not-started", [], since) == ([], True)


def test_session_tokens() -> None:
    secret = b"x" * 32
    token = auth.issue_session(secret, "42")
    assert auth.session_user(secret, token) == "42"
    assert auth.session_user(b"y" * 32, token) is None
    state = auth.issue_state(secret, "n", "login-id")
    assert auth.session_user(secret, state) is None  # wrong purpose
    assert auth.read_state(secret, state) == ("n", "login-id")
    assert auth.user_hash(secret, "42") != auth.user_hash(b"y" * 32, "42")


def test_hosted_connectors() -> None:
    spec = ConnectorSpec(id="letta/cloud", target="Letta", author="a", settings={"base_url": None})
    assert is_hosted_connector(spec)
    assert not is_hosted_connector(spec.model_copy(update={"settings": {"base_url": "http://x"}}))
    for never in ("offline/replay", "mem0/platform", "zep/cloud"):  # vendor terms or replay
        assert not is_hosted_connector(spec.model_copy(update={"id": never}))
    # Each release has its own jobs, so a resume runs the code the run started on.
    assert job_name("mem0/oss", "0.1.0") == "retrieve-mem0-oss-0-1-0"
    assert finish_job_name("0.2.0rc1") == "finish-0-2-0rc1"


def test_form_shows_the_models_of_each_preset() -> None:
    spec = ConnectorSpec(
        id="acme/cloud",
        target="Acme",
        author="a",
        settings={"llm_model": "llm-a", "top_k": 30},
        models=["llm_model"],
        presets={"small": {"top_k": 10}, "pro": {"llm_model": "llm-pro"}},
    )
    assert _models(spec) == {"small": {"llm_model": "llm-a"}, "pro": {"llm_model": "llm-pro"}}
    assert _models(spec.model_copy(update={"presets": {}})) == {"": {"llm_model": "llm-a"}}
    assert _models(spec.model_copy(update={"models": "vendor"})) == "vendor"


def test_egress_profiles_come_from_connector_specs() -> None:
    profiles = egress_profiles()
    assert profiles[subnet_name("letta/cloud")] == ["api.letta.com"]
    assert profiles["sn-baselines-bm25"] == []  # in-process: no internet at all
    assert "sn-offline-replay" not in profiles and "sn-zep-cloud" not in profiles  # never hosted


def test_github_app_commit() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    seen: list[httpx.Request] = []

    def github(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/access_tokens"):
            claims = jwt.decode(
                request.headers["Authorization"].removeprefix("Bearer "),
                key.public_key(),
                ["RS256"],
            )
            assert claims["iss"] == "123"
            return httpx.Response(201, json={"token": "ghs_install"})
        return httpx.Response(201, json={})

    http = httpx.Client(transport=httpx.MockTransport(github))
    token = installation_token(http, "123", "456", pem)
    commit_file(http, token, "memjudge/results", "runs/r.json", "{}", "Add run r")
    put = seen[-1]
    assert put.method == "PUT" and put.url.path == "/repos/memjudge/results/contents/runs/r.json"
    assert put.headers["Authorization"] == "Bearer ghs_install"
    body = json.loads(put.content)
    assert base64.b64decode(body["content"]) == b"{}" and "sha" not in body  # create only


def test_deploy_jobs_cover_finish_and_every_hosted_connector() -> None:
    jobs = deploy_jobs("0.1.0")
    assert jobs[0] == ("finish", "finish-0-1-0", "sn-finish", "4Gi")
    by_name = {j.name: j for j in jobs}
    assert by_name["retrieve-letta-cloud-0-1-0"].subnet == "sn-letta-cloud"
    assert by_name["retrieve-mem0-oss-0-1-0"].memory == "8Gi"  # its vector store is in RAM
    assert by_name["retrieve-zep-graphiti-0-1-0"].memory == "8Gi"  # FalkorDB is in RAM
    for never in ("offline-replay", "mem0-platform", "zep-cloud"):
        assert not any(never in name for name in by_name)


RUNNER = Path(__file__).resolve().parents[1]


def test_core_lock_is_the_full_lock_without_connector_sdks() -> None:
    core = lock_pins((RUNNER / "core.lock").read_text(encoding="utf-8"))
    full = lock_pins((RUNNER / "requirements.lock").read_text(encoding="utf-8"))
    assert core and all(full.get(name) == version for name, version in core.items())
    pyproject = tomllib.loads((RUNNER.parent / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    sdks = {
        canonical_name(re.split(r"[\[=<>~! ]", req)[0])
        for extra, reqs in extras.items()
        if extra not in ("all", "dev")
        for req in reqs
    }
    assert sdks and sdks <= full.keys()  # every connector SDK is in the image
    assert not sdks & core.keys()  # but not where the API and finish run


def test_core_env_check_finds_extra_packages_wrong_versions_and_startup_hooks(
    tmp_path: Path,
) -> None:
    lock = "pydantic==2.14.1 \\\n    --hash=sha256:00\n    # via memjudge\n"
    ok = [("pydantic", "2.14.1"), ("memjudge", "0.1.0"), ("memjudge_runner", "0.1.0")]
    assert core_env_problems(lock, ok, [tmp_path]) == []
    (tmp_path / "sdk.pth").write_text("import os\n")
    bad = [*ok[1:], ("Pydantic", "2.0"), ("mem0ai", "2.2.1")]
    assert core_env_problems(lock, bad, [tmp_path]) == [
        "installed but not in the lock: mem0ai==2.2.1",
        "pydantic is 2.0, the lock pins 2.14.1",
        f"startup hook: {tmp_path / 'sdk.pth'}",
    ]
    missing = core_env_problems(lock, ok[1:], [])
    assert missing == ["in the lock but not installed: pydantic==2.14.1"]


def _job(release: str, deployed: int) -> dict[str, object]:
    return {
        "metadata": {"labels": {"memjudge-release": release, "memjudge-deployed": str(deployed)}}
    }


def test_stale_releases_are_those_replaced_long_enough_ago() -> None:
    day = 86400
    jobs = [_job("0-1-0", 0), _job("0-1-0", 0), _job("0-1-1", 10 * day), _job("0-1-2", 30 * day)]
    jobs.append({"metadata": {"labels": {}}})  # not a runner job
    # 0.1.0 was replaced on day 10, 0.1.1 on day 30, and 0.1.2 is current.
    assert stale_releases(jobs, now=31 * day, days=14) == ["0-1-0"]
    assert stale_releases(jobs, now=45 * day, days=14) == ["0-1-0", "0-1-1"]
    assert stale_releases([_job("0-1-0", 0)], now=99 * day, days=14) == []


def test_wait_for_hosts_retries_until_the_network_is_up(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] < 3:  # Cloud NAT not ready yet
            raise httpx.ConnectError("timed out")
        return httpx.Response(404)  # any answer means reachable

    monkeypatch.setattr("time.sleep", lambda s: None)
    caplog.set_level(logging.INFO, logger="memjudge_runner.retrieve")
    wait_for_hosts(httpx.Client(transport=httpx.MockTransport(handler)), ["api.getzep.com"])
    assert calls[0] == 3
    ours = [r for r in caplog.records if r.name == "memjudge_runner.retrieve"]
    assert [r.getMessage().split(" after")[0] for r in ours] == [
        "waiting for the network to reach api.getzep.com",
        "network ready",
    ]

    clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("timed out")

    with pytest.raises(RuntimeError, match=r"could not reach api\.getzep\.com"):
        wait_for_hosts(httpx.Client(transport=httpx.MockTransport(down)), ["api.getzep.com"], 30)


def test_progress_report_posts_both_counts_and_always_the_final_one() -> None:
    posted: list[dict[str, object]] = []
    report = ProgressReport(posted.append, seconds=3600)
    report.sessions(0, 10)  # the first report goes out at once
    report.questions(0, 4)  # within the interval: held back
    report.turns(5, 20)  # held back
    report.sessions(10, 10)  # final: posted with everything so far
    assert posted == [
        {"ingested": 0, "ingest_total": 10, "ingest_unit": "sessions"},
        {"ingested": 10, "ingest_total": 10, "ingest_unit": "sessions", "done": 0, "total": 4},
    ]


_LISTEN = (
    "import socket, sys, time; s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); "
    "s.listen(); time.sleep(60)"
)


def test_local_service_runs_only_while_the_block_runs() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with local_service([sys.executable, "-c", _LISTEN, str(port)], port):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()  # listening
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()  # stopped


def test_local_service_that_exits_stops_the_run() -> None:
    with (
        pytest.raises(RuntimeError, match="exited with code 3"),
        local_service([sys.executable, "-c", "raise SystemExit(3)"], 1),
    ):
        pass


def test_local_service_output_goes_to_the_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, "memjudge_runner.retrieve")
    crash = "import sys; print('crash report'); sys.stdout.flush(); raise SystemExit(3)"
    with pytest.raises(RuntimeError), local_service([sys.executable, "-c", crash], 1):
        pass
    assert "crash report" in caplog.text
    assert "exited with code 3" in caplog.text
