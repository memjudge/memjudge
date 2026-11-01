"""Retrieve job: ingest the dataset into the memory system and retrieve for each question.

Runs connector code, so it is untrusted: it holds only the connector keys (its bundle is
sealed with a KMS key only this job can use), its internet egress is limited to the
connector's declared domains and Google APIs (what that cannot stop: ADR 0009, Egress),
and it never sees the judge key. It reports to the API with the run
token from its bundle and uploads its records through signed URLs.

It saves its records every ``CHECKPOINT_SECONDS`` and when it stops. A resumed run's job
starts from the saved records, so completed conversations are not run again. Any failure
stops the run (resumable) rather than failing it.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import httpx
from memjudge.connectors.loader import load_connector, load_spec
from memjudge.harness import Runner, RunOptions, RunStoppedError

from memjudge_runner.bundle import Blobs, Cipher, open_once
from memjudge_runner.config import CHECKPOINT_SECONDS, CONCURRENCY, LOCAL_SERVICES, Settings
from memjudge_runner.datasets import load_hosted
from memjudge_runner.redact import install, redact

log = logging.getLogger(__name__)

PROGRESS_SECONDS = 10.0


def short_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:300]}"


def throttled(report: Callable[[int, int], None], seconds: float) -> Callable[[int, int], None]:
    """Report at most every ``seconds``, and always the final count."""
    last = [0.0]

    def call(done: int, total: int) -> None:
        t = time.monotonic()
        if done >= total or t - last[0] >= seconds:
            last[0] = t
            try:
                report(done, total)
            except httpx.HTTPError as exc:  # progress is best effort
                log.warning("progress report failed: %s", exc.__class__.__name__)

    return call


def wait_for_hosts(http: httpx.Client, hosts: Iterable[str], seconds: float = 180.0) -> None:
    """Wait until each host answers over HTTPS (any status counts). Cloud NAT needs up to
    about a minute after a job starts before internet traffic flows."""
    start = time.monotonic()
    deadline = start + seconds
    hosts = list(hosts)
    if hosts:
        log.info("waiting for the network to reach %s", ", ".join(hosts))
    for host in hosts:
        while True:
            try:
                http.head(f"https://{host}/", timeout=10.0)
                break
            except httpx.TransportError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"The runner could not reach {host}. Try resuming the run later."
                    ) from None
                time.sleep(5.0)
    if hosts:
        log.info("network ready after %.0f s", time.monotonic() - start)


def _relay(proc: subprocess.Popen[str], name: str) -> None:
    """Log a local service's output, and its exit code if it dies, so a crash shows why."""
    assert proc.stdout is not None
    for line in proc.stdout:
        log.info("%s: %s", name, line.rstrip())
    if code := proc.wait():
        log.warning("%s exited with code %d", name, code)


@contextmanager
def local_service(command: list[str], port: int, seconds: float = 60.0) -> Iterator[None]:
    """Run ``command`` while the block runs, after waiting until it listens on ``port``."""
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    relay = threading.Thread(target=_relay, args=(proc, Path(command[0]).name), daemon=True)
    relay.start()
    try:
        deadline = time.monotonic() + seconds
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{Path(command[0]).name} exited with code {proc.returncode}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{Path(command[0]).name} did not start") from None
                time.sleep(0.5)
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        relay.join(timeout=10)  # its last lines, e.g. a crash report


class ProgressReport:
    """Retrieve's question and ingestion counts, posted together at most every ``seconds``
    and on each final count. Conversations report from several threads."""

    def __init__(self, post: Callable[[dict[str, Any]], None], seconds: float) -> None:
        self._post, self._seconds = post, seconds
        self._fields: dict[str, Any] = {}
        self._last: float | None = None
        self._lock = threading.Lock()

    def questions(self, done: int, total: int) -> None:
        self._set(done, total, {"done": done, "total": total})

    def sessions(self, done: int, total: int) -> None:
        self._set(done, total, {"ingested": done, "ingest_total": total, "ingest_unit": "sessions"})

    def turns(self, done: int, total: int) -> None:
        self._set(done, total, {"ingested": done, "ingest_total": total, "ingest_unit": "turns"})

    def _set(self, done: int, total: int, fields: dict[str, Any]) -> None:
        with self._lock:
            self._fields.update(fields)
            t = time.monotonic()
            if done < total and self._last is not None and t - self._last < self._seconds:
                return
            self._last = t
            body = dict(self._fields)
        try:
            self._post(body)
        except httpx.HTTPError as exc:  # progress is best effort
            log.warning("progress report failed: %s", exc.__class__.__name__)


class ApiClient:
    """The API's internal endpoints, authorised by the run token."""

    def __init__(self, base_url: str, run_id: str, token: str, http: httpx.Client) -> None:
        self._base = f"{base_url}/internal/runs/{run_id}"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._http = http

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        for attempt in range(4):
            try:
                resp = self._http.post(f"{self._base}/{path}", json=body, headers=self._headers)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError:
                if attempt == 3:
                    raise
            time.sleep(2**attempt)
        resp.raise_for_status()
        return resp.json()

    def progress(self, body: dict[str, Any]) -> None:
        self._post("progress", body)

    def upload_urls(self) -> dict[str, str]:
        urls: dict[str, str] = self._post("uploads")
        return urls

    def download_urls(self) -> dict[str, str]:
        urls: dict[str, str] = self._post("downloads")
        return urls

    def retrieved(self, endpoint: str | None) -> None:
        self._post("retrieved", {"endpoint": endpoint})

    def stopped(self, message: str) -> None:
        self._post("stopped", {"message": message})


class Checkpoint:
    """The run folder's records, saved to and restored from staging."""

    def __init__(
        self, api: ApiClient, http: httpx.Client, folder: Path, secrets: Iterable[str]
    ) -> None:
        self._api, self._http, self._folder = api, http, folder
        self._secrets = list(secrets)
        self._lock = threading.Lock()

    def restore(self) -> None:
        """Download what earlier attempts saved; nothing saved means start fresh."""
        for name, url in self._api.download_urls().items():
            resp = self._http.get(url)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            (self._folder / name).write_bytes(resp.content)

    def save(self) -> None:
        with self._lock:
            for name, url in self._api.upload_urls().items():
                path = self._folder / name
                if not path.exists():
                    continue
                data = path.read_bytes()
                data = data[: data.rfind(b"\n") + 1]  # whole records only: a writer may be mid-line
                # Finish redacts with the latest keys; this covers keys changed on resume.
                clean = redact(data.decode("utf-8"), self._secrets).encode("utf-8")
                resp = self._http.put(
                    url, content=clean, headers={"Content-Type": "application/octet-stream"}
                )
                resp.raise_for_status()

    @contextmanager
    def every(self, seconds: float) -> Iterator[None]:
        """Save in the background every ``seconds`` while the block runs."""
        done = threading.Event()

        def loop() -> None:
            while not done.wait(seconds):
                try:
                    self.save()
                except Exception as exc:  # the next save or the final one catches up
                    log.warning("checkpoint failed: %s", exc.__class__.__name__)

        thread = threading.Thread(target=loop, name="checkpoint", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()


def run_retrieve(
    run_id: str,
    bundle_name: str,
    settings: Settings,
    cipher: Cipher,
    blobs: Blobs,
    http: httpx.Client,
    work_dir: Path | None = None,
    redact_output: bool = False,
) -> int:
    """``redact_output`` redacts the keys from everything the process prints or logs."""
    bundle = open_once(
        cipher, blobs, settings.secrets_bucket, settings.retrieve_kms_key, run_id, bundle_name
    )
    keys: dict[str, str] = bundle["keys"]
    if redact_output:
        install(keys.values())
    log.info("retrieve job started for run %s", run_id)
    api = ApiClient(settings.api_url, run_id, bundle["run_token"], http)
    with tempfile.TemporaryDirectory() as tmp:
        out = work_dir or Path(tmp)
        checkpoint = Checkpoint(api, http, out, keys.values())
        try:
            if bundle.get("resume"):
                log.info("restoring the progress of the earlier attempts")
                checkpoint.restore()
            log.info("loading the dataset (%s)", bundle["dataset"])
            loaded = load_hosted(
                blobs, settings.datasets_bucket, bundle["dataset"], bundle["dataset_hash"]
            )
            spec = load_spec(bundle["connector"])
            wait_for_hosts(http, spec.allowed_domains)
            service = LOCAL_SERVICES.get(bundle["connector"])
            with local_service(*service) if service else nullcontext():
                log.info("starting the connector (%s)", bundle["connector"])
                connector = load_connector(bundle["connector"])
                instance = connector.instantiate(keys, spec.preset(bundle.get("preset")))
                runner = Runner(
                    loaded,
                    connector,
                    instance,
                    None,
                    out,
                    RunOptions(
                        budget_tokens=bundle["budget_tokens"],
                        concurrency=spec.concurrency(CONCURRENCY),
                    ),
                    run_id=run_id,
                )
                with checkpoint.every(CHECKPOINT_SECONDS):
                    report = ProgressReport(api.progress, PROGRESS_SECONDS)
                    runner.execute(report.questions, report.sessions, report.turns)
            checkpoint.save()
            api.retrieved(instance.endpoint)
            log.info("retrieval complete: judging starts in its own job")
        except Exception as exc:
            cause = str(exc) if isinstance(exc, RunStoppedError) else short_error(exc)
            message = redact(cause, keys.values())
            log.error("retrieve %s stopped: %s", run_id, message)
            try:
                checkpoint.save()
            except Exception as save_exc:
                log.error("final checkpoint failed: %s", save_exc.__class__.__name__)
            api.stopped(f"Retrieval stopped: {message}")
            return 1
    return 0
