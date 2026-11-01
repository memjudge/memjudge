"""``memjudge-runner``: one entrypoint per role, all in the same pinned image."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import httpx
import typer

from memjudge_runner.config import HOSTED_DATASETS, Settings

# Job tracebacks go to Cloud Logging: never print local variables (they can hold keys).
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)

HTTP_TIMEOUT = httpx.Timeout(30.0, read=120.0)


@app.command()
def api(port: Annotated[int, typer.Option(envvar="PORT")] = 8080) -> None:
    """Serve the Runner API (Cloud Run service)."""
    import uvicorn
    from google.cloud import firestore

    from memjudge_runner.api import Services, create_app
    from memjudge_runner.auth import OAuthApp
    from memjudge_runner.cloud import (
        CloudLogs,
        CloudRunJobs,
        FirestoreRuns,
        GcsBlobs,
        KmsCipher,
    )
    from memjudge_runner.options import load_options

    s = Settings.from_env()
    services = Services(
        settings=s,
        store=FirestoreRuns(firestore.Client(project=s.project)),
        cipher=KmsCipher(),
        blobs=GcsBlobs(),
        jobs=CloudRunJobs(s.project, s.region),
        logs=CloudLogs(s.project),
        http=httpx.Client(timeout=HTTP_TIMEOUT),
        options=load_options(s.options_file),
        oauth=OAuthApp.from_json(s.secret("github-oauth")),
        session_secret=s.secret("session-secret"),
    )
    uvicorn.run(create_app(services), host="0.0.0.0", port=port, proxy_headers=True)


@app.command()
def retrieve(run_id: str, bundle: str) -> None:
    """Ingest and retrieve for one run (Cloud Run job, one per connector)."""
    from memjudge_runner.cloud import GcsBlobs, KmsCipher
    from memjudge_runner.retrieve import run_retrieve

    with httpx.Client(timeout=HTTP_TIMEOUT) as http:
        code = run_retrieve(
            run_id, bundle, Settings.from_env(), KmsCipher(), GcsBlobs(), http, redact_output=True
        )
    raise typer.Exit(code)


@app.command()
def finish(run_id: str) -> None:
    """Validate, judge, sign and publish one run (Cloud Run job)."""
    from google.cloud import firestore

    from memjudge_runner.cloud import FirestoreRuns, GcsBlobs, KmsCipher
    from memjudge_runner.finish import FinishServices, run_finish
    from memjudge_runner.options import load_options
    from memjudge_runner.publish import GitHubApp, LivePublisher, R2Credentials

    s = Settings.from_env()
    with httpx.Client(timeout=HTTP_TIMEOUT) as http:
        services = FinishServices(
            settings=s,
            store=FirestoreRuns(firestore.Client(project=s.project)),
            cipher=KmsCipher(),
            blobs=GcsBlobs(),
            options=load_options(s.options_file),
            publisher=LivePublisher(
                R2Credentials.from_json(s.secret("r2")),
                GitHubApp.from_json(s.secret("github-app")),
                s.results_repo,
                http,
            ),
            http=http,
        )
        code = run_finish(run_id, services, redact_output=True)
    raise typer.Exit(code)


@app.command("build-options")
def build_options_cmd(out: Annotated[Path, typer.Option("--out", "-o")]) -> None:
    """Write the form's options (datasets, connectors, routes, estimates). Run at image build."""
    from memjudge_runner.options import build_options

    out.write_text(build_options().model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out}")


@app.command("check-core-env")
def check_core_env_cmd(lock: Path) -> None:
    """Fail unless this environment holds exactly LOCK's packages, MemJudge and the runner,
    and nothing that runs at startup. Run at image build from /opt/core (ADR 0009)."""
    import site
    from importlib.metadata import distributions

    from memjudge_runner.envcheck import core_env_problems

    installed = [(d.metadata["Name"], d.version) for d in distributions()]
    site_dirs = [Path(p) for p in site.getsitepackages()]
    problems = core_env_problems(lock.read_text(encoding="utf-8"), installed, site_dirs)
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"{len(installed)} packages, all from {lock}; no startup hooks")


@app.command()
def egress() -> None:
    """Print each hosted connector's subnet and allowed domains, one per line ("-" for
    none). infra/setup.sh builds the firewall rules from this."""
    from memjudge_runner.options import egress_profiles

    for subnet, domains in egress_profiles().items():
        typer.echo(f"{subnet} {','.join(domains) or '-'}")


@app.command("deploy-jobs")
def deploy_jobs_cmd() -> None:
    """Print this image's release label, then one line per job it deploys:
    kind, name, subnet, memory. The deploy workflow reads it."""
    from memjudge import __version__

    from memjudge_runner.config import release_label
    from memjudge_runner.deploy import deploy_jobs

    typer.echo(f"release {release_label(__version__)}")
    for job in deploy_jobs(__version__):
        typer.echo(" ".join(job))


@app.command("stale-releases")
def stale_releases_cmd(days: Annotated[int, typer.Option()] = 14) -> None:
    """Read `gcloud run jobs list --format=json` on stdin; print the releases whose jobs
    to delete (replaced more than DAYS ago)."""
    import json
    import sys
    import time

    from memjudge_runner.deploy import stale_releases

    for release in stale_releases(json.load(sys.stdin), time.time(), days):
        typer.echo(release)


@app.command("mirror-datasets")
def mirror_datasets(bucket: Annotated[str, typer.Option()]) -> None:
    """Copy the hosted datasets' verified upstream files into the private GCS mirror."""
    from memjudge_runner.cloud import GcsBlobs
    from memjudge_runner.datasets import push

    blobs = GcsBlobs()
    for name in HOSTED_DATASETS:
        typer.echo(f"{name}: gs://{bucket}/{push(blobs, bucket, name)}")


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    app()
