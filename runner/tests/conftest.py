from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fakes import TINY, FakeBlobs, FakeCipher, FakeJobs, FakeLogs
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from memjudge.datasets.loader import load_dataset

from memjudge_runner import auth
from memjudge_runner.api import Services, create_app
from memjudge_runner.config import Limits, Settings
from memjudge_runner.options import (
    ConnectorOption,
    DatasetOption,
    Options,
    RouteOption,
)
from memjudge_runner.runs import MemoryRuns

SESSION_SECRET = b"s" * 32
GOOD_JUDGE_KEY = "sk-ant-good-judge-key-0001"


def judge_provider(request: httpx.Request) -> httpx.Response:
    """Stands in for the judge provider's key check."""
    ok = request.headers.get("x-api-key") == GOOD_JUDGE_KEY
    return httpx.Response(200 if ok else 401, json={})


@pytest.fixture
def signing_key(tmp_path: Path) -> Ed25519PrivateKey:
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "secrets" / "signing-key" / "value"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return key


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        secrets_bucket="secrets",
        staging_bucket="staging",
        datasets_bucket="datasets",
        retrieve_kms_key="kms/retrieve",
        finish_kms_key="kms/finish",
        api_url="http://testserver",
        site_url="https://memjudge.test",
        artifact_url_prefix="https://artifacts.memjudge.test/runs/",
        secrets_dir=tmp_path / "secrets",
        git_commit="abc123",
        image_digest="sha256:feed",
        limits=Limits(),
    )


@pytest.fixture
def options() -> Options:
    loaded = load_dataset(str(TINY))
    return Options(
        budget_tokens=4096,
        judge_season="test",
        datasets=[
            DatasetOption(
                name=str(TINY),
                label=loaded.label,
                hash=loaded.hash,
                license=loaded.dataset.license,
                conversations=len(loaded.dataset.conversations),
                questions=len(loaded.dataset.questions),
            )
        ],
        connectors=[
            ConnectorOption(
                id="baselines/bm25",
                target="BM25",
                label="maintainer-written",
                required_keys=[],
                allowed_domains=[],
            ),
            ConnectorOption(
                id="letta/cloud",
                target="Letta Cloud",
                label="maintainer-written",
                required_keys=["LETTA_API_KEY"],
                allowed_domains=["api.letta.com"],
            ),
            ConnectorOption(
                id="recalld/cloud",
                target="Recalld",
                label="vendor-maintained",
                required_keys=["RECALLD_API_KEY"],
                allowed_domains=["us.recalld.ai"],
                presets=["recall", "search"],
            ),
        ],
        judge_routes=[
            RouteOption(
                route="anthropic", spec="anthropic:claude-sonnet-5-5", key_name="ANTHROPIC_API_KEY"
            )
        ],
        estimates={},
    )


@pytest.fixture
def services(settings: Settings, options: Options) -> Services:
    return Services(
        settings=settings,
        store=MemoryRuns(),
        cipher=FakeCipher(),
        blobs=FakeBlobs(),
        jobs=FakeJobs(),
        logs=FakeLogs(),
        http=httpx.Client(transport=httpx.MockTransport(judge_provider)),
        options=options,
        oauth=auth.OAuthApp("client-id", "client-secret"),
        session_secret=SESSION_SECRET,
    )


@pytest.fixture
def app(services: Services) -> FastAPI:
    app = create_app(services)
    blobs = services.blobs

    @app.put("/__fake_gcs/{bucket}/{name:path}")
    async def fake_upload(bucket: str, name: str, request: Request) -> None:
        blobs.put(bucket, name, await request.body())

    @app.get("/__fake_gcs/{bucket}/{name:path}")
    def fake_download(bucket: str, name: str) -> Response:
        if not blobs.exists(bucket, name):
            return Response(status_code=404)
        return Response(blobs.get(bucket, name))

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, base_url="https://testserver")  # the login cookie is Secure


def session(user_id: str = "1001") -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.issue_session(SESSION_SECRET, user_id)}"}


def run_request(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "dataset": str(TINY),
        "connector": "baselines/bm25",
        "judge_route": "anthropic",
        "connector_keys": {},
        "judge_key": GOOD_JUDGE_KEY,
        "affiliation": {"affiliated": False},
        "terms": "2026-10-17",
    }
    body.update(overrides)
    return body
