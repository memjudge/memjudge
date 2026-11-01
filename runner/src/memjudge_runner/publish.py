"""Publish a finished run: artifacts to R2, the signed result to the results repo.

The results repo is written only by the MemJudge GitHub App. The push triggers the
leaderboard rebuild (``results/.github/workflows/notify.yml``).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import jwt

GITHUB_API = "https://api.github.com"
GITHUB_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


class Publisher(Protocol):
    def upload_artifact(self, key: str, data: bytes) -> None: ...
    def commit_result(self, path: str, text: str, message: str) -> None: ...

    def download_artifact(self, key: str) -> bytes:
        """A published artifact (a re-judge reads the earlier run's stored contexts)."""

    def read_result(self, path: str) -> str:
        """A published signed result, as committed to the results repo."""


def envelope_text(envelope: dict[str, Any]) -> str:
    """Same layout as ``memjudge sign`` writes."""
    return json.dumps(envelope, indent=2, ensure_ascii=False)


def app_jwt(app_id: str, private_key_pem: bytes) -> str:
    """Short-lived JWT that authenticates as the GitHub App itself."""
    now = int(time.time())
    return jwt.encode({"iat": now - 60, "exp": now + 540, "iss": app_id}, private_key_pem, "RS256")


def installation_token(
    http: httpx.Client, app_id: str, installation_id: str, private_key_pem: bytes
) -> str:
    resp = http.post(
        f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
        headers={**GITHUB_HEADERS, "Authorization": f"Bearer {app_jwt(app_id, private_key_pem)}"},
    )
    resp.raise_for_status()
    token: str = resp.json()["token"]
    return token


def read_file(http: httpx.Client, token: str, repo: str, path: str) -> str:
    """``path`` on the default branch, as text."""
    resp = http.get(
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        headers={
            **GITHUB_HEADERS,
            "Accept": "application/vnd.github.raw+json",
            "Authorization": f"Bearer {token}",
        },
    )
    resp.raise_for_status()
    return resp.text


def commit_file(
    http: httpx.Client, token: str, repo: str, path: str, text: str, message: str
) -> None:
    """Create ``path`` on the default branch. Fails if it already exists (no overwrites)."""
    resp = http.put(
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        headers={**GITHUB_HEADERS, "Authorization": f"Bearer {token}"},
        json={"message": message, "content": base64.b64encode(text.encode()).decode()},
    )
    resp.raise_for_status()


@dataclass(frozen=True)
class GitHubApp:
    app_id: str
    installation_id: str
    private_key_pem: bytes

    @classmethod
    def from_json(cls, data: bytes) -> GitHubApp:
        d = json.loads(data)
        return cls(str(d["app_id"]), str(d["installation_id"]), d["private_key"].encode())


@dataclass(frozen=True)
class R2Credentials:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_json(cls, data: bytes) -> R2Credentials:
        d = json.loads(data)
        return cls(d["account_id"], d["access_key_id"], d["secret_access_key"], d["bucket"])


class LivePublisher:
    """R2 over its S3 API, and the results repo through the GitHub App."""

    def __init__(self, r2: R2Credentials, app: GitHubApp, repo: str, http: httpx.Client) -> None:
        import boto3

        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=r2.access_key_id,
            aws_secret_access_key=r2.secret_access_key,
            region_name="auto",
        )
        self._bucket = r2.bucket
        self._app = app
        self._repo = repo
        self._http = http

    def upload_artifact(self, key: str, data: bytes) -> None:
        self._s3.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType="application/x-ndjson"
        )

    def commit_result(self, path: str, text: str, message: str) -> None:
        commit_file(self._http, self._token(), self._repo, path, text, message)

    def download_artifact(self, key: str) -> bytes:
        data: bytes = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        return data

    def read_result(self, path: str) -> str:
        return read_file(self._http, self._token(), self._repo, path)

    def _token(self) -> str:
        return installation_token(
            self._http, self._app.app_id, self._app.installation_id, self._app.private_key_pem
        )
