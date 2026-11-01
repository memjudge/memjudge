"""Endpoint recording: every result says which API a run called, and flags custom ones."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from memjudge.connectors.base import (
    BaseConnector,
    Retrieval,
    Usage,
    is_custom_endpoint,
    public_url,
)
from memjudge.connectors.loader import load_connector
from memjudge.datasets.loader import load_dataset
from memjudge.harness import Runner, RunOptions
from memjudge.report.render import render_report
from memjudge.schema import Session
from tests.conftest import fake_judge_setup


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://eu.recalld.ai/", "https://eu.recalld.ai"),
        ("http://user:secret@localhost:8080/api/v2/?token=abc#x", "http://localhost:8080/api/v2"),
        ("http://[::1]:9000", "http://[::1]:9000"),
    ],
)
def test_public_url_strips_credentials(url: str, expected: str) -> None:
    assert public_url(url) == expected


def test_custom_endpoint_detection() -> None:
    domains = ["eu.recalld.ai", "us.recalld.ai"]
    assert not is_custom_endpoint("https://EU.recalld.ai/v1", domains)
    assert is_custom_endpoint("http://localhost:8080", domains)
    assert is_custom_endpoint("https://eu.recalld.ai.evil.com", domains)
    assert not is_custom_endpoint(None, domains)  # in-process, e.g. BM25


class _Api(BaseConnector):
    """BM25 behind a pretend API URL, to exercise endpoint recording."""

    def __init__(self, keys: Mapping[str, str], settings: Mapping[str, Any]) -> None:
        super().__init__(keys, settings)
        self.endpoint = str(settings["base_url"])
        self.inner = load_connector("baselines/bm25").instantiate({})

    def reset(self, namespace: str) -> None:
        self.inner.reset(namespace)

    def ingest(self, namespace: str, session: Session) -> Usage:
        return self.inner.ingest(namespace, session)

    def retrieve(
        self,
        namespace: str,
        query: str,
        asked_at: datetime | None,
        budget: int,
        top_k: int,
    ) -> Retrieval:
        return self.inner.retrieve(namespace, query, asked_at, budget, top_k)


def test_result_records_endpoint_and_flags_custom(tiny_path: Path, tmp_path: Path) -> None:
    loaded = load_dataset(str(tiny_path))
    conn = load_connector("recalld/cloud")  # only its spec (allowed domains) is used here
    url = "http://user:pw@localhost:8080/?k=1"
    result = Runner(
        loaded, conn, _Api({}, {"base_url": url}), fake_judge_setup(), tmp_path, RunOptions()
    ).run()
    assert result.connector.endpoint == "http://localhost:8080"
    assert result.connector.custom_endpoint is True
    assert result.connector.settings["base_url"] == "http://localhost:8080"  # no credentials
    assert "pw" not in (tmp_path / "result.json").read_text("utf-8")
    html = render_report([result], [], loaded, 30)
    assert "Custom endpoint" in html and "localhost:8080" in html


def test_in_process_connector_has_no_endpoint(tiny_path: Path, tmp_path: Path) -> None:
    loaded = load_dataset(str(tiny_path))
    conn = load_connector("baselines/bm25")
    result = Runner(
        loaded, conn, conn.instantiate({}), fake_judge_setup(), tmp_path, RunOptions()
    ).run()
    assert result.connector.endpoint is None and result.connector.custom_endpoint is False
