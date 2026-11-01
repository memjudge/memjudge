"""What the deploy workflow creates, and which old releases it deletes (ADR 0011).

Each release deploys its own jobs, named with the release, so a stopped run resumes on
the code it started with. Redeploying the same release replaces its jobs in place.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any, NamedTuple

from memjudge.connectors.loader import list_connectors, load_spec

from memjudge_runner.config import (
    DEFAULT_JOB_MEMORY,
    FINISH_SUBNET,
    JOB_MEMORY,
    finish_job_name,
    job_name,
    subnet_name,
)
from memjudge_runner.options import is_hosted_connector

RELEASE_LABEL = "memjudge-release"
DEPLOYED_LABEL = "memjudge-deployed"
"""Unix time of the release's latest deploy."""


class Job(NamedTuple):
    kind: str
    """retrieve or finish: picks the service account and the secrets."""
    name: str
    subnet: str
    memory: str


def deploy_jobs(version: str) -> list[Job]:
    """The finish job and one retrieve job per hosted connector, for release ``version``."""
    jobs = [Job("finish", finish_job_name(version), FINISH_SUBNET, DEFAULT_JOB_MEMORY)]
    for cid in list_connectors():
        if is_hosted_connector(load_spec(cid)):
            memory = JOB_MEMORY.get(cid, DEFAULT_JOB_MEMORY)
            jobs.append(Job("retrieve", job_name(cid, version), subnet_name(cid), memory))
    return jobs


def stale_releases(jobs: list[dict[str, Any]], now: float, days: int) -> list[str]:
    """Releases replaced by a newer deploy more than ``days`` ago, from the output of
    ``gcloud run jobs list --format=json``. The newest release is never stale."""
    deployed: dict[str, float] = {}
    for job in jobs:
        labels = job.get("metadata", {}).get("labels") or {}
        release, at = labels.get(RELEASE_LABEL), labels.get(DEPLOYED_LABEL)
        if release and at and at.isdigit():
            deployed[release] = max(deployed.get(release, 0.0), float(at))
    order = sorted(deployed, key=deployed.__getitem__)
    return [release for release, newer in pairwise(order) if now - deployed[newer] > days * 86400]
