"""Settings from the environment, as set on the Cloud Run service and job definitions.

Nothing per run comes from the environment: runs pass only their id (and, for retrieve,
the name of a sealed bundle) as job arguments.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from memjudge import __version__
from memjudge.sizes import TOP_K, questions_file, size_key

HOSTED_DATASETS = ("locomo", "longmemeval-s")
"""Curated datasets offered on the /run form (the launch set)."""
EXCLUDED_CONNECTORS = frozenset({"offline/replay", "mem0/platform", "zep/cloud"})
"""Never hosted. offline/replay replays stored outputs instead of calling a memory system.
Mem0 Platform's and Zep Cloud's terms forbid publishing benchmark results without the
vendor's consent, and a hosted run always publishes: they stay local-only until it is given."""
AFFILIATED_SYSTEMS = ("Letta", "Mem0", "Recalld", "Supermemory", "Zep")
"""Makers a submitter can name as affiliated. A fixed list, not free text: the answer is
published. "Another memory system" is sent as no name."""
TERMS_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
"""A start request carries the date of the Terms of Use the submitter agreed to."""
RUNS_LISTED = 20
"""How many of a user's runs the /run page lists, newest first."""
RECORD_DAYS = 90
"""Firestore deletes a run record (with its user HMAC) this long after the run starts."""
QUOTA_DAYS = 30
"""And a daily-limit counter this long after its day: longer than any run can last, since
a stop that refunds a run updates its day's counter."""

BUDGET_TOKENS = 8192
CONCURRENCY = 4
JUDGE_CONCURRENCY = 8
JUDGE_RETRY_ROUNDS = 2
ARTIFACT_NAMES = ("conversations.jsonl", *map(questions_file, TOP_K))
"""The retrieve job's records: one question file per result size (ADR 0018)."""


def judged_file(top_k: int) -> str:
    """Finish's judged records of one size in staging, so a resumed finish judges only what
    is left."""
    return f"judged-{size_key(top_k)}.jsonl"


JUDGED_SAVE_SECONDS = 60.0
"""How often finish saves judged records while judging: a crash re-judges (and leaves
uncounted) at most this much judge spend."""
CHECKPOINT_SECONDS = 300.0
"""How often retrieve uploads its progress: a crash loses at most this much work."""
RESUME_HOURS = 48
MAX_RESUMES = 3
JUDGE_HOSTS = {"anthropic": "api.anthropic.com", "openrouter": "openrouter.ai"}
"""The host each judge route calls (finish's subnet may reach only these, R2 and GitHub)."""
JOB_MEMORY = {"mem0/oss": "8Gi", "zep/graphiti": "8Gi"}
"""Retrieve job memory where 4Gi is not enough: Mem0 OSS keeps its vector store in /tmp,
which Cloud Run holds in RAM, and FalkorDB keeps Graphiti's graph in memory."""
LOCAL_SERVICES: dict[str, tuple[list[str], int]] = {
    "zep/graphiti": (
        [
            "/opt/falkordb/redis-server",
            "--bind",
            "127.0.0.1",
            "--port",
            "6379",
            "--dir",
            "/tmp",
            "--save",
            "",
            "--appendonly",
            "no",
            "--loglevel",  # the run's log gets warnings and crash reports, not startup chatter
            "warning",
            "--loadmodule",
            "/opt/falkordb/falkordb.so",
        ],
        6379,
    ),
}
"""Programs a connector needs on localhost (command, port), started by its retrieve job
before the connector and stopped after it: FalkorDB for Graphiti, empty for every run."""
DEFAULT_JOB_MEMORY = "4Gi"


@dataclass(frozen=True)
class Limits:
    user_active: int = 1
    user_daily: int = 3
    global_active: int = 3


@dataclass(frozen=True)
class Settings:
    project: str = ""
    region: str = ""
    secrets_bucket: str = ""
    """Sealed key bundles, under random names. Job service accounts can get and delete, not list."""
    staging_bucket: str = ""
    """Retrieval output, uploaded by retrieve through signed URLs and read by finish."""
    datasets_bucket: str = ""
    """Private mirror of the hosted datasets' pinned upstream files."""
    retrieve_kms_key: str = ""
    finish_kms_key: str = ""
    api_url: str = ""
    site_url: str = "https://memjudge.org"
    artifact_url_prefix: str = "https://artifacts.memjudge.org/runs/"
    results_repo: str = "memjudge/results"
    version: str = __version__
    """This image's release. Its jobs carry it in their names; a run resumes on its own."""
    secrets_dir: Path = Path("/secrets")
    options_file: Path = Path("/app/options.json")
    git_commit: str | None = None
    image_digest: str | None = None
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            project=env("MJR_PROJECT", ""),
            region=env("MJR_REGION", ""),
            secrets_bucket=env("MJR_SECRETS_BUCKET", ""),
            staging_bucket=env("MJR_STAGING_BUCKET", ""),
            datasets_bucket=env("MJR_DATASETS_BUCKET", ""),
            retrieve_kms_key=env("MJR_RETRIEVE_KMS_KEY", ""),
            finish_kms_key=env("MJR_FINISH_KMS_KEY", ""),
            api_url=env("MJR_API_URL", "").rstrip("/"),
            site_url=env("MJR_SITE_URL", cls.site_url).rstrip("/"),
            artifact_url_prefix=env("MJR_ARTIFACT_URL_PREFIX", cls.artifact_url_prefix),
            results_repo=env("MJR_RESULTS_REPO", cls.results_repo),
            secrets_dir=Path(env("MJR_SECRETS_DIR", str(cls.secrets_dir))),
            options_file=Path(env("MJR_OPTIONS_FILE", str(cls.options_file))),
            git_commit=env("MEMJUDGE_GIT_COMMIT") or None,
            image_digest=env("MEMJUDGE_IMAGE_DIGEST") or None,
        )

    def secret(self, name: str) -> bytes:
        """A secret mounted by Cloud Run at ``<secrets_dir>/<name>/value``."""
        return (self.secrets_dir / name / "value").read_bytes()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def release_label(version: str) -> str:
    """The release as a label value and name part, e.g. ``0-1-0``."""
    return _slug(version)


def job_name(connector_id: str, version: str) -> str:
    """Cloud Run job that retrieves with ``connector_id`` on release ``version``,
    e.g. ``retrieve-mem0-platform-0-1-0``. Each release deploys its own jobs."""
    return f"retrieve-{_slug(connector_id)}-{_slug(version)}"


def finish_job_name(version: str) -> str:
    """Cloud Run job that finishes runs of release ``version``, e.g. ``finish-0-1-0``."""
    return f"finish-{_slug(version)}"


FINISH_SUBNET = "sn-finish"


def subnet_name(connector_id: str) -> str:
    """The subnet a connector's retrieve jobs run in, e.g. ``sn-mem0-platform``. The
    firewall policy allows that subnet only the connector's declared domains."""
    return f"sn-{_slug(connector_id)}"
