"""The result payload: everything needed to verify, compare and reproduce a run (spec §8)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memjudge.datasets.selection import Selection
from memjudge.metrics import Metrics
from memjudge.sizes import SizeKey

RESULT_SCHEMA_VERSION: Literal["1"] = "1"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolInfo(_Model):
    version: str
    commit: str | None
    image_digest: str | None = None


class DatasetInfo(_Model):
    name: str
    version: str
    hash: str
    curated: bool
    license: str
    upstream_sha256: str | None
    errata_applied: int
    selection: Selection | None = Field(
        default=None, description="How part of the dataset was chosen; None for a full run."
    )
    conversation_ids: list[str] | None = Field(
        default=None, description="Conversations actually run; None for a full run."
    )
    questions: int


class ConnectorInfo(_Model):
    id: str
    code_hash: str
    target: str
    label: str
    sdk: str | None
    settings: dict[str, Any]
    models: dict[str, Any] | Literal["vendor"] = Field(
        default_factory=dict,
        description="Settings that name the models the system ran, or 'vendor' when the vendor "
        "picks them and does not say which. Empty for systems without a model (ADR 0014).",
    )
    endpoint: str | None = Field(
        default=None,
        description="Base URL the connector called, without credentials; None if in-process.",
    )
    custom_endpoint: bool = Field(
        default=False,
        description="True when the endpoint is not one of the connector's declared domains "
        "(localhost, staging, self-hosted). Such runs are never published.",
    )


class JudgeInfo(_Model):
    route: str
    model: str
    upstream_provider: str | None
    effort: str | None
    max_output_tokens: int
    prompt_version: str
    prompt_hash: str
    served_by: list[str]
    official: bool
    """True only when the judge matches the pinned judge for the season."""
    season: str | None


class RunConfigInfo(_Model):
    budget_tokens: int
    tokenizer: str
    concurrency: int
    judge_concurrency: int


class Artifact(_Model):
    name: str
    sha256: str
    url: str | None = None


class ReusedIngestionInfo(_Model):
    """Set when a run reused an earlier run's stored data and only re-ran retrieval.

    Ingestion metrics then come from that run. Such runs are local-only: published runs
    always ingest fresh.
    """

    run_id: str
    connector_code_hash: str
    """Connector code that did the ingestion (may differ from this run's connector)."""


class RejudgedFrom(_Model):
    """Set when this result judged an earlier run's stored contexts again, for a later season.

    Retrieval, ingestion and the connector are that run's; the judge, prompt and tool are this
    result's. The earlier result stays published under its own season.
    """

    run_id: str
    season: str | None
    """The season the earlier run was judged in (None if its judge was unofficial)."""


StopCause = Literal["submitter", "retrieval", "judge", "crash", "runner"]


class RunEvent(_Model):
    """A hosted run stopping or being resumed, as the runner recorded it (ADR 0016)."""

    at: datetime
    event: Literal["stopped", "resumed"]
    cause: StopCause | None = None
    """Why it stopped: the submitter pressed Stop, retrieval failed (the memory system or
    the retrieval job), the judge failed, the job crashed or hit its time limit, or the
    runner itself failed. None for a resume."""


class Affiliation(_Model):
    affiliated: bool | None = Field(
        description="Is the submitter affiliated with the memory system this run tests?"
    )
    system: str | None = None


class RunResult(_Model):
    schema_version: Literal["1"] = RESULT_SCHEMA_VERSION
    run_id: str
    mode: Literal["local", "hosted"]
    started_at: datetime
    finished_at: datetime
    tool: ToolInfo
    dataset: DatasetInfo
    connector: ConnectorInfo
    judge: JudgeInfo
    config: RunConfigInfo
    metrics: dict[SizeKey, Metrics] = Field(
        description="One entry per result size, top10 and top30 (ADR 0018). Ingestion is "
        "shared, so both carry the same ingestion numbers."
    )
    price_table_version: str
    artifacts: list[Artifact]
    affiliation: Affiliation
    reused_ingestion: ReusedIngestionInfo | None = None
    rejudged_from: RejudgedFrom | None = None
    resumes: int = Field(
        default=0,
        ge=0,
        description="Times the run was resumed after it stopped (memory system or judge "
        "failure, crash, or stopped by the submitter). Completed conversations were kept; "
        "unfinished ones were redone.",
    )
    history: list[RunEvent] = Field(
        default_factory=list,
        description="Hosted runs: when the run stopped (and why) and was resumed, as the "
        "runner recorded it. Empty for local runs and runs that never stopped.",
    )
    key_id: str | None = None
