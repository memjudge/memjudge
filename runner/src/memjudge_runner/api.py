"""Runner API (Cloud Run service): the /run form's backend.

Public endpoints serve the form: options, GitHub login, starting, stopping and resuming a
run, and its status. Internal endpoints are called by the retrieve job, authorised by the
run's random token, which only that job knows (it arrives inside the sealed bundle). Each
attempt gets a new token.

Users' keys exist in plain text here only while a request is handled: they are checked,
sealed and never stored or logged. A resume asks for them again.

Every attempt of a run uses the jobs of the release it started on (``RunRecord.version``),
so a resume after a new release runs the same code as the start.

A re-judge (ADR 0015) judges a published run's stored contexts again for the current
season: the API checks the copy of its result the site sends and starts finish directly;
finish checks the published original again before judging.
"""

# No `from __future__ import annotations`: FastAPI must resolve the dependency
# annotations, which refer to functions local to create_app.
import contextlib
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, Protocol
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from memjudge.harness import new_run_id
from memjudge.judge.setup import pinned_judge
from memjudge.result import RunResult
from memjudge.sizes import TOP_K
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memjudge_runner import auth
from memjudge_runner.bundle import Blobs, Cipher, seal
from memjudge_runner.config import (
    ARTIFACT_NAMES,
    TERMS_DATE,
    Settings,
    finish_job_name,
    job_name,
)
from memjudge_runner.finish import rejudge_problem
from memjudge_runner.logs import LogCache, Logs
from memjudge_runner.options import Options
from memjudge_runner.runs import (
    REJUDGE_ACTIVE,
    REJUDGE_DONE,
    IngestUnit,
    RunRecord,
    RunStore,
    now,
    resume_error,
    with_execution,
)

log = logging.getLogger(__name__)

STATE_COOKIE = "mj_oauth_state"
LOGIN_ID = r"^[A-Za-z0-9_-]{16,64}$"
"""The site's random id for one login; see auth.py."""
STUCK_STARTING = timedelta(minutes=10)
UPLOAD_URL_MINUTES = 60
LOG_UNREADABLE = "The log could not be read just now. It is tried again at the next refresh."


class Jobs(Protocol):
    def start(self, job: str, args: list[str]) -> str:
        """Start an execution of ``job`` with ``args``; returns the execution name."""

    def finished(self, execution: str) -> bool:
        """Whether the execution has ended (successfully or not)."""

    def cancel(self, execution: str) -> None:
        """Stop a running execution."""


@dataclass
class Services:
    settings: Settings
    store: RunStore
    cipher: Cipher
    blobs: Blobs
    jobs: Jobs
    logs: Logs
    http: httpx.Client
    options: Options
    oauth: auth.OAuthApp
    session_secret: bytes


class AffiliationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    affiliated: bool
    system: str | None = Field(default=None, max_length=100)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    connector: str
    preset: str | None = None
    judge_route: str
    connector_keys: dict[str, Annotated[str, Field(max_length=2000)]]
    judge_key: str = Field(min_length=1, max_length=2000)
    affiliation: AffiliationIn
    terms: str = Field(pattern=TERMS_DATE.pattern)
    """Date of the Terms of Use the submitter agreed to."""


class RejudgeRequest(BaseModel):
    """The published run's signed result as the site serves it, and the judge key."""

    model_config = ConfigDict(extra="forbid")

    envelope: dict[str, Any]
    judge_route: str
    judge_key: str = Field(min_length=1, max_length=2000)
    terms: str = Field(pattern=TERMS_DATE.pattern)


class ResumeRequest(BaseModel):
    """Every key again, also when only judging is left: finish removes the connector keys
    from the records."""

    model_config = ConfigDict(extra="forbid")

    connector_keys: dict[str, Annotated[str, Field(max_length=2000)]]
    judge_key: str = Field(min_length=1, max_length=2000)


class Progress(BaseModel):
    """Retrieve's counts. Each pair is optional: a report carries the counts it has."""

    done: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    ingested: int | None = Field(default=None, ge=0)
    ingest_total: int | None = Field(default=None, ge=0)
    ingest_unit: IngestUnit = "sessions"


class Retrieved(BaseModel):
    endpoint: str | None = Field(default=None, max_length=500)


class Stopped(BaseModel):
    message: str = Field(max_length=2000)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def check_judge_key(http: httpx.Client, route: str, key: str) -> str | None:
    """A cheap authenticated call, so a bad key fails now rather than after retrieval."""
    if route == "anthropic":
        resp = http.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            params={"limit": 1},
        )
    elif route == "openrouter":
        resp = http.get(
            "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}
        )
    else:
        return None
    if resp.status_code in (401, 403):
        return "The judge key was rejected by the provider."
    if resp.status_code >= 400:
        return f"Could not check the judge key (HTTP {resp.status_code}). Try again."
    return None


def clean_keys(given: dict[str, str], needed: list[str], target: str) -> dict[str, str]:
    keys = {k: v.strip() for k, v in given.items()}
    if set(keys) != set(needed) or not all(keys.values()):
        raise HTTPException(422, f"{target} needs: {', '.join(needed) or 'no keys'}.")
    return keys


def checked_judge_key(http: httpx.Client, route: str, given: str) -> str:
    key = given.strip()
    try:
        problem = check_judge_key(http, route, key)
    except httpx.HTTPError:
        problem = "Could not reach the judge provider to check the key. Try again."
    if problem:
        raise HTTPException(422, problem)
    return key


def create_app(svc: Services) -> FastAPI:
    settings = svc.settings
    site = urlsplit(settings.site_url)
    app = FastAPI(title="MemJudge Runner API", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"{site.scheme}://{site.netloc}"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    bearer = HTTPBearer(auto_error=False)
    callback_url = f"{settings.api_url}/auth/callback"
    log_cache = LogCache(svc.logs)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI echoes the input by default; the input may hold API keys.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    def current_user(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> str:
        user = creds and auth.session_user(svc.session_secret, creds.credentials)
        if not user:
            raise HTTPException(401, "Log in with GitHub to start a run.")
        return auth.user_hash(svc.session_secret, user)

    def running_job(
        run_id: str, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
    ) -> RunRecord:
        run = svc.store.get(run_id)
        if (
            creds is None
            or run is None
            or not hmac.compare_digest(token_hash(creds.credentials), run.run_token_hash)
        ):
            raise HTTPException(403, "Not allowed.")
        if run.state != "retrieving":
            raise HTTPException(409, f"Run is {run.state}.")
        return run

    def owned(run_id: str, user: str) -> RunRecord:
        run = svc.store.get(run_id)
        if run is None or run.user != user:
            raise HTTPException(404, "No such run.")
        return run

    def refresh(run: RunRecord) -> RunRecord:
        """Stop a run whose job ended without reporting (crash, timeout, out of memory),
        so it can be resumed. A run that never got a job fails."""
        if not run.active:
            return run
        changed = False
        if run.execution:
            try:
                ended = svc.jobs.finished(run.execution)
            except Exception as exc:
                log.warning("could not check execution %s: %s", run.execution, exc)
                ended = False
            changed = ended and svc.store.stop(
                run.run_id,
                "The run stopped unexpectedly (a crash or the time limit). It does not count "
                "against your daily limit.",
                counts=False,
                cause="crash",
                execution=run.execution,
            )
        elif run.state == "starting" and now() - run.created_at > STUCK_STARTING:
            changed = svc.store.fail(
                run.run_id, "The run could not be started. Try again later.", counts=False
            )
        return (svc.store.get(run.run_id) or run) if changed else run

    def seal_retrieve(run: RunRecord, run_token: str, keys: dict[str, str], resume: bool) -> str:
        return seal(
            svc.cipher,
            svc.blobs,
            settings.secrets_bucket,
            settings.retrieve_kms_key,
            run.run_id,
            {
                "run_token": run_token,
                "dataset": run.dataset,
                "dataset_hash": run.dataset_hash,
                "connector": run.connector,
                "preset": run.preset,
                "budget_tokens": run.budget_tokens,
                "keys": keys,
                "resume": resume,
            },
        )

    def seal_judge(run_id: str, judge_key: str, keys: dict[str, str]) -> str:
        return seal(
            svc.cipher,
            svc.blobs,
            settings.secrets_bucket,
            settings.finish_kms_key,
            run_id,
            {"judge_key": judge_key, "redact": [*keys.values(), judge_key]},
        )

    @app.get("/health")  # not /healthz: Cloud Run reserves some paths ending in "z"
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/options")
    def options() -> dict[str, Any]:
        # What is live, so the /run page can show which commit and image it talks to.
        runner = {
            "version": settings.version,
            "commit": settings.git_commit,
            "image_digest": settings.image_digest,
        }
        return {**svc.options.model_dump(mode="json"), "runner": runner}

    @app.get("/auth/login")
    def login(login_id: Annotated[str, Query(pattern=LOGIN_ID)]) -> RedirectResponse:
        nonce = secrets.token_urlsafe(16)
        state = auth.issue_state(svc.session_secret, nonce, login_id)
        resp = RedirectResponse(auth.authorize_url(svc.oauth, callback_url, state))
        resp.set_cookie(
            STATE_COOKIE,
            nonce,
            max_age=auth.STATE_SECONDS,
            secure=True,
            httponly=True,
            samesite="lax",
        )
        return resp

    @app.get("/auth/callback")
    def callback(request: Request, code: str = "", state: str = "") -> RedirectResponse:
        back = f"{settings.site_url}/run"
        read = auth.read_state(svc.session_secret, state)
        if not code or not read or read[0] != request.cookies.get(STATE_COOKIE):
            return RedirectResponse(f"{back}#error=login-expired")
        login_id = read[1]
        try:
            user_id = auth.github_user_id(svc.http, svc.oauth, code, callback_url)
        except (auth.LoginError, httpx.HTTPError):
            return RedirectResponse(f"{back}#error=login-failed")
        token = auth.issue_session(svc.session_secret, user_id)
        resp = RedirectResponse(f"{back}#token={token}&login_id={login_id}")
        resp.delete_cookie(STATE_COOKIE)
        return resp

    @app.post("/runs")
    def start_run(req: RunRequest, user: Annotated[str, Depends(current_user)]) -> dict[str, str]:
        opts = svc.options
        dataset = opts.dataset(req.dataset)
        connector = opts.connector(req.connector)
        route = opts.route(req.judge_route)
        if dataset is None or connector is None or route is None:
            raise HTTPException(422, "Pick a dataset, connector and judge route from the list.")
        allowed: list[str | None] = [*connector.presets] or [None]
        if req.preset not in allowed:
            raise HTTPException(422, "Pick a preset for this connector from the list.")
        keys = clean_keys(req.connector_keys, connector.required_keys, connector.target)
        affiliated, system = req.affiliation.affiliated, req.affiliation.system
        if system is not None and (not affiliated or system not in opts.affiliations):
            raise HTTPException(422, "Pick the system you are affiliated with from the list.")
        judge_key = checked_judge_key(svc.http, route.route, req.judge_key)

        for run in svc.store.active():
            refresh(run)
        run_id = new_run_id()
        run_token = secrets.token_urlsafe(32)
        started = now()
        record = RunRecord(
            run_id=run_id,
            user=user,
            day=f"{started:%Y%m%d}",
            dataset=dataset.name,
            connector=connector.id,
            preset=req.preset,
            judge_route=route.route,
            affiliation=req.affiliation.model_dump(),
            terms=req.terms,
            total=dataset.questions * len(TOP_K),  # retrieve searches each size (ADR 0018)
            created_at=started,
            updated_at=started,
            run_token_hash=token_hash(run_token),
            version=settings.version,
            dataset_hash=dataset.hash,
            budget_tokens=opts.budget_tokens,
            key_names=connector.required_keys,
        )
        refused = svc.store.create(record, settings.limits)
        if refused:
            raise HTTPException(429, refused)
        try:
            retrieve_bundle = seal_retrieve(record, run_token, keys, resume=False)
            svc.store.update(run_id, judge_object=seal_judge(run_id, judge_key, keys))
            execution = svc.jobs.start(
                job_name(record.connector, record.version), ["retrieve", run_id, retrieve_bundle]
            )
            svc.store.update(run_id, state="retrieving", **with_execution(record, execution))
        except Exception as exc:
            log.error("could not start run %s: %s", run_id, exc.__class__.__name__)
            svc.store.fail(run_id, "The run could not be started. Try again later.", counts=False)
            raise HTTPException(503, "The run could not be started. Try again later.") from None
        return {"run_id": run_id}

    @app.post("/runs/{run_id}/rejudge")
    def rejudge_run(
        run_id: str, req: RejudgeRequest, user: Annotated[str, Depends(current_user)]
    ) -> dict[str, str]:
        """Judge a published run's stored contexts again for the current season."""
        opts = svc.options
        route = opts.route(req.judge_route)
        if route is None:
            raise HTTPException(422, "Pick a judge route from the list.")
        try:
            original = RunResult.model_validate(req.envelope.get("payload"))
        except ValidationError:
            raise HTTPException(422, "That is not a MemJudge result.") from None
        if original.run_id != run_id:
            raise HTTPException(422, "The result is for another run.")
        season = pinned_judge().season
        # The same dataset version, by hash: the form names datasets its own way.
        offered = next((d for d in opts.datasets if d.hash == original.dataset.hash), None)
        problem = rejudge_problem(
            original, season, offered.hash if offered else None, opts.budget_tokens
        )
        if problem or offered is None:
            raise HTTPException(409, problem)
        judge_key = checked_judge_key(svc.http, route.route, req.judge_key)

        for run in svc.store.active():
            refresh(run)
        new_id = new_run_id()
        started = now()
        record = RunRecord(
            run_id=new_id,
            user=user,
            day=f"{started:%Y%m%d}",
            dataset=offered.name,
            connector=original.connector.id,
            judge_route=route.route,
            affiliation=original.affiliation.model_dump(),
            terms=req.terms,
            total=original.dataset.questions,
            created_at=started,
            updated_at=started,
            run_token_hash="",  # no retrieve job reports for a re-judge
            version=settings.version,
            dataset_hash=original.dataset.hash,
            budget_tokens=opts.budget_tokens,
            key_names=[],
            rejudge_of=run_id,
            rejudge_season=season,
        )
        refused = svc.store.create(record, settings.limits)
        if refused:
            raise HTTPException(409 if refused in (REJUDGE_ACTIVE, REJUDGE_DONE) else 429, refused)
        try:
            svc.store.update(
                new_id, judge_object=seal_judge(new_id, judge_key, {}), state="judging", done=0
            )
            execution = svc.jobs.start(finish_job_name(record.version), ["finish", new_id])
            svc.store.update(new_id, **with_execution(record, execution))
        except Exception as exc:
            log.error("could not start re-judge %s: %s", new_id, exc.__class__.__name__)
            svc.store.fail(
                new_id, "The re-judge could not be started. Try again later.", counts=False
            )
            raise HTTPException(
                503, "The re-judge could not be started. Try again later."
            ) from None
        return {"run_id": new_id}

    @app.get("/rejudges")
    def rejudges() -> dict[str, Any]:
        """Re-judges into the current season, by published run: public, so the leaderboard
        can show which runs are being re-judged."""
        for run in svc.store.active():
            if run.rejudge_of:
                refresh(run)
        season = pinned_judge().season
        return {"season": season, "runs": svc.store.rejudges(season)}

    @app.get("/runs")
    def my_runs(user: Annotated[str, Depends(current_user)]) -> dict[str, Any]:
        """The signed-in user's runs, so any browser can find one to follow or resume."""
        return {"runs": [refresh(r).public() for r in svc.store.for_user(user)]}

    @app.get("/runs/{run_id}")
    def run_status(run_id: str, user: Annotated[str, Depends(current_user)]) -> dict[str, Any]:
        return refresh(owned(run_id, user)).public()

    @app.get("/runs/{run_id}/log")
    def run_log(run_id: str, user: Annotated[str, Depends(current_user)]) -> dict[str, Any]:
        """The newest lines the run's jobs wrote, oldest first: for its owner only."""
        run = owned(run_id, user)
        lines, read = log_cache.lines(run.run_id, run.all_executions, run.created_at)
        return {
            "lines": [line.model_dump(mode="json") for line in lines],
            "error": None if read else LOG_UNREADABLE,
        }

    @app.post("/runs/{run_id}/stop")
    def stop_run(run_id: str, user: Annotated[str, Depends(current_user)]) -> dict[str, Any]:
        """Stop a run and keep its progress (the "pause"): resume it later."""
        run = refresh(owned(run_id, user))
        if run.state not in ("retrieving", "judging"):
            raise HTTPException(409, f"A run that is {run.state} cannot be stopped.")
        changed = svc.store.stop(run_id, "Stopped by you.", counts=True, cause="submitter")
        if changed and run.execution:
            try:
                svc.jobs.cancel(run.execution)
            except Exception as exc:  # the job finds the run stopped at its next report
                log.warning("could not cancel %s: %s", run.execution, exc.__class__.__name__)
        return (svc.store.get(run_id) or run).public()

    @app.post("/runs/{run_id}/resume")
    def resume_run(
        run_id: str, req: ResumeRequest, user: Annotated[str, Depends(current_user)]
    ) -> dict[str, str]:
        run = owned(run_id, user)
        problem = resume_error(run)
        if problem:
            raise HTTPException(409, problem)
        keys = clean_keys(req.connector_keys, run.key_names, run.connector)
        judge_key = checked_judge_key(svc.http, run.judge_route, req.judge_key)

        for other in svc.store.active():
            refresh(other)
        run_token = secrets.token_urlsafe(32)
        refused = svc.store.resume(
            run_id, settings.limits, run_token_hash=token_hash(run_token), execution=None
        )
        if refused:
            raise HTTPException(429, refused)
        try:
            svc.store.update(run_id, judge_object=seal_judge(run_id, judge_key, keys))
            if run.resume_from == "retrieve":
                bundle = seal_retrieve(run, run_token, keys, resume=True)
                job, args = job_name(run.connector, run.version), ["retrieve", run_id, bundle]
            else:
                job, args = finish_job_name(run.version), ["finish", run_id]
            svc.store.update(run_id, **with_execution(run, svc.jobs.start(job, args)))
        except Exception as exc:
            log.error("could not resume run %s: %s", run_id, exc.__class__.__name__)
            svc.store.stop(
                run_id,
                "The run could not be resumed. Try again later.",
                counts=True,
                cause="runner",
            )
            # This attempt never ran: the run stays as it was stopped.
            svc.store.update(run_id, resumes=run.resumes, history=run.history)
            raise HTTPException(503, "The run could not be resumed. Try again later.") from None
        if run.judge_object:  # the previous attempt's, if its finish never opened it
            with contextlib.suppress(Exception):
                svc.blobs.delete(settings.secrets_bucket, run.judge_object)
        return {"run_id": run_id}

    @app.post("/internal/runs/{run_id}/progress")
    def progress(body: Progress, run: Annotated[RunRecord, Depends(running_job)]) -> None:
        fields: dict[str, Any] = {"job_started": True}  # its first report ends start-up
        if body.done is not None:
            fields["done"] = min(body.done, run.total)
        if body.ingested is not None and body.ingest_total is not None:
            fields.update(
                ingested=min(body.ingested, body.ingest_total),
                ingest_total=body.ingest_total,
                ingest_unit=body.ingest_unit,
            )
        svc.store.update(run.run_id, **fields)

    @app.post("/internal/runs/{run_id}/uploads")
    def uploads(run: Annotated[RunRecord, Depends(running_job)]) -> dict[str, str]:
        return {
            name: svc.blobs.signed_put_url(
                settings.staging_bucket, f"{run.run_id}/{name}", UPLOAD_URL_MINUTES
            )
            for name in ARTIFACT_NAMES
        }

    @app.post("/internal/runs/{run_id}/downloads")
    def downloads(run: Annotated[RunRecord, Depends(running_job)]) -> dict[str, str]:
        """Where a resumed retrieve job reads the progress saved by earlier attempts."""
        return {
            name: svc.blobs.signed_get_url(
                settings.staging_bucket, f"{run.run_id}/{name}", UPLOAD_URL_MINUTES
            )
            for name in ARTIFACT_NAMES
        }

    @app.post("/internal/runs/{run_id}/retrieved")
    def retrieved(body: Retrieved, run: Annotated[RunRecord, Depends(running_job)]) -> None:
        # Atomic, and it spends the run token: a repeated call, or a Stop that landed after
        # running_job read the run, starts no finish.
        moved = svc.store.transition(
            run.run_id,
            "retrieving",
            state="judging",
            done=0,
            endpoint=body.endpoint,
            run_token_hash="",
        )
        if not moved:
            raise HTTPException(409, "Run is no longer retrieving.")
        try:
            execution = svc.jobs.start(finish_job_name(run.version), ["finish", run.run_id])
        except Exception as exc:
            log.error("could not start finish for %s: %s", run.run_id, exc.__class__.__name__)
            svc.store.fail(
                run.run_id, "The run could not be finished. Try again later.", counts=False
            )
            raise HTTPException(503, "Could not start finish.") from None
        svc.store.update(run.run_id, **with_execution(run, execution))

    @app.post("/internal/runs/{run_id}/stopped")
    def stopped(body: Stopped, run: Annotated[RunRecord, Depends(running_job)]) -> None:
        svc.store.stop(run.run_id, body.message[:500], counts=True, cause="retrieval")

    return app
