"""Finish job: validate the retrieval output, judge it, sign and publish the result.

Runs only MemJudge code: it never imports connector code (``connector_ref`` reads the spec
and hashes the folder). It alone can decrypt the judge key and read the signing key, the
R2 credentials and the GitHub App key.

The judged records are saved to staging while judging and at the end, so a resumed finish
(after the judge key ran out of credit, a crash or a publishing error) judges only the
questions still unjudged, and keeps the judge spend of earlier failed attempts.

A re-judge (ADR 0015) has no retrieve job: finish reads the earlier run's signed result from
the results repo, checks it, and judges its published contexts after checking them against
the signed hashes. The new result keeps that run's connector record and sets
``rejudged_from``.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from memjudge import __version__
from memjudge.canonical import sha256_hex
from memjudge.connectors.base import is_custom_endpoint
from memjudge.connectors.loader import connector_ref
from memjudge.datasets.loader import LoadedDataset
from memjudge.harness import (
    CONVERSATIONS_FILE,
    JudgeSetup,
    RunOptions,
    add_judge_spend,
    build_result,
    judge_records,
)
from memjudge.judge.client import KEY_ENV, JudgeModel
from memjudge.judge.setup import can_rejudge, make_judge_setup, pinned_judge
from memjudge.records import QuestionRecord
from memjudge.result import Affiliation, RejudgedFrom, RunResult, ToolInfo
from memjudge.signing import (
    VerificationError,
    key_id,
    load_private_key,
    sign_result,
    verify_envelope,
)
from memjudge.sizes import TOP_K, questions_file, size_key

from memjudge_runner.bundle import Blobs, Cipher, open_once
from memjudge_runner.config import (
    CONCURRENCY,
    JUDGE_CONCURRENCY,
    JUDGE_HOSTS,
    JUDGE_RETRY_ROUNDS,
    JUDGED_SAVE_SECONDS,
    Settings,
    judged_file,
)
from memjudge_runner.datasets import load_hosted
from memjudge_runner.options import Options
from memjudge_runner.publish import Publisher, envelope_text
from memjudge_runner.redact import install, redact
from memjudge_runner.retrieve import short_error, throttled, wait_for_hosts
from memjudge_runner.runs import RunRecord, RunStore
from memjudge_runner.validate import InvalidOutputError, clean_conversations, clean_questions

log = logging.getLogger(__name__)


class RunError(Exception):
    """The run cannot be published for a reason the user should see. Counts as a run."""


class JudgeIncompleteError(Exception):
    """The judge failed on some questions: the run stops, and a resume judges the rest."""


class RunEndedError(Exception):
    """The run left ``judging`` while this attempt worked on it (its owner stopped it and
    the cancel did not reach this job): publish nothing and leave the run as it is."""


def pinned_setup(route: str, key: str) -> JudgeSetup:
    pinned = pinned_judge()
    model = JudgeModel.parse(
        pinned.routes[route], effort=pinned.effort, max_output_tokens=pinned.max_output_tokens
    )
    return make_judge_setup(model, {KEY_ENV[route]: key})


@dataclass
class FinishServices:
    settings: Settings
    store: RunStore
    cipher: Cipher
    blobs: Blobs
    options: Options
    publisher: Publisher
    judge_setup: Callable[[str, str], JudgeSetup] = pinned_setup
    http: httpx.Client | None = None
    """When set, finish waits for the judge host before judging (Cloud NAT warm-up)."""


def run_finish(run_id: str, svc: FinishServices, redact_output: bool = False) -> int:
    """``redact_output`` redacts the users' keys from everything the process prints or logs."""
    run = svc.store.get(run_id)
    if run is None or run.state != "judging" or not run.judge_object:
        log.error("run %s is not waiting to be judged", run_id)
        return 1
    secrets: list[str] = []
    try:
        s = svc.settings
        bundle = open_once(
            svc.cipher, svc.blobs, s.secrets_bucket, s.finish_kms_key, run_id, run.judge_object
        )
        secrets = bundle["redact"]
        if redact_output:
            install(secrets)
        log.info("finish job started for run %s", run_id)
        with tempfile.TemporaryDirectory() as tmp:
            _finish(run, bundle, Path(tmp), svc)
    except JudgeIncompleteError as exc:
        svc.store.stop(run_id, str(exc), counts=True, cause="judge")
        return 1
    except RunEndedError:
        log.warning("run %s is no longer judging; nothing published", run_id)
        return 1
    except RunError as exc:
        svc.store.fail(run_id, str(exc), counts=True)
        return 1
    except Exception as exc:
        log.error("finish %s failed: %s", run_id, redact(short_error(exc), secrets))
        svc.store.stop(
            run_id,
            "The run could not be finished because of an internal error. "
            "It does not count against your daily limit.",
            counts=False,
            cause="runner",
        )
        return 1
    return 0


def _saved_verdicts(
    svc: FinishServices, run_id: str, top_k: int, questions: list[QuestionRecord]
) -> tuple[dict[str, QuestionRecord], dict[str, QuestionRecord]]:
    """Records of one result size an earlier finish of this run saved, for the same contexts:
    verdicts, and failed attempts (whose judge spend still counts). Only finish writes this
    file (retrieve can upload only its own record files)."""
    name = f"{run_id}/{judged_file(top_k)}"
    if not svc.blobs.exists(svc.settings.staging_bucket, name):
        return {}, {}
    contexts = {r.question_id: r.context for r in questions}
    saved: dict[str, QuestionRecord] = {}
    failed: dict[str, QuestionRecord] = {}
    for line in svc.blobs.get(svc.settings.staging_bucket, name).decode("utf-8").split("\n"):
        if not line.strip():
            continue
        rec = QuestionRecord.model_validate_json(line)
        if rec.retrieval_error or contexts.get(rec.question_id) != rec.context:
            continue
        if rec.correct is not None and not rec.judge_error:
            saved[rec.question_id] = rec
        elif rec.judge_error:
            failed[rec.question_id] = rec
    return saved, failed


def _original(run: RunRecord, loaded: LoadedDataset, svc: FinishServices) -> RunResult:
    """The published run a re-judge judges again, read from the results repo (written only
    by the runner) and checked again here: the API checked the copy the site sent."""
    path = f"runs/{run.rejudge_of}.json"
    try:
        envelope = json.loads(svc.publisher.read_result(path))
        public = load_private_key(svc.settings.secret("signing-key")).public_key()
        verify_envelope(envelope, {key_id(public): public})
        original = RunResult.model_validate(envelope["payload"])
    except (httpx.HTTPError, ValueError, KeyError, VerificationError) as exc:
        raise RunError(f"The published result of run {run.rejudge_of} could not be read.") from exc
    problem = rejudge_problem(original, run.rejudge_season, loaded.hash, svc.options.budget_tokens)
    if original.run_id != run.rejudge_of or problem:
        raise RunError(problem or "The published result is for another run.")
    return original


def rejudge_problem(
    original: RunResult, season: str | None, dataset_hash: str | None, budget: int
) -> str | None:
    """Why ``original`` cannot be re-judged into ``season`` (the current season), or None.
    ``dataset_hash`` is the hash of the dataset version on offer (None if not offered)."""
    if original.mode != "hosted" or not original.judge.official or original.dataset.selection:
        return "Only published hosted runs of a full dataset can be re-judged."
    if season != pinned_judge().season:
        return "The current season changed. Reload the page and try again."
    if original.judge.season == season:
        return "This run is already in the current season."
    if not can_rejudge(original.judge.season):
        return (
            "What the judge sees has changed since this run's season, so it cannot be "
            "re-judged. It needs a new run."
        )
    if dataset_hash != original.dataset.hash or budget != original.config.budget_tokens:
        return "The dataset or token budget changed since this run. It needs a new run."
    if set(RECORD_FILES) - {a.name for a in original.artifacts}:
        return "This run's stored contexts were not published."
    return None


RECORD_FILES = (CONVERSATIONS_FILE, *map(questions_file, TOP_K))
"""What a run publishes besides its result, and what a re-judge reads back."""


def _published(svc: FinishServices, original: RunResult, name: str) -> bytes:
    """One of the earlier run's published files, checked against its signed hash."""
    data = svc.publisher.download_artifact(f"runs/{original.run_id}/{name}")
    signed = next(a.sha256 for a in original.artifacts if a.name == name)
    if "sha256:" + sha256_hex(data) != signed:
        raise RunError(f"The published {name} of run {original.run_id} does not match its hash.")
    return data


def _judge_size(
    run: RunRecord,
    top_k: int,
    questions: list[QuestionRecord],
    saved: dict[str, QuestionRecord],
    failed: dict[str, QuestionRecord],
    loaded: LoadedDataset,
    setup: JudgeSetup,
    svc: FinishServices,
    base: int,
) -> list[QuestionRecord]:
    """Judge one result size's records not judged earlier, saving them to staging as it
    goes. ``base`` is the progress already done when it starts."""
    todo = [r for r in questions if r.question_id not in saved]
    log.info("judging %d %s answers (%d judged earlier)", len(todo), size_key(top_k), len(saved))

    def save(by_id: dict[str, QuestionRecord]) -> None:
        svc.blobs.put(
            svc.settings.staging_bucket,
            f"{run.run_id}/{judged_file(top_k)}",
            "".join(
                by_id[r.question_id].model_dump_json() + "\n"
                for r in questions
                if r.question_id in by_id
            ).encode("utf-8"),
        )

    # Saved while judging, so a crash neither re-judges nor loses the spend of what was done.
    done: dict[str, QuestionRecord] = {}
    last_save = [time.monotonic()]

    def keep(rec: QuestionRecord) -> None:
        earlier = failed.get(rec.question_id)
        done[rec.question_id] = add_judge_spend(rec, earlier) if earlier else rec
        if time.monotonic() - last_save[0] >= JUDGED_SAVE_SECONDS:
            last_save[0] = time.monotonic()
            try:
                save({**failed, **saved, **done})
            except Exception as exc:  # best effort: the final save below still runs
                log.warning("saving judged records failed: %s", exc.__class__.__name__)

    judge_records(
        todo,
        {q.id: q for q in loaded.dataset.questions},
        setup,
        JUDGE_CONCURRENCY,
        JUDGE_RETRY_ROUNDS,
        progress=throttled(lambda d, t: svc.store.update(run.run_id, done=base + d), 10.0),
        on_record=keep,
    )
    by_id = {**saved, **done}
    save(by_id)
    return [by_id[r.question_id] for r in questions]


def _finish(run: RunRecord, bundle: dict[str, Any], work: Path, svc: FinishServices) -> None:
    s, opts = svc.settings, svc.options
    dataset = opts.dataset(run.dataset)
    if dataset is None:
        raise RuntimeError(f"dataset {run.dataset} is not offered")
    loaded = load_hosted(svc.blobs, s.datasets_bucket, run.dataset, dataset.hash)
    original = _original(run, loaded, svc) if run.rejudge_of else None
    if original:
        ref = None
        raw = {name: _published(svc, original, name) for name in RECORD_FILES}
    else:
        ref = connector_ref(run.connector)
        if is_custom_endpoint(run.endpoint, ref.spec.allowed_domains):
            raise RunError(
                "The connector used an undeclared endpoint, so the run was not published."
            )
        raw = {
            name: svc.blobs.get(s.staging_bucket, f"{run.run_id}/{name}") for name in RECORD_FILES
        }

    secrets: list[str] = bundle["redact"]
    log.info("checking the retrieval output")
    try:
        questions = {
            k: clean_questions(
                raw[questions_file(k)], loaded, opts.budget_tokens, secrets, questions_file(k)
            )
            for k in TOP_K
        }
        conversations = clean_conversations(raw[CONVERSATIONS_FILE], loaded, secrets)
    except InvalidOutputError as exc:
        raise RunError(f"The retrieval output was invalid: {exc}") from exc

    if svc.http:
        wait_for_hosts(svc.http, [JUDGE_HOSTS[run.judge_route]])
    setup = svc.judge_setup(run.judge_route, bundle["judge_key"])
    earlier = {k: _saved_verdicts(svc, run.run_id, k, questions[k]) for k in TOP_K}
    total = sum(len(q) for q in questions.values())
    svc.store.update(
        run.run_id,
        done=sum(len(saved) for saved, _ in earlier.values()),
        total=total,
        job_started=True,
    )
    judged: dict[int, list[QuestionRecord]] = {}
    for i, k in enumerate(TOP_K):
        # Progress counts every size: the sizes judged above in full, the rest as saved.
        base = sum(len(questions[j]) for j in TOP_K[:i])
        base += sum(len(earlier[j][0]) for j in TOP_K[i:])
        judged[k] = _judge_size(run, k, questions[k], *earlier[k], loaded, setup, svc, base)
    failures = sum(1 for records in judged.values() for r in records if r.judge_error)
    if failures:
        raise JudgeIncompleteError(
            f"The judge failed on {failures} of {total} answers, so nothing was "
            "published yet. Check the judge key's credit and rate limits, then resume: "
            "the answers already judged are kept."
        )

    for k in TOP_K:
        (work / questions_file(k)).write_text(
            "".join(r.model_dump_json() + "\n" for r in judged[k]), encoding="utf-8"
        )
    (work / CONVERSATIONS_FILE).write_text(
        "".join(c.model_dump_json() + "\n" for c in conversations), encoding="utf-8"
    )
    tool = ToolInfo(version=__version__, commit=s.git_commit, image_digest=s.image_digest)
    if original:
        # The contexts, connector, affiliation, resumes and history are the earlier run's.
        result = build_result(
            work,
            run_id=run.run_id,
            started=run.created_at,
            dataset=loaded,
            connector=None,
            settings=None,
            judge_setup=setup,
            options=RunOptions(
                budget_tokens=opts.budget_tokens,
                concurrency=original.config.concurrency,
                judge_concurrency=JUDGE_CONCURRENCY,
                affiliation=original.affiliation,
                resumes=original.resumes,
                history=original.history,
            ),
            mode="hosted",
            tool=tool,
            rejudged_from=RejudgedFrom(run_id=original.run_id, season=original.judge.season),
            connector_info=original.connector,
        )
    else:
        assert ref is not None
        result = build_result(
            work,
            run_id=run.run_id,
            started=run.created_at,
            dataset=loaded,
            connector=ref,
            settings={**ref.spec.settings, **ref.spec.preset(run.preset)},
            judge_setup=setup,
            options=RunOptions(
                budget_tokens=opts.budget_tokens,
                concurrency=ref.spec.concurrency(CONCURRENCY),
                judge_concurrency=JUDGE_CONCURRENCY,
                affiliation=Affiliation(**run.affiliation),
                resumes=run.resumes,
                history=run.history,
            ),
            endpoint=run.endpoint,
            mode="hosted",
            tool=tool,
        )
    if not result.judge.official or result.dataset.selection or result.connector.custom_endpoint:
        raise RuntimeError("result is not publishable (judge, dataset or endpoint)")

    if not svc.store.transition(run.run_id, "judging", state="publishing"):
        raise RunEndedError
    log.info("publishing run %s", run.run_id)
    envelope = sign_result(
        json.loads(result.model_dump_json()),
        load_private_key(s.secret("signing-key")),
        s.artifact_url_prefix,
    )
    for artifact in result.artifacts:
        svc.publisher.upload_artifact(
            f"runs/{run.run_id}/{artifact.name}", (work / artifact.name).read_bytes()
        )
    message = (
        f"Add re-judged run {run.run_id} (from {run.rejudge_of})"
        if run.rejudge_of
        else f"Add hosted run {run.run_id}"
    )
    svc.publisher.commit_result(f"runs/{run.run_id}.json", envelope_text(envelope), message)
    svc.store.update(run.run_id, state="done", result_url=f"{s.site_url}/runs/{run.run_id}/")
    if run.rejudge_of:
        svc.store.set_rejudge(run, "done")
