"""``memjudge`` command-line interface."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table

from memjudge import __version__
from memjudge.connectors.base import BaseConnector, is_custom_endpoint, public_url
from memjudge.connectors.loader import (
    LoadedConnector,
    list_connectors,
    load_connector,
    load_spec,
)
from memjudge.datasets.loader import LoadedDataset, apply_selection, load_dataset
from memjudge.datasets.registry import DATASETS
from memjudge.datasets.selection import Selection, conversation_sizes
from memjudge.harness import (
    BATCH_FILE,
    CONVERSATIONS_FILE,
    RESULT_FILE,
    ReusedIngestion,
    Runner,
    RunOptions,
    RunStoppedError,
    build_result,
    judge_records,
    new_run_id,
    public_settings,
)
from memjudge.judge.client import JudgeModel
from memjudge.judge.judge import load_prompt
from memjudge.records import (
    BatchRecord,
    ConversationRecord,
    QuestionRecord,
    latest_by,
    read_jsonl,
)
from memjudge.result import Affiliation, RejudgedFrom, RunResult
from memjudge.sizes import TOP_K, questions_file, size_key

app = typer.Typer(
    help="MemJudge: open, vendor-neutral benchmark for AI memory systems.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)
RUN_META = "run.json"
# Judge tokenizers count more tokens than o200k for the same text (Claude: 1.40-1.48x).
JUDGE_TOKENS_PER_O200K = 1.5
# Context size per result size for the typical judge cost: above every memory system tested
# so far at that size.
TYPICAL_CONTEXT_TOKENS = {10: 700, 30: 2000}


def _env(env_file: Path | None) -> dict[str, str]:
    """Process environment plus an optional KEY=VALUE file. Values are never printed."""
    env = dict(os.environ)
    if env_file:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return env


def _settings(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise typer.BadParameter(f"--setting must be key=value, got {pair!r}")
        out[key] = yaml.safe_load(value)
    return out


def _affiliation(value: str | None, system: str | None) -> Affiliation:
    if value is None:
        return Affiliation(affiliated=None)
    if value not in ("yes", "no"):
        raise typer.BadParameter("--affiliated must be yes or no")
    return Affiliation(affiliated=value == "yes", system=system)


# Options shared by every command that loads part of a dataset.
ConversationOpt = Annotated[
    list[str],
    typer.Option(
        "--conversation",
        help="Only this conversation id (repeatable). See `memjudge conversations`.",
    ),
]
CategoryOpt = Annotated[
    list[str], typer.Option("--category", help="Only questions in this category (repeatable).")
]
SmallestOpt = Annotated[
    int | None, typer.Option(help="Only the N conversations with the fewest turns.")
]
SampleOpt = Annotated[int | None, typer.Option(help="Only N conversations chosen at random.")]
SeedOpt = Annotated[int, typer.Option(help="Seed for --sample.")]


def _selection(
    conversation: list[str],
    category: list[str],
    smallest: int | None,
    sample: int | None,
    seed: int,
) -> Selection:
    try:
        return Selection(
            conversation_ids=tuple(conversation),
            categories=tuple(category),
            smallest=smallest,
            sample=sample,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _load_dataset(spec: str, selection: Selection | None = None) -> LoadedDataset:
    with console.status(f"Loading dataset {spec}…"):
        loaded = load_dataset(spec)
    if selection is None:
        return loaded
    try:
        return apply_selection(loaded, selection)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


@app.command()
def run(
    dataset: Annotated[
        str, typer.Option(help="Curated dataset name or path to canonical JSON.")
    ] = "",
    connector: Annotated[str, typer.Option(help="Connector id, e.g. baselines/bm25.")] = "",
    judge_spec: Annotated[
        str | None,
        typer.Option("--judge", help="Override judge: route:model or openrouter:model@provider."),
    ] = None,
    budget: Annotated[int, typer.Option(help="Hard context token budget per query.")] = 8192,
    conversation: ConversationOpt = [],  # noqa: B006
    category: CategoryOpt = [],  # noqa: B006
    smallest: SmallestOpt = None,
    sample: SampleOpt = None,
    seed: SeedOpt = 0,
    concurrency: Annotated[int, typer.Option(help="Conversations processed in parallel.")] = 4,
    judge_concurrency: Annotated[int, typer.Option(help="Parallel judge calls.")] = 8,
    setting: Annotated[list[str], typer.Option(help="Connector setting override key=value.")] = [],  # noqa: B006
    affiliated: Annotated[
        str | None,
        typer.Option(help="yes/no: are you affiliated with the memory system you are testing?"),
    ] = None,
    affiliated_system: Annotated[
        str | None, typer.Option(help="Which system, if affiliated.")
    ] = None,
    env_file: Annotated[Path | None, typer.Option(help="File with KEY=VALUE API keys.")] = None,
    out: Annotated[Path, typer.Option(help="Directory for run folders.")] = Path("runs"),
    resume: Annotated[Path | None, typer.Option(help="Resume an interrupted run folder.")] = None,
    keep_data: Annotated[
        bool,
        typer.Option(
            help="Keep ingested data in the memory system so later runs can --reuse-ingestion. "
            "Delete it with `memjudge cleanup`."
        ),
    ] = False,
    reuse_ingestion: Annotated[
        Path | None,
        typer.Option(
            help="Skip ingestion and query the data kept by this earlier run (--keep-data). "
            "Reruns only retrieval and judging; local-only."
        ),
    ] = None,
    report: Annotated[
        bool, typer.Option(help="Write report-top10.html and report-top30.html.")
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Benchmark one connector on one dataset. Local runs are never published.

    Iterate cheaply first: --smallest 1 or --conversation <id>, then widen.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    env = _env(env_file)
    from memjudge.judge.setup import make_judge_setup, resolve_judge

    if resume:
        meta = json.loads((resume / RUN_META).read_text(encoding="utf-8"))
        run_dir, run_id = resume, meta["run_id"]
        started = datetime.fromisoformat(meta["started_at"])
        dataset, connector = meta["dataset"], meta["connector"]
        selection = Selection(**meta["selection"])
        overrides = meta["settings"]
        options = RunOptions(
            **meta["options"],
            affiliation=Affiliation(**meta["affiliation"]),
            resumes=int(meta.get("resumes", 0)) + 1,
        )
        judge_model = JudgeModel(**meta["judge"])
        keep_data = bool(meta.get("keep_data"))
        reused_from = meta.get("reuse_ingestion_from")
        reuse_ingestion = Path(reused_from) if reused_from else None
    else:
        if not dataset or not connector:
            raise typer.BadParameter("--dataset and --connector are required")
        run_id = new_run_id()
        run_dir = out / run_id
        started = datetime.now(UTC)
        overrides = _settings(setting)
        options = RunOptions(
            budget_tokens=budget,
            concurrency=concurrency,
            judge_concurrency=judge_concurrency,
            affiliation=_affiliation(affiliated, affiliated_system),
        )
        if reuse_ingestion and keep_data:
            raise typer.BadParameter("--reuse-ingestion runs never delete data; drop --keep-data")
        judge_model = resolve_judge(judge_spec, env)
        selection = _selection(conversation, category, smallest, sample, seed)

    loaded = _load_dataset(dataset, selection)
    conn = load_connector(connector)
    instance = conn.instantiate(env, overrides)
    setup = make_judge_setup(judge_model, env)
    options.cleanup = not keep_data
    capped = conn.spec.concurrency(options.concurrency)
    if capped < options.concurrency:
        console.print(f"{connector} runs at most {capped} conversations at once (ADR 0013).")
        options.concurrency = capped
    reuse = _reused_ingestion(reuse_ingestion, loaded, conn, instance) if reuse_ingestion else None

    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": run_id,
            "started_at": started.isoformat(),
            "dataset": dataset,
            "connector": connector,
            "selection": selection.model_dump(mode="json"),
            "settings": public_settings(overrides),
            "options": {
                "budget_tokens": options.budget_tokens,
                "concurrency": options.concurrency,
                "judge_concurrency": options.judge_concurrency,
            },
            "affiliation": options.affiliation.model_dump(),
            "judge": judge_model.model_dump(),
            "keep_data": keep_data,
            "reuse_ingestion_from": str(reuse_ingestion) if reuse_ingestion else None,
        }
        (run_dir / RUN_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        meta["resumes"] = options.resumes
        (run_dir / RUN_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    console.print(
        f"[bold]MemJudge {__version__}[/] · {loaded.label} · {conn.spec.id} · "
        f"judge {judge_model.spec}"
        + ("" if setup.info([]).official else " [yellow](unofficial judge)[/]")
    )
    if is_custom_endpoint(instance.endpoint, conn.spec.allowed_domains):
        console.print(
            f"[yellow]Custom endpoint {public_url(instance.endpoint or '')}: results are marked "
            "as such and are not comparable to the vendor's hosted product.[/]"
        )
    if loaded.selection:
        console.print(
            f"Partial run: {loaded.selection.describe()} → "
            f"{len(loaded.dataset.conversations)} conversations, "
            f"{len(loaded.dataset.questions)} questions"
        )
    if reuse:
        console.print(f"Reusing ingestion from run {reuse.run_id}: only retrieval and judging run.")
    if keep_data:
        console.print(f"Keeping ingested data. Delete it later with: memjudge cleanup {run_dir}")
    runner = Runner(loaded, conn, instance, setup, run_dir, options, run_id=run_id, reuse=reuse)
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        ingest = None if reuse else progress.add_task("ingested sessions", total=None)
        # one per question and size (ADR 0018)
        task = progress.add_task("answers", total=len(loaded.dataset.questions) * len(TOP_K))

        def on_ingest(done: int, total: int) -> None:
            if ingest is not None:
                progress.update(
                    ingest, description="ingested sessions", completed=done, total=total
                )

        def on_batch(done: int, total: int) -> None:
            if ingest is not None:
                progress.update(
                    ingest, description="ingested turns (batch)", completed=done, total=total
                )

        try:
            result = runner.run(
                lambda done, total: progress.update(task, completed=done, total=total),
                started=started,
                ingest_progress=on_ingest,
                batch_progress=on_batch,
            )
        except RunStoppedError as exc:
            progress.stop()
            console.print(f"[red]Stopped:[/] {exc}")
            console.print(
                "Completed conversations are saved. Fix the problem (service, credit, key), "
                f"then continue with: memjudge run --resume {run_dir}"
            )
            raise typer.Exit(1) from None
        except KeyboardInterrupt:
            progress.stop()
            console.print(f"[yellow]Interrupted.[/] Continue with: memjudge run --resume {run_dir}")
            # Worker threads still finishing a call would keep the process alive;
            # every finished record is already on disk.
            os._exit(130)
    _print_summary(result)
    if report:
        for path in _write_reports(run_dir, [result], loaded):
            console.print(f"Report: {path}")
    console.print(f"Result: {run_dir / RESULT_FILE}")


def _reused_ingestion(
    old_dir: Path, loaded: LoadedDataset, conn: LoadedConnector, instance: BaseConnector
) -> ReusedIngestion:
    """Check an earlier run's kept data can serve this run, and load its records."""

    def fail(msg: str) -> typer.BadParameter:
        return typer.BadParameter(f"cannot reuse ingestion from {old_dir}: {msg}")

    if instance.endpoint is None:
        raise fail(f"{conn.spec.id} runs in-process and keeps nothing; run it normally")
    meta_path, result_path = old_dir / RUN_META, old_dir / RESULT_FILE
    if not meta_path.exists() or not result_path.exists():
        raise fail("not a finished run folder (resume it first)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old = RunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    if meta.get("reuse_ingestion_from"):
        raise fail("it reused data itself; point at the run that ingested it")
    if not meta.get("keep_data"):
        raise fail("it deleted its data when it finished; run it again with --keep-data")
    if meta.get("data_deleted"):
        raise fail("its data was deleted with `memjudge cleanup`")
    if old.connector.id != conn.spec.id:
        raise fail(f"it used connector {old.connector.id}, not {conn.spec.id}")
    if old.dataset.hash != loaded.hash:
        raise fail("it used a different dataset version")
    if old.connector.endpoint != public_url(instance.endpoint):
        raise fail(f"its data is at {old.connector.endpoint}, not {public_url(instance.endpoint)}")
    records = latest_by(
        read_jsonl(old_dir / CONVERSATIONS_FILE, ConversationRecord), "conversation_id"
    )
    unusable = [
        c.id
        for c in loaded.dataset.conversations
        if c.id not in records or records[c.id].error or not records[c.id].namespace
    ]
    if unusable:
        raise fail(f"no usable ingested data for conversations {unusable}")
    batch_path = old_dir / BATCH_FILE
    batch = (
        BatchRecord.model_validate_json(batch_path.read_text(encoding="utf-8"))
        if batch_path.exists()
        else None
    )
    return ReusedIngestion(
        run_id=old.run_id,
        connector_code_hash=old.connector.code_hash,
        records=records,
        batch=batch,
    )


def _print_summary(result: RunResult) -> None:
    for top_k in TOP_K:
        m = result.metrics[size_key(top_k)]
        o = m.overall
        table = Table(
            title=f"{result.connector.id} on {result.dataset.name} {result.dataset.version}, "
            f"Top {top_k}"
        )
        table.add_column("Category")
        table.add_column("n", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("Tokens", justify="right")
        for name, c in [*m.by_category.items(), ("overall", o)]:
            table.add_row(
                name,
                str(c.questions),
                f"{c.accuracy:.1%}",
                f"{c.accuracy_ci95[0]:.1%}–{c.accuracy_ci95[1]:.1%}",
                f"{c.context_tokens_mean:,.0f}",
            )
        console.print(table)
        err = m.error_rate.value
        console.print(
            f"Latency p50 {o.retrieval_latency_p50_s or 0:.2f}s · "
            f"errors {0 if err is None else err:.1%} · flagged {o.flagged}"
        )
    ing = next(iter(result.metrics.values())).ingestion  # the same for every size
    note = " (batch, wall-clock)" if ing.mode == "batch" else " (all conversations added up)"
    console.print(f"Ingestion {ing.total_s:.1f}s{note}")


def _questions(run_dir: Path, top_k: int) -> list[QuestionRecord]:
    """A run folder's records of one result size, the latest per question."""
    path = run_dir / questions_file(top_k)
    return list(latest_by(read_jsonl(path, QuestionRecord), "question_id").values())


def _write_reports(
    run_dir: Path, runs: list[RunResult], dataset: LoadedDataset | None
) -> list[Path]:
    """One report per result size, from ``run_dir``'s records: ``report-top10.html``..."""
    from memjudge.report.render import render_report

    paths = []
    for top_k in TOP_K:
        path = run_dir / f"report-{size_key(top_k)}.html"
        html = render_report(runs, _questions(run_dir, top_k), dataset, top_k)
        path.write_text(html, encoding="utf-8")
        paths.append(path)
    return paths


def _dataset_for(run_dir: Path) -> LoadedDataset | None:
    meta_path = run_dir / RUN_META
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return _load_dataset(meta["dataset"], Selection(**meta["selection"]))


@app.command()
def report(
    run_dirs: Annotated[
        list[Path], typer.Argument(help="One or more run folders; the first is detailed.")
    ],
) -> None:
    """Render HTML reports, one per result size, into the first run folder. Pass 2–4 runs
    to compare them side by side."""
    if not 1 <= len(run_dirs) <= 4:
        raise typer.BadParameter("pass between 1 and 4 run folders")
    runs = [
        RunResult.model_validate_json((d / RESULT_FILE).read_text(encoding="utf-8"))
        for d in run_dirs
    ]
    hashes = {r.dataset.hash for r in runs}
    if len(hashes) > 1:
        console.print(
            "[yellow]Warning: runs use different dataset versions; they are not comparable.[/]"
        )
    for path in _write_reports(run_dirs[0], runs, _dataset_for(run_dirs[0])):
        console.print(f"Report: {path}")


@app.command()
def rejudge(
    run_dir: Annotated[Path, typer.Argument(help="Existing run folder.")],
    judge_spec: Annotated[str | None, typer.Option("--judge")] = None,
    out: Annotated[
        Path | None, typer.Option(help="New run folder (default: <run>-rejudged-<id>).")
    ] = None,
    judge_concurrency: Annotated[int, typer.Option()] = 8,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Re-score stored contexts with a (new) judge without re-running the memory system."""
    from memjudge.judge.setup import make_judge_setup, resolve_judge

    env = _env(env_file)
    meta = json.loads((run_dir / RUN_META).read_text(encoding="utf-8"))
    loaded = _load_dataset(meta["dataset"], Selection(**meta["selection"]))
    judge_model = resolve_judge(judge_spec, env)
    setup = make_judge_setup(judge_model, env)
    new_id = new_run_id()
    target = out or run_dir.with_name(f"{run_dir.name}-rejudged-{new_id[-6:]}")
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy(run_dir / CONVERSATIONS_FILE, target / CONVERSATIONS_FILE)
    if (run_dir / BATCH_FILE).exists():  # the run's batch ingestion (ADR 0010)
        shutil.copy(run_dir / BATCH_FILE, target / BATCH_FILE)
    new_meta = {
        **meta,
        "run_id": new_id,
        "judge": judge_model.model_dump(),
        "rejudged_from": meta["run_id"],
        "keep_data": False,  # the data belongs to the original run
        "reuse_ingestion_from": None,
    }
    (target / RUN_META).write_text(json.dumps(new_meta, indent=2), encoding="utf-8")

    by_id = {q.id: q for q in loaded.dataset.questions}
    for top_k in TOP_K:
        # Older runs did not store the question text; judge_records fills it in from the dataset.
        judged = judge_records(_questions(run_dir, top_k), by_id, setup, judge_concurrency)
        with (target / questions_file(top_k)).open("w", encoding="utf-8") as f:
            for rec in judged:
                f.write(rec.model_dump_json() + "\n")

    original = RunResult.model_validate_json((run_dir / RESULT_FILE).read_text("utf-8"))
    options = RunOptions(
        **meta["options"],
        affiliation=Affiliation(**meta["affiliation"]),
        resumes=original.resumes,  # the contexts are that run's, resumes included
    )
    result = build_result(
        target,
        run_id=new_id,
        started=datetime.now(UTC),
        dataset=loaded,
        connector=None,
        settings=None,
        judge_setup=setup,
        options=options,
        endpoint=original.connector.endpoint,
        reused=original.reused_ingestion,
        rejudged_from=RejudgedFrom(run_id=original.run_id, season=original.judge.season),
        # The contexts were retrieved by that run's connector code, not this checkout's.
        connector_info=original.connector,
    )
    _print_summary(result)
    _write_reports(target, [result], loaded)
    console.print(f"Re-judged run: {target}")


@app.command()
def estimate(
    dataset: Annotated[str, typer.Option()],
    budget: Annotated[int, typer.Option()] = 8192,
    judge_spec: Annotated[str | None, typer.Option("--judge")] = None,
    conversation: ConversationOpt = [],  # noqa: B006
    category: CategoryOpt = [],  # noqa: B006
    smallest: SmallestOpt = None,
    sample: SampleOpt = None,
    seed: SeedOpt = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON to stdout.")] = False,
) -> None:
    """Typical judge cost (700-token contexts at Top 10, 2,000 at Top 30) and upper bound (every
    context fills the budget). Both are estimates: the real cost depends on how much context the
    memory system returns.

    Every question is judged once per result size (ADR 0018). Takes the same selection
    options as `run`. Also shows what the memory system must ingest; its cost depends on the
    vendor and is not estimated.
    """
    from memjudge.judge.setup import pinned_judge
    from memjudge.pricing.cost import judge_model_key, load_price_table
    from memjudge.tokens import count_tokens

    loaded = _load_dataset(dataset, _selection(conversation, category, smallest, sample, seed))
    pinned = pinned_judge()
    jm = JudgeModel.parse(judge_spec or pinned.routes[pinned.default_route], effort=pinned.effort)
    prompt = load_prompt()
    base = count_tokens(prompt.system) + count_tokens(prompt.user_template)
    qs = loaded.dataset.questions
    sizes = len(TOP_K)
    fixed = sum(base + count_tokens(q.question + q.gold_answer) for q in qs) * sizes
    input_tokens = int((fixed + len(qs) * sizes * budget) * JUDGE_TOKENS_PER_O200K)
    typical_context = {k: min(budget, TYPICAL_CONTEXT_TOKENS[k]) for k in TOP_K}
    typical_input_tokens = int(
        (fixed + len(qs) * sum(typical_context.values())) * JUDGE_TOKENS_PER_O200K
    )
    output_tokens = len(qs) * sizes * 300  # verdict JSON plus low-effort reasoning, conservative
    convs = loaded.dataset.conversations
    turns = sum(c.turn_count() for c in convs)
    sessions = sum(len(c.sessions) for c in convs)
    turn_tokens = sum(count_tokens(t.text) for c in convs for s in c.sessions for t in s.turns)
    prices = load_price_table()
    model_key = judge_model_key(jm.route, jm.model, jm.upstream_provider)
    cost = prices.token_cost(model_key, input_tokens, output_tokens)
    typical = prices.token_cost(model_key, typical_input_tokens, output_tokens)
    if as_json:
        summary = {
            "dataset": loaded.label,
            "dataset_hash": loaded.hash,
            "budget_tokens": budget,
            "conversations": len(convs),
            "sessions": sessions,
            "turns": turns,
            "turn_tokens": turn_tokens,
            "questions": len(qs),
            "judge": jm.spec,
            "judge_input_tokens": input_tokens,
            "judge_output_tokens": output_tokens,
            "judge_cost_usd_typical": typical,
            "judge_cost_usd_upper_bound": cost,
        }
        typer.echo(json.dumps(summary, indent=2))
        return
    console.print(
        f"{loaded.label}: {len(loaded.dataset.conversations)} conversations, "
        f"{turns:,} turns, {len(qs):,} questions"
    )
    console.print(
        f"Judge {jm.spec}: up to {input_tokens:,} input + ~{output_tokens:,} output tokens"
    )
    contexts = ", ".join(f"{n:,} at Top {k}" for k, n in typical_context.items())
    console.print(
        f"Judge cost with typical contexts ({contexts} tokens): "
        f"{'not in price table' if typical is None else f'${typical:,.2f}'}"
    )
    console.print(
        f"Judge cost upper bound: {'not in price table' if cost is None else f'${cost:,.2f}'}"
    )
    console.print(
        f"Memory system: {sessions:,} sessions, {turns:,} write calls (one per turn), "
        f"{turn_tokens:,} turn tokens to ingest; {len(qs) * sizes:,} searches"
    )
    console.print("Memory-system cost depends on the vendor and is not estimated.")


@app.command()
def calibrate(
    labels: Annotated[Path, typer.Argument(help="Hand-labelled examples (JSON lines).")],
    judge_spec: Annotated[
        list[str],
        typer.Option("--judge", help="Candidate judge; repeat to compare several."),
    ] = [],  # noqa: B006
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure judge agreement, false-accept rate and cost on hand-labelled examples.

    Pass --judge several times to compare candidate judges on the same examples.
    """
    from memjudge.calibration import calibrate as run_calibration
    from memjudge.calibration import load_labels
    from memjudge.judge.setup import make_judge_setup, resolve_judge

    env = _env(env_file)
    examples = load_labels(labels)
    reports: list[dict[str, Any]] = []
    table = Table(title=f"Judge calibration on {len(examples)} examples")
    for col in ("Judge", "Agreement", "False accept", "False reject", "Errors", "Cost"):
        table.add_column(col, justify="left" if col == "Judge" else "right")
    specs: list[str | None] = [*judge_spec] or [None]
    for spec in specs:
        jm = resolve_judge(spec, env)
        setup = make_judge_setup(jm, env)
        with console.status(f"Judging with {jm.spec}…"):
            rep = run_calibration(examples, setup.client, setup.prompt, price=setup.price)
        reports.append({"judge": jm.spec, "effort": jm.effort, **rep.model_dump()})
        table.add_row(
            jm.spec,
            _pct(rep.agreement),
            _pct(rep.false_accept_rate),
            _pct(rep.false_reject_rate),
            str(rep.errors),
            "not reported" if rep.cost_usd is None else f"${rep.cost_usd:.4f}",
        )
    console.print(table)
    text = json.dumps({"prompt_hash": load_prompt().hash, "judges": reports}, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
    typer.echo(text)


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


@app.command()
def crosscheck(
    run_dirs: Annotated[list[Path], typer.Argument(help="Run folders to check.")],
    judge_spec: Annotated[
        str, typer.Option("--judge", help="Second judge, from a different vendor.")
    ],
    sample: Annotated[int, typer.Option(help="Questions sampled per run.")] = 300,
    seed: Annotated[int, typer.Option()] = 0,
    judge_concurrency: Annotated[int, typer.Option()] = 8,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Re-judge a seeded sample of each run with a second judge and report disagreement.

    Shows whether any system's score depends unusually on the choice of judge. Samples the
    Top 30 records, the larger contexts.
    """
    from memjudge.crosscheck import cross_check, sample_records
    from memjudge.judge.setup import make_judge_setup, resolve_judge

    env = _env(env_file)
    jm = resolve_judge(judge_spec, env)
    setup = make_judge_setup(jm, env)
    table = Table(title=f"Cross-judge check with {jm.spec}")
    for col in ("Connector", "n", "Disagree", "Acc. original", "Acc. second", "Delta"):
        table.add_column(col, justify="left" if col == "Connector" else "right")
    checks = []
    for run_dir in run_dirs:
        result = RunResult.model_validate_json((run_dir / RESULT_FILE).read_text("utf-8"))
        if (result.judge.route, result.judge.model) == (jm.route, jm.model):
            console.print(f"[yellow]{run_dir.name}: second judge equals the original judge.[/]")
        loaded = _dataset_for(run_dir)
        if loaded is None:
            raise typer.BadParameter(f"{run_dir} has no {RUN_META}")
        records = _questions(run_dir, max(TOP_K))
        with console.status(f"Re-judging a sample of {run_dir.name}…"):
            check = cross_check(
                sample_records(records, sample, seed),
                {q.id: q for q in loaded.dataset.questions},
                setup.client,
                setup.prompt,
                run_id=result.run_id,
                connector=result.connector.id,
                original_judge=f"{result.judge.route}:{result.judge.model}",
                second_judge=jm.spec,
                concurrency=judge_concurrency,
            )
        checks.append(check)
        table.add_row(
            check.connector,
            str(check.judged),
            _pct(check.disagreement_rate),
            _pct(check.accuracy_original),
            _pct(check.accuracy_second),
            _pct(check.accuracy_delta, signed=True),
        )
    console.print(table)
    summary = {
        "prompt_hash": setup.prompt.hash,
        "sample": sample,
        "seed": seed,
        "checks": [c.model_dump() for c in checks],
    }
    text = json.dumps(summary, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
    typer.echo(text)


@app.command()
def keygen(
    out: Annotated[Path, typer.Option(help="Directory for <key_id>.key and <key_id>.pub.")] = Path(
        "keys"
    ),
) -> None:
    """Generate an Ed25519 signing key pair. Keep the .key file secret; never commit it."""
    from memjudge.signing import generate_keypair

    priv, pub, kid = generate_keypair(out)
    console.print(f"Key id {kid}\nPrivate key: {priv} (secret)\nPublic key:  {pub}")


@app.command()
def sign(
    result_file: Annotated[Path, typer.Argument()],
    key: Annotated[Path, typer.Option(help="Ed25519 private key (PEM).")],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    artifact_url_prefix: Annotated[
        str | None,
        typer.Option(
            help="Where the artifacts were uploaded: sets each URL to <prefix><run_id>/<name>."
        ),
    ] = None,
) -> None:
    """Sign a result.json, writing a signed envelope (for the hosted runner)."""
    from memjudge.signing import load_private_key, sign_result

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    RunResult.model_validate(payload)
    envelope = sign_result(payload, load_private_key(key.read_bytes()), artifact_url_prefix)
    path = output or result_file.with_name("result.signed.json")
    path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Signed with key {envelope['signature']['key_id']}: {path}")


@app.command()
def verify(
    signed_file: Annotated[Path, typer.Argument()],
    keys: Annotated[
        Path | None, typer.Option(help="Folder of trusted .pub keys (default: bundled).")
    ] = None,
    artifacts: Annotated[
        Path | None, typer.Option(help="Folder with artifacts to check against their hashes.")
    ] = None,
) -> None:
    """Verify a signed result (and optionally its artifacts). Exit code 1 on failure."""
    from importlib import resources

    from memjudge.canonical import file_hash
    from memjudge.signing import VerificationError, load_public_keys, verify_envelope

    key_dir = keys or Path(str(resources.files("memjudge").joinpath("keys")))
    trusted = load_public_keys(key_dir)
    if not trusted:
        console.print(
            f"[red]INVALID[/]: no trusted .pub keys in {key_dir}; pass --keys with a folder "
            "of public keys."
        )
        raise typer.Exit(1)
    envelope = json.loads(signed_file.read_text(encoding="utf-8"))
    try:
        kid = verify_envelope(envelope, trusted)
        RunResult.model_validate(envelope["payload"])
    except (VerificationError, ValueError) as exc:
        console.print(f"[red]INVALID[/]: {exc}")
        raise typer.Exit(1) from None
    ok = True
    if artifacts:
        for art in envelope["payload"]["artifacts"]:
            path = artifacts / art["name"]
            actual = file_hash(path) if path.exists() else "missing"
            match = actual == art["sha256"]
            ok &= match
            console.print(f"{'ok ' if match else 'BAD'} {art['name']}")
    if not ok:
        console.print("[red]INVALID[/]: artifact hash mismatch")
        raise typer.Exit(1)
    console.print(f"[green]VALID[/] signature by key {kid}")


@app.command()
def cleanup(
    run_dir: Annotated[Path, typer.Argument(help="Run folder made with --keep-data.")],
    env_file: Annotated[Path | None, typer.Option(help="File with KEY=VALUE API keys.")] = None,
) -> None:
    """Delete the data a --keep-data run left in the memory system."""
    meta = json.loads((run_dir / RUN_META).read_text(encoding="utf-8"))
    if not meta.get("keep_data") or meta.get("data_deleted"):
        console.print("Nothing to delete: this run kept no data, or it was already deleted.")
        return
    conn = load_connector(meta["connector"])
    instance = conn.instantiate(_env(env_file), meta["settings"])
    records = latest_by(
        read_jsonl(run_dir / CONVERSATIONS_FILE, ConversationRecord), "conversation_id"
    )
    failed = 0
    try:
        for rec in records.values():
            if not rec.namespace or rec.reused_from:
                continue
            try:
                instance.restore_state(rec.namespace, rec.state)
                instance.reset(rec.namespace)
            except Exception as exc:
                failed += 1
                console.print(f"[red]{rec.conversation_id}: {exc.__class__.__name__}: {exc}[/]")
    finally:
        instance.close()
    if failed:
        console.print(f"[red]{failed} conversations could not be deleted; run cleanup again.[/]")
        raise typer.Exit(1)
    meta["data_deleted"] = True
    (run_dir / RUN_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.print(f"Deleted kept data for {len(records)} conversations.")


@app.command()
def conversations(
    dataset: Annotated[str, typer.Argument(help="Curated dataset name or canonical JSON path.")],
    category: CategoryOpt = [],  # noqa: B006
    limit: Annotated[int, typer.Option(help="Rows to show (0 = all).")] = 20,
) -> None:
    """List a dataset's conversations, smallest first, to choose what to run.

    Use an id with `run --conversation <id>`, or `run --smallest N`.
    """
    sel = Selection(categories=tuple(category)) if category else None
    loaded = _load_dataset(dataset, sel)
    rows = conversation_sizes(loaded.dataset)
    table = Table(title=f"{loaded.label}: {len(rows)} conversations, smallest first")
    for col in ("Conversation", "Sessions", "Turns", "~Tokens", "Questions"):
        table.add_column(col, justify="left" if col == "Conversation" else "right")
    for conv, questions, tokens in rows[: limit or None]:
        table.add_row(
            conv.id,
            str(len(conv.sessions)),
            f"{conv.turn_count():,}",
            f"{tokens:,}",
            str(questions),
        )
    console.print(table)
    if limit and len(rows) > limit:
        console.print(f"… {len(rows) - limit} more (use --limit 0 to show all)")


@app.command("datasets")
def list_datasets() -> None:
    """List curated datasets."""
    table = Table()
    for col in ("name", "version", "licence", "title"):
        table.add_column(col)
    for d in DATASETS.values():
        table.add_row(d.name, d.version, d.license, d.title)
    console.print(table)


@app.command("connectors")
def list_connectors_cmd() -> None:
    """List available connectors."""
    table = Table()
    for col in ("id", "target", "label", "keys"):
        table.add_column(col)
    for cid in list_connectors():
        spec = load_spec(cid)
        table.add_row(cid, spec.target, spec.label, ", ".join(spec.required_keys) or "-")
    console.print(table)


@app.command("validate-dataset")
def validate_dataset(path: Annotated[Path, typer.Argument()]) -> None:
    """Validate a canonical dataset JSON file and print its hash."""
    loaded = load_dataset(str(path))
    ds = loaded.dataset
    console.print(
        f"[green]valid[/] {ds.name} {ds.version}: "
        f"{len(ds.conversations)} conversations, {len(ds.questions)} questions"
    )
    typer.echo(loaded.hash)


@app.command()
def schema() -> None:
    """Print the JSON Schema of result.json."""
    typer.echo(json.dumps(RunResult.model_json_schema(), indent=2))


@app.callback(invoke_without_command=True)
def _main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    # Windows consoles and pipes default to a legacy code page that garbles symbols in
    # tables and messages; always write UTF-8.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def main() -> None:
    """Console entry point: a missing key or vendor SDK is an error line, not a traceback."""
    try:
        app()
    except (KeyError, ImportError) as exc:
        console.print(f"[red]Error:[/] {exc.args[0] if exc.args else exc}")
        raise SystemExit(2) from None
