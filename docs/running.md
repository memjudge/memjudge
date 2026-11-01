# Running benchmarks

## Install

```bash
pip install "memjudge[mem0,zep]"     # extras install vendor SDKs; "memjudge[all]" for every one
```

From a checkout: `pip install -e ".[dev,all]"`.

## Keys

Set keys as environment variables or put `KEY=value` lines in a file passed with
`--env-file`. Keys are never written to run folders or logs.

| Purpose | Variable |
| --- | --- |
| Judge (direct) | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY` |
| Judge (OpenRouter) | `OPENROUTER_API_KEY` |
| Mem0 Platform | `MEM0_API_KEY` |
| Mem0 open source | `MEM0_OSS_OPENAI_API_KEY` |
| Zep | `ZEP_API_KEY` |
| Zep Graphiti open source | `GRAPHITI_OPENAI_API_KEY` |
| Supermemory | `SUPERMEMORY_API_KEY` |
| Letta | `LETTA_API_KEY` |
| Recalld | `RECALLD_API_KEY` |

Without `--judge`, MemJudge uses the pinned judge on whichever route you have a key for.

## Run

```bash
memjudge estimate --dataset locomo                     # typical judge cost and upper bound
memjudge run --dataset locomo --connector mem0/platform \
  --smallest 1 --affiliated no                         # quick: the smallest conversation
memjudge run --dataset longmemeval-s --connector zep/cloud --budget 4096
```

### Start small, then widen

Full runs cost real money (judge calls, plus the memory system's own usage). While setting
up or tuning, run one small conversation, check the report, then widen:

```bash
memjudge conversations locomo                                   # sizes, smallest first
memjudge run --dataset locomo --connector recalld/cloud --conversation conv-30
memjudge run --dataset locomo --connector recalld/cloud --conversation conv-30 --conversation conv-26
memjudge run --dataset longmemeval-s --connector recalld/cloud --smallest 5
memjudge run --dataset longmemeval-s --connector recalld/cloud --category knowledge-update --sample 20
memjudge run --dataset locomo --connector recalld/cloud        # full run once it looks right
```

`memjudge estimate` accepts the same options, so you can price a selection first.

### When the memory system fails: stop and resume

If an ingestion fails, or 5 retrievals fail in a row, the run stops instead of recording
errors for the rest of the dataset (the service is down, out of credit, or the key is
wrong). A single failed retrieval is recorded as an error and the run goes on. Completed
conversations are saved; fix the problem, then continue:

```bash
memjudge run --resume runs/<id>
```

The resume wipes and redoes the conversations that were not complete, and skips the
others. The result records how many times the run was resumed (`resumes`). Hosted runs
work the same way: their owner can resume a stopped run from the form within 48 hours,
at most 3 times (ADR 0011). A hosted result also records when the run stopped, why, and
when it was resumed (`history`), and the run's owner can read its job log on the form
(ADR 0016). Local runs log with `--verbose`.

### Rerun retrieval without re-ingesting

Ingestion is usually the slow, expensive part. When you are changing retrieval only
(connector settings, or the memory system's search side), ingest once and keep the data:

```bash
memjudge run --dataset locomo --connector recalld/cloud --conversation conv-30 --keep-data
# ...change retrieval, then rerun only retrieval and judging against the kept data:
memjudge run --dataset locomo --connector recalld/cloud --conversation conv-30 \
  --reuse-ingestion runs/<first-run>
memjudge report runs/<first-run> runs/<second-run>      # compare side by side
memjudge cleanup runs/<first-run>                        # delete the kept data when done
```

- The reusing run must use the same connector, dataset version and endpoint, and only
  conversations the first run ingested successfully. Connector code and settings may differ.
- Its result and report say "ingestion reused from <run>"; its ingestion figures are the
  first run's. Such runs are local-only: published runs always ingest fresh.
- In-process connectors (BM25) keep nothing between runs; just run them normally.
- If a change might affect how data is stored, do a fresh run before trusting the numbers.

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--budget` | 8192 | Hard token budget per query |
| `--conversation ID` | all | Only this conversation (repeatable); ids from `memjudge conversations` |
| `--category NAME` | all | Only questions in this category (repeatable) |
| `--smallest N` | — | Only the N conversations with the fewest turns |
| `--sample N --seed S` | — | Only N conversations chosen at random |
| `--concurrency` | 4 | Conversations processed in parallel (lowered to the connector's `max_concurrency`, if set) |
| `--judge-concurrency` | 8 | Parallel judge calls |
| `--setting key=value` | — | Override a connector setting (recorded in the result) |
| `--setting base_url=URL` | vendor API | Call another server, e.g. `http://localhost:8080` (marked as a custom endpoint) |
| `--judge route:model` | pinned | `anthropic:…`, `openai:…`, `google:…`, `openrouter:vendor/model@provider` |
| `--resume runs/<id>` | — | Continue a stopped or interrupted run; completed conversations are skipped |
| `--keep-data` | off | Leave ingested data in the memory system for later `--reuse-ingestion` |
| `--reuse-ingestion runs/<id>` | — | Skip ingestion; query the data a `--keep-data` run kept |

## Outputs

`runs/<run-id>/` contains:

- `run.json` — how the run was started (used by `--resume` and `rejudge`)
- `conversations.jsonl` — ingestion timings, usage and errors per conversation
- `questions-top10.jsonl`, `questions-top30.jsonl` — per question and result size (ADR
  0018): question text, gold answer, context, tokens, latency, verdict, judge reason, raw
  output
- `result.json` — metrics per result size (`top10`, `top30`) and provenance (the signed
  payload on hosted runs)
- `report-top10.html`, `report-top30.html` — charts, per-category table, per-question
  verdicts

## Re-judging

```bash
memjudge rejudge runs/<id> --judge openrouter:google/gemini-x@google-ai-studio
```

copies the stored contexts into a new run folder and judges them again. Memory systems are
not called. This is how a new leaderboard season re-scores old runs.
