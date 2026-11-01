# How scoring works

## Pipeline

For each conversation in the dataset:

1. `reset(namespace)` — a fresh, empty namespace in the memory system.
2. `ingest(namespace, session)` for each session, in time order. Ingestion is timed until the
   data is **searchable**: connectors for asynchronous systems poll until processing ends.
3. For each question, twice: `retrieve(namespace, question, asked_at, budget_tokens, top_k)`
   with `top_k` 10 and 30, each timed. These are the two result sizes, **Top 10** and
   **Top 30**, scored separately from the same ingested data
   ([ADR 0018](decisions/0018-top-10-and-top-30.md)). The order of the two calls alternates
   from question to question.
4. The returned context is counted with one tokenizer (`o200k_base`) and **hard-truncated**
   to the run's token budget (default 8,192). Truncation is recorded per question.
5. The judge reads the question, gold answer and the (truncated) context and returns
   `{"verdict", "reason", "stale", "gold_suspect"}`.

There is no answering model: the question is whether the context **itself** lets a reader
reach the gold answer without guessing.

## Judge rules (prompt v3)

The prompt is in [`src/memjudge/judge/prompts/v3.md`](../src/memjudge/judge/prompts/v3.md);
its SHA-256 is recorded in every result.

| Question kind | Correct when |
| --- | --- |
| standard | The specific gold fact is present or directly derivable (e.g. date arithmetic). On-topic but different/vaguer values are **incorrect**. Unresolved contradictions are incorrect. |
| abstention | The context does **not** answer the question as asked. Empty or unrelated context is correct, and so is a real fact that clearly concerns a different person, thing, place, time or activity than the question names, even if it contains the distractor answer (e.g. LoCoMo's adversarial answer). Attributing the distractor to the one asked about, or mixing them up so that one cannot tell whom a fact belongs to, is incorrect. |
| knowledge_update | As standard, and `stale` is set if an outdated value for the fact appears. |

Facts about the conversation must come from the context. Well-established general
knowledge and common-sense inferences may link those facts to the gold answer (e.g. which
country a city is in); specialist knowledge or guesses may not. A gold answer that is
itself an inference ("likely …") is met when the context holds the facts it rests on. Some
benchmark questions need such a link by design; prompt v1 banned all outside knowledge and
so failed contexts that held every conversation fact the answer needed.

`gold_suspect` lets the judge flag a question whose gold answer looks wrong given the
context. Flagged questions are **kept and scored**, and listed for human review.

The judge is pinned per season in [`seasons.json`](../src/memjudge/judge/seasons.json) (route,
model, reasoning effort, max output, prompt version and hash); the last entry is the current
season ([seasons](seasons.md)). Sampling parameters are left at provider defaults because
several current models reject them; reasoning effort is pinned instead. Runs with any other
judge or prompt are marked `official: false`.

## Metrics

All metrics are reported overall and per category.

| Metric | Definition |
| --- | --- |
| Accuracy | correct / all questions. Retrieval or judge failures count as **not correct**. |
| 95% CI | Cluster bootstrap: 2,000 resamples of whole conversations (seed 0). Questions about one conversation are correlated, so this is wider — and more honest — than resampling questions. With fewer than 5 conversations (e.g. a small subset) questions are resampled instead. |
| Context tokens | Mean and p95 of tokens judged per query (after the budget). |
| Efficiency | Accuracy (%) per 1,000 mean context tokens. |
| Retrieval latency | p50 / p95 seconds per `retrieve` call. |
| Ingestion time | Total seconds, seconds per turn, per-session p50/p95 — until searchable. Per-session runs add up each conversation's time; batch runs record the upload's wall-clock time and are never ranked against per-session runs ([ADR 0017](decisions/0017-ingestion-time-by-path.md)). |
| Cost | Ingestion, retrieval and judge USD, total and per 1,000 questions. API-reported cost first, then the versioned price table (judge only); otherwise **not reported**. The total is only shown when every part is known. Judge cost includes retried calls (unparseable replies, failed attempts). |
| Abstention accuracy | Accuracy on abstention questions. |
| Stale-fact rate | Share of judged knowledge-update questions whose context contains an outdated value. |
| Evidence recall | Share of evidence turns returned, **only** for systems that report which turns an item came from. Others show "not reported" rather than a misleading text-match estimate. |
| Error rate | Failed or timed-out calls (ingest + retrieve + judge) / all calls. |
| Truncation rate | Share of queries whose context exceeded the budget. |
| Flagged questions | Count and ids of `gold_suspect` questions. |

Conversations are processed in parallel (4 at a time, or fewer when a connector's
`max_concurrency` caps it to fit a vendor's free plan). No metric depends on this: latency
and per-session ingestion time are measured per call and per session, not as the run's
total time ([ADR 0013](decisions/0013-connector-concurrency-cap.md)).

Each result records the models the system ran (`connector.models`), sent explicitly and
checked where the system reports them, or `vendor` when the vendor picks them and does not
say which ([ADR 0014](decisions/0014-explicit-models.md)). Every system that lets you
choose its LLM runs GPT-6 Luna, so the memory design is compared, not the model; embedding
models are each system's own ([fairness rules](publishing-results.md#fairness-rules)).

## Judge calibration

Before a judge is pinned for a season it is measured with `memjudge calibrate` on
~200 hand-labelled examples covering long contexts, temporal reasoning, abstention and
knowledge updates, including **adversarial "plausible but wrong" contexts** (right topic,
wrong value; vague; contradictory). Candidates are compared on the same examples:

```bash
memjudge calibrate labels.jsonl --judge anthropic:<model> --judge openai:<model> \
  --judge google:<model> -o calibration.json
```

The report gives agreement, false-accept rate, false-reject rate (overall and per tag) and
cost for each candidate. The selection rule, including the tie margin and the
conflict-of-interest tie-break, is fixed in
[ADR 0008](decisions/0008-judge-selection-and-cross-check.md). The full calibration table
is published with the season.

## Cross-judge check

Because a judge may favour text written by its own model family, every season a seeded
sample of each published run is re-judged with a second judge from a different vendor:

```bash
memjudge crosscheck runs/<run-a> runs/<run-b> --judge openai:<model> --sample 300
```

For each run it reports the share of sampled questions where the two judges disagree and
the accuracy under each judge. What matters is the comparison across systems: if one
system's accuracy moves much more than the others', its score depends on the judge.

Label format (JSON lines):

```json
{"id": "c1", "question": "When did Ana move?", "gold_answer": "March 2024", "kind": "standard",
 "context": "Ana moved in 2023.", "label": "incorrect", "tags": ["plausible-wrong"]}
```

## Comparing results

Results are only comparable within the same dataset hash, judge (season) and token budget.
Partial runs (`--conversation`, `--category`, `--smallest`, `--sample`) record the selection
and the conversation ids in the result, and are not comparable to full runs. Runs against
a custom endpoint (`custom_endpoint: true`: localhost, staging, self-hosted) test a different
deployment from the vendor's hosted product and are not comparable to it.
