# MemJudge

**An open, vendor-neutral benchmark for AI memory systems.**

MemJudge measures what a memory system (Mem0, Zep, Letta, Supermemory, …) actually
returns when an agent asks it something, and what that costs in tokens and time.

- **Judge-only scoring.** No answering LLM sits between the memory system and the judge, so
  a clever answer prompt can't hide weak retrieval. The judge reads the raw retrieved context.
- **One pinned judge.** One exact model snapshot, prompt hash and reasoning setting for every
  published run. Changing the judge starts a new season; stored contexts are re-judged.
- **Token-budget aware.** One fixed tokenizer, a hard per-query budget, and accuracy always
  reported against tokens returned. Dumping the whole conversation does not win.
- **Honest statistics.** 95% confidence intervals (cluster bootstrap over conversations) and
  sample sizes for every number; failures count as wrong, not as missing.
- **The real costs.** Ingestion time until data is searchable, retrieval latency p50/p95 and
  error/timeout rate. Anything a system doesn't report is shown as "not reported", never
  guessed.
- **Better ground truth.** Abstention and adversarial questions are scored properly,
  knowledge-update questions report a stale-fact rate, suspect gold answers are flagged
  (not dropped), and reviewed corrections ship as versioned errata.
- **Verifiable.** Results are canonicalised (RFC 8785) and signed (Ed25519); the dataset,
  connector code, judge prompt and raw per-question outputs are all hashed into the result.

Built and maintained by the founder of Recalld, a memory system — disclosed here and on
every Recalld run. See [docs/scoring.md](https://github.com/memjudge/memjudge/blob/main/docs/scoring.md)
for exactly how results are computed.

## Quick start

```bash
pip install memjudge              # core + BM25 baseline; add extras for vendor SDKs (below)
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
memjudge run --dataset locomo --connector baselines/bm25 --judge google:<model> --smallest 1
```

This writes `runs/<run-id>/` with `result.json`, raw `questions-top10.jsonl`,
`questions-top30.jsonl` and `conversations.jsonl`, and `report-top10.html` and
`report-top30.html` you can open in a browser. Local runs are for private testing and are
never published to the leaderboard.

| Command | What it does |
| --- | --- |
| `memjudge datasets` / `memjudge connectors` | List what you can run |
| `memjudge run` | Benchmark a connector on a dataset (`--resume <dir>` continues a run) |
| `memjudge report <run> [<run> …]` | HTML report; 2–4 runs gives a side-by-side comparison |
| `memjudge rejudge <run> --judge …` | Re-score stored contexts with another judge |
| `memjudge run … --reuse-ingestion <run>` | Rerun only retrieval on data kept with `--keep-data` |
| `memjudge cleanup <run>` | Delete data a `--keep-data` run left in the memory system |
| `memjudge estimate --dataset …` | Typical judge cost and upper bound, and what the memory system must ingest |
| `memjudge calibrate labels.jsonl --judge … --judge …` | Compare candidate judges: agreement, false-accept rate, cost |
| `memjudge crosscheck <run> … --judge …` | Re-judge a sample with a second judge; report disagreement per system |
| `memjudge verify result.signed.json` | Check a published result's signature (and artifacts) |
| `memjudge validate-dataset my.json` | Validate a custom dataset in the canonical format |

API keys are read from the environment or `--env-file`; they are never written to disk or
logs, and connectors only receive the keys they declare.

## Connectors

| Connector | System | Install extra | Keys | Notes |
| --- | --- | --- | --- | --- |
| `baselines/bm25` | BM25 keyword search over raw turns | – | – | Reference point; runs locally |
| `mem0/platform` | Mem0 Platform (hosted) | `mem0` | `MEM0_API_KEY` | Local only: Mem0's terms forbid benchmarking, so the hosted runner does not offer it |
| `mem0/oss` | Mem0 open source, in-process | `mem0-oss` | `MEM0_OSS_OPENAI_API_KEY` | Full multi-signal setup; also run `python -m spacy download en_core_web_sm`. Mem0 OSS has no `timestamp`, so the conversation date goes into Mem0's extraction prompt through its `prompt` argument. Refuses to run while `OPENROUTER_API_KEY` is set (Mem0 would use it) |
| `zep/graphiti` | Zep Graphiti open source, in-process | `graphiti` | `GRAPHITI_OPENAI_API_KEY` | Needs FalkorDB on `localhost:6379`: `docker run -p 6379:6379 falkordb/falkordb`. Same LLM and embedder as Mem0 OSS; the hosted runner starts its own FalkorDB, empty for every run |
| `zep/cloud` | Zep Cloud | `zep` | `ZEP_API_KEY` | Local only: Zep's terms forbid publishing benchmark results without its written consent, so the hosted runner does not offer it |
| `supermemory/cloud` | Supermemory (hosted) | `supermemory` | `SUPERMEMORY_API_KEY` | One document per session; waits until Supermemory has indexed it and built its memories (`dreaming: instant`, one extra billed operation per session) |
| `letta/cloud` | Letta (hosted) | `letta` | `LETTA_API_KEY` | Agent-managed memory with `openai/gpt-6-luna` and `openai/text-embedding-3-small` (as Mem0 OSS), checked on the created agent; one LLM step per turn, billed by Letta, or by OpenAI when the account has its own OpenAI key connected in Letta. Returns memory blocks, Letta's conversation search, then all archival memory (Letta Cloud has no working archival search). Runs 3 conversations at a time to fit Letta's free plan (3 agents) |
| `recalld/cloud` | Recalld (hosted) | – | `RECALLD_API_KEY` | Vendor-maintained by the MemJudge maintainer. Sends every model explicitly (`openai-gpt-6-luna`, `gemini-embedding-002-vertex`) and stops if Recalld reports others |
| `offline/replay` | Replay of recorded search results | – | – | No API calls |

Every connector names the models its system runs, or says the vendor picks them (ADR 0014); results show them.
Every run scores two result sizes from one ingestion (ADR 0018): **Top 10** and **Top 30**.
Each question is retrieved twice, once for up to 10 items and once for up to 30 (Zep and
Graphiti: 7 + 3 and 20 + 10 facts + entities; Letta: see above), and each size is judged and
scored on its own. The harness cuts every context to the same token budget.

Install one extra, several (`"memjudge[mem0,zep]"`) or all of them (`"memjudge[all]"`).
Each SDK is pinned to the version its connector was written against. Every hosted connector
also takes a `base_url` setting for self-hosted or staging servers; such runs are marked as
custom endpoints. Mem0 OSS uses its own key name so it never receives the judge's key.

## Datasets

| Name | Source | Licence |
| --- | --- | --- |
| `locomo` | [snap-research/locomo](https://github.com/snap-research/locomo) | CC BY-NC 4.0 |
| `longmemeval-s` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (cleaned) | MIT |
| `longmemeval-oracle` | LongMemEval, evidence sessions only (quick checks) | MIT |

Datasets are not redistributed: each is downloaded from a pinned upstream revision and its
SHA-256 is checked before use. Custom datasets work locally with `--dataset my.json`.

## Docs

- [Running benchmarks](https://github.com/memjudge/memjudge/blob/main/docs/running.md)
- [How scoring works](https://github.com/memjudge/memjudge/blob/main/docs/scoring.md)
- [How signing works](https://github.com/memjudge/memjudge/blob/main/docs/signing.md)
- [Adding a connector](https://github.com/memjudge/memjudge/blob/main/docs/adding-a-connector.md)
- [Adding a dataset](https://github.com/memjudge/memjudge/blob/main/docs/adding-a-dataset.md)
- [Design decisions](https://github.com/memjudge/memjudge/tree/main/docs/decisions)

## Contributing

See [CONTRIBUTING.md](https://github.com/memjudge/memjudge/blob/main/CONTRIBUTING.md).
Security issues: [SECURITY.md](https://github.com/memjudge/memjudge/blob/main/SECURITY.md).

## Licence

Code: [Apache-2.0](https://github.com/memjudge/memjudge/blob/main/LICENSE). Published results
(scores, verdicts, metrics): CC BY 4.0. Dataset text inside per-question records (questions,
answers, conversation excerpts) keeps its dataset's licence: LoCoMo is CC BY-NC 4.0
(non-commercial use only), LongMemEval is MIT. MemJudge is a free, non-commercial project. The
MemJudge name and logo are not covered by these licences; see
[TRADEMARK.md](https://github.com/memjudge/memjudge/blob/main/TRADEMARK.md).
