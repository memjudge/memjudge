# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Changes that alter scores are marked **[scoring]**.

## [Unreleased]

### Fixed
- `zep/graphiti` runs its fact and entity searches at the same time, as an app would, so
  retrieval latency is no longer the sum of the two. Accuracy is unchanged.

## [0.1.0] - 2026-11-02

First release.

### Added
- Core engine and `memjudge` CLI: `run` (with `--resume`), `report`, `rejudge`, `estimate`,
  `calibrate` (compares several judges), `crosscheck` (second-judge sample check), `keygen`,
  `sign`, `verify`, `cleanup`, `conversations`, `datasets`, `connectors`,
  `validate-dataset`, `schema`.
- Canonical dataset format with SHA-256 hashing, errata overlays and adapters for LoCoMo and
  LongMemEval (cleaned S and oracle), downloaded from pinned upstream revisions. LongMemEval
  abstention questions keep their question type as category **[scoring]**.
- Connectors: BM25 baseline, Mem0 Platform (v3), Mem0 open source (in-process, full
  multi-signal setup), Zep Cloud (v3), Zep Graphiti open source (in-process, local
  FalkorDB), Supermemory (v5), Letta (v1), Recalld (v1 API, vendor-maintained by the
  MemJudge maintainer), and an offline replay of recorded results. Each connector's SDK is
  an install extra (`memjudge[mem0]`, `[mem0-oss]`, `[zep]`, `[graphiti]`, `[supermemory]`,
  `[letta]`, or `[all]`), pinned to the version it was written against.
- Connector contract (ADR 0007): `retrieve` takes `top_k`. `connector.yaml` lists the
  settings that name models (`models:`, or `vendor` when the vendor hides them; ADR 0014),
  named setting overrides (`presets:`) and `max_concurrency` (ADR 0013). Every system that
  lets you choose its LLM runs GPT-6 Luna. Optional batch ingestion (`ingest_batch`,
  ADR 0010) uploads the whole run at once; Recalld uses it.
- Top 10 and Top 30 (ADR 0018): every run scores two result sizes from one ingestion. Each
  question is retrieved twice, for up to 10 and up to 30 items, and each size is judged on
  its own. The two calls alternate their order from question to question. Zep and Graphiti
  split the size into facts and entities, 20 + 10 and 7 + 3. **[scoring]**
- Judge-only scoring (ADR 0001) with routes for Anthropic, OpenAI, Google and OpenRouter
  (provider pinned, no fallback). A reply holding more than one JSON object counts its last
  valid verdict. **[scoring]**
- Seasons (ADR 0015): `src/memjudge/judge/seasons.json` lists every season's judging rules
  (judge, effort, prompt version and hash) with a name, start date, summary, details link and
  how earlier runs get in (`upgrade`: `re-judge` or `re-run`). Tests freeze each season's
  prompt file, and CI rejects changes to past seasons. Season 1 is Claude Sonnet 5.5 with
  judge prompt v3. **[scoring]**
- Re-judging: `memjudge rejudge` judges a run's stored contexts again and records
  `rejudged_from`, keeping the earlier run's connector record.
- Metrics: accuracy with cluster-bootstrap 95% CI, context tokens, efficiency, latency,
  ingestion time until searchable (`per-session` or `batch`, ADR 0017), abstention accuracy,
  stale-fact rate, evidence recall, error rate, truncation rate, flagged questions, judge
  cost, and the usage each memory system reports.
- A run stops when an ingestion fails or 5 retrievals fail in a row. Completed
  conversations are saved; `memjudge run --resume` redoes the unfinished ones.
- Every connector accepts a `base_url` setting; results record the endpoint called (without
  credentials) and flag custom endpoints (localhost, staging, self-hosted).
- Partial runs: `--conversation`, `--category`, `--smallest`, `--sample`, and
  `memjudge conversations` to list conversations by size.
- `--keep-data`, `--reuse-ingestion` and `memjudge cleanup`: rerun only retrieval and
  judging against data kept by an earlier run (local-only; labelled in the result).
- `memjudge estimate` (also `--json`): what the memory system must ingest (sessions, write
  calls, turn tokens), and the judge cost, typical and upper bound, for both result sizes.
- RFC 8785 + Ed25519 signing and verification. `memjudge verify` trusts the bundled hosted
  runner key `478df5eb628bfa74`; `memjudge sign --artifact-url-prefix` records each
  artifact's URL in the signed payload.
- HTML report per result size with ECharts (loaded from a CDN with subresource integrity)
  and a 2–4 run compare view.
- Hosted runner (`runner/`, package `memjudge-runner`, ADR 0009): Runner API, retrieve job
  and finish job on Google Cloud Run, with GitHub login. Each connector's jobs may reach only
  its declared domains (`memjudge-runner egress`). A stopped run can be resumed within 48
  hours, at most 3 times, on the jobs of its own release (ADR 0011). Owners see the job log
  and each result records its stops and resumes (ADR 0016). Anyone signed in can re-judge a
  published run into the current season with their own judge key. Mem0 Platform and Zep
  Cloud are local-only, because their terms forbid publishing benchmark results without the
  vendor's consent. `.github/workflows/runner.yml` builds the image and deploys one job set
  per release.

[Unreleased]: https://github.com/memjudge/memjudge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/memjudge/memjudge/releases/tag/v0.1.0
