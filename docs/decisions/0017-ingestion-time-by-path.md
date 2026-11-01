# 0017 — Ingestion time by path

Status: accepted (2026-10-30)

**Context.** A run ingests in one of two ways, and they time different things:

- **Per-session** (ADR 0007): `total_s` adds up each conversation's time until searchable.
  This is serial time. It does not depend on how many conversations run at once (ADR 0013).
- **Batch** (ADR 0010): `total_s` is the wall-clock time of one upload until all of it is
  searchable, with whatever parallelism the system uses inside.

ADR 0010 says the two are not comparable. But results did not say which path they used, and
the leaderboard ranked them in one "lower is better" row, so a batch run looked many times
faster than a per-session run of the same data.

**Decision.** Results record the path in `metrics.ingestion.mode`: `per-session` or `batch`.
Results without it are per-session. The numbers keep their meaning. Wherever ingestion time
is shown, batch values are labelled "batch, wall-clock". The compare page marks no best
ingestion time when the runs used different paths. A run that reuses or re-judges a batch run's
ingestion keeps that run's batch record, so its mode and time stay the batch's.

**Rejected: wall-clock time for every run.** Per-session runs have no ingestion-only clock:
each worker ingests a conversation, then retrieves and judges it, then takes the next, so
"first ingest call until all is searchable" would include most of the run's retrieval and
judging. Splitting runs into an ingest phase and a retrieval phase would not make it fair
either: per-session wall-clock is capped by the run's concurrency (4, or a connector's
`max_concurrency`), while a batch is not capped. Batch runs would still win by design, and
ingestion time would start to depend on concurrency.

**Consequences.** Batch and per-session ingestion times are never ranked against each other.
A system with a batch path that wants a comparable ingestion time runs per-session.
Accuracy and the other metrics are unaffected.
