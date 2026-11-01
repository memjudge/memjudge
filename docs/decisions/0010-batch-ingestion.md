# 0010 — Optional batch ingestion

Status: accepted (2026-10-11)

**Context.** Large datasets are expensive to ingest; LongMemEval-S alone has about 247,000
turns. Some memory systems accept a whole upload at once and process it at a lower price.
Recalld, for example, ingests the upload on the model provider's Flex tier, which bills
LLM tokens at half price but answers more slowly. ADR 0007 measures ingestion one message per
call, so a connector cannot use such a path in `ingest`.

**Decision.** Connectors may implement an optional `ingest_batch(conversations, state_dir)`.
The harness calls it once, before any conversation, with every namespace the run needs. The
connector uploads everything at once and returns only when all of it is searchable. The
harness then skips per-session ingestion and goes straight to the questions. If
`ingest_batch` fails, the run stops with that error: falling back to per-session ingestion
would measure something else at another cost. A connector
setting turns it on (for Recalld, `batch: true`), so the run's recorded settings show it was
used. The batch's time and usage count as the run's ingestion and are written to
`ingest_batch.json`, which is listed in the result's artifacts. State for resuming, such as
the batch id, lives in the run folder.

**Consequences.** Batch runs are local runs. They measure a different ingestion path, with
one long wait instead of per-message latency, so their ingestion time and cost are not
comparable with per-message runs. Accuracy should be the same as with per-message ingestion.
Connectors without `ingest_batch` are unaffected.
