# 0007 — Connector contract: blocking ingest

Status: accepted (2026-10-01)

**Context.** Many memory systems process writes asynchronously (Mem0 queues events; Zep
builds its graph in the background). Measuring only the write call hides the real
time-to-searchable, a top complaint from users, and querying too early lowers accuracy.

**Decision.** `ingest` must return only when the session is searchable. Connectors poll the
vendor's status API; the poll time counts as ingestion time and polling calls count as API
calls. Ingestion is per session; per-turn time is total time divided by turns.

**Consequences.** Ingestion numbers are comparable across synchronous and asynchronous
systems. Connectors need a timeout setting so a stuck queue fails the conversation rather
than the run.
