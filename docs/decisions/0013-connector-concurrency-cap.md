# 0013 — Connector concurrency cap

Status: accepted (2026-10-26)

**Context.** A run processes several conversations at once (4 by default, locally and on
the hosted runner). Some vendors' free plans limit how many stored objects an account may
hold at once. Letta's free plan allows 3 agents, and its connector creates one agent per
conversation, so a free account fails on the fourth conversation. Many submitters will use
free plans.

**Decision.** `connector.yaml` may set `max_concurrency`: the most conversations a run of
that connector processes at once. A run uses the lower of its own setting and the cap. The
cap is set only to fit a vendor plan limit, applies to every run of that connector (local
and hosted), and is recorded in the result (`config.concurrency`) and shown on the run page.
Letta Cloud: 3.

**Why it is neutral.** No published metric depends on concurrency:

- Each conversation runs in its own empty namespace, so verdicts are independent of what
  runs beside it.
- Retrieval latency is measured per `retrieve` call, and ingestion time per session, summed
  over conversations. Neither is wall-clock time for the whole run.

The only effect is load on the vendor's service: fewer parallel conversations means
slightly less load, which can only help that system's latency, never hurt it. The run's
total duration on the run page grows, but it is not a metric and is not compared.

**Rejected: one lower concurrency for every connector.** It would make every run slower
(about a third longer at 3 instead of 4) and push slow systems on large datasets towards
the hosted runner's 24-hour limit, with no fairness gain.
