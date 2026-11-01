# 0018 — Top 10 and Top 30

Status: accepted (2026-10-31). Replaces the single item count of ADR 0012.

**Context.** ADR 0012 has every connector ask for up to 30 items per question. The item count
changes results: a system that ranks its best items first loses little when it may return
only 10, while one that relies on returning many loses more. A single count hides that
difference, and letting each run pick its own count would invite picking the count that
suits one system.

**Decision.** Every run scores two result sizes from one ingestion:

- **Top 10** and **Top 30**: for each question the harness retrieves twice, once for up to
  10 items and once for up to 30. No other sizes exist, locally or hosted.
- The harness passes the size to the connector: `retrieve(namespace, query, asked_at,
  budget_tokens, top_k)`. Connectors have no item-count setting any more.
- A system that returns several kinds of item splits the size in the same proportion as
  before: Zep returns `round(2/3 × top_k)` facts and the rest as entities (20 + 10 and
  7 + 3). The Letta exception of ADR 0012 stays: memory blocks, the top `top_k` messages,
  then all archival memory.
- The two calls alternate their order from question to question, so neither size always
  gets the cold call when a system caches the query.
- Each size has its own records, `questions-top10.jsonl` and `questions-top30.jsonl`,
  judged separately. `metrics` holds one entry per size, `top10` and `top30`. Ingestion is
  shared, so both entries carry the same ingestion numbers.
- The token budget (ADR 0002) is unchanged and applies to both sizes.
- The leaderboard shows one board per size. Run and compare pages switch between them.

**Rejected: cutting Top 30 down to its first 10.** It is the same as a Top 10 call only for
systems that return a plain ranked list. Recalld's recall chooses what to return for the
limit it is given, Zep returns two lists, and Mem0 open source fuses several signals before
it cuts.

**Rejected: two separate runs.** That doubles ingestion (hours and the system's LLM bill
per run), and two ingestions of the same data differ, so the sizes would not be compared on
the same stored memory.

**Rejected: two linked results with their own run ids.** Same data, but twice the
publishing: two signatures, two ids and two records for the runner to keep in step.

**Consequences.** A run makes twice the retrieval calls and judges twice the contexts (about
$10 more judge cost on LoCoMo); ingestion is unchanged. The connector contract changes, so
every connector, including third-party ones, must accept `top_k`.
