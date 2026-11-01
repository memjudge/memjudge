# 0012 — Same retrieval size for every connector

Status: accepted (2026-10-18). The single item count is replaced by ADR 0018: every run
scores Top 10 and Top 30.

**Context.** The token budget (ADR 0002) caps what every system may return, but each
connector also asks its system for a number of items. These numbers differed (BM25 asked for
20, most others for 30). A system that returns fewer items than the budget allows is
handicapped, and different numbers look arbitrary to readers even when they are harmless.

**Decision.** Every connector asks for up to **30 items** per question by default. Where a
system returns several kinds of item, the kinds add up to 30 (Zep: 20 facts + 10 entities).
The token budget stays the real equaliser: items differ in size (a Mem0 fact is a short
sentence, a BM25 item is a whole turn), so the same item count is not the same amount of
context. Settings, including the item count, are recorded in every result and shown on the
run page.

**Exception.** Letta Cloud has no working archival search, so its connector returns the
agent's memory blocks, the top 30 messages from conversation search, and then all archival
memory. The budget still truncates it like any other system.

**Consequences.** Earlier BM25 results (20 items) are not comparable with new ones; the
connector's code hash changed, so they show as a different connector version. Runs with
other item counts remain possible locally and are labelled by their settings.
