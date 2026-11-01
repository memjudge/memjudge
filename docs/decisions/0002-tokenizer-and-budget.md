# 0002 — Harness-side tokenizer and hard token budget

Status: accepted (2026-10-01)

**Context.** Systems differ in how much text they return; returning more raises accuracy
(one vendor returned the whole conversation with top_k=50). Vendor token counts are not
comparable.

**Decision.** The harness counts every context with one tokenizer (`o200k_base`, tiktoken)
and truncates it to a per-run budget (default 8,192). Truncation rate, mean/p95 tokens and
efficiency (accuracy per 1k tokens) are reported. Connectors receive the budget as a hint.

**Consequences.** Results are compared at equal budgets. A system that ignores the budget is
cut off rather than rewarded. The tokenizer is a fixed proxy, not any model's exact count.
