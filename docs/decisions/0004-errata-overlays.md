# 0004 — Errata overlays for gold answers

Status: accepted (2026-10-01)

**Context.** An audit found ~6.4% of LoCoMo gold answers wrong. Silently fixing data, or
dropping questions, makes results incomparable and hides decisions.

**Decision.** Corrections are versioned YAML overlays, one per dataset version. Each change
records question, field, old value, new value, reason and evidence; the old value must
match. Applying errata changes the dataset hash. Questions the judge flags as suspect are
kept, scored and listed for review — never dropped automatically.

**Consequences.** Every correction is reviewable and reversible. Results before and after an
erratum have different dataset hashes and are not compared.
