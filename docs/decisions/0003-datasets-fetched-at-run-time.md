# 0003 — Curated datasets are fetched from pinned upstream sources

Status: accepted (2026-10-01)

**Context.** LoCoMo is CC BY-NC 4.0; LongMemEval is MIT but its haystacks come from
third-party chat datasets. MemJudge is run by a company.

**Decision.** v1 hosts no dataset copies. Each curated dataset is downloaded from a pinned
commit/revision URL; its SHA-256 is verified before conversion. The hash of the converted
canonical dataset (after errata) goes into every result.

**Consequences.** Runs need network access to the upstream host on first use (then cached).
If an upstream file disappears, the dataset version is retired.
