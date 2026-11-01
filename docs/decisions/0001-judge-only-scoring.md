# 0001 — Judge-only scoring

Status: accepted (2026-10-01)

**Context.** Published memory results usually put an answering LLM, with a vendor-chosen
prompt, between retrieval and the judge. Scores then mix retrieval quality with answer-prompt
quality, and disputes (Mem0 vs Zep on LoCoMo) turned on prompts and scoring choices.

**Decision.** The judge reads the raw retrieved context and decides whether it lets a reader
reach the gold answer. One pinned judge (model snapshot, prompt hash, reasoning effort) per
season. The judge is called over plain HTTP (httpx) for every route — Anthropic, OpenAI,
Google, OpenRouter with the upstream provider pinned and fallbacks disabled — instead of
four vendor SDKs.

**Consequences.** Scores are not directly comparable to answer-accuracy numbers in papers.
The judge must handle long contexts, so a mid-tier model is needed and calibration
(agreement + false-accept rate) is mandatory before a season starts.
