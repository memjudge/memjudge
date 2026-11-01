# Adding a connector

A connector is a folder `connectors/<system>/<name>/` with two files.

## `connector.yaml`

```yaml
id: acme/cloud                 # must equal the folder path
target: Acme Memory Cloud (v2 API)
author: Your Name
label: maintainer-written      # or vendor-maintained once a vendor owns it
sdk: acme-memory==1.4.2        # exact pin
required_keys: [ACME_API_KEY]  # the only keys the connector will receive
allowed_domains: [api.acme.ai] # hosted runs block all other egress
models: [llm_model]            # settings that name models; `vendor` if the vendor hides them
max_concurrency: 3             # optional: only to fit a vendor plan limit (ADR 0013)
settings:                      # defaults; all recorded in the result
  llm_model: gpt-6-luna        # always sent; never the account's default (ADR 0014)
  rerank: false
  base_url: null               # null = the vendor's production API
presets:                       # optional: named overrides a hosted run can pick
  reranked: { rerank: true }   # only existing settings, never base_url
```

Without presets a hosted run uses the defaults. With presets, the `/run` form offers
them and the run must pick one; the settings it ran with are in the result.

## `connector.py`

Define `class Connector(BaseConnector)` with three methods:

```python
from memjudge.connectors.base import BaseConnector, Item, Retrieval, Usage


class Connector(BaseConnector):
    def reset(self, namespace: str) -> None:
        """Delete everything under namespace. Must not fail if it does not exist."""

    def ingest(self, namespace: str, session: Session) -> Usage:
        """Store one session. Return only once the data is searchable."""

    def retrieve(self, namespace, query, asked_at, budget_tokens, top_k) -> Retrieval:
        """Return the context string an agent would be given: at most top_k items."""
```

Rules:

- **Finding data again.** If the system stores data under ids you create (rather than
  under the namespace itself), implement `save_state(namespace)` to return them (JSON, never
  keys) and `restore_state(namespace, state)` to load them back. This makes
  `--reuse-ingestion` and `memjudge cleanup` work. Connectors whose data is found by the
  namespace alone keep the defaults.
- **Models.** Send every model the system runs explicitly, from a setting listed under
  `models:`, so no account or server default applies. If the system reports which models
  it ran, check them and raise on a difference (ADR 0014).
- **Endpoint.** If the system is an API, accept a `base_url` setting (default: the vendor's
  production URL) and set `self.endpoint` to the URL actually used, in `__init__`. The
  result records it without credentials, and flags it as a custom endpoint when its host is
  not in `allowed_domains`. In-process systems leave `endpoint` as `None`.

- **One message per call.** Send each turn of the session as its own write call, in order,
  and wait until it is searchable before sending the next. Every system supports this, so
  ingestion is measured the same way for all of them, whatever batching they offer.
- **Block until searchable.** If the system processes writes in the background, poll its
  status API inside `ingest`. Ingestion time is measured until `ingest` returns.
- **Optional `ingest_batch`.** A system that accepts a whole upload at once may implement
  `ingest_batch(conversations, state_dir, progress)` (ADR 0010). It uploads the run once
  and returns when everything is searchable; the harness then skips per-session `ingest`.
  While it waits, call `progress(done, total)` with the turns processed so far, so the
  console shows the batch moving. Turn it on with a setting and keep it off by default.
  Its time and usage count as ingestion. The time is wall-clock, so results mark it `batch`
  and it is never ranked against per-session ingestion (ADR 0017).
- **Ask for `top_k` items.** The harness calls `retrieve` twice per question, with
  `top_k` 10 and 30 (ADR 0018). Pass it as the system's result count; never add a setting
  for it. If the system returns several kinds of item, split `top_k` between them in a fixed
  proportion that adds up to it.
- **Return exactly what an agent would see** in `Retrieval.context`. The harness counts and
  truncates it to the run's budget; respecting `budget_tokens` yourself avoids truncation.
- **Pass timestamps** (`turn.timestamp`, `session.started_at`, `asked_at`) if the system
  supports them.
- **Report usage** (`Usage.api_calls`, tokens, credits, `cost_usd`) when the API exposes it.
  Leave fields `None` otherwise — never estimate.
- Set `Item.source_turn_ids` when the system can say which turns an item came from; this
  enables the evidence-recall metric.
- Disable SDK telemetry and any network calls outside `allowed_domains`.
- Never log or print keys. Raise exceptions on failure; the harness records them.

## Tests

Add a contract test to `tests/test_connectors.py` that replaces the vendor SDK with an
in-memory fake (see the Mem0 and Zep tests) and a live smoke test skipped unless the key is
set. CI runs the contract tests on every PR.

If the connector needs a vendor SDK, add an install extra in `pyproject.toml` with the same
exact pin as `sdk`, and add it to the `all` extra (CI installs `.[dev,all]`). Then
regenerate `runner/requirements.lock`, the hash-pinned list the hosted runner image installs
(the command is in `runner/requirements.in`), then `runner/core.lock`, the same list without
connector SDKs (the command is in `runner/core.in`); the image build fails while a lock is
out of date.

## Review

Open a pull request. A maintainer reviews the code; vendors are invited to review
connectors for their system. A connector is identified by its **code hash** (all files in
the folder), so any change produces a new identity in results; exact duplicates are rejected.
