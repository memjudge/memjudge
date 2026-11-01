# 0014 — Explicit, checked models

Status: accepted (2026-10-26)

**Context.** Most memory systems run language and embedding models, and the model changes
the score: a stronger model extracts and selects better, at a higher cost and latency. Two
runs of the same system with different models are different results. Where a model comes
from an account default (Recalld) or a server default (Letta), a run's settings do not
show which model ran, and the default can change between runs or be changed by whoever
owns the account. A "vendor default" therefore says nothing a reader can rely on.

**Decision.**

- Every connector whose models can be chosen names them as settings, and sends every one
  explicitly; account and server defaults never apply. `connector.yaml` lists those
  settings under `models:`.
- The connector checks that the system ran the requested models, and stops the run if not.
  Recalld reports the models each memory call ran with (`X-Recalld-Models` response
  header); Letta reports the agent's model and embedding when the agent is created.
  In-process libraries (Mem0 open source, Graphiti) run exactly what they are given.
- Hosted products that pick their models themselves and do not say which (Mem0 Platform,
  Zep Cloud, Supermemory) declare `models: vendor`. Their results are labelled "models
  chosen by the vendor, not disclosed".
- The result records the models (`connector.models`). The `/run` form, the run page and
  the boards show them, so two runs of one system with different models are told apart.
- Other models are offered only as named presets, so each is its own, labelled result.

**Defaults.** Every system that lets you choose its LLM runs GPT-6 Luna: `gpt-6-luna` for
Mem0 open source and Graphiti, `openai/gpt-6-luna` for Letta, `openai-gpt-6-luna` for
Recalld. One LLM for all, so the memory design is compared, not the model. Embedders stay
each system's own choice, because they cannot all match: Letta offers only
`text-embedding-3-small` (also used by Mem0 open source and Graphiti), and Recalld offers
only `gemini-embedding-002-vertex`. Each is shown next to its result.

**Consequences.** Results show what ran. A Recalld gateway that does not report its
models stops a run, so the connector needs a gateway with `X-Recalld-Models`. Recalld
batch runs (local only) are not checked per call.
