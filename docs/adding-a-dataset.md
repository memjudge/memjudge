# Adding a dataset

## Canonical format

Every dataset is converted to one shape (`memjudge.schema.Dataset`):

```json
{
  "schema_version": "1", "name": "mydata", "version": "v1",
  "source": "https://…", "license": "MIT",
  "conversations": [
    {"id": "c1", "sessions": [
      {"id": "s1", "started_at": "2024-03-01T10:00:00", "turns": [
        {"id": "c1:1", "speaker": "Ana", "text": "…", "timestamp": "2024-03-01T10:00:00"}
      ]}
    ]}
  ],
  "questions": [
    {"id": "q1", "conversation_id": "c1", "question": "…", "gold_answer": "…",
     "category": "single-hop", "kind": "standard", "asked_at": "2024-06-01T00:00:00",
     "evidence_turn_ids": ["c1:1"]}
  ]
}
```

`kind` is `standard`, `abstention` (the answer was never in the conversation; optional
`distractor_answer`) or `knowledge_update` (the fact changed; gold is the latest value).

Validate a file with `memjudge validate-dataset mydata.json`. Custom files can be run
locally with `--dataset mydata.json`; they are never published.

## Curated datasets

1. Check the licence. If it allows redistribution, the converted copy may be hosted;
   otherwise the adapter downloads from the original source at run time.
2. Add an adapter in `src/memjudge/datasets/adapters/<name>.py` with
   `convert(raw) -> (conversations, questions)`.
3. Register it in `src/memjudge/datasets/registry.py` with a **pinned** URL (commit or
   revision, never a branch) and the file's SHA-256.
4. Add tests with a small synthetic sample in the real upstream layout.
5. Open a pull request with the source link. CI validates the schema; a maintainer approves.

## Errata

Gold-answer corrections live in `src/memjudge/datasets/errata/<name>-<version>.yaml`. Each
change names the question, field, old value, new value, reason and evidence. The old value
must match, and applying errata changes the dataset hash, so corrected and uncorrected
results are never mixed. Propose errata by pull request; a maintainer reviews each change
against the source conversation. Questions the judge flags as `gold_suspect` are the main
input to this review.
