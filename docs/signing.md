# How signing works

Hosted runs produce a signed envelope; `memjudge sign` and `memjudge verify` implement it:

```json
{
  "payload": { "...": "the full result.json, including key_id" },
  "signature": { "alg": "Ed25519", "key_id": "3f2a…", "value": "<base64>" }
}
```

- The payload is serialised with **RFC 8785 (JSON Canonicalization Scheme)**, so any
  implementation produces the same bytes, and signed with **Ed25519**.
- `key_id` is the first 16 hex characters of the SHA-256 of the raw public key.
- The payload contains: schema version, run id, timestamps, tool version + commit, dataset
  hash, connector id + code hash + settings + endpoint (the API URL called,
  without credentials, and whether it is a custom endpoint), judge route + model + effort +
  prompt hash, run config, metrics, price-table version, artifact hashes (and URLs once
  uploaded), the affiliation field and the key id.
- It never contains API keys or their hashes, IP addresses or e-mail addresses.
- The signature proves the result came from the official runner unchanged; it does not by
  itself prove which server was called. That guarantee comes from the hosted runner, which
  only allows a connector's declared domains and **refuses custom endpoints**, so a
  published result's endpoint is always the vendor's production API.

## Signing

```bash
memjudge sign runs/<id>/result.json --key keys/<key_id>.key \
  --artifact-url-prefix https://artifacts.memjudge.org/runs/ -o <run_id>.json
```

`--artifact-url-prefix` records where the raw artifacts were uploaded: each artifact URL
becomes `<prefix><run_id>/<name>`. The URLs are part of the signed payload.

## Verifying

```bash
memjudge verify result.signed.json                       # trusted keys bundled with memjudge
memjudge verify result.signed.json --keys path/to/keys   # your own trusted .pub files
memjudge verify result.signed.json --artifacts run-dir   # also check raw artifacts' SHA-256
```

The bundled key is the hosted runner's production key, `478df5eb628bfa74`.

Verification fails if the payload changed, the key id does not match, the key is not
trusted, or (with `--artifacts`) any artifact's hash differs.

## Keys

- `memjudge keygen --out keys/` creates `<key_id>.key` (secret) and `<key_id>.pub`.
- The production private key will live only in GCP Secret Manager and be used by the runner
  job. Only public keys are committed, under `src/memjudge/keys/`.
