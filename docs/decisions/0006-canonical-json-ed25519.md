# 0006 — RFC 8785 canonical JSON and Ed25519 signatures

Status: accepted (2026-10-01)

**Decision.** Results are serialised with RFC 8785 (`rfc8785` package) and signed with
Ed25519 (`cryptography`). The signed envelope is `{payload, signature}`; the key id is the
first 16 hex characters of SHA-256(raw public key). The same canonical form is used for
dataset hashes.

**Consequences.** Any language with a JCS implementation can verify results; the website
build verifies every result and skips invalid ones.
