# Security policy

## Reporting a vulnerability

E-mail **admin@memjudge.org** with details and steps to reproduce. Please do not open a
public issue. We aim to acknowledge reports within 3 working days and to agree a disclosure
date with you.

In scope: this repository (CLI, harness, connectors and the hosted runner), signing
and verification, key handling, and anything that could let a result be forged or a user's
API key be exposed.

## How keys are handled

- Local CLI: keys come from the environment or `--env-file`; they are never written to run
  folders, results or logs, and each connector only receives the keys it declares.
- Hosted runs: keys are sent over HTTPS to the runner API, encrypted there, passed to the
  job only as ciphertext, decrypted in memory and never logged or set as job environment
  variables.
- Results never contain keys, key hashes, IP addresses or e-mail addresses.
