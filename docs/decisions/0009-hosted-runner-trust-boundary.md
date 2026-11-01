# 0009 — Hosted runner: the trust boundary is the job

Status: accepted (2026-10-08)

**Context.** A hosted run executes connector code and vendor SDKs, which MemJudge reviews
but does not control. Any process in a Cloud Run container can get its service account's
token from the metadata server, so separate processes or users inside one container do
not keep secrets apart.

**Decision.** Each run uses two Cloud Run jobs from the same pinned image:

- `retrieve-<connector>` runs the connector. Its service account can decrypt only the
  connector keys (KMS key A), and its internet egress is limited to the connector's
  declared domains (see Egress below). It never holds the judge key, the signing
  key or publishing credentials.
- `finish` runs MemJudge code only and never imports connector code. It decrypts the
  judge key (KMS key B), rebuilds the retrieval records from the dataset (keeping only
  what the connector controls), judges, signs and publishes.

Keys reach a job only as KMS ciphertext bound to the run id, stored under a random name
that the job reads once and deletes; job service accounts cannot list the bucket. Job
arguments carry only the run id and that name. Keys are redacted from job output and
from published records.

**Two environments (2026-10-25).** Any wheel can install a `.pth` file or a
`sitecustomize` module, which Python runs at startup in every process of that environment,
with no import. So finish keeps connector code out only if no connector SDK is installed
where it runs. The image has two virtual environments, each from a hash-pinned lock:
`/opt/core` (MemJudge and the runner, `runner/core.lock`), which the API, the finish job
and the build commands run from, and `/opt/retrieve` (the same plus every connector's SDK,
`runner/requirements.lock`), which only retrieve jobs run from. `runner/infra/deploy.sh`
sets each job's command; starting a run changes only its arguments. The image build fails
if `/opt/core` holds a package that is not in its lock, or a startup hook.

**Consequences.** A compromised connector or SDK can affect only its own run's retrieval
output, which finish re-checks. It cannot forge signed results, read other runs' keys or
see the judge key. The cost is a second job start per run. Resuming stopped runs: ADR 0011.

**Egress (2026-10-25).** The network rules in `runner/infra/setup.sh` stop generic
malicious code, not an attacker who targets MemJudge:

- Blocked: connections to anything but the subnet's declared domains and Google APIs, and
  DNS lookups of any other name. A Cloud DNS response policy answers those itself, so no
  data leaves in lookups. In `run.app`, only the Runner API resolves.
- Not blocked: code in a retrieve job can still send the connector keys and run token it
  holds (1) to the attacker's own account on the vendor's API, which the job must reach;
  (2) to other sites at the same addresses as a declared domain, because firewall FQDN
  rules match addresses and several vendors share Cloudflare's; (3) to another project's
  storage bucket or Cloud Run service, through Google's private address. VPC Service
  Controls would close only (3), so it is not used.

The keys at risk are the ones that user gave for that connector's vendor. The defences
are review and the hash-pinned image dependencies (`runner/requirements.lock`). Users
should run hosted benchmarks with a dedicated key that has a spending limit, and revoke it
afterwards.
