# MemJudge hosted runner

Runs published MemJudge benchmarks on Google Cloud Run. Three entrypoints share one image:

- `memjudge-runner api`: the Runner API behind the `/run` form (FastAPI, Cloud Run service).
- `memjudge-runner retrieve <run_id> <object>`: ingests and retrieves with the connector
  (Cloud Run job, one per connector). Holds only the connector keys.
- `memjudge-runner finish <run_id>`: judges, validates, signs and publishes (Cloud Run job).
  Never imports connector code.

The image has two Python environments. Retrieve jobs run from `/opt/retrieve`, which has
every connector's SDK (`requirements.lock`). Everything else runs from `/opt/core`, which
has none (`core.lock`), so a connector dependency's startup hook (a `.pth` file) never
runs in the API or the finish job. The image build fails if `/opt/core` holds anything
else (`memjudge-runner check-core-env`).

See `docs/decisions/0009-hosted-runner-trust-boundary.md` for why the work is split this way,
`docs/decisions/0011-resuming-hosted-runs.md` for how a stopped run is resumed on the
jobs of the release it started on, and `docs/decisions/0016-run-log-and-history.md` for the
job log its owner sees and the history its result publishes.

## Google Cloud setup

`infra/setup.sh` creates everything the service and jobs use, and is safe to re-run: service
accounts, KMS keys, buckets, secrets, Firestore, Workload Identity Federation for the deploy
workflow, and the egress network.

- Each connector's retrieve jobs run in their own subnet (`sn-<connector>`). A global
  firewall policy lets that subnet reach only the domains its `connector.yaml` declares
  (`memjudge-runner egress` lists them), plus Google APIs and the Runner API through
  Private Google Access. Everything else is denied. `sn-finish` reaches only the judge
  providers, R2 and GitHub. A DNS response policy lets jobs look up only those names, so
  nothing leaves in DNS lookups. What these rules cannot stop: see Egress in ADR 0009.
- Adding a connector: give its subnet a range in `SUBNETS` in `setup.sh` and run it again.
- Cloud NAT needs up to about a minute after a job starts before internet traffic flows, so
  jobs first wait until their hosts answer (`wait_for_hosts`).

## Deploy

`.github/workflows/runner.yml` (Actions → **Deploy runner** → **Run workflow** on `main`, or
a `v*` tag) builds `runner/Dockerfile` and runs `infra/deploy.sh`:

- one job per hosted connector, `retrieve-<connector>-<release>`, in its subnet; the
  `finish-<release>` job; then the public `runner-api` service. `memjudge-runner
  deploy-jobs` lists them, so the names and subnets come from the code.
- Redeploying a release replaces its jobs. A release's jobs are deleted 14 days after a
  newer release was deployed (`memjudge-runner stale-releases`), so stopped runs can still
  resume on the code they started with (ADR 0011).
- Only the deploy job can deploy: Google accepts the workflow's token only from the
  repository's `production` environment. In the repository settings, that environment
  allows only `main` and `v*` tags, and a tag ruleset lets only admins create `v*` tags,
  because a `v*` tag deploys whatever commit it points to.
