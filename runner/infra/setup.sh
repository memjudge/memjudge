#!/usr/bin/env bash
# MemJudge hosted runner: one-time Google Cloud setup (ADR 0009 and 0011).
#
# Creates what the Runner API and the retrieve/finish jobs need: service accounts, KMS
# keys, buckets, secrets, Firestore, the per-connector egress network and GitHub Actions
# deploy access. Each step checks before it creates, so re-running it is safe: run it
# again after adding a connector (give it a subnet below) or a secret file.
#
# From the repository root, with gcloud logged in as a project owner:
#   PROJECT=memjudge-runner RUNNER=runner/.venv/Scripts/memjudge-runner runner/infra/setup.sh
#
# Secrets are read from $SECRETS_DIR (outside every repository) and never printed:
#   secrets.env       RESULTS_APP_ID, RESULTS_INSTALLATION_ID, LOGIN_CLIENT_ID,
#                     LOGIN_CLIENT_SECRET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   results-app.pem   the "MemJudge Results" GitHub App's private key
# A secret whose inputs are missing is skipped with a message.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT, e.g. PROJECT=memjudge-runner}"
RUNNER="${RUNNER:-memjudge-runner}"
PY="${PY:-python}"
SECRETS_DIR="${SECRETS_DIR:-$HOME/.memjudge/runner-secrets}"
# The signing key is only replaced when SIGNING_KEY_FILE names a key file: a plain re-run
# never touches the production key in Secret Manager.
SIGNING_KEY_FILE="${SIGNING_KEY_FILE:-}"

REGION=us-central1
BILLING_ACCOUNT=01307E-61AB48-527E2F
BUDGET=40GBP
R2_ACCOUNT_ID=b07ae0215e68b138c91c6da8dfc8c38d
R2_BUCKET=memjudge-artifacts
GITHUB_REPO=memjudge/memjudge
GITHUB_OWNER_ID=336665431 # the memjudge organisation's numeric id (names can be re-registered)
FINISH_DOMAINS="api.anthropic.com,openrouter.ai,$R2_ACCOUNT_ID.r2.cloudflarestorage.com,api.github.com"

# One /26 per egress profile, inside 10.10.0.0/22. Never renumber: append new connectors.
SUBNETS=(
  "sn-finish 10.10.0.0/26"
  "sn-baselines-bm25 10.10.0.64/26"
  "sn-letta-cloud 10.10.0.128/26"
  "sn-mem0-oss 10.10.0.192/26"
  "sn-mem0-platform 10.10.1.0/26"
  "sn-recalld-cloud 10.10.1.64/26"
  "sn-supermemory-cloud 10.10.1.128/26"
  "sn-zep-cloud 10.10.1.192/26"
  "sn-zep-graphiti 10.10.2.0/26"
)
ALL_SUBNETS=10.10.0.0/22

export CLOUDSDK_CORE_PROJECT="$PROJECT" # every gcloud call targets this project
NUMBER=$(gcloud projects describe "$PROJECT" --format="value(projectNumber)")
SA() { echo "$1@$PROJECT.iam.gserviceaccount.com"; }
API=$(SA runner-api) RETRIEVE=$(SA runner-retrieve) FINISH=$(SA runner-finish) DEPLOYER=$(SA runner-deployer)
SKIPPED=()

step() { echo "== $*"; }
exists() { "$@" >/dev/null 2>&1; }
retry() { # new service accounts take a few seconds to be visible to IAM
  for _ in 1 2 3 4 5 6; do "$@" >/dev/null 2>&1 && return 0; sleep 10; done
  "$@" >/dev/null
}
bind_project() { retry gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$1" --role="$2" --condition=None --quiet; }
bind_bucket() { retry gcloud storage buckets add-iam-policy-binding "gs://$1" --member="serviceAccount:$2" --role="$3"; }
bind_key() { retry gcloud kms keys add-iam-policy-binding "$1" --keyring=runner --location="$REGION" --member="serviceAccount:$2" --role="$3"; }
bind_secret() { retry gcloud secrets add-iam-policy-binding "$1" --member="serviceAccount:$2" --role=roles/secretmanager.secretAccessor; }

custom_role() { # name, permissions
  if exists gcloud iam roles describe "$1" --project="$PROJECT"; then
    gcloud iam roles update "$1" --project="$PROJECT" --permissions="$2" --quiet >/dev/null
  else
    gcloud iam roles create "$1" --project="$PROJECT" --title="$1" --stage=GA --permissions="$2" >/dev/null
  fi
}

# put_secret NAME: stores stdin as the secret's newest version, if it differs from it.
put_secret() {
  local value
  value=$(tr -d '\r') # Windows Python writes CRLF into pipes
  exists gcloud secrets describe "$1" ||
    gcloud secrets create "$1" --replication-policy=user-managed --locations="$REGION" >/dev/null
  local old
  old=$(gcloud secrets versions access latest --secret="$1" 2>/dev/null | tr -d '\r' || true)
  if [[ "$old" != "$value" ]]; then # compared in memory, never printed
    printf '%s' "$value" | gcloud secrets versions add "$1" --data-file=- >/dev/null
    echo "   $1: new version"
  fi
}

step "APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudkms.googleapis.com \
  firestore.googleapis.com secretmanager.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com cloudresourcemanager.googleapis.com compute.googleapis.com dns.googleapis.com \
  storage.googleapis.com billingbudgets.googleapis.com

step "Service accounts"
for sa in runner-api runner-retrieve runner-finish runner-deployer; do
  exists gcloud iam service-accounts describe "$(SA $sa)" ||
    gcloud iam service-accounts create "$sa" --display-name="MemJudge $sa" >/dev/null
done

step "Custom roles"
custom_role runnerJobController run.jobs.run,run.jobs.runWithOverrides,run.executions.get,run.executions.cancel,run.operations.get
custom_role runnerObjectGetDelete storage.objects.get,storage.objects.delete
custom_role runnerObjectCreateDelete storage.objects.create,storage.objects.delete

step "Artifact Registry"
exists gcloud artifacts repositories describe runner --location="$REGION" ||
  gcloud artifacts repositories create runner --repository-format=docker --location="$REGION" >/dev/null
retry gcloud artifacts repositories add-iam-policy-binding runner --location="$REGION" \
  --member="serviceAccount:$DEPLOYER" --role=roles/artifactregistry.writer
# Keep the 10 newest images; delete older ones after 14 days (a replaced release's jobs
# are deleted 14 days after the new release, ADR 0011).
gcloud artifacts repositories set-cleanup-policies runner --location="$REGION" \
  --policy=runner/infra/registry-cleanup.json --no-dry-run >/dev/null

step "Firestore"
exists gcloud firestore databases describe --database="(default)" ||
  gcloud firestore databases create --location="$REGION" --type=firestore-native >/dev/null
bind_project "$API" roles/datastore.user
bind_project "$FINISH" roles/datastore.user
bind_project "$API" roles/logging.viewer # shows a run's job log to its owner (ADR 0016)
for group in runs quota; do # Firestore deletes each record after its expire_at (privacy notice)
  gcloud firestore fields ttls list --format="value(name)" | grep -q "collectionGroups/$group/fields/expire_at" ||
    gcloud firestore fields ttls update expire_at --collection-group="$group" --enable-ttl --async >/dev/null
done

step "KMS"
exists gcloud kms keyrings describe runner --location="$REGION" ||
  gcloud kms keyrings create runner --location="$REGION"
for key in retrieve finish; do
  exists gcloud kms keys describe "$key" --keyring=runner --location="$REGION" ||
    gcloud kms keys create "$key" --keyring=runner --location="$REGION" --purpose=encryption
  bind_key "$key" "$API" roles/cloudkms.cryptoKeyEncrypter
done
bind_key retrieve "$RETRIEVE" roles/cloudkms.cryptoKeyDecrypter
bind_key finish "$FINISH" roles/cloudkms.cryptoKeyDecrypter

step "Buckets"
bucket() { # name, lifecycle file or ""
  exists gcloud storage buckets describe "gs://$1" ||
    gcloud storage buckets create "gs://$1" --location="$REGION" --uniform-bucket-level-access \
      --public-access-prevention --soft-delete-duration=0 >/dev/null
  if [[ -n "$2" ]]; then
    gcloud storage buckets update "gs://$1" --lifecycle-file="$2" >/dev/null
  fi
}
bucket "$PROJECT-secrets" runner/infra/lifecycle-3d.json # sealed key bundles
bucket "$PROJECT-staging" runner/infra/lifecycle-14d.json # retrieval output and saved progress
bucket "$PROJECT-datasets" "" # private mirror of the hosted datasets
P=projects/$PROJECT/roles
bind_bucket "$PROJECT-secrets" "$API" "$P/runnerObjectCreateDelete"
bind_bucket "$PROJECT-secrets" "$RETRIEVE" "$P/runnerObjectGetDelete" # no list: a job opens only its own bundle
bind_bucket "$PROJECT-secrets" "$FINISH" "$P/runnerObjectGetDelete"
bind_bucket "$PROJECT-staging" "$API" roles/storage.objectUser # signs the jobs' upload/download links
bind_bucket "$PROJECT-staging" "$FINISH" roles/storage.objectUser
bind_bucket "$PROJECT-datasets" "$RETRIEVE" roles/storage.objectViewer
bind_bucket "$PROJECT-datasets" "$FINISH" roles/storage.objectViewer

step "Runner API permissions"
bind_project "$API" "$P/runnerJobController"
retry gcloud iam service-accounts add-iam-policy-binding "$API" --member="serviceAccount:$API" \
  --role=roles/iam.serviceAccountTokenCreator # signs storage links as itself

step "Secrets"
if [[ -f "$SECRETS_DIR/secrets.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . <(tr -d '\r' <"$SECRETS_DIR/secrets.env")
  set +a
fi
if [[ -z "$SIGNING_KEY_FILE" ]]; then
  SKIPPED+=("signing-key: kept (set SIGNING_KEY_FILE to replace it)")
elif [[ -f "$SIGNING_KEY_FILE" ]]; then
  put_secret signing-key <"$SIGNING_KEY_FILE"
else SKIPPED+=("signing-key: no $SIGNING_KEY_FILE"); fi
exists gcloud secrets versions describe latest --secret=session-secret ||
  "$PY" -c "import secrets; print(secrets.token_hex(32))" | put_secret session-secret
if [[ -n "${RESULTS_APP_ID:-}" && -n "${RESULTS_INSTALLATION_ID:-}" && -f "$SECRETS_DIR/results-app.pem" ]]; then
  PEM=$(cat "$SECRETS_DIR/results-app.pem") "$PY" -c 'import json, os
print(json.dumps({"app_id": os.environ["RESULTS_APP_ID"], "installation_id": os.environ["RESULTS_INSTALLATION_ID"],
                  "private_key": os.environ["PEM"].replace("\r", "") + "\n"}))' | put_secret github-app
else SKIPPED+=("github-app: needs RESULTS_APP_ID, RESULTS_INSTALLATION_ID and results-app.pem"); fi
if [[ -n "${LOGIN_CLIENT_ID:-}" && -n "${LOGIN_CLIENT_SECRET:-}" ]]; then
  "$PY" -c 'import json, os
print(json.dumps({"client_id": os.environ["LOGIN_CLIENT_ID"], "client_secret": os.environ["LOGIN_CLIENT_SECRET"]}))' |
    put_secret github-oauth
else SKIPPED+=("github-oauth: needs LOGIN_CLIENT_ID and LOGIN_CLIENT_SECRET"); fi
if [[ -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" ]]; then
  R2_ACCOUNT_ID=$R2_ACCOUNT_ID R2_BUCKET=$R2_BUCKET "$PY" -c 'import json, os
print(json.dumps({"account_id": os.environ["R2_ACCOUNT_ID"], "bucket": os.environ["R2_BUCKET"],
                  "access_key_id": os.environ["R2_ACCESS_KEY_ID"], "secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"]}))' |
    put_secret r2
else SKIPPED+=("r2: needs R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"); fi
for s in github-oauth session-secret; do exists gcloud secrets describe "$s" && bind_secret "$s" "$API"; done
for s in signing-key r2 github-app; do exists gcloud secrets describe "$s" && bind_secret "$s" "$FINISH"; done

step "Network: VPC and one subnet per egress profile"
declare -A CIDR DOMAINS
for row in "${SUBNETS[@]}"; do read -r name range <<<"$row"; CIDR[$name]=$range; done
DOMAINS[sn-finish]=$FINISH_DOMAINS
while read -r name domains; do
  [[ -n "${CIDR[$name]:-}" ]] || { echo "No subnet range for $name: add it to SUBNETS in $0" >&2; exit 1; }
  DOMAINS[$name]=$([[ "$domains" == "-" ]] && echo "" || echo "$domains")
done < <("$RUNNER" egress | tr -d '\r')
exists gcloud compute networks describe runner ||
  gcloud compute networks create runner --subnet-mode=custom >/dev/null
NAT_SUBNETS=()
for row in "${SUBNETS[@]}"; do
  read -r name range <<<"$row"
  exists gcloud compute networks subnets describe "$name" --region="$REGION" ||
    gcloud compute networks subnets create "$name" --network=runner --region="$REGION" --range="$range" \
      --enable-private-ip-google-access >/dev/null
  [[ -n "${DOMAINS[$name]:-}" ]] && NAT_SUBNETS+=("$name") # no domains: no internet path at all
done

step "Network: Cloud NAT"
exists gcloud compute routers describe runner --region="$REGION" ||
  gcloud compute routers create runner --network=runner --region="$REGION" >/dev/null
NAT_LIST=$(IFS=,; echo "${NAT_SUBNETS[*]}")
if exists gcloud compute routers nats describe runner --router=runner --region="$REGION"; then
  gcloud compute routers nats update runner --router=runner --region="$REGION" \
    --nat-custom-subnet-ip-ranges="$NAT_LIST" >/dev/null
else
  gcloud compute routers nats create runner --router=runner --region="$REGION" \
    --nat-custom-subnet-ip-ranges="$NAT_LIST" --auto-allocate-nat-external-ips \
    --endpoint-types=ENDPOINT_TYPE_VM >/dev/null
fi

step "Network: private DNS for Google APIs and the Runner API"
VIP=199.36.153.8,199.36.153.9,199.36.153.10,199.36.153.11 # private.googleapis.com
API_HOST=runner-api-$NUMBER.$REGION.run.app
zone() { # zone, domain
  exists gcloud dns managed-zones describe "$1" ||
    gcloud dns managed-zones create "$1" --dns-name="$2." --visibility=private --networks=runner \
      --description="Private Google Access" >/dev/null
}
record() { # zone, name, type, data
  exists gcloud dns record-sets describe "$2." --type="$3" --zone="$1" ||
    gcloud dns record-sets create "$2." --type="$3" --ttl=300 --rrdatas="$4" --zone="$1" >/dev/null
}
zone googleapis googleapis.com
record googleapis private.googleapis.com A "$VIP"
record googleapis "*.googleapis.com" CNAME private.googleapis.com.
# run.app answers only for the Runner API, so jobs cannot look up other projects' services.
zone run-app run.app
record run-app "$API_HOST" A "$VIP"
if exists gcloud dns record-sets describe "*.run.app." --type=CNAME --zone=run-app; then
  gcloud dns record-sets delete "*.run.app." --type=CNAME --zone=run-app >/dev/null # earlier versions
  gcloud dns record-sets delete run.app. --type=A --zone=run-app >/dev/null
fi

step "Network: DNS allowlist"
# Cloud DNS answers every other name itself, so no lookup leaves Google and a job cannot
# carry data out in names it looks up (ADR 0009). It ignores a rule for the root, so each
# top-level domain in IANA's list gets one; its answer is a CNAME, which covers every record
# type, to a name the run-app zone does not have. Bypass rules let the names jobs need
# resolve as usual: the declared domains, Google APIs and the Runner API (the zones above
# answer those) and the metadata server. They go first, so a first run never blocks them.
RP=(--response-policy=runner-dns)
exists gcloud dns response-policies describe runner-dns ||
  gcloud dns response-policies create runner-dns --networks=runner \
    --description="Only the names the runner needs resolve" >/dev/null
HAVE=$(gcloud dns response-policies rules list runner-dns --format="value(ruleName)")
for name in $(printf '%s\n' "${DOMAINS[@]}" | tr ',' '\n' | sort -u) \
  "*.googleapis.com" "$API_HOST" metadata.google.internal; do
  rule=allow-${name//[^a-z0-9]/-}
  rule=${rule:0:63}
  grep -qx "$rule" <<<"$HAVE" ||
    gcloud dns response-policies rules create "$rule" "${RP[@]}" --dns-name="$name." \
      --behavior=bypassResponsePolicy >/dev/null
done
curl -fsS https://data.iana.org/TLD/tlds-alpha-by-domain.txt | grep -v '^#' | tr 'A-Z' 'a-z' |
  { grep -vxF -f <(sed -n 's/^tld-//p' <<<"$HAVE") || true; } |
  xargs -r -P 8 -I{} gcloud dns response-policies rules create "tld-{}" "${RP[@]}" \
    --dns-name="*.{}." --local-data="name=*.{}.,type=CNAME,ttl=300,rrdatas=blocked.run.app." \
    --no-user-output-enabled # about 1,450 rules: some 15 minutes the first time

step "Network: egress firewall policy"
FP=(--firewall-policy=runner-egress --global-firewall-policy)
exists gcloud compute network-firewall-policies describe runner-egress --global ||
  gcloud compute network-firewall-policies create runner-egress --global >/dev/null
gcloud compute network-firewall-policies describe runner-egress --global \
  --format="value(associations[].name)" | tr ';,' '\n\n' | grep -qx runner-egress ||
  gcloud compute network-firewall-policies associations create "${FP[@]}" --network=runner \
    --name=runner-egress >/dev/null
rule() { # priority, then the rule's flags
  local priority=$1
  shift
  if exists gcloud compute network-firewall-policies rules describe "$priority" "${FP[@]}"; then
    gcloud compute network-firewall-policies rules update "$priority" "${FP[@]}" "$@" >/dev/null
  else
    gcloud compute network-firewall-policies rules create "$priority" "${FP[@]}" --direction=EGRESS "$@" >/dev/null
  fi
}
# IP rules first: they are free, and FQDN (Standard tier) rules bill per GB they evaluate.
rule 100 --action=allow --layer4-configs=tcp:443 --src-ip-ranges="$ALL_SUBNETS" \
  --dest-ip-ranges=199.36.153.8/30,34.126.0.0/18 --description="Google APIs and run.app (private access)"
rule 110 --action=allow --layer4-configs=tcp:80,tcp:53,udp:53 --src-ip-ranges="$ALL_SUBNETS" \
  --dest-ip-ranges=169.254.169.254/32 --description="Metadata server and DNS"
i=0
for row in "${SUBNETS[@]}"; do
  read -r name range <<<"$row"
  if [[ -n "${DOMAINS[$name]:-}" ]]; then
    rule $((1000 + 10 * i)) --action=allow --layer4-configs=tcp:443 --src-ip-ranges="$range" \
      --dest-fqdns="${DOMAINS[$name]}" --description="$name"
  fi
  i=$((i + 1))
done
rule 2000000000 --action=deny --layer4-configs=all --dest-ip-ranges=0.0.0.0/0 --enable-logging \
  --description="Everything else"

step "GitHub Actions deploy access (Workload Identity Federation)"
exists gcloud iam workload-identity-pools describe github --location=global ||
  gcloud iam workload-identity-pools create github --location=global --display-name="GitHub Actions" >/dev/null
# Only a job in the repository's "production" environment can deploy: its deployment rules
# (main and v* tags) and the v* tag ruleset are set in the repository settings.
OIDC=(--location=global --workload-identity-pool=github
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"
  --attribute-condition="assertion.repository_owner_id=='$GITHUB_OWNER_ID' && assertion.repository=='$GITHUB_REPO' && (assertion.ref=='refs/heads/main' || assertion.ref.startsWith('refs/tags/v')) && assertion.environment=='production'")
if exists gcloud iam workload-identity-pools providers describe memjudge --location=global --workload-identity-pool=github; then
  gcloud iam workload-identity-pools providers update-oidc memjudge "${OIDC[@]}" >/dev/null
else
  gcloud iam workload-identity-pools providers create-oidc memjudge "${OIDC[@]}" \
    --issuer-uri=https://token.actions.githubusercontent.com >/dev/null
fi
retry gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$GITHUB_REPO"
bind_project "$DEPLOYER" roles/run.admin # run.admin: it makes the API service public
for sa in "$API" "$RETRIEVE" "$FINISH"; do
  retry gcloud iam service-accounts add-iam-policy-binding "$sa" --member="serviceAccount:$DEPLOYER" \
    --role=roles/iam.serviceAccountUser
done

step "Dataset mirror"
if [[ $(gcloud storage ls "gs://$PROJECT-datasets/datasets/" 2>/dev/null | wc -l) -lt 2 ]]; then
  GOOGLE_CLOUD_PROJECT="$PROJECT" "$RUNNER" mirror-datasets --bucket "$PROJECT-datasets"
fi

step "Budget"
gcloud billing budgets list --billing-account="$BILLING_ACCOUNT" --billing-project="$PROJECT" \
  --format="value(displayName)" 2>/dev/null | grep -qx "$PROJECT" ||
  gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" --billing-project="$PROJECT" \
    --display-name="$PROJECT" --budget-amount="$BUDGET" --filter-projects="projects/$PROJECT" \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0 >/dev/null

echo
echo "Done. For the deploy workflow (phase D):"
echo "  API URL           https://runner-api-$NUMBER.$REGION.run.app"
echo "  Images            $REGION-docker.pkg.dev/$PROJECT/runner"
echo "  Buckets           $PROJECT-secrets, $PROJECT-staging, $PROJECT-datasets"
echo "  KMS keys          projects/$PROJECT/locations/$REGION/keyRings/runner/cryptoKeys/{retrieve,finish}"
echo "  WIF provider      projects/$NUMBER/locations/global/workloadIdentityPools/github/providers/memjudge"
echo "  Deploy account    $DEPLOYER"
for s in "${SKIPPED[@]}"; do echo "Skipped $s"; done
