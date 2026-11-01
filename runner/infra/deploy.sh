#!/usr/bin/env bash
# Deploy the hosted runner from an image digest (ADR 0009, 0011): one job per hosted
# connector, the finish job, then the Runner API, and delete releases replaced more than
# 14 days ago. Run by .github/workflows/runner.yml; redeploying a release replaces it.
#
# Needs PROJECT, REGION, REPO, DIGEST, API_URL, SITE_URL, gcloud logged in as
# runner-deployer, and docker able to pull REPO.
set -euo pipefail
: "${PROJECT:?}" "${REGION:?}" "${REPO:?}" "${DIGEST:?}" "${API_URL:?}" "${SITE_URL:?}"

export CLOUDSDK_CORE_PROJECT="$PROJECT"
IMAGE="$REPO@$DIGEST"
COMMIT="${GITHUB_SHA:-$(git rev-parse HEAD)}"
KMS="projects/$PROJECT/locations/$REGION/keyRings/runner/cryptoKeys"
SA() { echo "runner-$1@$PROJECT.iam.gserviceaccount.com"; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The image itself says what it deploys, so naming rules live only in the runner code.
docker run --rm "$IMAGE" deploy-jobs >"$WORK/jobs.txt"
RELEASE=$(awk '$1 == "release" { print $2 }' "$WORK/jobs.txt")
echo "Deploying release $RELEASE from $IMAGE"

cat >"$WORK/env.yaml" <<EOF
MJR_PROJECT: "$PROJECT"
MJR_REGION: "$REGION"
MJR_SECRETS_BUCKET: "$PROJECT-secrets"
MJR_STAGING_BUCKET: "$PROJECT-staging"
MJR_DATASETS_BUCKET: "$PROJECT-datasets"
MJR_RETRIEVE_KMS_KEY: "$KMS/retrieve"
MJR_FINISH_KMS_KEY: "$KMS/finish"
MJR_API_URL: "$API_URL"
MJR_SITE_URL: "$SITE_URL"
MEMJUDGE_IMAGE_DIGEST: "$DIGEST"
MEMJUDGE_GIT_COMMIT: "$COMMIT"
EOF
LABELS="memjudge-release=$RELEASE,memjudge-deployed=$(date +%s)"

while read -r kind name subnet memory; do
  [[ "$kind" == release ]] && continue
  args=(--image="$IMAGE" --region="$REGION" --service-account="$(SA "$kind")"
    --network=runner --subnet="$subnet" --vpc-egress=all-traffic
    --cpu=2 --memory="$memory" --tasks=1 --max-retries=0 --task-timeout=24h
    --env-vars-file="$WORK/env.yaml" --labels="$LABELS")
  # Only retrieve jobs run from the environment with the connector SDKs (ADR 0009).
  if [[ "$kind" == finish ]]; then
    args+=(--command=/opt/core/bin/memjudge-runner)
    args+=(--set-secrets="/secrets/signing-key/value=signing-key:latest,/secrets/r2/value=r2:latest,/secrets/github-app/value=github-app:latest")
  else
    args+=(--command=/opt/retrieve/bin/memjudge-runner)
  fi
  echo "== job $name ($subnet, $memory)"
  gcloud run jobs deploy "$name" "${args[@]}" --quiet
done <"$WORK/jobs.txt"

# The API goes last, so it never starts jobs that don't exist yet.
echo "== service runner-api"
gcloud run deploy runner-api --image="$IMAGE" --region="$REGION" --service-account="$(SA api)" \
  --no-invoker-iam-check --cpu=1 --memory=1Gi --max-instances=3 \
  --env-vars-file="$WORK/env.yaml" --labels="memjudge-release=$RELEASE" \
  --set-secrets="/secrets/github-oauth/value=github-oauth:latest,/secrets/session-secret/value=session-secret:latest" \
  --quiet

echo "== stale releases"
gcloud run jobs list --region="$REGION" --format=json >"$WORK/all-jobs.json"
for stale in $(docker run -i --rm "$IMAGE" stale-releases --days 14 <"$WORK/all-jobs.json"); do
  for job in $(gcloud run jobs list --region="$REGION" --format="value(metadata.name)" \
    --filter="metadata.labels.memjudge-release=$stale"); do
    echo "deleting $job (release $stale)"
    gcloud run jobs delete "$job" --region="$REGION" --quiet
  done
done

echo "== smoke check"
healthy=""
for _ in 1 2 3 4 5 6; do
  if curl -fsS "$API_URL/health"; then healthy=yes && break; fi
  sleep 10
done
[[ -n "$healthy" ]] || { echo "The API did not answer on $API_URL/health" >&2; exit 1; }
echo
curl -fsS "$API_URL/options" | python3 -c '
import json, sys
o = json.load(sys.stdin)
datasets, connectors = [d["name"] for d in o["datasets"]], [c["id"] for c in o["connectors"]]
print("datasets:", ", ".join(datasets))
print("connectors:", ", ".join(connectors))
assert datasets and connectors, "the API offers nothing"'
