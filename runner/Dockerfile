# MemJudge hosted runner image: API, retrieve job and finish job.
# Build from the repository root:  docker build -f runner/Dockerfile .
# Deployed by digest only; the digest and commit are recorded in every hosted result.

# FalkorDB, the graph database the zep/graphiti connector needs. Its job starts it inside
# the container (LOCAL_SERVICES in config.py), empty for every run.
FROM falkordb/falkordb:v4.22.1@sha256:b0bdcac2cbaff989e77cb0610439b7a9289e86de033ccbbd4368b66fb680ae7a AS falkordb

FROM python:3.11.17-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/app/models/tiktoken \
    FASTEMBED_CACHE_PATH=/app/models/fastembed

# Same Debian release as the FalkorDB image; libgomp1 is the one library slim lacks.
RUN apt-get update  && apt-get install -y --no-install-recommends libgomp1  && rm -rf /var/lib/apt/lists/*
COPY --from=falkordb /usr/local/bin/redis-server /opt/falkordb/redis-server
COPY --from=falkordb /var/lib/falkordb/bin/falkordb.so /opt/falkordb/falkordb.so

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY connectors ./connectors
COPY runner ./runner

# Two environments with MemJudge and the runner (ADR 0009). /opt/core has no connector
# SDK: the API, the finish job and the build commands run from it. /opt/retrieve adds
# every connector's SDK, and only retrieve jobs run from it (deploy.sh sets their
# command), because a wheel's .pth file runs in every process of its environment at
# startup. Both are made without pip (so without setuptools and its .pth file) and filled
# by the base image's pip. Every third-party package, the spaCy model included, comes from
# a hash-pinned lock (runner/core.in and runner/requirements.in say how to regenerate
# them), so the image holding the signing key gets no unreviewed versions; pip check fails
# the build if a lock no longer covers what the packages need, and check-core-env if
# /opt/core holds anything else or a startup hook. Models that SDKs would otherwise
# download at run time are fetched now, so jobs need no egress to model hosts. They are
# made readable for the job user: some caches write root-only files (huggingface_hub's
# tree cache), and fastembed then cannot load the model offline.
RUN python -m venv --without-pip /opt/core \
 && python -m venv --without-pip /opt/retrieve \
 && pip --python /opt/core/bin/python install --require-hashes --no-deps -r runner/core.lock \
 && pip --python /opt/core/bin/python install --no-deps --no-build-isolation . ./runner \
 && pip --python /opt/core/bin/python check \
 && /opt/core/bin/memjudge-runner check-core-env runner/core.lock \
 && pip --python /opt/retrieve/bin/python install --require-hashes --no-deps -r runner/requirements.lock \
 && pip --python /opt/retrieve/bin/python install --no-deps --no-build-isolation . ./runner \
 && pip --python /opt/retrieve/bin/python check \
 && /opt/core/bin/python -c "from memjudge.tokens import count_tokens; count_tokens('warm')" \
 && /opt/retrieve/bin/python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')" \
 && chmod -R a+rX /app/models \
 && rm -rf /app/src /app/connectors /app/runner

# Form options and cost estimates. The datasets are downloaded to a throwaway cache in
# this same step: the image must not contain them (some licences forbid redistribution).
RUN MEMJUDGE_CACHE=/tmp/build-cache /opt/core/bin/memjudge-runner build-options -o /app/options.json \
 && rm -rf /tmp/build-cache

# Runtime: no network for model hubs or telemetry, writable state only under /tmp.
ENV HF_HUB_OFFLINE=1 \
    MEM0_TELEMETRY=False \
    HOME=/tmp \
    MEMJUDGE_CACHE=/tmp/memjudge \
    MJR_OPTIONS_FILE=/app/options.json

USER 10001:10001
ENTRYPOINT ["/opt/core/bin/memjudge-runner"]
CMD ["api"]
