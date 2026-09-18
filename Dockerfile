# syntax=docker/dockerfile:1.7
# GPU runtime image for the stance-model experiments (k8s/ Jobs on dkbl2).
# One image serves teacher-check (API only), stance-distill, stance-local-llm
# and score-distilled. Code is BAKED in at a git sha (scripts/k8s/build.sh
# tags the image with it); data is NOT in the image -- the Jobs mount the
# iran-sentiment-data PersistentVolumeClaim at /data (IRAN_DATA_DIR).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    IRAN_DATA_DIR=/data \
    TOKENIZERS_PARALLELISM=false

# Same uid/gid as `dk` on the nodes so files on the data volume stay dk's.
RUN groupadd -g 1000 iran && useradd -m -u 1000 -g 1000 -s /bin/bash iran

WORKDIR /app
COPY requirements-gpu.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-gpu.txt

COPY config ./config
COPY src ./src
COPY README.md CLAUDE.md ./
RUN chown -R iran:iran /app

ARG GIT_SHA=unknown
ENV IRAN_GIT_SHA=$GIT_SHA
USER iran
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
