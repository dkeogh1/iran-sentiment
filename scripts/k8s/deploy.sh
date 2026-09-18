#!/usr/bin/env bash
# Pin the image tag in every job kustomization and apply the base
# (namespace + PVC). Jobs are run with run-now.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
TAG="${1:-}"
if [ -z "$TAG" ]; then
    git diff-index --quiet HEAD -- || { echo "[deploy] ERROR: dirty tree -- pass the tag build.sh printed" >&2; exit 1; }
    TAG="$(git rev-parse --short=12 HEAD)"
fi
for f in k8s/jobs/*/kustomization.yaml; do sed -i -E "s/^(\s*newTag:).*/\1 $TAG/" "$f"; done
echo "[deploy] image tag -> $TAG"
kubectl apply -k k8s/
kubectl -n iran-sentiment get pvc
