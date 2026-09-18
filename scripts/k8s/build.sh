#!/usr/bin/env bash
# Build the GPU image IN the cluster (BuildKit daemon from homelab-infra,
# driven by buildctl over kube-pod://) and push it to the in-cluster
# registry. Mirrors quant/scripts/k8s/build.sh. No docker anywhere.
#   scripts/k8s/build.sh          # tag = git sha (-dirty.<stamp> if the tree has changes)
#   scripts/k8s/build.sh mytag
set -euo pipefail
BUILDKIT_VERSION="${BUILDKIT_VERSION:-v0.33.0}"   # must match homelab-infra buildkit.yaml
BUILD_NS="${BUILD_NS:-build}"
PUSH_REPO="${PUSH_REPO:-registry.registry.svc.cluster.local:5000/iran-sentiment}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
TAG="${1:-}"
if [ -z "$TAG" ]; then
    TAG="$(git rev-parse --short=12 HEAD)"
    git diff-index --quiet HEAD -- || { TAG="$TAG-dirty.$(date -u +%Y%m%dT%H%M%S)"; echo "[build] NOTE: dirty tree -> $TAG" >&2; }
fi
BUILDCTL="$HOME/.local/bin/buildctl"
if [ ! -x "$BUILDCTL" ] || ! "$BUILDCTL" --version 2>/dev/null | grep -q "${BUILDKIT_VERSION#v}"; then
    echo "[build] installing buildctl $BUILDKIT_VERSION"; mkdir -p "$HOME/.local/bin"; tmp="$(mktemp -d)"
    curl -fsSL "https://github.com/moby/buildkit/releases/download/${BUILDKIT_VERSION}/buildkit-${BUILDKIT_VERSION}.linux-amd64.tar.gz" | tar -xz -C "$tmp" bin/buildctl
    install -m 0755 "$tmp/bin/buildctl" "$BUILDCTL"; rm -rf "$tmp"
fi
POD="$(kubectl -n "$BUILD_NS" get pod -l app=buildkitd --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[ -n "$POD" ] || { echo "[build] ERROR: no running buildkitd pod in $BUILD_NS" >&2; exit 1; }
echo "[build] daemon: $BUILD_NS/$POD   image: $PUSH_REPO:$TAG"
"$BUILDCTL" --addr "kube-pod://${POD}?namespace=${BUILD_NS}" build \
    --frontend dockerfile.v0 --local context="$REPO" --local dockerfile="$REPO" \
    --opt "build-arg:GIT_SHA=$TAG" \
    --output "type=image,name=${PUSH_REPO}:${TAG},push=true,registry.insecure=true" --progress plain
echo "[build] pushed ${PUSH_REPO}:${TAG}"; echo "[build] next: scripts/k8s/deploy.sh $TAG"
