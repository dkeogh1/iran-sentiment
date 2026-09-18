#!/usr/bin/env bash
# Run one experiment Job and follow its log.
#   run-now.sh teacher-check | distill | local-llm | score-distilled
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS=iran-sentiment
WHAT="${1:?usage: run-now.sh teacher-check|distill|local-llm|score-distilled}"
[ -d "$REPO/k8s/jobs/$WHAT" ] || { echo "unknown job: $WHAT" >&2; exit 2; }
JOB="iran-$WHAT"
kubectl -n $NS delete job "$JOB" --ignore-not-found --wait=true
kubectl apply -k "$REPO/k8s/jobs/$WHAT"
echo "[run-now] job/$JOB -- waiting for the pod"
for _ in $(seq 1 90); do
    POD="$(kubectl -n $NS get pod -l job-name="$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    [ -n "$POD" ] && break; sleep 2
done
[ -n "${POD:-}" ] || { echo "[run-now] no pod appeared; kubectl -n $NS describe job/$JOB" >&2; exit 1; }
kubectl -n $NS wait --for=condition=Ready "pod/$POD" --timeout=900s >/dev/null 2>&1 || true
kubectl -n $NS logs -f "pod/$POD" || true
kubectl -n $NS wait --for=condition=complete "job/$JOB" --timeout=10s >/dev/null 2>&1 \
    && { echo "[run-now] job/$JOB COMPLETE"; exit 0; }
echo "[run-now] job/$JOB did not report Complete:"; kubectl -n $NS get "job/$JOB" -o wide; exit 1
