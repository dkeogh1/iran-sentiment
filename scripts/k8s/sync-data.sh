#!/usr/bin/env bash
# Move data between dkbl1's data/ (source of truth) and the PVC on dkbl2.
#   sync-data.sh push   # sentiment_all.parquet, reply_sentiment.parquet, replies_*.jsonl -> /data
#   sync-data.sh pull   # models/, *metrics*.json, teacher_check_*, local_llm_*, reply_sentiment.parquet <- /data
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
NS=iran-sentiment; POD=data-sync
MODE="${1:?usage: sync-data.sh push|pull}"
kubectl apply -f k8s/data-sync.yaml >/dev/null
kubectl -n $NS wait --for=condition=Ready pod/$POD --timeout=300s >/dev/null
case "$MODE" in
  push)
    kubectl -n $NS cp data/processed/sentiment_all.parquet $POD:/data/processed/sentiment_all.parquet
    [ -f data/processed/reply_sentiment.parquet ] && kubectl -n $NS cp data/processed/reply_sentiment.parquet $POD:/data/processed/reply_sentiment.parquet
    for f in data/raw/truthsocial/*.jsonl; do kubectl -n $NS cp "$f" "$POD:/data/raw/truthsocial/$(basename "$f")"; done
    kubectl -n $NS exec $POD -- sh -c 'du -sh /data/processed /data/raw/truthsocial'
    ;;
  pull)
    mkdir -p data/models data/processed
    kubectl -n $NS exec $POD -- sh -c 'ls /data/models 2>/dev/null' | while read -r m; do
        kubectl -n $NS cp "$POD:/data/models/$m" "data/models/$m"; done
    for f in $(kubectl -n $NS exec $POD -- sh -c 'cd /data/processed && ls teacher_check_* local_llm_* truthsocial_trump_stance.parquet 2>/dev/null'); do
        kubectl -n $NS cp "$POD:/data/processed/$f" "data/processed/$f"; done
    kubectl -n $NS exec $POD -- test -f /data/processed/reply_sentiment.parquet \
        && kubectl -n $NS cp $POD:/data/processed/reply_sentiment.parquet data/processed/reply_sentiment.parquet
    ls -la data/models data/processed | head -40
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
kubectl -n $NS delete pod $POD --wait=false >/dev/null
