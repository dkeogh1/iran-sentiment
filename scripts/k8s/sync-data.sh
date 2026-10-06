#!/usr/bin/env bash
# Move data between dkbl1's data/ (source of truth) and the PVC on dkbl2.
#   sync-data.sh push    # sentiment_all (+ _superseded) / reply_sentiment (+ its _columns.json) / stance_sample /
#                        # teacher_labels_replies_* / truthsocial_trump_stance parquet,
#                        # raw/truthsocial/*.jsonl, models/stance_distilled_final* -> PVC
#   sync-data.sh pull    # models/ (final dirs whole, other dirs their result files), teacher_check_*, local_llm_*,
#                        # truthsocial_trump_stance.parquet, reply_sentiment.parquet (+ _columns.json) <- PVC; then verify
#   sync-data.sh verify  # checksum dry-run of the pull set; lists anything on dkbl1 that differs from the PVC
#
# Goes over SSH straight into the PVC's local-path directory on dkbl2, not
# through kubectl cp. On 2026-09-19 a kubectl-cp pull -- one 1.7 GB stream
# through the API server on dkbl1, then a burst of short kubectl processes,
# one per small file -- coincided with dkbl1 hard-powering off mid-copy
# (homelab-infra docs/dkbl1-cstate.md: bursty idle/busy load is the known
# trigger, and the C-state pin was active). rsync is one process and one
# steady, rate-capped stream, resumable, and checked by checksum afterwards.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
NS=iran-sentiment; PVC=iran-sentiment-data
# dkbl2 over the LAN, through an ~/.ssh/config alias on dkbl1 so the address
# stays out of this public repo (k8s/README.md, *Data*). Tailscale SSH to dkbl2
# is in check mode, which wants a browser approval: no good for a script.
SYNC_HOST="${SYNC_HOST:-dkbl2-lan}"
# KB/s. ~20 MB/s keeps the transfer flat; the pulls that survived ran at 40-48.
BWLIMIT="${SYNC_BWLIMIT_KBPS:-20000}"
MODE="${1:?usage: sync-data.sh push|pull|verify}"

PV=$(kubectl -n $NS get pvc $PVC -o jsonpath='{.spec.volumeName}')
PVC_DIR=$(kubectl get pv "$PV" -o jsonpath='{.spec.local.path}{.spec.hostPath.path}')
[ -n "$PVC_DIR" ] || { echo "cannot resolve the host path of PVC $NS/$PVC" >&2; exit 1; }
rc=0; ssh "$SYNC_HOST" "command -v rsync >/dev/null || exit 3; test -d '$PVC_DIR' || exit 4" || rc=$?
case $rc in
  0) ;;
  3) echo "$SYNC_HOST: rsync missing" >&2; exit 1 ;;
  # local-path creates the storage dir 0700 root; dk needs to traverse it
  # (homelab-infra docs/incidents.md, 2026-09-19). It reverted once already.
  4) echo "$SYNC_HOST: cannot reach $PVC_DIR. If the PVC is Bound, on dkbl2 run:" \
          "sudo chmod o+x $(dirname "$PVC_DIR")" >&2; exit 1 ;;
  *) echo "$SYNC_HOST: ssh failed (exit $rc)" >&2; exit 1 ;;
esac

# -a keeps mtimes so re-runs are cheap; -c decides by checksum, which is what
# catches the 0-byte and truncated files a cut-off copy leaves behind; -W sends
# whole files (no delta computation on either end); --partial keeps a cut-off
# file so the next run resumes it.
RSYNC=(rsync -acW --partial --bwlimit="$BWLIMIT" --human-readable --info=progress2,stats1)
# Final model dirs come back whole; sweep / holdout dirs only their top-level
# result files (the trainer and CV dirs are large and reproducible).
MODEL_FILTER=(--include='/stance_distilled_final*/***' --include='/*/' --include='/*/*.json' --include='/*/*.parquet' --exclude='*')
PROCESSED_FILTER=(--include='teacher_check_*' --include='local_llm_*' --include='truthsocial_trump_stance.parquet' --include='reply_sentiment.parquet'
                  --include='reply_sentiment_columns.json' --exclude='*')

verify() {
    # Itemised checksum dry-run; only file lines count (directory mtimes may differ).
    local diff
    diff=$( { rsync -acn --itemize-changes "${MODEL_FILTER[@]}" "$SYNC_HOST:$PVC_DIR/models/" data/models/;
             rsync -acn --itemize-changes "${PROCESSED_FILTER[@]}" "$SYNC_HOST:$PVC_DIR/processed/" data/processed/; } | grep -E '^[<>]f' || true)
    if [ -n "$diff" ]; then
        echo "differs from the PVC:"; echo "$diff"; return 1
    fi
    echo "verify: data/models and data/processed match the PVC"
}

case "$MODE" in
  push)
    ssh "$SYNC_HOST" "mkdir -p '$PVC_DIR/processed' '$PVC_DIR/raw/truthsocial' '$PVC_DIR/models'"
    files=(data/processed/sentiment_all.parquet)
    # The feed parquet goes up too: pull brings it back, so a feed Job must
    # start from dkbl1's copy, not build a fresh one that overwrites it.
    # sentiment_all_superseded holds the Haiku scores of the posts
    # x-backfill-text changed; without it the teacher-check Job resamples
    # (stance_local.training_frame) and pays for ~400 new calls.
    # reply_sentiment_columns.json records which model wrote each reply-domain
    # column (stance_local.reply_columns_path); it travels with the parquet.
    for f in data/processed/reply_sentiment.parquet data/processed/reply_sentiment_columns.json \
             data/processed/stance_sample.parquet data/processed/teacher_labels_replies_*.parquet \
             data/processed/truthsocial_trump_stance.parquet data/processed/sentiment_all_superseded.parquet; do
        [ -f "$f" ] && files+=("$f"); done
    "${RSYNC[@]}" "${files[@]}" "$SYNC_HOST:$PVC_DIR/processed/"
    "${RSYNC[@]}" data/raw/truthsocial/*.jsonl "$SYNC_HOST:$PVC_DIR/raw/truthsocial/"
    # Final model dirs (~1.7 GB each), so the score Jobs run on a fresh PVC.
    "${RSYNC[@]}" data/models/stance_distilled_final* "$SYNC_HOST:$PVC_DIR/models/"
    ssh "$SYNC_HOST" "du -sh '$PVC_DIR/processed' '$PVC_DIR/raw/truthsocial' '$PVC_DIR/models'"
    ;;
  pull)
    mkdir -p data/models data/processed
    "${RSYNC[@]}" "${MODEL_FILTER[@]}" "$SYNC_HOST:$PVC_DIR/models/" data/models/
    "${RSYNC[@]}" "${PROCESSED_FILTER[@]}" "$SYNC_HOST:$PVC_DIR/processed/" data/processed/
    verify
    ls -la data/models data/processed | head -40
    ;;
  verify) verify ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
