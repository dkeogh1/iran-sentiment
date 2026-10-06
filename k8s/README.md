# Stance-model experiments on the cluster

Two ways to replace the per-post Haiku stance call with something that
runs free on dkbl2's RTX 3080, plus a check that Haiku is a fit teacher:

| Job | Command in the image | GPU | What it answers |
|---|---|---|---|
| `teacher-check` | `teacher-check` | no | does Opus 5 disagree with the Haiku labels, and where? |
| `distill` | `stance-distill` | yes | can a fine-tuned RoBERTa-large reproduce the teacher on held-out posts? |
| `local-llm` | `stance-local-llm` | yes | can Qwen2.5-7B (4-bit) with the same prompt match the teacher? |
| `score-distilled` | `score-distilled` | yes | population-level stance for every reply with the distilled model |
| `local-llm-qwen3`, `local-llm-qwen3-think` | `stance-local-llm --model Qwen/Qwen3-8B [--thinking]` | yes | the 2025 Qwen generation, reasoning off / on, same holdout sample |
| `distill-opus` | `stance-distill --recipe deb-128-1e5-3 --label-col score_opus --fit-all` | yes | the sweep winner retrained on the Opus 5 labels (`relabel merge` first, then `sync-data.sh push`); holdout eval + final model |
| `score-distilled-opus` | `score-distilled --model-dir .../stance_distilled_final_score_opus --col score_opus_distilled` | yes | population reply stance from the Opus-taught model (the reply scorer of record) |
| `distill-opus-mixed`, `score-distilled-mixed` | `stance-distill ... --with-replies` / `score-distilled --col score_mixed_distilled` | yes | the Opus retrain with the 958 Opus-labelled replies mixed in (reply tiers held out 20%), then replies rescored |
| `score-trump-feed` | `score-posts /data/raw/truthsocial/realDonaldTrump.jsonl ...` | yes | Trump's own Truth Social posts scored with the Opus-taught model |
| `sweep-opus` | `stance-sweep --label-col score_opus` | yes | the same sweep + CV + final fit on the Opus labels (~3 h) |
| `sweep` | `stance-sweep` | yes | every recipe in `DISTILL_SWEEP` on the shared split, 5-fold CV on the best, final fit on all labels (~3 h, resumable) |
| `reply-distill-lopo` | `reply-distill --lopo` | yes | does the posts-only scorer of record, fine-tuned on the v2 Opus reply labels, track them better, with the reply alone (`text`) or the pair (reply, parent post) (`ctx`)? Leave-one-post-out over the 8 posts, 2 seeds, against the published `score_opus_distilled` on the same rows -> `models/reply_ctx_lopo/` (~1.5-2.5 h, resumable) |
| `reply-distill-final` | `reply-distill --fit-all --variant ctx` | yes | the variant LOPO picks, fit on all 1,157 labelled replies -> `models/stance_distilled_final_reply_ctx` (~5-10 min) |
| `score-replies-ctx` | `score-replies-ctx --model-dir .../stance_distilled_final_reply_ctx --col score_ctx_distilled` | yes | every reply with text scored by that model, with its parent post at 384 tokens (~45-75 min). Not for `reply-population`: the model saw the labelled sample |
| `reply-crossfit-ctx` | `reply-distill --crossfit --variant ctx` | yes | per post, a fit on the other seven posts' labels scores that post's replies -> `score_ctx_crossfit`, the column `reply-population --col` can weight (~1.5-2 h, resumable per post) |

Layout follows the quant tenant conventions (homelab-infra
`docs/k8s-workloads.md`): one namespace, image built in-cluster and
pulled as `localhost:30500/iran-sentiment:<sha>`, pods as uid 1000,
Secret from `.env`. Data is a local-path PVC on dkbl2 filled by
`scripts/k8s/sync-data.sh`, which rsyncs over SSH straight into the PVC's
directory on dkbl2 (over the LAN, rate-capped, checksum-verified) rather
than through `kubectl cp` -- a kubectl-cp pull through the API server
coincided with dkbl1 hard-powering off on 2026-09-19. The source of truth
stays in `data/` on dkbl1. It reaches dkbl2 as `dkbl2-lan`, an alias in
dkbl1's `~/.ssh/config` (`HostName` = dkbl2's LAN address, `User dk`),
or whatever `SYNC_HOST` says.

```bash
scripts/k8s/build.sh                 # buildctl -> BuildKit -> registry (tag = git sha)
scripts/k8s/deploy.sh <tag>          # pin tag in k8s/jobs/*/kustomization.yaml, apply base
scripts/k8s/secrets.sh               # ANTHROPIC_API_KEY -> Secret (teacher-check only)
kubectl apply -f k8s/data-sync.yaml  # helper pod: local-path binds the PVC on first use
scripts/k8s/sync-data.sh push        # parquet + replies + final models -> PVC
scripts/k8s/run-now.sh teacher-check # ~$1 on Opus 5, prints per-tier agreement
scripts/k8s/run-now.sh distill       # ~15 min on the 3080
scripts/k8s/run-now.sh local-llm     # first run downloads ~15 GB of weights into /data/hf
scripts/k8s/run-now.sh score-distilled
scripts/k8s/run-now.sh sweep         # overnight; sweep_results.json / cv_results.json / sweep_summary.json
scripts/k8s/run-now.sh reply-distill-lopo   # models/reply_ctx_lopo/lopo_results.json + lopo_predictions.parquet
scripts/k8s/sync-data.sh pull        # models/, metrics, parquet <- PVC, then a checksum verify
scripts/k8s/sync-data.sh verify      # checksum dry-run only: what on dkbl1 differs from the PVC
```

Metrics land in `data/processed/*.json` and
`data/models/stance_distilled/distill_metrics.json`: Pearson, MAE, sign
agreement and sign-flip rate against the teacher, overall and per tier,
with RoBERTa valence on the same rows as the baseline to beat.

Decision rule: a candidate replaces Haiku only if its per-tier sign
agreement with the teacher is at least as good as Haiku's own agreement
with Opus 5 from the teacher check, `religious_authority` included.
Scores from a new scorer never mix with `score_llm` in one series --
rescore the whole set or keep a separate column.

Scoring runs at the model's training length, read from its `recipe.txt`
(`--max-len` overrides). A final model dir is never overwritten: to retrain
into one, move it aside first. Both rules come from 2026-09-30, when the
older reply scores turned out to come from a fit the sweep had replaced in
place (`docs/decisions.md`). On the 3080, 98.7k replies take ~20 min at 256
tokens and 15.6k take ~2 min at 128.

Reply-domain scorer (2026-10-06). The posts-only scorer never sees the
post a reply answers, which the v2 labels depend on (0.58 Pearson
against them, 0.32 under the ceasefire post, 0.14 under the deal post).
The `reply-*` Jobs fine-tune it warm on the labelled replies
(`settings.REPLY_DISTILL`: 3 epochs at 1e-5, effective batch 16, every
seed set before the model is built). `ctx` feeds the pair (reply, post)
cut only on the post (`stance_local.reply_inputs`); `text` feeds the
reply alone at the base model's 256 tokens. Pick the variant from
`lopo_results.json` (`summary`: pooled ALL and per post, seed mean with
min and max), never from the final fit's own rows; if it is `text`,
change `--variant` in `reply-distill-final` and `reply-crossfit-ctx`, and
the model dir and column in `score-replies-ctx`. Order: `reply-distill-lopo`,
pull and read; then `reply-crossfit-ctx` for `reply-population --col
score_ctx_crossfit`, and if a full-sample model is wanted for anything
else, `reply-distill-final` then `score-replies-ctx`. A column fit on the
labelled replies has in-sample errors on them, so `reply-population`
refuses it: its correction and interval would shrink with no gain in
accuracy. The cross-fit column scores each post with a model that never
saw that post's labels. `data/processed/reply_sentiment_columns.json`
records which model wrote each of these columns, and a column never takes
a second model's scores. Pull after each Job and before the next push:
push sends dkbl1's `reply_sentiment.parquet` and the column record up and
would replace scores the PVC holds but dkbl1 does not.

Known limits: the eGPU on dkbl2 occasionally wedges (homelab-infra
`docs/egpu-recovery.md`); a Job then stays Pending until the node
re-advertises the GPU. Delete `ns iran-sentiment` when the experiments
are over -- nothing here is scheduled.
