# Stance-model experiments on the cluster

Two ways to replace the per-post Haiku stance call with something that
runs free on dkbl2's RTX 3080, plus a check that Haiku is a fit teacher:

| Job | Command in the image | GPU | What it answers |
|---|---|---|---|
| `teacher-check` | `teacher-check` | no | does Opus 5 disagree with the Haiku labels, and where? |
| `distill` | `stance-distill` | yes | can a fine-tuned RoBERTa-large reproduce the teacher on held-out posts? |
| `local-llm` | `stance-local-llm` | yes | can Qwen2.5-7B (4-bit) with the same prompt match the teacher? |
| `score-distilled` | `score-distilled` | yes | population-level stance for all 83k replies with the distilled model |
| `local-llm-qwen3`, `local-llm-qwen3-think` | `stance-local-llm --model Qwen/Qwen3-8B [--thinking]` | yes | the 2025 Qwen generation, reasoning off / on, same holdout sample |
| `sweep` | `stance-sweep` | yes | every recipe in `DISTILL_SWEEP` on the shared split, 5-fold CV on the best, final fit on all labels (~3 h, resumable) |

Layout follows the quant tenant conventions (homelab-infra
`docs/k8s-workloads.md`): one namespace, image built in-cluster and
pulled as `localhost:30500/iran-sentiment:<sha>`, pods as uid 1000,
Secret from `.env`. Data is a local-path PVC on dkbl2 filled by
`kubectl cp`; the source of truth stays in `data/` on dkbl1.

```bash
scripts/k8s/build.sh                 # buildctl -> BuildKit -> registry (tag = git sha)
scripts/k8s/deploy.sh <tag>          # pin tag in k8s/jobs/*/kustomization.yaml, apply base
scripts/k8s/secrets.sh               # ANTHROPIC_API_KEY -> Secret (teacher-check only)
scripts/k8s/sync-data.sh push        # parquet + replies -> PVC
scripts/k8s/run-now.sh teacher-check # ~$1 on Opus 5, prints per-tier agreement
scripts/k8s/run-now.sh distill       # ~15 min on the 3080
scripts/k8s/run-now.sh local-llm     # first run downloads ~15 GB of weights into /data/hf
scripts/k8s/run-now.sh score-distilled
scripts/k8s/run-now.sh sweep         # overnight; sweep_results.json / cv_results.json / sweep_summary.json
scripts/k8s/sync-data.sh pull        # models/, metrics, parquet <- PVC
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

Known limits: the eGPU on dkbl2 occasionally wedges (homelab-infra
`docs/egpu-recovery.md`); a Job then stays Pending until the node
re-advertises the GPU. Delete `ns iran-sentiment` when the experiments
are over -- nothing here is scheduled.
