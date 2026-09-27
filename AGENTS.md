# iran-sentiment

Stance analysis of US political messaging on the 2026 Iran war: X broadcaster
accounts in political tiers, plus Trump's Truth Social feed and the replies to
six of his posts, scored by Claude models. `README.md` is the public write-up;
where the data stands is in [docs/STATUS.md](docs/STATUS.md).

This repo is **public** on GitHub: no IP addresses, account IDs, bucket names,
credit balances or personal data in anything committed.

## Ask first

Hand these to the user, with the estimate, instead of running them:

- **Paid runs.** Estimate first, then ask.
  - `collect` and `run-all` (X API, $0.005 per tweet read): run
    `python -m src.cli collect --estimate` (cache only, no API calls) and ask
    for the current X credit balance.
  - `relabel submit|resubmit` and `topic-label submit` (Anthropic Batch API):
    `relabel estimate` / `topic-label estimate` print the cost.
  - `analyze --llm`, `stance`, `teacher-check`, `reply-teacher-check` (direct
    Anthropic API): estimate from the post count (Haiku has run about
    $0.50-1.50 per 1,000 posts, README *Cost*; `teacher-check` ~$1 and
    `reply-teacher-check` ~$2 on Opus). Narrow `analyze --llm` with
    `--llm-tiers` / `--llm-accounts` to the subset that needs it.
- **The user's Truth Social account.** `ts-login` (the security code goes to
  the user), `collect-replies`, and `collect-truth` unless `--anonymous`. Free,
  but rate-limited and tied to the account.
- **Cluster changes.** The global list (`secrets.sh` is the user's, and so is
  deleting the namespace). Deploys are yours: `build.sh`, `deploy.sh`,
  `run-now.sh` for Jobs that call no paid API, and `sync-data.sh`, which rsyncs
  over SSH (the `dkbl2-lan` alias in `~/.ssh/config`) into dkbl2's PVC
  directory: this repo's one exception to reading tenant data through kubectl
  (see *Never*). A Job that calls a paid API (e.g. `teacher-check`) is a paid
  run: estimate and ask first.

## Never

- Never `kubectl cp` bulk data to or from the cluster: `scripts/k8s/sync-data.sh`
  is the only path (one rate-capped, checksum-verified rsync; why in
  [docs/decisions.md](docs/decisions.md)).
- Never re-fetch cached X data without `--force` and a reason: it is paid for.
- Never add a paid API call without a cap in `config/settings.py`, an estimate
  that makes no calls, and an on-disk cache that is skipped by default
  (`--force` to redo).
- Never schedule `collect` (no CronJob, no host cron): every run costs money.
- Never add one-off scripts. New behaviour is a subcommand in `src/cli.py`,
  kept thin over importable modules; tunables go in `config/settings.py`,
  lists (accounts, search terms, events, tracked posts) in their `config/`
  module. A one-off data migration is run and deleted, never committed.
- `data/raw/` holds ~83k replies from private individuals: keep rows out of
  git, issues and chat. Only the pipeline's own scoring calls send them out.

## Fast path

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,truthsocial]'
python -m pytest tests                # stub clients, no spend
ruff check <paths> && ruff format <paths>
```

- The system Python has no pip. If `python3 -m venv` fails, the user installs
  `python3.12-venv` with apt. Always work inside `.venv`.
- `.env` comes from the sops-encrypted `secrets.env` (the user runs
  `sops -d secrets.env > .env`; README *Setup*). Without the key, copy
  `.env.example`. `.env.example` lists every key the code reads.
- The tree is not clean under `ruff check` or `ruff format` yet (line length
  100): lint and format only the files you touch, never the whole repo in a
  feature commit.

## Running things

Everything goes through one Click CLI, `python -m src.cli <command>`:

```bash
python -m src.cli test       # verify X API credentials (free)
python -m src.cli status     # show what's cached, what's missing
python -m src.cli collect    # incremental fetch -- appends only new tweets since last run
python -m src.cli analyze    # score all cached data (VADER + RoBERTa; add --llm for Haiku stance)
python -m src.cli visualize  # regenerate all figures
python -m src.cli summary    # stats tables (default --score is the stance of record)
python -m src.cli phases     # tier x phase stance with CIs, split into war posts / topic share
                             #   [--source x|trump] [--by tier|user] [--topic llm|keyword|either]
python -m src.cli reply-population  # reply audience's Opus-corrected stance per post (free)
python -m src.cli teacher-retest    # Opus label test-retest on 497 posts (free)
python -m src.cli export-web # chart JSON for the dkweb blog post
python -m src.cli backup     # sync data/ to the off-box S3 bucket (after any paid run)
```

Refresh order (the paid steps are *Ask first*): Truth Social first, it is
free (`collect-truth`, `collect-replies`, `event-study`); then
`collect --estimate` and `collect`, refreshing heavy accounts before the
timeline horizon eats the gap; `analyze`; `relabel submit` -> `status` ->
`collect` -> `merge` for the new posts; `topic-label estimate|submit|status|collect`;
`phases`, `visualize`, `export-web`; `backup`.

## Layout

```
config/  settings.py (paths, caps, batch sizes, models, colours), accounts.py (tiers),
         timeline.py (events), tracked_posts.py (Trump posts for reply analysis)
src/     cli.py (every command), backup.py
  collectors/     x_collector.py, truthsocial_collector.py
  analysis/       sentiment.py (VADER, RoBERTa, LLM stance), event_study.py (replies),
                  relabel.py, topic_label.py (Batch API), inference.py (CIs, shift-share),
                  stance_local.py (GPU experiments)
  visualization/  plots.py, web_export.py
data/    raw/x/<handle>.jsonl, raw/truthsocial/, processed/sentiment_all.parquet,
         processed/reply_sentiment.parquet, models/   (gitignored)
k8s/, scripts/k8s/   GPU experiment Jobs
```

## Where things run

- The pipeline runs in the host venv on dkbl1 (CPU), by hand. Nothing is
  scheduled.
- GPU work (stance-model experiments) runs as one-shot Jobs on dkbl2's RTX 3080
  from `k8s/` (runbook and decision rule: [k8s/README.md](k8s/README.md)).
  Nothing stays deployed: the user deletes the namespace when experiments
  end. Reviving it is `build.sh`, `deploy.sh <tag>`, `sync-data.sh push`, then
  `run-now.sh <job>`, which you run (outside quant's windows), plus
  `secrets.sh` (teacher-check only), which the user runs.
- `data/` on dkbl1 is the source of truth. The PVC on dkbl2 is a working copy.
- `export-web` writes into the sibling repo `~/repos/dkweb`
  (`WEB_EXPORT_DIR`); the agent needs write access to it.

## Conventions and invariants

### Tiers and scorers

- `config/accounts.py` groups handles into tiers: admin, maga_prowar,
  maga_antiwar, opposition, media, religious_authority, plus `search`
  (keyword-query public-sentiment proxy). MTG is `@mtgreenee`.
- `settings.STANCE_SCORE_COL` (`score_opus`, Claude Opus 5 via `relabel`) is
  the stance of record; `summary`, `event-study` and the plots default to it.
  `score_opus_distilled` (DeBERTa distilled from the Opus labels) is the reply
  scorer of record. Other columns: `score_vader`, `score_transformer`
  (RoBERTa valence, a quick look only), `score_llm` (Haiku). Anything
  published or compared across tiers uses `score_opus`. Never mix scorers in
  one series; a new scorer gets its own column or rescores the whole set.
- **Report war-post stance, not the all-post mean.** The prompt makes Opus give
  off-topic posts ("Deport them!", "Amen") a stance from the author, so a
  tier's all-post mean moves with its war share. The about-the-war flag is the
  Haiku label from `topic-label` (`TOPIC_SOURCE = "either"` also accepts the
  keyword `WAR_TOPIC_PATTERN`, which alone misses about half the war posts).
- **Faith-based voices need LLM stance.** VADER and RoBERTa sign-flip religious
  anti-war rhetoric ("peace", "mercy" read as positive). Never report a
  valence score for `religious_authority` or any future faith-based tier
  unless flagged as miscalibrated. See README *The religious sign flip*.
- The LLM parsers strip ```` ```json ```` fences before `json.loads`
  (`sentiment.py`, `event_study.py`): Haiku wraps its JSON despite the
  prompt. Keep stripping; don't prompt-engineer it away.
- Reply population shares come from `reply-population` (Opus-corrected), not
  raw `score_opus_distilled`, which runs ~8 points pro-war.
- `analyze --llm` and `event-study` / `stance` are restart-safe and
  incremental: only posts or replies without a score are sent.

### X collection and budget

- Caps: `MAX_TWEETS_PER_USER` is a hard cap; sample prolific accounts with
  `ACCOUNT_CAP_OVERRIDES`. A real `collect` refuses to start when its maximum
  exceeds `X_RUN_BUDGET_USD` (`config/settings.py`).
- `collect` is incremental per account (only tweets newer than the cached
  `created_at`). Long gaps are walked oldest-slice-first in
  `GAP_FILL_SLICE_DAYS` windows with the cap shared across slices: the timeline
  endpoint is newest-first, so one capped fetch over a long gap would keep the
  newest N and drop the rest.
- X's user timeline returns only an account's latest ~3,200 tweets. Refresh
  heavy accounts at least every ~2 months or the gap is lost for good.
- Keyword search (`/search/recent`) reaches back only ~7 days.
- To add an account or search term, edit `config/accounts.py` and rerun.

### Truth Social

- Plain httpx/requests get 403 from Cloudflare; everything goes through
  `curl_cffi` (the `truthsocial` extra). `collect-truth` only appends runs
  that reach the cache edge (Truth Social ignores `min_id`, so it walks
  backward): authenticated, it refuses a batch that stops short; `--anonymous`
  holds pages in `<handle>.partial.jsonl` until it gets there, so a killed
  run resumes.
- Replies use the v2 descendants endpoint (`TS_DESCENDANTS_PATH`; v1 is dead)
  with the `TRUTHSOCIAL_TOKEN` that `ts-login` writes to `.env`. After a
  `ts-login`, the user re-encrypts `secrets.env`.
- truthbrush reads `TRUTHSOCIAL_USERNAME` / `TRUTHSOCIAL_PASSWORD`; parts of our
  code read `TRUTH_SOCIAL_*`. Keep both set. Details:
  [docs/truthsocial-api.md](docs/truthsocial-api.md).

### CPU and memory (dkbl1 powers off under bursty or sustained load)

Guardrails in `src/analysis/sentiment.py` + `config/settings.py`; keep them:

1. Batched inference (`ROBERTA_BATCH_SIZE` 16): never score posts one by one.
2. Thread cap (`TORCH_NUM_THREADS` 2, plus `OMP_NUM_THREADS` /
   `MKL_NUM_THREADS`), set before the pipeline is imported.
3. Checkpoints inside the RoBERTa loop every
   `ROBERTA_CHECKPOINT_EVERY_N_BATCHES` (25), not only between phases.
4. RSS ceiling (`ROBERTA_MAX_RSS_MB` 6144): abort cleanly, with a checkpoint,
   before the OOM killer does. RSS is logged at phase boundaries and
   checkpoints.
5. Resume: `analyze` restores prior scores by post id plus any RoBERTa
   checkpoint, so recovery is rerunning it.
6. LLM scoring skips scored posts and saves every `LLM_SAVE_EVERY_N` (100).
7. `del pipe; gc.collect()` in `try/finally`.

Baseline: RoBERTa peaks at ~1.3 GB RSS and takes ~1-1.7 s per 16-post batch
(~9 min for 5.6k posts, ~33 min for 26k replies). A big drift means a leak or
a removed thread cap. Treat any bulk transfer the same way: one steady,
rate-capped, resumable process.

### GPU Jobs (`k8s/`)

- One-shot Jobs only, on the quant tenant pattern (homelab-infra
  `docs/k8s-workloads.md`): image built in-cluster from `Dockerfile`, pinned by
  git sha with `deploy.sh <sha>`, uid 1000, Secret from `.env`, pods pinned to
  dkbl2 by `nodeSelector`, local-path PVC there. A new experiment is a Job dir
  under `k8s/jobs/`, a CLI subcommand and a row in `k8s/README.md`.
- A failed Job pages Slack (`KubeJobFailed`); read the pod's `kubectl logs`
  before proposing a fix.

### Backup

`backup` syncs `data/raw` and `data/processed` (Standard) and `data/models`
(Glacier IR) to the bucket in homelab-infra
`terraform/iran-sentiment-backups.tf`: no `--delete`, writer without
`DeleteObject`, versioned bucket. It needs `IRAN_BACKUP_S3_BUCKET` and the AWS
keys in `.env`. Run it after every paid `collect`, `relabel` or `topic-label`.
The host's restic drive is often unplugged, so S3 is the copy to count on.

### Blog charts (dkweb)

Rerun `export-web` after any rescoring, then `npm run build` in dkweb. Stance
colours are dkweb's `--viz-anti` / `--viz-pro` / `--viz-neutral` tokens in
`global.css`.

## Docs map

- [README.md](README.md): findings, methodology, stance-model experiments, setup, cost.
- [docs/STATUS.md](docs/STATUS.md): data coverage, open work, what a refresh needs.
- [docs/decisions.md](docs/decisions.md): dated decisions, why, and what was rejected.
- [docs/truthsocial-api.md](docs/truthsocial-api.md): endpoints, auth flow, rate limits.
- [k8s/README.md](k8s/README.md): GPU Job runbook and the scorer decision rule.
- homelab-infra `docs/secrets.md` (sops), `docs/k8s-workloads.md` (tenant
  conventions), `terraform/iran-sentiment-backups.tf` (backup bucket).
