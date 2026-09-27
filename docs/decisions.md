# Decisions

Newest first: what was decided, why, and what was rejected. The numbers
behind the stance-model entries are in README *Stance-model experiments* and
[k8s/README.md](../k8s/README.md).

## 2026-09-25: Off-box backup to S3, append-only

- **Decision:** `backup` syncs `data/` to a versioned bucket (homelab-infra
  `terraform/iran-sentiment-backups.tf`) with no `--delete` and a writer that
  cannot delete; models go to Glacier IR. Run by hand after every paid run.
- **Why:** the raw pulls and paid labels existed only on dkbl1, and the host's
  restic drive had been unplugged since April. Same pattern as quant.
- **Rejected:** relying on the host restic backup alone.

## 2026-09-23: Report war-post stance, not all-post means

- **Decision:** findings use war posts only (keyword or Haiku topic label,
  `TOPIC_SOURCE = "either"`), with account-day block-bootstrap intervals and a
  shift-share split, and every finding is checked under keyword, label and
  either.
- **Why:** the stance prompt makes Opus give off-topic posts ("Deport them!",
  "Amen") a stance from the author, so all-post means mostly track how much a
  camp talked about the war. On war posts the MAGA split is ~1.1 points, not
  the 0.48 the all-post means showed.
- **Rejected:** all-post means as the headline; the keyword pattern alone
  (it misses about half the war posts).

## 2026-09-20: Data moves to the cluster by rsync, never `kubectl cp`

- **Decision:** `scripts/k8s/sync-data.sh`: one rate-capped rsync over SSH
  straight into the PVC's directory on dkbl2, checksum-verified afterwards.
- **Why:** on 2026-09-19 dkbl1 hard-powered off during a `kubectl cp` pull (a
  1.7 GB stream through the API server, then a burst of one-process-per-file
  copies), leaving 0-byte and truncated model files. Bursty load is the known
  trigger (homelab-infra `docs/dkbl1-cstate.md`).
- **Rejected:** `kubectl cp` for anything bulk.

## 2026-09-19: Opus-taught DeBERTa scores the replies; the mixed-domain retrain does not

- **Decision:** `score_opus_distilled` (DeBERTa-v3-large, the 256-token sweep
  winner, CV 0.869 vs Opus) is the reply scorer of record, and reply
  population shares are corrected against the Opus-labelled sample
  (`reply-population`).
- **Why:** on 959 Opus-labelled replies the model reaches only 0.71 Pearson and
  64% sign agreement and leans ~8 points pro-war. Retraining with those
  replies mixed in (`score_mixed_distilled`) was no better on replies and
  slightly worse on posts.
- **Rejected:** `score_mixed_distilled` (kept as a comparison column);
  uncorrected distilled shares.

## 2026-09-18: Claude Opus 5 is the stance of record

- **Decision:** relabel every post with Opus 5 at low effort through the Batch
  API (`relabel`, ~$31 for 19.5k posts); `score_opus` is
  `settings.STANCE_SCORE_COL`.
- **Why:** a 497-post teacher check showed Haiku failing on `maga_prowar`
  (0.34 Pearson vs Opus, 24% sign flips): it reads vicious hawkish posts as
  anti-war and praise of peace as pro-war. A model distilled from Haiku
  labels topped out at 0.79 Pearson because the labels were the ceiling;
  Opus-taught, the same recipe reaches 0.88.
- **Rejected:** Haiku as teacher; open local LLMs with the same prompt
  (Qwen2.5-7B / Qwen3-8B, 4-bit: 0.47-0.59 Pearson, and thinking mode took
  2.6 h for 400 posts with 9% unparseable); the direct API (Batch is half
  price).

## 2026-09-18: GPU experiments are one-shot kustomize Jobs on dkbl2

- **Decision:** each experiment is a Job under `k8s/jobs/`, image built
  in-cluster and pinned by git sha, data on a local-path PVC on dkbl2. The
  pipeline itself stays on the host venv; nothing is scheduled, and the
  namespace is deleted when experiments end (done 2026-09-21).
- **Why:** quant tenant conventions (homelab-infra `docs/k8s-workloads.md`);
  the GPU is on dkbl2. `collect` costs money per run, so it stays manual.
- **Rejected:** a long-lived "workspace pod" with code and data copied in by
  kubectl; a CronJob for `collect`.

## 2026-09-16: Truth Social replies through our own v2 walker and a saved token

- **Decision:** `collect-replies` walks `/api/v2/statuses/:id/context/descendants`
  itself; `ts-login` handles the new-device security-code check once and
  saves `TRUTHSOCIAL_TOKEN`.
- **Why:** the v1 descendants endpoint truthbrush uses is dead, and
  `/context` is blocked by Cloudflare even with auth. See
  [truthsocial-api.md](truthsocial-api.md).
- **Rejected:** truthbrush's reply pagination.

## 2026-09-16: Gap-fill slicing and a per-run budget gate for X

- **Decision:** long gaps are walked oldest-slice-first in 14-day windows with
  the account cap shared across slices; `collect --estimate` prints the
  maximum spend from the cache; a real run refuses to start above
  `X_RUN_BUDGET_USD`; prolific accounts are sampled with
  `ACCOUNT_CAP_OVERRIDES`.
- **Why:** the timeline endpoint is newest-first, so one capped fetch over a
  multi-month gap keeps the newest N and silently drops the rest.
- **Rejected:** one capped fetch per account per run.

## 2026-04-20: LLM stance for the religious tier; CPU guardrails for RoBERTa

- **Decision:** `religious_authority` is always scored with an LLM stance
  model. RoBERTa runs batched (16), thread-capped (2), checkpointed inside the
  loop, under an RSS ceiling, with explicit model cleanup.
- **Why:** on 797 religious posts RoBERTa scored @Pontifex +0.36 where the LLM
  gave -0.50 (tier +0.26 vs -0.24): faith-based peace rhetoric is lexically
  positive. Secular anti-war voices were fine (@SenSanders differed by 0.06),
  so the fix is category-specific. The host powered off twice during
  `analyze`: first OOM on unbatched RoBERTa, then a thermal or power trip
  with all cores saturated.
- **Rejected:** trusting valence models for stance on faith-based voices;
  unbatched inference, then batch size 32.

## 2026-04: Hard caps, caching and estimates on paid APIs

- **Decision:** per-resource caps in `config/settings.py`, per-account JSONL
  caches that are skipped by default (`--force` to re-fetch), and a printed
  maximum spend before any paid run.
- **Why:** a single prolific account (@marklevinshow, 2,544 tweets, ~$13) plus
  a few others spent a $25 X top-up in one run before the rest were reached.
- **Rejected:** uncapped fetches; relying on X's 24-hour read dedup for
  budget planning.

## 2026-04: One CLI, config-driven, no one-off scripts

- **Decision:** every capability is a subcommand of `src/cli.py`, which stays
  thin over importable modules; tunables and lists live in `config/`.
- **Why:** early iterations left eight throwaway scripts in the repo root
  (`collect_missing.py`, `merge_and_rerun.py`, ...), and the user asked for
  reusable, config-driven code.
- **Rejected:** standalone scripts per task.
