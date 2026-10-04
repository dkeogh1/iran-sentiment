# Decisions

Newest first: what was decided, why, and what was rejected. The numbers
behind the stance-model entries are in README *Stance-model experiments* and
[k8s/README.md](../k8s/README.md).

## 2026-10-04: Retweets and ReTruths count as the account's messaging

- **Decision:** an X retweet or a Truth Social ReTruth counts as a post by
  the account that shared it, scored and counted like its own posts. The
  Truth Social collector now stores a ReTruth's text as `RT @acct: ...` with
  `reblog_of`; before, it kept only the ReTruth's own content, which is empty.
  Stored retweet text is cut at about 140 characters (no expansions, below),
  and 349 retweet/original pairs are both in the data (@POTUS of
  @WhiteHouse, @SecRubio and @StateDept), so that content counts twice
  inside a tier.
- **Why:** what an account chooses to amplify is part of its messaging, and
  for the administration it is most of it: 5,683 of the 17,837 X account
  posts are retweets (32%; 70% of admin), as are 39 of admin's 46 September
  war posts.
- **Rejected:** dropping retweets, at collection (`exclude=retweets`) or in
  the analysis: that measures authorship, not messaging, and would empty
  @POTUS.

## 2026-10-04: @POTUS stays in the admin tier

- **Decision:** @POTUS stays an admin account.
- **Why:** it is the presidency's official account, and under the retweet
  decision what it amplifies is administration messaging. It is not Trump's
  own voice: all 789 cached posts are retweets, 783 of them of @WhiteHouse.
  His own words are the Truth Social feed, reported on its own.
- **Rejected:** dropping it as a copy of @WhiteHouse; calling it Trump's own
  X account.

## 2026-10-04: Posts and replies with no text leave every denominator

- **Decision:** a post or reply with fewer than 3 characters once links and
  a leading `RT @x: ` are removed (`src/text_rules.py`,
  `settings.MIN_TEXT_CHARS`) has no text. It is never scored and leaves every
  share and mean: `inference.prepare` drops it, `relabel` skips it, and the
  `reply-population` estimand is now the replies with text. Same cut as the
  reply sampler's `media_only`.
- **Why:** an image, a video or a bare link gives the scorers nothing to
  judge, and counting it as an off-topic post moves the war share with how
  often an account posts pictures. Trump's 1,622 textless posts (of 4,360)
  put his war share at 26/8/8/8% by phase; without them it is 33/14/12/13%.
  The old reply method put about 70% of image-only replies on the pro-war
  side. On X, 490 posts leave, nearly all bare links; X war-post stance did
  not change. Replies: 92,306 of the 98,663 unique replies (98,668 rows)
  have text, and the audience across all eight
  posts moved from 46.3% pro-war / 35.0% anti-war to 47.2% / 37.3%.
- **Rejected:** counting them as off-topic (the old behaviour); a placeholder
  score (next entry); imputing them from the reply's valence bucket.

## 2026-10-04: The distilled scorer no longer scores empty text

- **Decision:** `score_with_distilled` returns NaN for rows without text and
  never sends them to the model. The `"."` stand-in for empty text is gone.
- **Why:** the stand-in scored a constant +0.111, inside the pro-war band
  (above +0.05), for all 1,622 textless Trump posts and the image-only
  replies. Rows scored before keep that value in the parquets; `phases`,
  `reply-population` and `export-web` drop them by the no-text rule.
- **Rejected:** any constant score for no text.

## 2026-10-04: X requests ask for `note_tweet` and `referenced_tweets`, no expansions

- **Decision:** timeline and search requests add `note_tweet` (the full text
  of a post over 280 characters) and `referenced_tweets` (stored as `ref`:
  retweeted, quoted, replied_to) to `tweet_fields`, and ask for no
  expansions.
- **Why:** `text` stops at 280 characters, so about 2,000 cached posts were
  stored and labelled cut there (about 14% of war posts; 32% of the
  religious tier's, 25% of pro-war MAGA's). New pulls get the whole post;
  the cached ones are not backfilled ([STATUS.md](STATUS.md)). X bills per
  resource returned, and how it bills posts returned as expansions is not
  documented, so `expansions=referenced_tweets.id` could add billed reads.
  Without it a retweet's text stays cut at about 140 characters. Measured
  on the 349 retweet/original pairs in the data, that costs little: the
  mean retweet-minus-original Opus score is -0.016 (mean absolute
  difference 0.075).
- **Rejected:** expansions for the full retweet text until their billing is
  known; `exclude=retweets` (first entry).

## 2026-10-04: The reply sample's random draws are frozen

- **Decision:** the random bucket draws behind the Opus-labelled reply
  sample (reply id, post, valence bucket at draw time) live in
  `data/processed/reply_sample_draws.parquet`. `stance` records a new
  post's draws as it samples; `reply-population` reads them instead of
  redrawing. It also dedupes reply ids, and a post with replies in a bucket
  that has no labelled draw comes out NaN, with a `coverage` column; before,
  it was renormalised over the covered buckets, and a post with no labels
  at all got a zero-width interval.
- **Why:** `reply-population` re-ran the draw on the current reply frame
  every time. It still matched the labelled draws, but a reply collection
  that changed a post's frame could have moved the draws off the paid
  labels. A duplicated reply id had inflated the join (1,137 rows for 1,136
  labelled draws).
- **Rejected:** redrawing on every run.

## 2026-09-30: Rescore every reply with the model of record, at its training length

- **Decision:** `score_opus_distilled` for all 98,668 replies and Trump's feed
  is the current `stance_distilled_final_score_opus` (`deb-256-1e5-3`) at 256
  tokens. The old reply scores stay as `score_opus_distilled_v0`, out of every
  series. Scoring defaults to the model's training length from `recipe.txt`,
  and `distill --fit-all` and the sweep refuse to write into a final dir that
  holds another model.
- **Why:** the 83,054 older replies had been scored with the 128-token
  `distill-opus` fit, which the sweep's final fit then replaced in the same
  directory (2026-09-19 05:10 UTC). The column matched no model on disk (0.915
  correlation on untruncated replies, where current-model reruns reproduce
  exactly), and the 15,614 new replies would have mixed two models in one
  series. The two fits are equally good against Opus (0.72 vs 0.71 Pearson
  on 958 labels); corrected reply shares moved by up to 5 points, inside their
  intervals. Scoring at 128 tokens alone moves nothing that matters (Trump's
  phase means by <= 0.01).
- **Rejected:** keeping `v0` for the older replies (its model is gone, so new
  replies cannot join the series); retraining a 128-token fit to match it (GPU
  training is not bit-reproducible); switching to `score_mixed_distilled`
  (better on the two unseen posts, 0.66 vs 0.60 Pearson, but the gap's CI
  includes zero).

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
- **Why:** on 958 Opus-labelled replies the model reaches only 0.71 Pearson and
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
