# Decisions

Newest first: what was decided, why, and what was rejected. The numbers
behind the stance-model entries are in README *Stance-model experiments* and
[k8s/README.md](../k8s/README.md).

## 2026-10-06: Truth Social quote posts count as messaging; the textless Trump posts were never ReTruths

- **Decision:** a Truth Social quote post counts as the account's messaging,
  like a ReTruth. One with no words of its own carries the quoted post's
  text, stored as `RT @acct: <quoted text>` with `quote_of`; one with words
  of its own keeps them as sent (after Truth Social's `RT: <link>`
  fallback) and gains `quote_of`. A bare `RT: <link>` has no text
  (`src/text_rules.py`). The collector does this for new pulls;
  `ts-fill-text` re-read by id the cached posts with no words of their own
  (free, anonymous) and rewrote them the same way.
- **Why:** quoting a post without comment is the same amplification as a
  ReTruth (2026-10-04, retweets and ReTruths). Until now the collector
  ignored `quote`, so a quote with no words was stored empty or as the bare
  fallback, which counted as text and was scored on a URL.
- **Correction:** the notes took the 1,622 textless Trump posts for
  ReTruths whose text the collector dropped, plus images and videos, and
  the 2026-10-04 retweet entry says the collector had stored ReTruths
  empty. None was a ReTruth: the cache already held 185 ReTruths with
  their text. Read again by id (1,820 reads: the 1,622 and 198 stored as
  only the fallback), they are 1,589 media-only posts (image or video, no
  caption) and 231 quote posts, all quoting an earlier post of Trump's
  own; only 10 of those quoted posts have text. Those 10 now carry it, so
  those words count twice, as they would in a ReTruth of his own post; the
  other 188 fallbacks have no text. His feed has 2,550 posts with text of
  4,360 (was 2,738) and 1,810 without (1,589 media-only, 221 quotes of his
  own posts with no text). What was made from the old text is in
  `*_superseded.parquet` (198 feed rows, 87 topic labels, 20 Trump-check
  labels), and the check was refilled to 400: on its 63 war posts (was 59)
  0.62 Pearson, 11% sign flips, the model's mean 0.03 below Opus's.
- **Effect:** his war posts and their means don't move (96/111/122/70 by
  phase, means +0.423/+0.358/+0.302/+0.385, topic `either`). His posts with
  text per phase fall from 290/804/996/541 to 273/732/938/501, so his war
  share rises from 33.1/13.8/12.2/12.9% to 35.2/15.2/13.0/14.0% (Haiku label
  alone: 26.9/9.7/8.2/8.5% to 28.6/10.7/8.7/9.2%). X tables and reply
  estimates are unchanged.
- **Rejected:** appending the quoted text to a quote with words of its own
  (the scorers would judge the quoted author's words as the poster's).

## 2026-10-06: X `collect --force` merges into the cache by id

- **Decision:** a forced X run (`collect --force`, timelines and searches)
  merges what it fetched into the cache by id: a post it returns replaces
  its cached version, and the posts it doesn't return stay. Where a post's
  text changes, its Opus and topic labels move to `*_superseded.parquet`
  and its `sentiment_all` row is archived and cleared, as `x-backfill-text`
  does, so `analyze` rescores it and the label runs redo it.
- **Why:** a forced run is capped, and the timeline reaches back only
  ~3,200 posts (search ~7 days), so replacing the cache with it dropped
  paid posts that can't be read again. And the labels are keyed by id, so
  a re-read that changed a post's text kept labels made from the old text.
- **Rejected:** replacing the cache with the forced run (the old
  behaviour); keeping a post's labels across a text change.

## 2026-10-05: Reply labels v2, made with the parent post, replace v1

- **Decision:** `reply-population`, and with it the published audience
  shares and `export-web`, weights the v2 Opus reply labels
  (`settings.REPLY_TEACHER_LABELS_VERSION = "v2"`). v2 labelled the 1,157
  sampled replies with text from the whole reply and the Trump post it
  answers. It asks for the author's position on the war, not the tone,
  and leaves out the author's handle (`stance_local.REPLY_TEACHER_V2_PREFIX`,
  prompt `reply-v2-2026-10-04`). The v1 labels are kept unchanged, as
  history. Still on v1: the mixed-domain distill's training rows
  (`stance_local.reply_label_frame`) and the mixed-vs-posts-only
  comparison in the 2026-09-19 and 2026-09-30 entries. Against v2, on the
  same 267 replies to the two unseen September posts, the posts-only
  model leads instead (0.61 vs 0.55 Pearson; gap interval -0.15 to
  +0.04).
- **Why:** v1 sent the broadcaster prompt only the first 100 characters
  of each reply (423 of its 1,225 labels were cut) and never the post
  being answered, so a reply that only cheered took the sign of the
  cheering. A rough keyword check (replies with an assent word and no war,
  peace, deal or strike word; aggregates only, no reply read): under the
  ceasefire, hold-off and deal posts, v1 put 74% of 87 such replies
  pro-war and 10% anti-war, v2 9% and 59%. Under the September strikes
  post, where cheering backs the strikes, both put them pro-war (86% and
  93% of 28). On the same 1,157 replies v1 and v2 agree only loosely
  (0.55 Pearson, 17% sign flips), and about as loosely where v1 saw the
  whole reply (0.56, 734 replies) as where it saw 100 characters (0.54,
  423). The parent post and the stance question drive the change, not the
  cut.
- **What it changes** (Opus-corrected, replies with text, v1 -> v2): the
  June 14 deal post goes from the largest pro-war share (62.8% pro, 22.4%
  anti) to more anti than pro (30.8% / 42.3%; mean -0.041 [-0.140,
  +0.061]). The May 18 hold-off post's audience becomes the most hawkish
  (53.4% / 38.9% -> 70.4% / 5.2%; mean +0.068 -> +0.372 [+0.293,
  +0.460]). The April 7 "civilisation will die" post's is roughly split
  (39.8% pro, 41.8% anti; mean -0.074 [-0.151, +0.008]). All eight posts: 45.1% pro, 27.9% anti (was
  47.2% / 37.3%). The distilled scorer tracks v2 less well than v1 (0.58
  vs 0.70 Pearson on the same replies) and RoBERTa valence not at all
  (-0.04), so the correction against the labelled sample carries more.
- **Rejected:** keeping v1 (it reads applause for a ceasefire as support
  for the war); relabelling only the 423 cut replies (v1 and v2 disagree
  as much on the replies v1 saw whole).

## 2026-10-05: The Trump-feed scorer is checked against Opus on the feed

- **Decision:** the Trump chart's caption (README and the blog) cites a
  check on his feed instead of the 0.88 measured on held-out X posts: Opus
  labels on 400 random feed posts with text (`teacher-check --source
  trump`, the broadcaster prompt the distilled model learned) against
  `score_opus_distilled`. The published Trump war-post levels stand.
- **Why:** the feed had no Opus labels, and the caption had to say it was
  unchecked. The check: on all 400 posts 0.74 Pearson, 1.5% sign flips,
  the model's mean 0.009 above Opus's; on the 59 war posts 0.61 Pearson,
  10% flips, its mean 0.03 below Opus's (0.398 vs 0.430). The level holds;
  single posts agree less well than on X. Each phase has only 9-20 of
  those war posts, too few to check phase by phase.
- **Rejected:** citing the X holdout's 0.88 for the feed.
- **Refilled 2026-10-06:** 20 of the 400 labels had been made from a bare
  quote link. The refilled check: 1.8% sign flips and the model's mean
  0.007 above Opus's on all 400; on its 63 war posts 0.62 Pearson, 11%
  flips, 0.379 vs 0.410, 11-22 a phase (2026-10-06).

## 2026-10-05: Teacher checks run through the Batch API

- **Decision:** `reply-teacher-check --v2` and `teacher-check --source
  trump` send full runs as Message Batches (`--batch submit|status|collect`,
  half price), with the direct path's prompts, cached-prefix blocks and
  model settings; direct calls stay for pilots. Each check keeps its own
  state file, refuses a submit while a batch is open or uncollected, and
  never resends a cached id. Every batch `collect` (these, `relabel`,
  `topic-label`) also reads a downloaded results file (`--results-file`).
- **Why:** a 16-call direct pilot was billed $0.069, $0.0043 a call. The
  prompt cache engaged for one parent post in eight (the rest fell under
  the cache minimum), so the estimate that cached every parent post was
  too low. At the pilot's rate the 1,157-reply run would have cost about
  $5.07 direct, against $3.44 estimated with every parent post cached. By
  Batch the 1,141 replies left after the pilot cost $2.10, and the Trump
  check's 400 posts $0.70. The SDK's results stream failed on this host for all four
  collects on 2026-10-04/05 (httpx `ReadError`, "Bad file descriptor");
  each was collected from the batch's downloaded `results_url`.
- **Rejected:** the direct API for full runs (twice the price, and the
  prefix cache saved little).

## 2026-10-05: Long X posts backfilled; war-post stance barely moves

- **Decision:** the X tables and figures use the full text of the 2,571
  long posts `x-backfill-text` recovered, and everything made from the cut
  text is redone from it: Opus (`relabel`), Haiku topic labels
  (`topic-label`), and VADER, RoBERTa and Haiku stance (`analyze`,
  `analyze --llm`). All 2,571 were relabelled, not only the war posts
  among them.
- **What ran:** the 2,981 candidates were read by id (a 100-post pilot,
  then 2,881; ≈ $14.90). 2,980 came back: 2,571 longer than the cached
  text (86%, close to the ~90% the length density predicted) and 409
  whole already; 1 was not returned. Before the raw cache was rewritten,
  their `sentiment_all` rows (2,571, holding the only copy of the cut-text
  Haiku scores), Opus labels (2,570) and topic labels (2,571) went to
  `*_superseded.parquet`. The scorers then redid only those posts, plus
  the few they already retry.
- **Effect** (X war posts in the phases, point estimates, before vs after
  on the same posts): tier means move by at most 0.048 under topic
  `either` (anti-war MAGA in expiry/strikes, -0.417 -> -0.465; pro-war
  MAGA in the same phase +0.407 -> +0.444), 0.065 under the Haiku label
  alone and 0.051 under keywords. No tier's mean changes sign under
  `either`, and pro-war MAGA stays above the administration in every
  phase. War shares rise by up to 2.2 points, because the full text shows
  more war content: 4,225 war posts instead of 4,045 (`either`), 929 of
  them backfilled. Retweets stay cut at about 140 characters (no
  expansions, 2026-10-04).
- **Rejected:** relabelling only the war posts among them (the war flag
  itself changes with the full text, and the all-post means use every
  post).

## 2026-10-04: Long X posts are re-read by id; labels from the cut are archived

- **Decision:** `x-backfill-text` re-reads by id (`GET /2/tweets` with
  `note_tweet`, 100 ids a request) the cached originals likely stored cut at
  280 characters, and puts the full text in the raw cache. Candidates: the
  text's length with entities unescaped, a reply's leading mentions and the
  trailing links left out and other links at 23, is 270-280, or 266-269
  ending mid-sentence; posts created after the last collect without
  `note_tweet` (2026-09-18) never are. Every batch is journalled
  (`data/processed/x_backfill_journal.jsonl`) before anything else is
  touched, so no read is bought twice. A changed post's Opus and topic
  labels move to `*_superseded.parquet` (appended, never deleted; a later
  label for the same post is archived too), and so does its `sentiment_all`
  row, cut text and every score, before the row is cleared: it is the only
  copy of the Haiku `score_llm`. `analyze` restores a prior score only onto
  the same text. The refresh order then redoes exactly those posts.
  `training_frame` reads a cleared post back from the archive as it was
  scored, so the teacher check's seeded sample stays the posts it already
  paid for, each with the text its labels saw; a batch submitted before the
  backfill (`relabel status` + `collect`, an old `topic-label collect
  --results-file`) is not collected over the changed posts again.
- **Why:** in the 12,154 cached originals the length runs at ~27 posts per
  character from 240 to 265, then 2,911 sit at 270-280 and 1 above (the cut
  is in code points: X's own weighting, emoji and CJK as 2, puts ~70 over
  280). At the lower density about 300 of the 2,911 would be whole posts, so
  ~90% are cut; at 266-269 only the mid-sentence endings beat the density
  (70 against ~24). 2,981 candidates in all, ~$14.91 to read; about 23 cut
  posts that end a sentence at 266-269 are missed. The direct teacher check
  holds 96 of them, so `teacher-retest` pairs its labels with the archived
  batch labels made from the same cut text.
- **Rejected:** `collect --force` over the old windows (re-buys every post,
  capped, and then replaced the cache; it merges by id since 2026-10-06);
  keeping the old labels by id (a re-fetch
  must not keep scores made from other text); deleting the superseded labels
  or clearing the Haiku scores without a copy (paid data; clearing alone
  would also have moved the teacher-check sample onto ~400 unlabelled
  posts); pinning the teacher-check sample to its cached ids instead (the
  Haiku side of each pair would have been gone); a candidate rule on ending
  alone (cut posts end a sentence about one time in eight).

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
  posts are retweets (32%; 70% of admin), as are 39 of admin's 48 September
  war posts (46 before the long-post backfill, 2026-10-05).
- **Rejected:** dropping retweets, at collection (`exclude=retweets`) or in
  the analysis: that measures authorship, not messaging, and would empty
  @POTUS.
- **Corrected 2026-10-06:** the cached ReTruths already had their text;
  Trump's textless posts were images, videos and quote posts (2026-10-06).

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
- **Since 2026-10-06:** with his bare quote fallbacks, 1,810 of Trump's
  posts have no text (none a ReTruth), and his war share is 35/15/13/14%
  (2026-10-06).

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
  the cached ones were re-read separately (`x-backfill-text`, 2026-10-04
  above; [STATUS.md](STATUS.md)). X bills per
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
