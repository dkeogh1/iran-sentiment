# Status

As of 2026-10-06. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 17,837 posts from 17 accounts plus 1,768 keyword-search
  rows (19,605 in all), Feb 1 -> Sep 18, 2026, though nine accounts start
  later (below). The May 10 -> Sep 18 refresh read 5,631 tweets for ~$28,
  with the five heaviest accounts (Loomer, Levin, Jones, Ravid, StateDept)
  sampled at 450. 490 posts have no text (nearly all bare links) and leave
  every series (`src/text_rules.py`). `score_opus` on all 19,115 posts with
  text (2026-10-05; the 3 posts whose answers the backfill relabel could
  not parse were resubmitted and merged). Haiku topic labels on 21,574 of
  the 21,665 posts with text (X and Trump's feed); the other 91 are link
  shares (*Open*).
- **Retweets** count as the account's messaging (`docs/decisions.md`):
  5,683 of the 17,837 account posts (32%; 70% of admin; @POTUS 789 of 789,
  783 of them @WhiteHouse's), stored cut at about 140 characters.
- **Posts over 280 characters** are whole. Requests ask for `note_tweet`
  since 2026-10-04, and `x-backfill-text` re-read the 2,981 cached
  originals that looked cut at 280 (2026-10-04): 2,980 came back, 2,571 of
  them (86%) longer than the cached text, now stored whole (median 461
  characters); 1 was not returned. What was made from the cut text is in
  `*_superseded.parquet` (2,571 `sentiment_all` rows, 2,570 Opus labels,
  2,571 topic labels), and all 2,571 were rescored and relabelled from the
  full text. They are 929 of the 4,225 war posts in the phases (22%, topic
  `either`). Effect: `docs/decisions.md`.
- **Late starts**: only 8 accounts begin on Feb 1-2. @VaticanNews starts
  Feb 21, then @POTUS Feb 27, @Pontifex Mar 3, @StateDept Mar 4, @VP Mar 17,
  @BarakRavid Mar 30, @RealAlexJones Apr 1, @LauraLoomer Apr 6 and
  @WhiteHouse Apr 9; their raw caches start there too (cause unverified).
  The strike phase, the base of every phase contrast, lacks them before
  those dates, and the late-March jump in the blog's weekly anti-war MAGA
  line is @RealAlexJones entering the data.
- **Holes from the ~3,200-tweet timeline horizon** (unrecoverable): from
  May 10 up to Jun 20 for @marklevinshow, Jul 4 for @LauraLoomer, Jul 17 for
  @WhiteHouse and Jul 31 for @RealAlexJones. The other 13 accounts have no
  horizon hole, but the Sep 18 run returned no posts for whole 14-day
  slices of @POTUS (May 4-18), @VP (May 22-Jun 5), @RealCandaceO
  (Jul 18-Aug 1) and @TuckerCarlson (Aug 1-15), and, after her horizon
  hole, for two slices of @LauraLoomer (Jul 19-Aug 16): silence or a
  failed fetch, unverified.
- **Capped slices**: a capped 14-day slice keeps its newest posts, a median
  of 0.7-2.9 days of the 14 for the five sampled accounts and @WhiteHouse.
  Capped on Sep 18, besides the five: @POTUS, @PeteHegseth, @VP and
  @VaticanNews in 9 slices, @RealCandaceO 7, @WhiteHouse 6, @mtgreenee and
  @Pontifex 1.
- **Keyword searches**: frozen at May 12, with a gap from about Apr 20 to
  May 5 (7-day recent search only).
- **Truth Social**: Trump's feed current through 2026-09-30 (anonymous
  refresh), 4,360 posts from Apr 4 (contiguous), 2,550 of them with text,
  scored with `score_opus_distilled`. Checked against Opus on the feed
  (`teacher-check --source trump`, 400 random posts with text; 2026-10-05,
  refilled 2026-10-06): 0.74 Pearson and 1.8% sign flips on all 400; on
  the 63 war posts among them 0.62 and 11%, the model's mean 0.03 below
  Opus's (0.379 vs 0.410). The 1,810 without text leave every share and
  mean: 1,589 images or videos with no caption and 221 quotes of his own
  posts that have no text (*Trump's posts with no text*, below).
  `truthsocial_trump_stance.parquet` still holds the old constant +0.111
  for 1,622 of them; the other 188 have no score.
  98,668 replies to 8 tracked posts, 98,663 unique (`data/raw/truthsocial/replies_*.jsonl`;
  `these_fools` 4,441 and `trump_strait` 11,173 added 2026-09-30, 94.7% and
  99.1% coverage), 92,306 of them with text (the `reply-population`
  estimand), scored with RoBERTa, `score_opus_distilled` and
  `score_mixed_distilled`, plus a 1,292-row Haiku stance sample (1 error to
  retry) and Opus labels on the 1,157 sampled replies with text. The
  published labels are v2 (2026-10-05, `reply-teacher-check --v2`): made
  from the whole reply and the Trump post it answers
  (`docs/decisions.md`). `reply-population` weights the 1,071 random draws
  among them (`settings.REPLY_TEACHER_LABELS_VERSION`) and writes
  `reply_population_score_opus_distilled_v2.csv`; the 1,200 draws are
  frozen in `data/processed/reply_sample_draws.parquet`. The v1 labels
  (1,225; 423 made from text cut to 100 characters, none with the parent
  post) and their last run, the unsuffixed CSV, are kept as history.
  `score_opus_distilled` was rescored 2026-09-30 at its 256-token
  training length; `score_opus_distilled_v0` keeps the old reply scores
  (`docs/decisions.md`).
- **Trump's posts with no text** (`ts-fill-text`, 2026-10-06, free,
  anonymous): these notes had called the 1,622 textless posts ReTruths
  whose text the collector dropped, plus images and videos. None was a
  ReTruth: the cache already held 185 ReTruths with their text. Read again
  by id, the 1,622 and the 198 stored as only Truth Social's quote
  fallback (`RT: <link>`, which had counted as text and been scored on the
  link) are 1,589 media-only posts and 231 quote posts (1,820 reads, no
  errors, none gone). All 231 quote an earlier post of his own, and only
  10 of those have text: those 10 now carry it, as
  `RT @realDonaldTrump: ...` with `quote_of`, so their words count twice,
  and the other 188 fallbacks have no text. Nothing is left to recover.
  What was made from the old text is in `*_superseded.parquet` (198 feed
  rows, 87 topic labels, 20 Trump-check labels); the 10 were rescored and
  topic-labelled, and the check was refilled to 400 (2 of its 20 new
  labels are those quote posts, relabelled from their new text). New pulls store quote
  posts the same way (`docs/decisions.md`). His war posts and their means
  don't move (96/111/122/70 by phase, topic `either`); with 188 fewer
  posts with text, his war share rises 0.8-2.1 points a phase
  (33.1/13.8/12.2/12.9% -> 35.2/15.2/13.0/14.0%). X tables and reply
  estimates are unchanged.
- **Timeline**: 84 events through 2026-09-15 in `config/timeline.py`;
  `ANALYSIS_END` = 2026-09-18. Sources for the Sep 2026 additions are in
  `docs/timeline_candidates_2026-05-12_to_2026-09-15.json`.
- **Spend, 2026-10-04/05** (approved runs): X ≈ $14.90 for the backfill
  reads (2,980 returned at $0.005). Anthropic, measured from the batch
  results' usage: `relabel` $5.04 (2,575 posts; the 3-post resubmit under a
  cent), `topic-label` $0.62 (2,890), Trump-feed check $0.70 (400), reply
  labels v2 $2.10 by batch (1,141) plus $0.07 for the 16-call direct
  pilot. `analyze --llm` (Haiku, 2,571 posts) is not metered: about $2 at
  Haiku's prices, from the relabel's token counts on the same prompt.
  About $25.40 in all.
- **Spend, 2026-10-06**: Anthropic by batch, from the results' usage:
  `topic-label` $0.023 (126 posts, 35 labelled), the Trump-feed check's
  refill $0.035 (20 posts); about $0.06. No X reads; `ts-fill-text` is
  free.

## Where things are

- The k8s namespace `iran-sentiment` is deleted again (2026-09-30, after the
  256-token scoring Jobs; its PVC with it). Models are in `data/models/` on
  dkbl1 and in the S3 backup.
- S3 backup first synced 2026-09-25: raw 72 MB and processed 50 MB in
  Standard, models 8.8 GB in Glacier IR, verified object for object. Last
  synced 2026-09-30 after the reply refresh and the GPU pull, as far as
  these notes record: run `backup` for the 2026-10-04/05 runs (the
  rewritten X raw cache, the relabels, the `*_superseded` archives, the v2
  and Trump-feed labels, the downloaded batch results),
  `data/processed/reply_sample_draws.parquet`, and the 2026-10-06 ones
  (the rewritten Trump raw cache, `ts_fill_text_journal.jsonl`, the new
  `*_superseded` rows, the refilled check and the new topic labels).
- Host backup (restic, homelab-infra): the SanDisk drive is unplugged and no
  host backup has run since 2026-04-23. The host backup script no longer
  needs Timeshift, so the drive can go back in; until then S3 is the only
  off-box copy of `data/`.
- Batch state files in `data/processed/`: `relabel_batch.json` (`relabel
  status|collect|merge|resubmit`), `topic_batch.json`,
  `teacher_batch_reply_v2.json` and `teacher_batch_trump.json` (`--batch
  status|collect`). The downloaded results sit beside them as
  `*_results_<date>.jsonl`. The Batch API keeps results for 29 days.
- `main` is pushed.

## Open

- The dkweb post `iran-war-stance` (7 Observable Plot charts, the stance
  colour tokens) merged to dkweb `main` on 2026-10-04 and deploys with it,
  behind Cloudflare Access until launch. Branch builds did run; the Sep 30
  "no preview" note predated the build finishing. The reply-scatter labels
  for the two new posts still need a visual check on the deployed page or
  `npm run preview` on 127.0.0.1 over an SSH tunnel. `export-web` reran
  2026-10-05 after the backfill and the v2 reply labels: `accounts`,
  `decomposition`, `maga_weekly` and `replies` JSONs changed and the
  post's prose was updated to them (the June deal and hold-off readings
  reverse; `docs/decisions.md`); pushed to dkweb `main`. The audience
  bars' x domain went to [-0.9, 0.9] and the tone-stance scatter's y domain
  to [-0.2, 0.5] so the v2 hold-off bar (ends at 0.83) and intervals (up
  to +0.46) fit; worth a look on the page. `export-web` reran 2026-10-06
  after `ts-fill-text`: only `trump_phases.json` changed (Trump's posts
  with text and war shares; the war-post counts and means did not), and
  the post's Trump prose and caption follow it.
- **Next X pull: Oct 25-30, 2026, no later than Nov 1.** On Sep 18
  @RealAlexJones's ~3,200-tweet timeline reached back only ~7 weeks, so his
  gap starts opening around Nov 1-6 (@WhiteHouse ~Nov 20, @LauraLoomer early
  Dec). One pull then covers ~6 weeks for about what a pull now would cost,
  because the heavy accounts hit their caps either way. Requests now ask for
  `note_tweet`, so long posts arrive whole. Before it: reconsider the
  per-slice cap split. It gives each 14-day slice an equal share and keeps
  that slice's newest posts, so for the heavy accounts it keeps only about
  the newest 1-3 days of each fortnight (*Capped slices*, above), and unused
  cap from quiet slices never reaches busy ones; holding the May-Sep density
  repeats that. Then size `ACCOUNT_CAP_OVERRIDES` for the gap so the
  accounts that hit slice caps (the five heavy ones plus @POTUS,
  @PeteHegseth, @VP, @VaticanNews, @RealCandaceO and @WhiteHouse) keep the
  May-Sep density of ~45-50 per two-week slice, `collect --estimate`, ask
  the user for the X credit balance, run with `--no-search`. Rough cost at
  that density: ~2,100 reads (~$10) plus ~$4 for the Opus relabel and topic
  labels; at today's 450/500 caps it would be ~$20-23 plus ~$7. After it:
  add timeline events from Sep 16 on, move `ANALYSIS_END` and extend
  `PHASES` (which end Sep 18, so Trump's feed after that is collected but
  outside every phase).
- **The Anthropic SDK's batch results stream breaks on this host** (httpx
  `ReadError`, "Bad file descriptor"): all four collects on 2026-10-04/05
  (`topic-label`, `relabel`, both teacher checks) failed on it. Workaround:
  read the batch (`GET /v1/messages/batches/<id>`, API key header),
  download its `results_url` to a file, and pass that to `collect
  --results-file` (`relabel`, `topic-label`, `--batch collect`). Worth
  making `collect` fall back to the download itself.
- The X teacher check needs no new labels after the backfill (0 of its 497
  to do), but 63 of its posts were backfilled: their direct Opus labels saw
  the cut text, while `sentiment_all` now holds full-text Haiku scores, so
  rerunning `teacher-check` (x) would compare different inputs on those 63.
  The 2026-09-18 report is unaffected, and `teacher-retest` pairs those
  labels with the archived batch labels made from the same cut text.
- `analyze` restores a prior score only onto the same text (2026-10-04),
  and archives a prior row with a paid score (Haiku's) before rescoring it,
  but the paid labels are keyed by id. `x-backfill-text`, X `collect
  --force` (which merges into the cache by id since 2026-10-06) and
  `ts-fill-text` move the labels of the posts whose text they change; a
  forced `collect-truth` merges by id but doesn't, so a Trump post whose
  text it changes keeps its topic label and feed score (`score-posts`
  rescores only rows that gain text).
- The mixed-domain distill is still on the v1 reply labels: its training
  rows (`stance_local.reply_label_frame`) and the mixed-vs-posts-only
  comparison (the mixed model ahead on the two unseen posts, 0.658 vs
  0.604 Pearson against v1, gap CI [-0.01, +0.12]). Against v2 on the same
  two posts the order flips (0.553 mixed vs 0.611 posts-only, 267
  replies; gap CI [-0.15, +0.04], replies resampled). Retrain on v2 rows
  only if the mixed model matters again.
- 91 posts with text are link shares Haiku will not judge (43 Trump's,
  48 on X, 39 of them Levin's: a link under a word or two such as "Amen"
  or "Right on"; it answers that it cannot open URLs and hits
  `max_tokens`). `war_flag` falls back to the keyword pattern for them by
  design. `topic-label` resubmits them on every run (a few cents). Worth
  recording them as unjudgeable.
- 5 duplicate reply ids in the April reply files (3 `civilisation_dies`,
  2 `power_plant_day`); `reply-population` and `export-web` drop the
  copies.
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).
