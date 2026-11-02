# Status

As of 2026-11-02. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 19,786 posts from 17 accounts plus 1,768 keyword-search
  rows, Feb 1 -> Nov 1, 2026, though nine accounts start later (below). The
  Sep 18 -> Nov 1 pull (2026-11-01, the 35 h window sampler, `docs/decisions.md`)
  read 1,949 posts plus 17 user lookups (~$9.92): 11 accounts sampled at 5
  per window, 6 take-all; per-window rows in
  `data/raw/x_sampling/windows_2026-11-01.jsonl`. It lost @marklevinshow's
  first 11 h and @VP's first 2 days to the timeline horizon (VP replies a
  lot, and replies count toward the 3,200). Its 1,903 new posts with text
  have RoBERTa scores and `score_opus` (2026-11-02; 2 unparseable answers
  resubmitted and merged), and Haiku topic labels with Trump's 672 new posts
  (2,570 of 2,666 labelled; the 96 failures are link shares, *Open*). Before it: the May 10 -> Sep 18 refresh read 5,631 tweets for ~$28,
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
  @WhiteHouse Apr 9; their raw caches start there too. X's counts
  (`x-counts`, 2026-11-02) show the posts exist (e.g. @WhiteHouse 877 in
  February, @LauraLoomer 1,544, @RealAlexJones 1,187), and full-archive
  search works on this app, so they can be filled (*Open*).
  The strike phase, the base of every phase contrast, lacks them before
  those dates, and the late-March jump in the blog's weekly anti-war MAGA
  line is @RealAlexJones entering the data.
- **Holes from the ~3,200-tweet timeline horizon** (fillable from the
  archive since the 2026-11-02 probe; `x-counts --report` has X's count per
  month beside what is held): from
  May 10 up to Jun 20 for @marklevinshow, Jul 4 for @LauraLoomer, Jul 17 for
  @WhiteHouse and Jul 31 for @RealAlexJones. The other 13 accounts have no
  horizon hole, but the Sep 18 run returned no posts for whole 14-day
  slices of @POTUS (May 4-18), @VP (May 22-Jun 5), @RealCandaceO
  (Jul 18-Aug 1) and @TuckerCarlson (Aug 1-15), and, after her horizon
  hole, for two slices of @LauraLoomer (Jul 19-Aug 16). X's counts say
  most were not silence (e.g. @POTUS 171 posts in May against 56 held).
- **@WhiteHouse: the timeline endpoint fails for it.** X counts 756-1,102
  posts a month (25-35 a day), but the Nov 1 pull got posts in only 5 of
  its 31 windows (25 posts); the cache holds 0-37% of X's count in every
  month. The Nov 1 pull also got 0 where X counts posts in 15 other windows
  (@POTUS 4, @VP 3, @BarakRavid 3, @marklevinshow 2, @StateDept 2,
  @RealCandaceO 1), and 4 instead of 5 in a few more (X returns short
  pages). Full-archive search returned every cached post on the probe days,
  so it is the route for @WhiteHouse and the empty windows.
- **Full-archive search works** (`x-archive probe`, 2026-11-02): for two
  closed @PeteHegseth days it returned every cached post, retweets
  included, plus 1 not cached; counts/all includes retweets and matches the
  cache (270 vs 268 posts, Feb 1 - Mar 2).
- **Capped slices**: a capped 14-day slice keeps its newest posts, a median
  of 0.7-2.9 days of the 14 for the five sampled accounts and @WhiteHouse.
  Capped on Sep 18, besides the five: @POTUS, @PeteHegseth, @VP and
  @VaticanNews in 9 slices, @RealCandaceO 7, @WhiteHouse 6, @mtgreenee and
  @Pontifex 1.
- **Keyword searches**: frozen at May 12, with a gap from about Apr 20 to
  May 5 (7-day recent search only).
- **Truth Social**: Trump's feed current through 2026-10-31 (anonymous
  refresh, 1,016 new posts), 5,376 posts from Apr 4 (contiguous), 3,222 of
  them with text, scored with `score_opus_distilled` (the 672 new posts with
  text on dkbl1's CPU, 2026-11-02, thread-capped). Checked against Opus on the feed
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
- **Timeline**: 138 events through 2026-10-31 in `config/timeline.py`;
  `ANALYSIS_END` = 2026-09-18 and `PHASES` end there until the phase-5
  decision (*Open*). Sources: `docs/timeline_candidates_2026-05-12_to_2026-09-15.json`
  and `docs/timeline_candidates_2026-09-16_to_2026-10-31.json`.
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
- **Spend, 2026-11-01/02** (approved runs): X ≈ $11.90: the pull $9.92
  (1,949 posts + 17 lookups), the archive probe $0.10 (16 posts + 2 counts
  pages), `x-counts` $1.89 (189 counts/all pages, 17 accounts + 4 terms,
  Feb 1 -> Nov 2, hourly). Anthropic, metered from the batch results'
  usage: `relabel` $2.90 (1,903 posts; the 2-post resubmit under a cent),
  `topic-label` $0.49 (2,666), about $3.39, drawn from the plan's API credit. Anthropic API spend
  from 2026-11 is covered by a monthly promotional credit on the user's
  Claude plan (expires each cycle, no rollover).

## Where things are

- The k8s namespace `iran-sentiment` is deleted again (2026-10-07, after
  the reply-scorer experiment on image 13717d7c504a; its PVC with it).
  Models, including
  `data/models/reply_ctx_lopo/`, are in `data/models/` on dkbl1 and in the
  S3 backup.
- S3 backup first synced 2026-09-25 (models 8.8 GB in Glacier IR, the
  rest Standard). Last synced 2026-11-02 after the X pull, the probe, the
  counts, the Trump scoring and the merged labels.
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
- **Next (2026-11-02):** labels for the Nov 1 pull are merged and backed
  up. Size an archive fill (`search/all`, same 35 h window grid) from
  `x-counts --report`: @WhiteHouse Feb 1 -> Nov 1, the horizon holes, the
  late starts and the empty windows; it needs an `x-archive fill` command.
  Decide phase 5 ("midterm hold", Sep 19 -> Nov 1) before `phases`,
  `visualize`, `export-web`. The Claude credit jobs (Trump's feed on Opus,
  reply prompt and sample checks, a stance-and-topic prompt, Opus 5.5 as a
  second rater) are being built; each is a paid run to estimate and ask.
  The timeline route stays the forward pull; the horizon matters less now
  that the archive works.
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
- Reply scorers against the v2 labels, settled 2026-10-07: a scorer that
  reads each reply with its Trump post beats the scorer of record
  leave-one-post-out (pooled 0.64 vs 0.58; June deal 0.62 vs 0.14) but
  would narrow the published intervals only modestly, so it stays an
  experiment (`docs/decisions.md`). The old mixed model
  (`score_mixed_distilled`, v1 rows) is history.
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
