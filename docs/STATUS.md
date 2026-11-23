# Status

As of 2026-11-23. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 25,477 posts from 17 accounts plus 1,768 keyword-search
  rows (27,245), Feb 1 -> Nov 22, 2026; every account starts Feb 1-4 (the
  archive fill, below). The series stop at `ANALYSIS_END`, Nov 21 (the
  pull's last whole UTC day). The Nov 1 -> Nov 22 pull (2026-11-22,
  `--no-search`, the 35 h window sampler, `docs/decisions.md`) read 875
  posts plus 17 user lookups (~$4.54) in 245 windows
  (`data/raw/x_sampling/windows_2026-11-22.jsonl`): 11 accounts sampled at
  5 per window, 6 take-all; the fill then added 72. Before it: the Sep 18 ->
  Nov 1 pull (2026-11-01) read 1,949 posts plus 17 lookups (~$9.92), and
  the May 10 -> Sep 18 refresh 5,631 tweets for ~$28, with the five
  heaviest accounts (Loomer, Levin, Jones, Ravid, StateDept) sampled at
  450. 724 account posts have no text (nearly all bare links) and leave
  every series (`src/text_rules.py`). `score_opus` on all 26,521 posts with
  text (24,753 account posts, 1,768 search rows); the 382 rows without it
  have no text. Haiku topic labels on 30,170 of the 30,237 posts with text
  (X accounts 24,722 of 24,753, keyword search 1,768 of 1,768, Trump 3,680
  of 3,716); the other 67 are link shares (*Open*). Since 2026-11-02 they
  feed only the comparison flags (`--topic either|llm`). Haiku stance
  (`score_llm`) covers 17,347 of the 24,753 account posts: none of the
  Nov 1 or Nov 22 pulls' or the fills'.
- **X archive fill** (`x-archive fill`; `docs/decisions.md` 2026-11-02 and
  2026-11-22): search/all filled the cache's holes on collect's 35 h grid,
  extended back to Feb 1, with collect's rule (a sampled account up to 5
  posts a window, a take-all one every post X counts). The full run
  (2026-11-03) planned 1,155 windows and read 5,290 posts in 2,890 requests
  ($26.45): 4,708 stored, 582 billed extras set aside in
  `x_archive/fill_<handle>_extra.jsonl` (outside the cache, not analysed),
  none already held. With the pilot's 36, 4,744 rows were filled then. 109
  windows stopped on their window budget though X's hour counts showed them
  whole; a cleanup rerun closed them with no request. The Nov 22 run
  (`X_ARCHIVE_FILL_MAX_USD` lowered to $27.64: billed + the $1 approved)
  filled the 38 windows the Nov 22 collect left empty (19) or short (19):
  94 posts in 66 requests ($0.47), 72 stored, 22 set aside; 3 windows
  stopped on their budget and a cleanup pass closed them with no request.
  All fill runs: 5,422 posts, $27.11; 4,816 rows filled (`"via":
  "search_all"`), 4,645 with text, all of those with `score_opus` and an
  Opus topic class. Every
  archive window is done and `x-archive fill --estimate` plans none. Filled
  rows by 2026-11-03: @WhiteHouse 796, @LauraLoomer 676, @RealAlexJones
  641, @BarakRavid 557, @marklevinshow 402, @StateDept 358, @POTUS 325,
  @VaticanNews 279, @VP 246, @PeteHegseth 194, @RealCandaceO 170, the six
  take-all accounts 8-37 each.
- **X coverage after the fill** (held / X's own count, `x-counts --report`,
  Feb-Oct, 2026-11-03): 24,476 of 67,024 posts (36.5%; 29.4% before). The
  take-all accounts (@SenSanders, @mtgreenee, @TuckerCarlson, @SecRubio,
  @USCCB, @Pontifex) hold 97-100% of X's count every month (a few months a
  little over). The sampled ones are at 5 a 35 h window wherever they were
  thin; their lowest month is now @LauraLoomer 6%, @RealAlexJones 7%,
  @marklevinshow 9%, @WhiteHouse 10%, @BarakRavid 13%, @StateDept 29%,
  @PeteHegseth 31%, @VP and @VaticanNews 33%, @RealCandaceO 37%, @POTUS 48%
  (0-31% before). Stretches read whole earlier stay dense: @marklevinshow,
  @PeteHegseth and @RealCandaceO 99-100% in Feb-Mar, @RealAlexJones 76% and
  @BarakRavid 72% in April (*Open*: uneven density). No archive window is a
  `FETCH SHORTFALL`; each of collect's 57 has an archive row (54 `sampled
  ok`, 2 `edge`, 1 `complete`). A handful of archive windows hold fewer than
  X counts (`short`: deleted since, or left out by search/all). Nov 1-22:
  `x-counts` (2026-11-22, 17 requests, $0.17) has X's hour counts; its
  report flagged 21 `FETCH SHORTFALL` windows among the Nov 22 collect's,
  and the Nov 22 fill planned every window collect left empty or short.
  Phase 6, like phase 5, is read at collect's 35 h design throughout, so
  its design weights nearly equal raw (media's phase-6 cell moves by under
  0.01: one @BarakRavid window weighs a little over 1).
- **@WhiteHouse**: the timeline endpoint fails for it (the Nov 1 pull got
  posts in 5 of its 31 windows; the Nov 22 pull got 32 posts, with 5
  windows empty next to dense ones); search/all serves it. It now holds
  1,495 posts from Feb 1: 5 in each of its 161 archive windows to Nov 1 and
  40 more in 8 windows of Nov 1-20 from the Nov 22 fill (10-44% of X's
  count every month Feb-Oct; October 110 of 1,102).
- **Late starts and horizon holes: filled.** Nine accounts started Feb 21 -
  Apr 9 (@VaticanNews, @POTUS, @Pontifex, @StateDept, @VP, @BarakRavid,
  @RealAlexJones, @LauraLoomer, @WhiteHouse); all start Feb 1-4 now. The
  ~3,200-tweet timeline horizon had emptied May 10 up to Jun 20 - Jul 31 for
  @marklevinshow, @LauraLoomer, @WhiteHouse and @RealAlexJones; their June
  holds 135 of X's 1,332, 103 of 1,361, 101 of 875 and 101 of 1,291. The Sep
  18 run's empty 14-day slices (@POTUS May 4-18, @VP May 22-Jun 5,
  @RealCandaceO Jul 18-Aug 1, @TuckerCarlson Aug 1-15, @LauraLoomer Jul
  19-Aug 16) are filled the same way (@POTUS May 56 -> 109 of 171), and so
  is the Nov 1 pull's first @marklevinshow window (5 posts); its first two
  @VP windows X counts at 0-1 and 0.
- **Opus topic class on every post** (`post-label`, the stance-topic prompt
  on claude-opus-5; the war-post flag of record since 2026-11-02): 28,469
  labels, all 24,753 X account posts with text (search rows out) and all
  3,716 of Trump's. 7,629 come from `--job stance-topic` (the
  keyword-or-Haiku candidates, X 5,661 and Trump 468, plus the 1,500-post
  random arm frozen in `post_random_draws.parquet`), 20,840 from the census
  (`--job stance-topic-census`, X 17,592 and Trump 3,248); the fill's posts
  went in on 2026-11-03 (1,024 candidates, 3,551 by census), the Nov 22
  pull's on 2026-11-22 (180 candidates, 1,242 by census, X and Trump).
  5,691 X and 381 Trump posts are `iran_war` (5,690 X inside the
  window). On 2026-11-03 no earlier filter had flagged 1,576 of the X war
  posts and 32 of Trump's; the random arm, drawn before the fill and
  scaled to the 18,322 unflagged X posts then, put the missed at 1,637
  [1,391, 1,921], around the census's 1,576. All in
  `post_labels_stance-topic_claude-opus-5.parquet`; its stance column
  (`score_opus_stance`, on every X war post) is a robustness check, not a
  series.
- **Alternate raters** (robustness checks, never a series;
  `docs/decisions.md`): Opus 5.5 on the 4,925 pre-fill candidates
  (`score_opus55`, `post_labels_rater_claude-opus-5-5.parquet`; the fill's
  1,024 new candidates and the Nov 22 pull's have none, so its checks rest
  on pre-fill posts) and on the 1,157 v2 replies of the eight older reply
  posts (`reply-label --job v2-rater`); stance-topic without the handle on
  1,957 paired X candidates (`post_nohandle_draws.parquet`, frozen).
- **Retweets** count as the account's messaging (`docs/decisions.md`):
  8,320 of the 25,477 account posts (33%; 67% of admin; @POTUS 1,295 of
  1,295, 1,277 of them @WhiteHouse's), stored cut at about 140 characters.
- **Posts over 280 characters** are whole. Requests ask for `note_tweet`
  since 2026-10-04, and `x-backfill-text` re-read the 2,981 cached
  originals that looked cut at 280 (2026-10-04): 2,980 came back, 2,571 of
  them (86%) longer than the cached text, now stored whole (median 461
  characters); 1 was not returned. What was made from the cut text is in
  `*_superseded.parquet` (2,571 `sentiment_all` rows, 2,570 Opus labels,
  2,571 topic labels), and all 2,571 were rescored and relabelled from the
  full text. They were 929 of the 4,225 war posts in the phases then (22%,
  topic `either`). Effect: `docs/decisions.md`. The archive fill's posts
  came whole (search/all with `note_tweet`); `x-backfill-text` skips them.
- **Full-archive search works** (`x-archive probe`, 2026-11-02): for two
  closed @PeteHegseth days it returned every cached post, retweets
  included, plus 1 not cached; counts/all includes retweets and matches the
  cache (270 vs 268 posts, Feb 1 - Mar 2). The fill (above) bounds its
  requests by id alone: X answers 400 to ids mixed with times.
- **Capped slices**: a capped 14-day slice keeps its newest posts, a median
  of 0.7-2.9 days of the 14 for the five sampled accounts and @WhiteHouse.
  Capped on Sep 18, besides the five: @POTUS, @PeteHegseth, @VP and
  @VaticanNews in 9 slices, @RealCandaceO 7, @WhiteHouse 6, @mtgreenee and
  @Pontifex 1. The fill topped the rest of each slice up to 5 a 35 h window.
- **Keyword searches**: frozen at May 12, with a gap from about Apr 20 to
  May 5 (7-day recent search only); the Nov 22 pull ran `--no-search`.
- **Truth Social**: Trump's feed current through 2026-11-22 15:21 UTC
  (anonymous refresh, 643 new posts from Oct 31 19:28 UTC: 103 in phase 5,
  the rest of Oct 31 and Nov 1; 528 in phase 6; 12 dated Nov 22, past
  `ANALYSIS_END`), 6,019 posts from Apr 4 (contiguous), 3,716 of them with
  text. His stance of record is `score_opus` since 2026-11-02 (Opus 5 on
  llm_prompt, the X tiers' model and prompt): `teacher-check --source
  trump --all` labelled all 3,716 (the check's 400 reused, 2,821 by batch
  on 2026-11-02, the one unparseable answer resent direct on 2026-11-03,
  494 by batch on 2026-11-22) and `--merge` wrote them onto
  `truthsocial_trump_stance.parquet`.
  `score-posts` keeps them on the rows it has (a row that gains text
  loses its scores and labels and is rescored); the posts it adds have
  none: merge again after it.
  `score_opus_distilled` stays for comparison (on 2026-11-02, against Opus
  on the then 3,222: 0.778 Pearson and 1.2% sign flips; on his 368 war
  posts, topic opus, 0.738, and it read the collapse phase 0.081 below
  Opus). The 2,303 posts without text leave every share and mean (as of
  2026-10-06, 1,589 images or videos with no caption and 221 quotes of his
  own posts that have no text: *Trump's posts with no text*, below).
  `truthsocial_trump_stance.parquet` still holds the old constant +0.111
  for 1,622 of the older ones.
  108,409 replies to 10 tracked posts, 108,404 unique
  (`data/raw/truthsocial/replies_*.jsonl`; `these_fools` 4,441 and
  `trump_strait` 11,173 added 2026-09-30, 94.7% and 99.1% coverage;
  `massie_hegseth` 4,869 and `no_deal_needed` 4,872 added 2026-11-22, 99.5%
  and 99.7%), 101,376 of them with text (the `reply-population`
  estimand), scored with RoBERTa and `score_opus_distilled` (the two new
  posts' 9,070 with text on dkbl1's CPU, 2026-11-22, 67 min; decisions.md),
  and the eight older posts' also with `score_mixed_distilled`; plus a
  1,592-row Haiku stance sample (3 errors to retry) and Opus labels on the
  1,430 sampled replies with text (1,157 for the eight older posts, 273
  for the two new ones). The label set of record is v2 (2026-10-05,
  `reply-teacher-check --v2`): made from the whole reply and the Trump
  post it answers (`docs/decisions.md`). Since 2026-11-22 `reply-population`
  pools the 1,344 random draws among them with 10,000 fresh draws
  (`reply_sample_draws_fresh.parquet`, frozen; 9,995 labelled by
  `reply-label --job v2-fresh`, 7,995 on 2026-11-02 and 2,000 on
  2026-11-22; 5 already had v2 labels): 11,344 labelled draws on ten
  posts (9,071 on eight from 2026-11-02; ten since 2026-11-22, the
  user's call, `docs/decisions.md`). Since 2026-11-23 (the owner's call,
  `docs/decisions.md`) the corrected figures take their census from
  `score_ctx_crossfit` (`settings.REPLY_POPULATION_COL`; below), written
  to `reply_population_score_ctx_crossfit_v2.csv`: same estimates (none
  moved outside its old interval), narrower intervals (the pooled mean's
  by 23%). The distilled-census file,
  `reply_population_score_opus_distilled_v2.csv`, is kept as history, and
  the first-draw and v2b files stay on it. `score_ctx_crossfit` is the
  parent-post reply scorer cross-fitted on the 11,425 pooled v2 labels
  (2026-11-23, `reply-crossfit-ctx` on dkbl2's 3080, 4 h 1 min, free):
  every reply with text is scored by a fit that never saw its post's
  labels; its raw scores do not compare across posts. The first draws, 1,500
  (1,200 for the eight posts, 150 for each new one), are frozen in
  `data/processed/reply_sample_draws.parquet`. The eight older posts'
  1,157 labelled replies also carry v2b (v2 without the praise rule,
  reported beside v2), cat-v2 (Opus reply categories made with the parent
  post; Haiku's categories in `stance_sample.parquet` stay as history) and
  Opus 5.5 (above); the two new posts' have none of these. Every batch
  that carried replies was deleted on the API at its collect.
  `reply-loyalty` (free) gives the loyalty gradient on the pooled draws
  (65,051 repliers; the 128 accounts created after
  `LOYALTY_REFERENCE_DATE`, Nov 1, count as age 0), unchanged by the
  census switch (rerun 2026-11-23). `reply-crowd` (free, 2026-11-23,
  1,000 replicates; `data/processed/reply_crowd_*_score_ctx_crossfit_v2.csv`)
  splits the change from the eight earlier posts to the two phase-6 posts
  into accounts seen in both periods and turnover, and sets returners
  against one-off repliers on the same post; published in the README and
  the dkweb post since 2026-11-23 (the owner's call). The v1 labels (1,225;
  423 made from text cut to 100 characters, none with the parent post) and
  their last run, the unsuffixed CSV, are kept as history.
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
- **Timeline**: 171 events through 2026-11-21 in `config/timeline.py`.
  Six phases since 2026-11-22 (the user's call, `docs/decisions.md`):
  phases 1-5 (Feb 1 -> Nov 1, the fifth "midterm hold", Sep 19 -> Nov 1)
  are closed, and phase 6, "midterms, standoff", runs from Nov 2 to
  `ANALYSIS_END`, which rolls to each pull's last whole UTC day
  (2026-11-21 for the Nov 22 pull). From 2026-11-02 to 2026-11-22 the
  analysis was frozen at the Nov 1 pull (`ANALYSIS_END` = 2026-11-01, five
  phases). On 2026-11-22 `phases` (X tiers and Trump, on `score_opus`
  and the Opus topic; raw, with the design- and volume-weighted CSVs
  beside it), `visualize`, `reply-population` and `reply-loyalty` reran
  on six phases and ten reply posts. Of `data/processed/`'s CSVs, those
  runs wrote `phases_x_tier_score_opus_opus*.csv`,
  `phase_contrasts_x_tier_score_opus_opus*.csv`,
  `phase_gaps_x_tier_score_opus_opus*.csv`, the two
  `*trump_user_score_opus_opus.csv`,
  `reply_population_score_opus_distilled_v2.csv` and
  `reply_loyalty_v2.csv`; every figure in `data/processed/figures/` and
  the README's two embedded figures are from the same day. The other CSVs
  predate the refresh (the Nov 3 five-phase run, a few older):
  `phases --by user` (X; the README's per-account cells come from
  `visualize`'s table, the same point estimates), the alternate topic
  definitions (`either`, `llm`, `keyword`), the alternate scorers
  (`score_opus55`, `score_opus_stance`, `score_opus_distilled`), the
  first-draw reply files and the v2 - v2b pairing (*Open*). `export-web`
  wrote the six-phase JSONs into dkweb's draft branch (*Open*). Sources:
  `docs/timeline_candidates_2026-05-12_to_2026-09-15.json`,
  `docs/timeline_candidates_2026-09-16_to_2026-10-31.json` and
  `docs/timeline_candidates_2026-11-01_to_2026-11-22.json` (80 candidates,
  33 selected).
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
- **Spend, 2026-11-02, the Claude label jobs** (approved runs; Anthropic
  only, billed usage from each collect and pilot, logs in
  `data/processed/credit_logs_2026-11-02/`; batch prices, except the three
  pilots run direct at full price: stance-topic $0.14, Opus 5.5 on posts
  $0.17, no-handle $0.13; pilots included in each job's figure):
  Trump's feed on Opus $5.07 (2,822 calls), v2b $2.24 (1,157), cat-v2
  $2.87, fresh reply draws $17.06 (7,995), stance-topic $8.77, Opus 5.5 on
  posts and replies $10.59, no-handle $4.21, the census $22.63 (16,047
  calls in 6 parts). $73.42 in all. No X reads.
- **Spend, 2026-11-03, the X archive fill and its labels** (approved runs):
  X $26.64 at $0.005 a post returned (requests free): the pilot $0.19 (38
  posts; two earlier tries refused with a 400 before any post was read),
  the full run $26.45 (5,290 posts in 2,890 requests), the cleanup pass $0.
  Anthropic, metered from the batch results' usage at batch prices:
  `relabel` $7.05 (4,575 posts; the 2 answers that were not valid JSON
  resent, under a cent), `topic-label` $0.78 (4,671), stance-topic $2.00
  (the 1,024 new candidates), the census $3.97 in two parts ($2.29 for
  2,000 calls, $1.68 for 1,551); Trump's one unparseable answer resent
  direct (one call, under a cent). $13.81 in all.
- **Spend, 2026-11-22, the Nov 22 refresh** (approved runs): X $5.18: the
  pull ~$4.54 (875 posts + 17 lookups, from the row counts), `x-counts`
  $0.17 (17 requests, Nov 1 -> Nov 22), the archive fill $0.47 (94 posts;
  the cleanup pass $0). Anthropic about $9.0, by batch unless marked
  direct, metered from the billed usage unless marked an estimate: Trump's
  feed check $1.10 (494
  posts); `relabel` ~$1.40 (858) and ~$0.11 (70), estimates; `topic-label`
  ~$0.21 (1,448) and ~$0.02 (148), estimates; stance-topic $0.27 (172) and
  $0.01 (8); the census $1.68 (1,180) and $0.07 (62); reply labels v2 $0.47
  (273); `v2-fresh` $2.26 (`massie_hegseth`, 1,000) and $1.14
  (`no_deal_needed`, 1,000); Haiku reply stance, direct (~340 calls, not
  metered, ~$0.3). About $14.2 in all; the Anthropic part from the plan's
  monthly API credit.

## Definitions of record

Since 2026-11-02, the user's calls (`docs/decisions.md`); the 2026-11-03
fill and the 2026-11-22 refresh changed none of the first two, only added
posts:

- **War posts**: Opus's topic class (`settings.TOPIC_SOURCE = "opus"`),
  which every X account post and Trump post with text now has; a post
  without a current label would count by the old either flag, with a
  warning. `--topic
  either|llm|keyword` stay for comparison.
- **Stance**: `score_opus` on X (unchanged) and on Trump's feed
  (`settings.TRUMP_STANCE_SCORE_COL`).
- **Replies**: v2 labels on the first and fresh draws pooled, on ten posts
  since 2026-11-22 (11,344; on eight, 9,071, from 2026-11-02;
  `REPLY_POPULATION_FRESH_DRAWS`; ten posts the user's call), with v2b
  beside them where the praise rule matters. The correction's census is
  `score_ctx_crossfit` since 2026-11-23 (`settings.REPLY_POPULATION_COL`,
  the owner's call). `reply-crowd`'s record is its post-stratified direct
  estimator; its GREG is a check.
- **Phases and weights** (2026-11-22): six phases, the sixth rolling to
  `ANALYSIS_END` (AGENTS.md *Phases and the analysis window*); raw, every
  post weighing 1, is the record, and `phases --weights design|volume`
  writes suffixed CSVs beside it.
- Reported beside them, never a series: Opus 5.5 (`score_opus55`,
  v2-rater), `score_opus_stance`, the no-handle subsample, cat-v2 and the
  loyalty gradient.

## Where things are

- The k8s namespace `iran-sentiment` is deleted again (2026-10-07, after
  the reply-scorer experiment on image 13717d7c504a; its PVC with it), and
  stayed down on 2026-11-22: the new replies were scored on dkbl1's CPU.
  Models, including
  `data/models/reply_ctx_lopo/`, are in `data/models/` on dkbl1 and in the
  S3 backup.
- S3 backup first synced 2026-09-25 (models 8.8 GB in Glacier IR, the
  rest Standard). The last sync on record ran 2026-11-22 at 20:01 UTC,
  after the X collect; none is on record after the day's later runs (the
  fill, the label jobs, the replies and their scoring): run `backup`.
- Archive fill state: `data/raw/x_archive/` (`fill_<handle>.json` per
  account, `fill_requests.jsonl`, the set-aside `fill_<handle>_extra.jsonl`)
  and `data/raw/x_sampling/windows_archive_2026-11-03.jsonl` (1,274 rows for
  1,162 windows; reruns, the pilot's and the cleanup pass, logged 110 of
  them more than once) and `windows_archive_2026-11-22.jsonl` (41 rows for
  38 windows).
- Host backup (restic, homelab-infra): the SanDisk drive is unplugged and no
  host backup has run since 2026-04-23. The host backup script no longer
  needs Timeshift, so the drive can go back in; until then S3 is the only
  off-box copy of `data/`.
- Batch state files in `data/processed/`: `relabel_batch.json` (`relabel
  status|collect|merge|resubmit`), `topic_batch.json`,
  `teacher_batch_reply_v2.json` and `teacher_batch_trump.json` (`--batch
  status|collect`), `teacher_batch_post_<job>.json` (`post-label`) and
  `teacher_batch_reply_<job>.json` (`reply-label`). The downloaded results
  and earlier batches' state sit beside them. The Batch API keeps results
  for 29 days. Label files: `post_labels_<job>_<model>.parquet`,
  `teacher_labels_replies_<set>_<model>.parquet`,
  `teacher_check_trump_claude-opus-5.parquet`.

## Open

- **The dkweb post** `iran-war-stance` is behind Cloudflare Access until
  launch (merged to dkweb `main` 2026-10-04, 2026-11-02 and 2026-11-23). dkweb
  is a private repo; the findings are public only through this repo.
  From 2026-11-22 (the user's call) it was extended to phase 6 on dkweb's
  local branch `iran-post-phase6`: export-web's six-phase, ten-post JSONs
  (replies.json on the cross-fit census since 2026-11-23), the post's text
  and the *Who came back in November* finding. The owner's calls of
  2026-11-23 (`docs/decisions.md`): the branch merged to `main` that day,
  with the post dated 2026-11-23 (it never launched, so no update note);
  the description stays; the decomposition chart stays strikes ->
  "midterm hold" (`settings.WEB_DECOMPOSITION_PHASE`). Still open:
  launching, the owner's step. Under the Opus flag
  @Pontifex (16 war posts) and @USCCB (15) fall under
  `WEB_MIN_ACCOUNT_POSTS` (20), so the account charts show 15 accounts, not
  17; the post says why the Pope is left out. Still worth a look on the
  page: the reply-scatter labels and the audience bars' domains, now with
  ten posts.
- **`reply-crowd`**: done and published 2026-11-23 (the owner's call,
  `docs/decisions.md`): the decisions entry approved, the split (P2)
  leading and returners (P1) second, in the README and the dkweb post.
  Still open, each the owner's call and paid: the labels its entry lists
  (a praise-rule check on the phase-6 first draws, a label retest, a
  census of the both-period accounts' earlier replies).
- **`score_ctx_crossfit`**: done 2026-11-23; the census of the published
  reply figures since that day (the owner's call, `docs/decisions.md`).
  The GPU namespace was deleted after the pull (2026-11-23); reviving it
  is in AGENTS.md *Where things run*, and every reply refresh now needs
  it (*What a refresh needs*).
- **The slope-1 assisted estimator** (`reply-population`'s `assisted_*`,
  the published corrected figures): whether to fix it, by GREG with an
  estimated slope or by publishing direct only, is the owner's call
  (`docs/decisions.md` 2026-11-23, reply-crowd's *Owner's calls*). A fix
  moves the published reply intervals. On the cross-fit census its
  interval is wider than direct (`opus_*`) on one post's mean
  (massie_hegseth, by 0.0002), four posts' pro-war shares (by up to
  0.006), six posts' anti-war shares and the ALL row's (ceasefire 37%
  wider, hold_off_attack 27%, deal_complete and power_plant_day 14%,
  trump_strait 7%, civilisation_dies 5%; ALL 11%) and seven posts'
  neutral shares and the ALL row's (by at most 0.003).
- **Phase 6 is open.** It closes at the next regime change (major
  US/Israeli strikes resume or a deal is signed) or on 2027-01-02,
  whichever comes first, the user's call; its last day then goes into
  `PHASES` and `ANALYSIS_END` stops rolling (`docs/decisions.md`
  2026-11-22). Its cells are thin: 20
  days, 41 pro-war and 44 anti-war MAGA war posts, 31 of the
  administration's, 13 of Trump's.
- **The next pull.** The heavy accounts' timeline reaches back only
  ~3,200 posts, so pull them by about mid-January at the latest, or the
  gap's windows come back empty and wait for a paid `x-archive fill`; a
  sampled account re-pulled within a few days gets a short last window at
  up to twice the density (AGENTS.md *X collection and budget*). The
  census's call cap needs raising before it, and stance-topic's too if the
  pull is much longer than Nov 22's (*What a refresh needs*).
- **Five-phase CSVs left over** (2026-11-22). `phases --by user` (X), the
  alternate topic definitions and scorers, `summary`, the first-draw reply
  files (with `--col score_opus_distilled`, the census the README quotes
  them on; the default is now `score_ctx_crossfit`) and the v2 - v2b
  pairing (direct, the same on either census) were not rerun on six
  phases (*Timeline*
  above): they are free but write `data/processed/`. The next free run
  should redo them, `phases --by user` first, so the README's phase-6
  per-account cells (from `visualize`'s table) have a CSV of record, and
  the README's whole-war tier table and robustness checks catch up.
- **Uneven density** (from the X archive fill, 2026-11-03). Each sampled
  account is dense where a pull read it whole (parts of Feb-Apr for most,
  and the newest days of each capped 14-day slice) and at 5 a 35 h window
  elsewhere, and raw `phases` weights every post equally. In the strike
  phase @marklevinshow is 78% of pro-war MAGA's posts (86% of its war
  posts) and @RealAlexJones 51% of anti-war MAGA's (65%). The anti-war MAGA
  weekly line's April rise is @RealAlexJones going from 20-25 posts a week
  (Feb-Mar, 5 a window) to 175-396 (Mar 30-Apr 13, 77-100% of X's count).
  `phases --weights design|volume` (2026-11-22, `docs/decisions.md`)
  weighs by the window manifests (`data/raw/x_sampling/`) and X's hour
  counts, as an option beside raw, which stays the record. With `x-counts`
  through Nov 22 every post has a weight: on 2026-11-22 design nearly
  equals raw in phases 5 and 6 (both read at 5 a window; media's phase-6
  cell moves by under 0.01), and volume keeps phase 6's
  direction (anti-war MAGA's war share 16% -> 31%, pro-war MAGA's 11% ->
  26%; under volume @RealAlexJones is 93% of anti-war MAGA's weighted war
  posts). Making a weighting the record is the user's call.
- The X teacher check needs no new labels after the backfill (0 of its 497
  to do), but 63 of its posts were backfilled: their direct Opus labels saw
  the cut text, while `sentiment_all` now holds full-text Haiku scores, so
  rerunning `teacher-check` (x) would compare different inputs on those 63.
  The 2026-09-18 report is unaffected, and `teacher-retest` pairs those
  labels with the archived batch labels made from the same cut text.
- `analyze` restores a prior score only onto the same text (2026-10-04),
  and archives a prior row with a paid score (Haiku's) before rescoring it,
  but the paid labels are keyed by id. `x-backfill-text`, X `collect
  --force` (which merges into the cache by id since 2026-10-06),
  `ts-fill-text` and a forced `collect-truth` move the labels of the posts
  whose text they change. Fixed 2026-11-22: a forced `collect-truth` used to keep them.
- Reply scorers against the v2 labels, settled 2026-10-07: a scorer that
  reads each reply with its Trump post beats the scorer of record
  leave-one-post-out (pooled 0.64 vs 0.58; June deal 0.62 vs 0.14) but
  would narrow the published intervals only modestly, so it stayed an
  experiment (`docs/decisions.md`). Refit on the pooled labels on
  2026-11-23 (held out, pooled 0.68 vs 0.57), its cross-fit became the
  census of the published reply figures. The old mixed model
  (`score_mixed_distilled`, v1 rows) is history. The 2026-11-22 drift check
  on the two new posts (0.58 Pearson against their v2-fresh labels) found
  no reason to refit.
- 67 posts with text have no Haiku topic label (36 Trump's, 31 on X, 26 of
  them Levin's). All carry a link: link shares Haiku will not judge (a
  link under a word or two such as "Amen" or "Right on"; it answers that
  it cannot open URLs and hits `max_tokens`). Each has an Opus topic class, so only the comparison flags
  (`either`, `llm`) count them, by the keyword pattern. `topic-label` stops
  sending a post after three no-verdict answers on its text:
  `topic_unjudgeable.parquet` holds 241 such answers on 108 posts, and 57
  of the 67 have reached three; `topic-label unjudgeable --results-file`
  records the earlier batches' (free).
- 5 duplicate reply ids in the April reply files (3 `civilisation_dies`,
  2 `power_plant_day`); `reply-population` and `export-web` drop the
  copies.
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).

## What a refresh needs

The order is in AGENTS.md *Running things*. While phase 6 is open, each
pull moves `ANALYSIS_END` to its last whole UTC day (AGENTS.md *Phases and
the analysis window*), so its new posts enter phase 6 once labelled; the
pull's partial last day waits for the next pull. Beyond `relabel` and
`topic-label`, since 2026-11-02:

- **X counts and holes** (paid, *Ask first*): `x-counts` for the new range
  first (`phases --weights` and the fill's plan need X's hour counts;
  2026-11-22: 17 requests, $0.17), then `x-archive fill --estimate` (free)
  plans the windows collect left short or empty, inside `ANALYSIS_END`; it
  plans none today. `X_ARCHIVE_FILL_MAX_USD` is $27.64 since 2026-11-22
  (billed $26.64 + the $1 approved for the Nov 22 fill) with $27.11 billed:
  room for 106 posts ($0.53), about one fill the size of Nov 22's (94
  posts). It is raised only on the user's say.
- **Opus topic labels for new posts** (paid, *Ask first*): `post-label
  estimate|submit|status|collect --job stance-topic` for new
  keyword-or-Haiku candidates, then `--job stance-topic-census` for the
  rest (it sends any post with text that has no label on its current text,
  candidates included). Until then a post counts by the either flag and
  `phases` warns with the count. Both jobs' caps bound the whole run
  (`post-label estimate` prints what each has used): stance-topic has used
  7,629 calls and $11.06 of 8,000 and $30, the census 20,840 calls and
  $28.35 of 21,500 and $33. That leaves 371 and 660 calls, and $4.65 of the
  census's dollars; the Nov 22 pull (three weeks of posts) needed 180 and
  1,242 calls ($0.28 and $1.75). So the census's call cap must be raised
  before the next pull, and stance-topic's 371 calls cover a pull the size
  of Nov 22's but not one much longer (a pull in mid-January would cover
  about eight weeks against Nov 22's three); raising either is the user's
  call.
- **Trump's stance**: after `score-posts`, `teacher-check --source trump
  --all --estimate`, `--batch submit|status|collect`, then `--merge` (again
  after any later `score-posts`).
- **Replies**: the ten tracked posts' draws (first and fresh) are frozen;
  phase 6's two posts are picked (`docs/decisions.md` 2026-11-22).
  `REPLY_FRESH_MAX_CALLS` / `REPLY_FRESH_MAX_USD` (10,500 and $25 since
  2026-11-22) bound the whole `v2-fresh` run: 9,995 calls and $20.45 used,
  so another post's 1,000 fresh draws ($1.14-2.26 a post on 2026-11-22)
  needs them raised. A new reply post is the user's call.
  `reply-population`, `reply-loyalty`, `reply-crowd` and export-web refuse
  a reply with text and no `score_ctx_crossfit`. New replies to a tracked
  post, with no new labels: `sync-data.sh push` (the PVC must hold them),
  the `reply-crossfit-ctx` Job (it refits that post's fold and rescores
  the post whole, ~24 min), then `sync-data.sh pull`. A new tracked post,
  or any new v2 or v2-fresh labels: every fold's training rows change and
  the column's record (`n_rows`, `rows_sha`) refuses them, so it is a full
  cross-fit into a new column (~4 h for ten folds, longer with more), and
  moving `settings.REPLY_POPULATION_COL` to that column changes the
  published census: the owner's call (`docs/decisions.md` 2026-11-23).
