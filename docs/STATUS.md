# Status

As of 2026-11-03. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 24,530 posts from 17 accounts plus 1,768 keyword-search
  rows (26,298), Feb 1 -> Nov 1, 2026; since the archive fill (below) every
  account starts Feb 1-4. The Sep 18 -> Nov 1 pull (2026-11-01, the 35 h
  window sampler, `docs/decisions.md`) read 1,949 posts plus 17 user lookups
  (~$9.92): 11 accounts sampled at 5 per window, 6 take-all; per-window rows
  in `data/raw/x_sampling/windows_2026-11-01.jsonl`. Before it: the May 10
  -> Sep 18 refresh read 5,631 tweets for ~$28, with the five heaviest
  accounts (Loomer, Levin, Jones, Ravid, StateDept) sampled at 450. 705
  account posts have no text (nearly all bare links) and leave every series
  (`src/text_rules.py`). `score_opus` on all 25,593 posts with text (23,825
  account posts, 1,768 search rows); the 363 rows without it have no text.
  Haiku topic labels on 28,719 of the 28,815 posts with text (X accounts
  23,769 of 23,825, keyword search 1,768 of 1,768, Trump 3,182 of 3,222);
  the other 96 are link shares (*Open*). Since 2026-11-02 they feed only
  the comparison flags (`--topic either|llm`). Haiku stance (`score_llm`)
  covers 17,347 of the 23,825 account posts: none of the Nov 1 pull's or
  the fill's.
- **X archive fill** (`x-archive fill`, 2026-11-03; `docs/decisions.md`
  2026-11-02): search/all filled the cache's holes on collect's 35 h grid,
  extended back to Feb 1, with collect's rule (a sampled account up to 5
  posts a window, a take-all one every post X counts). The full run planned
  1,155 windows and read 5,290 posts in 2,890 requests ($26.45): 4,708
  stored, 582 billed extras set aside in `x_archive/fill_<handle>_extra.jsonl`
  (outside the cache, not analysed), none already held. With the pilot's
  36, 4,744 rows are filled (`"via": "search_all"`), 4,575 with text, all
  with `score_opus` and an Opus topic class (below). 109 windows stopped on
  their window budget though X's hour counts showed them whole; a cleanup
  rerun closed them with no request. All 1,162 archive windows, the pilot's included, are done, and
  `x-archive fill --estimate` plans none. Filled rows: @WhiteHouse 796,
  @LauraLoomer 676, @RealAlexJones 641, @BarakRavid 557, @marklevinshow 402,
  @StateDept 358, @POTUS 325, @VaticanNews 279, @VP 246, @PeteHegseth 194,
  @RealCandaceO 170, the six take-all accounts 8-37 each.
- **X coverage after the fill** (held / X's own count, `x-counts --report`,
  Feb-Oct): 24,476 of 67,024 posts (36.5%; 29.4% before). The take-all
  accounts (@SenSanders, @mtgreenee, @TuckerCarlson, @SecRubio, @USCCB,
  @Pontifex) hold 97-100% of X's count every month (a few months a little
  over). The sampled ones are at 5 a 35 h window wherever they were thin;
  their lowest month is now @LauraLoomer 6%, @RealAlexJones 7%,
  @marklevinshow 9%, @WhiteHouse 10%, @BarakRavid 13%, @StateDept 29%,
  @PeteHegseth 31%, @VP and @VaticanNews 33%, @RealCandaceO 37%, @POTUS 48%
  (0-31% before). Stretches read whole earlier stay dense: @marklevinshow,
  @PeteHegseth and @RealCandaceO 99-100% in Feb-Mar, @RealAlexJones 76% and
  @BarakRavid 72% in April (*Open*: uneven density). No archive window is a
  `FETCH SHORTFALL`; each of collect's 57 has an archive row (54 `sampled
  ok`, 2 `edge`, 1 `complete`). A handful of archive windows hold fewer than
  X counts (`short`: deleted since, or left out by search/all).
- **@WhiteHouse**: the timeline endpoint fails for it (the Nov 1 pull got
  posts in 5 of its 31 windows); search/all serves it. It now holds 1,423
  posts from Feb 1, 5 in each of its 161 archive windows, 10-44% of X's
  count every month (October 110 of 1,102; it was 10).
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
  on claude-opus-5; the war-post flag of record since 2026-11-02): 27,047
  labels, all 23,825 X account posts with text (search rows out) and all
  3,222 of Trump's. 7,449 come from `--job stance-topic` (the
  keyword-or-Haiku candidates, X 5,503 and Trump 446, plus the 1,500-post
  random arm frozen in `post_random_draws.parquet`), 19,598 from the census
  (`--job stance-topic-census`, X 16,822 and Trump 2,776, in 8 parts); the
  fill's posts went in on 2026-11-03 (1,024 candidates, 3,551 by census).
  5,552 X and 368 Trump posts are `iran_war`; no earlier filter flagged
  1,576 and 32 of them. The random arm was drawn before the fill; scaled to
  today's 18,322 unflagged X posts it puts the missed at 1,637 [1,391,
  1,921], around the census's 1,576. All in
  `post_labels_stance-topic_claude-opus-5.parquet`; its stance column
  (`score_opus_stance`, on all 5,552 X war posts) is a robustness check,
  not a series.
- **Alternate raters** (robustness checks, never a series;
  `docs/decisions.md`): Opus 5.5 on the 4,925 pre-fill candidates
  (`score_opus55`, `post_labels_rater_claude-opus-5-5.parquet`; the fill's
  1,024 new candidates have none, so its checks rest on pre-fill posts) and
  on the 1,157 v2 replies (`reply-label --job v2-rater`); stance-topic
  without the handle on 1,957 paired X candidates
  (`post_nohandle_draws.parquet`, frozen).
- **Retweets** count as the account's messaging (`docs/decisions.md`):
  8,048 of the 24,530 account posts (33%; 67% of admin; @POTUS 1,239 of
  1,239, 1,222 of them @WhiteHouse's), stored cut at about 140 characters.
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
  May 5 (7-day recent search only).
- **Truth Social**: Trump's feed current through 2026-10-31 (anonymous
  refresh, 1,016 new posts), 5,376 posts from Apr 4 (contiguous), 3,222 of
  them with text. His stance of record is `score_opus` since 2026-11-02
  (Opus 5 on llm_prompt, the X tiers' model and prompt):
  `teacher-check --source trump --all` labelled all 3,222 (the check's 400
  reused, 2,821 by batch on 2026-11-02, the one unparseable answer resent
  direct on 2026-11-03) and `--merge` wrote them onto
  `truthsocial_trump_stance.parquet`.
  `score-posts` rewrites that parquet without them: merge again after it.
  `score_opus_distilled` (the 672 new posts with text on dkbl1's CPU,
  2026-11-02, thread-capped) stays for comparison: against Opus on the
  3,222, 0.778 Pearson and 1.2% sign flips; on his 368 war posts (topic
  opus) 0.738, and it reads the collapse phase 0.081 below Opus. The 1,810
  without text leave every share and
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
  label set of record is v2 (2026-10-05, `reply-teacher-check --v2`): made
  from the whole reply and the Trump post it answers
  (`docs/decisions.md`). Since 2026-11-02 `reply-population` pools the
  1,071 random draws among them with 8,000 fresh draws
  (`reply_sample_draws_fresh.parquet`, frozen; 7,995 labelled by
  `reply-label --job v2-fresh`, 5 already had v2 labels): 9,071 labelled
  draws, written to `reply_population_score_opus_distilled_v2.csv`. The
  first 1,200 draws are frozen in `data/processed/reply_sample_draws.parquet`.
  The same 1,157 replies also carry v2b (v2 without the praise rule,
  reported beside v2), cat-v2 (Opus reply categories made with the parent
  post; Haiku's categories in `stance_sample.parquet` stay as history) and
  Opus 5.5 (above). Every batch that carried replies was deleted on the API
  at its collect. `reply-loyalty` (free) gives the loyalty gradient on the
  pooled draws (60,852 repliers). The v1 labels (1,225; 423 made from
  text cut to 100 characters, none with the parent
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
- **Timeline**: 138 events through 2026-10-31 in `config/timeline.py`.
  The analysis is frozen at the Nov 1 pull (2026-11-02, the user's call):
  `ANALYSIS_END` = 2026-11-01 and a fifth phase, "midterm hold"
  (Sep 19 -> Nov 1), ending before the Nov 3 midterms. `phases`,
  `visualize` and `export-web` reran on five phases, again on 2026-11-02
  under the new definitions (below), and on 2026-11-03 after the archive
  fill: the `data/processed/` CSVs and figures, the README's two embedded
  figures and the dkweb JSONs (accounts, decomposition, maga_weekly and
  trump_phases changed; replies and events did not; *Open*). Sources: `docs/timeline_candidates_2026-05-12_to_2026-09-15.json`
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

## Definitions of record

Since 2026-11-02, the user's calls (`docs/decisions.md`); the 2026-11-03
fill changed none of them, only added posts:

- **War posts**: Opus's topic class (`settings.TOPIC_SOURCE = "opus"`),
  which every X account post and Trump post with text now has; a post
  without a current label would count by the old either flag, with a
  warning. `--topic
  either|llm|keyword` stay for comparison.
- **Stance**: `score_opus` on X (unchanged) and on Trump's feed
  (`settings.TRUMP_STANCE_SCORE_COL`).
- **Replies**: v2 labels on the first and fresh draws pooled (9,071;
  `REPLY_POPULATION_FRESH_DRAWS`), with v2b beside them where the praise
  rule matters.
- Reported beside them, never a series: Opus 5.5 (`score_opus55`,
  v2-rater), `score_opus_stance`, the no-handle subsample, cat-v2 and the
  loyalty gradient.

## Where things are

- The k8s namespace `iran-sentiment` is deleted again (2026-10-07, after
  the reply-scorer experiment on image 13717d7c504a; its PVC with it).
  Models, including
  `data/models/reply_ctx_lopo/`, are in `data/models/` on dkbl1 and in the
  S3 backup.
- S3 backup first synced 2026-09-25 (models 8.8 GB in Glacier IR, the
  rest Standard). Last synced 2026-11-03, after the archive fill, its $0
  cleanup pass, `analyze`, each of the day's label jobs, the outputs
  rerun after them and the docs commit.
- Archive fill state: `data/raw/x_archive/` (`fill_<handle>.json` per
  account, `fill_requests.jsonl`, the set-aside `fill_<handle>_extra.jsonl`)
  and `data/raw/x_sampling/windows_archive_2026-11-03.jsonl` (1,274 rows for
  1,162 windows; reruns, the pilot's and the cleanup pass, logged 110 of
  them more than once).
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

- **The dkweb post** `iran-war-stance` is on dkweb `main` (merged
  2026-10-04, and again 2026-11-02 from branch `iran-post-midterm-hold`:
  data to Nov 1, a midterm-hold section, five-phase charts, the export
  under the new definitions and the prose brought to it), behind
  Cloudflare Access until launch. dkweb is a private repo, so drafts merge
  to its `main` before launch (the user's call, 2026-11-02); the findings
  are public only through this repo. Under the Opus flag @Pontifex (16 war posts)
  and @USCCB (15) fall under `WEB_MIN_ACCOUNT_POSTS` (20), so the account
  charts show 15 accounts, not 17; the post says why the Pope is left out.
  Dated 2026-11-02, with no in-post update notes (it never launched).
  `export-web` rewrote its JSONs on dkweb `main` on 2026-11-03 after the
  fill; `npm run build` has not run since. Still worth a look on the page:
  the reply-scatter labels and the audience bars' domains. Launching it is
  the user's call.
- **The X archive fill ran** (2026-11-03, $26.64 with the pilot; *Data*
  and `docs/decisions.md`). What it leaves open, the user's calls:
  - Uneven density. Each sampled account is dense where a pull read it
    whole (parts of Feb-Apr for most, and the newest days of each capped
    14-day slice) and at 5 a 35 h window elsewhere, and `phases` weights every
    post equally. In the strike phase @marklevinshow is 78% of pro-war
    MAGA's posts (86% of its war posts) and @RealAlexJones 51% of anti-war
    MAGA's (65%). The anti-war MAGA weekly line's April rise is
    @RealAlexJones going from 20-25 posts a week (Feb-Mar, 5 a window) to
    175-396 (Mar 30-Apr 13, 77-100% of X's count). Weighting by the window
    manifests (`data/raw/x_sampling/`, which nothing reads yet) would even
    it out.
  - Extending past Nov 1 (`ANALYSIS_END`, the midterms). The timeline
    (`collect`) stays the forward pull; `x-counts` for the new range and
    `x-archive fill` (its grid ends at `ANALYSIS_END`) cover the windows it
    leaves short.
- `teacher-check --source trump --all --estimate` crashes (AttributeError)
  when no post is left to label: `stance_local.trump_check_estimate` calls
  `.str.len()` on the empty to-do frame's prompt column. Harmless while the
  feed is fully labelled (3,222 of 3,222); fix it before the next feed
  refresh needs an estimate.
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
- 96 posts with text have no Haiku topic label (40 Trump's, 56 on X, 43
  of them Levin's). All carry a link: link shares Haiku will not judge (a
  link under a word or two such as "Amen" or "Right on"; it answers that
  it cannot open URLs and hits `max_tokens`). Each has an Opus topic class, so only the comparison flags
  (`either`, `llm`) count them, by the keyword pattern. `topic-label` stops
  sending a post after three no-verdict answers on its text: the 2026-11-03
  batch recorded 96, one each, in `topic_unjudgeable.parquet`, and
  `topic-label unjudgeable --results-file` records the earlier batches'
  (free).
- 5 duplicate reply ids in the April reply files (3 `civilisation_dies`,
  2 `power_plant_day`); `reply-population` and `export-web` drop the
  copies.
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).

## What a refresh needs

The order is in AGENTS.md *Running things*. Posts after Nov 1 enter no
series while `ANALYSIS_END` stays there (the user's call); posts filled
inside the window do (the 2026-11-03 fill). Beyond `relabel` and
`topic-label`, since 2026-11-02:

- **X holes** (paid, *Ask first*): `x-archive fill --estimate` (free) plans
  the windows a collect left short or empty, inside `ANALYSIS_END` and
  where `x-counts` has X's count; it plans none today.
  `X_ARCHIVE_FILL_MAX_USD` ($32) has room for 1,072 more posts; it is
  raised only on the user's say.
- **Opus topic labels for new posts** (paid, *Ask first*): `post-label
  estimate|submit|status|collect --job stance-topic` for new
  keyword-or-Haiku candidates, then `--job stance-topic-census` for the
  rest (it sends any post with text that has no label on its current text,
  candidates included). Until then a post counts by the either flag and
  `phases` warns with the count. Both jobs' caps bound the whole run, so
  little room is left: stance-topic has used 7,449 calls and $10.77 of
  7,600 and $30, the census 19,598 and $26.60 of 20,000 and $33 (both
  raised 2026-11-03 for the fill's posts). A pull of more than a few
  hundred new posts needs them raised, the user's call.
- **Trump's stance**: after `score-posts`, `teacher-check --source trump
  --all --estimate` (it crashes while nothing is left to label, *Open*),
  `--batch submit|status|collect`, then `--merge` (again after any later
  `score-posts`).
- **Replies**: nothing. The draws are frozen; a new tracked post would
  need its own draws and labels.
