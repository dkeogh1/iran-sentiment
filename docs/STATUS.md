# Status

As of 2026-10-04. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 17,837 posts from 17 accounts plus 1,768 keyword-search
  rows (19,605 in all), Feb 1 -> Sep 18, 2026, though nine accounts start
  later (below). The May 10 -> Sep 18 refresh read 5,631 tweets for ~$28,
  with the five heaviest accounts (Loomer, Levin, Jones, Ravid, StateDept)
  sampled at 450. 490 posts have no text (nearly all bare links) and leave
  every series (`src/text_rules.py`). `score_opus` on 19,110 of the 19,115 posts
  with text after one resubmit (the 5 left are context-free quote-tweets);
  Haiku topic labels on 21,885 posts (X and Trump's feed, complete as of
  2026-09-30 except 458 link shares).
- **Retweets** count as the account's messaging (`docs/decisions.md`):
  5,683 of the 17,837 account posts (32%; 70% of admin; @POTUS 789 of 789,
  783 of them @WhiteHouse's), stored cut at about 140 characters.
- **Posts over 280 characters**: 2,981 cached originals look stored cut at
  280 (`x-backfill-text`'s rule; ~89% of them truly cut, from the length
  density below the pile-up), since `note_tweet` was not requested before
  2026-10-04: 883 of the 4,044 war posts in the phases (22%, topic
  `either`), 37% of the religious tier's, 30% of pro-war MAGA's, 29% of
  anti-war MAGA's, 18% of media's, 8% of admin's. Their Opus and topic
  labels saw only the cut text.
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
  refresh), 4,360 posts from Apr 4 (contiguous), scored with
  `score_opus_distilled`. 1,622 have no text and leave every share and
  mean: ReTruths whose text the collector dropped before 2026-10-04, and
  image or video posts (the cache can't tell which);
  `truthsocial_trump_stance.parquet` still holds their old constant +0.111.
  98,668 replies to 8 tracked posts, 98,663 unique (`data/raw/truthsocial/replies_*.jsonl`;
  `these_fools` 4,441 and `trump_strait` 11,173 added 2026-09-30, 94.7% and
  99.1% coverage), 92,306 of them with text (the `reply-population`
  estimand), scored with RoBERTa, `score_opus_distilled` and
  `score_mixed_distilled`, plus a 1,292-row Haiku stance sample (1 error to
  retry) and 1,225 Opus-labelled replies. 1,071 of those are random draws with text, which
  `reply-population` weights; the 1,200 draws are frozen in
  `data/processed/reply_sample_draws.parquet`. 423 of the 1,225 labels were
  made from text cut to 100 characters, and none saw the parent post.
  `score_opus_distilled` was rescored 2026-09-30 at its 256-token
  training length; `score_opus_distilled_v0` keeps the old reply scores
  (`docs/decisions.md`).
- **Timeline**: 84 events through 2026-09-15 in `config/timeline.py`;
  `ANALYSIS_END` = 2026-09-18. Sources for the Sep 2026 additions are in
  `docs/timeline_candidates_2026-05-12_to_2026-09-15.json`.

## Where things are

- The k8s namespace `iran-sentiment` is deleted again (2026-09-30, after the
  256-token scoring Jobs; its PVC with it). Models are in `data/models/` on
  dkbl1 and in the S3 backup.
- S3 backup first synced 2026-09-25: raw 72 MB and processed 50 MB in
  Standard, models 8.8 GB in Glacier IR, verified object for object. Last
  synced 2026-09-30 after the reply refresh and the GPU pull; run `backup`
  for `data/processed/reply_sample_draws.parquet` (new 2026-10-04).
- Host backup (restic, homelab-infra): the SanDisk drive is unplugged and no
  host backup has run since 2026-04-23. The host backup script no longer
  needs Timeshift, so the drive can go back in; until then S3 is the only
  off-box copy of `data/`.
- Batch state files: `data/processed/relabel_batch.json` (`relabel
  status|collect|merge|resubmit`) and `data/processed/topic_batch.json`. The
  Batch API keeps results for 29 days.
- `main` is pushed.

## Open

- The dkweb post `iran-war-stance` (7 Observable Plot charts, the stance
  colour tokens) merged to dkweb `main` on 2026-10-04 and deploys with it,
  behind Cloudflare Access until launch. Branch builds did run; the Sep 30
  "no preview" note predated the build finishing. The reply-scatter labels
  for the two new posts still need a visual check on the deployed page or
  `npm run preview` on 127.0.0.1 over an SSH tunnel. The 2026-10-04
  `export-web` (no-text rule) changed four of its JSONs, not yet committed
  in dkweb.
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
- **Paid, the user's call: relabel all 1,225 Opus reply labels with the full
  reply text and the parent post.** 423 were made from text cut to 100
  characters, and none saw the Trump post being answered: under the
  ceasefire, hold-off and deal posts, replies that only express agreement
  are labelled pro-war about three times in four (a rough keyword count,
  105 replies; none read or relabelled). The audience shares and the
  reply scorer checks all rest on these labels. Needs the teacher prompt to
  carry the parent first. Estimate: about $4-5 on the direct API that
  `reply-teacher-check` uses (~$2 a run so far; the full text and the parent
  roughly double each prompt), about half that by Batch.
- **Paid: run `x-backfill-text`** (built 2026-10-04, not run): it re-reads
  the 2,981 posts above by id with `note_tweet` (≈ $14.91 at $0.005 a read;
  the lookup's price is unverified, so pilot `--limit 100` and check the X
  console), journals every batch, rewrites the raw text, and moves what was
  made from the cut text to `*_superseded.parquet`: the Opus and topic
  labels, and the posts' `sentiment_all` rows (the only copy of their Haiku
  `score_llm`). It stops before journalling a batch once 20 or more posts
  read carry no `note_tweet` (~90% should), and refuses to start while a
  `collect` or another run holds the X cache. Then `analyze` rescores only
  those posts, and `relabel` and `topic-label` label only them (plus the 5
  and 319 they already resubmit); collecting a batch submitted before the
  backfill again skips them. The relabel costs more than `relabel
  estimate`'s average-post rate: `x-backfill-text` prints the full-length
  figure after the reads, at 2 characters a token, which errs high (~$6-7
  if the whole posts run 600-1,000 characters; ~$5-6 at the 2.8 the
  2026-10-04 reply pilot measured), plus ~$0.4-0.7 in topic labels.
- After `x-backfill-text`, `teacher-check` (x) stays fully cached:
  `training_frame` takes a changed post's Haiku score and cut text back
  from `sentiment_all_superseded.parquet`, so the seeded 497-post sample is
  unchanged (checked on a copy of the data: sample identical, 0 to label;
  without it ~400 new direct Opus calls). `sync-data.sh push` carries that
  archive, which the k8s teacher-check Job needs. Running `analyze --llm`
  on the changed posts afterwards (paid) gives them a full-text Haiku score
  instead, which the check would then compare with its cut-text Opus
  labels for the sampled posts among them (96 of the 497 are candidates).
- X `collect --force` still replaces an account's cache with what the forced
  run fetched, capped. Decide on merging by id before any forced backfill.
- `analyze` restores a prior score only onto the same text (2026-10-04),
  and archives a prior row with a paid score (Haiku's) before rescoring it,
  but the paid labels are still keyed by id: `x-backfill-text` moves the
  labels of the posts it changes, while any other re-fetch that changes a
  post's text (`collect --force`) keeps the old Opus and topic labels.
- **Free, but uses Truth Social: recover past ReTruth text** for the 1,622
  textless Trump posts. The collector now keeps a ReTruth's text, but only
  for posts it fetches; the cached ones need a status lookup the collector
  doesn't have yet. Some of the 1,622 are image or video posts with no text
  to recover. `score-posts` (the `score-trump-feed` Job) rescores rows that
  gain text.
- The mixed model beat the scorer of record on the two unseen posts (0.658
  vs 0.604 Pearson against Opus, gap CI [-0.01, +0.12]). Recheck when the
  next tracked posts get Opus labels, and after the reply relabel above.
- 458 posts are link shares Haiku will not judge (a bare URL, or "Amen" /
  "Right on" over a link: it answers that it cannot open URLs and hits
  `max_tokens`). 139 are bare links with no text and now leave every
  series; for the other 319 (183 Trump's, 112 Levin's) `war_flag` falls
  back to the keyword pattern by design. `topic-label` now skips the 139
  without text but still resubmits the other 319 on every run (~$0.05).
  Worth recording them as unjudgeable.
- 5 duplicate reply ids in the April reply files (3 `civilisation_dies`,
  2 `power_plant_day`); `reply-population` and `export-web` drop the
  copies.
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).
