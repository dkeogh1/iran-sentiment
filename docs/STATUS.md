# Status

As of 2026-09-30. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 19,605 posts from 17 accounts, Feb 1 -> Sep 18, 2026.
  The May 10 -> Sep 18 refresh read 5,631 tweets for ~$28, with the five
  heaviest accounts (Loomer, Levin, Jones, Ravid, StateDept) sampled at 450.
  `score_opus` on 19,452 of 19,457 labelled posts after one resubmit (the 5
  left are context-free quote-tweets); Haiku topic labels on 21,885 posts
  (X and Trump's feed, complete as of 2026-09-30 except 458 link shares).
- **Holes from the ~3,200-tweet timeline horizon** (unrecoverable): from
  May 10 up to Jun 20 for @marklevinshow, Jul 4 for @LauraLoomer, Jul 17 for
  @WhiteHouse and Jul 31 for @RealAlexJones. The other 13 accounts are
  continuous.
- **Keyword searches**: frozen at May 12, with a gap from about Apr 20 to
  May 5 (7-day recent search only).
- **Truth Social**: Trump's feed current through 2026-09-30 (anonymous
  refresh), 4,360 posts from Apr 4 (contiguous), scored with
  `score_opus_distilled`. 98,668 replies to 8 tracked posts
  (`data/raw/truthsocial/replies_*.jsonl`; `these_fools` 4,441 and
  `trump_strait` 11,173 added 2026-09-30, 94.7% and 99.1% coverage), scored
  with RoBERTa, `score_opus_distilled` and `score_mixed_distilled`, plus a
  1,292-row Haiku stance sample (1 error to retry) and 1,225 Opus-labelled
  replies. `score_opus_distilled` was rescored 2026-09-30 at its 256-token
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
  synced 2026-09-30 after the reply refresh and the GPU pull.
- Host backup (restic, homelab-infra): the SanDisk drive is unplugged and no
  host backup has run since 2026-04-23. The host backup script no longer
  needs Timeshift, so the drive can go back in; until then S3 is the only
  off-box copy of `data/`.
- Batch state files: `data/processed/relabel_batch.json` (`relabel
  status|collect|merge|resubmit`) and `data/processed/topic_batch.json`. The
  Batch API keeps results for 29 days.
- `main` is pushed.

## Open

- dkweb branch `iran-war-stance` (draft post, 7 Observable Plot charts, the
  stance colour tokens) is not merged.
- **Next X pull: Oct 25-30, 2026, no later than Nov 1.** On Sep 18
  @RealAlexJones's ~3,200-tweet timeline reached back only ~7 weeks, so his
  gap starts opening around Nov 1-6 (@WhiteHouse ~Nov 20, @LauraLoomer early
  Dec). One pull then covers ~6 weeks for about what a pull now would cost,
  because the heavy accounts hit their caps either way. Before it: size
  `ACCOUNT_CAP_OVERRIDES` for the gap so the sampled accounts (the five heavy
  ones plus @WhiteHouse and @VaticanNews) keep the May-Sep density of ~45-50
  per two-week slice, `collect --estimate`, ask the user for the X credit
  balance, run with `--no-search`. Rough cost at that density: ~2,100 reads
  (~$10) plus ~$4 for the Opus relabel and topic labels; at today's 450/500
  caps it would be ~$20-23 plus ~$7. After it: add timeline events
  from Sep 16 on, move `ANALYSIS_END` and extend `PHASES` (which end Sep 18,
  so Trump's feed after that is collected but outside every phase).
- The mixed model beat the scorer of record on the two unseen posts (0.658
  vs 0.604 Pearson against Opus, gap CI [-0.01, +0.12]). Recheck when the
  next tracked posts get Opus labels.
- 458 posts are link shares Haiku will not judge (a bare URL, or "Amen" /
  "Right on" over a link: it answers that it cannot open URLs and hits
  `max_tokens`); 183 are Trump's, 172 Levin's. `war_flag` falls back to the
  keyword pattern for them by design, but `topic-label` resubmits them on
  every run (~$0.07). Worth recording them as unjudgeable.
- 5 duplicate reply ids in the April reply files (3 `civilisation_dies`,
  2 `power_plant_day`); too few to move a number.
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).
