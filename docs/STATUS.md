# Status

As of 2026-09-30. Update the date and the lines that change after each refresh.

## Data

Two layers at different depths. Nothing runs on a schedule.

- **X broadcasters**: 19,605 posts from 17 accounts, Feb 1 -> Sep 18, 2026.
  The May 10 -> Sep 18 refresh read 5,631 tweets for ~$28, with the five
  heaviest accounts (Loomer, Levin, Jones, Ravid, StateDept) sampled at 450.
  `score_opus` on 19,452 of 19,457 labelled posts after one resubmit (the 5
  left are context-free quote-tweets); Haiku topic labels on 22,197 posts.
- **Holes from the ~3,200-tweet timeline horizon** (unrecoverable): from
  May 10 up to Jun 20 for @marklevinshow, Jul 4 for @LauraLoomer, Jul 17 for
  @WhiteHouse and Jul 31 for @RealAlexJones. The other 13 accounts are
  continuous.
- **Keyword searches**: frozen at May 12, with a gap from about Apr 20 to
  May 5 (7-day recent search only).
- **Truth Social**: Trump's feed current through 2026-09-30 (anonymous
  refresh), 4,360 posts from Apr 4 (contiguous), scored with
  `score_opus_distilled` and `score_opus_distilled_256`. 98,668 replies to 8
  tracked posts (`data/raw/truthsocial/replies_*.jsonl`; `these_fools` 4,441
  and `trump_strait` 11,173 added 2026-09-30, 94.7% and 99.1% coverage),
  scored with RoBERTa, `score_opus_distilled`, `score_opus_distilled_256` and
  `score_mixed_distilled`, plus a 1,292-row Haiku stance sample (1 error to
  retry) and 1,225 Opus-labelled replies.
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
- Heavy X accounts were last refreshed 2026-09-18: refresh them within about
  two months or the horizon opens new holes. Ask the user for the X credit
  balance before the pull.
- **The reply column of record mixes two models.** The 83,054 older replies'
  `score_opus_distilled` came from the 128-token `distill-opus` fit, which the
  `sweep-opus` final fit (`deb-256-1e5-3`) overwrote in
  `data/models/stance_distilled_final_score_opus/` on 2026-09-19 05:10 UTC;
  the feed and the 15,614 new replies were scored by the current model. On
  replies too short to truncate the old column correlates 0.915 with the
  current model, and every current-model run reproduces exactly. Both are
  about as good against Opus (0.713 vs 0.723 on 959 labels). Proposed:
  `score_opus_distilled_256` (current model at its training length, all
  98,668 replies) becomes the column of record; corrected reply shares move
  a few points, within their CIs. Waiting on the user; `export-web` and the
  README reply numbers wait on it too.
- Scoring at 256 vs 128 tokens alone moves nothing that matters: Trump's
  phase means by <= 0.01, reply agreement with Opus 0.692 -> 0.701.
- The mixed model on the two unseen posts: 0.658 vs 0.604 Pearson against
  Opus, gap CI [-0.01, +0.12]. A lean, not grounds to reverse 2026-09-19.
- Haiku topic labels are missing for the 273 feed posts from Sep 16 on; the
  ~58 of them inside the last phase use the keyword flag until the next
  `topic-label` run.
- 5 duplicate reply ids in the April reply files (3 `civilisation_dies`,
  2 `power_plant_day`); too few to move a number.
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).
