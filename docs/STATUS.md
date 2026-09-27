# Status

As of 2026-09-27. Update the date and the lines that change after each refresh.

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
- **Truth Social**: current through 2026-09-16. Trump's feed, 4,087 posts
  from Apr 4 (contiguous), scored with `score_opus_distilled`. 83,054 replies
  to 6 tracked posts (`data/raw/truthsocial/replies_*.jsonl`), scored with
  RoBERTa and `score_opus_distilled`, plus a 992-row Haiku stance sample and
  959 Opus-labelled replies.
- **Timeline**: 84 events through 2026-09-15 in `config/timeline.py`;
  `ANALYSIS_END` = 2026-09-18. Sources for the Sep 2026 additions are in
  `docs/timeline_candidates_2026-05-12_to_2026-09-15.json`.

## Where things are

- The k8s namespace `iran-sentiment` is deleted (its PVC with it). Models are
  in `data/models/` on dkbl1 and in the S3 backup.
- S3 backup first synced 2026-09-25: raw 72 MB and processed 50 MB in
  Standard, models 8.8 GB in Glacier IR, verified object for object.
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
- Reply candidates not yet tracked: the Jun 18 "These fools, who think I
  haven't been tough enough on Iran" post (4,702 replies, directly about the
  MAGA split) and the Sep 2 "TRUMP STRAIT" post (11,283).
- In `config/tracked_posts.py` but not collected: `armada` (Jan 28, before
  the window) and `epstein_hoax` (2025, a control).
