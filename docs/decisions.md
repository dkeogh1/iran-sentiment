# Decisions

Newest first: what was decided, why, and what was rejected. The numbers
behind the stance-model entries are in README *Stance-model experiments* and
[k8s/README.md](../k8s/README.md).

## 2026-11-02: The X holes are filled from search/all on collect's 35-hour grid; relabel gets caps

- **Decision:** `x-archive fill` (src/collectors/x_archive_fill.py) fills
  the X cache's holes from full-archive search: the late starts, the
  timeline-horizon holes (@WhiteHouse nearly every month, @LauraLoomer's
  February, March and June) and the collect windows that came back short
  (the 57 `FETCH SHORTFALL` windows among them). It is built and tested
  against a stub; the run is a paid run that waits for the user's approval
  of its estimate. relabel gets `RELABEL_MAX_CALLS` / `RELABEL_MAX_USD`,
  because the filled posts go through it.
- **Why search/all:** the timeline reaches back only ~3,200 posts and fails
  for @WhiteHouse, and the cursor has moved past the empty windows. The
  2026-11-02 probe found that search/all works on this app, returns
  retweets, and that counts/all counts them too. It bills $0.005 a post
  returned and nothing a request, so many small requests cost wall time,
  not money.
- **Why collect's grid and rule:** the filled stretches then follow the
  current design (2026-10-31): 35 h windows whose ends step through every
  clock hour and weekday, the newest 5 a window for a sampled account, every
  post for a take-all one. Each account's collect windows are reused as they
  are, and the grid steps back from the first one in 35 h steps to Feb 1,
  so every edge keeps its minute and second. A head under half a window
  merges into the first backward window, mirroring collect's tail rule.
  Nothing is added forward while the manifests end within half a window of
  the end of `ANALYSIS_END` (48 minutes today). A later collect run owns
  its whole span: an earlier run's windows inside it are dropped and the
  ones across its edges cut there, so a `collect --force` with longer
  windows replaces the incremental grid where it ran.
- **How it spends:** a window is picked when X's hour count, with the edge
  buckets counted whole, says it lacks a post. Its budget is that need plus
  9, so a window that still wants posts can always send one more request
  of search/all's 10-post minimum. Every fill run together stops at
  `X_ARCHIVE_FILL_MAX_USD`, tallied from a request log that records each
  request before it is sent (one answered but never logged counts at its
  `max_results`), and every request's `max_results` stays inside both
  limits. Held posts bound each request by
  id, so none is bought again; every request also carries `start_time` and
  `end_time`, because X uses `since_id` over `start_time` and, with no
  `start_time`, search/all reads only the last 30 days. A gap is skipped
  when every hour bucket it touches is held whole, checked when the walk
  reaches it rather than once before the first request. A sampled window's walk
  narrows each request with the hour counts and reads the crossing hour in
  15-minute slices; counts never end a walk early.
- **Estimate** (no call, 2026-11-02): 1,162 windows (259 late starts, 709
  empty, 194 short), 4,812 new posts, 5,070 reads ≈ $25.35 expected,
  6,111 ≈ $30.56 if posts bunch inside hours, ceiling $76.78;
  `--scope empty`: 968 windows, ≈ $24.29. The designer's scratch model put
  it at $25.60 / $24.33: it spread each edge bucket's count evenly, so it
  expected reads in edge buckets whose posts the cache already holds in the
  next window (about 46 reads, mostly the take-all accounts). The code
  tests each gap bucket by bucket, and narrows on what the window wants
  rather than on the expected count, which also needs fewer requests
  (2,671 against 2,779). The pilot (`--only WhiteHouse,PeteHegseth --since
  2026-08-01 --until 2026-08-08`: 8 windows, ≈ $0.18, ceiling $0.54) fits
  the shipped $1 cap. It checks that search/all serves @WhiteHouse, that
  the id bounds exclude held posts, and how billing counts.
- **Review fixes (2026-11-03, before any run):**
  - a request X answered but the run never logged (a kill, a power-off or
    a connection dropped mid-answer) was billed but missing from the
    tally, and the next run bought its range again. Each request now
    writes a pending line first;
  - with every gap's open flag fixed before the walk, a take-all window
    that found its last post could still try a gap whose bucket that post
    had closed, and stop on its budget. In a simulation of the full fill
    17-19 of 1,162 complete windows ended that way, marked truncated and
    planned again; none overspent;
  - after a `collect --force` (67 h windows for @WhiteHouse and
    @PeteHegseth, 77 h for @LauraLoomer), or an incremental run whose
    cursor sits before the earlier run's trailing empty windows, the old
    grid interleaved the two runs' windows into pieces as short as an hour,
    each with the full cap of 5: about 7.5 caps per 35 h against collect's
    5. Neither case is in today's manifest, so today's grid is unchanged;
  - an append after a line a kill cut was glued onto it, and both records
    were lost. `x_collector._append_jsonl` now ends the cut line first,
    which also covers collect.
- **relabel caps:** each submit and resubmit is checked against the
  estimate's ceiling, before any client is built: input from the llm_prompt
  usage fit, output at its p95 (113 tokens), batch prices, no cache.
  6,000 calls and $12 cover the fill's ~4,700 posts with text (≈ $7.6
  expected, ≈ $9.1 ceiling) with ~25% room. A larger set goes in `--limit`
  parts, each one passing the caps.
- **Rejected:**
  - one request over the whole window: it re-buys held posts and reads
    whole windows for 5 posts (≈ $49);
  - storing a page's extra posts in the cache: a sampled window would hold
    more than collect ever keeps. They go to a side file and are used
    first if a later walk wants them;
  - gating each run on the ceiling, which is 3x the expected spend: the
    run maximum is min(ceiling, the cap left);
  - the probe's 1.1 s pacing: search/all allows 300 requests per 15
    minutes, so a request every 3.1 s;
  - re-ranking windows that still hold posts from the old 14-day slices:
    the fill only adds the newest posts a window lacks, and never thins a
    window it paid for;
  - a `--count-slack` option for undercounted windows, which waits for the
    pilot.
- **Open (the user's calls):** the filled posts sit inside Feb 1 - Nov 1,
  so folding them into the published figures moves every one of them. And
  after the fill a sampled account is dense where Feb-Apr was read take-all
  and at 5 per window elsewhere: `phases` weights every post equally.

## 2026-11-02: Definitions of record after the Claude credit jobs

- **Decision (the user's calls, 2026-11-02, after the findings check on the
  seven collected label jobs):**
  1. **The war-post flag of record is Opus's topic class**
     (`settings.TOPIC_SOURCE = "opus"`): a post is a war post when its
     current `post-label` stance-topic label (claude-opus-5,
     post-stance-topic-2026-11-02) says `iran_war`, whichever set labelled
     it: the war candidates, the 1,500-post random arm (which now joins every
     series: same prompt, model and text check) or the census. A post with no
     label made from its text now counts by the old either flag, with a
     warning that counts them; no label file at all is an error, not a
     fallback. Every `inference.war_flag` reader takes it: `phases`,
     `visualize`, `export-web` (weekly series, accounts, the shift-share
     decomposition, Trump's phases), the Trump check's report and the
     `x-backfill-text` war counts; `--topic either|llm|keyword` stay for
     comparison. `summary` and `event-study`'s broadcaster windows read no
     topic flag (all-post means) and are unchanged. The stance of record on
     X stays score_opus.
  2. **A census labels every other post** so the flag rests on an Opus
     label everywhere: `post-label estimate|submit|status|collect --job
     stance-topic-census` sends stance-topic's prompt and model, into its
     label file and columns, to every post with text (X accounts, search
     rows out, and Trump's feed) without a current stance-topic label. It
     has its own batches and caps (`POST_CENSUS_MAX_CALLS` 17,000,
     `POST_CENSUS_MAX_USD` $28); its output is measured by stance-topic's
     50-call pilot and 6,375-call batch pooled (67.6 tokens a call, p95 90,
     input x 0.941), so it takes no pilot. Approved at about $22 of credit.
     The no-call estimate: 16,047 calls (13,271 X, 2,776 Trump), $24.53
     expected at batch prices with 89% cache reads (~$21.7 at the
     stance-topic batch's own rate, which read the cache on every request;
     ~$34 at the no-handle batch's ~59%), ceiling $56.95 with no cache hits.
     The dollar cap gates a submit on billed spend so far plus that
     submit's ceiling, so a cap at the ceiling ($60) would pass one submit
     of the whole census at any cache rate. It is held at $28 instead, ~25%
     over the approval, and the census goes in `--limit` parts collected
     between them: a run that bills high is refused before it passes the
     approval, and goes back to the user. Jobs that share a label file refuse to send while the
     other's batch is open.
  3. **Trump's stance of record is score_opus**
     (`settings.TRUMP_STANCE_SCORE_COL`): Opus 5 on llm_prompt, the X tiers'
     model and prompt, from `teacher-check --source trump --merge` (3,221 of
     his 3,222 posts with text). `phases --source trump` and export-web's
     trump_phases default to it; a post without it is left out with a
     count, never imputed. score_opus_distilled stays available (`--score`).
  4. **The reply figures are the pooled draws:** the first and fresh random
     draws together (1,071 + 8,000 = 9,071 labelled), as
     `REPLY_POPULATION_FRESH_DRAWS` already had it. export-web logs the
     basis and puts each post's labelled draws (`labelled`,
     `labelled_fresh`) in replies.json; it warns only when it exports
     without the fresh draws.
  5. **The praise rule stays:** v2 (praise for a deal, ceasefire or pause
     counts against the war) is the reply label set of record; v2b, the same
     prompt without the rule, is reported beside it (`reply-label report
     --job v2b`, `reply-population --labels v2b --first-draws`).
  6. **The loyalty gradient is a command:** `reply-loyalty` (free;
     inference.reply_loyalty) gives the Opus stance of low-, medium- and
     high-loyalty repliers and high minus low per tracked post, over every
     post pooled, and within posts (each post's gap averaged by post size),
     on the v2 pooled draws, design-weighted with reply-population's strata,
     resampling and finite-population correction. A replier's tier takes the
     account's age on `settings.LOYALTY_REFERENCE_DATE` (2026-11-01,
     ANALYSIS_END), not the run date, so tiers no longer drift;
     event-study's segment_by_loyalty keeps the run date. On the 2026-11-02
     data: 60,852 repliers (27,827 low, 29,768 medium, 3,330 high; a tier
     is per reply, from that reply's account snapshot, so the 73 repliers
     whose bio or account differs between replies count in two tiers and
     the tiers add to 60,925), high minus low +0.145 [+0.10, +0.20] pooled,
     +0.125 [+0.07, +0.18] within posts, reversed on the deal post (-0.14).
- **Why:** the either flag took in ~1,300 X posts Opus classes off the Iran
  war (keyword-only candidates are about a third Iran-war) and missed about
  29% [26%, 32%] of the Iran-war X posts (134 of the 1,500 random unflagged
  posts); Opus gives off-topic posts a stance from the author, so both errors
  move tier means. Trump's whole feed now has Opus labels from the X tiers'
  model and prompt, and with the analysis frozen the distilled model saves
  nothing; it also reads the collapse phase 0.07 [0.02, 0.12] too low. The
  fresh draws are a random sample of the same design as the first ones,
  and pooled they narrow the reply intervals (the loyalty gap's from
  [-0.06, +0.22] on the first draws to [+0.10, +0.20]).
- **Headline move, X by tier (data copy, before the census):** the
  administration's strike-to-midterm war-post change goes from -0.200
  [-0.30, -0.10] (either) to -0.131 [-0.27, -0.00] (opus), and pro-war MAGA
  minus the administration in the midterm hold from +0.146 [+0.01, +0.27]
  to +0.204 [+0.02, +0.39].
- **Rejected:** keeping the either flag (its errors are measured now, and
  they fall on the late phases where war posts are few); the Opus flag on
  the candidates only, with the random arm kept out (it leaves the ~29%
  missed posts out of every series; the census closes that); score_opus_distilled
  for Trump (a second scorer in the published series, and biased by phase);
  the first draws only for the replies (the fresh draws are the same
  design, 7.5 times as many); dropping the praise rule (a reply that cheers
  a deal does oppose escalation; v2b shows how much the rule moves).

## 2026-11-02: One alternate post-rater path; Trump's whole feed on score_opus; unjudgeable topic posts

- **Decision:** labels on broadcaster posts beyond score_opus go through one
  path, `post-label estimate|pilot|submit|status|collect|report --job <job>`
  (src/analysis/post_labels.py), whose parameters are the model, the prompt
  version and the post set, on the batch transport the teacher checks use.
  Each job has its own label file, state file, caps and report, and its
  columns carry the model's tag; none is ever merged into score_opus or
  sentiment_all.parquet. `phases --score <col>` joins a job's column from
  its label file onto the frame's war candidates (keyword OR Haiku) when the
  post's text is the one labelled, and `phases --topic opus` counts a
  candidate as a war post when its stance-topic label says `iran_war`.
  - `stance-topic` (claude-opus-5): a new prompt that classes the post
    (iran_war / war_or_peace_general / other_conflict / not_war) and scores
    the author's position on U.S. military action against Iran, null off
    the Iran war, with the handle where llm_prompt has it (just before the
    post). The topic definitions are topic_prompt's, the scale and "position,
    not tone" v2's; it adds that a repost counts as the author's post. The
    instructions come first, ~710 tokens, so they go as a cached prefix
    (Opus 5's minimum is 512). Columns `score_opus_stance`, `topic_opus`. Set:
    the 4,925 war candidates (X accounts 4,479, Trump 446) plus 1,500 X posts
    no filter flags (the smallest seeded hashes of their ids, frozen by the
    first pilot or submit), which only measure what the filters miss.
  - `rater` (claude-opus-5-5, effort low): llm_prompt, the score_opus prompt
    byte for byte, on the 4,925 candidates. Column `score_opus55`; the job
    refuses claude-opus-5 and Haiku (their `score_<tag>` are existing
    series). Its reply arm is `reply-label --job v2-rater`: the v2 prompt
    unchanged on the 1,157 v2 replies, in its own file, set against the v2
    labels by its report.
  - `stance-topic-nohandle`: stance-topic without the sentence naming the
    author, on a seeded paired subsample of the X candidates: all 272
    religious and 86 opposition posts, 400 each of admin, pro-war MAGA,
    anti-war MAGA and media (1,958). Its report pairs it with stance-topic's
    labels by tier; its columns are no series.
  - Reports (aggregates only): Pearson, Spearman, MAE, sign flips and
    Bland-Altman (mean and SD of new - score_opus, 95% limits) by tier and
    phase; the topic class against which filter flagged each candidate; the
    random arm's Iran-war share with a Wilson interval and what it implies
    for the unflagged pool; the no-handle shift by tier with its interval.
- **Trump's whole feed:** `TRUMP_TEACHER_CHECK_MAX_CALLS` raised from 500 to
  3,300 so `teacher-check --source trump --all` can label every post with
  text (3,222; the 400 labelled already are reused, 2,822 to do).
  `--merge` writes score_opus / label_opus onto the feed parquet by id
  (score_opus_distilled stays), claimed in the feed's own column record;
  `phases --source trump --score score_opus` then works. The Trump default
  in `phases` stays score_opus_distilled: switching it is the user's call.
  Not merged yet: score-posts may still be writing that parquet.
- **Pilots and caps:** `post-label pilot` (direct, 50 calls by default)
  sends the batch's own request and records the billed input (with cache
  writes and reads) and output tokens and each answer's stop reason; the
  estimate is then calibrated on it (input scale, output mean, p95 for the
  ceiling, whether the cache engaged). Only a pilot of the job's current
  prompt version, max_tokens and effort counts. Without one, batches of a
  model and prompt send at most `POST_PILOT_MAX_CALLS` (100) calls in all,
  except llm_prompt on claude-opus-5, whose output the usage fit measures;
  and once such a pilot has a max_tokens stop no batch goes, `--limit` or
  not (Opus 5.5 always thinks, and thinking counts against max_tokens).
  Each job's caps bound its whole run: every pilot (direct prices) and
  batch (billed, or its ceiling until collected), on any model, plus the
  next batch's no-call ceiling (no cache hits, output at the pilot's p95,
  twice the guess before a pilot). Batch collects now record output tokens
  a call (mean and p95) and stop reasons for every check.
- **Estimates (no call, 2026-11-02, batch prices):** Trump's feed $5.10 on
  the check batches' usage fit (0.3628 tokens a character + 35, 77 output;
  ceiling $6.06), against the old characters / 2 figure of $5.99.
  stance-topic $13.82 (6,425 calls, output guessed at 110, 89% cache reads;
  pre-pilot ceiling $34.39 over the $30 cap until a pilot replaces the
  guess), rater on Opus 5.5 $10.75 (4,925 calls, output guessed at twice
  Opus 5's 84; ceiling $18.96, cap $20), no-handle $4.17 (1,958 calls;
  ceiling $10.44 before a pilot, cap $9.50), the reply arm $1.69 (ceiling
  $3.14 at twice Opus 5's 67.8 output tokens while unmeasured, cap $4).
  Each 50-call pilot is about $0.21-0.23 direct (ceilings $0.39-0.54); the
  reply arm's 50-call batch pilot $0.07. The caps hold the run's ceiling up to a pilot p95 of
  ~160 output tokens on the stance-topic jobs and ~350 on the rater.
- **Topic-label failures:** each answer Haiku gives with no verdict (a
  link share: it starts explaining it can't open the link and runs out of
  its 32 tokens) is recorded in `topic_unjudgeable.parquet` with its batch
  and the length of the text it was sent (a submit records each length), and
  a post with three such answers on the same text
  (`TOPIC_UNJUDGEABLE_MIN_ATTEMPTS`) is not sent again
  (`--retry-unjudgeable` overrides; `topic-label unjudgeable
  --results-file` records a past batch's for free). It keeps no Haiku
  label, so `war_flag` counts it by the keyword pattern as before; errored
  or expired requests are retried, not recorded.
- **Review fixes (same day, before any spend):**
  - The post caps bounded each batch, so `submit --limit` batches (which
    the refusal message suggested) could together pass a dollar cap, and
    repeated `--limit 100` batches could skip the pilot: the caps now bound
    the run as v2-fresh's do (pilots included), the unmeasured gate counts
    the batches already sent, and no message offers `--limit` as a way
    round a cap. The same holds for the reply arm (`v2-rater`).
  - A max_tokens stop in the pilot now refuses every batch (`--limit`
    used to switch the check off), and a pilot of another prompt version or
    effort no longer counts as a measurement.
  - The reply ceiling priced output at the mean even unmeasured: it is twice
    Opus 5's length until measured, and a pilot-measured model is recorded
    with its p95 (`{"mean", "p95"}`), which the ceiling takes; Opus 5's
    1,141-call mean stays as both (over thousands of calls the mean bounds
    the total). v2-fresh without `--post` is only a pilot-sized `--limit`.
    A reply submit refuses once a batch of the job on that model had an
    answer cut at max_tokens (`REPLY_LABEL_MAX_TOKENS`).
  - Unjudgeable after one no-verdict was too soon: in the four topic batches
    to 2026-11-02, 30% of the posts got a verdict on the next try after one
    no-verdict, 25% after two, 20% after three or more (389 posts are
    unlabelled with a no-verdict: 214 with one, 102 with two, 73 with three
    or more). A resend costs ~$0.0001, so a post waits for three, counted
    per batch or file on the same text, and an answer is put down to the
    text the submit sent, never one changed since.
  - A label made from an older text counted as done, so the post was never
    labelled again while every reader dropped the label: it is now to do,
    and the new label replaces it (the old one to `*_superseded`); the
    text-change paths (x-backfill-text, `collect --force`, ts-fill-text)
    now move the post-label caches' labels aside as they do relabel's.
  - `column_owner` claimed score_opus for the rater on claude-opus-5 (a
    pair the job refuses), which sent the reader of a missing score_opus to
    a refused command: refused pairs own no column. `phases` on a
    post-label column (war candidates only) printed all-post, share and
    share-effect figures that were really candidate figures: they are NaN
    and only the war-post columns print.
  - The no-handle report's ALL row pooled a subsample that takes two tiers
    whole and 400 of each larger one: it now weights each tier's pairs by
    its pool over its pairs (the interval: see the second look below).
  - score-posts does keep score_opus (the doc said it drops it), but it
    reads the feed when it starts and writes it back whole when it ends, so
    a `--merge` meanwhile would be lost: it now writes the file atomically
    with a `<feed>.scoring` marker, and the merge refuses while the marker
    names a live process (on dkbl1: the GPU Job's run is on dkbl2, so a
    merge goes after its `sync-data.sh pull`, AGENTS.md).
- **Second look at the fixes (same day, before any spend):**
  - Every batch create (the checks, post-label, reply-label, relabel,
    topic-label) is sent once: the SDK retried a timed-out or dropped
    create with no idempotency key, so a batch the server had made would be
    made again and only the second recorded (billed outside the run caps,
    and a reply batch never deleted). A failure without an answer keeps a
    "submitting" state (relabel and topic-label now write one too) that
    every submit refuses on; a 4xx or 529 answer, which comes before any
    batch exists, puts the last state back.
  - relabel refuses any model but claude-opus-5 (the path rejected below
    still ran end to end, uncapped, at Opus 5's output length).
  - A pilot counts as the measurement only with `POST_PILOT_MIN_BILLED`
    (40) billed answers pooled: a `--n 1` pilot set the ceiling the dollar
    cap is checked against at one answer's length and opened the unmeasured
    gate. A smaller one is shown but stays a guess; its max_tokens stops
    still refuse.
  - A max_tokens stop in a collected post-label batch of the current
    request refuses the next submit, as the reply path already did (cut
    answers are not kept, so they would go again and be cut again).
  - A pilot is on record before its first call (in_progress, at its direct
    ceiling) and saves its labels and usage every `POST_PILOT_SAVE_EVERY`
    answers: a pilot killed outright (dkbl1 powers off) lost both, so its
    posts were paid again and its spend left the caps.
  - The collect of a reply batch syncs the disk before the delete, which
    cannot be undone (a new label file or results archive could still be
    in memory when the results left the API). The unjudgeable record, the
    column records and the relabel and topic-label caches are written
    temp-then-rename.
  - The no-handle ALL interval is the linearized one of its weighted ratio
    (e = 1{both iran_war} (shift - the weighted shift), stratified by
    tier), which counts that the posts both arms call iran_war are a random
    subset: the earlier "stratified" formula left that term out. Each
    tier's pairs are a sample of that tier's, with no finite-population
    correction, as in the per-tier rows (Opus's answers are draws too).
  - `phases --score score_opus_stance` says that its war posts are the
    `--topic` flag AND Opus's iran_war class (the score is null off it)
    and names the like-for-like baseline, `phases --score score_opus
    --topic opus`; export-web warns when replies.json pools the fresh
    draws, since the published prose cites the 1,071-draw figures.
- **Why:** every published stance rests on one model and one prompt that
  asks for "sentiment" and names the author; the Pope's -0.77 and the
  administration's keyword-only war posts rest on the loose filter; Trump's
  feed is the one published stance series not on score_opus. Each paid run
  still gets its own estimate and the user's go.
- **Rejected:** `relabel --model claude-opus-5-5` (no cap, a model-blind
  estimate, and before the tag fix it merged into score_opus); writing the
  new columns into sentiment_all.parquet (`analyze` rewrites it, and a read-
  time join cannot put a random-arm label into a series); Fable 5.1 as the
  second rater (the user chose Opus 5.5); the no-handle arm on all
  candidates (paired differences have little variance, so ~1,900 measure a
  tier's shift to about +-0.02); a prompt for stance-topic with the post
  first (its instructions could not be cached).

## 2026-11-02: One reply-label path; per-model prices and tags; batches deleted after collect

- **Decision:** further Opus reply labels go through one path, `reply-label
  estimate|submit|status|collect|report --job <job>`, whose parameters are
  the prompt version, the model and the id source, on the v2 run's prompt
  code, id-keyed label cache and batch transport. Three jobs, each with its
  own label file, state file, cap and report, none writing the v1 or v2
  files:
  - `v2b`: the v2 prompt less the one sentence that makes praise of a
    strike a position for the war and praise of a ceasefire, a deal or a
    hold-off one against (both halves, so the test stays symmetric), on the
    1,157 stance-sample replies with text. `reply-population --labels v2b`
    weights it.
  - `cat-v2`: the six categories of `event_study.STANCE_PROMPT`, word for
    word, asked v2's way (the parent post as the cached prefix, the whole
    reply, no author handle), on the same 1,157 replies. The Haiku
    categories stay as history. Their population shares weight the first
    random draws only (not the flipper cohort), as `reply-population` does.
  - `v2-fresh`: the v2 prompt unchanged on a second random sample,
    `reply-fresh-sample`: 1,000 replies with text per tracked post, split
    across the valence buckets in proportion to the post's replies, drawn
    within a bucket from those not already first draws, as the smallest
    seeded hashes of their ids, and recorded in its own manifest before any
    is sent. Same scorer as the v2 labels, so `reply-population` pools both
    files as the v2 label set, in its direct and its model-assisted columns
    (`REPLY_POPULATION_FRESH_DRAWS`; off reproduces the 1,071-draw figures,
    and which estimand is published stays the user's call); within a post
    and bucket the first and fresh draws together are one simple random
    sample, so the bucket weighting is unchanged. A drawn reply that already
    carries a v2 label keeps it. Sent one post per batch (`--post`,
    enforced; `--limit` for a pilot), under `REPLY_FRESH_MAX_CALLS` (8,500)
    and `REPLY_FRESH_MAX_USD` ($21.50), which bound the whole run: a submit
    counts the calls its earlier batches sent and the spend they were
    billed, and the draw refuses more than 1,000 a post or a manifest whose
    run would pass either cap.
- **Estimates:** calibrated on what the 2026-10-05 v2 batch was billed
  (1,141 requests): each post's prefix tokens, 0.350 tokens per reply
  character, 67.8 output tokens, a cache read on 89% of a post's requests
  where its cached block reaches the model's minimum (512 on Opus 5),
  another prompt's prefix moved by its extra instruction characters at 2.67
  a token. The ceiling has no cache hits: every request on a post whose
  block reaches the minimum writes it (1.25x), and only those requests
  carry a cache breakpoint, so the others can't pay a write the estimate
  leaves out. At Batch prices on claude-opus-5: v2b $2.11 (ceiling $2.66),
  cat-v2 $2.32 ($4.30, output at twice its 83-token guess while
  unmeasured), v2-fresh $14.90
  ($20.21) for 7,995 calls (5 drawn replies were labelled already); per
  post $1.32-2.29 (ceilings $2.02-3.40).
- **Shared fixes:** a per-model price table (`settings.MODEL_PRICES`: input,
  output, cache write and read multipliers, batch discount, cache minimum)
  prices every estimate and spend tally; a model with no row is refused.
  Column tags no longer collide (`claude-opus-5-5` was tagged `opus`, so a
  `relabel merge` of it would have replaced score_opus): Opus 5.5 is
  `opus55`, Fable 5.1 `fable51`, and `relabel merge` refuses a column
  `sentiment_all_columns.json` records as another model's. A collect whose
  results stream breaks downloads `results_url` itself, and a collect of a
  batch that carried replies deletes it on the API (`--keep-batch` to keep
  it), so private replies are not kept there for the 29 days results
  otherwise stay.
- **Why:** the June deal's anti-war reading rests on an instruction the
  prompt states outright; the Haiku categories never saw the parent post and
  named the author; the published reply intervals rest on 1,071 labelled
  draws. Each paid run still gets its own estimate and the user's go.
- **Review fixes (same day, before any spend):** the fresh caps were per
  batch, so eight per-post submits could each pass them, and `--n` had no
  bound: both now bound the run (above). The ceiling priced the cached
  block at 1x, under the 1.25x a missed cache costs: corrected, which put
  the run's ceiling over the first $20 cap, now $21.50 (the same ~6% slack
  as the call cap). A model whose output length is unmeasured (Opus 5.5:
  thinking may run longer) sends at most a 50-call pilot under a dollar cap
  until its figure is in `REPLY_OUTPUT_TOKENS`. A failed, pending or kept
  delete is retried by the next submit before it archives the state (it
  refuses while the delete fails); a 404 counts as deleted. The results
  download asks for `accept-encoding: identity` and never resumes an
  encoded body, so a Range offset always counts the bytes held.
  `reply-population` pools a post's fresh draws only once 98% of them are
  labelled (before its batch, only reused flipper labels sit there), and
  label sets are compared on the same draws: `--first-draws`, and the v2b
  report's paired v2 - v2b population difference
  (`inference.reply_population_paired`). The bootstrap gained the
  finite-population correction (the fresh draws sample up to a third of a
  bucket; the first draws sample 0.4-8% of theirs, so their intervals
  narrow by at most ~4%, mostly under 1%). The fresh report
  dropped its unweighted ALL row. cat-v2 keeps the six definitions word for
  word; "opposes THIS action" can read as opposing the de-escalation on the
  ceasefire posts, so its report counts, on those posts, the categories
  filed anti-war that v2 scored pro, to read on a 50-call pilot first.
- **Rejected:** writing the fresh labels into the v2 file (it would change
  the 1,157-reply v2 report and couple two batch checks to one parquet);
  excluding every labelled reply from the fresh frame (the flipper cohort
  could then never be drawn, a small bias the clean design avoids);
  `relabel --model` for a second rater (it has no cap and, before this,
  merged into score_opus); a direct-API path for the new jobs (Batch only,
  `--limit` for a pilot).

## 2026-11-01: X's own counts before any full-archive fill

- **Decision:** before any archive fill is bought for the X holes, fetch
  X's own post counts (`x-counts`: counts/all, $0.010 a request, 31 days a
  request at any granularity, so about $1.53 for every account over
  Feb-Nov in hour buckets) and run a tiny full-archive probe (`x-archive
  probe`, at most $0.12).
- **Why:** the 2026-11-01 collect logged ~22 windows that returned 0 posts
  next to dense truncated ones ("possible fetch failure"), and @WhiteHouse
  kept 25 posts in 31 windows (281 cached for April, 15 since Oct 1). The
  timeline can't say whether those windows were silent or failed, and the
  cursor has moved past them. Per-hour counts can, window by window
  (`x-counts --report`), and the monthly table shows the late starts and
  the May-Jul holes. Counts are billed per request, not per post.
- **What the probe decides:** search/all is the only way back past the
  ~3,200-post timeline horizon, but one docs page still calls the archive
  endpoints Enterprise-only (a forum report shows a 403
  client-not-enrolled), and the search index may drop native retweets from
  from: queries, which we count as messaging (2026-10-04). The probe asks
  for two whole UTC days of @PeteHegseth the cache holds whole (February,
  take-all; the uncapped Sep 13 slice), retweets and originals both, one
  page each, compares id for id, and counts his posts per day with and
  without `is:retweet`. `no archive access` ends the archive idea;
  `retweets dropped` makes a fill originals-only, which changes what an
  account's messaging means; `archive works incl. retweets` lets a fill be
  planned at $0.005 a post; `misses older posts` limits it to recent
  months. The `is:retweet` count also says whether the report's verdicts
  compare like with like.
- **Design:** only a 403 is a result (saved for the probe); a 401 (bad
  token) or a usage-capped 429 stops the run unfinished, and no further
  request is made either way; any other 429 waits a bounded number of
  times. A stretch X answers short of its documented 31-day paging is
  recorded only as far as its buckets reach, so the cache never holds
  false zeros. The counts cache key ignores the case of `from:` handles.
  Both caches sit outside the `data/raw/x/*.jsonl` glob; both commands
  take the collect lock, and their estimates make no call.
- **Rejected:** buying a fill on the strength of the docs, which contradict
  each other on access and say nothing reliable on retweets;
  `counts/recent` (7 days only); minute buckets (same price, 60 times the
  rows; the report gives a `lo-hi` range where hour buckets cut a window's
  edges).

## 2026-10-31: X gaps are walked in 35-hour windows; heavy accounts are sampled per window

- **Decision:** `collect` walks each account's gap oldest first, from its
  cursor, in fixed 35-hour windows (`GAP_FILL_WINDOW_HOURS`). The 11
  accounts in `ACCOUNT_WINDOW_CAPS` keep the newest <= 5 posts of every
  window, with no rollover: 48 per 14 days, the May-Sep density. Every
  other account is take-all: it reads every post, window after window, up
  to its backstop (`ACCOUNT_CAP_OVERRIDES` / `MAX_TWEETS_PER_USER`), and
  defers the windows it can't afford to the next run. No request asks for
  fewer than 5 posts (X's minimum, billed anyway); a window stops after 20
  pages; in an incremental run or a first pull each window's posts are
  appended as soon as it is read (a `--force` merge is written at the end,
  or with what it read when it fails); each window gets a row in
  `data/raw/x_sampling/` once its posts are stored; the plan maximum counts
  the $0.010 user lookup per account and is strict (a search stops once
  fewer than its 10-post minimum are left); `collect --only` narrows a run.
- **Why:** the 14-day split (2026-09-16) gave every slice an equal share of
  the cap, and the endpoint is newest-first, so a heavy account's capped
  slice kept only the newest ~0.9-1.6 days of each fortnight. The slices
  ended on the same weekday and hour, so those blocks were 72-78% Saturday
  and Sunday posts, and alternate weeks were empty. A 35-hour window
  (35 = 11 mod 24, coprime with 24) moves its end 11 hours later on the
  clock, and a day and 11 hours later in the week, each time: every 24 windows (exactly 5 weeks) end once at every clock hour
  and 3-4 times on every weekday. Whole-day lengths lock the weekday and
  hour; 36 h only alternates two hours. A per-window cap fixes the density
  at about 48 per 14 days over a gap of two weeks or more (within ~5%):
  a run's last window still gets 5 when it is only 17.5-35 h long, so a
  gap under 35 h is one window of 5, up to twice as dense, and sampled
  accounts should not be rerun within a few days. The planner lengthens
  the windows (to the next length coprime with 24) only when the caps
  would exceed the backstop. For a run at ~14:00 UTC on Nov 1 the plan is
  31 windows and at most 155 posts per sampled account (about 48.8 per 14
  days): 4,705 reads + 17 lookups, about $23.70. (At 00:00 UTC it is 30
  windows, 151 per sampled account, 4,661 reads, about $23.47.)
- **Forced runs on sampled accounts:** a forced plan covers Feb 1 to now
  at one density, so with a 450 backstop its windows grow to ~73 h. On a
  cache that stops half a window or more before now, that would sample
  the uncached tail at about half the incremental density and then move
  the cursor past it for good, so the planner skips the account until an
  incremental run has caught it up.
- **Window edges:** X's docs do not settle whether `end_time` is
  inclusive, so windows share their edge second (`[s, e]` then `[e, ...]`)
  and posts are deduped by id before they are stored. No second can be
  missed either way; at worst a post exactly on an edge second is bought
  twice. The take-all backstop counts every post returned, duplicates
  too, so the maximum stays strict.
- **Caveat:** phase-4 contrasts for the sampled accounts mix this design
  change with real change (their May-Sep blocks were weekend-heavy at fixed
  hours); check them against the take-all accounts before publishing.
- **Rejected:**
  - Rollover for sampled accounts (a quiet window's unused cap passed on).
    In both design-review replays it would give the mid-rate accounts
    (@POTUS, @StateDept, @RealCandaceO, @PeteHegseth, @VaticanNews) 7-15%
    more posts at the same war share and stance bias (within ~0.01), but it
    makes later windows denser (tilt ~1.0-1.3x), moves the share lost past
    the ~3,200-post horizon into later windows, and makes the inclusion
    rule depend on the run. min(n_w, 5) per window is a plain stratified
    sample, and the per-window manifest supports volume weights later.
  - 41-hour windows with rollover: the tail left unread, uneven weekdays,
    caps resized per run.
  - G/n slices with rolling shares: the length can land on 36 h.
  - Allocating reads by volume from the counts endpoint: paid; maybe later.
  - Repairing a torn last cache line after a crash: deferred. Appends now
    flush and fsync.

## 2026-10-07: The parent-post reply scorer stays an experiment

- **Decision:** `score_opus_distilled` stays the reply scorer of record. The
  reply scorer that reads each reply with its Trump post (`reply-distill`,
  variant `ctx`) is not cross-fitted or published.
- **Why:** leave-one-post-out against the v2 labels it beats the scorer of
  record where context decides stance (June deal 0.14 -> 0.62 Pearson,
  ceasefire 0.32 -> 0.70, hold-off 0.54 -> 0.76; pooled 0.58 -> 0.64,
  seeds 0.63-0.65) and loses some on the hawkish posts (power plant 0.72
  -> 0.58, strikes 0.77 -> 0.70). The published shares are unbiased either
  way; a better scorer only narrows `reply-population`'s intervals. On the
  same held-out predictions the mean intervals would narrow for three
  posts (deal about -35%, ceasefire -28%, hold-off -17%) and widen for
  three (+8% to +20%), and the pro / anti share intervals barely move.
  Not worth a published scorer whose per-post fits aren't kept.
- **Rejected:** cross-fitting and publishing `score_ctx_crossfit`;
  picking the scorer per post by which one did better (chosen on the
  same labels it would be judged by).

## 2026-10-06: A reply-trained scorer feeds `reply-population` only cross-fit

- **Decision:** the reply-domain scorer (`reply-distill`: the posts-only
  model fine-tuned on the v2 reply labels, the reply alone or with its
  parent post) is judged leave-one-post-out against the published scores,
  and reaches `reply-population` only as a cross-fit column: each post's
  replies are scored by a fit on the other seven posts' labels
  (`--crossfit`, `score_<variant>_crossfit`). `reply-population` refuses a
  column from a fit on all the labelled replies (`--fit-all`,
  `score-replies-ctx`); `reply_sentiment_columns.json` records which model
  wrote each such column.
- **Why:** the model-assisted estimate is the model's census plus the
  labelled sample's mean of Opus minus model. A model fit on that sample
  nearly reproduces its labels there, so the correction and its interval
  shrink while the model's error on the other replies goes uncorrected. A
  per-post cross-fit keeps census and correction from one model the
  post's sample is new to, so the estimate stays unbiased and a better
  scorer narrows the interval honestly.
- **Rejected:** weighting the full-sample fit's column; the full-sample
  fit for the census with out-of-fold predictions for the correction
  (mixes two models: biased by their difference).

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
- **Superseded** on 2026-10-31: 35-hour windows with per-window caps replace
  the 14-day slices and their equal shares.

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
