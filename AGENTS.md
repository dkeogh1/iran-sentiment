# iran-sentiment

Stance analysis of US political messaging on the 2026 Iran war: X broadcaster
accounts in political tiers, plus Trump's Truth Social feed and the replies to
eight of his posts, scored by Claude models. `README.md` is the public write-up;
where the data stands is in [docs/STATUS.md](docs/STATUS.md).

This repo is **public** on GitHub: no IP addresses, account IDs, bucket names,
credit balances or personal data in anything committed.

## Ask first

Hand these to the user, with the estimate, instead of running them:

- **Paid runs.** Estimate first, then ask.
  - `collect` and `run-all` (X API, $0.005 per tweet read): run
    `python -m src.cli collect --estimate` (cache only, no API calls) and ask
    for the current X credit balance. Same for `x-backfill-text` (one read
    per cached post cut at 280 characters; `--estimate`).
  - `x-counts` (X counts/all, $0.010 per request, 31 days per request:
    about $1.53 for every account over Feb-Nov at first, a range already
    cached is free) and `x-archive probe` (two search/all pages plus two
    counts/all requests, at most $0.12; a finished probe is reused free):
    both have `--estimate` (no API call, no credentials). `x-counts --report`
    is free.
  - `x-archive fill` (search/all, $0.005 a post returned, requests free):
    `x-archive fill --estimate` (no API call, no credentials; `--scope
    empty`, `--only`, `--since/--until`, `--limit` narrow it) prints the
    expected reads, the ceiling and the run maximum. Each window reads at
    most its need + `X_ARCHIVE_FILL_WINDOW_SLACK`; `X_ARCHIVE_FILL_MAX_USD`
    bounds every fill run together (tallied from
    `x_archive/fill_requests.jsonl`), and a run refuses past
    `X_RUN_BUDGET_USD`. The full fill ran on 2026-11-03 ($26.45, after a
    $0.19 pilot) with the cap at $32, and every window is done
    (`--estimate` plans none). Raise the cap only on the user's say. Ask for
    the X credit balance first; `backup` after.
  - `relabel submit|resubmit` and `topic-label submit` (Anthropic Batch API):
    `relabel estimate` / `topic-label estimate` print the cost. relabel
    labels with claude-opus-5 only; `RELABEL_MAX_CALLS` / `RELABEL_MAX_USD`
    bound each submit and resubmit, checked against the estimate's ceiling
    (input from the usage fit, output at its p95, batch prices); a larger
    set goes in `--limit` parts. `TOPIC_LABEL_MAX_CALLS` /
    `TOPIC_LABEL_MAX_USD` bound each topic-label submit the same way, before
    any client is built (ceiling: each prompt at 2.5 characters a token,
    every answer at its max_tokens). A second rater's model is `post-label
    --job rater`.
  - `analyze --llm`, `stance`, `teacher-check`, `reply-teacher-check` (direct
    Anthropic API): estimate from the post count (Haiku has run about
    $0.50-1.50 per 1,000 posts, README *Cost*; `teacher-check` ~$1 and
    `reply-teacher-check` ~$2 on Opus). Narrow `analyze --llm` with
    `--llm-tiers` / `--llm-accounts` to the subset that needs it.
  - `reply-teacher-check --v2` and `teacher-check --source trump`, direct or
    `--batch submit` (half price): `--estimate` prints both, calibrated on a
    measured pilot (~$0.0043 a call direct). `--batch status|collect` and
    `--report` are free.
  - `reply-label submit --job v2b|cat-v2|v2-fresh` (Batch API only):
    `reply-label estimate` prices it from the usage the 2026-10-05 v2 batch
    was billed (`settings.REPLY_V2_BATCH_USAGE`), at the model's own row of
    `settings.MODEL_PRICES`, and makes no call. Its ceiling (no cache hits:
    every request on a post whose prefix can cache writes it) is what the
    dollar caps are checked against. v2-fresh sends the fresh draws
    (`reply-fresh-sample`, free, records them first and refuses a run over
    the caps) one post at a time (`--post`, enforced; without it only a
    `--limit` pilot of at most `REPLY_UNMEASURED_PILOT_MAX_CALLS`);
    `REPLY_FRESH_MAX_CALLS` and `REPLY_FRESH_MAX_USD` bound the whole run,
    earlier batches included. Another `--model` writes its own label file;
    a job with a dollar cap sends a model whose output length is unmeasured
    no more than `REPLY_UNMEASURED_PILOT_MAX_CALLS` calls in all (its earlier
    batches of that prompt and model counted) until its tokens a call are in
    `REPLY_OUTPUT_TOKENS`, recorded as `{"mean", "p95"}` from the pilot's
    collect line (the ceiling takes the p95; unmeasured, twice Opus 5's
    length). A submit refuses once an earlier batch of the job on that model
    had an answer cut at max_tokens: raise `REPLY_LABEL_MAX_TOKENS`. Pilot
    cat-v2 first (`--limit 50`) and read the report's `check <post>` lines
    (de-escalation replies Opus files as anti-war that v2 scored pro) before
    the full batch. `v2-rater` (the second rater's reply arm,
    `settings.SECOND_RATER_MODEL`, claude-opus-5-5 by default) refuses
    claude-opus-5, the v2 labels' own model; its caps bound its whole run
    too, and its report leads with coverage (replies still unlabelled,
    failures by outcome, stop reasons). `status|collect|report` are free.
  - `teacher-check --source trump --all` (his whole feed on score_opus's
    model and prompt; `TRUMP_TEACHER_CHECK_MAX_CALLS` covers every post with
    text): `--all --estimate` adds a usage-fit figure from the check's own
    batches (all 3,222 posts with text are labelled since 2026-11-03; with
    none left the estimate crashes, STATUS *Open*). Then
    `--merge` (free) writes score_opus / label_opus onto the feed parquet by
    id. score-posts keeps them on the rows it has, and the posts it adds
    have none until the next `--merge`. Never run `--merge` while
    score-posts runs: it reads the parquet when it starts and writes it back
    as it read it when it ends (the merge refuses while score-posts'
    `<feed>.scoring` marker names a live process on dkbl1). On the GPU path
    the marker is on dkbl2, so run `--merge` only after `sync-data.sh pull`
    of a score-trump-feed Job, never between push and pull: the pull puts
    the PVC's feed parquet, without the merge, over dkbl1's. The pull also
    brings back the PVC's `teacher_check_*` files, so push sends them too
    (since 2026-11-02): never pull from a PVC that was not pushed after the
    last host-side `--all` collect, or its older label cache replaces
    dkbl1's (`sync-data.sh verify` lists what would change). Trump's stance
    of record is score_opus since 2026-11-02 (`settings.TRUMP_STANCE_SCORE_COL`).
  - `post-label pilot|submit --job stance-topic|rater|stance-topic-nohandle`
    (src/analysis/post_labels.py): `post-label estimate` makes no call and
    writes nothing. `pilot` (direct, `--n` 50 by default, at most
    `POST_PILOT_MAX_CALLS`) measures the billed tokens, the cache and how
    answers stop, and the estimate is calibrated on it. Only a pilot of the
    job's current prompt version, `POST_LABEL_MAX_TOKENS` and
    `POST_LABEL_EFFORT`, with `POST_PILOT_MIN_BILLED` (40) billed answers
    pooled, counts: without one, batches send at most `POST_PILOT_MAX_CALLS`
    calls in all, and once a pilot (of any size) or a collected batch of the
    current request has an answer stopped at max_tokens no batch goes,
    `--limit` or not. A pilot is on record (at its direct ceiling) before its
    first call and saves its labels every `POST_PILOT_SAVE_EVERY` answers.
    Pilot each job on each model first. Caps: `POST_*_MAX_CALLS` / `POST_*_MAX_USD` bound each
    job's whole run, every pilot (at direct prices) and batch on any model,
    plus the next batch at its ceiling (no cache hits, output at the pilot's
    p95): no `--limit` split gets past them; a pilot refuses too.
    `status|collect|report` are free.
  - `post-label submit --job stance-topic-census`: stance-topic's prompt,
    model, label file and columns on every other post with text (X
    accounts and Trump's feed), so the topic flag of record rests on an
    Opus label everywhere (19,598 calls: ~16,000 approved at about $22 on
    2026-11-02, then the X archive fill's 3,551 new posts on 2026-11-03).
    `post-label estimate --job stance-topic-census` makes no
    call and prints the calls by set, the expected cost and the ceiling
    against `POST_CENSUS_MAX_CALLS` / `POST_CENSUS_MAX_USD`. The caps
    (20,000 calls and $33 since 2026-11-03) sit at or under the approvals,
    far below a no-cache-hit ceiling: send the census in `--limit` parts
    (collected between parts; each is gated on billed spend so far plus its
    own ceiling), and a part the caps refuse goes back to the user. It
    takes no pilot (stance-topic's pilot and collected batch measure its
    output); `--limit N` sends a first part. It and stance-topic refuse to send
    while the other's batch is open. `status|collect` are free; its report
    is stance-topic's (`post-label report --job stance-topic`).
- **The user's Truth Social account.** `ts-login` (the security code goes to
  the user), `collect-replies`, and `collect-truth` unless `--anonymous`. Free,
  but rate-limited and tied to the account.
- **Cluster changes.** The global list (`secrets.sh` is the user's, and so is
  deleting the namespace). Deploys are yours: `build.sh`, `deploy.sh`,
  `run-now.sh` for Jobs that call no paid API, and `sync-data.sh`, which rsyncs
  over SSH (the `dkbl2-lan` alias in `~/.ssh/config`) into dkbl2's PVC
  directory: this repo's one exception to reading tenant data through kubectl
  (see *Never*). A Job that calls a paid API (e.g. `teacher-check`) is a paid
  run: estimate and ask first.

## Never

- Never `kubectl cp` bulk data to or from the cluster: `scripts/k8s/sync-data.sh`
  is the only path (one rate-capped, checksum-verified rsync; why in
  [docs/decisions.md](docs/decisions.md)).
- Never re-fetch cached X data without `--force` and a reason: it is paid for.
- Never add a paid API call without a cap in `config/settings.py`, an estimate
  that makes no calls, and an on-disk cache that is skipped by default
  (`--force` to redo).
- Never schedule `collect` (no CronJob, no host cron): every run costs money.
- Never add one-off scripts. New behaviour is a subcommand in `src/cli.py`,
  kept thin over importable modules; tunables go in `config/settings.py`,
  lists (accounts, search terms, events, tracked posts) in their `config/`
  module. A one-off data migration is run and deleted, never committed.
- `data/raw/` holds ~99k replies from private individuals: keep rows out of
  git, issues and chat. Only the pipeline's own scoring calls send them out.

## Fast path

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,truthsocial]'
python -m pytest tests                # stub clients, no spend
ruff check <paths> && ruff format <paths>
```

- The system Python has no pip. If `python3 -m venv` fails, the user installs
  `python3.12-venv` with apt. Always work inside `.venv`.
- `.venv` takes the CPU build of torch: dkbl1 has no GPU, and the default
  PyPI torch on Linux is the CUDA build (~3 GB of `nvidia-*` and `triton`
  wheels it can't use). Install it first and the `pip install -e` keeps it.
  The GPU image brings its own CUDA torch (`Dockerfile`).
- `.env` comes from the sops-encrypted `secrets.env` (the user runs
  `sops -d secrets.env > .env`; README *Setup*). Without the key, copy
  `.env.example`. `.env.example` lists every key the code reads.
- The tree is not clean under `ruff check` or `ruff format` yet (line length
  100): lint and format only the files you touch, never the whole repo in a
  feature commit.

## Running things

Everything goes through one Click CLI, `python -m src.cli <command>`:

```bash
python -m src.cli test       # verify X API credentials (one User read, $0.010)
python -m src.cli status     # show what's cached, what's missing
python -m src.cli collect    # incremental fetch -- appends only new tweets since last run
python -m src.cli x-backfill-text --estimate  # cached posts cut at 280 chars: reads to re-get them whole
python -m src.cli x-counts --estimate  # X's own per-hour post counts per account (counts/all), cached
python -m src.cli x-counts --report    # free: X's count vs what each collect window got, and by month
python -m src.cli x-archive probe --estimate  # does full-archive search work here, retweets included?
python -m src.cli x-archive fill --estimate   # the X holes on collect's grid from search/all: plan
                                              #   and price (no call); [--scope empty] [--only ...]
python -m src.cli ts-fill-text  # re-read by id the cached Truth Social posts with no words (free, anonymous;
                                #   done 2026-10-06, new pulls store quote posts themselves)
python -m src.cli analyze    # score all cached data (VADER + RoBERTa; add --llm for Haiku stance)
python -m src.cli visualize  # regenerate all figures
python -m src.cli summary    # stats tables (default --score is the stance of record)
python -m src.cli phases     # tier x phase stance with CIs, split into war posts / topic share
                             #   [--source x|trump] [--by tier|user] [--topic opus|either|llm|keyword]
                             #   [--score score_opus_stance|score_opus55|...] (post-label columns)
python -m src.cli reply-population  # reply audience's Opus stance per post, direct and corrected (free)
                                    #   [--labels v1|v2|v2b] [--first-draws]; v2 pools the fresh draws
python -m src.cli reply-label report --job v2b  # v2 vs v2b, paired on the first draws (free)
python -m src.cli reply-loyalty     # Opus stance by replier loyalty tier, high - low (free)
python -m src.cli reply-fresh-sample  # draw and record the fresh reply sample (free; --dry-run)
python -m src.cli reply-label estimate --job v2b|cat-v2|v2-fresh|v2-rater [--post SLUG]  # no API call
python -m src.cli post-label estimate --job stance-topic|rater|stance-topic-nohandle  # no API call
python -m src.cli post-label estimate --job stance-topic-census  # the census: no API call
python -m src.cli post-label report --job stance-topic  # rater agreement, filter misses (free)
python -m src.cli topic-label unjudgeable --results-file F  # record Haiku's no-verdict posts (free)
python -m src.cli teacher-retest    # Opus label test-retest on 497 posts (free)
python -m src.cli export-web # chart JSON for the dkweb blog post
python -m src.cli backup     # sync data/ to the off-box S3 bucket (after any paid run)
```

Refresh order (the paid steps are *Ask first*): Truth Social first, it is
free (`collect-truth`, `collect-replies`, `event-study`); then
`collect --estimate` and `collect`, refreshing heavy accounts before the
timeline horizon eats the gap; `analyze`; `relabel submit` -> `status` ->
`collect` -> `merge` for the new posts; `topic-label estimate|submit|status|collect`;
`post-label estimate|submit|status|collect --job stance-topic` for new war
candidates, then `--job stance-topic-census` for the rest (the topic flag of
record); `phases`, `visualize`, `export-web`; `backup`.

A batch `collect` (`relabel`, `topic-label`, `--batch collect`, `reply-label
collect`) streams results through the SDK, which breaks on this host (httpx
`ReadError`); it then downloads the batch's `results_url` itself (resuming a
cut download) and reads that. `--results-file` still reads a file you
downloaded. A collect of a batch that carried replies (`reply-teacher-check
--v2`, `reply-label`) deletes the batch on the API once every result is in,
so the replies are not kept there for 29 days; `--keep-batch` skips that.
A failed or kept delete is retried by the next collect without
`--keep-batch` and by the next submit, which refuses while it fails; a
batch already gone (404, say deleted in the Console) counts as deleted.
Every batch create is sent once (no SDK retry: it has no idempotency key,
so a lost answer would make a second, unrecorded batch). A create that
dies without an answer leaves a "submitting" state with no batch id, and
every submit refuses until you check the Console's Batches page and fill
in the id (or, if no batch was made, restore the state, as the message
says).

## Layout

```
config/  settings.py (paths, caps, batch sizes, models, colours), accounts.py (tiers),
         timeline.py (events), tracked_posts.py (Trump posts for reply analysis)
src/     cli.py (every command), backup.py
  collectors/     x_collector.py, x_backfill.py (posts cut at 280), truthsocial_collector.py,
                  ts_fill_text.py (Truth Social posts with no words), x_counts.py (counts/all
                  and its report), x_archive.py (search/all probe), x_archive_fill.py
                  (search/all fill on collect's window grid)
  analysis/       sentiment.py (VADER, RoBERTa, LLM stance), event_study.py (replies),
                  relabel.py, topic_label.py (Batch API), inference.py (CIs, shift-share,
                  reply population, fresh draws), stance_local.py (GPU experiments, teacher
                  checks, batch transport), reply_labels.py (v2b, cat-v2, v2-fresh),
                  batch_io.py (results download, batch delete), pricing.py (model prices, tags),
                  post_labels.py (alternate post raters: stance-topic, second rater, no-handle)
  visualization/  plots.py, web_export.py
data/    raw/x/<handle>.jsonl, raw/x_sampling/ (collect and fill windows), raw/x_counts/,
         raw/x_archive/ (probe, fill state, request log, set-aside posts), raw/truthsocial/,
         processed/sentiment_all.parquet,
         processed/reply_sentiment.parquet, models/   (gitignored)
k8s/, scripts/k8s/   GPU experiment Jobs
```

## Where things run

- The pipeline runs in the host venv on dkbl1 (CPU), by hand. Nothing is
  scheduled.
- GPU work (stance-model experiments) runs as one-shot Jobs on dkbl2's RTX 3080
  from `k8s/` (runbook and decision rule: [k8s/README.md](k8s/README.md)).
  Nothing stays deployed: the user deletes the namespace when experiments
  end. Reviving it is `build.sh`, `deploy.sh <tag>`, `kubectl apply -f
  k8s/data-sync.yaml` (binds the PVC), `sync-data.sh push`, then
  `run-now.sh <job>`, which you run (outside quant's windows), plus
  `secrets.sh` (teacher-check only) and, if the push says so, a `chmod o+x`
  on dkbl2's storage dir, which the user runs.
- `data/` on dkbl1 is the source of truth. The PVC on dkbl2 is a working copy.
- `export-web` writes into the sibling repo `~/repos/dkweb`
  (`WEB_EXPORT_DIR`); the agent needs write access to it.

## Conventions and invariants

### Tiers and scorers

- `config/accounts.py` groups handles into tiers: admin, maga_prowar,
  maga_antiwar, opposition, media, religious_authority, plus `search`
  (keyword-query public-sentiment proxy). MTG is `@mtgreenee`.
- `settings.STANCE_SCORE_COL` (`score_opus`, Claude Opus 5 via `relabel`) is
  the stance of record; `summary`, `phases` and `event-study`'s broadcaster
  half default to it, and `visualize` draws every `PLOT_SCORE_COLS` column.
  `event-study` on replies falls back to RoBERTa (`score_transformer`), and
  its critical / supportive shares stay `label_transformer` even with
  `--score`.
  Trump's feed has its own default, `settings.TRUMP_STANCE_SCORE_COL`
  (score_opus since 2026-11-02, from `teacher-check --source trump
  --merge`): `phases --source trump` and export-web's trump_phases read it,
  and a post without it is left out with a count, never imputed.
  `score_opus_distilled` (DeBERTa distilled from the Opus labels, run at its
  256-token training length) is the reply scorer of record for replies (and
  stays on Trump's feed for `--score`); `score_opus_distilled_v0` keeps the
  pre-2026-09-30 reply scores from an overwritten fit and never joins a
  series. Never overwrite
  a `data/models/stance_distilled_final*` dir: the data holds its scores.
  Other columns: `score_vader`, `score_transformer`
  (RoBERTa valence, a quick look only), `score_llm` (Haiku). Anything
  published or compared across tiers uses `score_opus`. Never mix scorers in
  one series; a new scorer gets its own column or rescores the whole set.
- **Report war-post stance, not the all-post mean.** The prompt makes Opus give
  off-topic posts ("Deport them!", "Amen") a stance from the author, so a
  tier's all-post mean moves with its war share. The about-the-war flag of
  record is Opus's topic class (`TOPIC_SOURCE = "opus"`, since 2026-11-02):
  a post is a war post when its current `post-label` stance-topic label
  (any set: candidates, random arm, census) says `iran_war`; a post with no
  label made from its text now counts by the either flag, with a warning
  that counts them (the census leaves ~0), and no label file at all is an
  error. Every `inference.war_flag` reader takes it (`phases`, `visualize`,
  `export-web`, the Trump check's report, the x-backfill war counts); a new
  reader must too, or fail loudly. `--topic either` (the keyword
  `WAR_TOPIC_PATTERN` OR the Haiku label from `topic-label`), `llm` and
  `keyword` stay for comparison: the either flag missed ~29% of the
  Iran-war X posts and took in ~1,300 that are not.
- **Faith-based voices need LLM stance.** VADER and RoBERTa sign-flip religious
  anti-war rhetoric ("peace", "mercy" read as positive). Never report a
  valence score for `religious_authority` or any future faith-based tier
  unless flagged as miscalibrated. See README *The religious sign flip*.
- The LLM parsers strip ```` ```json ```` fences before `json.loads`
  (`sentiment.py`, `event_study.py`): Haiku wraps its JSON despite the
  prompt. Keep stripping; don't prompt-engineer it away.
- Retweets, ReTruths and Truth Social quote posts count as the account's
  messaging (stored as `RT @acct: ...`; a quote with words of its own keeps
  them). Posts and replies with no text (`src/text_rules.py`: under 3
  characters once links and a leading `RT @x: ` or `RT: ` are removed)
  leave every denominator and are never scored.
- Reply population shares come from `reply-population` (Opus-corrected
  against the v2 reply labels, `settings.REPLY_TEACHER_LABELS_VERSION`), not
  raw `score_opus_distilled`, which runs ~15 points pro-war (63% vs 49% for
  the v2 Opus labels on the 1,157 labelled replies with text). Its direct
  columns (`opus_*`) weight the labels alone by bucket.
- Reply label files: one per prompt, model and id source
  (`stance_local.reply_labels_path`). The v2 label set is the stance
  sample's file plus the fresh draws' (`teacher_labels_replies_v2-fresh_*`,
  same prompt and model, never written into the v2 file); v2b (v2 without
  the praise rule) and cat-v2 (Opus categories with the parent post) have
  their own. The Haiku categories in `stance_sample.parquet` stay as
  history. The first random draws (`reply_sample_draws.parquet`) and the
  fresh ones (`reply_sample_draws_fresh.parquet`) are frozen: never redraw.
- Set two reply label sets against each other on the same draws only: v2
  pools the fresh draws, v1 and v2b label the first draws alone
  (`reply-population --first-draws`; `reply-label report --job v2b` for
  the paired v2 - v2b difference). A post's fresh draws join only once
  nearly all are labelled (`REPLY_FRESH_POOL_MIN_SHARE`): until its batch
  is collected the labelled ones are reused flipper labels. The reply
  bootstrap CIs carry the finite-population correction. The published
  reply figures are the pooled first + fresh draws (9,071; the user's call,
  2026-11-02): v2 is the label set of record, v2b reported beside it.
- `reply-loyalty` (inference.reply_loyalty, free) is the loyalty gradient:
  the Opus stance of low-, medium- and high-loyalty repliers and high minus
  low per post, pooled and within posts, on reply-population's sample,
  weights and intervals. A replier's tier takes the account's age on
  `settings.LOYALTY_REFERENCE_DATE` (fixed, so tiers do not move with the
  run date); event-study's loyalty table still uses the run date. A tier
  is per reply (that reply's account snapshot): a replier whose bio
  differs between replies is in two, and the output counts them.
- A model's relabel columns are `score_<tag>` with a tag no other model has
  (`settings.MODEL_TAGS`: claude-opus-5 `opus`, claude-opus-5-5 `opus55`,
  claude-fable-5-1 `fable51`); `relabel merge` refuses a column
  `sentiment_all_columns.json` records as another model's. Every estimate
  prices a model by its own row of `settings.MODEL_PRICES` and refuses one
  with none.
- Alternate post raters (`post-label`) keep their own label files
  (`post_labels_<job>_<model>.parquet`) and columns, named by the model's
  tag: `score_opus_stance` / `topic_opus` (the stance-topic prompt on
  claude-opus-5), `score_opus55` (the score_opus prompt on claude-opus-5-5;
  the rater job refuses claude-opus-5 and Haiku, whose `score_<tag>` are
  existing series), `score_opus_stance_nohandle` / `topic_opus_nohandle`.
  They are never written into sentiment_all.parquet or the Trump feed:
  `phases --score <col>` joins them from the label file onto every post the
  file labels when the post's text is the one labelled
  (`post_labels.attach`), and the topic flag of record reads `topic_opus`
  the same way. The stance-topic labels of every set join a series: the
  candidates, the random arm (since 2026-11-02) and the census
  (`stance-topic-census` writes stance-topic's file and columns); the
  rater's cover the candidates only, and `phases` counts the war posts
  without one. On such a column `phases` reports the war-post columns only
  (the all-post, share and share-effect figures are NaN in its CSVs: the
  column holds only the posts its job labels). The no-handle arm is a
  paired subsample for its report only (its ALL row weights each tier by
  its pool over its pairs). A label made from a text the post no longer has is
  never used, and the post is labelled again (the old label moves to the
  cache's `*_superseded` archive; x-backfill-text, `collect --force` and
  ts-fill-text move these caches' labels there too, as for relabel's).
  None of these goes into `STANCE_SCORE_COL` or `PLOT_SCORE_COLS`; making
  one a definition of record is the user's call, with a decisions.md entry
  (`topic_opus` is the topic flag of record since 2026-11-02).
  The random posts (`post_random_draws.parquet`) and the no-handle subsample
  (`post_nohandle_draws.parquet`) are frozen by the first pilot or submit:
  never redraw.
- `score_opus` on Trump's feed comes only from `teacher-check --source trump
  --merge` (the check's labels, claude-opus-5, llm_prompt), claimed in
  `truthsocial_trump_stance_columns.json` like relabel's columns.
- Each answer Haiku gives with no about-the-war verdict (link shares) is
  recorded in `topic_unjudgeable.parquet` (post, batch, the length of the
  text it was sent); a post with `TOPIC_UNJUDGEABLE_MIN_ATTEMPTS` (3) of
  them on the same text is not sent to `topic-label` again
  (`--retry-unjudgeable` overrides): retries still recover a fifth to a
  third. With no Haiku label, `war_flag` counts it by the keyword pattern
  as before. Errored or expired requests are retried, not recorded.
- A scorer fit on the Opus-labelled replies (`reply-distill`) enters
  `reply-population` only as a cross-fit column (`--crossfit`: each post
  scored by a fit that never saw its labels); `reply-population` refuses
  a full-sample one. `reply_sentiment_columns.json` records the model
  behind each such column; one column never takes two models' scores.
- `analyze --llm` and `event-study` / `stance` are restart-safe and
  incremental: only posts or replies without a score are sent. `analyze`
  restores a prior score only onto the same text, so a post whose text
  changed is rescored.

### X collection and budget

- Budget: a real `collect` refuses to start when its maximum exceeds
  `X_RUN_BUDGET_USD` (`config/settings.py`). The maximum counts the reads
  ($0.005 each) and one user lookup per account fetched
  (`X_USER_READ_COST_USD`, $0.010; none for a skipped account). `test`
  makes one lookup too, so it is not free. The maximum is strict: a
  search stops once fewer than X's 10-post minimum are left of its
  `MAX_TWEETS_PER_SEARCH`.
- `collect` is incremental per account (only tweets newer than the cached
  `created_at`). Each gap is walked oldest first, from the account's
  cursor, in fixed `GAP_FILL_WINDOW_HOURS` (35 h) windows, whose ends step
  through every clock hour and weekday: the timeline endpoint is
  newest-first, so one capped fetch over a long gap would keep the newest N
  and drop the rest.
  - Sampled accounts (`ACCOUNT_WINDOW_CAPS`, 5 per window, about 48 per 14
    days) keep the newest <= N posts of every window, with no rollover: a
    quiet window's unused reads are not passed on. The planner lengthens
    the windows only when the caps would add up to more than the account's
    backstop (`ACCOUNT_CAP_OVERRIDES`, else `MAX_TWEETS_PER_USER`). A run's
    last window still gets 5 when it is only 17.5-35 h long (a shorter gap
    waits for the next run), so a gap under 35 h is up to twice as dense:
    don't rerun sampled accounts within a few days.
  - `collect --force` skips a sampled account whose cache stops half a
    window or more before now when the forced plan lengthens its windows
    (it plans the whole collection window at one density, which would
    sample the uncached tail thinly for good): run an incremental `collect`
    first.
  - Every other account is take-all: it reads each window whole until its
    backstop runs out, then defers the windows left to the next run (an
    ERROR names them; if it recurs, add the account to
    `ACCOUNT_WINDOW_CAPS`).
  - A window logged `TRUNCATED` had more posts than it kept: by design for
    a sampled account; for a take-all one, the window's older posts are not
    read again. Warnings flag leading empty windows when the windows after
    them hold an estimated 1,600+ posts (`X_TIMELINE_HORIZON_POSTS` / 2;
    likely past the ~3,200-post horizon), and an empty window next to a
    dense truncated one (possible fetch failure).
  - Windows share their edge second and posts are deduped by id. No
    request asks for fewer than 5 posts (X's minimum, billed anyway) and a
    window stops after `X_MAX_PAGES_PER_WINDOW` pages.
  - In an incremental run or a first pull, each window's new posts are
    appended (fsync) as soon as it is read, so a killed run resumes from a
    window boundary. A `--force` merge is written once, at the end, or with
    what it read when it fails part-way. Each window gets a row (cap, got,
    new, truncated, kept span) in
    `data/raw/x_sampling/windows_<run end date>.jsonl` (`X_SAMPLING_DIR`,
    outside the `data/raw/x/*.jsonl` glob) once its posts are stored, for
    per-window volume weights.
  - `collect --only a,b` plans and collects only those handles; searches
    still run unless `--no-search`. An empty `--only` is refused, not read
    as every account.
- X's own counts tell a silent window from a failed fetch. `x-counts`
  caches counts/all per query and granularity in `data/raw/x_counts/`
  (`from:<handle>` per account, `--terms`, `--query`; hour buckets by
  default), writing each page as it arrives: a covered range is skipped, a
  killed run resumes, `--force` refetches. The cache key ignores the case
  of `from:` handles. A stretch X answers short of its 31-day paging is
  recorded only as far as its buckets reach (a WARNING; the next run plans
  the rest), never as zeros. A 403, a 401 (check the token) or a
  usage-capped 429 stops the run with no further request; any other 429
  waits per the rate-limit header, at most `X_429_MAX_RETRIES` times.
  `x-counts --report` (free) prints, per manifest window, its source
  (`collect`, or `archive` for an `x-archive fill` row; verdicts are
  counted per account and source), cap, got, held (cached now) and X's
  count (`lo-hi` when
  the window's edges cut hour buckets), with a verdict: `silent` (X counts
  0), `complete` (got everything X counts), `sampled ok` (got the cap, X
  counts more), `edge` (inside the edge-bucket range), `short` (under X's
  count and the cap, not zero: deleted since?), `FETCH SHORTFALL` (got
  less than min(cap, count) where X counts at least the cap, or got 0
  where X counts posts), `not counted`. Then X's count against the cache
  by month (`~` marks a month only partly counted). X's counts may leave
  out native retweets (unverified): a `+` (got more than X counts) is
  itself evidence; `--query 'from:<handle> is:retweet'` adds the `X RT`
  column.
- `x-archive probe` asks search/all (full archive) for one whole UTC day
  of @PeteHegseth in February and one in the uncapped Sep 13 slice, where
  the cache holds every post, and compares id for id; it also counts his
  posts per day with and without `is:retweet`. Its verdict (`no archive
  access`, `archive works, retweets dropped`, `archive works incl.
  retweets`, `archive misses older posts`, `inconclusive`) decides whether
  an archive fill is worth building. Only a 403 is a result: a 401, a
  usage cap or a counts step that stops short leaves the probe unfinished,
  and a rerun resumes it. Result and posts in `data/raw/x_archive/`; a
  finished probe is reused unless `--force`.
- `x-archive fill` (src/collectors/x_archive_fill.py) fills the holes
  (late starts, horizon holes, short collect windows) from search/all on
  collect's grid and with collect's rule, so the filled stretches follow
  the current design ([docs/decisions.md](docs/decisions.md) 2026-11-02).
  - Grid: each account's collect windows from the manifests, as they are,
    extended back to `COLLECTION_START` in 35 h steps (a head under half a
    window merges into the first, as collect merges a short tail); nothing
    forward while the manifests end within half a window of the end of
    `ANALYSIS_END`. A later collect run owns its whole span: an earlier
    run's windows inside it are dropped and the ones across its edges cut
    there, so a `collect --force` with longer windows replaces the
    incremental runs' grid where it ran.
  - Target: a sampled window up to its cap, a take-all window up to X's
    hour count; windows are picked with the edge buckets counted whole.
    Counts never end a walk: a window is done when it holds its target or
    every gap that may hold a post has been asked.
  - Requests: the posts a window holds cut it into gaps bounded by their
    ids (`since_id` / `until_id`), so no held post is bought again. A time
    edge goes as the id it maps to (`id_at`: an id carries its creation
    time in ms), because X answers 400 to any mix of ids with `start_time` /
    `end_time`. A gap is skipped only when
    every hour bucket it touches is held whole, checked when the walk
    reaches it (a bucket several gaps share closes once one of them yields
    its post). A sampled walk goes newest first, narrowed with the hour
    counts, the crossing hour in `X_ARCHIVE_FILL_SLICE_MINUTES` slices.
  - Keep rule: a sampled window keeps the newest cap - held. The rest of a
    page (search/all returns at least 10) were billed and go to
    `x_archive/fill_<handle>_extra.jsonl`, outside the cache's glob; a later
    walk takes them before it buys anything. Filled posts carry `"via":
    "search_all"`, and `x-backfill-text` skips them (read with note_tweet).
  - Writes: per request a pending line in `x_archive/fill_requests.jsonl`
    before it is sent, then the posts (fsync), then the answered line (the
    spend tally). A pending line with no answered line (a kill, a power-off
    or a dropped connection while X answered) counts at its max_results, so
    the tally errs high; an X error status closes it at 0. An append after a
    line a kill cut starts on a new line, so only the cut record is lost
    (in `data/raw/x/<handle>.jsonl` the strict reader still refuses the file
    until that one line is removed). Per window a row in
    `x_sampling/windows_archive_<date>.jsonl` (`source: "archive"`, `got` =
    what the window holds after), then its state in
    `x_archive/fill_<handle>.json`. A rerun skips windows done and plans the
    rest from the cache; `--force` plans done windows again (the old state
    kept aside).
  - collect's cursor is the newest cached post, and the fill writes nothing
    past the last collect window, so it never moves the cursor. Errors as
    in `x-counts`.
- Requests ask for `note_tweet` (full text over 280 characters) and
  `referenced_tweets`, never expansions: X bills per resource returned
  ([docs/decisions.md](docs/decisions.md)). `collect --force` merges the
  capped re-read into the cache by id and keeps what it didn't return; a
  post whose text changed has its labels moved to `*_superseded` and is
  rescored.
- X's user timeline returns only an account's latest ~3,200 tweets. Refresh
  heavy accounts at least every ~2 months, or the gap's windows come back
  empty and wait for `x-archive fill` (search/all).
- Keyword search (`/search/recent`) reaches back only ~7 days.
- To add an account or search term, edit `config/accounts.py` and rerun.

### Truth Social

- Plain httpx/requests get 403 from Cloudflare; everything goes through
  `curl_cffi` (the `truthsocial` extra). `collect-truth` only appends runs
  that reach the cache edge (Truth Social ignores `min_id`, so it walks
  backward): authenticated, it refuses a batch that stops short; `--anonymous`
  holds pages in `<handle>.partial.jsonl` until it gets there, so a killed
  run resumes.
- Replies use the v2 descendants endpoint (`TS_DESCENDANTS_PATH`; v1 is dead)
  with the `TRUTHSOCIAL_TOKEN` that `ts-login` writes to `.env`. After a
  `ts-login`, the user re-encrypts `secrets.env`.
- truthbrush reads `TRUTHSOCIAL_USERNAME` / `TRUTHSOCIAL_PASSWORD`; parts of our
  code read `TRUTH_SOCIAL_*`. Keep both set. Details:
  [docs/truthsocial-api.md](docs/truthsocial-api.md).

### CPU and memory (dkbl1 powers off under bursty or sustained load)

Guardrails in `src/analysis/sentiment.py` + `config/settings.py`; keep them:

1. Batched inference (`ROBERTA_BATCH_SIZE` 16): never score posts one by one.
2. Thread cap (`TORCH_NUM_THREADS` 2, plus `OMP_NUM_THREADS` /
   `MKL_NUM_THREADS`), set before the pipeline is imported.
3. Checkpoints inside the RoBERTa loop every
   `ROBERTA_CHECKPOINT_EVERY_N_BATCHES` (25), not only between phases.
4. RSS ceiling (`ROBERTA_MAX_RSS_MB` 6144): abort cleanly, with a checkpoint,
   before the OOM killer does. RSS is logged at phase boundaries and
   checkpoints.
5. Resume: `analyze` restores prior scores by post id plus any RoBERTa
   checkpoint, so recovery is rerunning it.
6. LLM scoring skips scored posts and saves every `LLM_SAVE_EVERY_N` (100).
7. `del pipe; gc.collect()` in `try/finally`.

Baseline: RoBERTa peaks at ~1.3 GB RSS and takes ~1-1.7 s per 16-post batch
(~9 min for 5.6k posts, ~33 min for 26k replies). A big drift means a leak or
a removed thread cap. Treat any bulk transfer the same way: one steady,
rate-capped, resumable process.

### GPU Jobs (`k8s/`)

- One-shot Jobs only, on the quant tenant pattern (homelab-infra
  `docs/k8s-workloads.md`): image built in-cluster from `Dockerfile`, pinned by
  git sha with `deploy.sh <sha>`, uid 1000, Secret from `.env`, pods pinned to
  dkbl2 by `nodeSelector`, local-path PVC there. A new experiment is a Job dir
  under `k8s/jobs/`, a CLI subcommand and a row in `k8s/README.md`.
- A failed Job pages Slack (`KubeJobFailed`); read the pod's `kubectl logs`
  before proposing a fix.

### Backup

`backup` syncs `data/raw` and `data/processed` (Standard) and `data/models`
(Glacier IR) to the bucket in homelab-infra
`terraform/iran-sentiment-backups.tf`: no `--delete`, writer without
`DeleteObject`, versioned bucket. It needs `IRAN_BACKUP_S3_BUCKET` and the AWS
keys in `.env`. Run it after every paid `collect`, `x-backfill-text`,
`x-counts`, `x-archive probe`, `x-archive fill`, `relabel` or `topic-label`.
The host's restic drive is often unplugged, so S3 is the copy to count on.

### Blog charts (dkweb)

Rerun `export-web` after any rescoring, then `npm run build` in dkweb.
`replies.json` pools the fresh draws (`REPLY_POPULATION_FRESH_DRAWS`, the
published basis since 2026-11-02: 9,071 draws); export-web warns only
when it exports without them. Stance colours are dkweb's `--viz-anti` / `--viz-pro` / `--viz-neutral` tokens in
`global.css`.

## Docs map

- [README.md](README.md): findings, methodology, stance-model experiments, setup, cost.
- [docs/STATUS.md](docs/STATUS.md): data coverage, open work, what a refresh needs.
- [docs/decisions.md](docs/decisions.md): dated decisions, why, and what was rejected.
- [docs/truthsocial-api.md](docs/truthsocial-api.md): endpoints, auth flow, rate limits.
- [k8s/README.md](k8s/README.md): GPU Job runbook and the scorer decision rule.
- homelab-infra `docs/secrets.md` (sops), `docs/k8s-workloads.md` (tenant
  conventions), `terraform/iran-sentiment-backups.tf` (backup bucket).
