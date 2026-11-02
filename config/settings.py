"""
Runtime settings for the iran-sentiment pipeline.

All tunables (paths, budget caps, batch sizes, model names) live here so
that scripts can stay thin and data-driven.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# IRAN_DATA_DIR relocates the whole data tree (the k8s Jobs mount their
# PersistentVolume at /data and set it); default is the checkout's data/.
DATA_DIR = Path(os.environ.get("IRAN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DIR / "figures"

X_RAW_DIR = RAW_DIR / "x"
# One row per `collect` window (x_collector.collect_user): its cap, what it
# got and whether more was left. It must stay outside X_RAW_DIR, because
# load_all_cached, `status` and x_backfill glob X_RAW_DIR/*.jsonl; `backup`
# syncs raw/ recursively, so it is backed up.
X_SAMPLING_DIR = RAW_DIR / "x_sampling"
# X's own post counts per query (`x-counts`, one JSON file per query and
# granularity) and the full-archive probe (`x-archive probe`: its result and
# the posts it returned). Both outside X_RAW_DIR for the same reason.
X_COUNTS_DIR = RAW_DIR / "x_counts"
X_ARCHIVE_DIR = RAW_DIR / "x_archive"
TRUTH_SOCIAL_RAW_DIR = RAW_DIR / "truthsocial"

# Checkpoints for resumable analysis
VADER_CHECKPOINT = PROCESSED_DIR / "checkpoint_vader.parquet"
ROBERTA_CHECKPOINT = PROCESSED_DIR / "checkpoint_roberta.parquet"
SENTIMENT_OUTPUT = PROCESSED_DIR / "sentiment_all.parquet"

# Scored Truth Social reply data (separate from broadcaster sentiment)
REPLY_SENTIMENT_OUTPUT = PROCESSED_DIR / "reply_sentiment.parquet"
# Trump's own Truth Social feed, scored by the Opus-taught distilled model
# (`score-posts ... --col score_opus_distilled`).
TRUMP_FEED_STANCE = PROCESSED_DIR / "truthsocial_trump_stance.parquet"
# The random bucket draws behind the Opus-labelled reply sample (id,
# tracked_slug, bucket at draw time), recorded once per tracked post so
# `reply-population` weights the sample that was labelled, not a fresh
# redraw from a reply frame that has grown since.
REPLY_DRAWS_MANIFEST = PROCESSED_DIR / "reply_sample_draws.parquet"


# ── Date window for data collection / analysis ────────────────────
COLLECTION_START = datetime(2026, 2, 1)
# End-of-window is "now" (evaluated at CLI startup), with a 30s buffer
# because the X API rejects end_time values that are too close to the
# current request time. Naive UTC to match COLLECTION_START's shape.
COLLECTION_END = (
    datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    - timedelta(seconds=30)
)


# ── Score columns ──────────────────────────────────────────────────
# score_opus (Claude Opus 5 via `relabel`) is the stance of record since
# 2026-09-18: Haiku (score_llm) reads angry hawkish posts as anti-war and
# put the pro-war MAGA tier at -0.06 where Opus puts it at +0.25. Columns
# absent from the frame are skipped by the plots.
STANCE_SCORE_COL = "score_opus"
PLOT_SCORE_COLS = ["score_vader", "score_transformer", "score_llm", "score_opus"]
# Posts and replies with fewer characters, once links and a leading
# "RT @x: " / "RT: " are removed (src/text_rules.py), carry no text to judge:
# an image or video with no caption, or a quote of a post with no text, on
# Truth Social; a bare t.co link on X. They are not scored and leave every
# share and mean, rather than counting as off-topic. Same cut as the reply
# sampler's media_only. 1,810 of Trump's 4,360 Truth Social posts fall under
# it (ts-fill-text, 2026-10-06: 1,589 media-only, 221 quotes of his own
# no-text posts); the distilled model had scored 1,622 of them a constant
# +0.111 and the 188 bare "RT: <link>" quote fallbacks on the link (~+0.03).
MIN_TEXT_CHARS = 3


# ── War-topic filter ───────────────────────────────────────────────
# Most posts are not about the war (StateDept, POTUS and the VP post about
# everything) and score ~0, so a tier's all-post mean moves when the SHARE of
# war posts moves even if the war posts themselves do not. `phases` splits
# each change into that share effect and the on-topic stance change. The
# filter is keyword-only on purpose: it must not depend on the score it is
# used to decompose. Case-insensitive.
WAR_TOPIC_PATTERN = (
    r"\biran|hormuz|tehran|khamenei|ayatollah|\birgc\b|islamic republic|persian gulf"
    r"|nuclear|enrich|\bwars?\b|warmonger|strikes?\b|airstrike|ceasefire|cease-fire"
    r"|\bbomb|missile|\bdrones?\b|\btroops\b|israel|\bidf\b|netanyahu|hezbollah|houthi"
    r"|middle east|blockade|islamabad|regime"
)
# About-the-war flag for `phases` / `export-web`: "keyword" (the regex,
# loose), "llm" (Haiku labels from `topic-label`, strict: it drops the Pope's
# war appeals that never name Iran), or "either" (one or the other, the
# default since 2026-09-23). Report only what holds under all three.
TOPIC_SOURCE = "either"
# `topic-label`: a post Haiku answered with no verdict (a link share) this
# many times on the same text counts as unjudgeable and is not sent again.
# Retries do recover some (the four batches to 2026-11-02: 30% of the posts
# got a verdict on the next try after one no-verdict, 25% after two, 20%
# after three or more) and a resend costs ~$0.0001, so it waits for three.
TOPIC_UNJUDGEABLE_MIN_ATTEMPTS = 3
# Group pairs `phases` reports gaps for (a - b, within each phase).
PHASE_GAPS = [("maga_prowar", "admin"), ("maga_prowar", "maga_antiwar")]
# Bootstrap for the phase tables: posts are resampled in account-day blocks
# within each account, so same-day posts move together. Each account keeps
# its total day count, but its days (and so its weight in a phase) are
# resampled across the whole series. The intervals are sampling noise for
# THESE accounts, not a claim about accounts we did not track.
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


# ── Event overlays ─────────────────────────────────────────────────
# Plots draw a marker line for every timeline event inside the analysis
# window but only print a label for events at or above this importance
# (1-5; pre-May events default to 3). Raise to 4 for a readable plot once
# the window spans the whole war.
EVENT_LABEL_MIN_IMPORTANCE = 4   # raised from 3 on 2026-09-18: the window now spans Feb-Sep


# ── Budget caps (per-account) ──────────────────────────────────────
# X charges ~$0.005/read on pay-as-you-go. These caps guard against a
# single prolific account (e.g. @marklevinshow posts 35+ times/day)
# blowing through the budget.
MAX_TWEETS_PER_USER = 500  # ≈ $2.50 cap per user: the backstop for take-all accounts
MAX_TWEETS_PER_SEARCH = 150  # ≈ $0.75 cap per search term
# GET /2/tweets/search/recent has a max_results minimum of 10: a search
# stops once fewer than that are left of its cap, so the cap stays strict.
X_MIN_SEARCH_READS_PER_REQUEST = 10
# A guard against empty search pages that still carry a next_token.
X_MAX_PAGES_PER_SEARCH = 20
X_READ_COST_USD = 0.005
# The get_user lookup (a User read) that resolves each fetched account's id:
# once per account per run, counted in the plan maximum (none for an account
# the plan skips). `test` makes one too, so it is not free either.
X_USER_READ_COST_USD = 0.010

# Sampled accounts: each keeps the newest <= N posts of every
# GAP_FILL_WINDOW_HOURS window, with no rollover (a quiet window's unused
# reads are not passed on). 5 per 35 h is 48 per 14 days, the May-Sep
# density (45-50 per two-week slice). The exception is a run's last window:
# when it is 17.5-35 h long it still gets 5, up to twice as dense. Over a
# gap of two weeks or more that is within ~5%, but a gap under 35 h is
# nothing but that window (a daily rerun would oversample by ~half), so
# don't rerun sampled accounts within a few days. Listed: the five accounts
# sampled at 450 on 2026-09-18 and the six that hit slice caps in 6-9
# slices then (collect_2026-09-18.log; @mtgreenee and @Pontifex hit one
# each and stay take-all). Accounts not listed are take-all: every post in
# the gap, up to their backstop (ACCOUNT_CAP_OVERRIDES /
# MAX_TWEETS_PER_USER).
ACCOUNT_WINDOW_CAPS: dict[str, int] = {
    "LauraLoomer": 5,
    "marklevinshow": 5,
    "RealAlexJones": 5,
    "BarakRavid": 5,
    "StateDept": 5,
    "POTUS": 5,
    "PeteHegseth": 5,
    "VP": 5,
    "VaticanNews": 5,
    "RealCandaceO": 5,
    "WhiteHouse": 5,
}

# Per-account overrides of MAX_TWEETS_PER_USER: a per-run backstop. For a
# sampled account it binds only on a gap over about four months
# (450 / 5 per window x 35 h = 131 days), when the planner lengthens the
# windows to fit; density comes from ACCOUNT_WINDOW_CAPS. (Until 2026-10-31
# these were the per-run sample sizes, shared evenly across 14-day slices.)
ACCOUNT_CAP_OVERRIDES: dict[str, int] = {
    "LauraLoomer": 450,
    "marklevinshow": 450,
    "RealAlexJones": 450,
    "BarakRavid": 450,
    "StateDept": 450,
}

# Hard ceiling on one `collect` run. The CLI prints the per-account plan
# and refuses to start if the plan's maximum spend exceeds this. Set it to
# the X credit you are actually willing to burn, never above the current
# balance; raise it deliberately, per run.
X_RUN_BUDGET_USD = 45.00   # raised 2026-09-18 after a top-up for the May->Sep refresh

# Long-post backfill (`x-backfill-text`, src/collectors/x_backfill.py). Until
# 2026-10-04 requests did not ask for note_tweet, so a post over 280
# characters was cached cut there; the last such collect wrote the cache on
# 2026-09-18 at 01:48 UTC, so a post created later was fetched whole.
X_CUT_TEXT_BEFORE = datetime.fromisoformat("2026-09-18T01:48:00+00:00")
# A cached original is a candidate when its cut length (x_backfill.cut_length)
# is at most 280 and at least X_BACKFILL_MIN_CHARS, or at least
# X_BACKFILL_OPEN_MIN_CHARS when it ends mid-sentence. Measured 2026-10-04 on
# the 12,154 cached originals: 27 per character from 240 to 265, then 2,911
# from 270 to 280 (~300 expected without a cut, so ~90% are cut) and 182 from
# 266 to 269 (110 expected), 70 of them ending mid-sentence (24 expected).
X_BACKFILL_MIN_CHARS = 270
X_BACKFILL_OPEN_MIN_CHARS = 266
# A run refuses to start with more reads to make than this (and, like
# `collect`, when they would cost more than X_RUN_BUDGET_USD). Approved on
# 2026-10-04: about $10-16, every post near 280 characters.
X_BACKFILL_MAX_READS = 3300
X_LOOKUP_BATCH = 100       # GET /2/tweets takes up to 100 ids per request
# A run stops, not journalling the batch, once it has read this many posts
# and none came with note_tweet (~90% should): X is not sending the field.
X_BACKFILL_NOTE_CHECK_MIN = 20


# ── Truth Social pacing ────────────────────────────────────────────
# The anonymous public API 429s after a handful of quick pages (seen
# 2026-09-16: 3 pages then 429). Pace requests and back off on 429
# using the response's Retry-After / X-RateLimit-Reset headers.
TS_PAGE_DELAY_S = 1.5        # sleep before every request
TS_DEFAULT_BACKOFF_S = 60    # when the 429 carries no usable header
TS_MIN_BACKOFF_S = 10
TS_MAX_BACKOFF_S = 330       # a full 5-minute window + slack
TS_MAX_RETRIES = 6           # per page
TS_AUTH_PAGE_DELAY_S = 1.0   # authenticated limit is 300 req / 5 min

# Reply tree endpoint. Truth Social's paginated descendants endpoint moved
# from /api/v1 to /api/v2 sometime between Apr and Sep 2026 (v1 now returns
# a bare "404 page not found"); it pages with a Link rel="next" header
# carrying an `offset`. truthbrush 0.2.5 still hardcodes v1, so the
# collector fetches replies itself. {id} is the status id.
TS_DESCENDANTS_PATH = "/api/v2/statuses/{id}/context/descendants"
# Incremental refresh uses truthbrush (authenticated, 300 req/5 min) when
# credentials are present; set False (or pass --anonymous) to force the
# paced public walk, e.g. while a new-device login check is pending.
TS_PREFER_AUTH = True
# No-text re-read (`ts-fill-text`, src/collectors/ts_fill_text.py): the cached
# posts with no words of their own (no text, or only the quote fallback
# "RT: <link>") are read again by id, anonymously, at TS_PAGE_DELAY_S with
# the 429 backoff above. A run stops once this many reads in a row fail
# (403 from Cloudflare, 5xx, network), or at once on a 429 that outlasts
# TS_MAX_RETRIES; the next run retries the failed ids after the unread ones.
TS_FILL_MAX_CONSECUTIVE_ERRORS = 3
# Round trip of one status read, for `--estimate` only: a guess, not
# measured. The pacing sleep comes on top.
TS_FILL_EST_REQUEST_S = 0.5


# ── Gap-fill windows ───────────────────────────────────────────────
# The X timeline endpoint returns newest-first. One capped fetch over a
# long gap would keep the newest N tweets, drop everything older, and then
# mark the account current -- a permanent hole (bitten after the May 10 ->
# Sep gap: 9 of 17 accounts were over the cap). collect_user instead walks
# each account's gap oldest first, from its cursor, in windows of this many
# hours. 35 = 11 (mod 24) shares no factor with 24, and 24 windows make
# exactly 5 weeks, so over every 24 windows the window ends sit 7 h apart
# through the week: every clock hour once, 3-4 ends on each weekday. A
# whole-day length locks the weekday and hour of every end, which is the
# May-Sep fault (14-day slices that kept each fortnight's newest day or
# so, mostly a weekend); 36 h only alternates two hours. The planner
# lengthens the windows (to the next length coprime with 24) only when an
# account's backstop is too small for its gap.
GAP_FILL_WINDOW_HOURS = 35
# GET /2/users/:id/tweets has a max_results minimum of 5: a smaller request
# still returns, and bills, up to 5. No request asks for fewer.
X_MIN_READS_PER_REQUEST = 5
# A guard against empty pages that still carry a next_token.
X_MAX_PAGES_PER_WINDOW = 20
# The user timeline reaches back only about this many posts. Leading
# windows that come back empty are reported as past it only when the
# windows after them hold an estimated half of it or more.
X_TIMELINE_HORIZON_POSTS = 3200


# ── X post counts and the full-archive probe ───────────────────────
# GET /2/tweets/counts/all (`x-counts`, src/collectors/x_counts.py): X's own
# count of the posts matching a query, per minute, hour or day, to tell a
# silent window from a failed fetch. Pay-per-use price (docs, 2026-11-01):
# $0.010 per request, whatever it returns. One response covers up to 31 days
# at any granularity, so the planner counts ceil(days / 31) requests per
# query (Feb 1 - Nov 1 is 9); a query may make X_COUNTS_PAGE_SLACK more
# before it stops, in case X pages more finely than that.
X_COUNTS_ALL_COST_USD = 0.010
X_COUNTS_DAYS_PER_PAGE = 31
X_COUNTS_PAGE_SLACK = 1
X_COUNTS_GRANULARITY = "hour"
# A run refuses to start with more requests planned than this, and stops at
# it (the slack can only add one per query). 17 accounts x 9 = 153.
X_COUNTS_MAX_REQUESTS = 300
# counts/all allows 300 requests per 15 minutes; one a second stays clear.
X_COUNTS_REQUEST_DELAY_S = 1.0
# A 429 on counts/all or search/all waits until x-rate-limit-reset (clamped
# to X_429_MAX_BACKOFF_S), or X_429_DEFAULT_BACKOFF_S without the header,
# at most X_429_MAX_RETRIES times per request; then the run stops. tweepy's
# own wait (collect) retries without limit.
X_429_DEFAULT_BACKOFF_S = 60
X_429_MAX_BACKOFF_S = 960
X_429_MAX_RETRIES = 3

# Full-archive probe (`x-archive probe`, src/collectors/x_archive.py): before
# any archive fill is bought, two GET /2/tweets/search/all calls (one page
# of max_results 10, $0.005 per post returned) on whole UTC days where the
# cache holds every post of the probe account, compared id for id, plus two
# counts/all calls (day granularity) for the account with and without
# is:retweet. Maximum $0.12. search/all allows one request a second.
X_ARCHIVE_PROBE_HANDLE = "PeteHegseth"
# (label, start, end), UTC, of the stretches the cache holds whole:
# February was collected take-all; the Sep 13 slice of the 2026-09-18
# collect was uncapped (that collect wrote the cache at 01:48 UTC on Sep
# 18). A probe day must also end before the account's first sampled
# `collect` window in the x_sampling manifests (Sep 17 20:59 for Hegseth).
X_ARCHIVE_PROBE_STRETCHES = [
    ("feb", "2026-02-01T00:00Z", "2026-03-01T00:00Z"),
    ("sep", "2026-09-13T00:00Z", "2026-09-18T01:48Z"),
]
X_ARCHIVE_PROBE_MIN_POSTS = 6
X_ARCHIVE_PROBE_MAX_POSTS = 10  # one page of max_results 10 holds the day
X_ARCHIVE_PROBE_COUNTS_RANGE = ("2026-02-01T00:00Z", "2026-03-03T00:00Z")
X_ARCHIVE_SEARCH_MAX_RESULTS = 10  # search/all's minimum
X_ARCHIVE_SEARCH_DELAY_S = 1.1
# Verdict thresholds: the share of the cached posts a probe day gets back.
X_ARCHIVE_PROBE_RECALL_OK = 0.8
X_ARCHIVE_PROBE_RECALL_LOW = 0.5


# ── Stance-model experiments (GPU Jobs on dkbl2, see k8s/README.md) ─
MODELS_DIR = DATA_DIR / "models"

# ── Off-box backup (`backup` command) ───────────────────────────────
# S3 bucket from homelab-infra terraform/iran-sentiment-backups.tf; the bucket
# name and the append-only writer's key live in .env
# (IRAN_BACKUP_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY). Nothing
# is scheduled: run it after a paid collect / relabel. (local dir, S3 prefix,
# storage class): raw and the paid labels are small and irreplaceable; the
# models are 8 GB and regenerable, so they go straight to Glacier IR.
BACKUP_SYNC = [
    (RAW_DIR, "raw/", "STANDARD"),
    (PROCESSED_DIR, "processed/", "STANDARD"),
    (MODELS_DIR, "models/", "GLACIER_IR"),
]
AWS_BIN = Path.home() / ".local" / "bin" / "aws"
# Teacher check: relabel a stratified sample with a stronger model and
# measure disagreement with the Haiku labels before distilling from them.
TEACHER_CHECK_MODEL = "claude-opus-5"
TEACHER_CHECK_N = 500
# Opus 5 thinks by default and thinking tokens count against max_tokens:
# the Haiku 200-token cap truncated its JSON (seen 2026-09-18). Low effort
# keeps the thinking short for a one-line classification.
TEACHER_MAX_TOKENS = 1024
TEACHER_EFFORT = "low"
# Claude API prices per model, $ per million tokens, and what multiplies
# them (platform.claude.com pricing, fetched 2026-11-01): a 5-minute cache
# write costs cache_write_mult x input, a read cache_read_mult x input, and
# the Batch API price is list x batch_discount; the multipliers stack.
# cache_min_tokens is the shortest prefix the model caches. Every estimator
# and spend tally prices a model by its own row (pricing.model_price) and
# refuses a model with none, so a new model is never priced at another's.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5": {
        "input": 5.0,
        "output": 25.0,
        "cache_write_mult": 1.25,
        "cache_read_mult": 0.10,
        "batch_discount": 0.5,
        "cache_min_tokens": 512,
    },
    "claude-opus-5-5": {
        "input": 4.0,
        "output": 20.0,
        "cache_write_mult": 1.25,
        "cache_read_mult": 0.05,
        "batch_discount": 0.5,
        "cache_min_tokens": 512,
    },
    "claude-fable-5-1": {
        "input": 10.0,
        "output": 50.0,
        "cache_write_mult": 1.25,
        "cache_read_mult": 0.025,
        "batch_discount": 0.5,
        "cache_min_tokens": 512,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_write_mult": 1.25,
        "cache_read_mult": 0.10,
        "batch_discount": 0.5,
        "cache_min_tokens": 4096,
    },
}
# The score_<tag> / label_<tag> suffix a model's labels get (relabel merge).
# These two keep their historical names; every other model is its family
# and version (claude-opus-5-5 -> opus55, claude-fable-5-1 -> fable51), so
# no two models share a column.
MODEL_TAGS = {"claude-opus-5": "opus", "claude-haiku-4-5": "haiku"}
# Columns already in the data whose model no record names: a merge writes
# score_opus only from claude-opus-5 (sentiment_all_columns.json records
# the rest as they are made).
SCORE_COLUMN_MODELS = {"score_opus": "claude-opus-5"}
# A batch collect streams the results; when the stream breaks (httpx
# ReadError on this host, 2026-09-23 and 2026-10-05) it downloads the
# batch's results_url to a file instead, resuming a cut download with a
# Range request, up to this many retries with a growing pause.
BATCH_RESULTS_DOWNLOAD_RETRIES = 6
BATCH_RESULTS_READ_TIMEOUT_S = 300
# Dry-run estimates (`--estimate`) count no tokens through the API: a prompt's
# input tokens are its characters over this. relabel measured 217 input
# tokens per X post on Opus 5 (2026-09-18), whose llm_prompt averages 460
# characters: 2.1. Rounded down so the estimate errs high; output tokens
# use relabel.EST_OUT_TOKENS (86, thinking included).
TEACHER_EST_CHARS_PER_TOKEN = 2.0
# Reply relabel v2 (`reply-teacher-check --v2`, 2026-10-04). The v1 reply
# labels (teacher_labels_replies_<model>.parquet) were made from the first
# 100 characters of 423 replies, with the broadcaster prompt and no sight of
# the Trump post replied to; v2 sends the full reply and the post. v1 stays
# as it is. A run refuses to start with more calls to make than the cap
# (1,157 sampled replies had text on 2026-10-04).
REPLY_TEACHER_V2_MAX_CALLS = 1300
# Measured 2026-10-04: what the 16-call `reply-teacher-check --v2 --limit 16`
# pilot was billed (its Spend log line) on Opus 5, and the characters of the
# 16 prompts it sent: ≈ $0.069 at direct-API prices, $0.0043 a call. The
# prefix cache engaged for one parent post of eight (the rest fell under the
# cache minimum), so the estimate that caches every shared prefix was too
# low; the prompts ran 2.8 characters a token, so the characters / 2 guess
# is high. `--estimate` also prices a run at these rates (input per prompt
# character, output per call), direct and at batch prices.
TEACHER_V2_PILOT = {
    "calls": 16, "prompt_chars": 26_446,
    "input_tokens": 5_359, "cache_creation_input_tokens": 2_067,
    "cache_read_input_tokens": 2_067, "output_tokens": 1_119,
}
# Which reply labels `reply-population` (and so export-web) weights: "v1"
# or "v2". Flip only once the v2 run is complete.
REPLY_TEACHER_LABELS_VERSION = "v2"
# Trump-feed check (`teacher-check --source trump`): Opus labels on a fixed
# random sample of Trump's Truth Social posts with text, against the
# distilled scorer used on his feed. Its 0.88 correlation with Opus was
# measured on held-out X posts, not on the feed.
TRUMP_TEACHER_CHECK_N = 400
# Raised from 500 on purpose (2026-11-02) so `teacher-check --source trump
# --all` can put his whole feed on score_opus: 3,222 posts with text after
# the Nov 1 pull, 400 already labelled by the check (reused, never re-sent),
# so 2,822 to do; 3,300 covers every post with text, with room for ~80 new
# ones, and still refuses anything past the feed.
TRUMP_TEACHER_CHECK_MAX_CALLS = 3300
TEACHER_SAMPLE_SEED = 42           # --limit pilot order and the Trump sample
# Reply labels beyond v2 (`reply-label`, src/analysis/reply_labels.py): v2b
# (v2 without its praise rule) and cat-v2 (the six reply categories with the
# parent post) on the stance sample reuse REPLY_TEACHER_V2_MAX_CALLS; the
# fresh random sample (`reply-fresh-sample`, then `reply-label --job
# v2-fresh --post <slug>`, one batch a post) has its own caps, sized to one
# run of REPLY_FRESH_PER_POST replies for each of the eight posts. They
# bound the whole run: a submit refuses when the calls its earlier batches
# sent plus its own are over the call cap, or their billed spend plus its
# ceiling (batch prices, no cache hits: every request on a cacheable post
# writes the prefix) is over the dollar cap; the draw refuses more than
# REPLY_FRESH_PER_POST a post or a manifest whose run would pass either cap.
REPLY_FRESH_PER_POST = 1000
REPLY_FRESH_SEED = 20261102
REPLY_FRESH_DRAWS_NAME = "reply_sample_draws_fresh.parquet"  # under PROCESSED_DIR
REPLY_FRESH_MAX_CALLS = 8500
# The 7,995-call run's ceiling is $20.21 on claude-opus-5 (expected $14.90):
# this leaves the ~6% slack the call cap has, for resubmitted failures.
REPLY_FRESH_MAX_USD = 21.5
# Whether reply-population (and so export-web) pools the fresh draws' v2
# labels with the first draws' once they exist: both its direct (opus_*) and
# its model-assisted columns use every labelled random draw. A post's fresh
# draws join only once this share of them carry a label (before its batch
# is collected, the only labelled ones are reused flipper labels, not a
# random part). Set False to reproduce the 1,071-draw figures.
REPLY_POPULATION_FRESH_DRAWS = True
REPLY_FRESH_POOL_MIN_SHARE = 0.98
# Measured on the 2026-10-05 v2 batch (1,141 requests, all end_turn, billed
# $2.10): input tokens per request = a per-post prefix (the instructions,
# the parent post and the reply wrapper; least squares with one slope) +
# 0.350 per reply character. The cached block is the prefix less about 14
# tokens of wrapper (ceasefire 734, hold-off 765, strikes 568 tokens read or
# written); only those three reached the 512-token minimum, and 396 of their
# 443 requests read the cache (the rest wrote it). Output 67.8 tokens a
# request. Another prompt's prefix is the post's v2 prefix plus its extra
# instruction characters at 2.67 a token (v2's instruction prose). A request
# carries a cache breakpoint only where its post's block reaches the
# model's minimum, so the ceiling is a true bound.
REPLY_V2_BATCH_USAGE = {
    "model": "claude-opus-5",
    "prefix_tokens": {
        "power_plant_day": 460.1,
        "civilisation_dies": 511.9,
        "deal_complete": 466.9,
        "ceasefire": 747.8,
        "hold_off_attack": 776.9,
        "strikes_resume_sep": 584.8,
        "these_fools": 443.3,
        "trump_strait": 433.5,
    },
    "wrapper_tokens": 14.0,
    "tokens_per_reply_char": 0.350,
    "output_tokens": 67.8,
    "cache_hit_share": 396 / 443,
    "instruction_chars_per_token": 2.67,
}
# Output tokens a stance answer, measured per model (a batch's billed output
# / calls). A bare figure is a large batch's mean (Opus 5's: 1,141 calls),
# used for the expected cost and the ceiling alike: over a run of hundreds
# or thousands of calls the mean is what bounds the output, not one call's
# p95. A model measured by a pilot batch is recorded as {"mean": ..., "p95":
# ...} from the collect's output line, and the ceiling takes the p95. A job
# with a dollar cap sends a model without a figure here no more than
# REPLY_UNMEASURED_PILOT_MAX_CALLS requests in all (a --limit pilot, its
# earlier batches of the same prompt and model counted): its estimate rests
# on Opus 5's length, at REPLY_UNMEASURED_CEILING_MULT times it at the
# ceiling, so the cap would bound a guess.
REPLY_OUTPUT_TOKENS: dict[str, float | dict[str, float]] = {
    "claude-opus-5": REPLY_V2_BATCH_USAGE["output_tokens"]
}
REPLY_UNMEASURED_PILOT_MAX_CALLS = 50
REPLY_UNMEASURED_CEILING_MULT = 2.0
# max_tokens of a reply-label batch request. Thinking counts against it, so
# a submit refuses once an earlier batch of the job on that model at this
# value had an answer cut at max_tokens (it would be cut again): raise it.
REPLY_LABEL_MAX_TOKENS = TEACHER_MAX_TOKENS
# cat-v2 answers with a category name and a reason instead of a score and a
# label: unmeasured, put at v2's output plus 15 tokens until a batch says.
REPLY_CAT_EST_OUTPUT_TOKENS = 83.0
# The second rater (`reply-label --job v2-rater`, `post-label --job rater`):
# the same prompt as the labels of record on another model, in its own files
# and columns (score_opus55), never score_opus or the v2 reply labels. Opus
# 5.5, not Fable 5.1 (the user's call, 2026-11-02). The reply arm's dollar
# cap bounds its whole run (every batch, pilot included) and covers the
# 1,157 v2 replies at the no-call ceiling, output at twice Opus 5's 67.8
# tokens while unmeasured (thinking is always on): $3.14.
SECOND_RATER_MODEL = "claude-opus-5-5"
SECOND_RATER_REPLY_MAX_USD = 4.0

# ── Alternate post raters (`post-label`, src/analysis/post_labels.py) ──
# Labels on broadcaster posts beyond score_opus: one path whose parameters
# are the model, the prompt version and the post set, each job in its own
# label file (post_labels_<job>_<model>.parquet) and columns, joined onto a
# frame when read (`phases --score`, post_labels.attach), never merged into
# sentiment_all.parquet or score_opus. The war candidates are the X
# accounts' posts with text (search rows out) and Trump's feed flagged by
# the keyword pattern OR the Haiku topic label (TOPIC_SOURCE "either").
#   stance-topic           a topic class plus the author's position on U.S.
#                          military action against Iran (null when not about
#                          Iran), with the author's handle, on the candidates
#                          and POST_RANDOM_N unflagged X posts, which only
#                          measure what the filters miss
#   rater                  llm_prompt (the score_opus prompt) on
#                          SECOND_RATER_MODEL, on the candidates
#   stance-topic-nohandle  stance-topic without the handle, on a seeded paired
#                          subsample of the X candidates: every post of the
#                          POST_NOHANDLE_ALL_TIERS, POST_NOHANDLE_PER_TIER of
#                          each of the POST_NOHANDLE_TIERS
# The random posts and the subsample are the smallest seeded hashes of their
# ids, recorded in a manifest by the first pilot or submit (never redrawn).
POST_LABEL_MAX_TOKENS = TEACHER_MAX_TOKENS  # thinking counts: no max_tokens stop in a pilot or batch
POST_LABEL_EFFORT = TEACHER_EFFORT
POST_RANDOM_N = 1500
POST_RANDOM_SEED = 20261102
POST_NOHANDLE_ALL_TIERS = ("religious_authority", "opposition")
POST_NOHANDLE_TIERS = ("admin", "maga_prowar", "maga_antiwar", "media")
POST_NOHANDLE_PER_TIER = 400
POST_NOHANDLE_SEED = 20261103
# `post-label pilot`: direct calls on the first N still to do (dealt across
# sets, sources and tiers) that record the billed input and output tokens
# and the stop reasons, which the estimate is then calibrated on. Only a
# pilot of the job's current prompt version, POST_LABEL_MAX_TOKENS and
# POST_LABEL_EFFORT counts. Batches of a model and prompt whose output
# length is unmeasured (no such pilot of POST_PILOT_MIN_BILLED billed
# answers, and not llm_prompt on claude-opus-5) send no more than
# POST_PILOT_MAX_CALLS in all, and no batch goes after such a pilot, or an
# earlier batch of the same request, had an answer stop at max_tokens.
POST_PILOT_N = 50
POST_PILOT_MAX_CALLS = 100
# A pilot counts as the measurement (its p95 the ceiling's output, the
# unmeasured gate open) only once it has this many billed answers, pooled
# over the job's pilots of the current request: a p95 of a handful says
# little, and the ceiling is what the dollar cap is checked against. A
# smaller pilot still refuses every batch after a max_tokens stop.
POST_PILOT_MIN_BILLED = 40
# A pilot saves its labels and its running usage every this many answers
# (its record is written before the first call, marked in_progress at its
# direct ceiling), so a run killed outright keeps what it paid for and still
# counts against the job's caps.
POST_PILOT_SAVE_EVERY = 10
# Caps for each job's whole run: every call it has made (pilots and
# batches, on any model) against the call cap, and their spend (pilots at
# direct prices, collected batches as billed, one not collected yet at its
# ceiling) plus the next batch's no-call ceiling against the dollar cap. The
# ceiling is at batch prices (input from the fit below, scaled by the
# pilot's billed/estimated input once there is one, with no cache hits:
# every request writes the stance-topic instructions; output at the pilot's
# p95 a call, or twice the guess while unmeasured); a pilot refuses too when
# it would pass either cap. Calls sized 2026-11-02 to one run with ~10% slack:
# stance-topic 4,925 candidates (X 4,479, Trump 446) + 1,500 random = 6,425;
# rater 4,925; nohandle 1,958 (272 religious, 86 opposition, 4 x 400). The
# dollar caps hold that run's ceiling at a pilot p95 of up to ~160 output
# tokens for the stance-topic jobs (expected ~$13.8 and ~$4.2 at the 110
# guess with 89% cache reads) and ~350 for the rater on Opus 5.5 (expected
# ~$10.8 at twice Opus 5's 84).
POST_STANCE_TOPIC_MAX_CALLS = 7000
POST_STANCE_TOPIC_MAX_USD = 30.0
POST_RATER_MAX_CALLS = 5400
POST_RATER_MAX_USD = 20.0
POST_NOHANDLE_MAX_CALLS = 2150
POST_NOHANDLE_MAX_USD = 9.5
# Input tokens of an llm_prompt request on claude-opus-5, least squares on
# billed usage against prompt characters, and its output tokens a call:
# X posts from the 2026-10-05 relabel batch (2,575 posts), Trump's feed from
# the 2026-10-05/06 check batches (398); output's mean and p95 a call from
# the same batches. A longer prompt adds its extra instruction characters at
# instruction_chars_per_token (v2's instruction prose, the 2026-10-05 reply
# batch). Another model is priced at its own row of MODEL_PRICES on these
# counts until a pilot measures it.
POST_PROMPT_TOKEN_FIT = {
    "model": "claude-opus-5",
    "x": {
        "tokens_per_char": 0.3365,
        "fixed_tokens": 51.0,
        "output_tokens": 84.0,
        "output_p95_tokens": 113.0,
    },
    "trump": {
        "tokens_per_char": 0.3628,
        "fixed_tokens": 35.0,
        "output_tokens": 77.0,
        "output_p95_tokens": 104.0,
    },
    "instruction_chars_per_token": 2.67,
}
# Output tokens a call guessed for a job until its pilot: stance-topic
# answers a topic, a score and a reason (96-110 by the 2026-11-01 sizing);
# another model than claude-opus-5 is guessed at twice the figure (Opus 5.5
# always thinks).
POST_OUTPUT_TOKENS_GUESS = {"stance-topic": 110.0}
# Distillation: fine-tune an encoder to regress score_llm.
DISTILL_BASE_MODEL = "roberta-large"
DISTILL_EPOCHS = 3
DISTILL_HOLDOUT = 0.2
DISTILL_BATCH_SIZE = 16
DISTILL_LR = 2e-5
DISTILL_MAX_LEN = 128
DISTILL_SEED = 42
# Distillation sweep (stance-sweep Job, ~2-3 h on the 3080): each recipe is
# fit on the same per-tier split as `distill`; the best by holdout Pearson
# gets DISTILL_CV_FOLDS-fold cross-validation for error bars and a final
# fit on ALL labels -> MODELS_DIR/stance_distilled_final.
DISTILL_SWEEP = [
    {"name": "rl-128-2e5-3",  "base_model": "roberta-large", "max_len": 128, "lr": 2e-5, "epochs": 3},
    {"name": "rl-128-1e5-4",  "base_model": "roberta-large", "max_len": 128, "lr": 1e-5, "epochs": 4},
    {"name": "rl-256-2e5-3",  "base_model": "roberta-large", "max_len": 256, "lr": 2e-5, "epochs": 3},
    {"name": "rl-128-2e5-5",  "base_model": "roberta-large", "max_len": 128, "lr": 2e-5, "epochs": 5},
    # DeBERTa-v3-large (435M params + a 128k-token embedding) does not fit
    # the 10 GB 3080 with fp32 AdamW states even at batch 4: 8-bit AdamW
    # (bitsandbytes) + gradient checkpointing + accumulation to an effective
    # batch of 16.
    {"name": "deb-128-1e5-3", "base_model": "microsoft/deberta-v3-large", "max_len": 128, "lr": 1e-5, "epochs": 3,
     "batch_size": 8, "grad_accum": 2, "optim": "adamw_bnb_8bit", "gradient_checkpointing": True},
    {"name": "deb-256-1e5-3", "base_model": "microsoft/deberta-v3-large", "max_len": 256, "lr": 1e-5, "epochs": 3,
     "batch_size": 4, "grad_accum": 4, "optim": "adamw_bnb_8bit", "gradient_checkpointing": True},
    {"name": "twr-128-3e5-4", "base_model": "cardiffnlp/twitter-roberta-base-sentiment-latest", "max_len": 128, "lr": 3e-5, "epochs": 4},
    {"name": "rb-128-3e5-4",  "base_model": "roberta-base", "max_len": 128, "lr": 3e-5, "epochs": 4},
]
DISTILL_CV_FOLDS = 5

# Reply-domain scorer (`reply-distill`, 2026-10-06). The posts-only scorer of
# record never sees the Trump post a reply answers, and the v2 labels depend
# on it (a reply cheering a ceasefire is against the war): 0.58 Pearson
# against v2 overall, 0.32 under the ceasefire post, 0.14 under the deal
# post. It is fine-tuned, warm from its own weights, on the labelled replies
# with the reply alone ("text", what it sees today) or with the pair (reply,
# parent post) ("ctx"), and judged leave-one-post-out. Base model: a dir
# under MODELS_DIR, whose published reply scores are REPLY_DISTILL_BASE_COL.
REPLY_DISTILL_BASE = "stance_distilled_final_score_opus"
REPLY_DISTILL_BASE_COL = "score_opus_distilled"
REPLY_DISTILL_VARIANTS = ("text", "ctx")
# "text" runs at the base model's training length (recipe.txt: 256); "ctx"
# needs room for the parent post too (the eight run 55-244 tokens, 90% of
# replies under 76). The pair is cut only on the post; a reply too long to
# leave this many tokens of its post is cut first.
REPLY_CTX_MAX_LEN = 384
REPLY_CTX_MIN_PARENT_TOKENS = 32
# Few epochs at a small LR from a fitted model. Effective batch 16 as
# batch 4 x accumulation 4 with the base recipe's 8-bit AdamW and gradient
# checkpointing (deb-256-1e5-3: DeBERTa-v3-large does not fit the 10 GB
# 3080 otherwise).
REPLY_DISTILL = {"epochs": 3, "lr": 1e-5, "batch_size": 4, "grad_accum": 4,
                 "optim": "adamw_bnb_8bit", "gradient_checkpointing": True}
REPLY_DISTILL_SEEDS = 2            # LOPO seeds: DISTILL_SEED, DISTILL_SEED + 1, ...
REPLY_SCORE_BATCH = 32             # inference batch at up to 384 tokens
REPLY_LOPO_DIR = "reply_ctx_lopo"  # under MODELS_DIR: results JSON + held-out predictions
# Final fits go to a stance_distilled_final* dir: never overwritten, and
# sync-data.sh carries those whole.
REPLY_FINAL_DIR = "stance_distilled_final_reply_{variant}"

# Local LLM: same prompt as the Claude scorer, run on the GPU in 4-bit.
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_LLM_EVAL_N = 1000
LOCAL_LLM_BATCH = 16
LOCAL_LLM_MAX_NEW_TOKENS = 80


# ── Sentiment analysis ─────────────────────────────────────────────
# Transformer model name (HuggingFace hub)
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Batch size for RoBERTa inference. Lowered from 32 to 16 after the
# mini PC powered off mid-run on the 8k-tweet dataset — likely thermal
# from sustained 100% CPU. Smaller batches + thread cap keep load down.
ROBERTA_BATCH_SIZE = 16

# How often to force garbage collection during inference
GC_EVERY_N_BATCHES = 10

# Persist partial RoBERTa results every N batches so a mid-phase crash
# only loses up to this many batches of work (a full re-run on 8k posts
# takes ~30 min on this hardware — recovering is way more costly than
# a small write).
ROBERTA_CHECKPOINT_EVERY_N_BATCHES = 25

# Hard memory ceiling for the analyze process. If RSS exceeds this in
# the RoBERTa loop, we checkpoint and abort cleanly so the next run
# can resume. Set well above the observed ~925 MB model footprint to
# leave headroom for tokenizer activations.
ROBERTA_MAX_RSS_MB = 6144

# Cap PyTorch threads to keep CPU load below thermal-trip threshold on
# small fanless / passive-cooled hardware. RoBERTa inference is mostly
# matmul; 2 threads is a sweet spot vs runtime on this machine.
TORCH_NUM_THREADS = 2

# LLM-based scoring (optional)
LLM_MODEL = "claude-haiku-4-5-20251001"


# ── Truth Social web-client credentials ────────────────────────────
# These are NOT user credentials — they are the client_id / client_secret
# extracted from Truth Social's own public web-app JS bundle. The upstream
# `truthbrush` library hardcodes the same values. They only work when
# paired with a real user username/password (loaded from .env), so they
# don't grant any access on their own. Kept here (vs inline in the
# collector) so secret scanners don't flag what looks like a
# CLIENT_SECRET literal in application code.
TRUTH_SOCIAL_WEB_CLIENT_ID = "9X1Fdd-pxNsAgEDNi_SfhJWi8T-vLuV2WVzKIbkTCw4"
TRUTH_SOCIAL_WEB_CLIENT_SECRET = "ozF8jzI4968oTKFkEnsBC-UbLPCdrSv0MkXGQu2o_-M"

# New-device security-code flow (`python -m src.cli ts-login`), read from
# the web app on 2026-09-16 (bundle index-BbOBI-WJ.js + chunk
# sign-in-modal-Bqv85TDn.js):
#   1. POST /oauth/v2/token (password grant) -> 403 security_code_required
#      with challenge_id + supported_delivery_methods
#   2. POST /oauth/v2/choose_delivery_method
#      {username, challenge_id, delivery_method: "email"|"sms"}  (no creds)
#   3. POST /oauth/v2/verify_security_code with the password-grant body
#      + challenge_id + security_code -> access_token
# Every call to step 1 mints a NEW challenge, so steps 2-3 must use the
# challenge_id from the same run.
TS_SECURITY_CODE_DELIVERY_ENDPOINT = "/oauth/v2/choose_delivery_method"
# Headers the web app sends on auth calls; the device check may key on them.
TS_AUTH_HEADERS = {"Browser": "Chrome", "OS": "Linux"}

# LLM calls are I/O-bound (waiting on Anthropic API) — modest
# concurrency cuts runtime ~5x with zero CPU/memory pressure. Raise
# if the Anthropic tier allows; lower if rate limits bite.
LLM_CONCURRENCY = 5

# Persist sentiment_all.parquet every N LLM completions so a mid-run
# crash only loses the last batch of in-flight scores, not the whole
# run (a full LLM pass on 8k posts is a ~$12 / 30-min job).
LLM_SAVE_EVERY_N = 100


# ── dkweb export (`export-web`) ─────────────────────────────────────
# Chart JSON for the blog post on dankeogh.com (repo ~/repos/dkweb, Astro +
# Observable Plot rendered at build time). The post folder holds the MDX and
# these files side by side.
WEB_EXPORT_DIR = Path.home() / "repos" / "dkweb" / "src" / "content" / "blog" / "iran-war-stance"
WEB_WEEKLY_TIERS = ["maga_prowar", "maga_antiwar", "admin"]
WEB_SMOOTH_WEEKS = 3          # centred, post-weighted rolling window for the weekly chart
WEB_MIN_WINDOW_POSTS = 15     # windows with fewer war posts are dropped
WEB_MIN_ACCOUNT_POSTS = 20    # accounts with fewer war posts are left off the strip
WEB_TIER_LABELS = {
    "admin": "Administration", "maga_prowar": "Pro-war MAGA", "maga_antiwar": "Anti-war MAGA",
    "opposition": "Opposition", "media": "Media", "religious_authority": "Vatican & bishops",
}
WEB_POST_LABELS = {
    "power_plant_day": "\u201cPower Plant Day\u201d", "civilisation_dies": "\u201cA whole civilisation will die\u201d",
    "ceasefire": "Two-week ceasefire", "hold_off_attack": "\u201cHold off\u201d on the attack",
    "deal_complete": "\u201cThe Deal is complete\u201d", "strikes_resume_sep": "Strikes near Hormuz",
    "these_fools": "\u201cThese fools\u201d", "trump_strait": "\u201cTRUMP STRAIT\u201d",
}
WEB_EVENTS = [
    ("2026-02-28", "Strikes"), ("2026-04-08", "Ceasefire"), ("2026-06-17", "Islamabad MOU"),
    ("2026-07-08", "MOU collapses"), ("2026-08-17", "MOU expires"), ("2026-09-01", "Strikes resume"),
]


# ── Plotting ───────────────────────────────────────────────────────
# Colors used to encode tiers in the tier comparison plot
TIER_COLORS = {
    "admin": "#d62728",                 # red
    "maga_prowar": "#ff7f0e",           # orange
    "maga_antiwar": "#2ca02c",          # green
    "opposition": "#1f77b4",            # blue
    "media": "#9467bd",                 # purple
    "search": "#17becf",                # cyan (public sentiment proxy)
    "religious_authority": "#bcbd22",   # olive (Pope/Vatican/USCCB)
}

# Rolling window for smoothing time-series plots (pandas offset alias)
PLOT_ROLLING_WINDOW = "1D"
PLOT_SMOOTH_DAYS = 5


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    for d in (
        DATA_DIR, RAW_DIR, PROCESSED_DIR, FIGURES_DIR,
        X_RAW_DIR, TRUTH_SOCIAL_RAW_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
