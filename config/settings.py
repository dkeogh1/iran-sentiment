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
# "RT @x: " are removed (src/text_rules.py), carry no text to judge: image /
# video only or a ReTruth stored before 2026-10-04 on Truth Social, a bare
# t.co link on X. They are not scored and leave every share and mean,
# rather than counting as off-topic. Same cut as the reply sampler's
# media_only. 1,622 of Trump's 4,360 Truth Social posts fell under it on
# 2026-09-30 (the distilled model had scored each one a constant +0.111, as
# it did the image-only replies).
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
MAX_TWEETS_PER_USER = 500       # ≈ $2.50 cap per user
MAX_TWEETS_PER_SEARCH = 150     # ≈ $0.75 cap per search term
X_READ_COST_USD = 0.005

# Per-account overrides of MAX_TWEETS_PER_USER for a run. The point is to
# SAMPLE high-volume accounts (Levin, Loomer, Jones post 35+/day) rather
# than let the cap silently truncate them. Empty = every account uses the
# default cap.
ACCOUNT_CAP_OVERRIDES: dict[str, int] = {
    # 2026-09-18 May 10 -> Sep 18 refresh: the five accounts projected far
    # over the cap are sampled at ~450 (spread across ~10 two-week slices).
    # The rest ran at 500, but split_cap gives every slice an equal share,
    # so a busy fortnight is cut there too: 8 more accounts hit a slice cap
    # (collect_2026-09-18.log: @POTUS, @PeteHegseth, @VP, @VaticanNews in 9
    # slices, @RealCandaceO 7, @WhiteHouse 6, @mtgreenee and @Pontifex 1).
    # A capped slice keeps its newest posts.
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


# ── Gap-fill slicing ───────────────────────────────────────────────
# The X timeline endpoint returns newest-first. One capped fetch over a
# long gap would keep the newest N tweets, drop everything older, and then
# mark the account current -- a permanent hole (bitten after the May 10 ->
# Sep gap: 9 of 17 accounts were over the cap). collect_user instead walks
# the gap oldest-slice-first in windows of this many days, giving each
# slice an even share of the account's cap, so a cap hit costs a slice its
# older days (the endpoint is newest-first: a capped slice keeps its last
# day or two) instead of deleting months.
GAP_FILL_SLICE_DAYS = 14


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
REPLY_TEACHER_LABELS_VERSION = "v1"
# Trump-feed check (`teacher-check --source trump`): Opus labels on a fixed
# random sample of Trump's Truth Social posts with text, against the
# distilled scorer used on his feed. Its 0.88 correlation with Opus was
# measured on held-out X posts, not on the feed.
TRUMP_TEACHER_CHECK_N = 400
TRUMP_TEACHER_CHECK_MAX_CALLS = 500
TEACHER_SAMPLE_SEED = 42           # --limit pilot order and the Trump sample
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
