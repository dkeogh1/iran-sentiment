"""
Runtime settings for the iran-sentiment pipeline.

All tunables (paths, budget caps, batch sizes, model names) live here so
that scripts can stay thin and data-driven.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
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


# ── Date window for data collection / analysis ────────────────────
COLLECTION_START = datetime(2026, 2, 1)
# End-of-window is "now" (evaluated at CLI startup), with a 30s buffer
# because the X API rejects end_time values that are too close to the
# current request time. Naive UTC to match COLLECTION_START's shape.
COLLECTION_END = (
    datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    - timedelta(seconds=30)
)


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
    # over the cap are sampled at ~450 (spread across ~10 two-week slices);
    # everything else fits under 500 and is fetched in full.
    "LauraLoomer": 450,
    "marklevinshow": 450,
    "RealAlexJones": 450,
    "BarakRavid": 450,
    "StateDept": 450,
}

# Hard ceiling on one `collect` run. The CLI prints the per-account plan
# and refuses to start if the plan's maximum spend exceeds this. Set it to
# the X credit you are actually willing to burn (balance was $4.40 on
# 2026-09-16); raise it deliberately, per run.
X_RUN_BUDGET_USD = 45.00   # raised 2026-09-18 after a top-up for the May->Sep refresh


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
# slice an even share of the account's cap, so a cap hit thins a slice
# instead of deleting months.
GAP_FILL_SLICE_DAYS = 14


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
