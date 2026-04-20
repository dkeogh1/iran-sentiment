"""
Runtime settings for the iran-sentiment pipeline.

All tunables (paths, budget caps, batch sizes, model names) live here so
that scripts can stay thin and data-driven.
"""

from datetime import datetime
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
COLLECTION_END = datetime(2026, 4, 21)


# ── Budget caps (per-account) ──────────────────────────────────────
# X charges ~$0.005/read on pay-as-you-go. These caps guard against a
# single prolific account (e.g. @marklevinshow posts 35+ times/day)
# blowing through the budget.
MAX_TWEETS_PER_USER = 500       # ≈ $2.50 cap per user
MAX_TWEETS_PER_SEARCH = 150     # ≈ $0.75 cap per search term
X_READ_COST_USD = 0.005


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
