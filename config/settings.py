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
SENTIMENT_OUTPUT = PROCESSED_DIR / "sentiment_all.parquet"

# Scored Truth Social reply data (separate from broadcaster sentiment)
REPLY_SENTIMENT_OUTPUT = PROCESSED_DIR / "reply_sentiment.parquet"


# ── Date window for data collection / analysis ────────────────────
COLLECTION_START = datetime(2026, 2, 1)
COLLECTION_END = datetime(2026, 4, 10)


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

# Batch size for RoBERTa inference. Keep small to cap peak memory —
# we OOM'd a mini PC on a prior run without batching.
ROBERTA_BATCH_SIZE = 32

# How often to force garbage collection during inference
GC_EVERY_N_BATCHES = 10

# LLM-based scoring (optional)
LLM_MODEL = "claude-haiku-4-5-20251001"


# ── Plotting ───────────────────────────────────────────────────────
# Colors used to encode tiers in the tier comparison plot
TIER_COLORS = {
    "admin": "#d62728",          # red
    "maga_prowar": "#ff7f0e",    # orange
    "maga_antiwar": "#2ca02c",   # green
    "opposition": "#1f77b4",     # blue
    "media": "#9467bd",          # purple
    "search": "#17becf",         # cyan (public sentiment proxy)
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
