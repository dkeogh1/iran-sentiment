"""
Visualization module — time-series sentiment plots with event overlays.

Every public plot function takes a DataFrame + score column and returns
a matplotlib Figure. Saving is controlled by the `save` flag so the same
functions can be used from notebooks without writing to disk.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from config import settings
from config.timeline import EVENTS, ANALYSIS_START, ANALYSIS_END

logger = logging.getLogger(__name__)

# Colors used to mark key events by category
EVENT_CATEGORY_COLORS = {
    "military": "#8b0000",
    "diplomatic": "#006400",
    "political": "#00008b",
    "media": "#4b0082",
    "protest": "#ff4500",
}


# ── Helpers ─────────────────────────────────────────────────────────

def _fig_path(name: str) -> Path:
    settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return settings.FIGURES_DIR / name


def _add_event_markers(ax: plt.Axes, y_top: float = 0.6) -> None:
    """Overlay vertical event markers on a time-series axis."""
    for event in EVENTS:
        if not (ANALYSIS_START <= event.date <= ANALYSIS_END):
            continue
        color = EVENT_CATEGORY_COLORS.get(event.category, "gray")
        ts = pd.Timestamp(event.date, tz="UTC")
        ax.axvline(ts, color=color, alpha=0.4, linestyle=":", linewidth=1)
        ax.annotate(
            event.label,
            xy=(ts, y_top),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=6,
            rotation=90,
            ha="center",
            va="bottom",
            color=color,
            alpha=0.8,
        )


def _format_date_axis(ax: plt.Axes, fig: plt.Figure) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=45)


# ── 1. Overall timeline ────────────────────────────────────────────

def plot_sentiment_timeline(
    df: pd.DataFrame,
    score_col: str = "score_vader",
    title: str | None = None,
    by_user: bool = False,
    save: bool = True,
) -> plt.Figure:
    """
    Time-series of sentiment with a volume panel underneath and
    vertical markers for key events.
    """
    fig, (ax_sent, ax_vol) = plt.subplots(
        2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True
    )
    fig.suptitle(title or f"Iran War Sentiment ({score_col})", fontsize=16, fontweight="bold")

    df = df.dropna(subset=[score_col]).copy()
    df_idx = df.set_index("created_at")

    if by_user:
        for user, group in df_idx.groupby("user"):
            daily = group[score_col].resample(settings.PLOT_ROLLING_WINDOW).mean()
            ax_sent.plot(daily.index, daily.values, alpha=0.7, label=f"@{user}", linewidth=1.5)
        ax_sent.legend(loc="upper left", fontsize=8, ncol=2)
    else:
        daily = df_idx[score_col].resample(settings.PLOT_ROLLING_WINDOW).mean()
        smoothed = daily.rolling(3, center=True, min_periods=1).mean()
        ax_sent.fill_between(daily.index, daily.values, alpha=0.15, color="steelblue")
        ax_sent.plot(smoothed.index, smoothed.values, color="steelblue", linewidth=2, label="Rolling mean")

    ax_sent.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_sent.set_ylabel("Sentiment Score")
    ax_sent.set_ylim(-1.05, 1.05)
    _add_event_markers(ax_sent, y_top=ax_sent.get_ylim()[1])

    # Volume panel
    volume = df_idx[score_col].resample(settings.PLOT_ROLLING_WINDOW).count()
    ax_vol.bar(volume.index, volume.values, width=0.8, color="steelblue", alpha=0.4)
    ax_vol.set_ylabel("Post count")
    ax_vol.set_xlabel("Date")

    _format_date_axis(ax_vol, fig)
    fig.tight_layout()

    if save:
        suffix = "_by_user" if by_user else ""
        fig.savefig(_fig_path(f"sentiment_timeline_{score_col}{suffix}.png"),
                    dpi=150, bbox_inches="tight")

    return fig


# ── 2. Tier comparison (the money chart) ────────────────────────────

def plot_tier_comparison(
    df: pd.DataFrame,
    score_col: str = "score_vader",
    smooth_days: int = settings.PLOT_SMOOTH_DAYS,
    save: bool = True,
) -> plt.Figure:
    """
    Rolling-mean sentiment by tier on a single axis — the best view of
    how different political camps' messaging shifted over the war.
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    df = df.dropna(subset=[score_col]).copy()
    df_idx = df.set_index("created_at")

    for tier, color in settings.TIER_COLORS.items():
        subset = df_idx[df_idx["tier"] == tier]
        if subset.empty:
            continue
        daily = subset[score_col].resample("1D").mean()
        smoothed = daily.rolling(smooth_days, center=True, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values,
                color=color, linewidth=2.5,
                label=f"{tier} (n={len(subset)})")
        ax.fill_between(smoothed.index, smoothed.values, alpha=0.1, color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    _add_event_markers(ax, y_top=ax.get_ylim()[1])

    ax.set_title(
        f"Iran War Sentiment by Political Tier ({smooth_days}-day rolling mean)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel(f"Sentiment Score ({score_col})")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=10, ncol=2)
    _format_date_axis(ax, fig)
    fig.tight_layout()

    if save:
        fig.savefig(_fig_path(f"tier_comparison_{score_col}.png"),
                    dpi=150, bbox_inches="tight")
    return fig


# ── 3. Account heatmap ─────────────────────────────────────────────

def plot_account_heatmap(
    df: pd.DataFrame,
    score_col: str = "score_vader",
    freq: str = "W",
    save: bool = True,
) -> plt.Figure:
    """Heatmap: average sentiment per account per time period."""
    df = df.dropna(subset=[score_col]).copy()
    df["period"] = df["created_at"].dt.tz_localize(None).dt.to_period(freq).dt.to_timestamp()
    pivot = df.pivot_table(values=score_col, index="user", columns="period", aggfunc="mean")
    pivot.columns = pivot.columns.strftime("%b %d")

    fig, ax = plt.subplots(figsize=(16, max(6, len(pivot) * 0.5)))
    sns.heatmap(
        pivot, cmap="RdYlGn", center=0, annot=True, fmt=".2f",
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    )
    ax.set_title(f"Sentiment Heatmap by Account ({score_col})", fontsize=14)
    ax.set_ylabel("")
    fig.tight_layout()

    if save:
        fig.savefig(_fig_path(f"account_heatmap_{score_col}.png"),
                    dpi=150, bbox_inches="tight")
    return fig


# ── 4. Public search sentiment ─────────────────────────────────────

def plot_public_search(
    df: pd.DataFrame,
    score_col: str = "score_vader",
    save: bool = True,
) -> plt.Figure | None:
    """
    Time-series of keyword-search tweets — a near-realtime public
    sentiment proxy. Only ~7 days of data available from /search/recent.
    """
    search_df = df[df["user"].astype(str).str.startswith("search:")].copy()
    if search_df.empty:
        logger.info("No search data to plot")
        return None

    fig, ax = plt.subplots(figsize=(14, 5))
    search_idx = search_df.set_index("created_at")
    hourly = search_idx[score_col].resample("1h").mean()

    ax.plot(hourly.index, hourly.values, color="darkblue", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.2, color="darkblue")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(
        f"Public 'Iran war' search sentiment (n={len(search_df)}, last ~7 days)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel(f"Sentiment ({score_col})")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    if save:
        fig.savefig(_fig_path(f"public_search_{score_col}.png"),
                    dpi=150, bbox_inches="tight")
    return fig


# ── Orchestration ───────────────────────────────────────────────────

def generate_all(
    df: pd.DataFrame,
    score_cols: list[str] | None = None,
) -> list[Path]:
    """
    Regenerate all standard figures. Returns the list of output paths.

    Excludes the keyword-search 'user' rows from account-level plots.
    """
    score_cols = score_cols or ["score_vader", "score_transformer"]
    df_accounts = df[~df["user"].astype(str).str.startswith("search:")]
    written: list[Path] = []

    for score_col in score_cols:
        if score_col not in df.columns:
            logger.warning("Skipping %s — not in dataframe", score_col)
            continue

        plot_sentiment_timeline(df_accounts, score_col=score_col,
                                title=f"Iran War Sentiment — All Accounts ({score_col})")
        plot_sentiment_timeline(df_accounts, score_col=score_col, by_user=True,
                                title=f"Iran War Sentiment — Per Account ({score_col})")
        plot_tier_comparison(df_accounts, score_col=score_col)
        plot_account_heatmap(df_accounts, score_col=score_col)
        plot_public_search(df, score_col=score_col)
        plt.close("all")

    for f in sorted(settings.FIGURES_DIR.glob("*.png")):
        written.append(f)
    return written
