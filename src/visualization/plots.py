"""
Visualization module — time-series sentiment plots with event overlays.

Every public plot function takes a DataFrame + score column and returns
a matplotlib Figure. Saving is controlled by the `save` flag so the same
functions can be used from notebooks without writing to disk.

Figures 1-4 average every post. The war-post figures (weekly tier panels,
account x phase heatmap) are the published method, inference.prepare
(posts with text, the about-the-war flag from settings.TOPIC_SOURCE), and
are the README's headline and per-account views.
"""

import logging
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

from config import settings
from config.timeline import ANALYSIS_END, ANALYSIS_START, EVENTS, PHASES
from src.analysis import inference as inf

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
        if event.importance < settings.EVENT_LABEL_MIN_IMPORTANCE:
            continue
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


# ── War-post figures (the published method) ─────────────────────────

# The blog's stance scale (dkweb src/styles/global.css, light theme:
# --viz-anti / --viz-neutral / --viz-pro), so a figure here and a chart in
# the post give anti-war and pro-war the same colours.
STANCE_COLORS = {"anti": "#2a78d6", "neutral": "#cfcdc5", "pro": "#e34948"}
HEATMAP_MIN_CELL_POSTS = 5  # account x phase cells with fewer war posts are left blank
_INK, _MUTED, _HAIRLINE, _CONTEXT = "#1a1a1a", "#666666", "#e0e0e0", "#c8c8c8"
_SCORERS = {"score_opus": "Claude Opus 5", "score_llm": "Claude Haiku 4.5",
            "score_opus_distilled": "distilled DeBERTa"}
_TOPIC_SOURCES = {"either": "the Haiku topic label or the keyword pattern",
                  "llm": "the Haiku topic label", "keyword": "the keyword pattern"}


def stance_cmap() -> LinearSegmentedColormap:
    """Diverging map for vmin=-1, vmax=+1: anti-war blue, neutral grey at 0,
    pro-war red, as the blog draws them (the old RdYlGn heatmap put red on
    the anti-war end)."""
    return LinearSegmentedColormap.from_list(
        "stance", [STANCE_COLORS["anti"], STANCE_COLORS["neutral"], STANCE_COLORS["pro"]])


def _scorer(score_col: str) -> str:
    name = _SCORERS.get(score_col)
    return f"{name} ({score_col})" if name else score_col


def _method_note(score_col: str) -> str:
    topic = _TOPIC_SOURCES.get(settings.TOPIC_SOURCE, settings.TOPIC_SOURCE)
    return (f"War posts: flagged by {topic}. Posts with no text are left out. "
            f"Scorer: {_scorer(score_col)}, -1 anti-war to +1 pro-war.")


def _text_color(rgba) -> str:
    """Ink or white, whichever has the higher WCAG contrast on this fill."""
    def lum(c):
        r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c[:3])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    bg = lum(rgba)
    ink = (bg + 0.05) / (lum(to_rgba(_INK)) + 0.05)
    white = 1.05 / (bg + 0.05)
    return _INK if ink >= white else "white"


def _war_tiers(d: pd.DataFrame) -> list[str]:
    """X tiers in the frame (prepare has dropped search), in TIER_COLORS order."""
    present = set(d["tier"].astype(str)) - {"search"}
    known = [t for t in settings.TIER_COLORS if t in present]
    return known + sorted(present - set(known))


def weekly_war_stance(d: pd.DataFrame, score_col: str,
                      tiers: list[str] | None = None) -> pd.DataFrame:
    """Weekly war-post stance per tier from a prepare()d frame, smoothed as
    web_export.maga_weekly smooths the blog's weekly chart: a centred
    WEB_SMOOTH_WEEKS-week window, sum of scores over sum of posts, so a thin
    week cannot swing it. Each tier gets every week from its first war post
    to its last; `smooth` is NaN where the window holds fewer than
    WEB_MIN_WINDOW_POSTS war posts, so a plot shows a gap, not a join."""
    tiers = tiers or _war_tiers(d)
    w = d[d["on_topic"] & d["tier"].isin(tiers)].copy()
    w["week"] = (pd.to_datetime(w["created_at"], utc=True).dt.tz_localize(None)
                 .dt.to_period("W-SUN").dt.start_time)
    out = []
    for tier, g in w.groupby("tier"):
        wk = g.groupby("week")[score_col].agg(["sum", "count"])
        wk = wk.reindex(pd.date_range(wk.index.min(), wk.index.max(), freq="7D"), fill_value=0)
        roll = wk.rolling(settings.WEB_SMOOTH_WEEKS, center=True, min_periods=1).sum()
        keep = roll["count"] >= settings.WEB_MIN_WINDOW_POSTS
        out.append(pd.DataFrame({
            "tier": tier, "week": wk.index, "n": wk["count"].astype(int).to_numpy(),
            "stance": (wk["sum"] / wk["count"].where(wk["count"] > 0)).to_numpy(),
            "n_window": roll["count"].astype(int).to_numpy(),
            "smooth": (roll["sum"] / roll["count"]).where(keep).to_numpy()}))
    cols = ["tier", "week", "n", "stance", "n_window", "smooth"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def _lone_points(y: np.ndarray) -> list[bool]:
    """Kept weeks with no kept week either side. A line through NaNs draws
    nothing for them (nor does fill_between), so they need a marker."""
    kept = ~np.isnan(np.asarray(y, dtype=float))
    prev = np.r_[False, kept[:-1]]
    nxt = np.r_[kept[1:], False]
    return (kept & ~prev & ~nxt).tolist()


def _stance_yaxis(ax: plt.Axes) -> None:
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1 anti", "-0.5", "0", "+0.5", "+1 pro"])
    ax.grid(axis="y", color=_HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_HAIRLINE)
    ax.tick_params(colors=_MUTED, labelsize=8)


def plot_tier_war_weekly(
    df: pd.DataFrame,
    score_col: str = settings.STANCE_SCORE_COL,
    save: bool = True,
    path: Path | None = None,
) -> plt.Figure:
    """Weekly war-post stance by tier, one panel per X tier with the other
    tiers in grey behind it. Panels, not one axis: admin and pro-war MAGA,
    and anti-war MAGA and the Vatican, sit close enough to tangle. The area
    between the line and zero takes the blog's pro / anti colour."""
    d = inf.prepare(df, score_col)
    tiers = _war_tiers(d)
    wk = weekly_war_stance(d, score_col, tiers)
    wk["x"] = wk["week"] + pd.Timedelta(days=3.5)  # mid-week, so phase lines fall true
    n_war = d[d["on_topic"]].groupby("tier").size()
    start = pd.Timestamp(PHASES[0][1])
    end = pd.Timestamp(PHASES[-1][2]) + pd.Timedelta(days=1)

    ncols = 3
    nrows = max(1, math.ceil(len(tiers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows + 1.2),
                             sharex=True, sharey=True, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(tiers):
            ax.set_visible(False)
            continue
        tier = tiers[i]
        _stance_yaxis(ax)
        ax.axhline(0, color=_MUTED, linewidth=0.8, alpha=0.6)
        for _, p_start, _ in PHASES[1:]:
            ax.axvline(pd.Timestamp(p_start), color=_HAIRLINE, linewidth=1, zorder=0)
        if i < ncols:
            for label, p_start, p_end in PHASES:
                mid = pd.Timestamp(p_start) + (pd.Timestamp(p_end) - pd.Timestamp(p_start)) / 2
                ax.text(mid, 1.0, label.replace(", ", ",\n"), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color=_MUTED, linespacing=1.1)
        for other, g in wk.groupby("tier"):
            if other != tier:
                yo = g["smooth"].to_numpy(dtype=float)
                ax.plot(g["x"], yo, color=_CONTEXT, linewidth=1, zorder=1, marker="o",
                        markersize=2.5, markeredgewidth=0, markevery=_lone_points(yo))
        g = wk[wk["tier"] == tier]
        y = g["smooth"].to_numpy(dtype=float)
        ax.fill_between(g["x"], 0, y, where=y > 0, interpolate=True,
                        color=STANCE_COLORS["pro"], alpha=0.25, linewidth=0, zorder=2)
        ax.fill_between(g["x"], 0, y, where=y < 0, interpolate=True,
                        color=STANCE_COLORS["anti"], alpha=0.25, linewidth=0, zorder=2)
        ax.plot(g["x"], y, color=_INK, linewidth=2, solid_capstyle="round", zorder=3,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5,
                markevery=_lone_points(y))
        if not g["smooth"].notna().any():
            ax.text(0.5, 0.5, f"no {settings.WEB_SMOOTH_WEEKS}-week window reaches "
                    f"{settings.WEB_MIN_WINDOW_POSTS} war posts", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=_MUTED,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
        ax.set_title(settings.WEB_TIER_LABELS.get(tier, tier), loc="left", fontsize=11, color=_INK)
        ax.set_title(f"{int(n_war.get(tier, 0)):,} war posts", loc="right", fontsize=8, color=_MUTED)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes[:, 0]:
        ax.set_ylabel(f"war-post stance ({score_col})", fontsize=9, color=_MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.005, 0.985, f"War-post stance by tier, weekly: {_scorer(score_col)}",
             fontsize=14, fontweight="bold", color=_INK, va="top")
    fig.text(0.005, 0.945,
             f"Centred {settings.WEB_SMOOTH_WEEKS}-week window weighted by posts; weeks whose "
             f"window holds fewer than {settings.WEB_MIN_WINDOW_POSTS} war posts are left out; "
             f"a dot is a kept week with neither neighbour kept. Grey: the other tiers. "
             f"Light rules: phase boundaries. " + _method_note(score_col),
             fontsize=8.5, color=_MUTED, va="top", wrap=True)
    if save:
        fig.savefig(path or _fig_path(f"tier_war_weekly_{score_col}.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
    return fig


def account_war_phase_table(df: pd.DataFrame,
                            score_col: str = settings.STANCE_SCORE_COL) -> pd.DataFrame:
    """War-post count and mean per account x phase plus a whole-war column:
    the point estimates `phases --by user` writes (phase_stats on the same
    prepare), with no bootstrap since the figure draws no intervals."""
    parts = []
    for phases in (PHASES, inf.WHOLE_WAR):
        d = inf.prepare(df, score_col, group="user", phases=phases)
        st = inf.phase_stats(inf.bootstrap_cells(d, score_col, n_boot=0))
        st["phase"] = st["phase"].astype(str)
        st["tier"] = st["group"].map(d.groupby("group")["tier"].first())
        parts.append(st[["group", "tier", "phase", "n_on", "mean_on"]])
    return pd.concat(parts, ignore_index=True)


def plot_account_war_heatmap(
    df: pd.DataFrame,
    score_col: str = settings.STANCE_SCORE_COL,
    save: bool = True,
    path: Path | None = None,
) -> plt.Figure:
    """War-post stance per account x phase (plus the whole war), accounts
    sorted pro-war to anti-war on the whole war. Cells with fewer than
    HEATMAP_MIN_CELL_POSTS war posts are blank, not a noisy mean."""
    t = account_war_phase_table(df, score_col)
    whole = inf.WHOLE_WAR[0][0]
    cols = inf.PHASE_ORDER + [whole]
    mean = t.pivot(index="group", columns="phase", values="mean_on").reindex(columns=cols)
    n = t.pivot(index="group", columns="phase", values="n_on").reindex(columns=cols).fillna(0)
    mean = mean.where(n >= HEATMAP_MIN_CELL_POSTS)
    mean = mean.loc[mean[whole].sort_values(ascending=False, na_position="last").index]
    tier = t.drop_duplicates("group").set_index("group")["tier"]
    rows = [f"@{u}  ({settings.WEB_TIER_LABELS.get(tier[u], tier[u])})" for u in mean.index]
    ticks = [f"{label}\n{s:%b %-d}–{e:%b %-d}" for label, s, e in PHASES]
    ticks.append(f"{whole}\n{inf.WHOLE_WAR[0][1]:%b %-d}–{inf.WHOLE_WAR[0][2]:%b %-d}")

    cmap, norm = stance_cmap(), Normalize(-1, 1)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(mean) + 2.2))
    sns.heatmap(
        mean.set_axis(rows, axis=0).set_axis(ticks, axis=1), mask=mean.isna().to_numpy(),
        cmap=cmap, vmin=-1, vmax=1, linewidths=2, linecolor="white", ax=ax,
        cbar_kws={"label": f"war-post stance ({score_col})", "shrink": 0.6,
                  "ticks": [-1, -0.5, 0, 0.5, 1]},
    )
    vals = mean.to_numpy()
    for (i, j), v in np.ndenumerate(vals):
        if np.isnan(v):
            continue
        ax.text(j + 0.5, i + 0.5, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color=_text_color(cmap(norm(v))))
    ax.axvline(len(inf.PHASE_ORDER), color="white", linewidth=8)  # set the whole war apart
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="both", length=0, labelsize=9, colors=_INK)
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(["-1 anti", "-0.5", "0", "+0.5", "+1 pro"])
    cbar.ax.tick_params(labelsize=8, colors=_MUTED)
    cbar.outline.set_visible(False)
    ax.set_title(
        f"War-post stance by account and phase: {_scorer(score_col)}",
        loc="left", fontsize=13, fontweight="bold", color=_INK, pad=44,
    )
    fig.text(0.01, 0.0,
             f"Mean of each account's war posts in the phase. Blank: fewer than "
             f"{HEATMAP_MIN_CELL_POSTS} war posts. " + _method_note(score_col),
             fontsize=8, color=_MUTED, va="top", wrap=True)
    fig.tight_layout()
    if save:
        fig.savefig(path or _fig_path(f"account_war_phases_{score_col}.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
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
    score_cols = score_cols or settings.PLOT_SCORE_COLS
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

    # War-post views for the stance of record only: the other scorers are
    # shown all-post above to make their miscalibrations visible.
    if settings.STANCE_SCORE_COL in df.columns:
        plot_tier_war_weekly(df, score_col=settings.STANCE_SCORE_COL)
        plot_account_war_heatmap(df, score_col=settings.STANCE_SCORE_COL)
        plt.close("all")

    written.extend(sorted(settings.FIGURES_DIR.glob("*.png")))
    return written
