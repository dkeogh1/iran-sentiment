"""
Chart-ready JSON for the dkweb blog post (Astro + Observable Plot, rendered to
SVG at build time; see dkweb README "Charts in posts"). Every file is a flat
list of records so the MDX can hand it straight to a Plot mark. Numbers come
from the same functions as `phases` / `reply-population`, never recomputed.

  maga_weekly.json      weekly stance on war posts: pro-war MAGA, anti-war MAGA, admin
  events.json           the six milestones drawn as rules on the time charts
  accounts.json         each X account's war-post stance (Opus) with a 95% CI,
                        plus RoBERTa valence on the same posts (the sign-flip chart)
  decomposition.json    each tier's strike-phase -> September change, split into
                        "talked about the war less" and "changed its war stance"
  trump_phases.json     Trump's own Truth Social war posts by phase
                        (settings.TRUMP_STANCE_SCORE_COL: score_opus, Opus 5)
  replies.json          Trump's reply audience per post (replies with text):
                        Opus-corrected anti / neutral / pro shares, net stance
                        with CI, RoBERTa valence, and the labelled random draws
                        it rests on (first + fresh, pooled)

War posts are inference.war_flag's settings.TOPIC_SOURCE (Opus's stance-topic
class since 2026-11-02), read once per prepared frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from config import settings
from config.accounts import ACCOUNT_NAMES
from config.timeline import PHASES
from config.tracked_posts import TRACKED_POSTS
from src.analysis import inference as inf

logger = logging.getLogger(__name__)


def _write(out_dir: Path, name: str, records: list[dict]) -> Path:
    p = out_dir / name
    p.write_text(json.dumps(records, indent=1, default=str))
    return p


def _r(x, nd: int = 3):
    return None if pd.isna(x) else round(float(x), nd)


def maga_weekly(d: pd.DataFrame, score: str) -> list[dict]:
    """Weekly war-post stance per tier plus a post-weighted centred rolling
    mean over WEB_SMOOTH_WEEKS calendar weeks (a thin week cannot swing it,
    and a missing week is a gap, not a join). Windows with fewer than
    WEB_MIN_WINDOW_POSTS war posts are dropped."""
    w = d[d["on_topic"] & d["tier"].isin(settings.WEB_WEEKLY_TIERS)].copy()
    w["week"] = (
        pd.to_datetime(w["created_at"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
        .dt.start_time
    )
    rows = []
    for tier, g in w.groupby("tier"):
        wk = g.groupby("week")[score].agg(["sum", "count"])
        wk = wk.reindex(pd.date_range(wk.index.min(), wk.index.max(), freq="7D"), fill_value=0)
        roll = wk.rolling(settings.WEB_SMOOTH_WEEKS, center=True, min_periods=1).sum()
        for week, r in wk.iterrows():
            n_win = int(roll.loc[week, "count"])
            if n_win < settings.WEB_MIN_WINDOW_POSTS:
                continue
            rows.append(
                {
                    "week": week.date().isoformat(),
                    "tier": settings.WEB_TIER_LABELS.get(tier, tier),
                    "stance": _r(r["sum"] / r["count"]) if r["count"] else None,
                    "n": int(r["count"]),
                    "smooth": _r(roll.loc[week, "sum"] / n_win),
                    "n_window": n_win,
                }
            )
    return rows


def accounts(df: pd.DataFrame, score: str) -> list[dict]:
    d = inf.prepare(df, score, group="user", phases=inf.WHOLE_WAR)
    st = inf.phase_stats(inf.bootstrap_cells(d, score)).set_index("group")
    rob = d[d["on_topic"]].groupby("user")["score_transformer"].mean()
    tier = d.groupby("user")["tier"].first()
    rows = []
    for user, r in st.iterrows():
        if r["n_on"] < settings.WEB_MIN_ACCOUNT_POSTS:
            continue
        rows.append(
            {
                "account": ACCOUNT_NAMES.get(user, f"@{user}"),
                "handle": user,
                "tier": settings.WEB_TIER_LABELS.get(tier[user], tier[user]),
                "stance": _r(r["mean_on"]),
                "lo": _r(r["mean_on_lo"]),
                "hi": _r(r["mean_on_hi"]),
                "valence": _r(rob.get(user)),
                "n": int(r["n_on"]),
            }
        )
    return sorted(rows, key=lambda x: x["stance"])


def decomposition(d: pd.DataFrame, score: str) -> list[dict]:
    boot = inf.bootstrap_cells(d, score)
    c = inf.phase_contrasts(boot)
    c = c[c["phase"] == PHASES[-1][0]]
    parts = [
        ("share_effect", "Talked about the war less (or more)"),
        ("on_topic_effect", "Changed its stance on the war"),
        ("off_topic_effect", "Everything else it posted"),
    ]
    rows = []
    for _, r in c.iterrows():
        if r["group"] not in settings.WEB_TIER_LABELS:
            continue
        for key, label in parts:
            rows.append(
                {
                    "tier": settings.WEB_TIER_LABELS[r["group"]],
                    "part": label,
                    "value": _r(r[key]),
                    "lo": _r(r[f"{key}_lo"]),
                    "hi": _r(r[f"{key}_hi"]),
                    "total": _r(r["all"]),
                }
            )
    return rows


def trump_phases(score: str | None = None) -> list[dict]:
    """Trump's war posts by phase on `score` (settings.TRUMP_STANCE_SCORE_COL,
    score_opus, by default). A post with text in the phases but no score is
    left out with a logged count, never imputed; a feed without the column
    raises (score-posts rewrites the parquet without score_opus: `teacher-check
    --source trump --merge` writes it back). Each tick's dates are clipped to
    the feed: it was collected from Apr 4, so the first phase reads Apr 4,
    not Feb 1."""
    score = score or settings.TRUMP_STANCE_SCORE_COL
    df = pd.read_parquet(settings.TRUMP_FEED_STANCE)
    if score not in df.columns:
        raise LookupError(
            f"{settings.TRUMP_FEED_STANCE.name} has no {score}: `teacher-check --source trump "
            "--merge` writes it onto the feed (score-posts rewrites the parquet without it)"
        )
    text = inf.has_text(df["text"]) & inf.assign_phase(df["created_at"]).notna()
    missing = int((text & df[score].isna()).sum())
    if missing:
        logger.warning(
            "export-web: %d of Trump's %d posts with text in the phases have no %s: left out "
            "of trump_phases.json",
            missing,
            int(text.sum()),
            score,
        )
    d = inf.prepare(df, score, group="user")
    st = inf.phase_stats(inf.bootstrap_cells(d, score))
    days = pd.to_datetime(df["created_at"], utc=True).dt.date
    first, last = days.min(), days.max()
    ticks = {
        label: f"{max(start, first):%b %-d}–{min(end, last):%b %-d}\n{label}"
        for label, start, end in PHASES
    }
    return [
        {
            "phase": str(r["phase"]),
            "tick": ticks[str(r["phase"])],
            "stance": _r(r["mean_on"]),
            "lo": _r(r["mean_on_lo"]),
            "hi": _r(r["mean_on_hi"]),
            "share": _r(r["share"]),
            "n_war": int(r["n_on"]),
            "n": int(r["n"]),
        }
        for _, r in st.iterrows()
    ]


def _log_reply_basis(rp: pd.DataFrame) -> None:
    """Say how many labelled random draws replies.json rests on, first and
    fresh. The pooled draws (REPLY_POPULATION_FRESH_DRAWS) are the published
    basis since 2026-11-02 (the user's call); exporting without the fresh
    draws (the 1,071-draw figures) is warned about."""
    per = rp[rp["post"] != "ALL"]
    if "n_labelled_fresh" not in per:
        return
    n, fresh = int(per["n_labelled"].sum()), int(per["n_labelled_fresh"].sum())
    if fresh:
        logger.info(
            "export-web: replies.json rests on %d labelled random draws pooled: %d first "
            "draws + %d fresh draws (REPLY_POPULATION_FRESH_DRAWS, the published basis "
            "since 2026-11-02)",
            n,
            n - fresh,
            fresh,
        )
    else:
        logger.warning(
            "export-web: replies.json rests on the %d first draws only, no fresh draws: the "
            "published basis is the pooled first + fresh draws (REPLY_POPULATION_FRESH_DRAWS"
            "%s)",
            n,
            " is False" if not settings.REPLY_POPULATION_FRESH_DRAWS else ", none labelled",
        )


def replies() -> list[dict]:
    """Per tracked post, over its replies with text (as reply_population). A
    post whose Opus estimate is NaN (replies in a valence bucket with no
    labelled draw) is left out with a warning, not exported as nulls. Each
    record carries the labelled random draws it rests on (labelled, of them
    labelled_fresh), and the log says the same in all (_log_reply_basis)."""
    rp = inf.reply_population()
    _log_reply_basis(rp)
    val = pd.read_parquet(
        settings.REPLY_SENTIMENT_OUTPUT, columns=["id", "tracked_slug", "text", "score_transformer"]
    )
    val = val[~val["id"].astype(str).duplicated() & inf.has_text(val["text"])]
    val = val.groupby("tracked_slug")["score_transformer"].mean()
    posts = {p.slug: p for p in TRACKED_POSTS}
    rows = []
    for _, r in rp[rp["post"] != "ALL"].iterrows():
        if r[["assisted_mean", "assisted_pro", "assisted_anti"]].isna().any():
            logger.warning(
                "export-web: %s left out of replies.json: %.0f%% of its replies are in "
                "a valence bucket with no Opus-labelled draw",
                r["post"],
                100 * (1 - r["coverage"]),
            )
            continue
        p = posts[r["post"]]
        pro, anti = r["assisted_pro"], r["assisted_anti"]
        # Shares to 4 decimals: the chart labels them in whole percent, and a share
        # cut to 3 that ends in 5 (0.4548 -> 0.455) rounds up a point there.
        rows.append(
            {
                "post": settings.WEB_POST_LABELS.get(r["post"], p.label),
                "slug": r["post"],
                "date": p.event_date.isoformat(),
                "replies": int(r["n_replies"]),
                "anti": _r(anti, 4),
                "neutral": _r(1 - pro - anti, 4),
                "pro": _r(pro, 4),
                "stance": _r(r["assisted_mean"]),
                "lo": _r(r["assisted_mean_lo"]),
                "hi": _r(r["assisted_mean_hi"]),
                "valence": _r(val.get(r["post"])),
                "labelled": int(r["n_labelled"]),
                "labelled_fresh": int(r.get("n_labelled_fresh", 0)),
            }
        )
    return sorted(rows, key=lambda x: x["date"])


def export(out_dir: Path, score: str = settings.STANCE_SCORE_COL) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    d = inf.prepare(df, score)
    events = [{"date": dt, "label": label} for dt, label in settings.WEB_EVENTS]
    return [
        _write(out_dir, "maga_weekly.json", maga_weekly(d, score)),
        _write(out_dir, "events.json", events),
        _write(out_dir, "accounts.json", accounts(df, score)),
        _write(out_dir, "decomposition.json", decomposition(d, score)),
        _write(out_dir, "trump_phases.json", trump_phases()),
        _write(out_dir, "replies.json", replies()),
    ]
