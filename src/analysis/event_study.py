"""
Event-study analysis for the Iran-sentiment project.

Two analysis modes live here:

  1. Per-post reply analysis — for each tracked Trump Truth Social
     post (config/tracked_posts.py), compute the audience-sentiment
     summary the NYT article relied on: N, mean sentiment, bootstrap
     95% CI, and fractions critical/neutral/supportive. This is the
     replication target for the NYT's ~50/25/25 and Epstein ~33/33/33
     headline numbers.

  2. Event-window comparison — for each timeline event
     (config/timeline.py), compare mean broadcaster sentiment in the
     ±window_hours before and after the event, with bootstrapped
     confidence intervals on the difference. Quasi-experimental
     leverage around sharp events like the Easter rant and the
     ceasefire reversal.

Also exposes a loyalty-prior scorer that segments repliers by how
"MAGA" they look *before* the war (bio keywords + account age) so we
can test whether high-loyalty accounts turn negative — the within-
person flip that distinguishes a real civil war from base noise.

Notes on measurement:
  - "critical" / "supportive" collapse sentiment and stance. On a
    pro-war post, a negative-sentiment reply is almost always a
    critical one, but not always (e.g. "I love you Mr President but
    please don't do this" scores positive on sentiment while being
    stance-critical). The LLM scorer handles this better than VADER
    or RoBERTa. Flag disagreements between scorers in the output.
  - Bootstrap defaults to n=2000 resamples, which gives ~1% CI
    precision and runs in well under a second for reply counts up
    to ~50k.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


# ── Score-column helpers ───────────────────────────────────────────

#: Ordered preference — use the most context-aware score that's present.
SCORE_COLS_IN_PREFERENCE = ("score_llm", "score_transformer", "score_vader")
LABEL_COLS_IN_PREFERENCE = ("label_llm", "label_transformer", "label_vader")


def pick_score_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in SCORE_COLS_IN_PREFERENCE:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError(f"No score column in frame; have {list(df.columns)}")


def pick_label_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in LABEL_COLS_IN_PREFERENCE:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError(f"No label column in frame; have {list(df.columns)}")


# ── Bootstrap helpers ──────────────────────────────────────────────

def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """
    Return (mean, lower_bound, upper_bound) at the (1-alpha) level.
    Vanilla percentile bootstrap. Handles small N gracefully.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(vals.mean())
    if len(vals) < 2:
        return (mean, mean, mean)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_resamples, len(vals)))
    samples = vals[idx].mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return (mean, float(lo), float(hi))


def bootstrap_diff_ci(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for (mean(a) - mean(b)). Resamples within each group
    independently (unequal-N friendly).
    """
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"), float("nan"))

    diff = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    ai = rng.integers(0, len(a), size=(n_resamples, len(a)))
    bi = rng.integers(0, len(b), size=(n_resamples, len(b)))
    samples = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return (diff, float(lo), float(hi))


# ── Per-post reply summary ─────────────────────────────────────────

@dataclass
class PostSummary:
    slug: str
    n: int
    mean_score: float
    ci_low: float
    ci_high: float
    pct_critical: float
    pct_neutral: float
    pct_supportive: float
    score_col: str
    label_col: str

    def as_row(self) -> dict:
        return asdict(self)


def summarize_post_replies(
    df: pd.DataFrame,
    slug: str,
    *,
    score_col: str | None = None,
    label_col: str | None = None,
) -> PostSummary:
    """
    Compute the NYT-style breakdown for replies to a single tracked post.
    Expects `df` to be the scored reply frame (reply_sentiment.parquet),
    which must contain a `tracked_slug` column.
    """
    sub = df[df["tracked_slug"] == slug]
    score_col = pick_score_col(sub, score_col)
    label_col = pick_label_col(sub, label_col)

    mean, lo, hi = bootstrap_mean_ci(sub[score_col])

    # Label → critical/neutral/supportive mapping. We treat the label
    # as a stance proxy: "negative" toward a pro-war post ≈ critical.
    n = len(sub)
    if n == 0:
        return PostSummary(slug, 0, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"),
                           score_col, label_col)

    labels = sub[label_col].str.lower()
    pct_crit = (labels == "negative").mean() * 100
    pct_neut = (labels == "neutral").mean() * 100
    pct_supp = (labels == "positive").mean() * 100

    return PostSummary(
        slug=slug, n=n, mean_score=mean, ci_low=lo, ci_high=hi,
        pct_critical=pct_crit, pct_neutral=pct_neut, pct_supportive=pct_supp,
        score_col=score_col, label_col=label_col,
    )


def summarize_all_posts(
    df: pd.DataFrame,
    *,
    score_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """One row per tracked-post slug in the frame."""
    slugs = sorted(df["tracked_slug"].dropna().unique())
    rows = [
        summarize_post_replies(df, slug, score_col=score_col, label_col=label_col).as_row()
        for slug in slugs
    ]
    return pd.DataFrame(rows)


# ── Loyalty-prior segmentation ─────────────────────────────────────

#: Bio keyword regexes that suggest a user self-identifies as MAGA-loyal.
#: Conservative list — false positives here inflate the "high loyalty"
#: bucket and weaken the analysis. Extend deliberately.
LOYALTY_BIO_KEYWORDS = (
    "maga", "trump", "patriot", "ultra maga", "america first", "kag",
    "conservative", "1776", "q17", "wwg1wga", "constitution", "christian",
    "god fearing", "god bless america", "save america",
)


def loyalty_score(record: dict | pd.Series) -> float:
    """
    Return a 0–1 "how MAGA does this account look before the war?" score.
    Two additive signals:

      - Bio keywords (0–0.7): up to 0.7 from explicit self-identification.
      - Account age (0–0.3): older accounts get a mild boost, capped at
        3+ years on Truth Social.

    This is a *prior*, not a measurement of current stance. High-loyalty
    accounts turning critical is the strongest civil-war evidence we can
    extract from this data.
    """
    account = (
        record.get("account") if isinstance(record, dict) else record.get("account", {})
    ) or {}
    bio = (account.get("bio") or "").lower()

    bio_hits = sum(1 for kw in LOYALTY_BIO_KEYWORDS if kw in bio)
    bio_component = min(0.7, bio_hits * 0.2)  # 3 hits saturates

    age_component = 0.0
    created = account.get("created_at")
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            years = (datetime.now(timezone.utc) - created_dt).days / 365.25
            age_component = min(0.3, max(0.0, years / 10))  # 3yr ≈ 0.3
        except Exception:
            pass

    return bio_component + age_component


def loyalty_tier(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def segment_by_loyalty(
    df: pd.DataFrame,
    *,
    score_col: str | None = None,
) -> pd.DataFrame:
    """
    Return mean sentiment + bootstrap CI per loyalty tier, optionally
    grouped by tracked_slug. Expects a scored reply frame that has an
    `account` column (dict) — if the column is missing, loyalty falls
    back to 'unknown' for every row.
    """
    df = df.copy()
    score_col = pick_score_col(df, score_col)

    if "account" in df.columns:
        df["loyalty_score"] = df["account"].apply(
            lambda a: loyalty_score({"account": a}) if isinstance(a, dict) else 0.0
        )
        df["loyalty_tier"] = df["loyalty_score"].apply(loyalty_tier)
    else:
        df["loyalty_score"] = 0.0
        df["loyalty_tier"] = "unknown"

    group_cols = ["tracked_slug", "loyalty_tier"] if "tracked_slug" in df.columns else ["loyalty_tier"]

    rows: list[dict] = []
    for key, sub in df.groupby(group_cols):
        mean, lo, hi = bootstrap_mean_ci(sub[score_col])
        row = {"n": len(sub), "mean": mean, "ci_low": lo, "ci_high": hi,
               "score_col": score_col}
        if isinstance(key, tuple):
            for col, val in zip(group_cols, key):
                row[col] = val
        else:
            row[group_cols[0]] = key
        rows.append(row)

    out = pd.DataFrame(rows)
    ordering = ["low", "medium", "high", "unknown"]
    out["loyalty_tier"] = pd.Categorical(out["loyalty_tier"], ordering, ordered=True)
    return out.sort_values(group_cols).reset_index(drop=True)


# ── Event-window comparison (broadcaster data) ─────────────────────

@dataclass
class EventWindowResult:
    event_date: str
    event_label: str
    window_hours: int
    n_pre: int
    n_post: int
    mean_pre: float
    mean_post: float
    diff: float
    diff_ci_low: float
    diff_ci_high: float
    score_col: str

    def as_row(self) -> dict:
        return asdict(self)


def event_window(
    df: pd.DataFrame,
    event_datetime: datetime,
    *,
    window_hours: int = 48,
    score_col: str | None = None,
    filter_expr: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Slice a scored broadcaster frame into ±window_hours around an event.
    Returns (pre_scores, post_scores, score_col_used).
    `filter_expr` is a pandas query() string (e.g. "tier == 'admin'").
    """
    score_col = pick_score_col(df, score_col)
    sub = df if filter_expr is None else df.query(filter_expr)

    created = pd.to_datetime(sub["created_at"], utc=True)
    center = pd.Timestamp(event_datetime).tz_convert("UTC") if pd.Timestamp(event_datetime).tzinfo \
             else pd.Timestamp(event_datetime, tz="UTC")
    lo = center - pd.Timedelta(hours=window_hours)
    hi = center + pd.Timedelta(hours=window_hours)

    pre_mask = (created >= lo) & (created < center)
    post_mask = (created >= center) & (created < hi)

    pre = sub.loc[pre_mask, score_col].to_numpy()
    post = sub.loc[post_mask, score_col].to_numpy()
    return pre, post, score_col


def compare_events(
    df: pd.DataFrame,
    events: Iterable,
    *,
    window_hours: int = 48,
    score_col: str | None = None,
    filter_expr: str | None = None,
) -> pd.DataFrame:
    """
    Run `event_window` on every Event in `events` (from config.timeline)
    and return a DataFrame of pre/post means with bootstrap CI on the
    difference.
    """
    rows: list[dict] = []
    for ev in events:
        ev_dt = datetime(ev.date.year, ev.date.month, ev.date.day, tzinfo=timezone.utc)
        pre, post, col = event_window(
            df, ev_dt,
            window_hours=window_hours,
            score_col=score_col,
            filter_expr=filter_expr,
        )
        if len(pre) == 0 and len(post) == 0:
            continue
        diff, lo, hi = bootstrap_diff_ci(post, pre)
        rows.append(
            EventWindowResult(
                event_date=ev.date.isoformat(),
                event_label=ev.label,
                window_hours=window_hours,
                n_pre=len(pre),
                n_post=len(post),
                mean_pre=float(pre.mean()) if len(pre) else float("nan"),
                mean_post=float(post.mean()) if len(post) else float("nan"),
                diff=diff,
                diff_ci_low=lo,
                diff_ci_high=hi,
                score_col=col,
            ).as_row()
        )
    return pd.DataFrame(rows)


# ── Reply scoring (load-or-score) ──────────────────────────────────

def load_or_score_replies(force: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of scored reply records. Loads from
    settings.REPLY_SENTIMENT_OUTPUT if present, else runs the shared
    sentiment pipeline on data/raw/truthsocial/replies_*.jsonl and
    caches the parquet.

    Scored in-place so the event-study command is one call instead of
    requiring a separate `analyze --replies` step.
    """
    from src.collectors.truthsocial_collector import load_all_cached_replies
    from src.analysis.sentiment import analyze as run_analyze, save as _save

    out = settings.REPLY_SENTIMENT_OUTPUT
    if out.exists() and not force:
        logger.info("Loading cached reply sentiment from %s", out)
        return pd.read_parquet(out)

    replies = load_all_cached_replies()
    if not replies:
        logger.warning("No cached replies found — run `collect-replies` first.")
        return pd.DataFrame()

    logger.info("Scoring %d replies through VADER + RoBERTa", len(replies))
    df = run_analyze(replies, use_vader=True, use_transformer=True, use_llm=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved scored replies → %s", out)
    return df


# ── Stance scoring (LLM-based) ────────────────────────────────────

STANCE_CATEGORIES = [
    "pro_war_supportive",
    "pro_trump_antiwar",
    "antiwar_betrayal",
    "antiwar_opposition",
    "pro_war_critical",
    "neutral_other",
]

STANCE_PROMPT = """\
Score the STANCE of this Truth Social reply to a Trump post about the Iran war.

Context: This is a reply to President Trump's post on Truth Social. \
The parent post was about the Iran war (threatening strikes, demanding \
Iran open the Strait of Hormuz, or announcing a ceasefire).

Reply text: \"\"\"{text}\"\"\"
Reply by: @{user}

Classify the reply's STANCE (not just sentiment) into exactly one category:
- pro_war_supportive: supports Trump AND supports the war/strikes
- pro_trump_antiwar: supports Trump generally but opposes THIS action \
("I love you but please stop")
- antiwar_betrayal: feels personally betrayed, often mentions voting \
for Trump ("voted 3x", "losing me as a supporter")
- antiwar_opposition: opposes war on moral/political grounds (could be \
from left or right)
- pro_war_critical: supports the war but criticizes Trump for not going \
far enough or for backing down
- neutral_other: off-topic, unclear, or genuinely neutral

Respond with ONLY valid JSON:
{{"stance": "<one of the categories above>", "confidence": <0.0-1.0>, \
"reason": "<10 words max>"}}"""

STANCE_OUTPUT = settings.PROCESSED_DIR / "stance_sample.parquet"


def _parse_json_response(text: str) -> dict:
    """Strip markdown fencing if present, then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def stratified_stance_sample(
    df: pd.DataFrame,
    n_per_bucket: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a stratified sample from the scored reply frame.

    For each tracked post, samples `n_per_bucket` from each of three
    sentiment buckets (critical/mid/supportive) so the LLM sees the
    full spectrum — not just the dominant-negative tail.

    Also adds a flipper cohort: accounts that were supportive on
    power_plant_day but turned critical on civilisation_dies.
    """
    score_col = pick_score_col(df)
    samples: list[pd.DataFrame] = []

    for slug in sorted(df["tracked_slug"].dropna().unique()):
        sub = df[df["tracked_slug"] == slug]
        for lo, hi in [(-999, -0.3), (-0.3, 0.3), (0.3, 999)]:
            bucket = sub[(sub[score_col] >= lo) & (sub[score_col] < hi)]
            n = min(n_per_bucket, len(bucket))
            if n > 0:
                samples.append(bucket.sample(n=n, random_state=seed))

    # Flipper cohort
    user_slug_score = df.groupby(["user", "tracked_slug"])[score_col].mean()
    pivot = user_slug_score.unstack()
    if "power_plant_day" in pivot.columns and "civilisation_dies" in pivot.columns:
        flippers = pivot[
            (pivot["power_plant_day"] > 0.1) & (pivot["civilisation_dies"] < -0.1)
        ]
        flipper_replies = df[
            (df["user"].isin(flippers.index))
            & (df["tracked_slug"] == "civilisation_dies")
        ]
        n = min(n_per_bucket, len(flipper_replies))
        if n > 0:
            samples.append(flipper_replies.sample(n=n, random_state=seed))

    return pd.concat(samples).drop_duplicates("id").reset_index(drop=True)


def score_stance(
    df: pd.DataFrame,
    *,
    model: str = settings.LLM_MODEL,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run LLM stance classification on a (sampled) reply frame.

    Returns a DataFrame with columns: id, tracked_slug, user, text,
    score_transformer, stance, confidence, reason.

    Caches to STANCE_OUTPUT. Pass `force=True` to re-score.
    """
    if STANCE_OUTPUT.exists() and not force:
        logger.info("Loading cached stance scores from %s", STANCE_OUTPUT)
        return pd.read_parquet(STANCE_OUTPUT)

    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic — required for stance scoring")

    from dotenv import load_dotenv
    load_dotenv(override=True)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env for stance scoring")

    client = anthropic.Anthropic()
    score_col = pick_score_col(df)
    results: list[dict] = []

    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stance"):
        text = (row.get("text") or "")[:500]
        user = row.get("user", "")

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": STANCE_PROMPT.format(text=text, user=user),
                }],
            )
            data = _parse_json_response(resp.content[0].text)
            results.append({
                "id": row["id"],
                "tracked_slug": row.get("tracked_slug"),
                "user": user,
                "text": text[:100],
                score_col: row.get(score_col),
                "stance": data.get("stance"),
                "confidence": data.get("confidence"),
                "reason": data.get("reason"),
            })
        except Exception as e:
            logger.debug("Stance error on %s: %s", row.get("id"), e)
            results.append({
                "id": row["id"],
                "tracked_slug": row.get("tracked_slug"),
                "user": user,
                "text": text[:100],
                score_col: row.get(score_col),
                "stance": "error",
                "confidence": 0,
                "reason": str(e)[:80],
            })

    result_df = pd.DataFrame(results)
    STANCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(STANCE_OUTPUT, index=False)
    logger.info("Saved %d stance scores → %s", len(result_df), STANCE_OUTPUT)
    return result_df


def stance_summary(df: pd.DataFrame) -> None:
    """Print stance analysis tables to stdout."""
    import click

    score_col = [c for c in df.columns if c.startswith("score_")][0] if \
        any(c.startswith("score_") for c in df.columns) else None

    valid = df[df["stance"] != "error"]
    errors = len(df) - len(valid)

    click.echo(f"\n{'='*70}")
    click.echo(f"STANCE ANALYSIS ({len(valid)} scored, {errors} errors)")
    click.echo(f"{'='*70}")

    # Overall distribution
    click.echo("\n--- Overall stance distribution ---")
    counts = valid["stance"].value_counts()
    total = len(valid)
    for stance, n in counts.items():
        bar = "█" * int(n / total * 40)
        click.echo(f"  {stance:22s}  {n:4d}  ({n/total:5.1%})  {bar}")

    # By post
    click.echo("\n--- Stance by post (%) ---")
    ct = pd.crosstab(
        valid["tracked_slug"], valid["stance"], normalize="index"
    ) * 100
    click.echo(ct.round(1).to_string())

    # Stance vs RoBERTa score (the misclassification check)
    if score_col:
        click.echo(f"\n--- Stance vs {score_col} (RoBERTa blindspot check) ---")
        means = (
            valid.groupby("stance")[score_col]
            .agg(["mean", "count"])
            .sort_values("mean")
        )
        for stance, row in means.iterrows():
            flag = ""
            if stance == "pro_war_critical" and row["mean"] < -0.1:
                flag = " ← RoBERTa sees negative, but stance is PRO-war"
            elif stance == "pro_trump_antiwar" and row["mean"] > 0:
                flag = " ← RoBERTa sees positive, but stance is ANTI-war"
            click.echo(
                f"  {stance:22s}  mean={row['mean']:+.3f}  n={int(row['count'])}{flag}"
            )
