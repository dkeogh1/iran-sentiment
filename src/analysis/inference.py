"""
Uncertainty and composition checks behind the published numbers.

  phase_stats        tier (or account) x phase means, split into war posts and
                     everything else, with account-day block-bootstrap CIs
  phase_contrasts    each phase against the first: the all-post change split
                     into a topic-share effect and on-topic / off-topic stance
                     changes (a symmetric shift-share decomposition)
  phase_gaps         between-group gaps within each phase (e.g. pro-war MAGA
                     minus admin), all posts and war posts only
  reply_population   per-post Opus stance of Trump's reply audience, estimated
                     from the Opus-labelled reply sample weighted back to the
                     population, directly and model-assisted (distilled score
                     + the sample's mean correction)
  draw_fresh_sample  a second, frozen random sample of replies per post
                     (proportional across the valence buckets) whose v2
                     labels join the first draws in reply_population
  reply_category_population
                     the six reply categories' population shares, Opus
                     (cat-v2) and Haiku (`stance`), on the first draws
  teacher_retest     Opus labelled the same 497 posts twice (direct teacher
                     check, then the batch relabel): how much does it move?

Everything here reads cached parquets; nothing calls an API. Two writes:
reply_population (and so export-web) records the reply sample's draws in
REPLY_DRAWS_MANIFEST the first time it sees a post `stance` has sampled, and
draw_fresh_sample (`reply-fresh-sample`) records the fresh draws in their own
manifest before any of them is sent.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from config.timeline import PHASES

logger = logging.getLogger(__name__)

STANCE_BAND = 0.05  # |score| above this counts as pro / anti (sign3 in stance_local)
STANCE_BUCKET_NAMES = ("critical", "mid", "supportive")  # event_study.STANCE_BUCKETS, in order


# ── Phase / topic tagging ──────────────────────────────────────────

def assign_phase(created_at: pd.Series, phases=PHASES) -> pd.Series:
    """Phase label per row (UTC date, both ends inclusive); NaN outside."""
    d = pd.to_datetime(created_at, utc=True).dt.date
    out = pd.Series(np.nan, index=created_at.index, dtype=object)
    for label, start, end in phases:
        out[(d >= start) & (d <= end)] = label
    return out


def has_text(text: pd.Series) -> pd.Series:
    """True where a post or reply has text to judge (src/text_rules.py).
    Image-, video- and link-only rows are not off-topic or neutral: they
    leave the denominator."""
    from src.text_rules import has_text as one
    return text.map(one).astype(bool)


def on_topic(text: pd.Series, pattern: str = settings.WAR_TOPIC_PATTERN) -> pd.Series:
    return text.fillna("").str.contains(pattern, case=False, regex=True)


def war_flag(d: pd.DataFrame, source: str = settings.TOPIC_SOURCE) -> pd.Series:
    """About-the-war flag per row. "keyword": the pattern. "llm": the Haiku
    topic label, the pattern where there is none (link-only posts Haiku
    would not judge). "either": flagged by one or the other -- Haiku is
    strict (it drops the Pope's war appeals that never name Iran), the
    pattern is loose (it misses unnamed strikes, catches "culture war").
    "opus": an "either" post whose Opus stance-topic label says iran_war
    (post_labels.opus_war_flag; an unlabelled candidate counts as "either"
    does, a post neither filter flags never counts)."""
    kw = on_topic(d["text"])
    if source == "keyword":
        return kw
    if source == "opus":
        from src.analysis.post_labels import opus_war_flag
        return opus_war_flag(d, war_flag(d, "either"))
    from src.analysis.topic_label import load_labels
    lab = load_labels()
    if lab is None:
        logger.warning("no topic labels yet (run `topic-label`); using the keyword filter")
        return kw
    m = d["id"].astype(str).map(lab.set_index("id")["about_war"])
    llm = m.where(m.notna(), kw).astype(bool)
    return (llm | kw) if source == "either" else llm


def prepare(df: pd.DataFrame, score_col: str, group: str = "tier",
            topic_source: str = settings.TOPIC_SOURCE, phases=PHASES) -> pd.DataFrame:
    """Rows with text and a score inside a phase, tagged with phase / topic /
    day. Rows without text (has_text: 1,810 of Trump's Truth Social posts,
    images or videos with no caption and quotes of his own no-text posts)
    are dropped first, so they leave every share and
    mean instead of counting as off-topic posts with a constant score.
    Pass phases=WHOLE_WAR for one cell per group over the whole window."""
    d = df[df[score_col].notna() & has_text(df["text"])].copy()
    if "user" in d:
        d = d[~d["user"].astype(str).str.startswith("search:")]
    d["phase"] = assign_phase(d["created_at"], phases)
    d = d[d["phase"].notna()]
    d["on_topic"] = war_flag(d, topic_source)
    d["day"] = pd.to_datetime(d["created_at"], utc=True).dt.date
    d["group"] = d[group].astype(str)
    return d


# ── Block bootstrap ────────────────────────────────────────────────

def _clusters(d: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """One row per account-day: group, phase, and the on/off-topic sums."""
    x = d.assign(s_on=np.where(d["on_topic"], d[score_col], 0.0),
                 n_on=d["on_topic"].astype(int),
                 s_off=np.where(d["on_topic"], 0.0, d[score_col]),
                 n_off=(~d["on_topic"]).astype(int))
    return (x.groupby(["user", "day", "group", "phase"], observed=True)
             [["s_on", "n_on", "s_off", "n_off"]].sum().reset_index())


def _block_weights(users: pd.Series, n_boot: int, seed: int) -> np.ndarray:
    """(n_clusters, 1 + n_boot) resampling counts: column 0 is the observed
    data (all ones); each other column resamples every account's days with
    replacement, keeping the account's day count fixed."""
    rng = np.random.default_rng(seed)
    w = np.ones((len(users), 1 + n_boot))
    for _, idx in users.groupby(users.values).indices.items():
        k = len(idx)
        w[idx, 1:] = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).T
    return w


def _cell(c: pd.DataFrame, w: np.ndarray) -> dict[str, np.ndarray]:
    """Replicate vectors (length 1 + n_boot) for one group x phase cell."""
    s_on, n_on = w.T @ c["s_on"].values, w.T @ c["n_on"].values
    s_off, n_off = w.T @ c["s_off"].values, w.T @ c["n_off"].values
    n = n_on + n_off
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"share": n_on / n, "mean_all": (s_on + s_off) / n,
                "mean_on": s_on / n_on, "mean_off": s_off / n_off}


def _ci(v: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    boot = v[1:][~np.isnan(v[1:])]
    if len(boot) == 0:
        return float(v[0]), float("nan"), float("nan")
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(v[0]), float(lo), float(hi)


def bootstrap_cells(d: pd.DataFrame, score_col: str, n_boot: int = settings.BOOTSTRAP_N,
                    seed: int = settings.BOOTSTRAP_SEED) -> dict:
    """{(group, phase): replicate dict} plus observed counts, shared weights
    so contrasts and gaps can be formed replicate by replicate."""
    c = _clusters(d, score_col)
    w = _block_weights(c["user"], n_boot, seed)
    cells, counts = {}, {}
    for (g, p), idx in c.groupby(["group", "phase"], observed=True).indices.items():
        cells[(g, p)] = _cell(c.iloc[idx], w[idx])
        sub = c.iloc[idx]
        counts[(g, p)] = {"n": int(sub["n_on"].sum() + sub["n_off"].sum()),
                          "n_on": int(sub["n_on"].sum()),
                          "n_accounts": int(sub["user"].nunique()), "n_days": int(len(sub))}
    return {"cells": cells, "counts": counts}


# ── Tables ─────────────────────────────────────────────────────────

PHASE_ORDER = [p[0] for p in PHASES]
WHOLE_WAR = [("whole war", PHASES[0][1], PHASES[-1][2])]


def phase_stats(boot: dict) -> pd.DataFrame:
    rows = []
    for (g, p), rep in boot["cells"].items():
        row = {"group": g, "phase": p, **boot["counts"][(g, p)]}
        for k, v in rep.items():
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = _ci(v)
        rows.append(row)
    out = pd.DataFrame(rows)
    order = PHASE_ORDER + [p for p in out["phase"].unique() if p not in PHASE_ORDER]
    out["phase"] = pd.Categorical(out["phase"], order, ordered=True)
    return out.sort_values(["group", "phase"]).reset_index(drop=True)


def decompose(a: dict, b: dict) -> dict[str, np.ndarray]:
    """Change from cell a to cell b: all = share + on + off, where
    share = dS * (mean on - mean off, averaged), on = S_avg * d(mean on),
    off = (1 - S_avg) * d(mean off). Exact for every replicate."""
    ds = b["share"] - a["share"]
    s_avg = (a["share"] + b["share"]) / 2
    gap_avg = ((a["mean_on"] - a["mean_off"]) + (b["mean_on"] - b["mean_off"])) / 2
    return {"all": b["mean_all"] - a["mean_all"],
            "share_effect": ds * gap_avg,
            "on_topic_effect": s_avg * (b["mean_on"] - a["mean_on"]),
            "off_topic_effect": (1 - s_avg) * (b["mean_off"] - a["mean_off"]),
            "on_topic_change": b["mean_on"] - a["mean_on"]}


def phase_contrasts(boot: dict, base: str = PHASE_ORDER[0]) -> pd.DataFrame:
    rows = []
    for (g, p), rep in boot["cells"].items():
        if p == base or (g, base) not in boot["cells"]:
            continue
        dec = decompose(boot["cells"][(g, base)], rep)
        row = {"group": g, "phase": p}
        for k, v in dec.items():
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = _ci(v)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["phase"] = pd.Categorical(out["phase"], PHASE_ORDER, ordered=True)
    return out.sort_values(["group", "phase"]).reset_index(drop=True)


def phase_gaps(boot: dict, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        for p in PHASE_ORDER:
            if (a, p) not in boot["cells"] or (b, p) not in boot["cells"]:
                continue
            ca, cb = boot["cells"][(a, p)], boot["cells"][(b, p)]
            row = {"pair": f"{a} - {b}", "phase": p}
            for k in ("mean_all", "mean_on", "share"):
                row[k], row[f"{k}_lo"], row[f"{k}_hi"] = _ci(ca[k] - cb[k])
            rows.append(row)
    return pd.DataFrame(rows)


# The phases columns that need every post with text scored: on a column
# that only the war candidates carry (a post-label job's), each would be a
# figure over the candidates under its all-post name.
ALL_POST_COLUMNS = (
    "share",
    "mean_all",
    "mean_off",
    "all",
    "share_effect",
    "on_topic_effect",
    "off_topic_effect",
)


def war_posts_only(table: pd.DataFrame) -> pd.DataFrame:
    """A phase_stats / phase_contrasts / phase_gaps table with every
    ALL_POST_COLUMNS figure (and its interval) blanked to NaN, the war-post
    columns (mean_on, on_topic_change) and counts left as they are."""
    cols = [f"{c}{s}" for c in ALL_POST_COLUMNS for s in ("", "_lo", "_hi")]
    out = table.copy()
    for c in cols:
        if c in out:
            out[c] = np.nan
    return out


def topic_filter_check(d: pd.DataFrame, score_col: str) -> dict:
    """How well the keyword filter separates stance-bearing posts: mean |stance|
    and share of posts with |stance| >= 0.3 inside and outside the filter."""
    a = d[score_col].abs()
    strong = a >= 0.3
    return {"on_topic_share": float(d["on_topic"].mean()),
            "mean_abs_on": float(a[d["on_topic"]].mean()),
            "mean_abs_off": float(a[~d["on_topic"]].mean()),
            "strong_share_on": float(strong[d["on_topic"]].mean()),
            "strong_share_off": float(strong[~d["on_topic"]].mean()),
            "strong_captured": float(d.loc[strong, "on_topic"].mean())}


# ── Reply audience: sample -> population ───────────────────────────

def stance_bucket(score: pd.Series) -> pd.Series:
    """Index into event_study.STANCE_BUCKETS (0 critical, 1 mid, 2 supportive)."""
    from src.analysis.event_study import STANCE_BUCKETS
    edges = [STANCE_BUCKETS[0][0]] + [hi for _, hi in STANCE_BUCKETS]
    return pd.cut(score, edges, right=False, labels=False)


def _per_post(d: pd.DataFrame) -> str:
    return ", ".join(f"{s} ({n})" for s, n in d.groupby("tracked_slug").size().items())


def reply_draws(
    replies: pd.DataFrame,
    n_per_bucket: int = 50,
    score_col: str = "score_transformer",
    fresh: bool = False,
) -> pd.DataFrame:
    """The random draws of the Opus-labelled reply sample (_stance_draws),
    and with `fresh` the fresh draws as well (draw_fresh_sample's manifest),
    each tagged with its draw_set ("stance" for the first). A fresh draw is
    drawn from the replies with text not already among the first draws, so
    within a post and bucket the two together are one simple random sample
    and reply_population weights them alike."""
    draws = _stance_draws(replies, n_per_bucket, score_col)
    if not fresh:
        return draws
    first = draws.assign(id=draws["id"].astype(str), draw_set="stance")
    fr = read_fresh_draws()
    if fr.empty:
        return first
    dup = fr["id"].isin(set(first["id"]))
    if dup.any():
        logger.warning(
            "reply draws: %d fresh draws are also first draws; counted once: %s",
            int(dup.sum()),
            _per_post(fr[dup]),
        )
    return pd.concat(
        [first, fr.loc[~dup, ["id", "tracked_slug", "bucket", "draw_set"]]], ignore_index=True
    )


def _stance_draws(
    replies: pd.DataFrame, n_per_bucket: int = 50, score_col: str = "score_transformer"
) -> pd.DataFrame:
    """The random bucket draws of the Opus-labelled reply sample: id,
    tracked_slug, bucket at draw time, kept in REPLY_DRAWS_MANIFEST so a later
    reply collection never moves a post's draws. Posts it lacks are drawn now
    with event_study.bucket_draws on `replies` (the whole reply frame in file
    order, as `stance` draws; it samples each post on its own, so drawing only
    the new posts gives the same rows as drawing them all). `stance` passes
    its own n_per_bucket and score column.

    New draws are recorded only once STANCE_OUTPUT holds every one of them:
    drawn before `stance` samples the post, or by a `stance` run that died
    before saving them, they could come from a different frame than the one
    whose draws get labelled. Until then they serve this run only (a post
    never sampled has no labels and comes out NaN either way)."""
    from src.analysis.event_study import STANCE_OUTPUT, bucket_draws

    path = settings.REPLY_DRAWS_MANIFEST
    have = (pd.read_parquet(path) if path.exists()
            else pd.DataFrame(columns=["id", "tracked_slug", "bucket"]))
    sampled: set[tuple[str, str]] = set()
    if STANCE_OUTPUT.exists():
        s = pd.read_parquet(STANCE_OUTPUT, columns=["id", "tracked_slug"])
        sampled = set(zip(s["tracked_slug"], s["id"].astype(str)))

    def in_stance(d: pd.DataFrame) -> pd.Series:
        return pd.Series([k in sampled for k in zip(d["tracked_slug"], d["id"].astype(str))],
                         index=d.index, dtype=bool)

    stale = have[~in_stance(have)]
    if len(stale):
        logger.warning("reply draws: %d recorded draws are not in %s and carry no label: %s",
                       len(stale), STANCE_OUTPUT.name, _per_post(stale))
    missing = sorted(set(replies["tracked_slug"].dropna()) - set(have["tracked_slug"]))
    if not missing:
        return have
    drawn = bucket_draws(replies[replies["tracked_slug"].isin(missing)],
                         n_per_bucket=n_per_bucket, score_col=score_col)
    new = pd.DataFrame({"id": drawn["id"].astype(str), "tracked_slug": drawn["tracked_slug"],
                        "bucket": stance_bucket(drawn[score_col]).astype(int),
                        "recorded_at": pd.Timestamp.now(tz="UTC")}).drop_duplicates("id")
    if new.empty:
        return have
    hit = in_stance(new)
    keep = new["tracked_slug"].map(hit.groupby(new["tracked_slug"]).all())
    sampled_posts = {slug for slug, _ in sampled}
    off = new[~keep & new["tracked_slug"].isin(sampled_posts) & ~hit]
    if len(off):
        # `stance` drew these posts from a different reply frame (it has
        # grown since): today's draws only partly meet the labels.
        logger.warning("reply draws: draws not in %s, so not recorded; this run's "
                       "estimate rests on the labelled overlap: %s",
                       STANCE_OUTPUT.name, _per_post(off))
    waiting = new[~keep & ~new["tracked_slug"].isin(sampled_posts)]
    if len(waiting):
        logger.info("reply draws: not sampled by `stance` yet, not recorded: %s",
                    _per_post(waiting))
    rec = new[keep]
    if len(rec):
        have = rec if have.empty else pd.concat([have, rec], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(path, index=False)
        logger.info("reply draws: recorded %d draws for %s -> %s", len(rec), _per_post(rec), path)
    parts = [d for d in (have, new[~keep]) if len(d)]
    return pd.concat(parts, ignore_index=True) if parts else have


def _stance_shares(y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": y,
        "pro": (y > STANCE_BAND).astype(float),
        "anti": (y < -STANCE_BAND).astype(float),
        "neutral": (np.abs(y) <= STANCE_BAND).astype(float),
    }


def _weighted(
    p: pd.DataFrame,
    smp: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
    nan_ok: bool = False,
):
    """(coverage, replicates or None, uncovered strata) for one post (or ALL):
    each array (one value per row of smp, the labelled draws) averaged within
    its (post, draw-time bucket) stratum and weighted by that stratum's share
    of the population p (current buckets). Replicate 0 is the sample, then
    n_boot resamples of the labelled rows within each stratum, one shared
    resample per stratum for every array. Each stratum's replicate
    deviations from its sample mean are scaled by sqrt(1 - n_h / N_h), its
    labelled draws over its population (the finite-population correction:
    the fresh draws sample up to a third of a stratum). Replicates are None
    when part of the population sits in a stratum with no labelled draw
    (coverage < 1). nan_ok averages the non-NaN values of a stratum (labels
    some rows lack)."""
    strata = smp.groupby(["tracked_slug", "bucket"]).indices
    n_pop = p.groupby(["tracked_slug", "bucket"]).size()
    total = float(n_pop.sum())
    coverage = float(n_pop[n_pop.index.isin(list(strata))].sum()) / total if total else 0.0
    if coverage < 1:
        uncovered = [
            f"{s}/{STANCE_BUCKET_NAMES[b]}" for s, b in n_pop.index if (s, b) not in strata
        ]
        return coverage, None, uncovered
    reps = {k: np.zeros(1 + n_boot) for k in arrays}
    for key, idx in strata.items():
        n_h = n_pop.get(key, 0)
        wt = n_h / total  # a draw-time bucket since emptied weighs nothing
        fpc = np.sqrt(max(0.0, 1.0 - len(idx) / n_h)) if n_h else 0.0
        pick = np.vstack([np.arange(len(idx)), rng.integers(0, len(idx), (n_boot, len(idx)))])
        for k, a in arrays.items():
            if not nan_ok:
                m = a[idx][pick].mean(axis=1)
            else:
                with warnings.catch_warnings():  # a resample of only NaN rows is a NaN replicate
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m = np.nanmean(a[idx][pick], axis=1)
            m[1:] = m[0] + fpc * (m[1:] - m[0])
            reps[k] += wt * m
    return coverage, reps, []


def _reply_frames(col: str, fresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(population, random draws, replies without text per post) as
    reply_population weighs them: the population is each post's replies
    with text and a `col` and RoBERTa score, one row per id, bucketed by
    current valence; the draws are reply_draws (with the fresh ones when
    `fresh`), each with its draw_set and draw-time bucket."""
    replies = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    replies["id"] = replies["id"].astype(str)
    draws = reply_draws(replies, fresh=fresh)  # fixed first
    if "draw_set" not in draws:
        draws = draws.assign(draw_set="stance")
    draws = draws.assign(id=draws["id"].astype(str), bucket=draws["bucket"].astype(int))
    replies = replies.drop_duplicates("id")  # a few ids were collected twice
    text = has_text(replies["text"])
    n_no_text = (~text).groupby(replies["tracked_slug"]).sum()
    pop = replies[text & replies[col].notna() & replies["score_transformer"].notna()].copy()
    unscored = text & replies[col].isna()
    if unscored.any():
        logger.warning(
            "reply population: %d replies with text have no %s and are left out: %s",
            int(unscored.sum()),
            col,
            _per_post(replies[unscored]),
        )
    pop["bucket"] = stance_bucket(pop["score_transformer"]).astype(int)
    return pop, draws, n_no_text


def _whole_fresh_posts(lab: pd.DataFrame, pop: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """lab less the fresh draws of each post whose fresh draws in the
    population are not yet labelled to settings.REPLY_FRESH_POOL_MIN_SHARE:
    until a post's batch is collected its only labelled fresh draws are
    those whose label was reused (the flipper cohort), not a random part."""
    fr = draws[(draws["draw_set"] != "stance") & draws["id"].isin(set(pop["id"]))]
    if fr.empty:
        return lab
    drawn = fr.groupby("tracked_slug")["id"].nunique()
    is_fresh = lab["draw_set"] != "stance"
    got = lab[is_fresh].groupby("tracked_slug")["id"].nunique().reindex(drawn.index, fill_value=0)
    short = (got / drawn) < settings.REPLY_FRESH_POOL_MIN_SHARE
    if short.any():
        logger.info(
            "reply population: fresh draws left out until labelled (labelled / drawn): %s",
            ", ".join(f"{s} ({got[s]}/{drawn[s]})" for s in drawn.index[short]),
        )
    return lab[~(is_fresh & lab["tracked_slug"].isin(set(drawn.index[short])))]


def reply_population(
    col: str = "score_opus_distilled",
    model: str = settings.TEACHER_CHECK_MODEL,
    n_boot: int = settings.BOOTSTRAP_N,
    seed: int = settings.BOOTSTRAP_SEED,
    labels_version: str | None = None,
    fresh: bool | None = None,
) -> pd.DataFrame:
    """Per tracked post: the distilled model's population numbers, the Opus
    stance estimated directly from the labelled sample (opus_*: the
    design-weighted mean and pro / anti / neutral shares of the labels), and
    the model-assisted estimate (assisted_*: population model value + the
    sample's weighted Opus-minus-model correction). The Opus labels are the
    reply teacher's label set `labels_version` (default
    settings.REPLY_TEACHER_LABELS_VERSION; stance_local.REPLY_LABEL_SETS: v2
    is the stance sample's file plus the fresh draws').

    The estimand is the replies with text: the population is each post's
    replies with has_text and a `col` score, one row per id (n_no_text counts
    the image / GIF-only ones left out). The sample is the random draws
    (reply_draws; the flipper cohort and an earlier extra batch are not
    random) that are in the population and carry a label of the set, each
    weighted by its draw-time bucket's population size (current valence)
    over its labelled count. The fresh draws join (n_labelled_fresh counts
    them) when `fresh` (default settings.REPLY_POPULATION_FRESH_DRAWS) and
    the set has the fresh file (v2), and for a post only once its fresh
    draws are labelled to settings.REPLY_FRESH_POOL_MIN_SHARE. Two label
    sets compare on the same draws only: v1 and v2b label the first draws
    alone, so set them against v2 with fresh=False, or with
    reply_population_paired. If any of a post's population sits in a bucket
    with no labelled draw, its opus_* and assisted_* are NaN rather than
    renormalised over the covered buckets; `coverage` is the share of its
    population in labelled buckets, and ALL is NaN unless every post is
    covered. CIs resample the labelled rows within each bucket; the model
    terms are a census and carry no sampling error.

    A `col` whose model was fit on the labelled replies is refused (its
    recorded fit, stance_local.reply_column_entry): its errors on the sample
    are in-sample, so the correction would shrink with no gain in accuracy.
    A cross-fit column (`reply-distill --crossfit`) scores each post with a
    model that never saw that post's labels."""
    from src.analysis.stance_local import (
        REPLY_LABEL_SETS,
        read_reply_label_set,
        reply_column_entry,
        reply_labels_path,
    )

    fit = reply_column_entry(col)
    if fit and fit.get("trained_on_reply_labels") and not fit.get("crossfit"):
        raise ValueError(
            f"{col} comes from {fit.get('model_dir')}, fit on the labelled replies "
            "this corrects against: weight a cross-fit column instead "
            "(reply-distill --crossfit)"
        )
    version = labels_version or settings.REPLY_TEACHER_LABELS_VERSION
    labels = read_reply_label_set(model, version)
    pools = "v2-fresh" in REPLY_LABEL_SETS.get(version, ())
    want = settings.REPLY_POPULATION_FRESH_DRAWS if fresh is None else fresh
    pop, draws, n_no_text = _reply_frames(col, want and pools)
    lab = (
        pop.drop(columns="bucket")
        .merge(draws[["id", "bucket", "draw_set"]], on="id")
        .merge(labels[["id", "score_teacher"]], on="id")
    )
    if want and pools:
        lab = _whole_fresh_posts(lab, pop, draws)
    elif want and reply_labels_path(model, "v2-fresh").exists():
        logger.warning(
            "reply population: the %s labels are on the first draws only, while the v2 "
            "figures pool the fresh draws: set the two on the same draws (--first-draws, "
            "or `reply-label report --job v2b` for v2 against v2b, paired)",
            version,
        )
    rng = np.random.default_rng(seed)

    rows = []
    for slug in list(pop["tracked_slug"].dropna().unique()) + ["ALL"]:
        p = pop if slug == "ALL" else pop[pop["tracked_slug"] == slug]
        smp = lab if slug == "ALL" else lab[lab["tracked_slug"] == slug]
        model_pop = {k: float(v.mean()) for k, v in _stance_shares(p[col].values).items()}
        y = _stance_shares(smp["score_teacher"].values)
        f = _stance_shares(smp[col].values)
        arrays = {
            **{f"opus_{k}": y[k] for k in model_pop},
            **{f"assisted_{k}": y[k] - f[k] for k in model_pop},
        }
        coverage, reps, uncovered = _weighted(p, smp, arrays, n_boot, rng)
        row = {
            "post": slug,
            "labels": version,
            "n_replies": len(p),
            "n_no_text": int(n_no_text.sum() if slug == "ALL" else n_no_text.get(slug, 0)),
            "n_labelled": int(smp["id"].nunique()),
            "n_labelled_fresh": int(smp.loc[smp["draw_set"] != "stance", "id"].nunique()),
            "coverage": coverage,
        }
        for k, v in model_pop.items():
            row[f"model_{k}"] = v
        if reps is None:
            logger.warning(
                "reply population: %s has %.1f%% of its replies in buckets with no "
                "labelled draw (%s); Opus estimates left NaN",
                slug,
                100 * (1 - coverage),
                ", ".join(uncovered),
            )
            reps = {name: np.full(1 + n_boot, np.nan) for name in arrays}
        else:
            for k, v in model_pop.items():
                reps[f"assisted_{k}"] += v
        for name, v in reps.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = _ci(v)
        rows.append(row)
    return pd.DataFrame(rows)


def reply_population_paired(
    a: str = "v2",
    b: str = "v2b",
    col: str = "score_opus_distilled",
    model: str = settings.TEACHER_CHECK_MODEL,
    n_boot: int = settings.BOOTSTRAP_N,
    seed: int = settings.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Two reply label sets' direct population estimates on the same draws:
    the first random draws (not the fresh ones) that carry a label of both,
    weighted as reply_population weighs them, with one shared within-bucket
    resample, so each post's b - a difference (diff_*) gets a paired CI
    and a sampling shift cannot pass for the prompt's effect. Per post and
    ALL: {a}_* and {b}_* mean and pro / anti / neutral shares, diff_*. No
    API."""
    from src.analysis.stance_local import read_reply_label_set

    pop, draws, _ = _reply_frames(col, fresh=False)
    la = read_reply_label_set(model, a)[["id", "score_teacher"]].rename(
        columns={"score_teacher": "y_a"}
    )
    lb = read_reply_label_set(model, b)[["id", "score_teacher"]].rename(
        columns={"score_teacher": "y_b"}
    )
    lab = (
        pop.drop(columns="bucket")
        .merge(draws[["id", "bucket"]], on="id")
        .merge(la, on="id")
        .merge(lb, on="id")
    )
    rng = np.random.default_rng(seed)
    rows = []
    for slug in list(pop["tracked_slug"].dropna().unique()) + ["ALL"]:
        p = pop if slug == "ALL" else pop[pop["tracked_slug"] == slug]
        smp = lab if slug == "ALL" else lab[lab["tracked_slug"] == slug]
        ya, yb = _stance_shares(smp["y_a"].values), _stance_shares(smp["y_b"].values)
        arrays = {
            **{f"{a}_{k}": ya[k] for k in ya},
            **{f"{b}_{k}": yb[k] for k in yb},
            **{f"diff_{k}": yb[k] - ya[k] for k in ya},
        }
        coverage, reps, uncovered = _weighted(p, smp, arrays, n_boot, rng)
        if reps is None:
            logger.warning(
                "reply population %s vs %s: %s has %.1f%% of its replies in buckets with no "
                "draw labelled by both (%s); left NaN",
                a,
                b,
                slug,
                100 * (1 - coverage),
                ", ".join(uncovered),
            )
            reps = {name: np.full(1 + n_boot, np.nan) for name in arrays}
        row = {
            "post": slug,
            "labels": f"{b} - {a}",
            "n_replies": len(p),
            "n_labelled": int(smp["id"].nunique()),
            "coverage": coverage,
        }
        for name, v in reps.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = _ci(v)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Fresh reply draws and the reply categories ─────────────────────


def fresh_draws_path() -> Path:
    """The fresh draws' manifest, next to REPLY_DRAWS_MANIFEST."""
    return settings.PROCESSED_DIR / settings.REPLY_FRESH_DRAWS_NAME


def read_fresh_draws() -> pd.DataFrame:
    path = fresh_draws_path()
    if not path.exists():
        return pd.DataFrame(columns=["id", "tracked_slug", "bucket", "draw_set"])
    d = pd.read_parquet(path)
    return d.assign(id=d["id"].astype(str), bucket=d["bucket"].astype(int))


def _allocate(n: int, sizes: pd.Series, room: pd.Series) -> dict[int, int]:
    """n split across buckets in proportion to `sizes` (largest remainder),
    none past its `room`; what a full bucket cannot take goes to the rest."""
    alloc = dict.fromkeys(sizes.index, 0)
    left = min(n, int(room.reindex(sizes.index, fill_value=0).sum()))
    while left > 0:
        open_ = [b for b in sizes.index if alloc[b] < room.get(b, 0)]
        share = sizes[open_] / sizes[open_].sum() * left
        whole = np.floor(share).astype(int)
        order = (share - whole).sort_values(ascending=False, kind="stable").index
        add = whole.to_dict()
        for b in order[: left - int(whole.sum())]:
            add[b] += 1
        for b in open_:
            alloc[b] += min(add[b], int(room.get(b, 0)) - alloc[b])
        new_left = min(n, int(room.reindex(sizes.index, fill_value=0).sum())) - sum(alloc.values())
        if new_left == left:
            break
        left = new_left
    return {int(b): int(k) for b, k in alloc.items()}


def draw_fresh_sample(
    replies: pd.DataFrame | None = None,
    n_per_post: int = settings.REPLY_FRESH_PER_POST,
    seed: int = settings.REPLY_FRESH_SEED,
    write: bool = False,
    check=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fresh draws of every post, a per post x bucket count table): for each
    tracked post the manifest lacks, n_per_post replies with text, split
    across the valence buckets in proportion to the post's replies with text
    in each, drawn within a bucket from those not already first draws
    (REPLY_DRAWS_MANIFEST) as the smallest seeded hashes of their ids. A
    reply that already carries a v2 label (the flipper cohort, an earlier
    extra batch) can be drawn: its label is reused, never paid again. A post
    the manifest holds is frozen and never redrawn. With `write` the new
    draws are recorded (temp file, then rename) before anything is sent,
    once `check(manifest)` (when given: reply_labels.refuse_fresh_run, the
    run's calls and dollar ceiling against its caps) has not raised.

    The run is sized by its caps: n_per_post above settings
    REPLY_FRESH_PER_POST is refused, and so is recording a manifest of more
    draws than REPLY_FRESH_MAX_CALLS; a bigger run is a deliberate settings
    change."""
    if n_per_post > settings.REPLY_FRESH_PER_POST:
        raise ValueError(
            f"{n_per_post} fresh draws a post is over REPLY_FRESH_PER_POST "
            f"({settings.REPLY_FRESH_PER_POST}) in config/settings.py: raise it deliberately"
        )
    if replies is None:
        replies = pd.read_parquet(
            settings.REPLY_SENTIMENT_OUTPUT,
            columns=["id", "tracked_slug", "text", "score_transformer"],
        )
    r = replies.assign(id=replies["id"].astype(str)).drop_duplicates("id")
    r = r[has_text(r["text"]) & r["score_transformer"].notna() & r["tracked_slug"].notna()]
    r = r.assign(bucket=stance_bucket(r["score_transformer"]).astype(int))
    first = set()
    if settings.REPLY_DRAWS_MANIFEST.exists():
        first = set(
            pd.read_parquet(settings.REPLY_DRAWS_MANIFEST, columns=["id"])["id"].astype(str)
        )
    have = read_fresh_draws()
    frozen = set(have["tracked_slug"])
    picked, summary = [], []
    for slug in sorted(r["tracked_slug"].unique()):
        f = r[r["tracked_slug"] == slug]
        sizes = f.groupby("bucket").size()
        elig = f[~f["id"].isin(first)]
        room = elig.groupby("bucket").size()
        if slug in frozen:
            got = have[have["tracked_slug"] == slug].groupby("bucket").size()
            alloc = {int(b): int(got.get(b, 0)) for b in sizes.index}
        else:
            alloc = _allocate(n_per_post, sizes, room)
            for b, k in alloc.items():
                e = elig[elig["bucket"] == b]
                key = e["id"].map(lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
                picked.append(
                    e.assign(_key=key).sort_values("_key").head(k)[["id", "tracked_slug", "bucket"]]
                )
        for b in sizes.index:
            summary.append(
                {
                    "post": slug,
                    "bucket": STANCE_BUCKET_NAMES[int(b)],
                    "with_text": int(sizes[b]),
                    "first_draws": int(sizes[b] - room.get(b, 0)),
                    "drawn": alloc.get(int(b), 0),
                    "frozen": slug in frozen,
                }
            )
    new = (
        pd.concat(picked, ignore_index=True)
        if picked
        else pd.DataFrame(columns=["id", "tracked_slug", "bucket"])
    )
    new = new.assign(
        draw_set=f"fresh-{seed}",
        seed=seed,
        n_per_post=n_per_post,
        recorded_at=pd.Timestamp.now(tz="UTC"),
    )
    out = pd.concat([have, new], ignore_index=True) if len(have) else new
    if write and len(new):
        if len(out) > settings.REPLY_FRESH_MAX_CALLS:
            raise ValueError(
                f"{len(out)} fresh draws would be recorded, over REPLY_FRESH_MAX_CALLS "
                f"({settings.REPLY_FRESH_MAX_CALLS}) in config/settings.py: raise it "
                "deliberately, or draw fewer a post"
            )
        if check is not None:
            check(out)
        path = fresh_draws_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(path)
        logger.info(
            "fresh reply draws: recorded %d draws for %s -> %s", len(new), _per_post(new), path
        )
    return out.reset_index(drop=True), pd.DataFrame(summary)


def reply_category_population(
    model: str = settings.TEACHER_CHECK_MODEL,
    n_boot: int = settings.BOOTSTRAP_N,
    seed: int = settings.BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Per tracked post (and ALL) and category (event_study.STANCE_CATEGORIES):
    the category's share of the post's replies with text, from the Opus
    categories (reply_labels job cat-v2) on the first random draws only
    (REPLY_DRAWS_MANIFEST: not the flipper cohort, not the fresh draws),
    weighted as reply_population weights them, with within-bucket bootstrap
    CIs; and the same for the Haiku categories `stance` gave those draws
    (event_study.STANCE_PROMPT, no parent post), with the same weights and
    resamples. NaN where a bucket has no labelled draw."""
    from src.analysis.event_study import STANCE_CATEGORIES, STANCE_OUTPUT
    from src.analysis.stance_local import reply_labels_path

    replies = pd.read_parquet(
        settings.REPLY_SENTIMENT_OUTPUT, columns=["id", "tracked_slug", "text", "score_transformer"]
    )
    replies["id"] = replies["id"].astype(str)
    draws = reply_draws(replies)
    draws = draws.assign(id=draws["id"].astype(str), bucket=draws["bucket"].astype(int))
    replies = replies.drop_duplicates("id")
    pop = replies[has_text(replies["text"]) & replies["score_transformer"].notna()].copy()
    pop["bucket"] = stance_bucket(pop["score_transformer"]).astype(int)
    cats = pd.read_parquet(reply_labels_path(model, "cat-v2"), columns=["id", "category"])
    cats = cats.assign(id=cats["id"].astype(str)).dropna().drop_duplicates("id")
    haiku = pd.read_parquet(STANCE_OUTPUT, columns=["id", "stance"])
    haiku = haiku.assign(id=haiku["id"].astype(str)).drop_duplicates("id")
    lab = (
        pop[["id", "tracked_slug"]]
        .merge(draws[["id", "bucket"]], on="id")
        .merge(cats, on="id")
        .merge(haiku, on="id", how="left")
    )
    rng = np.random.default_rng(seed)
    rows = []
    for slug in list(pop["tracked_slug"].dropna().unique()) + ["ALL"]:
        p = pop if slug == "ALL" else pop[pop["tracked_slug"] == slug]
        smp = lab if slug == "ALL" else lab[lab["tracked_slug"] == slug]
        judged = smp["stance"].isin(STANCE_CATEGORIES).values
        arrays = {}
        for c in STANCE_CATEGORIES:
            arrays[f"opus:{c}"] = (smp["category"] == c).values.astype(float)
            arrays[f"haiku:{c}"] = np.where(judged, (smp["stance"] == c).values, np.nan)
        coverage, reps, _ = _weighted(p, smp, arrays, n_boot, rng, nan_ok=True)
        for c in STANCE_CATEGORIES:
            row = {
                "post": slug,
                "category": c,
                "n_replies": len(p),
                "n_labelled": int(smp["id"].nunique()),
                "coverage": coverage,
            }
            for kind in ("opus", "haiku"):
                v = reps[f"{kind}:{c}"] if reps is not None else np.full(1 + n_boot, np.nan)
                row[f"{kind}_share"], row[f"{kind}_lo"], row[f"{kind}_hi"] = _ci(v)
            rows.append(row)
    return pd.DataFrame(rows)


# ── Opus test-retest ───────────────────────────────────────────────

def teacher_retest(model: str = settings.TEACHER_CHECK_MODEL) -> tuple[pd.DataFrame, dict]:
    """The teacher check (direct API, 2026-09-18) and the batch relabel scored
    the same posts independently with the same prompt and effort. Agreement by
    tier plus the per-label noise SD implied by the differences."""
    from src.analysis.relabel import labels_path
    from src.analysis.stance_local import agreement_by_tier
    from src.superseded import superseded_path

    tag = model.replace("/", "_")
    direct = pd.read_parquet(settings.PROCESSED_DIR / f"teacher_check_{tag}.parquet")
    batch = pd.read_parquet(labels_path(model))
    posts = pd.read_parquet(settings.SENTIMENT_OUTPUT, columns=["id", "tier"])
    for f in (direct, batch, posts):
        f["id"] = f["id"].astype(str)
    # Both labels of a pair must come from the same text: a post whose text was
    # backfilled since (x-backfill-text) pairs with the batch label made from
    # the cut, now in the superseded archive (its first entry), not its relabel.
    if superseded_path(labels_path(model)).exists():
        old = pd.read_parquet(superseded_path(labels_path(model)))
        old = old.assign(id=old["id"].astype(str)).drop_duplicates("id")
        batch = pd.concat([old, batch[~batch["id"].isin(old["id"])]], ignore_index=True)
    j = (direct.rename(columns={"score_teacher": "direct"})[["id", "direct"]]
         .merge(batch.rename(columns={"score_teacher": "batch"})[["id", "batch"]], on="id")
         .merge(posts, on="id").dropna(subset=["direct", "batch"]))
    diff = j["direct"] - j["batch"]
    summary = {"n": int(len(j)), "identical": float((diff == 0).mean()),
               "within_0.1": float((diff.abs() <= 0.1).mean()),
               "noise_sd_per_label": float(diff.std() / np.sqrt(2)),
               "mean_shift": float(diff.mean())}
    return agreement_by_tier(j, "batch", "direct"), summary
