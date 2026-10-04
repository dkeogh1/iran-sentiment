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
  teacher_retest     Opus labelled the same 497 posts twice (direct teacher
                     check, then the batch relabel): how much does it move?

Everything here reads cached parquets; nothing calls an API. One write:
reply_population (and so export-web) records the reply sample's draws in
REPLY_DRAWS_MANIFEST the first time it sees a post `stance` has sampled.
"""

from __future__ import annotations

import logging

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
    pattern is loose (it misses unnamed strikes, catches "culture war")."""
    kw = on_topic(d["text"])
    if source == "keyword":
        return kw
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
    day. Rows without text (has_text: 1,622 of Trump's Truth Social posts,
    image / video posts and ReTruths whose text the collector used to drop)
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


def reply_draws(replies: pd.DataFrame, n_per_bucket: int = 50,
                score_col: str = "score_transformer") -> pd.DataFrame:
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


def reply_population(col: str = "score_opus_distilled", model: str = settings.TEACHER_CHECK_MODEL,
                     n_boot: int = settings.BOOTSTRAP_N, seed: int = settings.BOOTSTRAP_SEED,
                     labels_version: str | None = None) -> pd.DataFrame:
    """Per tracked post: the distilled model's population numbers, the Opus
    stance estimated directly from the labelled sample, and the model-assisted
    estimate (population model value + the sample's weighted Opus-minus-model
    correction). The Opus labels are the reply teacher's `labels_version`
    (default settings.REPLY_TEACHER_LABELS_VERSION; stance_local.reply_labels_path).

    The estimand is the replies with text: the population is each post's
    replies with has_text and a `col` score, one row per id (n_no_text counts
    the image / GIF-only ones left out). The sample is the random bucket draws
    of stratified_stance_sample (reply_draws; the flipper cohort and an
    earlier extra batch are not random) that are in the population and carry
    an Opus label, each weighted by its draw-time bucket's population size
    (current valence) over its labelled count. If any of a post's population
    sits in a bucket with no labelled draw, its opus_* and assisted_* are NaN
    rather than renormalised over the covered buckets; `coverage` is the
    share of its population in labelled buckets, and ALL is NaN unless every
    post is covered. CIs resample the labelled rows within each bucket; the
    model terms are a census and carry no sampling error."""
    replies = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    replies["id"] = replies["id"].astype(str)
    draws = reply_draws(replies)                     # fixed before any filtering
    draws = draws.assign(id=draws["id"].astype(str), bucket=draws["bucket"].astype(int))
    replies = replies.drop_duplicates("id")          # a few ids were collected twice
    text = has_text(replies["text"])
    n_no_text = (~text).groupby(replies["tracked_slug"]).sum()
    pop = replies[text & replies[col].notna() & replies["score_transformer"].notna()].copy()
    pop["bucket"] = stance_bucket(pop["score_transformer"]).astype(int)
    from src.analysis.stance_local import reply_labels_path
    version = labels_version or settings.REPLY_TEACHER_LABELS_VERSION
    labels = pd.read_parquet(reply_labels_path(model, version))
    labels["id"] = labels["id"].astype(str)
    labels = labels[labels["score_teacher"].notna()].drop_duplicates("id")
    lab = (pop.drop(columns="bucket").merge(draws[["id", "bucket"]], on="id")
              .merge(labels[["id", "score_teacher"]], on="id"))
    rng = np.random.default_rng(seed)

    def stats(y: np.ndarray) -> dict[str, np.ndarray]:
        return {"mean": y, "pro": (y > STANCE_BAND).astype(float), "anti": (y < -STANCE_BAND).astype(float)}

    rows = []
    for slug in list(pop["tracked_slug"].dropna().unique()) + ["ALL"]:
        p = pop if slug == "ALL" else pop[pop["tracked_slug"] == slug]
        smp = lab if slug == "ALL" else lab[lab["tracked_slug"] == slug]
        strata = smp.groupby(["tracked_slug", "bucket"]).indices
        n_pop = p.groupby(["tracked_slug", "bucket"]).size()
        total = float(n_pop.sum())
        coverage = float(n_pop[n_pop.index.isin(list(strata))].sum()) / total if total else 0.0
        model_pop = {k: float(v.mean()) for k, v in stats(p[col].values).items()}
        row = {"post": slug, "labels": version, "n_replies": len(p),
               "n_no_text": int(n_no_text.sum() if slug == "ALL" else n_no_text.get(slug, 0)),
               "n_labelled": int(smp["id"].nunique()), "coverage": coverage}
        for k, v in model_pop.items():
            row[f"model_{k}"] = v
        # replicate 0 = observed sample, then n_boot within-bucket resamples
        reps = {f"{kind}_{k}": np.zeros(1 + n_boot) for kind in ("opus", "assisted") for k in model_pop}
        if coverage < 1:
            uncovered = [f"{s}/{STANCE_BUCKET_NAMES[b]}" for s, b in n_pop.index
                         if (s, b) not in strata]
            logger.warning("reply population: %s has %.1f%% of its replies in buckets with no "
                           "labelled draw (%s); Opus estimates left NaN",
                           slug, 100 * (1 - coverage), ", ".join(uncovered))
            reps = {name: np.full(1 + n_boot, np.nan) for name in reps}
        else:
            for key, idx in strata.items():
                wt = n_pop.get(key, 0) / total    # a draw-time bucket since emptied weighs nothing
                y = stats(smp["score_teacher"].values[idx])
                f = stats(smp[col].values[idx])
                pick = np.vstack([np.arange(len(idx)),
                                  rng.integers(0, len(idx), (n_boot, len(idx)))])
                for k in model_pop:
                    reps[f"opus_{k}"] += wt * y[k][pick].mean(axis=1)
                    reps[f"assisted_{k}"] += wt * (y[k] - f[k])[pick].mean(axis=1)
            for k, v in model_pop.items():
                reps[f"assisted_{k}"] += v
        for name, v in reps.items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = _ci(v)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Opus test-retest ───────────────────────────────────────────────

def teacher_retest(model: str = settings.TEACHER_CHECK_MODEL) -> tuple[pd.DataFrame, dict]:
    """The teacher check (direct API, 2026-09-18) and the batch relabel scored
    the same posts independently with the same prompt and effort. Agreement by
    tier plus the per-label noise SD implied by the differences."""
    from src.analysis.relabel import labels_path
    from src.analysis.stance_local import agreement_by_tier

    tag = model.replace("/", "_")
    direct = pd.read_parquet(settings.PROCESSED_DIR / f"teacher_check_{tag}.parquet")
    batch = pd.read_parquet(labels_path(model))
    posts = pd.read_parquet(settings.SENTIMENT_OUTPUT, columns=["id", "tier"])
    for f in (direct, batch, posts):
        f["id"] = f["id"].astype(str)
    j = (direct.rename(columns={"score_teacher": "direct"})[["id", "direct"]]
         .merge(batch.rename(columns={"score_teacher": "batch"})[["id", "batch"]], on="id")
         .merge(posts, on="id").dropna(subset=["direct", "batch"]))
    diff = j["direct"] - j["batch"]
    summary = {"n": int(len(j)), "identical": float((diff == 0).mean()),
               "within_0.1": float((diff.abs() <= 0.1).mean()),
               "noise_sd_per_label": float(diff.std() / np.sqrt(2)),
               "mean_shift": float(diff.mean())}
    return agreement_by_tier(j, "batch", "direct"), summary
