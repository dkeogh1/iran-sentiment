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

Everything here reads cached parquets; nothing calls an API.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import settings
from config.timeline import PHASES

logger = logging.getLogger(__name__)

STANCE_BAND = 0.05  # |score| above this counts as pro / anti (sign3 in stance_local)


# ── Phase / topic tagging ──────────────────────────────────────────

def assign_phase(created_at: pd.Series, phases=PHASES) -> pd.Series:
    """Phase label per row (UTC date, both ends inclusive); NaN outside."""
    d = pd.to_datetime(created_at, utc=True).dt.date
    out = pd.Series(np.nan, index=created_at.index, dtype=object)
    for label, start, end in phases:
        out[(d >= start) & (d <= end)] = label
    return out


def on_topic(text: pd.Series, pattern: str = settings.WAR_TOPIC_PATTERN) -> pd.Series:
    return text.fillna("").str.contains(pattern, case=False, regex=True)


def war_flag(d: pd.DataFrame, source: str = settings.TOPIC_SOURCE) -> pd.Series:
    """About-the-war flag per row: the Haiku topic label where one exists
    (source="llm"), the keyword pattern otherwise."""
    kw = on_topic(d["text"])
    if source != "llm":
        return kw
    from src.analysis.topic_label import load_labels
    lab = load_labels()
    if lab is None:
        logger.warning("no topic labels yet (run `topic-label`); using the keyword filter")
        return kw
    m = d["id"].astype(str).map(lab.set_index("id")["about_war"])
    return m.where(m.notna(), kw).astype(bool)


def prepare(df: pd.DataFrame, score_col: str, group: str = "tier",
            topic_source: str = settings.TOPIC_SOURCE, phases=PHASES) -> pd.DataFrame:
    """Rows with a score inside a phase, tagged with phase / topic / day.
    Pass phases=WHOLE_WAR for one cell per group over the whole window."""
    d = df[df[score_col].notna()].copy()
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

def reply_population(col: str = "score_opus_distilled", model: str = settings.TEACHER_CHECK_MODEL,
                     n_boot: int = settings.BOOTSTRAP_N, seed: int = settings.BOOTSTRAP_SEED) -> pd.DataFrame:
    """Per tracked post: the distilled model's population numbers, the Opus
    stance estimated directly from the labelled sample, and the model-assisted
    estimate (population model value + the sample's weighted Opus-minus-model
    correction). Only the random bucket draws of stratified_stance_sample are
    used (the flipper cohort and an earlier extra batch are not random), each
    weighted by its bucket's population size over its sample size. CIs
    resample the labelled rows within each bucket; the model terms are a
    census and carry no sampling error."""
    from src.analysis.event_study import STANCE_BUCKETS, bucket_draws

    replies = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    replies["id"] = replies["id"].astype(str)
    replies = replies[replies[col].notna()]
    labels = pd.read_parquet(settings.PROCESSED_DIR / f"teacher_labels_replies_{model.replace('/', '_')}.parquet")
    labels["id"] = labels["id"].astype(str)
    draws = set(bucket_draws(replies, score_col="score_transformer")["id"])
    lab = replies[replies["id"].isin(draws)].merge(labels[["id", "score_teacher"]], on="id")

    def bucket(s: pd.Series) -> pd.Series:
        edges = [STANCE_BUCKETS[0][0]] + [hi for _, hi in STANCE_BUCKETS]
        return pd.cut(s, edges, right=False, labels=False)

    replies["bucket"] = bucket(replies["score_transformer"])
    lab["bucket"] = bucket(lab["score_transformer"])
    rng = np.random.default_rng(seed)

    def stats(y: np.ndarray) -> dict[str, np.ndarray]:
        return {"mean": y, "pro": (y > STANCE_BAND).astype(float), "anti": (y < -STANCE_BAND).astype(float)}

    rows = []
    for slug in list(replies["tracked_slug"].dropna().unique()) + ["ALL"]:
        pop = replies if slug == "ALL" else replies[replies["tracked_slug"] == slug]
        smp = lab if slug == "ALL" else lab[lab["tracked_slug"] == slug]
        strata = smp.groupby(["tracked_slug", "bucket"]).indices
        n_pop = pop.groupby(["tracked_slug", "bucket"]).size()
        model_pop = {k: float(v.mean()) for k, v in stats(pop[col].values).items()}
        row = {"post": slug, "n_replies": int(len(pop)), "n_labelled": int(len(smp))}
        for k, v in model_pop.items():
            row[f"model_{k}"] = v
        # replicate 0 = observed sample, then n_boot within-bucket resamples
        reps = {f"{kind}_{k}": np.zeros(1 + n_boot) for kind in ("opus", "assisted") for k in model_pop}
        total = float(n_pop[list(strata)].sum())
        for key, idx in strata.items():
            wt = n_pop[key] / total
            y = stats(smp["score_teacher"].values[idx])
            f = stats(smp[col].values[idx])
            pick = np.vstack([np.arange(len(idx)), rng.integers(0, len(idx), (n_boot, len(idx)))])
            for k in model_pop:
                reps[f"opus_{k}"] += wt * y[k][pick].mean(axis=1)
                reps[f"assisted_{k}"] += wt * (y[k] - f[k])[pick].mean(axis=1)
        for k in model_pop:
            reps[f"assisted_{k}"] += model_pop[k]
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
