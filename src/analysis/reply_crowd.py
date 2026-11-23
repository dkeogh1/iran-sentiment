"""
Same accounts or a different crowd? (`reply-crowd`; docs/decisions.md 2026-11-23)

The reply audience's Opus stance (v2 labels on the random draws, first and
fresh pooled) split by fixed census groups of accounts, never by a label or
a score:

  R1   returners (the account replied to another tracked post) against
       one-off repliers on the same post, per post and within posts
  R2   the move from the earlier posts to the late phase's split into
       "accounts seen in both periods" (same_accounts) and turnover
       (late_only_term + earlier_only_term), post-standardised with "both"
       as the anchor: change = same_accounts + turnover in every replicate.
       same_accounts is the post effect plus any change of mind, which this
       cannot separate; the split is accounting, not attribution.
  pairs every ordered pair of posts (q before p) split the same way on the
       accounts that replied to both, and the noise-corrected RMS of each
       term over the pairs of a kind: the reference the R2 terms are read
       against
  panel exploratory: post-centred label correlation of the same account on
       two posts (account-cluster bootstrap); never a headline
  census exact counts: prior repliers, group sizes, account ages, tiers

Estimator of record: post-stratified domain means on the census cell totals
(post x RoBERTa bucket x group), collapsing to a Hajek ratio over the
domain's buckets when a cell has fewer than REPLY_CROWD_MIN_CELL_DRAWS draws,
NaN (named) when a cell with population has none. A GREG with an estimated
slope on the census `--col` score is a check. One replicate-weight matrix
(stratum resample with the finite-population correction, per-post rng
streams) serves every estimate, so identities and contrasts hold replicate
by replicate. Aggregates only; reads data, writes nothing but its CSVs (the
CLI). No API.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import settings, timeline
from config.tracked_posts import TRACKED_POSTS
from src.analysis.inference import (
    _ci,
    _reply_frames,
    _stance_shares,
    _whole_fresh_posts,
    assign_phase,
    has_text,
    replier_key,
    replier_tiers,
)

logger = logging.getLogger(__name__)

OUTCOMES = ("mean", "pro", "anti", "neutral")
R1_GROUPS = ("returner", "one_off")
R1_THREE = ("ret_1", "ret_2plus", "one_off")
R2_GROUPS = ("both", "earlier_only", "late_only")
PAIR_GROUPS = ("both_qp", "q_only", "p_only")
TERMS = (
    "earlier",
    "late",
    "change",
    "both_earlier",
    "both_late",
    "same_accounts",
    "late_only_term",
    "earlier_only_term",
    "turnover",
    "post_mix_earlier",
    "post_mix_late",
    "did",
    "did_sd",
)
HEADLINE_TERMS = ("change", "same_accounts", "turnover", "did")
PAIR_TERMS = ("pair_change", "same_accounts", "composition")
LOYALTY_LEVELS = ("low", "loyal")  # medium and high merged
_SHARE_KS = ("pro", "anti", "neutral")
_SHARE_AUX = ("pro", "anti")  # GREG's share auxiliary; neutral is in the cell intercepts
DRAW_SETS = ("all", "first", "fresh")
_URL = re.compile(r"https?://\S+|www\.\S+")


# ── Posts and contrasts ────────────────────────────────────────────


class Contrast(NamedTuple):
    name: str
    early: list[str]
    late: list[str]


def _tracked() -> dict:
    return {p.slug: (p.event_date, i) for i, p in enumerate(TRACKED_POSTS)}


def post_order(slugs) -> list[str]:
    """The slugs in time order (event_date, then TRACKED_POSTS order);
    ValueError on a slug TRACKED_POSTS lacks."""
    tracked = _tracked()
    missing = sorted(set(slugs) - set(tracked))
    if missing:
        raise ValueError(f"not in config/tracked_posts.TRACKED_POSTS: {', '.join(missing)}")
    return sorted(set(slugs), key=lambda s: tracked[s])


def _phase_of(slugs: list[str]) -> pd.Series:
    when = pd.Series([pd.Timestamp(_tracked()[s][0]) for s in slugs], index=slugs)
    return assign_phase(when, timeline.PHASES)


def contrast_posts(early_phases, late_phases, slugs) -> tuple[list[str], list[str]]:
    """(earlier posts, late posts), each in time order: the tracked posts in
    `slugs` whose event_date falls in one of `early_phases` / `late_phases`
    (labels of config/timeline.PHASES, read when called). ValueError on a
    label not in PHASES, the first phase used as late, overlapping sides, an
    earlier phase not before every late one, an empty side or a slug
    TRACKED_POSTS lacks. Posts dated after the late phase are left out with
    a log line, and so is a base phase with no post."""
    labels = [p[0] for p in timeline.PHASES]
    early_phases, late_phases = tuple(early_phases), tuple(late_phases)
    bad = [x for x in (*early_phases, *late_phases) if x not in labels]
    if bad:
        raise ValueError(f"not a phase in config/timeline.PHASES: {bad}")
    if labels[0] in late_phases:
        raise ValueError(f"the first phase ({labels[0]!r}) has no earlier phase: not a late side")
    if set(early_phases) & set(late_phases):
        raise ValueError(f"a phase on both sides: {sorted(set(early_phases) & set(late_phases))}")
    if (
        early_phases
        and late_phases
        and max(map(labels.index, early_phases)) >= min(map(labels.index, late_phases))
    ):
        raise ValueError(
            f"the earlier phases must all precede the late phase: {early_phases} -> {late_phases}"
        )
    order = post_order(slugs)
    phase = _phase_of(order)
    early = [s for s in order if phase[s] in early_phases]
    late = [s for s in order if phase[s] in late_phases]
    last = max(labels.index(x) for x in late_phases)
    after = [
        s
        for s in order
        if s not in early
        and s not in late
        and isinstance(phase[s], str)
        and labels.index(phase[s]) > last
    ]
    if after:
        logger.info("reply crowd: posts dated after %s left out: %s", late_phases, after)
    for x in early_phases:
        if not (phase == x).any():
            logger.info("reply crowd: base phase %r has no tracked post with replies", x)
    if not early or not late:
        raise ValueError(
            f"reply crowd: an empty side ({early_phases} -> {late_phases}: "
            f"{len(early)} earlier, {len(late)} late posts)"
        )
    return early, late


def default_contrast(slugs, late_phase: str | None = None) -> Contrast:
    """Every phase before the late phase (settings.REPLY_CROWD_LATE_PHASE by
    default, never PHASES[-1]) against it."""
    late_phase = late_phase or settings.REPLY_CROWD_LATE_PHASE
    labels = [p[0] for p in timeline.PHASES]
    if late_phase not in labels:
        raise ValueError(f"not a phase in config/timeline.PHASES: {late_phase!r}")
    early, late = contrast_posts(labels[: labels.index(late_phase)], (late_phase,), slugs)
    return Contrast("all earlier", early, late)


def crowd_contrasts(slugs, late_phase: str | None = None) -> list[Contrast]:
    """The record (all earlier phases -> the late phase), each earlier phase
    with posts on its own, the near base (settings.REPLY_CROWD_NEAR_BASE)
    and the benchmark (settings.REPLY_CROWD_BENCHMARK, no late phase in it).
    An optional contrast that does not fit the posts is logged and skipped."""
    main = default_contrast(slugs, late_phase)
    late_phase = late_phase or settings.REPLY_CROWD_LATE_PHASE
    out = [main]
    phase = _phase_of(post_order(slugs))
    labels = [p[0] for p in timeline.PHASES]
    for label in labels[: labels.index(late_phase)]:
        if (phase == label).any():
            out.append(Contrast(f"base: {label}", *contrast_posts((label,), (late_phase,), slugs)))
    for name, (early, late) in (
        ("near", (settings.REPLY_CROWD_NEAR_BASE, (late_phase,))),
        ("benchmark", settings.REPLY_CROWD_BENCHMARK),
    ):
        try:
            out.append(Contrast(name, *contrast_posts(early, late, slugs)))
        except ValueError as e:
            logger.info("reply crowd: %s contrast skipped: %s", name, e)
    return out


# ── Participation and groups ───────────────────────────────────────


def participation(replies: pd.DataFrame, on: str = "all") -> pd.DataFrame:
    """id, tracked_slug, replier (replier_key: the account id), has_text, m
    (the account's replies with text on the post) per reply, one row per id.
    on="all" keeps every collected reply (the record: group membership counts
    a reply with no text too), "text" only those with text. A reply with no
    account record has replier None: counted, in no group."""
    if on not in ("all", "text"):
        raise ValueError(f"participation on {on!r}: 'all' or 'text'")
    r = replies.drop_duplicates("id")
    text = r["has_text"].astype(bool) if "has_text" in r else has_text(r["text"])
    acct = r["account"] if "account" in r else pd.Series(None, index=r.index, dtype=object)
    out = pd.DataFrame(
        {
            "id": r["id"].astype(str).to_numpy(),
            "tracked_slug": r["tracked_slug"].to_numpy(),
            "replier": acct.map(replier_key).astype(object).to_numpy(),
            "has_text": text.to_numpy(dtype=bool),
        }
    )
    if "created" in r:
        out["created"] = r["created"].to_numpy()
    known = out["replier"].notna() & out["has_text"]
    m = out[known].groupby(["replier", "tracked_slug"]).size()
    key = pd.MultiIndex.from_arrays([out["replier"], out["tracked_slug"]])
    out["m"] = m.reindex(key).fillna(0).astype(int).to_numpy()
    return out[out["has_text"]].reset_index(drop=True) if on == "text" else out


def returner_groups(part: pd.DataFrame) -> pd.DataFrame:
    """id, returner (the account replied to at least one other tracked post
    in `part`, earlier or later), n_other_posts; known accounts only."""
    k = part[part["replier"].notna()]
    n_posts = k.groupby("replier")["tracked_slug"].nunique()
    other = k["replier"].map(n_posts).astype(int) - 1
    return pd.DataFrame(
        {
            "id": k["id"].to_numpy(),
            "returner": (other >= 1).to_numpy(),
            "n_other_posts": other.to_numpy(),
        }
    )


def r1_groups(part: pd.DataFrame, three: bool = False) -> pd.Series:
    """id -> returner / one_off (three: ret_1 / ret_2plus / one_off)."""
    g = returner_groups(part)
    if three:
        lab = np.where(
            g["n_other_posts"] >= 2, "ret_2plus", np.where(g["returner"], "ret_1", "one_off")
        )
    else:
        lab = np.where(g["returner"], "returner", "one_off")
    return pd.Series(lab, index=g["id"].to_numpy())


def _two_sided(part: pd.DataFrame, a: list[str], b: list[str], names) -> pd.Series:
    k = part[part["replier"].notna() & part["tracked_slug"].isin([*a, *b])]
    in_a = set(k.loc[k["tracked_slug"].isin(a), "replier"])
    in_b = set(k.loc[k["tracked_slug"].isin(b), "replier"])
    both = k["replier"].isin(in_a) & k["replier"].isin(in_b)
    side = np.where(k["tracked_slug"].isin(a), names[1], names[2])
    return pd.Series(np.where(both, names[0], side), index=k["id"].to_numpy())


def crowd_groups(part: pd.DataFrame, early: list[str], late: list[str]) -> pd.Series:
    """id -> both / earlier_only / late_only for the replies to the
    contrast's posts: relative to these posts only, recomputed per base."""
    return _two_sided(part, list(early), list(late), R2_GROUPS)


def pair_groups(part: pd.DataFrame, q: str, p: str) -> pd.Series:
    """id -> both_qp / q_only / p_only for the replies to posts q and p."""
    return _two_sided(part, [q], [p], PAIR_GROUPS)


# ── Frames ─────────────────────────────────────────────────────────


@dataclass
class CrowdFrames:
    """pop: one row per reply in the population (text, a `col` score and a
    RoBERTa score); lab: its labelled random draws (draw-time bucket, label
    y, draw_set); part: every collected reply's account (participation).
    The text is not kept."""

    pop: pd.DataFrame
    lab: pd.DataFrame
    part: pd.DataFrame
    col: str
    posts: list[str]
    n_unknown: int = 0


def _norm_text(t: str) -> str:
    t = _URL.sub(" ", str(t).lower())
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def dup_text_accounts(replier: pd.Series, slug: pd.Series, text: pd.Series) -> set:
    """Accounts with the same normalised text (lower case, links and
    punctuation removed) on two or more posts."""
    d = pd.DataFrame(
        {"r": replier.to_numpy(), "s": slug.to_numpy(), "t": text.map(_norm_text).to_numpy()}
    )
    d = d[d["r"].notna() & (d["t"] != "")]
    n = d.groupby(["r", "t"])["s"].nunique()
    return set(n[n >= 2].index.get_level_values(0))


def _created(acct: pd.Series) -> pd.Series:
    from src.analysis.event_study import account_created

    return pd.to_datetime(acct.map(account_created), utc=True)


def column_guard(col: str) -> None:
    """reply_population's rule: a column fit on the labelled replies that is
    not a cross-fit is refused (its errors on the sample are in-sample)."""
    from src.analysis.stance_local import reply_column_entry

    fit = reply_column_entry(col)
    if fit and fit.get("trained_on_reply_labels") and not fit.get("crossfit"):
        raise ValueError(
            f"{col} comes from {fit.get('model_dir')}, fit on the labelled replies: "
            "use a cross-fit column (reply-distill --crossfit)"
        )


def _reply_columns() -> list[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(settings.REPLY_SENTIMENT_OUTPUT).schema_arrow.names


def _read_participation(extra: tuple[str, ...] = ()) -> pd.DataFrame:
    names = _reply_columns()
    want = dict.fromkeys(("id", "tracked_slug", "text", "account", *extra))
    cols = [c for c in want if c in names]
    r = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT, columns=cols)
    r["id"] = r["id"].astype(str)
    r = r.drop_duplicates("id")
    r["has_text"] = has_text(r["text"])
    if "account" in r:
        r["created"] = _created(r["account"])
    return r.drop(columns="text")


def _check_column(col: str, raw: pd.DataFrame | None, drawn_posts) -> None:
    """ValueError when `col` is not in the reply parquet (raw None), or when
    a column other than the record's (settings.DISTILLED_SCORER_COL) is
    missing on a reply with text and a RoBERTa score of a post with random
    draws: a partly written column (a cross-fit Job cut short) would shrink
    the population, and the record estimator's base must not move with the
    GREG check's column."""
    if raw is None:
        raise ValueError(
            f"{col} is not in {settings.REPLY_SENTIMENT_OUTPUT.name} yet: run the Job that "
            "writes it and `scripts/k8s/sync-data.sh pull` first"
        )
    if col == settings.DISTILLED_SCORER_COL:
        return
    miss = raw["has_text"] & raw[col].isna() & raw["tracked_slug"].isin(set(drawn_posts))
    if "score_transformer" in raw:
        miss &= raw["score_transformer"].notna()
    if miss.any():
        n = raw.loc[miss, "tracked_slug"].value_counts().sort_index()
        raise ValueError(
            f"{col} is missing on {int(miss.sum())} replies with text of posts with random "
            f"draws ({', '.join(f'{p}: {k}' for p, k in n.items())}): a partly written "
            "column; finish its Job and pull before using it"
        )


def crowd_frames(
    col: str | None = None,
    model: str | None = None,
    draws: str = "all",
    status_on: str | None = None,
    exclude: tuple[str, ...] = (),
) -> CrowdFrames:
    """The population, labelled draws (v2, after _whole_fresh_posts) and
    participation, read once. draws: all / first / fresh. exclude: any of
    heavy, dup_text, new_accounts (robustness)."""
    col = col or settings.REPLY_POPULATION_COL
    from src.analysis.stance_local import read_reply_label_set

    column_guard(col)
    if col not in _reply_columns():
        _check_column(col, None, ())
    labels = read_reply_label_set(model or settings.TEACHER_CHECK_MODEL, "v2")
    pop, drw, _ = _reply_frames(col, fresh=True)
    raw = _read_participation((col, "score_transformer"))
    _check_column(col, raw, drw["tracked_slug"].unique())
    lab = (
        pop[["id", "tracked_slug"]]
        .merge(drw[["id", "bucket", "draw_set"]], on="id")
        .merge(labels[["id", "score_teacher"]], on="id")
    )
    lab = _whole_fresh_posts(lab, pop, drw)
    part = participation(raw, status_on or settings.REPLY_CROWD_STATUS_ON)
    acct = pop["account"] if "account" in pop else pd.Series(None, index=pop.index, dtype=object)
    replier = acct.map(replier_key).astype(object)
    dup = dup_text_accounts(replier, pop["tracked_slug"], pop["text"])
    tiers = replier_tiers(pop.assign(account=acct))
    p = pd.DataFrame(
        {
            "id": pop["id"].astype(str).to_numpy(),
            "tracked_slug": pop["tracked_slug"].to_numpy(),
            "bucket": pop["bucket"].astype(int).to_numpy(),
            "f": pop[col].astype(float).to_numpy(),
            "replier": replier.to_numpy(),
            "created": _created(acct).to_numpy(),
            "loyalty": tiers["loyalty"].to_numpy(),
            "dup_text": replier.isin(dup).to_numpy(),
        }
    )
    unknown = p["replier"].isna()
    if unknown.any():
        logger.warning(
            "reply crowd: %d replies with text have no account: in no group", int(unknown.sum())
        )
    p = p[~unknown].reset_index(drop=True)
    p["m"] = p.groupby(["replier", "tracked_slug"])["id"].transform("size").astype(int)
    lab = (
        lab[["id", "bucket", "draw_set", "score_teacher"]]
        .rename(columns={"score_teacher": "y"})
        .merge(p.drop(columns="bucket"), on="id")
        .reset_index(drop=True)
    )
    fr = CrowdFrames(p, lab, part, col, post_order(p["tracked_slug"].unique()), int(unknown.sum()))
    fr = select_draws(fr, draws)
    for name in exclude:
        if name == "heavy":
            fr = exclude_heavy(fr)
        elif name == "dup_text":
            fr = exclude_dup_text(fr)
        elif name == "new_accounts":
            first = _tracked()[default_contrast(fr.posts).early[0]][0]
            fr = exclude_new_accounts(fr, pd.Timestamp(first, tz="UTC"))
        else:
            raise ValueError(f"unknown exclusion {name!r}")
    return fr


def select_draws(fr: CrowdFrames, draws: str) -> CrowdFrames:
    """The first draws, the fresh draws, or all (two independent samples)."""
    if draws not in DRAW_SETS:
        raise ValueError(f"draws {draws!r}: one of {DRAW_SETS}")
    if draws == "all":
        return fr
    first = fr.lab["draw_set"] == "stance"
    keep = first if draws == "first" else ~first
    return replace(fr, lab=fr.lab[keep].reset_index(drop=True))


def with_participation(fr: CrowdFrames, on: str) -> CrowdFrames:
    """Groups from every reply ("all") or the replies with text ("text")."""
    if on not in ("all", "text"):
        raise ValueError(f"participation on {on!r}: 'all' or 'text'")
    part = fr.part[fr.part["has_text"]].reset_index(drop=True) if on == "text" else fr.part
    return replace(fr, part=part)


def _drop_accounts(fr: CrowdFrames, accounts: set) -> CrowdFrames:
    return replace(
        fr,
        pop=fr.pop[~fr.pop["replier"].isin(accounts)].reset_index(drop=True),
        lab=fr.lab[~fr.lab["replier"].isin(accounts)].reset_index(drop=True),
        part=fr.part[~fr.part["replier"].isin(accounts)].reset_index(drop=True),
    )


def exclude_heavy(fr: CrowdFrames, k: int | None = None) -> CrowdFrames:
    """Drop the replies of accounts with k+ text replies on that post
    (settings.REPLY_CROWD_HEAVY_REPLIES), from population and sample; the
    groups keep them."""
    k = k or settings.REPLY_CROWD_HEAVY_REPLIES
    return replace(
        fr,
        pop=fr.pop[fr.pop["m"] < k].reset_index(drop=True),
        lab=fr.lab[fr.lab["m"] < k].reset_index(drop=True),
    )


def exclude_dup_text(fr: CrowdFrames) -> CrowdFrames:
    """Drop accounts with identical normalised text on two or more posts."""
    return _drop_accounts(fr, set(fr.pop.loc[fr.pop["dup_text"], "replier"]))


def exclude_new_accounts(fr: CrowdFrames, after: pd.Timestamp) -> CrowdFrames:
    """Drop accounts created after `after` (the base's first post: they
    could not have replied to it)."""
    created = fr.part.get("created", fr.pop["created"])
    rep = fr.part["replier"] if "created" in fr.part else fr.pop["replier"]
    return _drop_accounts(fr, set(rep[created > after]))


# ── Replicate weights ──────────────────────────────────────────────


def _post_streams(seed: int) -> dict[str, np.random.SeedSequence]:
    """One rng stream per tracked post, keyed by its index in TRACKED_POSTS:
    adding a post leaves the others' replicates as they were."""
    kids = np.random.SeedSequence(seed).spawn(len(TRACKED_POSTS))
    return {p.slug: kids[i] for i, p in enumerate(TRACKED_POSTS)}


def design_weights(lab: pd.DataFrame, pop: pd.DataFrame) -> np.ndarray:
    """w_r = N_h / n_h per labelled draw, h = (post, draw-time bucket), N_h
    the population in that bucket now."""
    n_pop = pop.groupby(["tracked_slug", "bucket"]).size()
    key = pd.MultiIndex.from_arrays([lab["tracked_slug"], lab["bucket"]])
    N = n_pop.reindex(key).fillna(0).to_numpy(dtype=float)
    n = lab.groupby(["tracked_slug", "bucket"])["id"].transform("size").to_numpy(dtype=float)
    return N / n


def replicate_weights(
    lab: pd.DataFrame,
    pop: pd.DataFrame,
    n_boot: int,
    seed: int,
    fpc: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """(n_lab, 1 + n_boot) float32: w_r^(b) = w_r (1 - a_h + a_h c_r^(b)),
    a_h = sqrt(1 - n_h / N_h) (1 without fpc), c the draw's count in a
    with-replacement resample of its stratum (1 at b = 0). A stratum mean
    under these weights is _weighted's FPC-scaled resample; weights stay
    above 0, so no cell empties in a replicate. Each post's strata take
    their own stream (_post_streams); with `rng` given every stratum draws
    from it in _weighted's order instead (the equivalence test)."""
    W = np.empty((len(lab), 1 + n_boot), dtype=np.float32)
    n_pop = pop.groupby(["tracked_slug", "bucket"]).size()
    streams = None if rng is not None else _post_streams(seed)
    gens: dict[str, np.random.Generator] = {}
    for key, idx in lab.groupby(["tracked_slug", "bucket"]).indices.items():
        if rng is not None:
            g = rng
        else:
            if key[0] not in streams:
                raise ValueError(f"{key[0]} is not in TRACKED_POSTS")
            g = gens.setdefault(key[0], np.random.default_rng(streams[key[0]]))
        n, N = len(idx), float(n_pop.get(key, 0))
        pick = g.integers(0, n, (n_boot, n))
        c = np.bincount((pick + n * np.arange(n_boot)[:, None]).ravel(), minlength=n * n_boot)
        c = c.reshape(n_boot, n).T
        a = (np.sqrt(max(0.0, 1.0 - n / N)) if N else 0.0) if fpc else 1.0
        w = N / n
        W[idx, 0] = w
        W[idx, 1:] = w * (1.0 - a + a * c)
    return W


def _account_xi(accounts: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_accounts, n_boot) Poisson(1) multipliers, one per account in
    sorted order (stable for a given account set)."""
    rng = np.random.default_rng([seed, 1])
    return rng.poisson(1.0, (len(accounts), n_boot)).astype(np.float32)


def account_weights(lab: pd.DataFrame, pop: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    """(n_lab, 1 + n_boot) float32: design weights times a Poisson(1) draw per
    account, shared by all of its draws on every post; no stratum resample,
    no FPC (the account-cluster sensitivity)."""
    accounts = np.sort(lab["replier"].astype(str).unique())
    xi = _account_xi(accounts, n_boot, seed)
    row = np.searchsorted(accounts, lab["replier"].astype(str).to_numpy())
    w = design_weights(lab, pop).astype(np.float32)
    W = np.empty((len(lab), 1 + n_boot), dtype=np.float32)
    W[:, 0] = w
    W[:, 1:] = w[:, None] * xi[row]
    return W


# ── Domain means ───────────────────────────────────────────────────


@dataclass
class DomainSet:
    """Per (post, group): replicate vectors of each outcome's mean (mu), the
    group's share of the post (pi: census, or estimated for hajek), census
    totals and counts, and which domains collapsed, are NaN or suppressed."""

    mu: dict
    pi: dict
    A: dict
    A_post: dict
    accounts: dict
    replies: dict
    labelled: dict
    cells: pd.DataFrame
    collapsed: list
    nan_cells: list
    empty_cells: list
    suppressed: list
    n_rep: int

    def mean(self, post: str, g: str, k: str) -> np.ndarray:
        v = self.mu.get((post, g, k))
        return np.full(self.n_rep, np.nan) if v is None else v

    def share(self, post: str, g: str) -> np.ndarray:
        v = self.pi.get((post, g))
        return np.zeros(self.n_rep) if v is None else v

    def groups_of(self, post: str) -> list[str]:
        return [g for (p, g) in self.pi if p == post]

    def post_mean(self, post: str, k: str) -> np.ndarray:
        out = np.zeros(self.n_rep)
        for g in self.groups_of(post):
            pi = self.share(post, g)
            if np.any(pi > 0):
                out = out + pi * self.mean(post, g, k)
        return out

    def n_flags(self, posts) -> tuple[int, int]:
        """(collapsed, NaN) domains on `posts`: NaN for a cell with
        population and no draw; suppressed ones are n_suppressed's."""
        posts = set(posts)
        return (
            sum(p in posts for p, _ in self.collapsed),
            sum(p in posts for p, _ in {(c[0], c[2]) for c in self.nan_cells}),
        )

    def n_suppressed(self, posts) -> int:
        posts = set(posts)
        return sum(p in posts for p, _, _ in self.suppressed)

    def nan_names(self) -> list[str]:
        return ["/".join(map(str, c)) for c in self.nan_cells]

    def suppressed_names(self) -> list[str]:
        return [f"{p}/{g} ({n} accounts)" for p, g, n in self.suppressed]


def _log_flags(table: str, ds: DomainSet) -> None:
    """Name the NaN and suppressed domains of a table once (post slug,
    bucket, group and an account count: no reply or account)."""
    if ds.nan_cells:
        logger.warning(
            "reply crowd %s: NaN domains (a cell with population and no draw): %s",
            table,
            ", ".join(ds.nan_names()),
        )
    if ds.suppressed:
        logger.warning(
            "reply crowd %s: suppressed domains (under %d accounts): %s",
            table,
            settings.REPLY_CROWD_MIN_ACCOUNTS,
            ", ".join(ds.suppressed_names()),
        )


def _phis(y: np.ndarray, ks) -> dict[str, np.ndarray]:
    s = _stance_shares(np.asarray(y, dtype=float))
    s["mean_sq"] = np.asarray(y, dtype=float) ** 2
    return {k: s[k] for k in ks}


def _cell_sums(codes: np.ndarray, values: dict, W: np.ndarray, n_cells: int) -> dict:
    """Per cell and replicate, sum_r W[r, b] v_r for each value array (rows
    with code -1 are in no cell)."""
    from scipy import sparse

    rows = np.flatnonzero(codes >= 0)
    out = {}
    for name, v in values.items():
        M = sparse.csr_matrix(
            (np.asarray(v, dtype=float)[rows], (codes[rows], rows)), shape=(n_cells, len(codes))
        )
        out[name] = np.asarray(M @ W, dtype=float)
    return out


def _centred(S: dict, idx: np.ndarray, Sa: np.ndarray):
    """f(prod, u, v): the within-cell centred cross-product of u and v
    (S[prod] the cell sums of w a u v) pooled over the cells idx, per
    replicate."""
    ok = Sa > 0

    def f(prod: str, u: str, v: str) -> np.ndarray:
        return np.where(ok, S[prod][idx] - S[u][idx] * S[v][idx] / Sa, 0).sum(0)

    return f


def domain_means(
    fr: CrowdFrames,
    groups: pd.Series,
    W: np.ndarray,
    outcomes=OUTCOMES,
    per_account: bool = False,
    estimator: str = "poststrat",
    assist: bool = False,
    min_cell: int | None = None,
) -> DomainSet:
    """Domain means on the cells (post, bucket, group) of the replies that
    `groups` (id -> group) assigns. poststrat (the record): each domain's
    cell means weighted by the census cell totals; a domain with a cell of
    1 to min_cell-1 draws (settings.REPLY_CROWD_MIN_CELL_DRAWS by default)
    collapses to the Hajek ratio over its buckets (counted); a cell with
    population and no draw makes the domain NaN (named), never
    renormalised. hajek: the ratio over the domain's draws with estimated
    group shares (reply_loyalty's form). assist: GREG on the census `col`
    score f, slopes per post over its cells (within-cell, so the cell
    intercepts are the partition's), refit in every replicate: the mean on
    x = f (mean_sq on f^2); the pro / anti / neutral shares together on one
    auxiliary vector (pro(f), anti(f)) with a 2 x 2 slope matrix, so their
    g-weights do not depend on the outcome and the shares sum to 1.
    a_i = 1 (replies) or 1 / m (per_account)."""
    if estimator not in ("poststrat", "hajek"):
        raise ValueError(f"estimator {estimator!r}")
    if assist and estimator != "poststrat":
        raise ValueError("the assisted estimator is post-stratified")
    ks = tuple(outcomes) + (("mean_sq",) if "mean" in outcomes else ())
    min_cell = settings.REPLY_CROWD_MIN_CELL_DRAWS if min_cell is None else min_cell
    min_acc = settings.REPLY_CROWD_MIN_ACCOUNTS
    R = W.shape[1]

    pg = fr.pop["id"].map(groups)
    pop = fr.pop[pg.notna()].assign(g=pg[pg.notna()])
    lg = fr.lab["id"].map(groups)
    a_pop = 1.0 / pop["m"].to_numpy(float) if per_account else np.ones(len(pop))
    a_lab = 1.0 / fr.lab["m"].to_numpy(float) if per_account else np.ones(len(fr.lab))

    keys = ["tracked_slug", "bucket", "g"]
    lab_cells = fr.lab.assign(g=lg)[lg.notna()][keys]
    cells = pd.MultiIndex.from_frame(
        pd.concat([pop[keys], lab_cells]).drop_duplicates().sort_values(keys)
    )
    n_cells = len(cells)
    pop_code = cells.get_indexer(pd.MultiIndex.from_frame(pop[keys]))
    lab_code = np.full(len(fr.lab), -1)
    if len(lab_cells):
        lab_code[lab_cells.index.to_numpy()] = cells.get_indexer(
            pd.MultiIndex.from_frame(lab_cells)
        )

    A = np.bincount(pop_code, weights=a_pop, minlength=n_cells)
    n_c = np.bincount(lab_code[lab_code >= 0], minlength=n_cells)
    phy = _phis(fr.lab["y"].to_numpy(), ks)
    values = {"a": a_lab, **{f"y_{k}": a_lab * phy[k] for k in ks}}
    # GREG auxiliaries: aux[k] names the x columns outcome k is regressed on.
    X, aux = {}, {}
    if assist:
        fl, fp = fr.lab["f"].to_numpy(float), pop["f"].to_numpy(float)
        xl, xp = _stance_shares(fl), _stance_shares(fp)
        xl["mean_sq"], xp["mean_sq"] = fl**2, fp**2
        for k in ks:
            aux[k] = _SHARE_AUX if k in _SHARE_KS else (k,)
        for x in {x for v in aux.values() for x in v}:
            values[f"x_{x}"] = a_lab * xl[x]
            X[x] = np.bincount(pop_code, weights=a_pop * xp[x], minlength=n_cells)
        for k in ks:
            for i, x in enumerate(aux[k]):
                values[f"xy_{x}_{k}"] = a_lab * xl[x] * phy[k]
                for x2 in aux[k][i:]:
                    values.setdefault(f"xx_{x}_{x2}", a_lab * xl[x] * xl[x2])
    S = _cell_sums(lab_code, values, W, n_cells)

    cdf = cells.to_frame(index=False).assign(A=A, n=n_c)
    post_of = cdf["tracked_slug"].to_numpy()
    beta = {}  # (post, k) -> (R, len(aux[k])) slopes
    if assist:
        with np.errstate(invalid="ignore", divide="ignore"):
            for post in np.unique(post_of):
                idx = np.flatnonzero((post_of == post) & (n_c > 0))
                Sa = S["a"][idx]
                centred = _centred(S, idx, Sa)
                for k in ks:
                    xs = aux[k]
                    if len(xs) == 1:
                        x = xs[0]
                        num = centred(f"xy_{x}_{k}", f"x_{x}", f"y_{k}")
                        den = centred(f"xx_{x}_{x}", f"x_{x}", f"x_{x}")
                        beta[(post, k)] = np.where(den > 1e-12, num / den, 0.0)[:, None]
                        continue
                    C = np.empty((R, len(xs), len(xs)))
                    for i, x in enumerate(xs):
                        for j in range(i, len(xs)):
                            x2 = xs[j]
                            C[:, i, j] = C[:, j, i] = centred(f"xx_{x}_{x2}", f"x_{x}", f"x_{x2}")
                    c = np.stack([centred(f"xy_{x}_{k}", f"x_{x}", f"y_{k}") for x in xs], axis=1)
                    # Truncated pseudo-inverse: a direction with (about) no
                    # within-cell variance gets slope 0, the same for every
                    # outcome, so the shares still sum to 1.
                    val, vec = np.linalg.eigh(C)
                    tol = 1e-9 * np.maximum(Sa.sum(0), 1.0)
                    inv = np.where(val > tol[:, None], 1.0 / np.where(val > 0, val, 1.0), 0.0)
                    P = np.einsum("rij,rj,rkj->rik", vec, inv, vec)
                    beta[(post, k)] = np.einsum("rij,rj->ri", P, c)

    names = dict(enumerate(("critical", "mid", "supportive")))
    mu, pi, Ad, acc, reps, labd = {}, {}, {}, {}, {}, {}
    collapsed, nan_cells, empty, suppressed = [], [], [], []
    acc_n = pop.groupby(["tracked_slug", "g"])["replier"].nunique()
    rep_n = pop.groupby(["tracked_slug", "g"]).size()
    A_post = pd.Series(A).groupby(post_of).sum().to_dict()
    sa_post = {}
    dom = cdf.groupby(["tracked_slug", "g"]).indices
    for (post, g), idx in dom.items():
        Ac, nc = A[idx], n_c[idx]
        has_pop = Ac > 0
        Ad[(post, g)] = float(Ac.sum())
        acc[(post, g)] = int(acc_n.get((post, g), 0))
        reps[(post, g)] = int(rep_n.get((post, g), 0))
        labd[(post, g)] = int(nc.sum())
        hole = has_pop & (nc == 0)
        for i in idx[hole]:
            name = (post, names.get(int(cdf["bucket"][i]), cdf["bucket"][i]), g)
            (empty if estimator == "hajek" else nan_cells).append(name)
        Sa = S["a"][idx]
        sa_post[(post, g)] = Sa.sum(0)
        if estimator == "poststrat" and hole.any():
            continue
        if nc.sum() == 0:
            continue
        ratio = estimator == "hajek" or (nc[has_pop] < min_cell).any()
        if ratio and estimator == "poststrat":
            collapsed.append((post, g))
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in ks:
                Sy = S[f"y_{k}"][idx]
                if ratio:
                    m = Sy.sum(0) / Sa.sum(0)
                    if assist:
                        for i, x in enumerate(aux[k]):
                            xbar = S[f"x_{x}"][idx].sum(0) / Sa.sum(0)
                            m = m + beta[(post, k)][:, i] * (X[x][idx].sum() / Ac.sum() - xbar)
                else:
                    sel = has_pop
                    cm = Sy[sel] / Sa[sel]
                    if assist:
                        for i, x in enumerate(aux[k]):
                            xc = S[f"x_{x}"][idx][sel] / Sa[sel]
                            Xbar = X[x][idx][sel] / Ac[sel]
                            cm = cm + beta[(post, k)][None, :, i] * (Xbar[:, None] - xc)
                    m = ((Ac[sel] / Ac.sum())[:, None] * cm).sum(0)
                mu[(post, g, k)] = m
        if acc[(post, g)] < min_acc:
            suppressed.append((post, g, acc[(post, g)]))
            for k in ks:
                mu.pop((post, g, k), None)
    for post, g in dom:
        if estimator == "hajek":
            tot = sum(v for (p, _), v in sa_post.items() if p == post)
            with np.errstate(invalid="ignore", divide="ignore"):
                pi[(post, g)] = np.where(tot > 0, sa_post[(post, g)] / tot, 0.0)
        else:
            share = Ad[(post, g)] / A_post[post] if A_post[post] else 0.0
            pi[(post, g)] = np.full(R, share)
    return DomainSet(
        mu, pi, Ad, A_post, acc, reps, labd, cdf, collapsed, nan_cells, empty, suppressed, R
    )


# ── The decomposition (R2) ─────────────────────────────────────────


def _post_weights(ds: DomainSet, posts: list[str], how: str) -> np.ndarray:
    if how not in ("size", "equal"):
        raise ValueError(f"post weights {how!r}: size or equal")
    v = np.array([ds.A_post.get(p, 0.0) if how == "size" else 1.0 for p in posts], dtype=float)
    return v / v.sum()


def _sd(ds: DomainSet, post: str, k: str) -> np.ndarray:
    m = ds.post_mean(post, k)
    sq = ds.post_mean(post, "mean_sq") if k == "mean" else m
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum(sq - m**2, 0.0))


def _term(ds: DomainSet, post: str, g: str, anchor: str, k: str) -> np.ndarray:
    pi = ds.share(post, g)
    if not np.any(pi > 0):
        return np.zeros(ds.n_rep)
    return pi * (ds.mean(post, g, k) - ds.mean(post, anchor, k))


def decompose(
    ds: DomainSet,
    early: list[str],
    late: list[str],
    k: str = "mean",
    post_weights: str = "size",
) -> dict[str, np.ndarray]:
    """The R2 terms (replicate vectors) for outcome k, post-standardised
    (each period's posts weighted by size, shared by every term) with "both"
    as the anchor: change = same_accounts + turnover exactly. did: the gap
    of both-period accounts to the rest, late minus earlier; did_sd the same
    with each post's gap over its label SD."""
    both, eo, lo = R2_GROUPS
    wE, wL = _post_weights(ds, early, post_weights), _post_weights(ds, late, post_weights)

    def avg(w, posts, f):
        return sum(wi * f(p) for wi, p in zip(w, posts))

    with np.errstate(invalid="ignore", divide="ignore"):
        t = {
            "earlier": avg(wE, early, lambda p: ds.post_mean(p, k)),
            "late": avg(wL, late, lambda p: ds.post_mean(p, k)),
            "both_earlier": avg(wE, early, lambda p: ds.mean(p, both, k)),
            "both_late": avg(wL, late, lambda p: ds.mean(p, both, k)),
            "late_only_term": avg(wL, late, lambda p: _term(ds, p, lo, both, k)),
            "earlier_only_term": -avg(wE, early, lambda p: _term(ds, p, eo, both, k)),
        }
        t["change"] = t["late"] - t["earlier"]
        t["same_accounts"] = t["both_late"] - t["both_earlier"]
        t["turnover"] = t["late_only_term"] + t["earlier_only_term"]
        for side, posts in (("earlier", early), ("late", late)):
            Ab = np.array([ds.A.get((p, both), 0.0) for p in posts])
            raw = avg(Ab / Ab.sum(), posts, lambda p: ds.mean(p, both, k))
            t[f"post_mix_{side}"] = t[f"both_{side}"] - raw

        def gap(p, other):
            return ds.mean(p, both, k) - ds.mean(p, other, k)

        t["did"] = avg(wL, late, lambda p: gap(p, lo)) - avg(wE, early, lambda p: gap(p, eo))
        t["did_sd"] = avg(wL, late, lambda p: gap(p, lo) / _sd(ds, p, k)) - avg(
            wE, early, lambda p: gap(p, eo) / _sd(ds, p, k)
        )
    return t


def check_identity(t: dict[str, np.ndarray]) -> None:
    """change = same_accounts + turnover in every replicate (a run-time
    assert: the accounting is exact or something is wrong)."""
    if not np.allclose(t["change"], t["same_accounts"] + t["turnover"], equal_nan=True, atol=1e-9):
        raise AssertionError("reply crowd: change != same_accounts + turnover")


# ── R1: returners against one-off repliers ─────────────────────────


def _put(row: dict, key: str, v: np.ndarray) -> None:
    row[key], row[f"{key}_lo"], row[f"{key}_hi"] = _ci(np.asarray(v, dtype=float))


def _within(
    ds: DomainSet, posts: list[str], k: str, g1: str, g2: str, how: str = "size"
) -> np.ndarray:
    """The posts' own g1 - g2 gaps averaged by post size; NaN when any
    post's gap is NaN at replicate 0 (a NaN or suppressed domain: never
    renormalised over the other posts)."""
    gaps = {p: ds.mean(p, g1, k) - ds.mean(p, g2, k) for p in posts}
    if not posts or not all(np.isfinite(gaps[p][0]) for p in posts):
        return np.full(ds.n_rep, np.nan)
    w = _post_weights(ds, posts, how)
    return sum(wi * gaps[p] for wi, p in zip(w, posts))


def _pooled(ds: DomainSet, posts: list[str], g: str, k: str) -> np.ndarray:
    A = np.array([ds.A.get((p, g), 0.0) for p in posts])
    return sum(a / A.sum() * ds.mean(p, g, k) for a, p in zip(A, posts) if a > 0)


def loyalty_groups(fr: CrowdFrames, r1: pd.Series) -> pd.Series:
    """id -> "<returner|one_off>|<low|loyal>" (medium and high merged)."""
    tier = np.where(
        fr.pop["loyalty"].isin(("medium", "high")),
        "loyal",
        np.where(fr.pop["loyalty"] == "low", "low", None),
    )
    g = fr.pop["id"].map(r1).to_numpy(object)
    ok = pd.notna(g) & pd.notna(tier)
    return pd.Series(
        [f"{a}|{b}" for a, b in zip(g[ok], tier[ok])], index=fr.pop["id"].to_numpy()[ok]
    )


def _loyalty_within(
    fr: CrowdFrames, W: np.ndarray, r1: pd.Series, k: str, per_account: bool, how: str = "size"
) -> tuple[np.ndarray, dict]:
    """R1's within-posts gap standardised to each post's census tier mix:
    sum_P (A_P / A) sum_T kappa_PT (mu(returner, T) - mu(one_off, T)),
    Hajek domains. Returns it (NaN when any post's gap is NaN: never
    renormalised) and its flags: empty_cells (population, no draw: the
    Hajek ratio leaves them out), suppressed_domains, nan_posts."""
    ds = domain_means(fr, loyalty_groups(fr, r1), W, (k,), per_account, "hajek")
    gaps, sizes, nan_posts = [], [], []
    for p in fr.posts:
        tot = sum(ds.A.get((p, f"{g}|{t}"), 0.0) for g in R1_GROUPS for t in LOYALTY_LEVELS)
        if tot <= 0:
            continue
        gp = np.zeros(ds.n_rep)
        for t in LOYALTY_LEVELS:
            kappa = sum(ds.A.get((p, f"{g}|{t}"), 0.0) for g in R1_GROUPS) / tot
            if kappa > 0:
                gp = gp + kappa * (ds.mean(p, f"returner|{t}", k) - ds.mean(p, f"one_off|{t}", k))
        if not np.isfinite(gp[0]):
            nan_posts.append(p)
        gaps.append(gp)
        sizes.append(tot if how == "size" else 1.0)
    flags = {
        "empty_cells": len(ds.empty_cells),
        "suppressed_domains": len(ds.suppressed),
        "nan_posts": len(nan_posts),
    }
    if nan_posts:
        logger.warning("reply crowd: loyalty-standardised gap NaN on %s", ", ".join(nan_posts))
        _log_flags("loyalty-standardised R1", ds)
    if not gaps or nan_posts:
        return np.full(ds.n_rep, np.nan), flags
    w = np.array(sizes) / sum(sizes)
    return sum(wi * g for wi, g in zip(w, gaps)), flags


def reply_returners(
    fr: CrowdFrames,
    W: np.ndarray,
    contrast: Contrast,
    W_acct: np.ndarray | None = None,
    assist: bool = True,
    per_account: bool = False,
) -> pd.DataFrame:
    """R1 per post (in time order), within posts, within earlier / late,
    within posts loyalty-standardised (direct mean and pro) and the ALL
    pooled row (descriptive: the post mix differs by group)."""
    r1 = r1_groups(fr.part)
    ests = {"direct": domain_means(fr, r1, W, OUTCOMES, per_account)}
    if assist:
        ests["assisted"] = domain_means(fr, r1, W, OUTCOMES, per_account, assist=True)
    acct = domain_means(fr, r1, W_acct, OUTCOMES, per_account) if W_acct is not None else None
    three = domain_means(fr, r1_groups(fr.part, three=True), W, ("mean",), per_account)
    ds = ests["direct"]
    _log_flags("R1", ds)
    period = {**{p: "earlier" for p in contrast.early}, **{p: "late" for p in contrast.late}}
    rows = []

    def flags(posts) -> dict:
        coll, nan = ds.n_flags(posts)
        return {
            "collapsed": coll,
            "nan_domains": nan,
            "suppressed_domains": ds.n_suppressed(posts),
        }

    for p in fr.posts:
        row = {
            "post": p,
            "period": period.get(p, "other"),
            "n_replies": sum(ds.replies.get((p, g), 0) for g in R1_GROUPS),
            "share_returner": float(ds.share(p, "returner")[0]),
            "accounts_returner": ds.accounts.get((p, "returner"), 0),
            "accounts_one_off": ds.accounts.get((p, "one_off"), 0),
            "labelled_returner": ds.labelled.get((p, "returner"), 0),
            "labelled_one_off": ds.labelled.get((p, "one_off"), 0),
            **flags([p]),
        }
        for est, d in ests.items():
            for k in OUTCOMES:
                for g in R1_GROUPS:
                    _put(row, f"{est}_{k}_{g}", d.mean(p, g, k))
                _put(row, f"{est}_{k}_gap", d.mean(p, "returner", k) - d.mean(p, "one_off", k))
        if acct is not None:
            for k in OUTCOMES:
                _, row[f"direct_{k}_gap_acct_lo"], row[f"direct_{k}_gap_acct_hi"] = _ci(
                    acct.mean(p, "returner", k) - acct.mean(p, "one_off", k)
                )
        for g in ("ret_1", "ret_2plus"):
            _put(row, f"direct_mean_{g}", three.mean(p, g, "mean"))
        rows.append(row)
    for name, posts in (
        ("within posts", fr.posts),
        ("within earlier", contrast.early),
        ("within late", contrast.late),
    ):
        posts = [p for p in posts if p in fr.posts]
        row = {
            "post": name,
            "period": "",
            "n_replies": int(sum(r["n_replies"] for r in rows if r["post"] in posts)),
            **flags(posts),
        }
        for est, d in ests.items():
            for k in OUTCOMES:
                _put(row, f"{est}_{k}_gap", _within(d, posts, k, *R1_GROUPS))
        if acct is not None:
            for k in OUTCOMES:
                _, row[f"direct_{k}_gap_acct_lo"], row[f"direct_{k}_gap_acct_hi"] = _ci(
                    _within(acct, posts, k, *R1_GROUPS)
                )
        rows.append(row)
    row = {"post": "within posts, loyalty-standardised", "period": ""}
    for k in ("mean", "pro"):
        v, lf = _loyalty_within(fr, W, r1, k, per_account)
        _put(row, f"direct_{k}_gap", v)
        row["nan_domains"] = lf["nan_posts"]
        row["suppressed_domains"] = lf["suppressed_domains"]
        row["empty_cells"] = lf["empty_cells"]
    rows.append(row)
    row = {
        "post": "ALL pooled (post mix differs)",
        "period": "",
        "n_replies": int(sum(r["n_replies"] for r in rows[: len(fr.posts)])),
        **flags(fr.posts),
    }
    for est, d in ests.items():
        for k in OUTCOMES:
            m = {g: _pooled(d, fr.posts, g, k) for g in R1_GROUPS}
            for g in R1_GROUPS:
                _put(row, f"{est}_{k}_{g}", m[g])
            _put(row, f"{est}_{k}_gap", m["returner"] - m["one_off"])
    rows.append(row)
    return pd.DataFrame(rows)


# ── R2: the change table ───────────────────────────────────────────


def _group_counts(fr: CrowdFrames, groups: pd.Series, c: Contrast) -> dict:
    pop = fr.pop.assign(g=fr.pop["id"].map(groups)).dropna(subset=["g"])
    lab = fr.lab.assign(g=fr.lab["id"].map(groups)).dropna(subset=["g"])
    out = {}
    for g in R2_GROUPS:
        out[f"accounts_{g}"] = int(pop.loc[pop["g"] == g, "replier"].nunique())
        out[f"replies_{g}"] = int((pop["g"] == g).sum())
        out[f"labelled_{g}"] = int((lab["g"] == g).sum())
    e = pop[pop["tracked_slug"].isin(c.early)]
    late = pop[pop["tracked_slug"].isin(c.late)]
    out["share_both_of_earlier_replies"] = float((e["g"] == "both").mean())
    out["share_both_of_late_replies"] = float((late["g"] == "both").mean())
    # Each period's own accounts: a "both" account (both sides by any reply,
    # textless ones too) may have replies with text on one side only.
    ea, la = e["replier"].nunique(), late["replier"].nunique()
    eb = e.loc[e["g"] == "both", "replier"].nunique()
    lb = late.loc[late["g"] == "both", "replier"].nunique()
    out["accounts_both_earlier"], out["accounts_both_late"] = int(eb), int(lb)
    out["share_both_of_earlier_accounts"] = eb / ea if ea else float("nan")
    out["share_both_of_late_accounts"] = lb / la if la else float("nan")
    return out


def reply_crowd_change(
    fr: CrowdFrames,
    W: np.ndarray,
    contrasts: list[Contrast],
    W_acct: np.ndarray | None = None,
    assist: bool = True,
    per_account: bool = False,
    post_weights: str = "size",
) -> pd.DataFrame:
    """R2, long: one row per (contrast, estimator, outcome) with every term
    and its interval; headline terms carry the account-cluster interval."""
    rows = []
    for c in contrasts:
        groups = crowd_groups(fr.part, c.early, c.late)
        counts = _group_counts(fr, groups, c)
        ests = {"direct": domain_means(fr, groups, W, OUTCOMES, per_account)}
        if assist:
            ests["assisted"] = domain_means(fr, groups, W, OUTCOMES, per_account, assist=True)
        acct = (
            domain_means(fr, groups, W_acct, OUTCOMES, per_account) if W_acct is not None else None
        )
        for est, ds in ests.items():
            if est == "direct":
                _log_flags(f"R2 [{c.name}]", ds)
            coll, nan = ds.n_flags(c.early + c.late)
            for k in OUTCOMES:
                t = decompose(ds, c.early, c.late, k, post_weights)
                check_identity(t)
                row = {
                    "contrast": c.name,
                    "early_posts": ",".join(c.early),
                    "late_posts": ",".join(c.late),
                    "estimator": est,
                    "weight": "account" if per_account else "reply",
                    "outcome": k,
                    **counts,
                    "collapsed": coll,
                    "nan_domains": nan,
                    "suppressed_domains": ds.n_suppressed(c.early + c.late),
                }
                for term in TERMS:
                    _put(row, term, t[term])
                if acct is not None and est == "direct":
                    ta = decompose(acct, c.early, c.late, k, post_weights)
                    for term in HEADLINE_TERMS:
                        _, row[f"{term}_acct_lo"], row[f"{term}_acct_hi"] = _ci(ta[term])
                rows.append(row)
    return pd.DataFrame(rows)


# ── Pairs and the reference distribution ───────────────────────────


def reference_rms(ts: list[np.ndarray], alpha: float = 0.05) -> dict:
    """Over the pairs of a kind: rms_raw = sqrt(mean t^2), mean_var = mean
    bootstrap variance, rms_corrected = sqrt(max(0, mean(t^2 - v))) (the
    sampling noise taken out), and its percentile interval from
    mean_j((t_j^(b))^2 - 2 v_j) over the shared replicates."""
    T = np.vstack(ts) if ts else np.empty((0, 2))
    T = T[np.isfinite(T[:, 0])]
    if not len(T):
        return {"n_pairs": 0}
    v = np.nanvar(T[:, 1:], axis=1, ddof=1)
    S = float(np.mean(T[:, 0] ** 2 - v))
    Sb = np.nanmean(T[:, 1:] ** 2 - 2 * v[:, None], axis=0)
    lo, hi = np.nanquantile(Sb, [alpha / 2, 1 - alpha / 2])
    return {
        "n_pairs": len(T),
        "rms_raw": float(np.sqrt(np.mean(T[:, 0] ** 2))),
        "mean_var": float(np.mean(v)),
        "rms_corrected": float(np.sqrt(max(0.0, S))),
        "rms_lo": float(np.sqrt(max(0.0, lo))),
        "rms_hi": float(np.sqrt(max(0.0, hi))),
    }


def _kind(a: str, b: str) -> str | None:
    return None if "other" in (a, b) else f"{a}-{b}"


def reply_crowd_pairs(
    fr: CrowdFrames, W: np.ndarray, contrast: Contrast, outcomes=OUTCOMES
) -> pd.DataFrame:
    """Every ordered pair q before p of the contrast's posts: pair_change =
    same_accounts (the accounts on both, the same set on both sides) +
    composition; then per kind and term the noise-corrected RMS over its
    pairs (summary rows). Descriptive."""
    period = {**{p: "earlier" for p in contrast.early}, **{p: "late" for p in contrast.late}}
    posts = [p for p in fr.posts if p in period]
    rows, store = [], {}
    for i, q in enumerate(posts):
        for p in posts[i + 1 :]:
            kind = _kind(period[q], period[p])
            ds = domain_means(fr, pair_groups(fr.part, q, p), W, outcomes)
            _log_flags(f"pair {q} -> {p}", ds)
            have = ds.cells[ds.cells["A"] > 0]
            coll, nan = ds.n_flags([q, p])
            for k in outcomes:
                t = {
                    "pair_change": ds.post_mean(p, k) - ds.post_mean(q, k),
                    "same_accounts": ds.mean(p, "both_qp", k) - ds.mean(q, "both_qp", k),
                }
                t["composition"] = t["pair_change"] - t["same_accounts"]
                row = {
                    "kind": kind,
                    "q": q,
                    "p": p,
                    "outcome": k,
                    "accounts_both": ds.accounts.get((p, "both_qp"), 0),
                    "labelled_q_both": ds.labelled.get((q, "both_qp"), 0),
                    "labelled_p_both": ds.labelled.get((p, "both_qp"), 0),
                    "min_cell": int(have["n"].min()) if len(have) else 0,
                    "collapsed": coll,
                    "nan_domains": nan,
                    "suppressed_domains": ds.n_suppressed([q, p]),
                }
                for term in PAIR_TERMS:
                    _put(row, term, t[term])
                    store.setdefault((kind, term, k), []).append(t[term])
                rows.append(row)
    for (kind, term, k), ts in store.items():
        if term == "pair_change":
            continue
        row = {"kind": f"summary:{kind}", "term": term, "outcome": k, **reference_rms(ts)}
        if row.get("n_pairs") == 1:
            row["note"] = "one pair: its own row carries the value and CI"
            row["rms_corrected"] = row["rms_lo"] = row["rms_hi"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── The panel (exploratory) ────────────────────────────────────────


def _wcorr(x: np.ndarray, y: np.ndarray, V: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        s = V.sum(0)
        mx, my = x @ V / s, y @ V / s
        cxy = (x * y) @ V / s - mx * my
        vx, vy = (x * x) @ V / s - mx**2, (y * y) @ V / s - my**2
        return cxy / np.sqrt(vx * vy)


def _panel_units(fr: CrowdFrames, contrast: Contrast) -> pd.DataFrame:
    lab = fr.lab
    d = pd.DataFrame(
        {
            "id": lab["id"],
            "post": lab["tracked_slug"],
            "acct": lab["replier"],
            "y": lab["y"],
            "f": lab["f"],
            "w": design_weights(lab, fr.pop),
            "dup": lab["dup_text"],
        }
    )
    sw = d.groupby("post")["w"].transform("sum")
    d["e"] = d["y"] - (d["w"] * d["y"]).groupby(d["post"]).transform("sum") / sw
    d["ef"] = d["f"] - d["post"].map(fr.pop.groupby("tracked_slug")["f"].mean())
    d["m"] = d.groupby(["acct", "post"])["id"].transform("size")
    multi = d[d.groupby("acct")["id"].transform("size") >= 2]
    u = multi.merge(multi, on="acct", suffixes=("1", "2"))
    u = u[u["id1"] != u["id2"]]
    order = {p: i for i, p in enumerate(fr.posts)}
    period = {**{p: "earlier" for p in contrast.early}, **{p: "late" for p in contrast.late}}
    same = u["post1"] == u["post2"]
    u = u[same | (u["post1"].map(order) < u["post2"].map(order))].copy()
    same = u["post1"] == u["post2"]
    u["v"] = np.where(
        same,
        u["w1"] * u["w2"] / (u["m1"] * (u["m1"] - 1)).clip(lower=1),
        u["w1"] * u["w2"] / (u["m1"] * u["m2"]),
    )
    u["kind"] = [
        "same post" if s else _kind(period.get(a, "other"), period.get(b, "other"))
        for s, a, b in zip(same, u["post1"], u["post2"])
    ]
    return u[u["kind"].notna()]


def reply_crowd_panel(fr: CrowdFrames, n_boot: int, seed: int, contrast: Contrast) -> pd.DataFrame:
    """Exploratory: for an account with labelled draws on two posts (or two
    on one post), the weighted Pearson r of its post-centred labels, per
    kind, with an account-cluster bootstrap. Takes no column (model_r, on
    the census score of the same units, is a diagnostic only). Never a
    transition table, a kept-side share or a switching rate."""
    u = _panel_units(fr, contrast)
    rows = []
    min_acc = settings.REPLY_CROWD_PANEL_MIN_ACCOUNTS
    for kind in ("same post", "earlier-earlier", "earlier-late", "late-late"):
        k = u[u["kind"] == kind]
        accounts = np.sort(k["acct"].astype(str).unique())
        n_units = len(k) // 2 if kind == "same post" else len(k)
        row = {"kind": kind, "accounts": len(accounts), "units": n_units}
        tot = k.groupby("acct")["v"].sum()
        row["kish_n"] = float(tot.sum() ** 2 / (tot**2).sum()) if len(tot) else 0.0
        note = [
            (
                "exploratory: post-centred, regression to the mean and label noise; "
                "not a switching rate"
            )
        ]
        if len(accounts) < min_acc:
            note.append(f"suppressed: under {min_acc} accounts")
            for c in ("r_opus", "r_opus_lo", "r_opus_hi", "r_opus_no_dup_text", "model_r"):
                row[c] = float("nan")
        else:
            xi = _account_xi(accounts, n_boot, seed + 2)
            ix = np.searchsorted(accounts, k["acct"].astype(str).to_numpy())
            V = k["v"].to_numpy()[:, None] * np.hstack([np.ones((len(k), 1)), xi[ix]])
            row["r_opus"], row["r_opus_lo"], row["r_opus_hi"] = _ci(
                _wcorr(k["e1"].to_numpy(), k["e2"].to_numpy(), V)
            )
            nd = k[~(k["dup1"] | k["dup2"])]
            row["r_opus_no_dup_text"] = float(
                _wcorr(nd["e1"].to_numpy(), nd["e2"].to_numpy(), nd["v"].to_numpy()[:, None])[0]
            )
            row["model_r"] = float(
                _wcorr(k["ef1"].to_numpy(), k["ef2"].to_numpy(), k["v"].to_numpy()[:, None])[0]
            )
            if len(accounts) < 2 * min_acc:
                note.append("few accounts: a wide interval")
        note.append("model_r is a diagnostic, not for publication")
        row["note"] = "; ".join(note)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Census facts ───────────────────────────────────────────────────


def _quantile_dates(s: pd.Series) -> list:
    s = s.dropna()
    if not len(s):
        return [None] * 5
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95], interpolation="nearest")
    return [pd.Timestamp(x).date().isoformat() for x in q]


def reply_crowd_census(fr: CrowdFrames, contrasts: list[Contrast]) -> pd.DataFrame:
    """Per post in time order, exact counts: repliers and replies, the share
    of its replies (and accounts) from accounts seen on an earlier tracked
    post, from returners, group account counts for each contrast, and by
    group (R1 and the record's R2 groups) the account-created quantiles,
    tier mix and accounts created after settings.REPLY_CROWD_NEW_ACCOUNT_DATE.
    A group under REPLY_CROWD_MIN_ACCOUNTS accounts is suppressed."""
    min_acc = settings.REPLY_CROWD_MIN_ACCOUNTS
    new_after = pd.Timestamp(settings.REPLY_CROWD_NEW_ACCOUNT_DATE)
    part = fr.part[fr.part["replier"].notna()]
    r1 = r1_groups(fr.part)
    r2 = crowd_groups(fr.part, contrasts[0].early, contrasts[0].late)
    per_contrast = [(c, crowd_groups(fr.part, c.early, c.late)) for c in contrasts]
    rows, seen = [], set()
    for p in fr.posts:
        pp = fr.pop[fr.pop["tracked_slug"] == p]
        acc_p = set(part.loc[part["tracked_slug"] == p, "replier"])
        prior = pp["replier"].isin(seen)
        row = {
            "post": p,
            "event_date": _tracked()[p][0].isoformat(),
            "repliers": len(acc_p),
            "replies": int((fr.part["tracked_slug"] == p).sum()),
            "replies_with_text": len(pp),
            "prior_replier_share": float(prior.mean()) if len(pp) else float("nan"),
            "prior_replier_share_accounts": len(acc_p & seen) / len(acc_p)
            if acc_p
            else float("nan"),
            "other_post_share": float((pp["id"].map(r1) == "returner").mean()),
        }
        seen |= acc_p
        for c, g in per_contrast:
            if p in c.early or p in c.late:
                gp = pp.assign(g=pp["id"].map(g))
                for name in R2_GROUPS:
                    row[f"accounts_{name} [{c.name}]"] = int(
                        gp.loc[gp["g"] == name, "replier"].nunique()
                    )
        for name, gmap in (("returner", r1), ("one_off", r1), *((x, r2) for x in R2_GROUPS)):
            d = pp[pp["id"].map(gmap) == name]
            accts = d.drop_duplicates("replier")
            n = len(accts)
            row[f"accounts_{name}"] = n
            ok = n >= min_acc
            for qn, v in zip(
                ("q05", "q25", "q50", "q75", "q95"), _quantile_dates(accts["created"])
            ):
                row[f"created_{qn}_{name}"] = v if ok else None
            for t in ("low", "medium", "high"):
                row[f"tier_{t}_{name}"] = float((d["loyalty"] == t).mean()) if ok else np.nan
            row[f"created_after_{new_after.date().isoformat()}_{name}"] = (
                int((accts["created"] > new_after).sum()) if ok else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ── Robustness ─────────────────────────────────────────────────────

_R1_HEAD = ("mean", "pro", "neutral")
_R2_HEAD = ("neutral", "pro", "mean")


def _headline(
    fr,
    W,
    contrast,
    *,
    estimator="poststrat",
    assist=False,
    per_account=False,
    post_weights="size",
    tables=("R1", "R2"),
    min_cell=None,
) -> list[dict]:
    out = []
    if "R1" in tables:
        ds = domain_means(
            fr, r1_groups(fr.part), W, _R1_HEAD, per_account, estimator, assist, min_cell
        )
        for k in _R1_HEAD:
            v = _within(ds, fr.posts, k, *R1_GROUPS, how=post_weights)
            out.append(
                dict(
                    zip(
                        ("table", "term", "outcome", "value", "lo", "hi"),
                        ("R1", "within_posts_gap", k, *_ci(v)),
                    )
                )
            )
    if "R2" in tables:
        ds = domain_means(
            fr,
            crowd_groups(fr.part, contrast.early, contrast.late),
            W,
            _R2_HEAD,
            per_account,
            estimator,
            assist,
            min_cell,
        )
        for k in _R2_HEAD:
            t = decompose(ds, contrast.early, contrast.late, k, post_weights)
            check_identity(t)
            for term in HEADLINE_TERMS:
                out.append(
                    dict(
                        zip(
                            ("table", "term", "outcome", "value", "lo", "hi"),
                            ("R2", term, k, *_ci(t[term])),
                        )
                    )
                )
    return out


def _side_balance(fr, W, c: Contrast) -> list[dict]:
    """Side balance pro / (pro + anti) by group and period, levels only."""
    ds = domain_means(fr, crowd_groups(fr.part, c.early, c.late), W, ("pro", "anti"))
    out = []
    for period, posts, groups in (
        ("earlier", c.early, ("both", "earlier_only")),
        ("late", c.late, ("both", "late_only")),
    ):
        w = _post_weights(ds, posts, "size")
        for g in groups:
            pro = sum(wi * ds.mean(p, g, "pro") for wi, p in zip(w, posts))
            anti = sum(wi * ds.mean(p, g, "anti") for wi, p in zip(w, posts))
            with np.errstate(invalid="ignore", divide="ignore"):
                v = _ci(pro / (pro + anti))
            out.append(
                {
                    "table": "levels",
                    "term": f"side_balance_{g}_{period}",
                    "outcome": "side_balance",
                    "value": v[0],
                    "lo": v[1],
                    "hi": v[2],
                    "note": "praise rule: praise for a deal counts against the war",
                }
            )
    return out


def reply_crowd_robustness(
    fr: CrowdFrames,
    W: np.ndarray,
    W_acct: np.ndarray,
    contrasts: list[Contrast],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """The headline terms (R1 within-posts gap; R2 change, same_accounts,
    turnover, did on the record's contrast) under each variant of
    docs/decisions.md 2026-11-23. Every variant but per_account is reply-
    weighted, whatever weight the record tables of the same run take."""
    c = contrasts[0]
    rows: list[dict] = []

    def add(variant, items):
        rows.extend({"variant": variant, **r} for r in items)

    def rebuilt(f):
        return replicate_weights(f.lab, f.pop, n_boot, seed)

    add("direct", _headline(fr, W, c))
    add("no_collapse", _headline(fr, W, c, min_cell=1))
    add("hajek", _headline(fr, W, c, estimator="hajek"))
    add(f"assisted:{fr.col}", _headline(fr, W, c, assist=True))
    add("per_account", _headline(fr, W, c, per_account=True))
    add("cluster_account", _headline(fr, W_acct, c))
    add("no_fpc", _headline(fr, replicate_weights(fr.lab, fr.pop, n_boot, seed, fpc=False), c))
    add("status_on_text", _headline(with_participation(fr, "text"), W, c))
    for d in ("first", "fresh"):
        f = select_draws(fr, d)
        add(f"draws_{d}", _headline(f, rebuilt(f), c))
    first = pd.Timestamp(_tracked()[c.early[0]][0], tz="UTC")
    for name, f in (
        ("exclude_new_accounts", exclude_new_accounts(fr, first)),
        ("exclude_heavy", exclude_heavy(fr)),
        ("exclude_dup_text", exclude_dup_text(fr)),
    ):
        add(name, _headline(f, rebuilt(f), c))
    add("equal_post_weights", _headline(fr, W, c, post_weights="equal"))
    for p in c.early:
        rest = [q for q in c.early if q != p]
        if rest:
            add(f"leave_out_{p}", _headline(fr, W, Contrast(c.name, rest, c.late), tables=("R2",)))
    for other in contrasts[1:]:
        name = {"near": "base_near", "benchmark": "benchmark"}.get(other.name, other.name)
        add(name, _headline(fr, W, other, tables=("R2",)))
    r1 = r1_groups(fr.part)
    add(
        "loyalty_standardised",
        [
            dict(
                zip(
                    ("table", "term", "outcome", "value", "lo", "hi"),
                    ("R1", "within_posts_gap", k, *_ci(_loyalty_within(fr, W, r1, k, False)[0])),
                )
            )
            for k in ("mean", "pro")
        ],
    )
    add("side_balance", _side_balance(fr, W, c))
    return pd.DataFrame(rows)


# ── Entry point ────────────────────────────────────────────────────


def _close_to_population(fr: CrowdFrames, W: np.ndarray) -> list[str]:
    """Posts whose post-stratified mean sits further from the bucket-
    stratified direct mean (reply_population's opus_mean) than that
    estimate's half-width: logged, not fatal (a sampling difference)."""
    g = r1_groups(fr.part)
    ps = domain_means(fr, g, W, ("mean",))
    hj = domain_means(fr, g, W, ("mean",), estimator="hajek")
    off = []
    for p in fr.posts:
        a, b = ps.post_mean(p, "mean"), hj.post_mean(p, "mean")
        v, lo, hi = _ci(b)
        if np.isfinite(a[0]) and abs(a[0] - v) > (hi - lo) / 2:
            off.append(p)
    if off:
        logger.warning("reply crowd: post-stratified mean outside the direct half-width: %s", off)
    return off


def reply_crowd(
    col: str | None = None,
    model: str | None = None,
    n_boot: int | None = None,
    seed: int | None = None,
    late_phase: str | None = None,
    per_account: bool = False,
    robustness: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every reply-crowd table and a summary for the console. Reads the
    frames once; no API, no writes."""
    col = col or settings.REPLY_POPULATION_COL
    n_boot = settings.BOOTSTRAP_N if n_boot is None else n_boot
    seed = settings.BOOTSTRAP_SEED if seed is None else seed
    fr = crowd_frames(col, model)
    contrasts = crowd_contrasts(fr.posts, late_phase)
    c = contrasts[0]
    W = replicate_weights(fr.lab, fr.pop, n_boot, seed)
    W_acct = account_weights(fr.lab, fr.pop, n_boot, seed)
    off = _close_to_population(fr, W)
    tables = {
        "returners": reply_returners(fr, W, c, W_acct, per_account=per_account),
        "change": reply_crowd_change(fr, W, contrasts, W_acct, per_account=per_account),
        "pairs": reply_crowd_pairs(fr, W, c),
        "panel": reply_crowd_panel(fr, n_boot, seed, c),
        "census": reply_crowd_census(fr, contrasts),
    }
    if robustness:
        tables["robustness"] = reply_crowd_robustness(fr, W, W_acct, contrasts, n_boot, seed)
    phase = _phase_of(fr.posts)
    summary = {
        "labels": "v2",
        "col": col,
        "n_boot": n_boot,
        "weight": "account (1 / m)" if per_account else "reply",
        "n_replies": len(fr.pop),
        "n_labelled": len(fr.lab),
        "n_first": int((fr.lab["draw_set"] == "stance").sum()),
        "n_fresh": int((fr.lab["draw_set"] != "stance").sum()),
        "repliers": int(fr.pop["replier"].nunique()),
        "replies_all": len(fr.part),
        "no_account": fr.n_unknown,
        "early": [(p, phase[p]) for p in c.early],
        "late": [(p, phase[p]) for p in c.late],
        "contrasts": [x.name for x in contrasts],
        "off_population": off,
    }
    return tables, summary


def output_names(
    col: str, per_account: bool = False, late_phase: str | None = None
) -> dict[str, str]:
    """The CSV file name of each table, under settings.PROCESSED_DIR. A
    late phase other than settings.REPLY_CROWD_LATE_PHASE suffixes every
    file (`_late-<phase slug>`), so a variant never writes over the record;
    per_account suffixes the tables it changes (returners, change). The
    panel (model_r) and the census (the population with a `col` score) name
    the column."""
    late = ""
    if late_phase is not None and late_phase != settings.REPLY_CROWD_LATE_PHASE:
        late = "_late-" + re.sub(r"[^a-z0-9]+", "-", late_phase.lower()).strip("-")
    pa = "_per_account" if per_account else ""
    return {
        "returners": f"reply_crowd_returners_{col}_v2{pa}{late}.csv",
        "change": f"reply_crowd_change_{col}_v2{pa}{late}.csv",
        "pairs": f"reply_crowd_pairs_{col}_v2{late}.csv",
        "panel": f"reply_crowd_panel_{col}_v2{late}.csv",
        "census": f"reply_crowd_census_{col}_v2{late}.csv",
        "robustness": f"reply_crowd_robustness_{col}_v2{late}.csv",
    }
