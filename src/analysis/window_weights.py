"""
Optional window weights for the X broadcaster series: `phases --weights
none|design|volume` (default settings.PHASE_WEIGHTS, "none"). Raw, every post
weighing 1, stays the definition of record (the owner's call, 2026-11-22;
docs/decisions.md). Nothing here calls an API or writes a file.

Why. A sampled account (settings.ACCOUNT_WINDOW_CAPS) holds the newest 5 posts
of each 35 h window on most of its grid (collect since 2026-10-31, and the
x-archive fill) but every post wherever an earlier pull read it whole (parts
of Feb-Apr, the newest days of each 14-day slice before 2026-10-31). Raw, a
dense stretch counts many times over a sparse one, and a dense account over
the rest of its tier (docs/STATUS.md, *Uneven density*).

The grid is the x-archive fill's: x_archive_fill.collect_rows over the collect
manifests in settings.X_SAMPLING_DIR, then account_grid's backward windows to
COLLECTION_START, each window [s, e) (windows share their edge second; a post
at it is the later window's). For a window w of a sampled account:

  h_w    the account's rows of the frame in [s, e): every post held, with
         text or not, since X's count counts them all
  N_w    X's count over [s, e) from the `x-counts` hour buckets
         (x_counts.CountsCache): a bucket wholly inside counts whole; an
         edge bucket (cut by s or e) counts the posts the frame holds in its
         part inside the window, plus the bucket's posts the frame holds
         nowhere in its hour (X's count less the held, at least 0) times the
         share of the hour inside the window, so a held post at an edge
         counts only in the window that holds it. Never under h_w: X counts
         the posts there now, and one deleted since a pull read it is held
         but not counted (the 2026-11-02 probe found that counts/all takes
         in retweets and matches the cache)
  cap_w  the window's cap (the collect manifest's, else x_collector._window_cap)

and a post in w weighs

  none    1 (the record)
  volume  N_w / h_w (1 where h_w >= N_w): an account's posts stand for its
          real output, so each account counts by how much it posted
  design  min(cap_w, N_w) / h_w: each window weighs as the posts the current
          design keeps there (its cap, or all it had when fewer), so a stretch
          an earlier pull read whole counts like a 5-a-window one

Weight 1 under every scheme: take-all accounts (read whole: what they hold is
what X counts, so their windows need no weight), a sampled account's posts
outside every grid window (counted), keyword-search rows and Trump's Truth
Social feed (no grid). A window that holds nothing carries no weight, so
under volume what X counts there is lost, not spread (counted too).

Missing counts. N_w needs the hour counts over the whole window. A window past
their coverage (from 2026-11-02 until `x-counts` runs for it) still gets an
exact weight where it needs no count: a collect window the run did not
truncate read every post the timeline served (N_w = h_w), and under design a
window holding at least its cap weighs cap_w / h_w whatever X counts. Any
other post there gets no weight (NaN), and `require` (called by the weighted
bootstrap and weekly series) refuses an analysis that uses one, naming the
accounts: `x-counts` for the range is a paid call (about $0.01 per account per
31 days), so the estimate goes to the owner first. Not a fallback to design
weights: in a truncated window that would weigh the posts cap_w / h_w = 1,
the very undercount volume weights exist to undo, inside a cell weighted by
volume everywhere else.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import settings
from src.analysis.inference import WEIGHT_COL
from src.collectors import x_archive_fill as xf
from src.collectors import x_counts as xcn

logger = logging.getLogger(__name__)

SCHEMES = ("none", "design", "volume")
HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)


class WeightsUnavailable(RuntimeError):
    """Posts an analysis uses have no window weight (their window's count is
    not cached and the weight depends on it)."""


def _check_scheme(scheme: str) -> None:
    if scheme not in SCHEMES:
        raise ValueError(f"no weighting {scheme!r}: one of {', '.join(SCHEMES)}")


def _held_in(held: np.ndarray, a: datetime, b: datetime) -> int:
    """Posts of `held` (sorted datetime64[us]) in [a, b)."""
    lo, hi = np.searchsorted(held, np.array([a, b], dtype="datetime64[us]"), side="left")
    return int(hi - lo)


def apportioned_count(
    counts: xcn.CountsCache, s: datetime, e: datetime, held: np.ndarray | None = None
) -> float | None:
    """X's count over [s, e) (module doc, N_w before its floor at h_w): each
    hour bucket wholly inside counts whole; an edge bucket counts the posts
    `held` in its part inside [s, e), plus its posts held nowhere in its hour
    (X's count less the held there, at least 0) times the share of the hour
    inside [s, e). `held` is the account's posts as sorted naive-UTC
    datetime64[us] (None: nothing held, so each edge bucket is pro-rated
    whole). None unless every bucket the window touches is covered
    (CountsCache.span_counts)."""
    if counts.span_counts(s, e) is None:
        return None
    held = np.empty(0, dtype="datetime64[us]") if held is None else held
    n, t = 0.0, s.replace(minute=0, second=0, microsecond=0)
    while t < e:
        a, b = max(t, s), min(t + HOUR, e)
        c = counts._count(t)
        if a == t and b == t + HOUR:
            n += c
        else:
            unplaced = max(0, c - _held_in(held, t, t + HOUR))
            n += _held_in(held, a, b) + unplaced * ((b - a) / HOUR)
        t += HOUR
    return n


def account_windows(
    handle: str,
    manifest_rows: list[dict],
    start: datetime | None = None,
    end: datetime | None = None,
    cursor: datetime | None = None,
    counts: xcn.CountsCache | None = None,
    held: np.ndarray | None = None,
) -> pd.DataFrame:
    """One sampled account's grid (module doc), oldest first: s, e (naive
    UTC), cap, origin (backward | manifest | no_manifest), truncated (the
    collect run's flag for a manifest window as the manifest has it, else
    None) and x_count (X's apportioned count, NaN where the counts do not
    cover the window; apportioned_count, with `held` the account's posts as
    sorted naive-UTC datetime64[us]). `start` / `end` default to the fill's
    (COLLECTION_START, the midnight after ANALYSIS_END); `cursor` (the
    newest post + 1 s) shapes the grid only for an account with no manifest
    rows."""
    start = start or xf.fill_start()
    end = end or xf.fill_end()
    w = settings.ACCOUNT_WINDOW_CAPS.get(handle)
    rows = xf.collect_rows(handle, manifest_rows, end)
    grid = xf.account_grid(rows, w, start, end, cursor)
    # the flag as written: a row without it (or null) is not "read whole"
    trunc = {(r["s"], r["e"]): r["row"].get("truncated") for r in rows}
    counts = counts or xcn.CountsCache(xcn.handle_query(handle), "hour")
    out = []
    for win in grid:
        x = apportioned_count(counts, win.s, win.e, held)
        out.append(
            {
                "handle": handle,
                "s": win.s,
                "e": win.e,
                "cap": win.cap,
                "origin": win.origin,
                "truncated": trunc.get((win.s, win.e)) if win.origin == "manifest" else None,
                "x_count": np.nan if x is None else x,
            }
        )
    cols = ["handle", "s", "e", "cap", "origin", "truncated", "x_count"]
    return pd.DataFrame(out, columns=cols)


def _read_whole(truncated) -> bool:
    """A collect window's manifest says the run kept every post it read: an
    explicit false, never a missing or null flag."""
    return isinstance(truncated, (bool, np.bool_)) and not truncated


def _naive_utc(created_at: pd.Series) -> np.ndarray:
    return pd.to_datetime(created_at, utc=True).dt.tz_localize(None).to_numpy("datetime64[us]")


def window_weights(
    times: np.ndarray, windows: pd.DataFrame, scheme: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """(weight per post, the windows with held / count_used / weight added)
    for one sampled account: `times` its posts' naive-UTC datetime64, `windows`
    its account_windows. A post outside every window weighs 1; a post whose
    weight needs a count the cache lacks gets NaN (module doc)."""
    _check_scheme(scheme)
    n = len(times)
    win = windows.sort_values("s").reset_index(drop=True).copy()
    if win.empty:
        return np.ones(n), win.assign(held=0, count_used=np.nan, weight=np.nan)
    starts = win["s"].to_numpy("datetime64[us]")
    ends = win["e"].to_numpy("datetime64[us]")
    i = np.searchsorted(starts, times, side="right") - 1
    inside = (i >= 0) & (times < ends[np.clip(i, 0, None)])
    held = np.bincount(i[inside], minlength=len(win)).astype(float)
    x = win["x_count"].to_numpy(dtype=float)
    cap = win["cap"].to_numpy(dtype=float)
    complete = np.array([_read_whole(v) for v in win["truncated"]], dtype=bool)
    # N_w: X's count, never under what the window holds; where uncounted, what
    # it holds when the run read it whole; else unknown
    count = np.where(np.isnan(x), np.where(complete, held, np.nan), np.maximum(x, held))
    with np.errstate(invalid="ignore", divide="ignore"):
        if scheme == "volume":
            wt = count / held
        elif scheme == "design":
            wt = np.minimum(cap, count) / held
            # at or over its cap a window weighs cap / held whatever X counts
            wt = np.where(np.isnan(count) & (held >= cap), cap / held, wt)
        else:
            wt = np.ones(len(win))
    wt = np.where(held > 0, wt, np.nan)
    out = np.ones(n)
    out[inside] = wt[i[inside]]
    win["held"] = held.astype(int)
    win["count_used"] = count
    win["weight"] = wt
    return out, win


def post_weights(
    df: pd.DataFrame,
    scheme: str,
    manifest_rows: list[dict] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """(weight per row of df, on its index; every sampled account's window
    table from window_weights, with the posts outside its grid in an
    `outside` attribute). Rows of an account not in ACCOUNT_WINDOW_CAPS
    (take-all, search:, Trump's feed) weigh 1. `manifest_rows` default to
    every collect manifest (x_counts.load_manifests)."""
    _check_scheme(scheme)
    vals = np.ones(len(df))
    if scheme == "none" or df.empty:
        return pd.Series(vals, index=df.index, name=WEIGHT_COL), pd.DataFrame()
    rows = xcn.load_manifests() if manifest_rows is None else manifest_rows
    users = df["user"].astype(str)
    tables, outside = [], {}
    for handle in sorted(set(users) & set(settings.ACCOUNT_WINDOW_CAPS)):
        sel = (users == handle).to_numpy()
        times = _naive_utc(df.loc[sel, "created_at"])
        newest = pd.Timestamp(times.max()).to_pydatetime()
        cursor = newest.replace(microsecond=0) + SECOND
        win = account_windows(handle, rows, start, end, cursor, held=np.sort(times))
        wt, table = window_weights(times, win, scheme)
        vals[sel] = wt
        tables.append(table)
        outside[handle] = int(len(times) - table["held"].sum()) if len(table) else len(times)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out.attrs["outside"] = outside
    return pd.Series(vals, index=df.index, name=WEIGHT_COL), out


def attach(
    df: pd.DataFrame,
    scheme: str,
    manifest_rows: list[dict] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(df with WEIGHT_COL, the window table): weight the full frame, before
    inference.prepare drops the rows with no text or score, so h_w counts every
    post held. inference.bootstrap_cells and the weekly series use the column
    wherever the frame carries it. "none" adds no column."""
    _check_scheme(scheme)
    if scheme == "none":
        return df, pd.DataFrame()
    w, table = post_weights(df, scheme, manifest_rows, start, end)
    out = df.assign(**{WEIGHT_COL: w.to_numpy()})
    out.attrs["window_weights"] = scheme
    logger.info(
        "%s window weights: %d posts of %d sampled accounts weighted, %d with no weight",
        scheme,
        int(df["user"].astype(str).isin(settings.ACCOUNT_WINDOW_CAPS).sum()),
        table["handle"].nunique() if len(table) else 0,
        int(w.isna().sum()),
    )
    return out, table


def require(d: pd.DataFrame) -> None:
    """Refuse a weighted analysis frame with a post that has no weight
    (module doc, *Missing counts*), naming the accounts, the posts and their
    span."""
    if WEIGHT_COL not in d:
        return
    miss = d[d[WEIGHT_COL].isna()]
    if miss.empty:
        return
    by = miss.groupby(miss["user"].astype(str)).size()
    t = pd.to_datetime(miss["created_at"], utc=True)
    handles = ",".join(by.index)
    raise WeightsUnavailable(
        f"{len(miss)} posts have no window weight: their windows have no cached hour counts "
        f"and the weight depends on X's count ({', '.join(f'@{h} {n}' for h, n in by.items())}; "
        f"{t.min():%Y-%m-%d %H:%M} to {t.max():%Y-%m-%d %H:%M} UTC). "
        f"`x-counts --only {handles} --estimate` prices the counts (a paid call: ask the "
        "owner first); raw (no weights) needs none"
    )


def summary(table: pd.DataFrame, scheme: str) -> list[str]:
    """Lines for the CLI: per sampled account, its windows with posts, the
    windows whose count was not cached (and how many of them still got a
    weight), posts outside the grid, empty windows X counts posts in, and
    the spread of its weights."""
    if table.empty:
        return [f"{scheme} weights: no sampled account in the frame (every post weighs 1)"]
    out = [
        (
            f"{scheme} weights ({WEIGHT_COL}) per sampled account: windows holding posts, of "
            "them without cached counts (weighted anyway / not), posts outside the grid, empty "
            "windows X counts posts in (their count), weight min / median / max"
        )
    ]
    outside = table.attrs.get("outside", {})
    for handle, t in table.groupby("handle", sort=True):
        has = t[t["held"] > 0]
        unc = has[has["x_count"].isna()]
        unw = int(unc["weight"].isna().sum())
        empty = t[(t["held"] == 0) & (t["x_count"] > 0)]
        ws = has["weight"].dropna()
        spread = f"{ws.min():.2f} / {ws.median():.2f} / {ws.max():.2f}" if len(ws) else "-"
        out.append(
            f"  @{handle:16s}{len(has):>5d} windows, {len(unc)} uncounted "
            f"({len(unc) - unw} / {unw}), {outside.get(handle, 0)} outside, "
            f"{len(empty)} empty ({empty['x_count'].sum():.0f}), weights {spread}"
        )
    return out


def account_shares(d: pd.DataFrame, war_only: bool = True, by: str = "tier") -> pd.DataFrame:
    """Each account's share of its group's posts per phase (war posts only
    by default) on a prepare()d frame: raw (post counts) and, where the frame
    carries WEIGHT_COL, weighted (the weights' sum). Columns: group, phase,
    user, posts, share, and with weights weight, share_w."""
    x = d[d["on_topic"]] if war_only else d
    keys = [by, "phase", "user"]
    agg = {"posts": ("user", "size")}
    if WEIGHT_COL in x:
        agg["weight"] = (WEIGHT_COL, "sum")
    t = x.groupby(keys, observed=True).agg(**agg).reset_index()
    tot = t.groupby([by, "phase"], observed=True)
    t["share"] = t["posts"] / tot["posts"].transform("sum")
    if "weight" in t:
        t["share_w"] = t["weight"] / tot["weight"].transform("sum")
    return t.rename(columns={by: "group"})
