"""
Full-archive fill (`x-archive fill`): GET /2/tweets/search/all fills the X
cache's holes (timeline-horizon holes, late starts, windows a collect got
short) on collect's own 35 h window grid and with collect's per-window rule,
so the filled stretches follow the current design (docs/decisions.md
2026-11-02). search/all bills $0.005 a post returned; requests are free.

  - Grid (account_grid): each account's collect windows from the x_sampling
    manifests, used as they are (collect_rows: a later run owns its whole
    span, so a `collect --force` with longer windows replaces the earlier
    runs' grid where it ran), extended backward in GAP_FILL_WINDOW_HOURS
    steps to COLLECTION_START (a head under half a window merges into the
    first backward window, as collect merges a short tail), and forward to
    the end of config.timeline.ANALYSIS_END only when half a window or more
    is left there. An account with no manifest rows gets plan_windows over
    the whole range.
  - Target (plan_window): a sampled account's window (ACCOUNT_WINDOW_CAPS)
    should hold its cap, a take-all account's window everything X counts
    (the `x-counts` hour buckets). Windows are picked with the edge buckets
    counted whole, so none that could be missing posts is skipped;
    `--scope empty` keeps only the windows that hold nothing.
  - Requests: from:<handle>, newest first, the collector's tweet fields, no
    expansions. The posts a window holds (and posts bought before but set
    aside) cut it into gaps bounded by their ids (since_id / until_id), so
    no held post is bought again. A time edge goes as the id it maps to
    (id_at: X's ids carry their creation time in ms), because X answers 400
    to any mix of ids and start_time / end_time.
    A gap is skipped only when every hour bucket it touches is held whole,
    by the posts held when the walk reaches it (a bucket several gaps share
    closes once one of them yields its missing post).
    A sampled window's walk goes newest first: next_lower narrows each
    request's start to the last hour that should hold what the window still
    wants, and reads that crossing hour in X_ARCHIVE_FILL_SLICE_MINUTES
    slices. A take-all window reads every open gap whole. Counts never end a
    walk: a window is done when it holds its target or every open gap has
    been asked.
  - Keep rule: a sampled window keeps the newest cap - held posts. The rest
    of a page (search/all returns at least 10) were billed and go to
    X_ARCHIVE_DIR/fill_<handle>_extra.jsonl, outside the cache's glob; a
    later walk takes them before it buys anything.
  - Spend is strict: a window reads at most its need plus
    X_ARCHIVE_FILL_WINDOW_SLACK, the run at most X_ARCHIVE_FILL_MAX_USD less
    what earlier runs billed (billed_so_far: fill_requests.jsonl, reconciled
    with the stored posts, erring high), and every request's max_results
    stays inside both.
  - Storage, per request: a pending line in fill_requests.jsonl before it
    is sent (a request answered but never logged counts at its max_results),
    then the kept posts appended to data/raw/x/<handle>.jsonl with "via":
    "search_all" and fsynced, then the extras, then the answered line. An
    append after a line a kill cut starts on a new line
    (x_collector._append_jsonl), so the cut record is the only one lost.
    A finished window writes its
    manifest row (x_sampling/windows_archive_<date>.jsonl, source
    "archive"), then its state (fill_<handle>.json). A rerun skips windows
    marked done and plans the rest from the cache. collect's cursor is the
    newest cached post (x_collector._latest_created_at), so older posts
    appended out of order never move it back.
  - Errors as in x-counts: 403, 401 and a usage-capped 429 stop the run with
    no further request; another 429 waits per its header at most
    X_429_MAX_RETRIES times.
"""

import bisect
import itertools
import json
import logging
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import requests
import tweepy

from config import settings, timeline
from src.collectors import x_collector as xc
from src.collectors import x_counts as xcn
from src.collectors.x_archive import SEARCH_ENDPOINT

logger = logging.getLogger(__name__)

HOUR = timedelta(hours=1)
SECOND = timedelta(seconds=1)
# X's ids are snowflakes: id >> 22 is the post's creation in ms since this
# epoch (every cached post's id falls inside its created_at second).
SNOWFLAKE_EPOCH_MS = 1288834974657
EPS = 1e-9
VIA = "search_all"
SCOPES = ("below", "empty")
KINDS = ("late_start", "empty", "short")


class CountsMissing(RuntimeError):
    """An account has no hour counts over part of the fill's range."""


class _WindowStop(Exception):
    """A window stops with posts it still wants (its budget, the run's, its
    request guard)."""


# ── Paths and small helpers ─────────────────────────────────────────


def fill_start() -> datetime:
    return settings.COLLECTION_START


def fill_end() -> datetime:
    """Midnight after config.timeline.ANALYSIS_END (an inclusive date)."""
    return datetime.combine(timeline.ANALYSIS_END + timedelta(days=1), time())


def state_path(handle: str) -> Path:
    return settings.X_ARCHIVE_DIR / f"fill_{handle}.json"


def extras_path(handle: str) -> Path:
    return settings.X_ARCHIVE_DIR / f"fill_{handle}_extra.jsonl"


def log_path() -> Path:
    return settings.X_ARCHIVE_DIR / "fill_requests.jsonl"


def manifest_path(run_start: datetime) -> Path:
    return settings.X_SAMPLING_DIR / f"windows_archive_{run_start:%Y-%m-%d}.jsonl"


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def id_at(t: datetime) -> int:
    """The lowest id X can give a post created at naive-UTC `t` or later: a
    since_id of id_at(t) - 1 bounds like start_time t (inclusive), an
    until_id of id_at(t) like end_time t (exclusive)."""
    ms = round(t.replace(tzinfo=UTC).timestamp() * 1000)
    return (ms - SNOWFLAKE_EPOCH_MS) << 22


def _floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def _load_lines(path: Path) -> list[dict]:
    """A JSONL file's records; a line that does not parse (a write cut by a
    kill) is skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            logger.warning("%s: skipping a line that does not parse", path.name)
    return out


def load_state(handle: str) -> dict:
    p = state_path(handle)
    return json.loads(p.read_text()) if p.exists() else {"handle": handle, "windows": {}}


def set_aside(handle: str, stamp: str) -> Path | None:
    """--force: rename the account's state with a stamp, never delete it
    (run_fill then writes it back without the windows it plans again). The
    request log and the extras stay: the spend tally is cumulative."""
    p = state_path(handle)
    if not p.exists():
        return None
    dest = p.with_name(f"{p.stem}.{stamp.replace(':', '').replace('-', '')}{p.suffix}")
    p.rename(dest)
    return dest


# ── Grid ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    s: datetime
    e: datetime
    cap: int | None  # a sampled account's cap for this window; None for take-all
    origin: str  # backward | manifest | forward
    hours: int


def collect_rows(handle: str, manifest_rows: list[dict], end: datetime) -> list[dict]:
    """
    The account's collect windows ({s, e, row}) that start before `end`,
    oldest first: manifest rows with no source (or source "timeline"), so
    the fill's own rows never shape the grid. Runs are laid down oldest
    first (by run_end), and a later run owns its whole span, from its first
    window's start to its last window's end. An earlier run's window that
    starts inside that span is dropped; one that straddles its start ends
    there (an incremental run starts at its cursor: inside the previous
    run's last window, or before it when that run's last windows got
    nothing); one that straddles its end starts there (a forced run that
    stopped part-way). So a `collect --force`, whose windows can be longer,
    replaces the incremental runs' grid wherever it ran, and two runs'
    windows never interleave.
    """
    runs: dict[str, list[dict]] = {}
    for r in manifest_rows:
        if r.get("handle") != handle or (r.get("source") or "timeline") != "timeline":
            continue
        s, e = xcn.parse_ts(r["window_start"]), xcn.parse_ts(r["window_end"])
        if e > s:
            runs.setdefault(r.get("run_end") or "", []).append({"s": s, "e": e, "row": r})
    out: list[dict] = []
    for run in sorted(runs):
        own: list[dict] = []
        for w in sorted(runs[run], key=lambda w: w["s"]):
            while own and w["s"] <= own[-1]["s"]:
                own.pop()
            if own and w["s"] < own[-1]["e"]:
                own[-1] = {**own[-1], "e": w["s"]}
            own.append(w)
        a, b = own[0]["s"], max(w["e"] for w in own)
        kept = []
        for w in out:
            if w["e"] <= a or w["s"] >= b:
                kept.append(w)
                continue
            if w["s"] < a:
                kept.append({**w, "e": a})
            if w["e"] > b:
                kept.append({**w, "s": b})
        out = sorted(kept + own, key=lambda w: w["s"])
    return [w for w in out if w["s"] < end]


def account_grid(rows: list[dict], w: int | None, start: datetime, end: datetime) -> list[Window]:
    """
    The fill's windows for one account, oldest first (module doc): the
    collect windows `rows` (collect_rows) as they are, preceded by windows of
    GAP_FILL_WINDOW_HOURS stepping back from the first one to `start`, so
    their edges keep its minute and second and their ends keep stepping
    through the clock and the week. A head under half a window merges into
    the first backward window (collect's tail rule, mirrored); a longer one
    stands alone. Forward windows (plan_windows) only when the collect
    windows end half a window or more before `end`. Caps: sampled (w) a
    collect window keeps its manifest cap, any other window
    x_collector._window_cap; take-all None.
    """
    h = settings.GAP_FILL_WINDOW_HOURS
    step = timedelta(hours=h)

    def cap(s: datetime, e: datetime, hours: int) -> int | None:
        return xc._window_cap(w, s, e, hours) if w else None

    if not rows:
        return [Window(s, e, cap(s, e, h), "forward", h) for s, e in xc.plan_windows(start, end, h)]
    out: list[Window] = []
    first = rows[0]["s"]
    if first > start:
        edges = [first]
        while edges[0] - step >= start:
            edges.insert(0, edges[0] - step)
        head = edges[0] - start
        if head > timedelta(0):
            if head < step / 2 and len(edges) > 1:
                edges[0] = start
            else:
                edges.insert(0, start)
        out += [Window(s, e, cap(s, e, h), "backward", h) for s, e in itertools.pairwise(edges)]
    for r in rows:
        row, hours = r["row"], int(r["row"].get("hours") or h)
        if not w:
            c = None
        elif row.get("mode") == "sample" and row.get("cap"):
            c = int(row["cap"])
        else:
            c = cap(r["s"], r["e"], hours)
        out.append(Window(r["s"], r["e"], c, "manifest", hours))
    last = rows[-1]["e"]
    if end - last >= step / 2:
        out += [Window(s, e, cap(s, e, h), "forward", h) for s, e in xc.plan_windows(last, end, h)]
    return out


# ── One account: the cache, the extras, the counts ─────────────────


@dataclass(frozen=True)
class Bound:
    """A request edge: a time, or a known post (its time and id), which the
    request excludes exactly through since_id / until_id."""

    t: datetime
    id: str | None = None


@dataclass
class Gap:
    lo: Bound
    hi: Bound
    open: bool  # an hour bucket it touches may hold a post nobody holds
    extra_below: str | None  # the id of the set-aside post at lo, if it is one


class Account:
    """One account's fill inputs, kept current as a run stores posts."""

    def __init__(self, handle: str, tier: str, manifest_rows: list[dict], start, end):
        self.handle, self.tier = handle, tier
        self.w = settings.ACCOUNT_WINDOW_CAPS.get(handle)
        self.mode = "sample" if self.w else "take_all"
        self.counts = xcn.CountsCache(xcn.handle_query(handle), "hour")
        self.rows = collect_rows(handle, manifest_rows, end)
        self.grid = account_grid(self.rows, self.w, start, end)
        self.cached: dict[str, datetime] = {}
        for r in xc._load_jsonl(xc._account_cache_path(handle)):
            pid, ts = str(r.get("id")), r.get("created_at")
            if ts and pid not in self.cached:
                self.cached[pid] = xcn.parse_ts(ts)
        self.extras: dict[str, dict] = {}
        for r in _load_lines(extras_path(handle)):
            pid = str(r["id"])
            if pid not in self.cached:
                self.extras[pid] = r
        self._points: list[tuple[datetime, int, str]] = sorted(
            [(t, int(pid), pid) for pid, t in self.cached.items()]
            + [(xcn.parse_ts(r["created_at"]), int(pid), pid) for pid, r in self.extras.items()]
        )
        self._times = [p[0] for p in self._points]
        self._buckets = Counter(_floor_hour(p[0]) for p in self._points)
        self.state = load_state(handle)

    # known posts: the cache and the set-aside extras
    def known_in(self, s: datetime, e: datetime) -> list[tuple[datetime, int, str]]:
        i, j = bisect.bisect_left(self._times, s), bisect.bisect_left(self._times, e)
        return self._points[i:j]

    def held_in(self, s: datetime, e: datetime) -> int:
        return sum(1 for _, _, pid in self.known_in(s, e) if pid in self.cached)

    def first_cached(self) -> datetime | None:
        return min(self.cached.values()) if self.cached else None

    def count_at(self, t: datetime) -> int:
        return self.counts._count(t)

    def bucket_open(self, t: datetime) -> bool:
        """X counts more posts in the hour bucket at t than are known."""
        return self.count_at(t) > self._buckets.get(t, 0)

    def range_open(self, lo: Bound, hi: Bound) -> bool:
        """Any hour bucket the range [lo, hi] touches is open (bucket_open),
        by the posts known now."""
        top = hi.t + SECOND if hi.id else hi.t
        t = _floor_hour(lo.t)
        while t < top:
            if self.bucket_open(t):
                return True
            t += HOUR
        return False

    def gaps(self, s: datetime, e: datetime) -> list[Gap]:
        """[s, e) cut at the known posts, newest gap first."""
        pts = self.known_in(s, e)
        bounds = [Bound(s)] + [Bound(t, pid) for t, _, pid in pts] + [Bound(e)]
        out = []
        for lo, hi in reversed(list(itertools.pairwise(bounds))):
            extra = lo.id if lo.id and lo.id in self.extras else None
            out.append(Gap(lo, hi, self.range_open(lo, hi), extra))
        return out

    def _add_point(self, t: datetime, pid: str) -> None:
        k = (t, int(pid), pid)
        i = bisect.bisect_left(self._points, k)
        if i < len(self._points) and self._points[i] == k:
            return
        self._points.insert(i, k)
        self._times.insert(i, t)
        self._buckets[_floor_hour(t)] += 1

    def add(self, rec: dict, extra: bool) -> None:
        pid, t = str(rec["id"]), xcn.parse_ts(rec["created_at"])
        if extra:
            self.extras[pid] = rec
        else:
            self.cached[pid] = t
            self.extras.pop(pid, None)
        self._add_point(t, pid)

    def is_known(self, pid: str) -> bool:
        return pid in self.cached or pid in self.extras


# ── The expected-reads model and the walk's narrowing ──────────────


class Mass:
    """
    X's hour counts over [s, e) less the known posts, bucket by bucket, each
    bucket's rest spread evenly inside it; an edge bucket counts the share
    of it inside the window. lo / hi / mid are X's count with the edge
    buckets left out / counted whole / pro-rated.
    """

    def __init__(self, s: datetime, e: datetime, count_at, known_times: list[datetime]):
        self.s, self.e = s, e
        self.segs: list[tuple[datetime, datetime, float]] = []
        self.lo = self.hi = 0
        self.mid = 0.0
        t = _floor_hour(s)
        while t < e:
            a, b = max(t, s), min(t + HOUR, e)
            share = (b - a) / HOUR
            n = count_at(t)
            held = bisect.bisect_left(known_times, b) - bisect.bisect_left(known_times, a)
            self.segs.append((a, b, max(0.0, n * share - held)))
            self.hi += n
            self.mid += n * share
            if share >= 1:
                self.lo += n
            t += HOUR

    def unheld(self, a: datetime, b: datetime) -> float:
        m = 0.0
        for x, y, v in self.segs:
            lo, hi = max(a, x), min(b, y)
            if hi > lo and v:
                m += v * ((hi - lo) / (y - x))
        return m


def narrow_start(m: Mass, floor: datetime, cursor: datetime, want: float) -> datetime:
    """The latest hour boundary in (floor, cursor) from which the model holds
    `want` posts up to the cursor; floor if none does."""
    t = _floor_hour(cursor)
    if t == cursor:
        t -= HOUR
    while t > floor:
        if m.unheld(t, cursor) >= want - EPS:
            return t
        t -= HOUR
    return floor


def next_lower(m: Mass, floor: datetime, cursor: datetime, want: float, slice_min: int) -> datetime:
    """
    A sampled walk's next request is [next_lower, cursor). The start is
    narrowed to the hour where the model's posts reach `want` (narrow_start).
    With slices, when that hour holds more than the window wants: the whole
    hours after it go first in one request if the model puts posts there;
    otherwise the crossing hour's newest slice is read, and the hours above
    it (empty by the counts) ride along in the same request, so no stretch is
    left unasked.
    """
    a = narrow_start(m, floor, cursor, want)
    if a <= floor or not slice_min or m.unheld(a, cursor) <= want + EPS:
        return a
    b = min(cursor, a + HOUR)
    if b < cursor and m.unheld(b, cursor) > EPS:
        return b
    return max(a, b - timedelta(minutes=slice_min))


def simulate(
    m: Mass, gaps: list[Gap], need: float, take_all: bool, slice_min: int
) -> tuple[float, int]:
    """Expected (reads, requests) of a window's walk when the model's posts
    are what search/all returns: the same gaps and narrowing as fill_window,
    each request returning min(max_results, the model's posts in it)."""
    lo_res = settings.X_ARCHIVE_SEARCH_MAX_RESULTS
    per = settings.X_ARCHIVE_FILL_MAX_RESULTS
    read, req = 0.0, 0
    if take_all:
        for g in gaps:
            if g.open:
                mass = m.unheld(g.lo.t, g.hi.t)
                read += mass
                req += max(1, math.ceil(mass / per - EPS))
        return read, req
    left = need
    for g in gaps:
        if left <= EPS:
            break
        if not g.open:
            continue
        cursor = g.hi.t
        while left > EPS and cursor > g.lo.t:
            lower = next_lower(m, g.lo.t, cursor, left, slice_min)
            got = min(min(per, max(lo_res, math.ceil(left - EPS))), m.unheld(lower, cursor))
            req += 1
            read += got
            left -= got
            cursor = lower
    return read, req


# ── Planning (no call) ──────────────────────────────────────────────


def _month(t: datetime) -> str:
    return f"{t:%Y-%m}"


def plan_window(acct: Account, win: Window) -> dict:
    """
    One window against its target, from the cache, the extras, the counts
    and the account's fill state (no call): held (cached posts in [s, e)),
    X's count (lo, hi, mid), kind, need_hi, budget (need_hi + the slack),
    whether the fill touches it, and the expected reads and requests (the
    slice walk, and whole hours as an upper figure). A window an earlier
    run stopped is planned again for at least one more request.
    """
    s, e = win.s, win.e
    span = acct.counts.span_counts(s, e)
    if span is None:
        return {"handle": acct.handle, "s": s, "e": e, "uncounted": True}
    known = acct.known_in(s, e)
    held = sum(1 for _, _, pid in known if pid in acct.cached)
    m = Mass(s, e, acct.count_at, [t for t, _, _ in known])
    x_lo, x_hi = span
    sample = win.cap is not None
    entry = (acct.state.get("windows") or {}).get(xcn.z(s)) or {}
    if entry.get("end") != xcn.z(e):
        entry = {}
    if sample:
        need_hi = min(win.cap, x_hi) - held if held < win.cap else 0
        target = min(win.cap, x_hi)
    else:
        need_hi = x_hi - held
        target = x_hi
    if entry.get("status") == "stopped" and (not sample or held < win.cap):
        need_hi = max(need_hi, 1)
    first = acct.first_cached()
    kind = "late_start" if first is None or e <= first else "empty" if held == 0 else "short"
    wp = {
        "handle": acct.handle,
        "tier": acct.tier,
        "mode": acct.mode,
        "s": s,
        "e": e,
        "cap": win.cap,
        "hours": win.hours,
        "origin": win.origin,
        "held": held,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "x_mid": m.mid,
        "target": target,
        "kind": kind,
        "month": _month(s),
        "need_hi": max(0, need_hi),
        "touch": need_hi >= 1,
        "status": entry.get("status"),
    }
    if not wp["touch"]:
        return wp
    wp["budget"] = need_hi + settings.X_ARCHIVE_FILL_WINDOW_SLACK
    gaps = acct.gaps(s, e)
    unheld = m.unheld(s, e)
    if sample:
        # the walk narrows on what the window wants (counts never end it),
        # so the model does too; it stores what the model's posts allow
        need = float(win.cap - held)
        wp["new_exp"] = min(max(0.0, min(win.cap, m.mid) - held), unheld)
    else:
        need = unheld
        wp["new_exp"] = min(max(0.0, m.mid - held), unheld)
    wp["exp_slice"], wp["req"] = simulate(
        m, gaps, need, not sample, settings.X_ARCHIVE_FILL_SLICE_MINUTES
    )
    wp["exp_hour"], _ = simulate(m, gaps, need, not sample, 0)
    return wp


def plus_windows(acct: Account) -> int:
    """Collect windows that got more posts than X counts with the edge
    buckets whole (the report's '+'): a sign X's counts leave posts out."""
    n = 0
    for r in acct.rows:
        row = r["row"]
        span = acct.counts.span_counts(
            xcn.parse_ts(row["window_start"]), xcn.parse_ts(row["window_end"])
        )
        n += bool(span and row.get("got", 0) > span[1])
    return n


def billed_so_far() -> dict:
    """What every fill run has billed, erring high: the posts each answered
    request returned; each request with a pending line and no answered line
    (X may have served it while the run was killed, powered off or lost the
    connection) at its max_results; and any stored fill post no answered
    line names (a run killed between storing and logging). A request killed
    between those two writes is counted twice. {posts, requests, unanswered,
    unlogged, usd}."""
    lines = _load_lines(log_path())
    answered = {x.get("req") for x in lines if not x.get("pending")}
    posts = requests_n = unanswered = 0
    ids: set[str] = set()
    for line in lines:
        if line.get("pending"):
            if not line.get("req") or line["req"] not in answered:
                unanswered += 1
                requests_n += 1
                posts += int(line.get("max_results") or settings.X_ARCHIVE_FILL_MAX_RESULTS)
            continue
        requests_n += 1
        posts += int(line.get("returned") or 0)
        ids.update(str(i) for i in line.get("ids") or [])
    stored: set[str] = set()
    for path in sorted(settings.X_RAW_DIR.glob("*.jsonl")):
        if path.stem.startswith("search_"):
            continue
        stored.update(str(r.get("id")) for r in xc._load_jsonl(path) if r.get("via") == VIA)
    for path in sorted(settings.X_ARCHIVE_DIR.glob("fill_*_extra.jsonl")):
        stored.update(str(r.get("id")) for r in _load_lines(path))
    unlogged = len(stored - ids)
    total = posts + unlogged
    return {
        "posts": total,
        "requests": requests_n,
        "unanswered": unanswered,
        "unlogged": unlogged,
        "usd": total * settings.X_READ_COST_USD,
    }


def cap_posts() -> int:
    """X_ARCHIVE_FILL_MAX_USD in posts."""
    return math.floor(settings.X_ARCHIVE_FILL_MAX_USD / settings.X_READ_COST_USD + 1e-9)


def plan_fill(
    accounts: dict[str, list[str]],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    scope: str = "below",
    force: bool = False,
    limit: int | None = None,
    manifest_rows: list[dict] | None = None,
) -> dict:
    """
    What a fill run would do, from the cache, the extras, the counts, the
    manifests, the fill states and the request log (no call, no
    credentials). Windows whose window_start falls in [since, until), below
    their target (scope "empty": holding nothing), not marked done (unless
    force), in order of (window_start, handle), the first `limit` of them.
    Raises CountsMissing for an account whose windows the hour counts do not
    cover. Returns the windows, totals by account, month and kind, the
    ceiling, what earlier runs billed, the run maximum and a refusal (None
    or the reason).
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
    start, end = fill_start(), fill_end()
    since = max(since or start, start)
    until = min(until or end, end)
    rows = xcn.load_manifests() if manifest_rows is None else manifest_rows
    windows: list[dict] = []
    accts: dict[str, Account] = {}
    by_acct: dict[str, dict] = {}
    missing: list[str] = []
    for tier, handles in accounts.items():
        for h in handles:
            acct = Account(h, tier, rows, start, end)
            accts[h] = acct
            tot = {"mode": acct.mode, "win": 0, "plus": plus_windows(acct)}
            tot.update({k: 0 for k in KINDS})
            tot.update(new=0.0, slice=0.0, hour=0.0, ceiling=0, req=0)
            by_acct[h] = tot
            uncounted = 0
            for win in acct.grid:
                if not since <= win.s < until:
                    continue
                wp = plan_window(acct, win)
                if wp.get("uncounted"):
                    uncounted += 1
                    continue
                if not wp["touch"] or (scope == "empty" and wp["held"]):
                    continue
                if wp["status"] == "done" and not force:
                    continue
                windows.append(wp)
            if uncounted:
                missing.append(h)
    if missing:
        raise CountsMissing(
            "no hour counts cached over part of the fill's range for "
            + ", ".join(f"@{h}" for h in missing)
            + f": `x-counts --only {','.join(missing)} --estimate` prices them "
            "(a paid call: ask the user first)"
        )
    windows.sort(key=lambda w: (w["s"], w["handle"]))
    if limit is not None:
        windows = windows[:limit]
    months: dict[str, dict] = {}
    kinds = {k: {"win": 0, "new": 0.0, "slice": 0.0, "hour": 0.0, "ceiling": 0} for k in KINDS}
    for wp in windows:
        t = by_acct[wp["handle"]]
        t["win"] += 1
        t[wp["kind"]] += 1
        for k, src in (("new", "new_exp"), ("slice", "exp_slice"), ("hour", "exp_hour")):
            t[k] += wp[src]
            kinds[wp["kind"]][k] += wp[src]
        t["ceiling"] += wp["budget"]
        t["req"] += wp["req"]
        kinds[wp["kind"]]["win"] += 1
        kinds[wp["kind"]]["ceiling"] += wp["budget"]
        mo = months.setdefault(wp["month"], {"slice": 0.0, "ceiling": 0, "win": 0})
        mo["slice"] += wp["exp_slice"]
        mo["ceiling"] += wp["budget"]
        mo["win"] += 1
    ceiling = sum(w["budget"] for w in windows)
    billed = billed_so_far()
    left = max(0, cap_posts() - billed["posts"])
    run_max = min(ceiling, left)
    expected = sum(w["exp_slice"] for w in windows)
    req = sum(w["req"] for w in windows)
    cost = settings.X_READ_COST_USD
    floor = settings.X_ARCHIVE_SEARCH_MAX_RESULTS
    refusal = None
    if windows:
        if run_max * cost > settings.X_RUN_BUDGET_USD + 1e-9:
            refusal = (
                f"the run maximum ${run_max * cost:.2f} exceeds X_RUN_BUDGET_USD "
                f"${settings.X_RUN_BUDGET_USD:.2f}"
            )
        elif run_max < floor:
            refusal = (
                f"X_ARCHIVE_FILL_MAX_USD ${settings.X_ARCHIVE_FILL_MAX_USD:.2f} is used up "
                f"(${billed['usd']:.2f} billed by earlier fill runs; {left} posts left, a "
                f"request takes at least {floor}): raise it deliberately in config/settings.py "
                "after the user approves the estimate"
            )
        elif req > settings.X_ARCHIVE_FILL_MAX_REQUESTS:
            refusal = (
                f"{req:,} requests planned is over X_ARCHIVE_FILL_MAX_REQUESTS "
                f"({settings.X_ARCHIVE_FILL_MAX_REQUESTS:,}): use --only, --since/--until or "
                "--limit"
            )
    return {
        "windows": windows,
        "accounts": by_acct,
        "accts": accts,
        "months": dict(sorted(months.items())),
        "kinds": kinds,
        "scope": scope,
        "since": since,
        "until": until,
        "force": force,
        "limit": limit,
        "new": sum(w["new_exp"] for w in windows),
        "expected": expected,
        "expected_hour": sum(w["exp_hour"] for w in windows),
        "ceiling": ceiling,
        "requests": req,
        "billed": billed,
        "cap_left": left,
        "run_max": run_max,
        "refusal": refusal,
    }


# ── Run ─────────────────────────────────────────────────────────────


def _record(tweet, handle: str, tier: str) -> dict:
    return {**xc._timeline_record(tweet, handle, tier), "via": VIA}


class FillRun:
    """One run's counters and its writes (module doc)."""

    def __init__(self, client, plan: dict):
        self.client, self.plan = client, plan
        self.started = _now()
        self.run_id = xcn.z(self.started)
        self.left = plan["run_max"]
        self.cap = xcn.RequestCap(
            settings.X_ARCHIVE_FILL_MAX_REQUESTS, settings.X_ARCHIVE_FILL_DELAY_S
        )
        self.read = self.kept = self.extra = self.reread = self.promoted = 0
        self.done: list[dict] = []
        self.stopped: list[dict] = []
        self.stop: str | None = None
        self.error: str | None = None

    # one request
    def _request(self, wr: dict, lo: Bound, hi: Bound, max_results: int, token: str | None):
        """Send one request and store its answer. A pending log line goes
        first, so a request X answers but the run never logs (a kill, a
        power-off, a connection dropped mid-answer) is still counted, at its
        max_results (billed_so_far); the answered line, with the same req
        id, is written after the posts. An X error status serves no posts:
        its line is closed at 0 before the error propagates."""
        acct = wr["acct"]
        params = {
            "query": xcn.handle_query(acct.handle),
            "max_results": max_results,
            "sort_order": "recency",
            "tweet_fields": xc._TWEET_FIELDS,
        }
        # Ids only: X answers 400 to any mix of since_id / until_id with
        # start_time / end_time (the 2026-11-03 pilot), so a time edge goes
        # as the id it maps to (id_at), which bounds exactly as the time would.
        params["since_id"] = lo.id or str(id_at(lo.t) - 1)
        params["until_id"] = hi.id or str(id_at(hi.t))
        if token:
            params["next_token"] = token
        self.cap.before()
        line = {
            "run": self.run_id,
            "req": uuid.uuid4().hex,
            "handle": acct.handle,
            "window_start": xcn.z(wr["s"]),
            "lo_t": xcn.z(lo.t),
            "hi_t": xcn.z(hi.t),
            "since_id": params.get("since_id"),
            "until_id": params.get("until_id"),
            "next_token": bool(token),
            "max_results": max_results,
        }
        xc._append_jsonl(log_path(), [{**line, "at": xcn.z(_now()), "pending": True}])
        try:
            resp = xcn.call_with_backoff(self.client.search_all_tweets, SEARCH_ENDPOINT, **params)
        except (xcn.XRefused, tweepy.errors.HTTPException) as err:
            xc._append_jsonl(
                log_path(),
                [
                    {
                        **line,
                        "at": xcn.z(_now()),
                        "returned": 0,
                        "ids": [],
                        "error": type(err).__name__,
                    }
                ],
            )
            raise
        self.cap.after()
        wr["requests"] += 1
        recs = sorted(
            (_record(t, acct.handle, acct.tier) for t in resp.data or []),
            key=lambda r: int(r["id"]),
            reverse=True,
        )
        n = len(recs)
        if n > max_results:
            logger.warning(
                "@%s: asked for %d posts, got %d (all billed)", acct.handle, max_results, n
            )
        self.read += n
        self.left -= n
        wr["read"] += n
        keep, extra, reread = [], [], []
        for r in recs:
            pid = r["id"]
            if acct.is_known(pid) or any(pid == k["id"] for k in keep + extra):
                reread.append(pid)
                continue
            inside = wr["s"] <= xcn.parse_ts(r["created_at"]) < wr["e"]
            if inside and (wr["want"] is None or len(keep) < wr["want"]):
                keep.append(r)
            else:
                extra.append({**r, "fill_window": xcn.z(wr["s"])})
        if keep:
            xc._append_jsonl(xc._account_cache_path(acct.handle), keep)
            for r in keep:
                acct.add(r, extra=False)
        if extra:
            xc._append_jsonl(extras_path(acct.handle), extra)
            for r in extra:
                acct.add(r, extra=True)
        xc._append_jsonl(
            log_path(),
            [
                {
                    **line,
                    "at": xcn.z(_now()),
                    "returned": n,
                    "ids": [r["id"] for r in recs],
                    "kept": len(keep),
                    "extra": len(extra),
                    "reread": len(reread),
                    "more": bool((resp.meta or {}).get("next_token")),
                }
            ],
        )
        self.kept += len(keep)
        self.extra += len(extra)
        self.reread += len(reread)
        wr["new"] += len(keep)
        wr["extra"] += len(extra)
        wr["reread"] += len(reread)
        if wr["want"] is not None:
            wr["want"] -= len(keep)
        if reread:
            logger.warning(
                "@%s: %d post(s) already held came back (billed): %s",
                acct.handle,
                len(reread),
                ", ".join(reread[:5]),
            )
        return (resp.meta or {}).get("next_token")

    def _max_results(self, wr: dict) -> int:
        """The next request's max_results, or _WindowStop when the window's
        or the run's reads left are under search/all's minimum."""
        floor = settings.X_ARCHIVE_SEARCH_MAX_RESULTS
        if wr["budget"] - wr["read"] < floor:
            raise _WindowStop("window budget reached")
        if self.left < floor:
            raise _WindowStop("run maximum reached")
        if wr["requests"] >= settings.X_ARCHIVE_FILL_MAX_REQUESTS_PER_WINDOW:
            raise _WindowStop("window request guard reached")
        want = settings.X_ARCHIVE_FILL_MAX_RESULTS if wr["want"] is None else wr["want"]
        return min(
            settings.X_ARCHIVE_FILL_MAX_RESULTS,
            max(floor, want),
            wr["budget"] - wr["read"],
            self.left,
        )

    def _read_range(self, wr: dict, lo: Bound, hi: Bound) -> None:
        """One range, following next_token while the window wants posts."""
        token = None
        while True:
            token = self._request(wr, lo, hi, self._max_results(wr), token)
            if not token or (wr["want"] is not None and wr["want"] <= 0):
                return

    def _promote(self, wr: dict, pid: str) -> None:
        """A post bought before and set aside goes into the cache, free."""
        acct = wr["acct"]
        rec = {k: v for k, v in acct.extras[pid].items() if k != "fill_window"}
        xc._append_jsonl(xc._account_cache_path(acct.handle), [rec])
        acct.add(rec, extra=False)
        self.promoted += 1
        wr["new"] += 1
        if wr["want"] is not None:
            wr["want"] -= 1

    def _walk(self, wr: dict) -> None:
        """The gaps are cut once, before the first request, but each one is
        read only while an hour bucket it touches is still open by the posts
        stored so far: a bucket shared by several gaps closes once one of
        them yields its missing post."""
        acct, s, e = wr["acct"], wr["s"], wr["e"]
        gaps = acct.gaps(s, e)
        if wr["want"] is None:  # take-all: every post
            for g in gaps:
                if g.extra_below:
                    self._promote(wr, g.extra_below)
            for g in gaps:
                if g.open and acct.range_open(g.lo, g.hi):
                    self._read_range(wr, g.lo, g.hi)
            return
        known = acct.known_in(s, e)
        m = Mass(s, e, acct.count_at, [t for t, _, _ in known])
        slice_min = settings.X_ARCHIVE_FILL_SLICE_MINUTES
        for g in gaps:
            if wr["want"] <= 0:
                return
            if g.open and acct.range_open(g.lo, g.hi):
                cursor, upper = g.hi.t, g.hi
                while wr["want"] > 0 and cursor > g.lo.t:
                    lower = next_lower(m, g.lo.t, cursor, wr["want"], slice_min)
                    lo = g.lo if lower <= g.lo.t else Bound(lower)
                    self._read_range(wr, lo, upper)
                    cursor, upper = lower, Bound(lower)
            if g.extra_below and wr["want"] > 0:
                self._promote(wr, g.extra_below)

    def fill_window(self, wp: dict) -> dict:
        """Walk one planned window (module doc), then write its manifest row
        and its state. A tweepy, network or X refusal error still writes them
        (the window stopped) before it propagates."""
        acct: Account = self.plan["accts"][wp["handle"]]
        s, e = wp["s"], wp["e"]
        held = acct.held_in(s, e)
        sample = wp["cap"] is not None
        wr = {
            "acct": acct,
            "s": s,
            "e": e,
            "budget": wp["budget"],
            "want": wp["cap"] - held if sample else None,
            "read": 0,
            "new": 0,
            "extra": 0,
            "reread": 0,
            "requests": 0,
        }
        status, stop, failure = "done", None, None
        try:
            if wr["want"] is None or wr["want"] > 0:
                self._walk(wr)
        except _WindowStop as why:
            status, stop = "stopped", str(why)
        except (xcn.XRefused, xcn.RunCapReached) as err:
            status, stop, failure = "stopped", str(err), err
        except (tweepy.errors.TweepyException, requests.exceptions.RequestException) as err:
            status, stop, failure = "stopped", f"{type(err).__name__}: {err}", err
        row = self._finish(wp, wr, held, status, stop)
        (self.done if status == "done" else self.stopped).append(row)
        logger.info(
            "@%s %s..%s %s: held %d -> %d (read %d, extra %d, %d requests)%s",
            acct.handle,
            f"{s:%Y-%m-%d %H:%M}",
            f"{e:%m-%d %H:%M}",
            status,
            held,
            row["got"],
            wr["read"],
            wr["extra"],
            wr["requests"],
            f" -- {stop}" if stop else "",
        )
        if failure is not None:
            raise failure
        return row

    def _finish(self, wp: dict, wr: dict, held: int, status: str, stop: str | None) -> dict:
        acct = wr["acct"]
        s, e = wp["s"], wp["e"]
        mine = [(t, pid) for t, _, pid in acct.known_in(s, e) if pid in acct.cached]
        got = len(mine)
        sample = wp["cap"] is not None
        row = {
            "run_end": self.run_id,
            "handle": acct.handle,
            "tier": acct.tier,
            "mode": acct.mode,
            "hours": wp["hours"],
            "window_start": xcn.z(s),
            "window_end": xcn.z(e),
            "cap": wp["cap"] if sample else held + wp["budget"],
            "got": got,
            "new": wr["new"],
            "truncated": status != "done" or (sample and wp["x_hi"] > got),
            "oldest_kept": mine[0][0].isoformat() + "+00:00" if mine else None,
            "newest_kept": mine[-1][0].isoformat() + "+00:00" if mine else None,
            "source": "archive",
            "origin": wp["origin"],
            "held_before": held,
            "read": wr["read"],
            "extra": wr["extra"],
            "reread": wr["reread"],
            "requests": wr["requests"],
            "budget": wp["budget"],
            "x_lo": wp["x_lo"],
            "x_hi": wp["x_hi"],
            "target": wp["target"],
            "kind": wp["kind"],
            "status": status,
            "stop": stop,
        }
        xc._append_jsonl(manifest_path(self.started), [row])  # the row, then the state
        acct.state.setdefault("windows", {})[xcn.z(s)] = {
            "end": xcn.z(e),
            "cap": wp["cap"],
            "held_before": held,
            "need_hi": wp["need_hi"],
            "budget": wp["budget"],
            "read": wr["read"],
            "new": wr["new"],
            "extra": wr["extra"],
            "reread": wr["reread"],
            "requests": wr["requests"],
            "status": status,
            "stop": stop,
            "run": self.run_id,
            "finished_at": xcn.z(_now()),
        }
        acct.state.setdefault("created_at", self.run_id)
        xcn._save_json(state_path(acct.handle), acct.state)
        return row

    def result(self) -> dict:
        billed = billed_so_far()
        return {
            "run": self.run_id,
            "done": self.done,
            "stopped": self.stopped,
            "kept": self.kept,
            "promoted": self.promoted,
            "extra": self.extra,
            "reread": self.reread,
            "read": self.read,
            "requests": self.cap.used,
            "billed": billed,
            "stop": self.stop,
            "error": self.error,
            "left": len(self.plan["windows"]) - len(self.done) - len(self.stopped),
        }


def run_fill(client, plan: dict) -> dict:
    """
    Fill the planned windows in plan order. A forced plan first sets the
    state of each account it plans aside (renamed) and writes it back
    without the planned windows. The run stops, with every window walked so far
    saved, when its maximum is spent (the windows left stay planned), at
    X_ARCHIVE_FILL_MAX_REQUESTS, or on an error: 403, 401 and a usage-capped
    429 make no further request; tweepy and network errors too. Returns the
    run's counts (FillRun.result); `error` is set when an error stopped it.
    """
    run = FillRun(client, plan)
    floor = settings.X_ARCHIVE_SEARCH_MAX_RESULTS
    if plan["force"]:
        planned: dict[str, set[str]] = {}
        for wp in plan["windows"]:
            planned.setdefault(wp["handle"], set()).add(xcn.z(wp["s"]))
        for h, starts in planned.items():
            acct = plan["accts"][h]
            moved = set_aside(h, run.run_id)
            if moved:
                logger.info("@%s: fill state set aside as %s", h, moved.name)
            # windows outside this plan keep what they were marked
            kept = {k: v for k, v in (acct.state.get("windows") or {}).items() if k not in starts}
            acct.state = {**acct.state, "windows": kept}
            xcn._save_json(state_path(h), acct.state)
    for wp in plan["windows"]:
        if run.left < floor:
            run.stop = (
                f"the run maximum ({plan['run_max']} posts) is spent; the windows left "
                "stay planned for the next run"
            )
            break
        try:
            run.fill_window(wp)
        except xcn.XRefused as e:
            run.error = f"{e}. No further requests were made{e.hint}"
            break
        except xcn.RunCapReached as e:
            run.error = f"{e}; rerun to fill the windows left"
            break
        except (tweepy.errors.TweepyException, requests.exceptions.RequestException) as e:
            run.error = f"{type(e).__name__}: {e}; what was read is saved, rerun to resume"
            break
    return run.result()


# ── Text ────────────────────────────────────────────────────────────


def _usd(n: float) -> str:
    return f"${n * settings.X_READ_COST_USD:,.2f}"


def estimate_lines(plan: dict) -> list[str]:
    """The --estimate text: by account, by month and kind, totals, billed so
    far, the run maximum against both caps, and the wall time."""
    rng = f"{plan['since']:%Y-%m-%d %H:%M}..{plan['until']:%Y-%m-%d %H:%M} UTC"
    out = [
        (
            f"x-archive fill (--scope {plan['scope']}): search/all on collect's "
            f"{settings.GAP_FILL_WINDOW_HOURS} h grid, windows starting {rng}"
            + (f", the first {plan['limit']}" if plan["limit"] else "")
            + (", done windows planned again (--force)" if plan["force"] else "")
        ),
        (
            f"${settings.X_READ_COST_USD} a post returned (X_READ_COST_USD); requests are "
            "free; no user lookup. Planned from the cache and X's hour counts: no API call."
        ),
        "",
        (
            f"  {'account':<15}{'mode':<9}{'win':>5}{'late':>6}{'empty':>6}{'short':>6}"
            f"{'new':>7}{'reads':>7}{'hour':>7}{'ceiling':>9}{'$ exp':>8}{'$ ceil':>8}"
            f"{'req':>6}{'+':>4}"
        ),
    ]
    tot = Counter()
    for h, a in plan["accounts"].items():
        if not a["win"] and not a["plus"]:
            continue
        out.append(
            f"  {h[:14]:<15}{a['mode']:<9}{a['win']:>5}{a['late_start']:>6}{a['empty']:>6}"
            f"{a['short']:>6}{a['new']:>7,.0f}{a['slice']:>7,.0f}{a['hour']:>7,.0f}"
            f"{a['ceiling']:>9,}{_usd(a['slice']):>8}{_usd(a['ceiling']):>8}{a['req']:>6,}"
            f"{a['plus']:>4}"
        )
        for k in ("win", "late_start", "empty", "short", "new", "slice", "hour", "ceiling", "req"):
            tot[k] += a[k]
        tot["plus"] += a["plus"]
    out.append(
        f"  {'TOTAL':<24}{tot['win']:>5}{tot['late_start']:>6}{tot['empty']:>6}"
        f"{tot['short']:>6}{tot['new']:>7,.0f}{tot['slice']:>7,.0f}{tot['hour']:>7,.0f}"
        f"{tot['ceiling']:>9,}{_usd(tot['slice']):>8}{_usd(tot['ceiling']):>8}{tot['req']:>6,}"
        f"{tot['plus']:>4}"
    )
    out.append(
        "  (new: posts expected to be stored; reads: expected, the crossing hour in "
        f"{settings.X_ARCHIVE_FILL_SLICE_MINUTES}-minute slices; hour: whole hours; ceiling: "
        "each window's need + "
        f"{settings.X_ARCHIVE_FILL_WINDOW_SLACK}; +: collect windows that got more than X counts)"
    )
    if plan["months"]:
        out.append(
            "\n  By month of window start (as --since/--until select): windows, "
            "expected reads, ceiling"
        )
        for mo, m in plan["months"].items():
            out.append(
                f"    {mo}  {m['win']:>5}  {m['slice']:>7,.0f} {_usd(m['slice']):>8}"
                f"  {m['ceiling']:>7,} {_usd(m['ceiling']):>8}"
            )
        out.append("  By kind: windows, new posts, expected reads, ceiling")
        for k, m in plan["kinds"].items():
            out.append(
                f"    {k:<11}{m['win']:>5}  {m['new']:>7,.0f}  {m['slice']:>7,.0f} "
                f"{_usd(m['slice']):>8}  {m['ceiling']:>7,} {_usd(m['ceiling']):>8}"
            )
    b = plan["billed"]
    secs = plan["requests"] * settings.X_ARCHIVE_FILL_DELAY_S
    notes = []
    if b["unanswered"]:
        notes.append(f"{b['unanswered']} sent with no answer logged, counted at their max_results")
    if b["unlogged"]:
        notes.append(f"{b['unlogged']} posts stored without a log line")
    billed_note = f" ({'; '.join(notes)})" if notes else ""
    out += [
        "",
        (
            f"Windows to fill: {len(plan['windows']):,} (late start {tot['late_start']}, "
            f"empty {tot['empty']}, short {tot['short']}); new posts expected "
            f"{plan['new']:,.0f}"
        ),
        (
            f"Expected reads: {plan['expected']:,.0f} ≈ {_usd(plan['expected'])} (X's hourly "
            f"counts spread evenly inside each hour); {plan['expected_hour']:,.0f} ≈ "
            f"{_usd(plan['expected_hour'])} if posts bunch (whole hours)"
        ),
        (
            f"Ceiling (each window's need + {settings.X_ARCHIVE_FILL_WINDOW_SLACK}): "
            f"{plan['ceiling']:,} ≈ {_usd(plan['ceiling'])}"
        ),
        (
            f"Billed by earlier fill runs: {b['posts']:,} posts ≈ ${b['usd']:.2f} in "
            f"{b['requests']:,} requests{billed_note}, of X_ARCHIVE_FILL_MAX_USD "
            f"${settings.X_ARCHIVE_FILL_MAX_USD:.2f}"
        ),
        (
            f"Run maximum: {plan['run_max']:,} posts ≈ {_usd(plan['run_max'])} = min(ceiling, "
            f"the cap's {plan['cap_left']:,} left); X_RUN_BUDGET_USD "
            f"${settings.X_RUN_BUDGET_USD:.2f}"
        ),
        (
            f"Requests ≈ {plan['requests']:,} at X_ARCHIVE_FILL_DELAY_S "
            f"{settings.X_ARCHIVE_FILL_DELAY_S:g} s ≈ {secs / 3600:.1f} h"
        ),
    ]
    if plan["windows"] and plan["expected"] > plan["run_max"]:
        out.append(
            f"NOTE: the expected reads are over the run maximum: a run stops part-way at "
            f"{_usd(plan['run_max'])}, earliest windows first; the rest stay planned "
            "(raise X_ARCHIVE_FILL_MAX_USD deliberately for the whole fill)"
        )
    if plan["refusal"]:
        out.append(f"A run would refuse: {plan['refusal']}.")
    if not plan["windows"]:
        out.append("Nothing to fill in this range.")
    return out


def result_lines(res: dict) -> list[str]:
    b = res["billed"]
    cap = settings.X_ARCHIVE_FILL_MAX_USD
    out = [
        (
            f"\nx-archive fill run {res['run']}: {len(res['done'])} windows done, "
            f"{len(res['stopped'])} stopped, {res['left']} not reached"
        ),
        (
            f"  posts stored {res['kept'] + res['promoted']:,} ({res['promoted']} of them "
            f"bought earlier and set aside), set aside {res['extra']:,}, already held "
            f"{res['reread']}"
        ),
        (
            f"  this run: {res['read']:,} posts read ≈ {_usd(res['read'])} in "
            f"{res['requests']:,} requests; all fill runs: {b['posts']:,} ≈ ${b['usd']:.2f} "
            f"of X_ARCHIVE_FILL_MAX_USD ${cap:.2f}"
        ),
    ]
    short = [
        r for r in res["done"] if r["got"] < min(r["target"], r["x_lo"]) and r["mode"] == "sample"
    ] + [r for r in res["done"] if r["mode"] == "take_all" and r["got"] < r["x_lo"]]
    if short:
        out.append(
            f"  {len(short)} done windows hold fewer posts than X counts inside them "
            "(deleted since, or search/all leaving posts out): x-counts --report lists them"
        )
    for r in res["stopped"][:20]:
        out.append(f"  stopped @{r['handle']} {r['window_start'][:16]}: {r['stop']}")
    if len(res["stopped"]) > 20:
        out.append(f"  ... and {len(res['stopped']) - 20} more stopped windows")
    if res["stop"]:
        out.append(f"  {res['stop']}")
    out.append(
        "Next: x-counts --report (free), backup, then analyze and the label steps "
        "(AGENTS.md, Ask first)."
    )
    return out
