"""
X's own post counts (GET /2/tweets/counts/all) per query, cached, and the
free report that reads them against the `collect` windows.

Why: a collect window that comes back empty next to a dense truncated one
may be a silent account or a failed fetch, and nothing revisits it (the
cursor is the newest post). X's per-hour count of from:<handle> tells the
two apart, at $0.010 a request.

Design:
  - One JSON file per (query, granularity) in settings.X_COUNTS_DIR
    (CountsCache): the buckets (start, end, tweet_count), the covered
    intervals, fetched_at and a log of the fetches. A requested range is
    aligned to the granularity (both ends floored), so no partial bucket is
    stored.
  - plan_query / plan_run work out the requests from the cache alone (no
    call, no credentials): only the part of the range not yet covered,
    ceil(days / 31) requests per uncovered stretch.
  - fetch_query follows next_token and rewrites the file (atomic replace)
    after every page, so a killed run resumes with what it got. A query
    stops at its page cap (its plan + X_COUNTS_PAGE_SLACK), a run at its
    request cap (RequestCap). A stretch counts as covered only when X's
    buckets reach across it or X answered every page the 31-day paging
    implies; otherwise only what the buckets span, so an unexpected short
    answer never stores months of false zeros.
  - The cache key is the query with whitespace collapsed and from:
    operands lowercased (query_key): X matches usernames
    case-insensitively, so one count is never bought twice by spelling.
  - 403 raises AccessDenied, 401 CredentialsRejected, a usage-capped 429
    UsageCapped, all at once and with no further request; any other 429
    waits per x-rate-limit-reset, at most X_429_MAX_RETRIES times
    (call_with_backoff).
  - report_lines: per collect window, X's count against what the run got
    and what the cache holds, with a verdict; per account and month, X's
    count against the cache. Counts may leave out native retweets
    (unverified), so a window can hold more than X counts.
"""

import bisect
import hashlib
import itertools
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import tweepy

from config import settings
from src.collectors import x_collector as xc

logger = logging.getLogger(__name__)

GRANULARITIES = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
}
COUNTS_ENDPOINT = "GET /2/tweets/counts/all"

# Module level so tests can stand in for the clock.
_sleep = time.sleep
_clock = time.monotonic


class XRefused(RuntimeError):
    """X refused a request in a way no retry fixes. Raised at once; the run
    makes no further request. `hint` is appended to the stop message."""

    hint = ""

    def __init__(self, endpoint: str, status: int | None, detail: str):
        self.endpoint, self.status, self.detail = endpoint, status, detail
        super().__init__(f"{endpoint} answered {status}: {detail}")


class AccessDenied(XRefused):
    """403 (e.g. client-not-enrolled): this app may not use the endpoint. A
    result about the endpoint; the probe saves it as one."""


class CredentialsRejected(XRefused):
    """401: the bearer token is wrong, revoked or regenerated. Says nothing
    about the endpoint, so nothing is saved as a result."""

    hint = "; check X_BEARER_TOKEN in .env, then rerun"


class UsageCapped(XRefused):
    """429 with X's usage-capped problem type: the account's usage cap is
    reached, which waiting does not clear."""

    hint = "; the account's usage cap is reached (see the Developer Console)"


class RunCapReached(RuntimeError):
    """The run has made as many requests as its cap allows."""


def get_client() -> tweepy.Client:
    """A client without tweepy's unbounded 429 wait (call_with_backoff waits)."""
    return xc.get_client(wait_on_rate_limit=False)


# ── Time and intervals (naive UTC) ──────────────────────────────────


def parse_ts(ts: str) -> datetime:
    """An RFC 3339 time as naive UTC (3.11+ reads a trailing 'Z')."""
    dt = datetime.fromisoformat(ts)
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def z(dt: datetime) -> str:
    """Naive UTC as RFC 3339, whole seconds, 'Z'."""
    return dt.replace(microsecond=0).isoformat() + "Z"


def floor_to(dt: datetime, granularity: str) -> datetime:
    """dt rounded down to a bucket boundary."""
    dt = dt.replace(second=0, microsecond=0)
    if granularity in ("hour", "day"):
        dt = dt.replace(minute=0)
    if granularity == "day":
        dt = dt.replace(hour=0)
    return dt


def merge_intervals(intervals) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def uncovered(start: datetime, end: datetime, covered) -> list[tuple[datetime, datetime]]:
    """The parts of [start, end) no interval in `covered` holds."""
    gaps, t = [], start
    for a, b in merge_intervals(covered):
        if b <= t:
            continue
        if a >= end:
            break
        if a > t:
            gaps.append((t, a))
        t = max(t, b)
        if t >= end:
            break
    if t < end:
        gaps.append((t, end))
    return gaps


def _covered_seconds(start: datetime, end: datetime, covered) -> float:
    gaps = uncovered(start, end, covered)
    return (end - start).total_seconds() - sum((b - a).total_seconds() for a, b in gaps)


# ── The cache ───────────────────────────────────────────────────────


def handle_query(handle: str) -> str:
    return f"from:{handle}"


_FROM_OPERAND = re.compile(r"(?<![\w:])from:(\w+)", re.IGNORECASE)


def query_key(query: str) -> str:
    """The query as the cache knows it: whitespace collapsed and every from:
    operand lowercased, since X matches usernames case-insensitively. So
    'from:whitehouse  is:retweet' and 'from:WhiteHouse is:retweet' share a
    file and are bought once."""
    q = " ".join(query.split())
    return _FROM_OPERAND.sub(lambda m: f"from:{m.group(1).lower()}", q)


def query_slug(query: str) -> str:
    """A file-safe name for a query's key; the hash keeps two queries that
    differ only in punctuation apart."""
    key = query_key(query)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")[:60] or "q"
    return f"{safe}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def cache_path(query: str, granularity: str) -> Path:
    return settings.X_COUNTS_DIR / f"{query_slug(query)}.{granularity}.json"


def _save_json(path: Path, obj: dict) -> None:
    """Temp file, fsync, rename: the file holds paid requests, so a run
    killed mid-write leaves the old one whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _bucket_count(bucket: dict) -> int:
    """X's docs name the field tweet_count in examples, post_count in the
    schema; take either."""
    n = bucket.get("tweet_count", bucket.get("post_count"))
    if n is None:
        raise ValueError(f"count bucket without a count: {bucket}")
    return int(n)


class CountsCache:
    """One query's cached counts at one granularity (see the module doc)."""

    def __init__(self, query: str, granularity: str):
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {', '.join(GRANULARITIES)}")
        self.query, self.granularity = query, granularity
        self.step = GRANULARITIES[granularity]
        self.path = cache_path(query, granularity)
        self.buckets: dict[datetime, tuple[datetime, int]] = {}
        self.covered: list[tuple[datetime, datetime]] = []
        self.fetched_at: str | None = None
        self.fetches: list[dict] = []
        if self.path.exists():
            data = json.loads(self.path.read_text())
            held = (query_key(data.get("query") or ""), data.get("granularity"))
            if held != (query_key(query), granularity):
                raise ValueError(
                    f"{self.path} holds {data.get('query')!r} at {data.get('granularity')}"
                )
            for b in data.get("buckets", []):
                self.buckets[parse_ts(b["start"])] = (parse_ts(b["end"]), int(b["tweet_count"]))
            self.covered = merge_intervals(
                (parse_ts(a), parse_ts(b)) for a, b in data.get("covered", [])
            )
            self.fetched_at = data.get("fetched_at")
            self.fetches = data.get("fetches", [])

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def merge(self, buckets: list[dict], span: tuple[datetime, datetime] | None) -> None:
        """Buckets replace any cached bucket with the same start; `span` (what
        X has answered for) joins the covered intervals."""
        for b in buckets:
            self.buckets[parse_ts(b["start"])] = (parse_ts(b["end"]), _bucket_count(b))
        if span is not None:
            self.covered = merge_intervals([*self.covered, span])

    def save(self) -> None:
        _save_json(
            self.path,
            {
                "query": self.query,
                "granularity": self.granularity,
                "covered": [[z(a), z(b)] for a, b in self.covered],
                "fetched_at": self.fetched_at,
                "fetches": self.fetches,
                "buckets": [
                    {"start": z(s), "end": z(e), "tweet_count": n}
                    for s, (e, n) in sorted(self.buckets.items())
                ],
            },
        )

    def _count(self, t: datetime) -> int:
        """A covered bucket X sent no row for counts 0."""
        return self.buckets.get(t, (None, 0))[1]

    def span_counts(self, start: datetime, end: datetime) -> tuple[int, int] | None:
        """(lo, hi) for X's count in [start, end): lo sums the buckets wholly
        inside, hi adds the buckets its edges cut. None unless every bucket
        it touches is covered."""
        first = floor_to(start, self.granularity)
        last = floor_to(end, self.granularity)
        if last < end:
            last += self.step
        if uncovered(first, last, self.covered):
            return None
        lo = hi = 0
        t = first
        while t < end:
            n = self._count(t)
            hi += n
            if t >= start and t + self.step <= end:
                lo += n
            t += self.step
        return lo, hi

    def total(self, start: datetime, end: datetime) -> tuple[int, float]:
        """(count, covered share) over [start, end), both on bucket
        boundaries: the covered buckets' sum and the share of the span
        covered."""
        if end <= start:
            return 0, 0.0
        n = sum(
            c
            for s, (e, c) in self.buckets.items()
            if start <= s and e <= end and not uncovered(s, e, self.covered)
        )
        share = _covered_seconds(start, end, self.covered) / (end - start).total_seconds()
        return n, share

    def by_day(self, start: datetime, end: datetime) -> dict[str, int] | None:
        """Day totals over [start, end) (whole days), None unless covered."""
        if uncovered(start, end, self.covered):
            return None
        out: dict[str, int] = {}
        for s, (_, c) in self.buckets.items():
            if start <= s < end:
                out[f"{s:%Y-%m-%d}"] = out.get(f"{s:%Y-%m-%d}", 0) + c
        t = start
        while t < end:
            out.setdefault(f"{t:%Y-%m-%d}", 0)
            t += timedelta(days=1)
        return dict(sorted(out.items()))


# ── Planning (no call) ──────────────────────────────────────────────


def build_queries(
    accounts: dict[str, list[str]] | None,
    terms: list[str] | None = None,
    raw: str | None = None,
) -> list[str]:
    """from:<handle> per account, then the search terms, then one raw query,
    deduped by query_key in that order (the first spelling stays)."""
    qs = [handle_query(h) for hs in (accounts or {}).values() for h in hs]
    qs += list(terms or [])
    if raw and raw.strip():
        qs.append(" ".join(raw.split()))
    out: dict[str, str] = {}
    for q in qs:
        out.setdefault(query_key(q), q)
    return list(out.values())


def pages_for(start: datetime, end: datetime) -> int:
    """Requests one stretch takes: a response covers up to 31 days."""
    secs = int((end - start).total_seconds())
    per = settings.X_COUNTS_DAYS_PER_PAGE * 86400
    return max(1, -(-secs // per))


def plan_query(
    query: str,
    start: datetime,
    end: datetime,
    granularity: str,
    force: bool = False,
    slack: int | None = None,
) -> dict:
    """
    What fetching one query over [start, end) would request, from the cache
    alone. Both ends are floored to the granularity. Without force, only the
    stretches the cache does not cover are planned. page_cap is the planned
    requests plus `slack` (default X_COUNTS_PAGE_SLACK), the most the query
    may make.
    """
    slack = settings.X_COUNTS_PAGE_SLACK if slack is None else slack
    s, e = floor_to(start, granularity), floor_to(end, granularity)
    cache = CountsCache(query, granularity)
    if e <= s:
        todo, reason = [], "empty range"
    else:
        todo = [(s, e)] if force else uncovered(s, e, cache.covered)
        reason = None if todo else f"cached ({granularity})"
    pages = sum(pages_for(a, b) for a, b in todo)
    cap = pages + slack if pages else 0
    cost = settings.X_COUNTS_ALL_COST_USD
    return {
        "query": query,
        "granularity": granularity,
        "start": s,
        "end": e,
        "todo": todo,
        "pages": pages,
        "page_cap": cap,
        "cost_usd": pages * cost,
        "max_cost_usd": cap * cost,
        "skip_reason": reason,
    }


def plan_run(
    queries: list[str],
    start: datetime,
    end: datetime,
    granularity: str,
    force: bool = False,
) -> dict:
    """Every query's plan, the planned requests and the run's maximum: the
    page caps' sum, never over X_COUNTS_MAX_REQUESTS (where the run stops)."""
    plans = [plan_query(q, start, end, granularity, force=force) for q in queries]
    pages = sum(p["pages"] for p in plans)
    max_requests = min(sum(p["page_cap"] for p in plans), settings.X_COUNTS_MAX_REQUESTS)
    cost = settings.X_COUNTS_ALL_COST_USD
    return {
        "queries": plans,
        "pages": pages,
        "cost_usd": pages * cost,
        "max_requests": max_requests,
        "max_cost_usd": max_requests * cost,
    }


def _span_text(todo, s: datetime, e: datetime) -> str:
    if not todo:
        return "-"
    if todo == [(s, e)]:
        return "all"
    days = sum((b - a).total_seconds() for a, b in todo) / 86400
    return f"{len(todo)} gap(s), {days:.1f} d"


def plan_lines(plan: dict) -> list[str]:
    """The --estimate table and its total."""
    out = [
        f"  {'query':<36}{'gran':>7}  {'range (UTC)':<33}{'to fetch':>18}{'requests':>10}{'$':>7}"
    ]
    for p in plan["queries"]:
        rng = f"{p['start']:%Y-%m-%d %H:%M}..{p['end']:%Y-%m-%d %H:%M}"
        what = p["skip_reason"] or _span_text(p["todo"], p["start"], p["end"])
        out.append(
            f"  {p['query'][:35]:<36}{p['granularity']:>7}  {rng:<33}{what:>18}"
            f"{p['pages']:>10}{p['cost_usd']:>7.2f}"
        )
    out.append(
        f"\nPlanned: {plan['pages']} requests ≈ ${plan['cost_usd']:.2f} at "
        f"${settings.X_COUNTS_ALL_COST_USD:.3f} each. Maximum {plan['max_requests']} "
        f"(each query may make {settings.X_COUNTS_PAGE_SLACK} more; run cap "
        f"X_COUNTS_MAX_REQUESTS = {settings.X_COUNTS_MAX_REQUESTS}) ≈ "
        f"${plan['max_cost_usd']:.2f}  (budget X_RUN_BUDGET_USD = "
        f"${settings.X_RUN_BUDGET_USD:.2f})"
    )
    return out


# ── Fetching ────────────────────────────────────────────────────────


class RequestCap:
    """Counts a run's answered requests against its cap and spaces them
    `delay_s` apart. A request that raised is not counted: X bills the
    requests it answers."""

    def __init__(self, limit: int, delay_s: float):
        self.limit, self.delay_s = limit, delay_s
        self.used = 0
        self._last: float | None = None

    def before(self) -> None:
        if self.used >= self.limit:
            raise RunCapReached(f"the run's cap of {self.limit} requests is reached")
        if self._last is not None:
            wait = self.delay_s - (_clock() - self._last)
            if wait > 0:
                _sleep(wait)

    def after(self) -> None:
        self.used += 1
        self._last = _clock()


def _status(e: tweepy.errors.HTTPException) -> int | None:
    return getattr(getattr(e, "response", None), "status_code", None)


def _problem(e: tweepy.errors.HTTPException) -> dict:
    """X's problem body (title, type, detail, ...), or {}."""
    try:
        body = e.response.json()
    except (AttributeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _detail(e: tweepy.errors.HTTPException) -> str:
    """X's problem title / reason / detail (client-not-enrolled comes as a
    reason), else tweepy's message."""
    body = _problem(e)
    parts = [str(body[k]) for k in ("title", "reason", "detail") if body.get(k)]
    return "; ".join(parts) if parts else str(e)


def _usage_capped(e: tweepy.errors.HTTPException) -> bool:
    """A 429 for the usage cap (problem type .../usage-capped, title
    UsageCapExceeded), not the rate limit."""
    body = _problem(e)
    title = str(body.get("title") or "").lower().replace(" ", "")
    return "usage-capped" in str(body.get("type") or "") or "usagecap" in title


def _backoff_s(response) -> int:
    """Seconds to wait after a 429: until x-rate-limit-reset (or
    retry-after), clamped to [1, X_429_MAX_BACKOFF_S]; the default without
    a usable header."""
    headers = getattr(response, "headers", None) or {}
    wait = None
    try:
        if headers.get("x-rate-limit-reset"):
            wait = int(headers["x-rate-limit-reset"]) - int(time.time()) + 1
        elif headers.get("retry-after"):
            wait = int(headers["retry-after"])
    except (TypeError, ValueError):
        wait = None
    if wait is None:
        wait = settings.X_429_DEFAULT_BACKOFF_S
    return max(1, min(wait, settings.X_429_MAX_BACKOFF_S))


def call_with_backoff(fn, endpoint: str, **params):
    """fn(**params). Raises AccessDenied on 403, CredentialsRejected on 401
    and UsageCapped on a usage-capped 429, at once; waits out any other 429
    at most X_429_MAX_RETRIES times before letting it raise."""
    for attempt in itertools.count():
        try:
            return fn(**params)
        except tweepy.errors.Unauthorized as e:
            raise CredentialsRejected(endpoint, _status(e), _detail(e)) from None
        except tweepy.errors.Forbidden as e:
            raise AccessDenied(endpoint, _status(e), _detail(e)) from None
        except tweepy.errors.TooManyRequests as e:
            if _usage_capped(e):
                raise UsageCapped(endpoint, _status(e), _detail(e)) from None
            if attempt >= settings.X_429_MAX_RETRIES:
                raise
            wait = _backoff_s(getattr(e, "response", None))
            logger.warning(
                "%s: 429 rate limited; waiting %d s (retry %d of %d)",
                endpoint,
                wait,
                attempt + 1,
                settings.X_429_MAX_RETRIES,
            )
            _sleep(wait)
    raise AssertionError("unreachable")


def _page_span(data: list[dict], a: datetime, b: datetime):
    """What one page of a paged stretch answered for: its buckets' extent."""
    if not data:
        return None
    lo = max(a, min(parse_ts(x["start"]) for x in data))
    hi = min(b, max(parse_ts(x["end"]) for x in data))
    return (lo, hi) if lo < hi else None


def fetch_query(client, plan: dict, cap: RequestCap) -> dict:
    """
    Fetch a planned query's uncovered stretches, following next_token. The
    cache file is rewritten after every page: the page's buckets, and the
    span X has answered for, so a killed run resumes with only what is left.
    While more pages follow that span is the page's extent. Once none do it
    is the whole stretch, but only if the stretch's buckets reach across it
    or X answered every page pages_for() expects; otherwise X answered short
    of its documented 31-day paging, and only the buckets' extent is
    recorded (a WARNING; the query comes back incomplete, the next run plans
    the rest). Stops the query at plan["page_cap"] requests; the run's cap
    raises RunCapReached. Returns {query, requests, complete, stopped,
    total}.
    """
    cache = CountsCache(plan["query"], plan["granularity"])
    used, stopped, short = 0, None, []
    for a, b in plan["todo"]:
        token, entry, pages, spans = None, None, 0, []
        while True:
            if used >= plan["page_cap"]:
                stopped = (
                    f"page cap {plan['page_cap']} reached with pages left (X paged more "
                    "finely than planned); the rest is fetched by the next run"
                )
                logger.warning("%s: %s", plan["query"], stopped)
                break
            cap.before()
            resp = call_with_backoff(
                client.get_all_tweets_count,
                COUNTS_ENDPOINT,
                query=plan["query"],
                start_time=z(a),
                end_time=z(b),
                granularity=plan["granularity"],
                next_token=token,
            )
            cap.after()
            used += 1
            pages += 1
            data = list(resp.data or [])
            token = (resp.meta or {}).get("next_token")
            span = _page_span(data, a, b)
            if span:
                spans.append(span)
            gap = None
            if token:
                cache.merge(data, span)
            elif not uncovered(a, b, spans) or pages >= pages_for(a, b):
                cache.merge(data, (a, b))
            else:
                cache.merge(data, span)
                got = merge_intervals(spans)
                reach = f"{z(got[0][0])}..{z(got[-1][1])}" if got else "nothing"
                gap = (
                    f"X answered {pages} page(s) with no next_token for {z(a)}..{z(b)}, "
                    f"where its {settings.X_COUNTS_DAYS_PER_PAGE}-day paging implies "
                    f"{pages_for(a, b)}, and its buckets cover {reach}: only that is "
                    "recorded as counted; the next run plans the rest"
                )
                logger.warning("%s: %s", plan["query"], gap)
                short.append(gap)
            now = z(datetime.now(UTC).replace(tzinfo=None))
            if entry is None:
                entry = {"start": z(a), "end": z(b), "pages": 0}
                cache.fetches.append(entry)
            done = not token and gap is None
            entry.update(pages=entry["pages"] + 1, complete=done, fetched_at=now)
            cache.fetched_at = now
            cache.save()
            if not token:
                break
        if stopped:
            break
    total, _ = cache.total(plan["start"], plan["end"])
    notes = [*short, *([stopped] if stopped else [])]
    return {
        "query": plan["query"],
        "requests": used,
        "complete": not notes,
        "stopped": "; ".join(notes) or None,
        "total": total,
    }


# ── Report (free, no call) ──────────────────────────────────────────

VERDICTS = ["silent", "complete", "sampled ok", "edge", "short", "FETCH SHORTFALL", "not counted"]


def is_retweet(rec: dict) -> bool:
    if any((r or {}).get("type") == "retweeted" for r in rec.get("ref") or []):
        return True
    return str(rec.get("text") or "").startswith("RT @")


def verdict(lo: int, hi: int, cap: int, got: int) -> str:
    """
    One collect window against X's count, given as (lo, hi) because hour
    buckets cut the window's edges:
      silent           X counts nothing and the run got nothing;
      complete         the run got every post X counts (or more: counts
                       may leave out retweets);
      sampled ok       the run got its cap, X counts more;
      edge             what the run got sits inside the edge-bucket range;
      FETCH SHORTFALL  the run got less than min(cap, count) where X counts
                       at least the cap, or got nothing where X counts posts;
      short            got less than X counts, under the cap and not zero
                       (deleted since, or posts the timeline does not serve).
    """
    if hi == 0:
        return "silent" if got == 0 else "complete"
    want_lo, want_hi = min(cap, lo), min(cap, hi)
    if got >= want_hi:
        return "complete" if got >= hi else "sampled ok"
    if got < want_lo:
        return "FETCH SHORTFALL" if lo >= cap or got == 0 else "short"
    return "edge"


def load_manifests(paths=None) -> list[dict]:
    """Rows of the given manifests, else of every windows_*.jsonl in
    settings.X_SAMPLING_DIR."""
    paths = (
        [Path(p) for p in paths]
        if paths
        else sorted(settings.X_SAMPLING_DIR.glob("windows_*.jsonl"))
    )
    rows: list[dict] = []
    for p in paths:
        rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    return rows


def _cached_posts(handle: str) -> list[tuple[datetime, bool]]:
    """(created_at, is retweet) of the account's cached posts, by time."""
    recs = xc._load_jsonl(xc._account_cache_path(handle))
    return sorted((parse_ts(r["created_at"]), is_retweet(r)) for r in recs if r.get("created_at"))


def _held(posts: list[tuple[datetime, bool]], start: datetime, end: datetime) -> tuple[int, int]:
    """(posts, retweets) cached in [start, end)."""
    times = [t for t, _ in posts]
    i, j = bisect.bisect_left(times, start), bisect.bisect_left(times, end)
    return j - i, sum(1 for _, rt in posts[i:j] if rt)


def window_rows(manifest_rows: list[dict], handles: set[str], granularity: str) -> list[dict]:
    """Every manifest window of these handles with X's count (lo, hi), what
    the run got, what the cache holds now ([start, end)) and a verdict."""
    caches: dict[str, CountsCache] = {}
    posts: dict[str, list] = {}
    out = []
    rows = [r for r in manifest_rows if r.get("handle") in handles]
    for r in sorted(rows, key=lambda r: (r["handle"], r["window_start"], r.get("run_end", ""))):
        h = r["handle"]
        if h not in caches:
            caches[h] = CountsCache(handle_query(h), granularity)
            posts[h] = _cached_posts(h)
        s, e = parse_ts(r["window_start"]), parse_ts(r["window_end"])
        span = caches[h].span_counts(s, e)
        held, held_rt = _held(posts[h], s, e)
        row = {
            "handle": h,
            "src": r.get("source") or "collect",
            "mode": r.get("mode"),
            "start": s,
            "end": e,
            "cap": r["cap"],
            "got": r["got"],
            "held": held,
            "held_rt": held_rt,
            "lo": span[0] if span else None,
            "hi": span[1] if span else None,
        }
        row["verdict"] = "not counted" if span is None else verdict(*span, r["cap"], r["got"])
        row["over"] = span is not None and r["got"] > span[1]
        out.append(row)
    return out


def _months(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    out, t = [], start
    while t < end:
        nxt = (t.replace(day=1) + timedelta(days=32)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        out.append((t, min(nxt, end)))
        t = nxt
    return out


def monthly_rows(handle: str, start: datetime, end: datetime, granularity: str) -> list[dict]:
    """Per calendar month of [start, end): X's count of from:<handle> (and of
    from:<handle> is:retweet when that is cached), with the covered share,
    against the posts and retweets the cache holds."""
    s, e = floor_to(start, granularity), floor_to(end, granularity)
    cache = CountsCache(handle_query(handle), granularity)
    rt_cache = CountsCache(f"{handle_query(handle)} is:retweet", granularity)
    posts = _cached_posts(handle)
    out = []
    for m0, m1 in _months(s, e):
        x, share = cache.total(m0, m1)
        x_rt, rt_share = rt_cache.total(m0, m1) if rt_cache.exists else (None, 0.0)
        held, held_rt = _held(posts, m0, m1)
        out.append(
            {
                "month": f"{m0:%Y-%m}",
                "x": x if share else None,
                "share": share,
                "x_rt": x_rt if rt_share else None,
                "rt_share": rt_share,
                "held": held,
                "held_rt": held_rt,
            }
        )
    return out


def _count_text(lo, hi) -> str:
    if lo is None:
        return "-"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _x_text(n, share) -> str:
    if n is None:
        return "-"
    return f"{n}" if share >= 0.999 else f"~{n} ({share:.0%})"


RETWEET_NOTE = (
    "Note: X's counts may leave out native retweets (unverified; the search index may drop "
    "them from from: queries). So 'got' or 'held' above X's count ('+' column) is possible, "
    "and is itself evidence about retweets: compare 'held' with 'held RT', and run "
    "`x-counts --query 'from:<handle> is:retweet'` for the 'X RT' column."
)


def report_lines(
    accounts: dict[str, list[str]],
    manifest_paths=None,
    start: datetime | None = None,
    end: datetime | None = None,
    granularity: str | None = None,
) -> list[str]:
    """The `x-counts --report` text: windows, verdicts by account, months.
    A window's row says its source: `collect` (the timeline) or `archive`
    (an `x-archive fill` row, whose got is what the window held after the
    fill); verdicts are counted per account and source, so a window the
    fill revisited is not counted twice in one line."""
    g = granularity or settings.X_COUNTS_GRANULARITY
    start = start or settings.COLLECTION_START
    end = end or settings.COLLECTION_END
    handles = [h for hs in accounts.values() for h in hs]
    rows = window_rows(load_manifests(manifest_paths), set(handles), g)
    out = [f"X counts (from:<handle>, {g} buckets) against the collect and fill windows"]
    if not rows:
        out.append("  no manifest windows for these accounts")
    for h, group in itertools.groupby(rows, key=lambda r: r["handle"]):
        group = list(group)
        by_src = Counter(r["src"] for r in group)
        out.append(
            f"\n@{h}  {group[0]['mode']}, "
            + ", ".join(f"{n} {src} windows" for src, n in sorted(by_src.items()))
        )
        out.append(
            f"  {'window (UTC)':<31}{'src':<9}{'cap':>5}{'got':>5}{'held':>6}{'X count':>9}  "
            f"{'+':<2}verdict"
        )
        for r in group:
            win = f"{r['start']:%Y-%m-%d %H:%M}..{r['end']:%m-%d %H:%M}"
            out.append(
                f"  {win:<31}{r['src']:<9}{r['cap']:>5}{r['got']:>5}{r['held']:>6}"
                f"{_count_text(r['lo'], r['hi']):>9}  {'+' if r['over'] else '':<2}{r['verdict']}"
            )
    if rows:
        out.append("\nVerdicts by account and source (+ = got more than X counts)")
        short = {
            "sampled ok": "sampled",
            "FETCH SHORTFALL": "SHORTFALL",
            "not counted": "uncounted",
        }
        names = [short.get(v, v) for v in VERDICTS]
        out.append(
            f"  {'account':<24}{'windows':>8}" + "".join(f"{n:>11}" for n in names) + f"{'+':>5}"
        )
        for h, group in itertools.groupby(rows, key=lambda r: r["handle"]):
            group = list(group)
            for src in sorted({r["src"] for r in group}, key=lambda s: (s != "collect", s)):
                part = [r for r in group if r["src"] == src]
                c = Counter(r["verdict"] for r in part)
                name = f"@{h}" if src == "collect" else f"@{h} {src}"
                out.append(
                    f"  {name:<24}{len(part):>8}"
                    + "".join(f"{c.get(v, 0):>11}" for v in VERDICTS)
                    + f"{sum(r['over'] for r in part):>5}"
                )
    out.append(f"\nX count against the cache by month ({start:%Y-%m-%d}..{end:%Y-%m-%d})")
    for h in handles:
        cache = CountsCache(handle_query(h), g)
        if not cache.exists:
            out.append(f"\n@{h}: no {g} counts cached (x-counts --only {h})")
            continue
        out.append(f"\n@{h}  (counts fetched {cache.fetched_at})")
        out.append(
            f"  {'month':<9}{'X count':>14}{'X RT':>14}{'held':>7}{'held RT':>9}"
            f"{'held orig':>11}{'held/X':>8}"
        )
        for m in monthly_rows(h, start, end, g):
            ratio = f"{m['held'] / m['x']:.2f}" if m["x"] else "-"
            out.append(
                f"  {m['month']:<9}{_x_text(m['x'], m['share']):>14}"
                f"{_x_text(m['x_rt'], m['rt_share']):>14}{m['held']:>7}{m['held_rt']:>9}"
                f"{m['held'] - m['held_rt']:>11}{ratio:>8}"
            )
    out.append("\n" + RETWEET_NOTE)
    return out
