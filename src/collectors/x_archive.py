"""
Full-archive search probe (GET /2/tweets/search/all), run before any
archive fill is bought.

It asks: does search/all work for this app at all (one docs page says the
archive endpoints are Enterprise-only, and a forum report shows a 403
client-not-enrolled)? Does a from: query return native retweets? Does it
reach February as well as September? And does counts/all count retweets?

  - choose_windows picks, from the cache alone, one whole UTC day in each of
    settings.X_ARCHIVE_PROBE_STRETCHES that the cache holds whole. The day
    is "closed": it ends inside its stretch and before the account's first
    sampled or truncated `collect` window in the x_sampling manifests, so
    every post of it is cached. It holds 6-10 cached posts, retweets and
    originals both. Deterministic: the most even mix, then the fewest
    posts, then the earliest day.
  - run_probe makes one search/all request per day (max_results 10, the
    collector's tweet fields), then counts/all at day granularity over
    X_ARCHIVE_PROBE_COUNTS_RANGE for from:<handle> and for
    from:<handle> is:retweet, through x_counts, so those counts land in the
    shared counts cache (and a range it already covers costs nothing).
  - State goes to X_ARCHIVE_DIR/probe_<handle>.json after every step and
    the returned posts to probe_<handle>_posts.jsonl, so a killed probe
    resumes with the steps it has not done. A finished probe is reused
    unless --force, which sets the old files aside (never deletes them).
  - A 403 is a result ("no archive access"), saved like any other; no
    further request is made. A 401 (bad token) or a usage-capped 429 also
    stops the run at once, but says nothing about the endpoint: the probe
    stays unfinished and a rerun resumes it. So does a counts step that
    did not cover its range.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from config import settings
from config.accounts import X_ACCOUNTS
from src.collectors import x_collector as xc
from src.collectors import x_counts as xcn

logger = logging.getLogger(__name__)

SEARCH_ENDPOINT = "GET /2/tweets/search/all"


class NoProbeWindow(RuntimeError):
    """The cache holds no closed day that fits a stretch."""


def _naive(ts: str) -> datetime:
    return xcn.parse_ts(ts)


def _now() -> str:
    return xcn.z(datetime.now(UTC).replace(tzinfo=None))


def state_path(handle: str) -> Path:
    return settings.X_ARCHIVE_DIR / f"probe_{handle}.json"


def posts_path(handle: str) -> Path:
    return settings.X_ARCHIVE_DIR / f"probe_{handle}_posts.jsonl"


def load_state(handle: str) -> dict | None:
    p = state_path(handle)
    return json.loads(p.read_text()) if p.exists() else None


def _tier(handle: str) -> str:
    return next((t for t, hs in X_ACCOUNTS.items() if handle in hs), "unknown")


# ── Window choice (cache only) ──────────────────────────────────────


def complete_until(handle: str, start: datetime, end: datetime) -> datetime:
    """The end of the stretch the cache holds whole: `end`, or the start of
    the first manifest window overlapping [start, end) that may have left
    posts unread (sampled, or truncated)."""
    until = end
    for r in xcn.load_manifests():
        if r.get("handle") != handle:
            continue
        if r.get("mode") == "take_all" and not r.get("truncated"):
            continue  # read whole
        ws, we = _naive(r["window_start"]), _naive(r["window_end"])
        if we > start and ws < until:
            until = max(start, ws)
    return until


def choose_window(
    posts: list[dict], label: str, start: datetime, end: datetime, until: datetime
) -> dict | None:
    """The closed UTC day in [start, min(end, until)) with
    X_ARCHIVE_PROBE_MIN_POSTS..MAX_POSTS cached posts, retweets and
    originals both: the most even mix, then the fewest posts, then the
    earliest. None if no day fits."""
    stop = min(end, until)
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if day < start:
        day += timedelta(days=1)
    best = None
    while day + timedelta(days=1) <= stop:
        nxt = day + timedelta(days=1)
        inside = [p for p in posts if day <= _naive(p["created_at"]) < nxt]
        n, rt = len(inside), sum(1 for p in inside if xcn.is_retweet(p))
        lo, hi = settings.X_ARCHIVE_PROBE_MIN_POSTS, settings.X_ARCHIVE_PROBE_MAX_POSTS
        if lo <= n <= hi and 0 < rt < n:
            key = (-min(rt, n - rt), n, day)
            if best is None or key < best[0]:
                best = (key, day, inside, rt)
        day = nxt
    if best is None:
        return None
    _, day, inside, rt = best
    return {
        "label": label,
        "start": xcn.z(day),
        "end": xcn.z(day + timedelta(days=1)),
        "cached_ids": sorted(str(p["id"]) for p in inside),
        "cached_rt_ids": sorted(str(p["id"]) for p in inside if xcn.is_retweet(p)),
        "cached": len(inside),
        "cached_rt": rt,
        "cached_orig": len(inside) - rt,
    }


def choose_windows(handle: str) -> list[dict]:
    """One closed day per stretch in settings.X_ARCHIVE_PROBE_STRETCHES."""
    posts = xc._load_jsonl(xc._account_cache_path(handle))
    out = []
    for label, s, e in settings.X_ARCHIVE_PROBE_STRETCHES:
        start, end = _naive(s), _naive(e)
        until = complete_until(handle, start, end)
        w = choose_window(posts, label, start, end, until)
        if w is None:
            raise NoProbeWindow(
                f"@{handle}: no whole UTC day in {label} ({start:%Y-%m-%d %H:%M}.."
                f"{min(end, until):%Y-%m-%d %H:%M}) holds "
                f"{settings.X_ARCHIVE_PROBE_MIN_POSTS}-{settings.X_ARCHIVE_PROBE_MAX_POSTS} "
                "cached posts mixing retweets and originals"
            )
        out.append(w)
    return out


# ── Plan (no call) ──────────────────────────────────────────────────


def counts_queries(handle: str) -> list[str]:
    q = xcn.handle_query(handle)
    return [q, f"{q} is:retweet"]


def _counts_range() -> tuple[datetime, datetime]:
    s, e = settings.X_ARCHIVE_PROBE_COUNTS_RANGE
    return _naive(s), _naive(e)


def _new_state(handle: str, windows: list[dict]) -> dict:
    return {
        "handle": handle,
        "created_at": _now(),
        "finished_at": None,
        "complete": False,
        "windows": windows,
        "searches": {},
        "counts": {},
        "denied": None,
        "pending": [],
        "verdict": None,
        "counts_reading": None,
    }


def plan_probe(handle: str | None = None, force: bool = False) -> dict:
    """
    What a probe run would do, from the cache and the saved state alone. A
    finished probe is reused (reuse=True, nothing to pay) unless force. An
    unfinished one resumes with its saved windows and the steps it has not
    done. Returns {handle, force, reuse, state, windows, searches (labels to
    run), counts (plans, some with nothing to request), requests, max_reads,
    max_cost_usd}.
    """
    handle = handle or settings.X_ARCHIVE_PROBE_HANDLE
    saved = None if force else load_state(handle)
    if saved and saved.get("complete"):
        return {
            "handle": handle,
            "force": False,
            "reuse": True,
            "state": saved,
            "windows": saved["windows"],
            "searches": [],
            "counts": [],
            "requests": 0,
            "max_reads": 0,
            "max_cost_usd": 0.0,
        }
    state = saved or _new_state(handle, choose_windows(handle))
    searches = [w["label"] for w in state["windows"] if w["label"] not in state["searches"]]
    s, e = _counts_range()
    counts = [
        p
        for q in counts_queries(handle)
        if q not in state["counts"]
        for p in [xcn.plan_query(q, s, e, "day", force=force, slack=0)]
    ]
    reads = len(searches) * settings.X_ARCHIVE_SEARCH_MAX_RESULTS
    pages = sum(p["pages"] for p in counts)
    return {
        "handle": handle,
        "force": force,
        "reuse": False,
        "state": state,
        "windows": state["windows"],
        "searches": searches,
        "counts": counts,
        "requests": len(searches) + pages,
        "max_reads": reads,
        "max_cost_usd": reads * settings.X_READ_COST_USD + pages * settings.X_COUNTS_ALL_COST_USD,
    }


# ── Run ─────────────────────────────────────────────────────────────


def _save_state(state: dict) -> None:
    xcn._save_json(state_path(state["handle"]), state)


def set_aside(handle: str) -> list[Path]:
    """--force: rename a previous probe's files with their creation stamp,
    never delete them (the posts were paid for)."""
    old = load_state(handle)
    stamp = (old or {}).get("created_at") or _now()
    stamp = stamp.replace(":", "").replace("-", "")
    moved = []
    for p in (state_path(handle), posts_path(handle)):
        if p.exists():
            dest = p.with_name(f"{p.stem}.{stamp}{p.suffix}")
            p.rename(dest)
            moved.append(dest)
    return moved


def _search_day(client, handle: str, window: dict, cap: xcn.RequestCap) -> dict:
    cap.before()
    resp = xcn.call_with_backoff(
        client.search_all_tweets,
        SEARCH_ENDPOINT,
        query=xcn.handle_query(handle),
        start_time=window["start"],
        end_time=window["end"],
        max_results=settings.X_ARCHIVE_SEARCH_MAX_RESULTS,
        tweet_fields=xc._TWEET_FIELDS,
    )
    cap.after()
    tier = _tier(handle)
    recs = [
        {**xc._timeline_record(t, handle, tier), "probe_window": window["label"]}
        for t in resp.data or []
    ]
    if recs:
        xc._append_jsonl(posts_path(handle), recs)  # before the state says done
    returned = [r["id"] for r in recs]
    cached, cached_rt = set(window["cached_ids"]), set(window["cached_rt_ids"])
    got = set(returned)
    return {
        "fetched_at": _now(),
        "returned_ids": returned,
        "returned": len(returned),
        "returned_rt": sum(1 for r in recs if xcn.is_retweet(r)),
        "more": bool((resp.meta or {}).get("next_token")),
        "missing_ids": sorted(cached - got),
        "extra_ids": sorted(got - cached),
        "hits_rt": len(got & cached_rt),
        "hits_orig": len(got & (cached - cached_rt)),
    }


def _held_by_day(handle: str, start: datetime, end: datetime) -> dict[str, list[int]]:
    """{date: [posts, retweets]} cached over [start, end)."""
    out: dict[str, list[int]] = {}
    t = start
    while t < end:
        out[f"{t:%Y-%m-%d}"] = [0, 0]
        t += timedelta(days=1)
    for r in xc._load_jsonl(xc._account_cache_path(handle)):
        ts = _naive(r["created_at"])
        if start <= ts < end:
            d = out[f"{ts:%Y-%m-%d}"]
            d[0] += 1
            d[1] += xcn.is_retweet(r)
    return out


def run_probe(client, plan: dict) -> dict:
    """
    Run a planned probe: the search/all days not yet done, then the counts
    not yet done, saving the state after each step. A 403 records `denied`
    and stops (no further request): a result, so the probe is finished. A
    counts step that did not cover its range (X paged more finely than
    planned, or answered short) is left out of the state and listed in
    `pending`, and the probe stays unfinished so a rerun resumes it. Any
    other error, a 401 or a usage-capped 429 included, propagates with the
    steps done so far saved. A forced plan first sets the previous probe's
    files aside. Returns the state, with its verdict.
    """
    state = plan["state"]
    handle = state["handle"]
    if plan["force"]:
        set_aside(handle)
    _save_state(state)
    cap = xcn.RequestCap(max(plan["requests"], 1), settings.X_ARCHIVE_SEARCH_DELAY_S)
    windows = {w["label"]: w for w in state["windows"]}
    pending: list[str] = []
    try:
        for label in plan["searches"]:
            state["searches"][label] = _search_day(client, handle, windows[label], cap)
            _save_state(state)
        for p in plan["counts"]:
            res = xcn.fetch_query(client, p, cap) if p["pages"] else None
            cache = xcn.CountsCache(p["query"], "day")
            days = cache.by_day(p["start"], p["end"])
            if (res and not res["complete"]) or days is None:
                why = (res or {}).get("stopped") or "range not covered"
                pending.append(f"{p['query']}: {why}")
                continue
            state["counts"][p["query"]] = {
                "requests": res["requests"] if res else 0,
                "fetched_at": cache.fetched_at,
                "days": days,
            }
            _save_state(state)
    except xcn.AccessDenied as e:
        state["denied"] = {"endpoint": e.endpoint, "status": e.status, "detail": e.detail}
    s, e = _counts_range()
    state["pending"] = pending
    state["held_by_day"] = _held_by_day(handle, s, e)
    state["verdict"] = verdict(state)
    state["counts_reading"] = counts_reading(state)
    state["complete"] = bool(state["denied"]) or not pending
    state["finished_at"] = _now() if state["complete"] else None
    _save_state(state)
    return state


# ── Verdicts ────────────────────────────────────────────────────────


def _share(a: int, b: int) -> float | None:
    return a / b if b else None


def verdict(state: dict) -> str:
    """
    One line from the search/all days, the oldest window first:
      no archive access              search/all answered 403 (a 401 is a
                                     bad token, never a verdict);
      archive misses older posts     the newest day gets back >= RECALL_OK of
                                     what we hold, the oldest < RECALL_LOW;
      archive works, retweets dropped  no cached retweet came back, and the
                                     originals did (>= RECALL_OK);
      archive works incl. retweets   >= RECALL_OK of all and of the
                                     retweets came back;
      inconclusive                   anything else (numbers attached).
    Recall counts the posts of the kinds that come back at all: originals
    only when retweets are dropped. A cached post can also be missing
    because it (or the retweet) was deleted since.
    """
    denied = state.get("denied")
    if denied and denied["endpoint"] == SEARCH_ENDPOINT:
        return "no archive access"
    ok, low = settings.X_ARCHIVE_PROBE_RECALL_OK, settings.X_ARCHIVE_PROBE_RECALL_LOW
    wins = [
        (w, state["searches"][w["label"]])
        for w in state["windows"]
        if w["label"] in state["searches"]
    ]
    if not wins or len(wins) < len(state["windows"]):
        return "inconclusive: not every probe day was searched"
    c_rt = sum(w["cached_rt"] for w, _ in wins)
    c_orig = sum(w["cached_orig"] for w, _ in wins)
    h_rt = sum(r["hits_rt"] for _, r in wins)
    h_orig = sum(r["hits_orig"] for _, r in wins)
    rt_dropped = c_rt > 0 and h_rt == 0 and sum(r["returned_rt"] for _, r in wins) == 0

    def recall(w, r):
        if rt_dropped:
            return _share(r["hits_orig"], w["cached_orig"])
        return _share(r["hits_rt"] + r["hits_orig"], w["cached"])

    old, new = recall(*wins[0]), recall(*wins[-1])
    tail = "; retweets dropped" if rt_dropped else ""
    if len(wins) > 1 and new is not None and old is not None and new >= ok and old < low:
        return f"archive misses older posts{tail}"
    orig_recall = _share(h_orig, c_orig) or 0.0
    if rt_dropped and orig_recall >= ok:
        return "archive works, retweets dropped"
    all_recall = _share(h_rt + h_orig, c_rt + c_orig) or 0.0
    if all_recall >= ok and (_share(h_rt, c_rt) or 0.0) >= ok:
        return "archive works incl. retweets"
    return (
        f"inconclusive: {h_orig}/{c_orig} cached originals and {h_rt}/{c_rt} cached "
        "retweets came back"
    )


def counts_reading(state: dict) -> str:
    """Whether counts/all counts native retweets, from the is:retweet count
    against the retweets cached over the same days."""
    denied = state.get("denied")
    if denied and denied["endpoint"] == xcn.COUNTS_ENDPOINT:
        return f"counts/all: no access ({denied['status']})"
    q_all, q_rt = counts_queries(state["handle"])
    days_all = (state["counts"].get(q_all) or {}).get("days")
    days_rt = (state["counts"].get(q_rt) or {}).get("days")
    if days_all is None or days_rt is None:
        return "counts: not fetched (or not covered)"
    held = state.get("held_by_day") or {}
    held_n = sum(v[0] for v in held.values())
    held_rt = sum(v[1] for v in held.values())
    x_all, x_rt = sum(days_all.values()), sum(days_rt.values())
    nums = f"X counts {x_all} posts, {x_rt} retweets; the cache holds {held_n}, {held_rt} retweets"
    if not held_rt:
        return f"counts: no cached retweets to compare ({nums})"
    ok = settings.X_ARCHIVE_PROBE_RECALL_OK
    low = round(1 - ok, 9)  # 1 - 0.8 is 0.19999999999999996
    if x_rt >= ok * held_rt:
        return f"counts include retweets ({nums})"
    if x_rt <= low * held_rt:
        return f"counts leave out retweets ({nums})"
    return f"counts include some retweets ({nums})"


# ── Text ────────────────────────────────────────────────────────────


def window_lines(windows: list[dict]) -> list[str]:
    return [
        f"  {w['label']}: {w['start'][:16]} .. {w['end'][:16]} UTC, {w['cached']} cached posts "
        f"({w['cached_rt']} retweets, {w['cached_orig']} originals)"
        for w in windows
    ]


def report_lines(state: dict) -> list[str]:
    """The probe's result, id for id, the daily counts and the verdict."""
    done = f"finished {state.get('finished_at')}" if state.get("complete") else "UNFINISHED"
    out = [f"\nFull-archive probe for @{state['handle']} ({done})"]
    denied = state.get("denied")
    if denied:
        out.append(
            f"  {denied['endpoint']} answered {denied['status']}: {denied['detail']}. "
            "No further requests were made."
        )
    for w in state["windows"]:
        r = state["searches"].get(w["label"])
        if r is None:
            out.append(f"  {w['label']}: not searched")
            continue
        more = "; X has more on a next page (not read)" if r["more"] else ""
        out.append(
            f"  {w['label']} {w['start'][:10]}: returned {r['returned']} ({r['returned_rt']} "
            f"retweets), cached {w['cached']} ({w['cached_rt']} retweets); back: "
            f"{r['hits_orig']}/{w['cached_orig']} originals, {r['hits_rt']}/{w['cached_rt']} "
            f"retweets; missing {len(r['missing_ids'])}, extra {len(r['extra_ids'])}{more}"
        )
        if r["missing_ids"]:
            out.append(f"    missing ids: {', '.join(r['missing_ids'])}")
        if r["extra_ids"]:
            out.append(f"    extra ids: {', '.join(r['extra_ids'])}")
    q_all, q_rt = counts_queries(state["handle"])
    days_all = (state["counts"].get(q_all) or {}).get("days")
    days_rt = (state["counts"].get(q_rt) or {}).get("days") or {}
    held = state.get("held_by_day") or {}
    if days_all:
        out.append(f"\n  {'day':<12}{'X':>6}{'X RT':>6}{'held':>6}{'held RT':>9}{'held orig':>11}")
        for d, n in days_all.items():
            h = held.get(d, [0, 0])
            rt = days_rt.get(d, "-")
            out.append(f"  {d:<12}{n:>6}{rt:>6}{h[0]:>6}{h[1]:>9}{h[0] - h[1]:>11}")
    for note in state.get("pending") or []:
        out.append(f"  counts step left for a rerun: {note}")
    out.append(f"\n  {state.get('counts_reading')}")
    out.append(f"Verdict: {state.get('verdict')}")
    return out
