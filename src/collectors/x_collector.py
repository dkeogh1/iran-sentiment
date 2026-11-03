"""
Collector for X (Twitter) posts via the v2 API.

Design:
  - Per-account caching: each user's tweets go to data/raw/x/<handle>.jsonl
    so reruns skip already-collected accounts.
  - Hard budget caps from config.settings to prevent runaway API spend.
  - Gap-fill windows: each account's gap is walked oldest first in fixed
    windows of settings.GAP_FILL_WINDOW_HOURS (35 h, so window ends step
    through every clock hour and weekday). A sampled account
    (settings.ACCOUNT_WINDOW_CAPS) keeps the newest <= N posts of every
    window, with no rollover; any other account is take-all: it reads every
    post, window after window, until its backstop (ACCOUNT_CAP_OVERRIDES /
    MAX_TWEETS_PER_USER) runs out and defers the rest to the next run.
    An incremental run or a first pull appends each window's posts as soon
    as it is read; a forced run merges into the cache once, at the end (or,
    if it fails part-way, with what it read before the error propagates).
    Each window gets a row in settings.X_SAMPLING_DIR once its posts are
    stored.
  - A forced run merges into the cache by id (merge_by_id): the timeline
    reaches back only ~3,200 posts and search ~7 days, so what it can't
    re-read stays.
  - plan_account / estimate_run compute the per-account plan and maximum
    spend WITHOUT touching the API, so the CLI can show and gate the cost.
  - Config-driven: accepts the accounts dict from config.accounts.
"""

import fcntl
import itertools
import json
import logging
import math
import os
import re
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import tweepy
from dotenv import load_dotenv

from config import settings

load_dotenv()

logger = logging.getLogger(__name__)


def get_client(wait_on_rate_limit: bool = True) -> tweepy.Client:
    """Build a tweepy client from the X_BEARER_TOKEN env var. tweepy's own
    429 wait retries without limit; x-counts and x-archive pass False and
    back off themselves, a bounded number of times."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN not set — add it to .env")
    return tweepy.Client(bearer_token=token, wait_on_rate_limit=wait_on_rate_limit)


class CacheBusy(RuntimeError):
    """Another collect or x-backfill-text holds the X cache."""


@contextmanager
def cache_lock():
    """
    One paid writer of the raw X cache at a time: `collect` and
    `x-backfill-text` both plan from the cache, buy reads and rewrite these
    files, so two at once would buy the same reads or lose appended records.
    `x-counts` and `x-archive` take it too, so one paid X run goes at a
    time. Raises CacheBusy at once rather than wait. The lock file sits in
    the data root, outside what `backup` syncs.
    """
    path = settings.DATA_DIR / ".x_cache.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CacheBusy(
                "another collect or x-backfill-text (or x-counts / x-archive) is running"
            ) from None
        yield


# ── Helpers ─────────────────────────────────────────────────────────


def _account_cache_path(handle: str) -> Path:
    return settings.X_RAW_DIR / f"{handle}.jsonl"


def _search_cache_path(query: str) -> Path:
    safe = query.replace(" ", "_").replace("/", "_").replace("#", "hash")
    return settings.X_RAW_DIR / f"search_{safe}.jsonl"


def _save_jsonl(path: Path, records: list[dict]) -> None:
    """Temp file, fsync, rename: the file holds paid reads, so a run killed
    mid-write leaves the old one whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, records: list[dict]) -> None:
    """Append and fsync: a collect appends each window's paid reads as it goes,
    so a run killed later keeps them. A file whose last line a kill cut (no
    final newline) gets the newline first, so the cut record is the only one
    lost and the next record starts on its own line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                lead = "\n"
    with open(path, "a") as f:
        f.write(lead)
        f.writelines(json.dumps(r) + "\n" for r in records)
        f.flush()
        os.fsync(f.fileno())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def _latest_created_at(records: list[dict]) -> datetime | None:
    """Return the max created_at across cached records, or None if empty."""
    timestamps = []
    for r in records:
        ts = r.get("created_at")
        if not ts:
            continue
        try:
            timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except ValueError:
            continue
    return max(timestamps) if timestamps else None


# Fields of the post itself, never expansions: X bills per resource returned,
# so an expansion could add billed reads. note_tweet holds the full text of a
# post over 280 characters; referenced_tweets marks retweets, quotes, replies.
_TWEET_FIELDS = ["created_at", "public_metrics", "lang", "note_tweet", "referenced_tweets"]

_RT_PREFIX = re.compile(r"RT @\w+: ")


def _full_text(tweet) -> str:
    """
    The post's whole text. `text` stops at 280 characters; a longer post's
    full text is in note_tweet, which tweepy's Tweet model does not parse.
    A retweet keeps its "RT @handle: " prefix so it still reads as one.
    """
    note = ((tweet.data or {}).get("note_tweet") or {}).get("text")
    if not note:
        return tweet.text
    rt = _RT_PREFIX.match(tweet.text or "")
    return rt.group(0) + note if rt and not note.startswith(rt.group(0)) else note


def _with_refs(record: dict, tweet) -> dict:
    """Add `ref` ([{type, id}], type retweeted|quoted|replied_to) when the post
    references another. Old cache records lack the key."""
    refs = (tweet.data or {}).get("referenced_tweets") or []
    if refs:
        record["ref"] = [{"type": r.get("type"), "id": str(r.get("id"))} for r in refs]
    return record


# ── Forced runs: merge into the cache ──────────────────────────────

FORCED_SUPERSEDED_REASON = "collect --force: labelled from the text a re-read replaced"


def merge_by_id(existing: list[dict], fetched: list[dict]) -> tuple[list[dict], dict]:
    """
    A full-window run (`--force`) merged into the cache by id. It never gets
    every cached post back (the timeline reaches only ~3,200 posts, search ~7
    days, and the run is capped), and the rest were paid for and can't be
    read again: they stay. A post it returned replaces its cached record, so
    the text, metrics and refs X sends now win. Cached posts keep their place
    in the file; posts new to it are appended in fetch order, as an
    incremental run appends them.

    Returns (merged, stats): stats counts refreshed, added and kept ids, and
    `changed` maps each refreshed id whose text differs to its new text.
    """
    fresh = {str(r["id"]): r for r in fetched}
    merged: list[dict] = []
    refreshed: set[str] = set()
    kept: set[str] = set()
    changed: dict[str, str] = {}
    for r in existing:
        pid = str(r.get("id"))
        new = fresh.get(pid)
        if new is None:
            merged.append(r)
            kept.add(pid)
            continue
        if (new.get("text") or "") != (r.get("text") or ""):
            changed[pid] = new.get("text")
        merged.append(new)
        refreshed.add(pid)
    added = [r for pid, r in fresh.items() if pid not in refreshed]
    merged.extend(added)
    stats = {"refreshed": len(refreshed), "added": len(added), "kept": len(kept)}
    return merged, {**stats, "changed": changed}


def _supersede_changed(changes: dict[str, str]) -> None:
    """
    `analyze` restores a score only onto the text it was made from, but the
    paid Opus and topic label caches are keyed by id alone. So, as
    x-backfill-text does before it rewrites the cache: move the labels of the
    posts whose text changed to their *_superseded archives, then archive
    their sentiment_all rows (Haiku's score_llm has no other copy) and clear
    them onto the new text. Called before the cache write: a run killed in
    between leaves the old text cached with its labels archived, which costs
    a relabel, never a label left on text it wasn't made from.
    """
    if not changes:
        return
    from src.collectors import x_backfill as xb  # imports this module

    moved = xb.supersede_labels(set(changes), FORCED_SUPERSEDED_REASON)
    cleared, archived = xb.clear_scores(changes, FORCED_SUPERSEDED_REASON)
    logger.info(
        "%d re-read posts have new text: labels moved to *_superseded %s; "
        "%d sentiment_all rows cleared (%d archived)",
        len(changes),
        {k: v for k, v in moved.items() if v},
        cleared,
        archived,
    )


def _save_merged(cache: Path, existing: list[dict], fetched: list[dict], who: str) -> list[dict]:
    """Merge a full-window run into a non-empty cache (merge_by_id), moving
    what was made from any text it changed first, and write it."""
    merged, st = merge_by_id(existing, fetched)
    _supersede_changed(st["changed"])
    _save_jsonl(cache, merged)
    logger.info(
        "%s: re-read merged into the cache by id: %d refreshed (%d with new text), "
        "%d added, %d kept that it did not return (total: %d)",
        who,
        st["refreshed"],
        len(st["changed"]),
        st["added"],
        st["kept"],
        len(merged),
    )
    return merged


# ── User timeline collection ───────────────────────────────────────


def _naive_utc(dt: datetime) -> datetime:
    """Strip tz after converting to UTC (the API formatter appends 'Z')."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _iso(dt: datetime) -> str:
    """A request time edge: naive UTC, whole seconds, 'Z'."""
    assert dt.tzinfo is None and dt.microsecond == 0, dt
    return dt.isoformat() + "Z"


def _hm(dt: datetime) -> str:
    return f"{dt:%Y-%m-%dT%H:%M}"


def _created(rec: dict) -> datetime:
    return datetime.fromisoformat(rec["created_at"])  # 3.11+ reads "Z"


def account_cap(handle: str) -> int:
    """Per-run backstop for an account (settings override or default)."""
    return settings.ACCOUNT_CAP_OVERRIDES.get(handle, settings.MAX_TWEETS_PER_USER)


def _coprime_hours(h: int) -> int:
    """The first window length >= h hours that shares no factor with 24, so
    the window ends keep stepping through the clock."""
    while math.gcd(h % 24, 24) != 1:
        h += 1
    return h


def plan_windows(
    start: datetime,
    end: datetime,
    hours: int | None = None,
) -> list[tuple[datetime, datetime]]:
    """
    Split [start, end] into contiguous windows of `hours` (default
    settings.GAP_FILL_WINDOW_HOURS, read per call), OLDEST FIRST, on whole
    seconds: start is rounded up and end floored. Every window but the last
    is exactly `hours` long. A remainder under half a window is merged into
    the last window (up to 1.5x long); a longer one is its own, shorter,
    window. The last window ends at `end`. Pure: no I/O.
    """
    h = hours or settings.GAP_FILL_WINDOW_HOURS
    length = timedelta(hours=h)
    if start.microsecond:
        start = start.replace(microsecond=0) + timedelta(seconds=1)
    end = end.replace(microsecond=0)
    if end <= start:
        return []
    edges = [start]
    while edges[-1] + length <= end:
        edges.append(edges[-1] + length)
    if edges[-1] < end:
        if end - edges[-1] < length / 2 and len(edges) > 1:
            edges[-1] = end
        else:
            edges.append(end)
    return list(itertools.pairwise(edges))


def _window_cap(w: int, s: datetime, e: datetime, h: int) -> int:
    """A sampled account's cap for one window: w for a full window, in
    proportion (rounded half up) for a longer merged tail, never under w."""
    secs = int((e - s).total_seconds())
    full = h * 3600
    return max(w, (2 * w * secs + full) // (2 * full))


def plan_account(
    handle: str,
    start: datetime = settings.COLLECTION_START,
    end: datetime = settings.COLLECTION_END,
    force: bool = False,
    cap: int | None = None,
) -> dict:
    """
    Work out what a collect run would do for one account, from the cache
    and settings alone (no API call).

    `cap` overrides the per-run backstop (ACCOUNT_CAP_OVERRIDES /
    MAX_TWEETS_PER_USER). An account in settings.ACCOUNT_WINDOW_CAPS is
    sampled (mode "sample"): every window gets its own cap, w for a full
    window, and the windows are lengthened (to the next length coprime with
    24) only when the caps would add up to more than the backstop. Any other
    account is take-all (mode "take_all"): caps is None and the backstop is
    the run's maximum, read greedily window after window.

    A forced plan for a sampled account covers the whole [start, end] at
    one density. When that lengthens the windows and the cache stops at
    least half a window before `end`, the account is skipped: the uncached
    tail would be sampled more thinly than an incremental run samples it,
    and the cursor would then move past it for good.

    Returns a dict with: handle, cached (int), incremental (bool), window
    ((start, end) or None when skipped), slices (the windows), caps (per
    window, or None for take-all), cap (the backstop), max_reads,
    skip_reason, mode, window_cap (w or None), hours, lookups (1, or 0 when
    skipped) and max_cost_usd (reads plus the user lookup).
    """
    cache = _account_cache_path(handle)
    existing = _load_jsonl(cache) if cache.exists() else []
    incremental = bool(existing) and not force
    skip_reason = None
    start, end = _naive_utc(start), _naive_utc(end)

    if incremental:
        latest = _latest_created_at(existing)
        if latest is None:
            incremental = False  # unparseable cache -> full pull
        else:
            # +1 second so the boundary tweet is not refetched
            start = _naive_utc(latest + timedelta(seconds=1))
            if start >= end:
                skip_reason = "already past window end"

    backstop = cap if cap is not None else account_cap(handle)
    w = settings.ACCOUNT_WINDOW_CAPS.get(handle)
    floor = settings.X_MIN_READS_PER_REQUEST
    if w is not None and w < floor:
        raise ValueError(
            f"ACCOUNT_WINDOW_CAPS[{handle!r}] = {w} is under X's {floor}-post request minimum"
        )
    if skip_reason is None and backstop < floor:
        skip_reason = f"cap {backstop} is under X's {floor}-post request minimum"

    h = settings.GAP_FILL_WINDOW_HOURS
    windows: list[tuple[datetime, datetime]] = []
    caps: list[int] | None = None if w is None else []
    if skip_reason is None and w is not None:
        gap_h = (end - start).total_seconds() / 3600
        if gap_h < h / 2:
            # The cursor does not move, so nothing is lost.
            skip_reason = "gap shorter than half a window; left for the next run"
        else:
            if w * gap_h / h > backstop:
                h = _coprime_hours(max(h, math.ceil(w * gap_h / backstop)))
            while True:
                windows = plan_windows(start, end, h)
                caps = [_window_cap(w, s, e, h) for s, e in windows]
                if sum(caps) <= backstop or len(windows) == 1:
                    break
                h = _coprime_hours(h + 1)
            if sum(caps) > backstop:  # a single window over the backstop
                caps = [backstop]
            latest = _latest_created_at(existing) if force else None
            if latest is not None and h > settings.GAP_FILL_WINDOW_HOURS:
                cursor = _naive_utc(latest + timedelta(seconds=1))
                if (end - cursor).total_seconds() / 3600 >= settings.GAP_FILL_WINDOW_HOURS / 2:
                    skip_reason = (
                        f"cache ends {_hm(cursor)}: run an incremental collect before "
                        f"--force (its {h} h windows would sample the uncached tail more "
                        "thinly, for good)"
                    )
    elif skip_reason is None:
        windows = plan_windows(start, end, h)
    if skip_reason is None and not windows:
        skip_reason = "nothing to fetch (empty window)"

    if skip_reason:
        windows, max_reads, lookups = [], 0, 0
        caps = None if w is None else []
    else:
        max_reads = sum(caps) if caps is not None else backstop
        lookups = 1
    return {
        "handle": handle,
        "cached": len(existing),
        "incremental": incremental,
        "window": None if skip_reason else (start, end),
        "slices": windows,
        "caps": caps,
        "cap": backstop,
        "max_reads": max_reads,
        "max_cost_usd": (
            max_reads * settings.X_READ_COST_USD + lookups * settings.X_USER_READ_COST_USD
        ),
        "skip_reason": skip_reason,
        "mode": "take_all" if w is None else "sample",
        "window_cap": w,
        "hours": h,
        "lookups": lookups,
    }


def estimate_run(
    accounts: dict[str, list[str]],
    search_terms: list[str] | None = None,
    force: bool = False,
) -> dict:
    """
    Plan a whole `collect` run without calling the API.

    Returns {"accounts": [plan, ...], "searches": [{query, max_reads,
    max_cost_usd}], "max_reads": int, "lookups": int, "max_cost_usd": float}.
    max_cost_usd counts the reads and one user lookup per account fetched.
    Search terms always count their full cap: /search/recent only reaches
    back 7 days, so a stale cache does not reduce what a refresh can read.
    """
    plans = []
    for tier, handles in accounts.items():
        for handle in handles:
            plan = plan_account(handle, force=force)
            plan["tier"] = tier
            plans.append(plan)
    searches = [
        {
            "query": q,
            "max_reads": settings.MAX_TWEETS_PER_SEARCH,
            "max_cost_usd": settings.MAX_TWEETS_PER_SEARCH * settings.X_READ_COST_USD,
        }
        for q in (search_terms or [])
    ]
    max_reads = sum(p["max_reads"] for p in plans) + sum(s["max_reads"] for s in searches)
    return {
        "accounts": plans,
        "searches": searches,
        "max_reads": max_reads,
        "lookups": sum(p["lookups"] for p in plans),
        "max_cost_usd": (
            sum(p["max_cost_usd"] for p in plans) + sum(s["max_cost_usd"] for s in searches)
        ),
    }


def _timeline_record(tweet, handle: str, tier: str) -> dict:
    return _with_refs(
        {
            "id": str(tweet.id),
            "user": handle,
            "tier": tier,
            "text": _full_text(tweet),
            "created_at": tweet.created_at.isoformat(),
            "metrics": dict(tweet.public_metrics) if tweet.public_metrics else {},
            "lang": tweet.lang,
            "platform": "x",
        },
        tweet,
    )


def _fetch_window(
    client: tweepy.Client,
    user_id,
    handle: str,
    tier: str,
    start: datetime,
    end_time: datetime,
    cap: int,
) -> tuple[list[dict], bool]:
    """
    Pull up to `cap` tweets from one user in [start, end_time]. The API
    returns newest-first, so a cap hit keeps the newest tweets in the
    window. No request asks for fewer than X_MIN_READS_PER_REQUEST (X's
    minimum, billed even when fewer are wanted), so the loop stops once
    fewer than that are left to take; it also stops after
    X_MAX_PAGES_PER_WINDOW pages. Every post returned is kept, because
    each was billed.

    Returns (tweets, truncated): truncated means more posts were left when
    the loop stopped (cap reached, request floor or page guard). A window
    filled exactly, with no next page, is not truncated.
    """
    out: list[dict] = []
    token = None
    pages = 0
    while (
        cap - len(out) >= settings.X_MIN_READS_PER_REQUEST
        and pages < settings.X_MAX_PAGES_PER_WINDOW
    ):
        max_results = min(100, cap - len(out))
        resp = client.get_users_tweets(
            id=user_id,
            start_time=_iso(start),
            end_time=_iso(end_time),
            max_results=max_results,
            pagination_token=token,
            tweet_fields=_TWEET_FIELDS,
        )
        pages += 1
        data = resp.data or []
        if len(data) > max_results:
            logger.warning(
                "@%s: asked for %d posts, got %d -- kept them all (billed)",
                handle,
                max_results,
                len(data),
            )
        out.extend(_timeline_record(t, handle, tier) for t in data)
        token = (resp.meta or {}).get("next_token")
        if not token:
            break
    if token is not None and pages >= settings.X_MAX_PAGES_PER_WINDOW:
        logger.warning(
            "@%s: window %s..%s stopped at the %d-page guard with posts left",
            handle,
            _hm(start),
            _hm(end_time),
            settings.X_MAX_PAGES_PER_WINDOW,
        )
    return out, token is not None


def _kept_span_hours(row: dict) -> float | None:
    if not row["got"]:
        return None
    span = datetime.fromisoformat(row["newest_kept"]) - datetime.fromisoformat(row["oldest_kept"])
    return span.total_seconds() / 3600


def _estimated_posts(row: dict, s: datetime, e: datetime) -> float:
    """How many posts a walked window held: its count when it was read
    whole; for a truncated one, the rate of its kept posts (the newest)
    over the whole window, never under what it got."""
    span = _kept_span_hours(row)
    if not row["truncated"] or row["got"] < 2 or not span:
        return row["got"]
    return max(row["got"], (row["got"] - 1) / span * (e - s).total_seconds() / 3600)


def _warn_empty_windows(
    handle: str, windows: list[tuple[datetime, datetime]], walked: list[dict], hours: int
) -> None:
    """
    (a) Leading windows that returned nothing are most likely past the
    ~3,200-post timeline horizon when the windows after them hold an
    estimated half of it or more (_estimated_posts): one warning. Before
    fewer posts, an empty start is just a quiet one.
    (b) Any other empty window next to a dense truncated window (kept span
    under hours/4) is more likely a failed fetch than a silent account, and
    nothing revisits it: the cursor is the newest post.
    """
    lead = 0
    while lead < len(walked) and walked[lead]["got"] == 0:
        lead += 1
    after = sum(_estimated_posts(r, *windows[i]) for i, r in enumerate(walked) if i >= lead)
    horizon = 0 < lead < len(walked) and after >= settings.X_TIMELINE_HORIZON_POSTS / 2
    if horizon:
        logger.warning(
            "@%s: first %d window(s) %s..%s returned 0 posts, and the windows after them hold "
            "an estimated ~%d -- likely past the ~%d-post timeline horizon (lost; empty "
            "responses are free)",
            handle,
            lead,
            _hm(windows[0][0]),
            _hm(windows[lead - 1][1]),
            round(after),
            settings.X_TIMELINE_HORIZON_POSTS,
        )

    def dense_truncated(row: dict) -> bool:
        span = _kept_span_hours(row)
        return row["truncated"] and span is not None and span < hours / 4

    for i, row in enumerate(walked):
        if row["got"] or (horizon and i < lead):
            continue
        near = [walked[j] for j in (i - 1, i + 1) if 0 <= j < len(walked)]
        if any(dense_truncated(r) for r in near):
            logger.warning(
                "@%s: window %s..%s returned 0 posts next to a dense truncated window -- "
                "possible fetch failure; later runs start after the newest post and will "
                "not fetch this window again",
                handle,
                _hm(windows[i][0]),
                _hm(windows[i][1]),
            )


def collect_user(
    client: tweepy.Client,
    handle: str,
    tier: str,
    start: datetime = settings.COLLECTION_START,
    end: datetime = settings.COLLECTION_END,
    max_tweets: int | None = None,
    force: bool = False,
) -> list[dict]:
    """
    Pull tweets from a user's timeline, walking the gap oldest first in
    settings.GAP_FILL_WINDOW_HOURS windows (plan_account). A sampled account
    (settings.ACCOUNT_WINDOW_CAPS) keeps the newest <= cap posts of every
    window, with no rollover from quiet windows; any other account is
    take-all, reading every post window after window until its backstop
    runs out, when the windows left are deferred to the next run. Window
    edges are shared seconds ([s, e] then [e, ...]) and posts are deduped by
    id before they are stored, so no second is missed whether X treats
    end_time as inclusive or not.

    Behavior when a cache file exists:
      - force=True   : re-fetch the whole [start, end] window and merge it
                       into the cache by id (merge_by_id): re-read posts are
                       refreshed, the rest kept. Returns the merged set. The
                       merge is written once, at the end; a run that fails
                       part-way merges what it read before the error
                       propagates. (An unparseable cache is merged the same
                       way.)
      - force=False  : incremental -- fetch only tweets created after the
                       latest cached one, appending each window's new posts
                       as soon as it is read, so a killed run resumes from a
                       window boundary. Returns the full merged set so
                       downstream code sees both old and new.
    With no cache (a first pull), each window is appended as it is read, as
    in an incremental run, so a failed first pull resumes incrementally.

    `max_tweets` overrides the per-run backstop for this call only; by
    default it comes from settings.ACCOUNT_CAP_OVERRIDES / MAX_TWEETS_PER_USER.
    Every window walked gets a row in settings.X_SAMPLING_DIR, written once
    its posts are in the cache.

    Why the plan's maximum (max_reads plus lookups) is never exceeded:
      - sample mode: posts returned <= requested <= cap_i in each window, so
        the total <= sum(caps) = max_reads;
      - take-all mode: each request asks for <= the backstop still left,
        counted on every post returned (duplicates too), so the total <=
        the backstop;
      - exactly one user lookup per planned account, none when skipped;
      - the planner is pure: it reads only the cache and settings.
    """
    cache = _account_cache_path(handle)
    existing = _load_jsonl(cache) if cache.exists() else []

    plan = plan_account(handle, start=start, end=end, force=force, cap=max_tweets)
    if plan["skip_reason"]:
        (logger.warning if force else logger.info)(
            "@%s: %s -- skipping", handle, plan["skip_reason"]
        )
        return existing
    incremental = plan["incremental"]
    windows = plan["slices"]
    sample = plan["mode"] == "sample"
    how = f"sample {plan['window_cap']}/window" if sample else f"take-all, backstop {plan['cap']}"
    if incremental:
        logger.info(
            "@%s: incremental from %s (cached: %d, %s, %d window(s) of %d h, max %d reads)",
            handle,
            plan["window"][0].isoformat(),
            len(existing),
            how,
            len(windows),
            plan["hours"],
            plan["max_reads"],
        )

    try:
        user = client.get_user(username=handle)
    except tweepy.errors.HTTPException as e:
        logger.error("@%s: lookup failed -- %s", handle, e)
        return existing

    if not user.data:
        logger.warning("@%s: not found", handle)
        return existing

    run_end = windows[-1][1]
    manifest = settings.X_SAMPLING_DIR / f"windows_{run_end:%Y-%m-%d}.jsonl"
    # An incremental run or a first pull appends each window as it is read.
    # A merge (--force, or an unparseable cache) writes the cache once, so its
    # manifest rows wait for that write: a row always means its posts are in.
    append = incremental or not existing
    seen = {str(r.get("id")) for r in existing} if incremental else set()
    remaining = plan["max_reads"]
    new_tweets: list[dict] = []
    walked: list[dict] = []
    try:
        for i, (s, e) in enumerate(windows):
            if sample:
                cap_i = plan["caps"][i]
            else:
                if remaining < settings.X_MIN_READS_PER_REQUEST:
                    break
                cap_i = remaining
            got, truncated = _fetch_window(client, user.data.id, handle, tier, s, e, cap_i)
            remaining -= len(got)  # raw count, duplicates too: the maximum stays strict
            new = []
            for r in got:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    new.append(r)
            new_tweets.extend(new)
            by_time = sorted(got, key=_created)
            row = {
                "run_end": _iso(run_end),
                "handle": handle,
                "tier": tier,
                "mode": plan["mode"],
                "hours": plan["hours"],
                "window_start": _iso(s),
                "window_end": _iso(e),
                "cap": cap_i,
                "got": len(got),
                "new": len(new),
                "truncated": truncated,
                "oldest_kept": by_time[0]["created_at"] if by_time else None,
                "newest_kept": by_time[-1]["created_at"] if by_time else None,
            }
            if append:
                if new:
                    _append_jsonl(cache, new)  # per window: a killed run resumes from here
                _append_jsonl(manifest, [row])
            walked.append(row)
            logger.info(
                "@%s: window %s..%s cap %d -> %d%s",
                handle,
                _hm(s),
                _hm(e),
                cap_i,
                len(got),
                " (TRUNCATED)" if truncated else "",
            )
    except BaseException:
        if not append:  # keep the billed reads of the windows already walked
            logger.error(
                "@%s: failed after %d of %d windows -- merging the %d posts read so far "
                "into the cache before the error propagates",
                handle,
                len(walked),
                len(windows),
                len(new_tweets),
            )
            if new_tweets:
                _save_merged(cache, existing, new_tweets, f"@{handle} [{tier}] (partial)")
            if walked:
                _append_jsonl(manifest, walked)
        raise

    # Take-all: the backstop ran out. A window cut by it keeps its newest
    # posts, and the cursor (the newest stored post) moves past the rest;
    # windows not reached are read by the next run.
    if not sample and walked and remaining < settings.X_MIN_READS_PER_REQUEST:
        notes = []
        if walked[-1]["truncated"]:
            cs, ce = windows[len(walked) - 1]
            notes.append(
                f"window {_hm(cs)}..{_hm(ce)} keeps only its newest posts, the rest are not "
                "read again"
            )
        if len(walked) < len(windows):
            notes.append(
                f"windows from {_hm(windows[len(walked)][0])} on not read, deferred to the next run"
            )
        if notes:
            logger.error(
                "@%s: backstop %d reached; %s (add the account to ACCOUNT_WINDOW_CAPS if this "
                "recurs)",
                handle,
                plan["cap"],
                "; ".join(notes),
            )
    _warn_empty_windows(handle, windows, walked, plan["hours"])

    flag = " (TRUNCATED)" if any(r["truncated"] for r in walked) else ""

    if incremental:
        if new_tweets:
            logger.info(
                "@%s [%s]: +%d new tweets%s (total: %d)",
                handle,
                tier,
                len(new_tweets),
                flag,
                len(existing) + len(new_tweets),
            )
        else:
            logger.info("@%s [%s]: no new tweets since last fetch", handle, tier)
        return existing + new_tweets

    if existing:
        merged = _save_merged(cache, existing, new_tweets, f"@{handle} [{tier}]{flag}")
        if walked:
            _append_jsonl(manifest, walked)
        return merged
    if not cache.exists():  # `status` tells a collected account from one never collected
        _save_jsonl(cache, [])
    logger.info("@%s [%s]: %d tweets%s", handle, tier, len(new_tweets), flag)
    return new_tweets


# ── Keyword search collection ──────────────────────────────────────


def _search_record(tweet, query: str) -> dict:
    return _with_refs(
        {
            "id": str(tweet.id),
            "author_id": str(tweet.author_id),
            "user": f"search:{query}",
            "tier": "search",
            "text": _full_text(tweet),
            "created_at": tweet.created_at.isoformat(),
            "metrics": dict(tweet.public_metrics) if tweet.public_metrics else {},
            "lang": tweet.lang,
            "platform": "x",
        },
        tweet,
    )


def collect_search(
    client: tweepy.Client,
    query: str,
    max_total: int = settings.MAX_TWEETS_PER_SEARCH,
    force: bool = False,
) -> list[dict]:
    """
    Run a recent-search query. The /search/recent endpoint only
    returns the last ~7 days of tweets, so this is a near-realtime
    public sentiment proxy rather than historical data.

    Behavior matches collect_user: if a cache exists and not force,
    fetch only tweets newer than the latest cached one and append; with
    force, merge what the 7-day search returns into the cache by id.
    """
    cache = _search_cache_path(query)
    existing = _load_jsonl(cache) if cache.exists() else []

    incremental = bool(existing) and not force
    start_time = None
    if incremental:
        latest = _latest_created_at(existing)
        if latest is not None:
            from datetime import timedelta

            start_time = latest + timedelta(seconds=1)
            if start_time.tzinfo is not None:
                start_time = start_time.astimezone(UTC).replace(tzinfo=None)
            # /search/recent only goes back ~7 days. If the cached
            # latest is older than that, the API will 400. Clamp
            # start_time forward to the window edge (with a 5-minute
            # buffer for clock drift) and warn — there is no way to
            # backfill the gap without the paid full-archive endpoint.
            search_floor = (
                datetime.now(UTC).replace(tzinfo=None, microsecond=0)
                - timedelta(days=7)
                + timedelta(minutes=5)
            )
            if start_time < search_floor:
                logger.warning(
                    "search '%s': cached latest %s is older than 7-day window; "
                    "clamping start_time to %s — gap of %s unfetched",
                    query,
                    latest.isoformat(),
                    search_floor.isoformat(),
                    search_floor - start_time,
                )
                start_time = search_floor
            logger.info(
                "search '%s': incremental from %s (cached: %d)",
                query,
                start_time.isoformat(),
                len(existing),
            )
        else:
            incremental = False

    # No request asks for more than what is left of max_total (the plan
    # counts max_total per term) or for under X's 10-post minimum, so a
    # search never reads more than max_total; every post returned was billed
    # and is kept. Empty pages that keep a next_token stop at a page guard.
    new_tweets: list[dict] = []
    token = None
    pages = 0
    while (
        max_total - len(new_tweets) >= settings.X_MIN_SEARCH_READS_PER_REQUEST
        and pages < settings.X_MAX_PAGES_PER_SEARCH
    ):
        kwargs = {
            "query": query,
            "max_results": min(100, max_total - len(new_tweets)),
            "next_token": token,
            "tweet_fields": [*_TWEET_FIELDS, "author_id"],
        }
        if start_time is not None:
            kwargs["start_time"] = start_time.isoformat() + "Z"
        resp = client.search_recent_tweets(**kwargs)
        pages += 1
        new_tweets.extend(_search_record(t, query) for t in resp.data or [])
        token = (resp.meta or {}).get("next_token")
        if not token:
            break
    if token is not None and pages >= settings.X_MAX_PAGES_PER_SEARCH:
        logger.warning(
            "search '%s': stopped at the %d-page guard with posts left",
            query,
            settings.X_MAX_PAGES_PER_SEARCH,
        )

    if incremental:
        if new_tweets:
            _append_jsonl(cache, new_tweets)
            logger.info(
                "search '%s': +%d new (total: %d)",
                query,
                len(new_tweets),
                len(existing) + len(new_tweets),
            )
        else:
            logger.info("search '%s': no new tweets since last fetch", query)
        return existing + new_tweets

    if existing:
        return _save_merged(cache, existing, new_tweets, f"search '{query}'")
    logger.info("search '%s': %d tweets", query, len(new_tweets))
    _save_jsonl(cache, new_tweets)
    return new_tweets


# ── Orchestration ───────────────────────────────────────────────────


def collect_all(
    accounts: dict[str, list[str]],
    search_terms: list[str] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """
    Run the full collection pipeline for a config-supplied account dict.

    Returns a {source -> count} dict summarizing what was collected.
    """
    client = get_client()
    summary: dict[str, int] = {}

    for tier, handles in accounts.items():
        for handle in handles:
            try:
                tweets = collect_user(client, handle, tier, force=force)
                summary[handle] = len(tweets)
            except Exception as e:
                logger.error("Failed @%s: %s", handle, e)
                summary[handle] = 0

    if search_terms:
        for query in search_terms:
            try:
                tweets = collect_search(client, query, force=force)
                summary[f"search:{query}"] = len(tweets)
            except Exception as e:
                logger.error("Search '%s' failed: %s", query, e)
                summary[f"search:{query}"] = 0

    return summary


def load_all_cached() -> list[dict]:
    """Load every cached JSONL file in the X raw directory."""
    all_posts: list[dict] = []
    for path in sorted(settings.X_RAW_DIR.glob("*.jsonl")):
        all_posts.extend(_load_jsonl(path))

    # Deduplicate by tweet id (search results may overlap with user timelines)
    seen: set[str] = set()
    deduped: list[dict] = []
    for post in all_posts:
        pid = post.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(post)
    return deduped
