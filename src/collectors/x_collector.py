"""
Collector for X (Twitter) posts via the v2 API.

Design:
  - Per-account caching: each user's tweets go to data/raw/x/<handle>.jsonl
    so reruns skip already-collected accounts.
  - Hard budget caps from config.settings to prevent runaway API spend.
  - Gap-fill slicing: long fetch windows are walked oldest-slice-first with
    the cap shared across slices (see settings.GAP_FILL_SLICE_DAYS).
  - plan_account / estimate_run compute the per-account plan and maximum
    spend WITHOUT touching the API, so the CLI can show and gate the cost.
  - Config-driven: accepts the accounts dict from config.accounts.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tweepy
from dotenv import load_dotenv

from config import settings

load_dotenv()

logger = logging.getLogger(__name__)


def get_client() -> tweepy.Client:
    """Build a tweepy client from the X_BEARER_TOKEN env var."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN not set — add it to .env")
    return tweepy.Client(bearer_token=token, wait_on_rate_limit=True)


# ── Helpers ─────────────────────────────────────────────────────────

def _account_cache_path(handle: str) -> Path:
    return settings.X_RAW_DIR / f"{handle}.jsonl"


def _search_cache_path(query: str) -> Path:
    safe = query.replace(" ", "_").replace("/", "_").replace("#", "hash")
    return settings.X_RAW_DIR / f"search_{safe}.jsonl"


def _save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


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


# ── User timeline collection ───────────────────────────────────────

def _naive_utc(dt: datetime) -> datetime:
    """Strip tz after converting to UTC (the API formatter appends 'Z')."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def account_cap(handle: str) -> int:
    """Per-run tweet cap for an account (settings override or default)."""
    return settings.ACCOUNT_CAP_OVERRIDES.get(handle, settings.MAX_TWEETS_PER_USER)


def plan_slices(
    start: datetime,
    end: datetime,
    slice_days: int = settings.GAP_FILL_SLICE_DAYS,
) -> list[tuple[datetime, datetime]]:
    """
    Split [start, end] into contiguous windows of at most `slice_days`,
    OLDEST FIRST. The last slice is short so the final edge is exactly `end`.
    """
    if end <= start:
        return []
    step = timedelta(days=slice_days)
    slices: list[tuple[datetime, datetime]] = []
    s = start
    while s < end:
        e = min(s + step, end)
        slices.append((s, e))
        s = e
    return slices


def split_cap(cap: int, n_slices: int) -> list[int]:
    """
    Share `cap` across slices so the list sums to exactly `cap`. The
    remainder goes to the NEWEST slices (end of the list).
    """
    if n_slices <= 0:
        return []
    base, extra = divmod(cap, n_slices)
    return [base + (1 if i >= n_slices - extra else 0) for i in range(n_slices)]


def plan_account(
    handle: str,
    start: datetime = settings.COLLECTION_START,
    end: datetime = settings.COLLECTION_END,
    force: bool = False,
) -> dict:
    """
    Work out what a collect run would do for one account, from the cache
    alone (no API call).

    Returns a dict with: handle, cached (int), incremental (bool),
    window (tuple | None -- None means nothing to fetch), slices, caps
    (per slice, sums to cap), cap, max_reads, max_cost_usd, skip_reason.
    """
    cache = _account_cache_path(handle)
    existing = _load_jsonl(cache) if cache.exists() else []
    incremental = bool(existing) and not force
    skip_reason = None

    if incremental:
        latest = _latest_created_at(existing)
        if latest is None:
            incremental = False  # unparseable cache -> full pull
        else:
            # +1 second so the boundary tweet is not refetched
            start = _naive_utc(latest + timedelta(seconds=1))
            if start >= end:
                skip_reason = "already past window end"

    slices = [] if skip_reason else plan_slices(start, end)
    cap = account_cap(handle)
    caps = split_cap(cap, len(slices))
    max_reads = sum(caps)
    return {
        "handle": handle,
        "cached": len(existing),
        "incremental": incremental,
        "window": None if skip_reason else (start, end),
        "slices": slices,
        "caps": caps,
        "cap": cap,
        "max_reads": max_reads,
        "max_cost_usd": max_reads * settings.X_READ_COST_USD,
        "skip_reason": skip_reason,
    }


def estimate_run(
    accounts: dict[str, list[str]],
    search_terms: list[str] | None = None,
    force: bool = False,
) -> dict:
    """
    Plan a whole `collect` run without calling the API.

    Returns {"accounts": [plan, ...], "searches": [{query, max_reads,
    max_cost_usd}], "max_reads": int, "max_cost_usd": float}.
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
        "max_cost_usd": max_reads * settings.X_READ_COST_USD,
    }


def _fetch_window(
    client: tweepy.Client,
    user_id,
    handle: str,
    tier: str,
    start: datetime,
    end: datetime,
    cap: int,
) -> tuple[list[dict], bool]:
    """
    Pull up to `cap` tweets from one user in [start, end]. The API returns
    newest-first, so a cap hit keeps the newest tweets in the window.
    Returns (tweets, capped).
    """
    out: list[dict] = []
    pagination_token = None
    if cap <= 0:
        return out, False

    while len(out) < cap:
        remaining = cap - len(out)
        per_page = min(100, remaining)

        resp = client.get_users_tweets(
            id=user_id,
            start_time=start.isoformat() + "Z",
            end_time=end.isoformat() + "Z",
            max_results=max(5, per_page),  # API requires min 5
            pagination_token=pagination_token,
            tweet_fields=["created_at", "public_metrics", "lang"],
        )

        if resp.data:
            for tweet in resp.data:
                if len(out) >= cap:
                    break
                out.append({
                    "id": str(tweet.id),
                    "user": handle,
                    "tier": tier,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "metrics": dict(tweet.public_metrics) if tweet.public_metrics else {},
                    "lang": tweet.lang,
                    "platform": "x",
                })

        if resp.meta and resp.meta.get("next_token") and len(out) < cap:
            pagination_token = resp.meta["next_token"]
        else:
            break

    return out, len(out) >= cap


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
    Pull tweets from a user's timeline with a hard cap, walking the window
    oldest-slice-first (settings.GAP_FILL_SLICE_DAYS) with the cap shared
    across slices.

    Behavior when a cache file exists:
      - force=True   : ignore cache, re-fetch the whole [start, end] window,
                       overwrite the cache.
      - force=False  : incremental -- fetch only tweets created after the
                       latest cached one and append them to the cache.
                       Returns the full merged set so downstream code sees
                       both old and new.

    `max_tweets` overrides the per-run cap for this call only; by default
    the cap comes from settings.ACCOUNT_CAP_OVERRIDES / MAX_TWEETS_PER_USER.
    """
    cache = _account_cache_path(handle)
    existing = _load_jsonl(cache) if cache.exists() else []

    plan = plan_account(handle, start=start, end=end, force=force)
    if max_tweets is not None:
        plan["caps"] = split_cap(max_tweets, len(plan["slices"]))
        plan["cap"] = max_tweets

    if plan["skip_reason"]:
        logger.info("@%s: %s -- skipping", handle, plan["skip_reason"])
        return existing
    incremental = plan["incremental"]
    if incremental:
        logger.info("@%s: incremental from %s (cached: %d, %d slice(s), cap %d)",
                    handle, plan["window"][0].isoformat(), len(existing),
                    len(plan["slices"]), plan["cap"])

    try:
        user = client.get_user(username=handle)
    except tweepy.errors.HTTPException as e:
        logger.error("@%s: lookup failed -- %s", handle, e)
        return existing

    if not user.data:
        logger.warning("@%s: not found", handle)
        return existing

    new_tweets: list[dict] = []
    any_capped = False
    for (s, e), cap in zip(plan["slices"], plan["caps"]):
        got, capped = _fetch_window(client, user.data.id, handle, tier, s, e, cap)
        any_capped = any_capped or capped
        new_tweets.extend(got)
        logger.info("@%s: slice %s..%s -> %d%s", handle, s.date(), e.date(),
                    len(got), " (CAPPED)" if capped else "")

    capped = " (CAPPED)" if any_capped else ""

    if incremental:
        if new_tweets:
            _append_jsonl(cache, new_tweets)
            logger.info("@%s [%s]: +%d new tweets%s (total: %d)",
                        handle, tier, len(new_tweets), capped,
                        len(existing) + len(new_tweets))
        else:
            logger.info("@%s [%s]: no new tweets since last fetch", handle, tier)
        return existing + new_tweets

    logger.info("@%s [%s]: %d tweets%s", handle, tier, len(new_tweets), capped)
    _save_jsonl(cache, new_tweets)
    return new_tweets


# ── Keyword search collection ──────────────────────────────────────

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
    fetch only tweets newer than the latest cached one and append.
    """
    cache = _search_cache_path(query)
    existing = _load_jsonl(cache) if cache.exists() else []

    incremental = bool(existing) and not force
    start_time = None
    if incremental:
        latest = _latest_created_at(existing)
        if latest is not None:
            from datetime import timedelta, timezone
            start_time = latest + timedelta(seconds=1)
            if start_time.tzinfo is not None:
                start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)
            # /search/recent only goes back ~7 days. If the cached
            # latest is older than that, the API will 400. Clamp
            # start_time forward to the window edge (with a 5-minute
            # buffer for clock drift) and warn — there is no way to
            # backfill the gap without the paid full-archive endpoint.
            search_floor = (
                datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
                - timedelta(days=7) + timedelta(minutes=5)
            )
            if start_time < search_floor:
                logger.warning(
                    "search '%s': cached latest %s is older than 7-day window; "
                    "clamping start_time to %s — gap of %s unfetched",
                    query, latest.isoformat(), search_floor.isoformat(),
                    search_floor - start_time,
                )
                start_time = search_floor
            logger.info("search '%s': incremental from %s (cached: %d)",
                        query, start_time.isoformat(), len(existing))
        else:
            incremental = False

    new_tweets: list[dict] = []
    pagination_token = None

    while len(new_tweets) < max_total:
        remaining = max_total - len(new_tweets)
        per_page = min(100, remaining)

        kwargs = dict(
            query=query,
            max_results=max(10, per_page),
            next_token=pagination_token,
            tweet_fields=["created_at", "author_id", "public_metrics", "lang"],
        )
        if start_time is not None:
            kwargs["start_time"] = start_time.isoformat() + "Z"

        resp = client.search_recent_tweets(**kwargs)

        if resp.data:
            for tweet in resp.data:
                if len(new_tweets) >= max_total:
                    break
                new_tweets.append({
                    "id": str(tweet.id),
                    "author_id": str(tweet.author_id),
                    "user": f"search:{query}",
                    "tier": "search",
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "metrics": dict(tweet.public_metrics) if tweet.public_metrics else {},
                    "lang": tweet.lang,
                    "platform": "x",
                })

        if resp.meta and resp.meta.get("next_token") and len(new_tweets) < max_total:
            pagination_token = resp.meta["next_token"]
        else:
            break

    if incremental:
        if new_tweets:
            _append_jsonl(cache, new_tweets)
            logger.info("search '%s': +%d new (total: %d)",
                        query, len(new_tweets), len(existing) + len(new_tweets))
        else:
            logger.info("search '%s': no new tweets since last fetch", query)
        return existing + new_tweets

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
