"""
Collector for X (Twitter) posts via the v2 API.

Design:
  - Per-account caching: each user's tweets go to data/raw/x/<handle>.jsonl
    so reruns skip already-collected accounts.
  - Hard budget caps from config.settings to prevent runaway API spend.
  - Config-driven: accepts the accounts dict from config.accounts.
"""

import json
import logging
import os
from datetime import datetime
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

def collect_user(
    client: tweepy.Client,
    handle: str,
    tier: str,
    start: datetime = settings.COLLECTION_START,
    end: datetime = settings.COLLECTION_END,
    max_tweets: int = settings.MAX_TWEETS_PER_USER,
    force: bool = False,
) -> list[dict]:
    """
    Pull tweets from a user's timeline with a hard cap.

    Behavior when a cache file exists:
      - force=True   : ignore cache, re-fetch the whole [start, end] window,
                       overwrite the cache.
      - force=False  : incremental — fetch only tweets created after the
                       latest cached one (capped at max_tweets new) and
                       append them to the cache. Returns the full merged
                       set so downstream code sees both old and new.
    """
    cache = _account_cache_path(handle)
    existing = _load_jsonl(cache) if cache.exists() else []

    # Incremental path — narrow the window to (latest_cached, end]
    incremental = bool(existing) and not force
    if incremental:
        latest = _latest_created_at(existing)
        if latest is None:
            # Cache exists but has no parseable timestamps — fall back to full pull
            incremental = False
        else:
            # +1 second so we don't refetch the boundary tweet
            from datetime import timedelta, timezone
            window_start = latest + timedelta(seconds=1)
            # Strip timezone for the API formatter below
            if window_start.tzinfo is not None:
                window_start = window_start.astimezone(timezone.utc).replace(tzinfo=None)
            if window_start >= end:
                logger.info("@%s: cached up to %s — already past window end, skipping",
                            handle, latest.isoformat())
                return existing
            start = window_start
            logger.info("@%s: incremental from %s (cached: %d)",
                        handle, start.isoformat(), len(existing))

    try:
        user = client.get_user(username=handle)
    except tweepy.errors.HTTPException as e:
        logger.error("@%s: lookup failed — %s", handle, e)
        return existing

    if not user.data:
        logger.warning("@%s: not found", handle)
        return existing

    new_tweets: list[dict] = []
    pagination_token = None

    while len(new_tweets) < max_tweets:
        remaining = max_tweets - len(new_tweets)
        per_page = min(100, remaining)

        resp = client.get_users_tweets(
            id=user.data.id,
            start_time=start.isoformat() + "Z",
            end_time=end.isoformat() + "Z",
            max_results=max(5, per_page),  # API requires min 5
            pagination_token=pagination_token,
            tweet_fields=["created_at", "public_metrics", "lang"],
        )

        if resp.data:
            for tweet in resp.data:
                if len(new_tweets) >= max_tweets:
                    break
                new_tweets.append({
                    "id": str(tweet.id),
                    "user": handle,
                    "tier": tier,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "metrics": dict(tweet.public_metrics) if tweet.public_metrics else {},
                    "lang": tweet.lang,
                    "platform": "x",
                })

        if resp.meta and resp.meta.get("next_token") and len(new_tweets) < max_tweets:
            pagination_token = resp.meta["next_token"]
        else:
            break

    capped = " (CAPPED)" if len(new_tweets) >= max_tweets else ""

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
