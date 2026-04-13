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


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


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

    If a cached file already exists for this handle and `force` is False,
    the cached data is returned without hitting the API.
    """
    cache = _account_cache_path(handle)
    if cache.exists() and not force:
        existing = _load_jsonl(cache)
        logger.info("@%s: cached (%d tweets) — skipping", handle, len(existing))
        return existing

    try:
        user = client.get_user(username=handle)
    except tweepy.errors.HTTPException as e:
        logger.error("@%s: lookup failed — %s", handle, e)
        return []

    if not user.data:
        logger.warning("@%s: not found", handle)
        return []

    tweets: list[dict] = []
    pagination_token = None

    while len(tweets) < max_tweets:
        remaining = max_tweets - len(tweets)
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
                if len(tweets) >= max_tweets:
                    break
                tweets.append({
                    "id": str(tweet.id),
                    "user": handle,
                    "tier": tier,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "metrics": dict(tweet.public_metrics) if tweet.public_metrics else {},
                    "lang": tweet.lang,
                    "platform": "x",
                })

        if resp.meta and resp.meta.get("next_token") and len(tweets) < max_tweets:
            pagination_token = resp.meta["next_token"]
        else:
            break

    capped = " (CAPPED)" if len(tweets) >= max_tweets else ""
    logger.info("@%s [%s]: %d tweets%s", handle, tier, len(tweets), capped)

    _save_jsonl(cache, tweets)
    return tweets


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
    """
    cache = _search_cache_path(query)
    if cache.exists() and not force:
        existing = _load_jsonl(cache)
        logger.info("search '%s': cached (%d tweets) — skipping", query, len(existing))
        return existing

    tweets: list[dict] = []
    pagination_token = None

    while len(tweets) < max_total:
        remaining = max_total - len(tweets)
        per_page = min(100, remaining)

        resp = client.search_recent_tweets(
            query=query,
            max_results=max(10, per_page),
            next_token=pagination_token,
            tweet_fields=["created_at", "author_id", "public_metrics", "lang"],
        )

        if resp.data:
            for tweet in resp.data:
                if len(tweets) >= max_total:
                    break
                tweets.append({
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

        if resp.meta and resp.meta.get("next_token") and len(tweets) < max_total:
            pagination_token = resp.meta["next_token"]
        else:
            break

    logger.info("search '%s': %d tweets", query, len(tweets))
    _save_jsonl(cache, tweets)
    return tweets


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
