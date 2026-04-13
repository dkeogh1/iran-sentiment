"""
Collector for Truth Social posts.

HTTP stack: `curl_cffi` with Chrome TLS impersonation. Plain httpx
gets 403'd at Truth Social's Cloudflare edge because Cloudflare
fingerprints the TLS handshake (JA3/JA4), independent of headers or
auth tokens. `curl_cffi` links against a patched libcurl that
impersonates real browser TLS stacks, which passes the check.

Endpoints and their auth requirements:
  - /accounts/lookup, /accounts/:id/statuses — anonymous, no token
  - /statuses/:id                             — anonymous, no token
  - /statuses/:id/context (reply tree)        — REQUIRES a user bearer
    token, even for public posts. Truth Social does not honor the
    OAuth `client_credentials` grant, so there is no way to get an
    app-only token — a real account is required.

Caching mirrors the X collector: one JSONL per handle at
`data/raw/truthsocial/<handle>.jsonl`, skipped on reruns unless
`force=True`.

The legacy `collect_via_truthbrush` path is kept as a fallback for the
reply tree — if you later wire up credentials via TRUTHSOCIAL_USERNAME
/ TRUTHSOCIAL_PASSWORD in .env, truthbrush's `pull_comments` can be
used from `collect_replies`.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path

from curl_cffi import requests as cffi_requests

from config import settings

logger = logging.getLogger(__name__)

#: Per-account cache directory (from settings — do not hardcode)
RAW_DIR = settings.TRUTH_SOCIAL_RAW_DIR

# Truth Social's public API (Mastodon-compatible) base URL
TS_API_BASE = "https://truthsocial.com/api/v1"

# Truth Social sits behind Cloudflare, which fingerprints the TLS
# handshake (JA3/JA4) and returns 403 to plain httpx/requests clients.
# `curl_cffi` links against a patched libcurl that impersonates real
# browser TLS stacks. `chrome` picks the latest supported Chrome build.
_IMPERSONATE = "chrome"
_DEFAULT_TIMEOUT = 30.0


def _ts_get(url: str, params: dict | None = None, timeout: float = _DEFAULT_TIMEOUT):
    """Single GET helper with browser TLS impersonation."""
    return cffi_requests.get(
        url, params=params, impersonate=_IMPERSONATE, timeout=timeout,
    )


def _account_cache_path(handle: str) -> Path:
    """Deterministic cache path so reruns are idempotent."""
    return RAW_DIR / f"{handle}.jsonl"


# ── Strategy 1: truthbrush ──────────────────────────────────────────

def collect_via_truthbrush(
    username: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Use the truthbrush library to pull posts from a Truth Social account."""
    try:
        from truthbrush import Api
    except ImportError:
        logger.error(
            "truthbrush not installed. Run: pip install 'iran-sentiment[truthsocial]'"
        )
        return []

    api = Api()
    posts: list[dict] = []

    for status in api.pull_statuses(username):
        created = datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
        if created.date() < start_date:
            break  # statuses come in reverse chronological order
        if created.date() > end_date:
            continue

        posts.append(
            {
                "id": status["id"],
                "user": username,
                "text": _strip_html(status.get("content", "")),
                "created_at": status["created_at"],
                "metrics": {
                    "reblogs": status.get("reblogs_count", 0),
                    "favourites": status.get("favourites_count", 0),
                    "replies": status.get("replies_count", 0),
                },
                "platform": "truthsocial",
            }
        )

    logger.info("truthbrush: collected %d posts from @%s", len(posts), username)
    return posts


# ── Strategy 2: public Mastodon-compat API ──────────────────────────

def collect_via_public_api(
    username: str,
    start_date: date,
    end_date: date,
    *,
    max_posts: int | None = None,
) -> list[dict]:
    """
    Fetch posts from Truth Social's public Mastodon-compatible API.

    Works anonymously for public accounts like @realDonaldTrump. Uses
    curl_cffi to impersonate a Chrome TLS fingerprint — plain httpx
    gets 403'd at the Cloudflare edge.
    """
    posts: list[dict] = []

    # Look up account ID
    resp = _ts_get(f"{TS_API_BASE}/accounts/lookup", params={"acct": username})
    if resp.status_code != 200:
        logger.warning(
            "Could not look up @%s: HTTP %s %s",
            username, resp.status_code, resp.text[:200],
        )
        return []

    account_id = resp.json()["id"]

    # Paginate statuses — Mastodon uses max_id as the keyset cursor
    max_id = None
    while True:
        params: dict = {"limit": 40}
        if max_id:
            params["max_id"] = max_id

        resp = _ts_get(
            f"{TS_API_BASE}/accounts/{account_id}/statuses",
            params=params,
        )
        if resp.status_code != 200:
            logger.warning(
                "Statuses fetch failed for @%s: HTTP %s",
                username, resp.status_code,
            )
            break

        batch = resp.json()
        if not batch:
            break

        for status in batch:
            created = datetime.fromisoformat(
                status["created_at"].replace("Z", "+00:00")
            )
            if created.date() < start_date:
                logger.info(
                    "@%s: hit start_date boundary (%s), stopping",
                    username, start_date,
                )
                return posts
            if created.date() <= end_date:
                posts.append(
                    {
                        "id": status["id"],
                        "user": username,
                        "text": _strip_html(status.get("content", "")),
                        "created_at": status["created_at"],
                        "metrics": {
                            "reblogs": status.get("reblogs_count", 0),
                            "favourites": status.get("favourites_count", 0),
                            "replies": status.get("replies_count", 0),
                        },
                        "platform": "truthsocial",
                    }
                )
                if max_posts is not None and len(posts) >= max_posts:
                    logger.info("@%s: hit max_posts cap (%d)", username, max_posts)
                    return posts

        max_id = batch[-1]["id"]

    logger.info("Public API: collected %d posts from @%s", len(posts), username)
    return posts


# ── Helpers ─────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Naive HTML tag removal for Truth Social post content."""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _save_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── User timeline collection ───────────────────────────────────────

def collect_user(
    handle: str,
    tier: str,
    *,
    start: date | datetime = settings.COLLECTION_START,
    end: date | datetime = settings.COLLECTION_END,
    force: bool = False,
) -> list[dict]:
    """
    Fetch a Truth Social account's posts into a deterministic cache.

    Mirrors `x_collector.collect_user`: if the cache file exists and
    `force` is False, returns the cached posts without hitting the API.
    Each record is tagged with `tier` so downstream analysis can group
    by tier the same way it does for X data.

    Strategy: public Mastodon-compatible API (no auth, free). Accounts
    that aren't publicly exposed need truthbrush with credentials —
    call `collect_via_truthbrush` directly in that case.
    """
    cache = _account_cache_path(handle)
    # Treat an empty cache file as "not collected". An earlier failed run
    # (rate limit, Cloudflare block, etc.) leaves a zero-byte file behind,
    # and we don't want that to poison future runs. A legitimately empty
    # window is rare and cheap to re-fetch, so this is the safer default.
    if cache.exists() and not force:
        existing = _load_jsonl(cache)
        if existing:
            logger.info("@%s: cached (%d posts) — skipping", handle, len(existing))
            return existing
        logger.info("@%s: empty cache file — treating as not collected", handle)

    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end

    posts = collect_via_public_api(handle, start_date, end_date)
    for p in posts:
        p["tier"] = tier  # annotate for downstream grouping

    logger.info("@%s [%s]: %d posts", handle, tier, len(posts))
    if posts:
        _save_jsonl(cache, posts)
    else:
        logger.warning(
            "@%s: no posts fetched — NOT writing cache so a retry can proceed",
            handle,
        )
    return posts


# ── Orchestration ──────────────────────────────────────────────────

def collect_all(
    accounts: dict[str, list[str]],
    *,
    start: date | datetime = settings.COLLECTION_START,
    end: date | datetime = settings.COLLECTION_END,
    force: bool = False,
) -> dict[str, int]:
    """
    Run the Truth Social collection pipeline for a tiered accounts dict
    (e.g. `TRUTH_SOCIAL_ACCOUNTS` from config.accounts).

    Returns {handle: post_count}. Accounts that fail to look up are
    recorded as 0 rather than aborting the whole run.
    """
    summary: dict[str, int] = {}
    for tier, handles in accounts.items():
        for handle in handles:
            try:
                posts = collect_user(handle, tier, start=start, end=end, force=force)
                summary[handle] = len(posts)
            except Exception as e:
                logger.error("Failed @%s on Truth Social: %s", handle, e)
                summary[handle] = 0
    return summary


def load_all_cached() -> list[dict]:
    """Load every cached per-account JSONL (excludes reply files)."""
    all_posts: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        if path.stem.startswith("replies_"):
            continue
        all_posts.extend(_load_jsonl(path))

    seen: set[str] = set()
    deduped: list[dict] = []
    for post in all_posts:
        pid = post.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(post)
    return deduped


# ── Reply collection ───────────────────────────────────────────────

def _reply_record(status: dict, parent_id: str) -> dict:
    """
    Flatten a Mastodon-compatible status dict into the shape our
    sentiment pipeline expects, preserving the fields we need for
    loyalty-prior segmentation (account age, bio, follower count).
    """
    acct = status.get("account") or {}
    return {
        "id": status["id"],
        "parent_id": parent_id,
        "in_reply_to_id": status.get("in_reply_to_id"),
        "user": acct.get("username", ""),
        "text": _strip_html(status.get("content", "")),
        "created_at": status["created_at"],
        "metrics": {
            "reblogs": status.get("reblogs_count", 0),
            "favourites": status.get("favourites_count", 0),
            "replies": status.get("replies_count", 0),
        },
        # Loyalty-prior features — used by event_study to segment
        # repliers by how long they've been on Truth Social and what
        # their bio claims about them.
        "account": {
            "id": acct.get("id"),
            "username": acct.get("username", ""),
            "display_name": acct.get("display_name", ""),
            "bio": _strip_html(acct.get("note", "")),
            "created_at": acct.get("created_at"),
            "followers_count": acct.get("followers_count", 0),
            "following_count": acct.get("following_count", 0),
            "statuses_count": acct.get("statuses_count", 0),
            "verified": acct.get("verified", False),
        },
        "platform": "truthsocial",
        "source": "reply",
    }


def _get_truthbrush_api():
    """
    Get an authenticated truthbrush Api instance.

    Reads TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD from .env
    (truthbrush picks them up from the environment automatically).
    Raises RuntimeError with a clear message if either is missing or
    if truthbrush isn't installed.
    """
    try:
        from truthbrush.api import Api
    except ImportError:
        raise RuntimeError(
            "truthbrush is required for reply collection. "
            "Install with: pip install 'truthbrush>=0.2'"
        )

    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)

    if not os.environ.get("TRUTHSOCIAL_USERNAME") or not os.environ.get("TRUTHSOCIAL_PASSWORD"):
        raise RuntimeError(
            "Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD in .env "
            "to collect replies. See CLAUDE.md for details."
        )

    return Api()


def collect_replies(
    post_id: str,
    *,
    label: str | None = None,
    include_all: bool = True,
    only_direct: bool = True,
) -> list[dict]:
    """
    Fetch replies to a Truth Social post using truthbrush.

    Uses truthbrush's `pull_comments` which calls the paginated
    `/v1/statuses/{id}/context/descendants` endpoint with Link-header
    pagination and built-in rate-limit backoff. This is the only way
    to get the full reply tree (standard `/context` is Cloudflare-
    blocked for unauthenticated clients and truncates to ~200 for
    authenticated ones).

    Args:
        post_id: Truth Social status ID.
        label: Slug used for the cache filename (defaults to post_id).
        include_all: If True, fetch every reply (can be thousands).
                     If False, truthbrush defaults to 40.
        only_direct: If True, only keep replies directly to the post
                     (in_reply_to_id == post_id), filtering out sub-
                     threads. This matches the NYT's analysis scope.

    Returns:
        List of reply dicts in our standard record shape, with account
        metadata for loyalty-prior scoring.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    label = label or post_id

    api = _get_truthbrush_api()

    raw_replies: list[dict] = []
    for status in api.pull_comments(post_id, include_all=include_all, only_first=only_direct):
        raw_replies.append(status)
        if len(raw_replies) % 500 == 0:
            logger.info("  ... %d replies so far for %s", len(raw_replies), label)

    replies = [_reply_record(s, parent_id=post_id) for s in raw_replies]
    logger.info("Collected %d replies to post %s", len(replies), post_id)

    # Fetch the parent post's replies_count to report coverage
    try:
        resp = _ts_get(f"{TS_API_BASE}/statuses/{post_id}")
        if resp.status_code == 200:
            reported = resp.json().get("replies_count", 0) or 0
            pct = (100 * len(replies) / reported) if reported else 0
            logger.info(
                "Coverage: %d/%d replies (%.1f%%)", len(replies), reported, pct,
            )
    except Exception:
        pass  # non-critical — don't block on this

    # Write cache
    out = RAW_DIR / f"replies_{label}.jsonl"
    with open(out, "w") as f:
        for r in replies:
            f.write(json.dumps(r) + "\n")
    logger.info("Saved replies → %s", out)

    return replies


def load_cached_replies(label: str) -> list[dict]:
    """Load a cached reply JSONL by tracked-post slug (or post_id)."""
    path = RAW_DIR / f"replies_{label}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_cached_replies() -> list[dict]:
    """Load every cached reply file, tagging each record with its source slug."""
    out: list[dict] = []
    for path in RAW_DIR.glob("replies_*.jsonl"):
        slug = path.stem.removeprefix("replies_")
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["tracked_slug"] = slug
                out.append(rec)
    return out
