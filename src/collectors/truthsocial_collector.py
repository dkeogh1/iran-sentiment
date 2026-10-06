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
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from curl_cffi import requests as cffi_requests

from config import settings
from src.text_rules import has_text

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
        url,
        params=params,
        impersonate=_IMPERSONATE,
        timeout=timeout,
    )


def _retry_after_seconds(resp) -> float:
    """Seconds to wait after a 429, from Retry-After / X-RateLimit-Reset,
    clamped to [TS_MIN_BACKOFF_S, TS_MAX_BACKOFF_S]."""
    wait = None
    headers = getattr(resp, "headers", {}) or {}
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra:
        try:
            wait = float(ra)
        except ValueError:
            pass
    if wait is None:
        reset = headers.get("x-ratelimit-reset") or headers.get("X-RateLimit-Reset")
        if reset:
            try:
                dt = datetime.fromisoformat(reset.replace("Z", "+00:00"))
                wait = (dt - datetime.now(UTC)).total_seconds()
            except ValueError:
                pass
    if wait is None:
        wait = settings.TS_DEFAULT_BACKOFF_S
    return max(settings.TS_MIN_BACKOFF_S, min(wait, settings.TS_MAX_BACKOFF_S))


def _ts_get_paced(url: str, params: dict | None = None):
    """
    GET with pacing and 429 backoff. Sleeps TS_PAGE_DELAY_S before every
    call; on 429 waits per the response headers and retries up to
    TS_MAX_RETRIES times. Returns the final response (may still be 429),
    at once: every caller stops on it, so a wait after it would only idle.
    """
    tries = settings.TS_MAX_RETRIES + 1
    for attempt in range(tries):
        time.sleep(settings.TS_PAGE_DELAY_S)
        resp = _ts_get(url, params=params)
        if resp.status_code != 429 or attempt + 1 == tries:
            break  # no backoff after the last try: the caller stops on it
        wait = _retry_after_seconds(resp)
        logger.warning(
            "429 from %s (attempt %d/%d) -- backing off %.0fs",
            url.rsplit("/", 2)[-1],
            attempt + 1,
            tries,
            wait,
        )
        time.sleep(wait)
    return resp


def _account_cache_path(handle: str) -> Path:
    """Deterministic cache path so reruns are idempotent."""
    return RAW_DIR / f"{handle}.jsonl"


# ── Strategy 1: truthbrush ──────────────────────────────────────────


def _has_truthbrush_creds() -> bool:
    import os

    from dotenv import load_dotenv

    load_dotenv(override=True)
    return bool(os.environ.get("TRUTHSOCIAL_USERNAME") and os.environ.get("TRUTHSOCIAL_PASSWORD"))


def collect_via_truthbrush(
    username: str,
    start_date: date,
    end_date: date,
    *,
    since_id: str | None = None,
    created_after: datetime | None = None,
) -> list[dict]:
    """
    Use the truthbrush library (authenticated) to pull posts from a Truth
    Social account. `since_id` / `created_after` bound the walk from the
    old end for incremental refreshes; the library paginates newest-first
    and stops at the bound.
    """
    try:
        api = _get_truthbrush_api()
    except RuntimeError as e:
        logger.error("%s", e)
        return []

    posts: list[dict] = []
    if created_after is not None and created_after.tzinfo is None:
        created_after = created_after.replace(tzinfo=UTC)

    for status in api.pull_statuses(username, since_id=since_id, created_after=created_after):
        created = datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
        if created.date() < start_date:
            break  # statuses come in reverse chronological order
        if created.date() > end_date:
            continue

        posts.append(
            {
                "id": status["id"],
                "user": username,
                **_status_content(status),
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
    max_id: str | None = None,
    stop_at_id: str | None = None,
    on_batch=None,
) -> list[dict]:
    """
    Fetch posts from Truth Social's public Mastodon-compatible API,
    newest-first with `max_id` keyset pagination, stopping when a status
    is older than `start_date` or has id <= `stop_at_id` (the cache edge).

    Works anonymously for public accounts like @realDonaldTrump. Uses
    curl_cffi to impersonate a Chrome TLS fingerprint -- plain httpx gets
    403'd at the Cloudflare edge. Truth Social IGNORES Mastodon's `min_id`
    (returns the newest page regardless; observed 2026-09-16), so there is
    no forward walk -- callers that need interruption safety use
    `max_id` to resume and `on_batch` to persist pages as they land.

    Sets `collect_via_public_api.last_complete` (bool): True when the walk
    reached its natural end, False when it was cut short (HTTP error,
    no-progress page, max_posts cap).
    """
    posts: list[dict] = []
    collect_via_public_api.last_complete = False

    resp = _ts_get_paced(f"{TS_API_BASE}/accounts/lookup", params={"acct": username})
    if resp.status_code != 200:
        logger.warning(
            "Could not look up @%s: HTTP %s %s", username, resp.status_code, resp.text[:200]
        )
        return []
    account_id = resp.json()["id"]

    cursor = max_id
    seen: set[str] = set()
    while True:
        params: dict = {"limit": 40}
        if cursor:
            params["max_id"] = cursor

        resp = _ts_get_paced(f"{TS_API_BASE}/accounts/{account_id}/statuses", params=params)
        if resp.status_code != 200:
            logger.warning(
                "Statuses fetch failed for @%s: HTTP %s -- stopping with %d posts (INCOMPLETE)",
                username,
                resp.status_code,
                len(posts),
            )
            return posts

        batch = resp.json()
        if not batch:
            break
        batch = sorted(batch, key=lambda st: int(st["id"]), reverse=True)
        if all(st["id"] in seen for st in batch):
            logger.error(
                "@%s: page made no progress (server ignored max_id?) -- stopping (INCOMPLETE)",
                username,
            )
            return posts

        page: list[dict] = []
        done = False
        for status in batch:
            seen.add(status["id"])
            if stop_at_id is not None and int(status["id"]) <= int(stop_at_id):
                done = True
                break
            created = datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
            if created.date() < start_date:
                logger.info("@%s: hit start_date boundary (%s), stopping", username, start_date)
                done = True
                break
            if created.date() <= end_date:
                page.append(
                    {
                        "id": status["id"],
                        "user": username,
                        **_status_content(status),
                        "created_at": status["created_at"],
                        "metrics": {
                            "reblogs": status.get("reblogs_count", 0),
                            "favourites": status.get("favourites_count", 0),
                            "replies": status.get("replies_count", 0),
                        },
                        "platform": "truthsocial",
                    }
                )
                if max_posts is not None and len(posts) + len(page) >= max_posts:
                    logger.info("@%s: hit max_posts cap (%d)", username, max_posts)
                    posts.extend(page)
                    if on_batch is not None and page:
                        on_batch(page)
                    return posts

        posts.extend(page)
        if on_batch is not None and page:
            on_batch(page)
        logger.info(
            "@%s: page -> %d kept (total %d, back to %s)",
            username,
            len(page),
            len(posts),
            posts[-1]["created_at"][:10] if posts else "-",
        )
        if done:
            break
        cursor = batch[-1]["id"]

    collect_via_public_api.last_complete = True
    logger.info("Public API: collected %d posts from @%s", len(posts), username)
    return posts


collect_via_public_api.last_complete = False


# ── Helpers ─────────────────────────────────────────────────────────


def _strip_html(html: str) -> str:
    """Naive HTML tag removal for Truth Social post content."""
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


# Truth Social opens a quote post's content with a fallback for clients that
# don't show quotes, "RT: <quoted status uri>", then the poster's own words if
# any. The fallback is not words, and alone (198 of Trump's 4,360 cached
# posts, 2026-10-06) it carried a post with nothing to judge past has_text's
# cut: the link goes, but "RT:" is MIN_TEXT_CHARS long.
_QUOTE_FALLBACK_RE = re.compile(r"^\s*RT:\s*https?://\S+\s*")


def _drop_bare_fallback(text: str) -> str:
    """`text` less a leading quote fallback ("RT: <uri>") when nothing with
    text follows it, so the post reads as no text; otherwise unchanged. A
    quote post with words of its own keeps the fallback before them, as the
    cache has always stored it (40 of Trump's posts)."""
    rest = _QUOTE_FALLBACK_RE.sub("", text, count=1)
    return text if rest == text or has_text(rest) else rest.strip()


def _text_of(status: dict) -> str:
    """A status's own content as text, a bare quote fallback dropped."""
    return _drop_bare_fallback(_strip_html(status.get("content") or ""))


def _quoted(status: dict) -> dict | None:
    """The status a quote post embeds (`quote`), or None. Truth Social sends
    `quote_id` with it, and `quote` null when the quoted post is gone."""
    q = status.get("quote")
    return q if isinstance(q, dict) else None


def _carried_text(status: dict) -> tuple[str, str]:
    """(acct, text) of the words a status carries: its own text, or, for a
    quote post with no text of its own (has_text), the quoted post's own
    text. One level down only."""
    own = _text_of(status)
    q = _quoted(status)
    if not has_text(own) and q is not None:
        inner = _text_of(q)
        if has_text(inner):
            return (q.get("account") or {}).get("acct", ""), inner
    return (status.get("account") or {}).get("acct", ""), own


def _status_content(status: dict) -> dict:
    """
    `text` plus the optional `reblog_of` / `quote_of` / `media` keys for a
    status record.

    A ReTruth carries no content of its own, so its text is the reblogged
    post's, prefixed "RT @acct: " like an X retweet: a ReTruth counts as the
    account's messaging (docs/decisions.md, 2026-10-04). A quote post with no
    text of its own (src/text_rules.has_text: empty, an emoji, or only
    Truth Social's "RT: <quoted uri>" fallback) is the same amplification,
    so it is stored the same way, "RT @<quoted acct>: <quoted text>", with
    `quote_of`. A quote post WITH text of its own keeps its text as sent
    (the fallback, then its words) and gains `quote_of`: the decision is
    about what an account amplifies, and its own words are its message;
    appending the quoted text would make the scorers judge the quoted
    author's words as well. The words of a ReTruth of a quote post with no
    text are the quoted post's (_carried_text).

    A ReTruth or quote of a post with no text (e.g. a lone emoji or a bare
    link) keeps its own text, empty for a ReTruth and without the fallback
    for a quote, since the prefix alone would carry it past that cut.
    `media` is the attachment count, the reblogged post's for a ReTruth;
    each key appears only when set.
    """
    out = {"text": _text_of(status)}
    reblog = status.get("reblog")
    if isinstance(reblog, dict):
        out["reblog_of"] = reblog.get("id")
        acct, inner = _carried_text(reblog)
        if not out["text"] and has_text(inner):
            out["text"] = f"RT @{acct}: {inner}"
    else:
        quote = _quoted(status)
        quote_of = status.get("quote_id") or (quote or {}).get("id")
        if quote_of:
            out["quote_of"] = str(quote_of)
            if not has_text(out["text"]) and quote is not None:
                acct, inner = _carried_text(quote)
                if has_text(inner):
                    out["text"] = f"RT @{acct}: {inner}"
    media = (
        status.get("media_attachments")
        or (reblog.get("media_attachments") if isinstance(reblog, dict) else None)
        or []
    )
    if media:
        out["media"] = len(media)
    return out


def _save_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── User timeline collection ───────────────────────────────────────


def _latest_created_at(records: list[dict]) -> datetime | None:
    stamps = []
    for r in records:
        ts = r.get("created_at")
        if not ts:
            continue
        try:
            stamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        except ValueError:
            continue
    return max(stamps) if stamps else None


def collect_user(
    handle: str,
    tier: str,
    *,
    start: date | datetime = settings.COLLECTION_START,
    end: date | datetime = settings.COLLECTION_END,
    force: bool = False,
    use_auth: bool | None = None,
) -> list[dict]:
    """
    Fetch a Truth Social account's posts into a deterministic cache.

    `use_auth` picks the incremental strategy: True = truthbrush
    (authenticated, fast), False = anonymous public API (paced, slow but
    needs no login), None = settings.TS_PREFER_AUTH and credentials present.

    Mirrors `x_collector.collect_user`:
      - force=True   : re-fetch the whole [start, end] window and merge it
                       into the cache by id, the fetched version winning.
      - force=False  : incremental -- if a non-empty cache exists, fetch
                       only from the day of the latest cached post, drop
                       ids already cached, and append. Returns the full
                       merged set (newest first, like the API).
    Each record is tagged with `tier` so downstream analysis can group
    by tier the same way it does for X data.

    Strategy: public Mastodon-compatible API (no auth, free). Accounts
    that aren't publicly exposed need truthbrush with credentials --
    call `collect_via_truthbrush` directly in that case.
    """
    cache = _account_cache_path(handle)
    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end

    # Treat an empty cache file as "not collected". An earlier failed run
    # (rate limit, Cloudflare block, etc.) leaves a zero-byte file behind,
    # and we don't want that to poison future runs.
    existing = _load_jsonl(cache) if cache.exists() and not force else []
    incremental = bool(existing)
    if incremental:
        latest = _latest_created_at(existing)
        if latest is None:
            incremental, existing = False, []
        else:
            # The page walker stops at day granularity, so start on the
            # latest cached day and dedupe the overlap by id below.
            start_date = max(start_date, latest.date())
            if latest.date() > end_date:
                logger.info(
                    "@%s: cached through %s -- already past window end, skipping",
                    handle,
                    latest.isoformat(),
                )
                return existing
            logger.info("@%s: incremental from %s (cached: %d)", handle, start_date, len(existing))
    elif cache.exists() and not force:
        logger.info("@%s: empty cache file -- treating as not collected", handle)

    if incremental:
        newest_id = max((r["id"] for r in existing if r.get("id")), key=int)
        if use_auth is None:
            use_auth = settings.TS_PREFER_AUTH and _has_truthbrush_creds()
        if use_auth:
            # Authenticated: 300 req / 5 min with the library's own backoff,
            # so a full walk back to the cache edge normally completes in
            # one go. Contiguity is verified below before anything is
            # appended (the library returns the whole list at the end).
            posts = collect_via_truthbrush(
                handle,
                start_date,
                end_date,
                since_id=newest_id,
                created_after=latest,
            )
            reached_edge = bool(posts) and min(
                datetime.fromisoformat(p["created_at"].replace("Z", "+00:00")) for p in posts
            ).date() <= latest.date() + timedelta(days=1)
            complete = reached_edge or not posts
            if posts and not reached_edge:
                logger.error(
                    "@%s: fetched %d posts but the oldest (%s) does not touch the cache "
                    "edge (%s) -- NOT appending, rerun",
                    handle,
                    len(posts),
                    min(p["created_at"] for p in posts),
                    latest.isoformat(),
                )
                return existing
        else:
            # Anonymous: walk BACKWARD from the newest post down to the cache
            # edge. Pages land in <handle>.partial.jsonl as they arrive; the
            # partial is merged into the cache only once the edge is reached,
            # so an interrupted run never leaves a hole and a rerun resumes
            # from the partial's oldest id.
            partial_path = cache.with_suffix(".partial.jsonl")
            # Drop partial posts the cache already holds: a forced or
            # authenticated run since the partial was written may have
            # fetched them, and appending them again would duplicate ids.
            cached_ids = {r.get("id") for r in existing}
            partial = [r for r in _load_jsonl(partial_path) if r.get("id") not in cached_ids]
            resume_from = (
                min((r["id"] for r in partial if r.get("id")), key=int) if partial else None
            )
            if partial:
                logger.info(
                    "@%s: resuming from partial (%d posts, oldest id %s)",
                    handle,
                    len(partial),
                    resume_from,
                )
            seen_ids = cached_ids | {r.get("id") for r in partial}

            def _flush(page: list[dict]) -> None:
                fresh = [p for p in page if p.get("id") not in seen_ids]
                for p in fresh:
                    p["tier"] = tier
                    seen_ids.add(p["id"])
                if fresh:
                    with open(partial_path, "a") as f:
                        f.writelines(json.dumps(r) + "\n" for r in fresh)
                    partial.extend(fresh)

            collect_via_public_api(
                handle,
                start_date,
                end_date,
                max_id=resume_from,
                stop_at_id=newest_id,
                on_batch=_flush,
            )
            complete = getattr(collect_via_public_api, "last_complete", True)
            if not complete:
                logger.warning(
                    "@%s [%s]: INCOMPLETE -- %d posts held in %s; rerun to continue",
                    handle,
                    tier,
                    len(partial),
                    partial_path.name,
                )
                return existing
            if partial:
                with open(cache, "a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in partial)
            # Also when every partial post was already cached (nothing new).
            partial_path.unlink(missing_ok=True)
            if partial:
                logger.info(
                    "@%s [%s]: +%d new posts (total: %d)",
                    handle,
                    tier,
                    len(partial),
                    len(existing) + len(partial),
                )
            else:
                logger.info("@%s [%s]: no new posts since last fetch", handle, tier)
            return sorted(partial + existing, key=lambda r: r["created_at"], reverse=True)
    else:
        posts = collect_via_public_api(handle, start_date, end_date)
        complete = getattr(collect_via_public_api, "last_complete", True)
    for p in posts:
        p["tier"] = tier  # annotate for downstream grouping

    if incremental:
        seen = {r.get("id") for r in existing}
        new = [p for p in posts if p.get("id") not in seen]
        if new:
            with open(cache, "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
            logger.info(
                "@%s [%s]: +%d new posts (total: %d)%s",
                handle,
                tier,
                len(new),
                len(existing) + len(new),
                "" if complete else " -- INCOMPLETE, rerun to continue",
            )
        else:
            logger.info(
                "@%s [%s]: no new posts since last fetch%s",
                handle,
                tier,
                "" if complete else " (fetch failed before any new page)",
            )
        return sorted(new + existing, key=lambda r: r["created_at"], reverse=True)

    # A forced walk must not shrink the cache it was meant to refresh: one cut
    # short (the anonymous API 429s after ~5 pages) or run over a narrower
    # --since/--until window never reaches the rest. Merge into it by id, the
    # fetched version of a post winning.
    prior = _load_jsonl(cache) if force and cache.exists() else []
    if prior:
        if not complete:
            logger.warning(
                "@%s [%s]: forced walk INCOMPLETE -- merging %d fetched posts into "
                "the %d cached instead of replacing them",
                handle,
                tier,
                len(posts),
                len(prior),
            )
        if not posts:  # cache untouched; newest first like every other return
            return sorted(prior, key=lambda r: r.get("created_at") or "", reverse=True)
        fetched = {p["id"] for p in posts}
        posts = sorted(
            posts + [r for r in prior if r.get("id") not in fetched],
            key=lambda r: r.get("created_at") or "",
            reverse=True,
        )
    elif not complete:
        logger.warning(
            "@%s [%s]: walk INCOMPLETE -- the older end of the window is missing", handle, tier
        )

    logger.info("@%s [%s]: %d posts", handle, tier, len(posts))
    if posts:
        _save_jsonl(cache, posts)
    else:
        logger.warning(
            "@%s: no posts fetched -- NOT writing cache so a retry can proceed",
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
        **_status_content(status),
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


# ── Login / security-code flow ─────────────────────────────────────

TS_OAUTH_BASE = "https://truthsocial.com"


class SecurityCodeRequired(Exception):
    """The password grant was refused pending a new-device security code."""

    def __init__(self, challenge_id: str, delivery_methods: list[dict], raw: dict):
        super().__init__("security code required")
        self.challenge_id = challenge_id
        self.delivery_methods = delivery_methods
        self.raw = raw


def _password_grant_body(username: str, password: str) -> dict:
    return {
        "client_id": settings.TRUTH_SOCIAL_WEB_CLIENT_ID,
        "client_secret": settings.TRUTH_SOCIAL_WEB_CLIENT_SECRET,
        "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
        "grant_type": "password",
        "scope": "read",
        "username": username,
        "password": password,
    }


def _auth_post(path: str, body: dict):
    return cffi_requests.post(
        f"{TS_OAUTH_BASE}{path}",
        json=body,
        impersonate="chrome136",
        headers={"Authorization": "", **settings.TS_AUTH_HEADERS},
        timeout=_DEFAULT_TIMEOUT,
    )


def request_token(username: str, password: str) -> str:
    """
    Password grant. Returns the access token, or raises
    SecurityCodeRequired with the challenge the server issued.
    """
    resp = _auth_post("/oauth/v2/token", _password_grant_body(username, password))
    if resp.status_code == 200:
        return resp.json()["access_token"]
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code == 403 and data.get("error") == "security_code_required":
        raise SecurityCodeRequired(
            data.get("challenge_id", ""),
            data.get("supported_delivery_methods", []),
            data,
        )
    raise RuntimeError(f"token exchange failed: HTTP {resp.status_code} {resp.text[:300]}")


def request_security_code_delivery(username: str, password: str, challenge_id: str, method: str):
    """
    Ask the server to send the security code via `method` (email | sms).
    Mirrors the web app's sign-in modal exactly: {username, challenge_id,
    delivery_method} and nothing else. Returns the raw response so the
    caller can show it.
    """
    body = {"username": username, "challenge_id": challenge_id, "delivery_method": method}
    return _auth_post(settings.TS_SECURITY_CODE_DELIVERY_ENDPOINT, body)


def verify_security_code(username: str, password: str, challenge_id: str, code: str) -> str:
    """Exchange challenge_id + security_code for an access token (web-app flow)."""
    body = {
        **_password_grant_body(username, password),
        "challenge_id": challenge_id,
        "security_code": code,
    }
    resp = _auth_post("/oauth/v2/verify_security_code", body)
    if resp.status_code != 200:
        raise RuntimeError(
            f"verify_security_code failed: HTTP {resp.status_code} {resp.text[:300]}"
        )
    return resp.json()["access_token"]


def save_token_to_env(token: str, env_path: Path | None = None) -> Path:
    """Persist TRUTHSOCIAL_TOKEN in .env so truthbrush reuses it (no re-login)."""
    from dotenv import set_key

    env_path = env_path or (settings.PROJECT_ROOT / ".env")
    set_key(str(env_path), "TRUTHSOCIAL_TOKEN", token, quote_mode="never")
    return env_path


def _get_truthbrush_api():
    """
    Get an authenticated truthbrush Api instance.

    Prefers TRUTHSOCIAL_TOKEN from .env (issued once by `ts-login`, which
    handles Truth Social's new-device security-code check); falls back to
    TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD, which truthbrush uses for
    a fresh password grant. Raises RuntimeError with a clear message if
    neither is available or if truthbrush isn't installed.
    """
    try:
        from truthbrush.api import Api
    except ImportError:
        raise RuntimeError(
            "truthbrush is required for reply collection. "
            "Install with: pip install 'truthbrush>=0.2'"
        )

    import os

    from dotenv import load_dotenv

    load_dotenv(override=True)

    token = os.environ.get("TRUTHSOCIAL_TOKEN")
    if token:
        return Api(token=token)
    if not os.environ.get("TRUTHSOCIAL_USERNAME") or not os.environ.get("TRUTHSOCIAL_PASSWORD"):
        raise RuntimeError(
            "Set TRUTHSOCIAL_TOKEN (run `python -m src.cli ts-login`) or "
            "TRUTHSOCIAL_USERNAME / TRUTHSOCIAL_PASSWORD in .env to collect replies."
        )
    return Api()


_TS_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)


def _get_token() -> str:
    import os

    from dotenv import load_dotenv

    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)
    token = os.environ.get("TRUTHSOCIAL_TOKEN")
    if not token:
        raise RuntimeError("No TRUTHSOCIAL_TOKEN in .env -- run `python -m src.cli ts-login`")
    return token


def _ts_get_auth_paced(url: str, params: dict | None, token: str):
    """Authenticated GET with pacing and 429 backoff (mirrors _ts_get_paced)."""
    headers = {"Authorization": f"Bearer {token}", "User-Agent": _TS_USER_AGENT}
    tries = settings.TS_MAX_RETRIES + 1
    for attempt in range(tries):
        time.sleep(settings.TS_AUTH_PAGE_DELAY_S)
        resp = cffi_requests.get(
            url, params=params, headers=headers, impersonate="chrome136", timeout=_DEFAULT_TIMEOUT
        )
        if resp.status_code != 429 or attempt + 1 == tries:
            break
        wait = _retry_after_seconds(resp)
        logger.warning("429 (attempt %d/%d) -- backing off %.0fs", attempt + 1, tries, wait)
        time.sleep(wait)
    return resp


def _next_link(resp) -> str | None:
    for link in (resp.headers.get("link") or resp.headers.get("Link") or "").split(","):
        parts = link.split(";")
        if len(parts) == 2 and parts[1].strip() == 'rel="next"':
            return parts[0].strip().strip("<>")
    return None


def iter_descendants(
    post_id: str, *, only_direct: bool = True, token: str | None = None, sort: str = "oldest"
):
    """
    Yield reply statuses to `post_id` from settings.TS_DESCENDANTS_PATH,
    following Link rel="next" until the server stops paging. With
    `only_direct`, keep only replies whose in_reply_to_id is the post
    (the NYT analysis scope) -- sub-thread replies are skipped.
    """
    token = token or _get_token()
    url = TS_OAUTH_BASE + settings.TS_DESCENDANTS_PATH.format(id=post_id)
    params: dict | None = {"sort": sort}
    page = 0
    while url:
        resp = _ts_get_auth_paced(url, params, token)
        if resp.status_code != 200:
            raise RuntimeError(
                f"descendants page {page} -> HTTP {resp.status_code}: {resp.text[:200]}"
            )
        batch = resp.json()
        if not batch:
            break
        for st in batch:
            if not only_direct or st.get("in_reply_to_id") == post_id:
                yield st
        page += 1
        url, params = _next_link(resp), None  # the next link carries its own query


def collect_replies(
    post_id: str,
    *,
    label: str | None = None,
    include_all: bool = True,
    only_direct: bool = True,
) -> list[dict]:
    """
    Fetch replies to a Truth Social post via the paginated descendants
    endpoint (settings.TS_DESCENDANTS_PATH, v2 since 2026-09) using the
    token from `ts-login`. This is the only way to get the full reply
    tree (standard `/context` is Cloudflare-blocked for unauthenticated
    clients and truncates to ~200 for authenticated ones).

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

    raw_replies: list[dict] = []
    for status in iter_descendants(post_id, only_direct=only_direct):
        raw_replies.append(status)
        if len(raw_replies) % 500 == 0:
            logger.info("  ... %d replies so far for %s", len(raw_replies), label)
        if not include_all and len(raw_replies) >= 40:
            break

    replies = [_reply_record(s, parent_id=post_id) for s in raw_replies]
    logger.info("Collected %d replies to post %s", len(replies), post_id)

    # Fetch the parent post's replies_count to report coverage
    try:
        resp = _ts_get(f"{TS_API_BASE}/statuses/{post_id}")
        if resp.status_code == 200:
            reported = resp.json().get("replies_count", 0) or 0
            pct = (100 * len(replies) / reported) if reported else 0
            logger.info(
                "Coverage: %d/%d replies (%.1f%%)",
                len(replies),
                reported,
                pct,
            )
    except Exception:
        pass  # non-critical — don't block on this

    # Write cache
    out = RAW_DIR / f"replies_{label}.jsonl"
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in replies)
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
