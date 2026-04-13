"""
Specific Trump Truth Social posts to analyze at the reply level.

These are the posts the NYT analysis hinged on: the Easter rant, the
"whole civilization will die" post, and the ceasefire announcement.
We treat each one as a discrete event — an observation unit for
pre/post sentiment comparison and for replicating the NYT's
critical/supportive/neutral breakdown.

Two ways to identify a post:
  1. `post_id` — the Truth Social status ID (most reliable, doesn't
     drift if Trump edits or the NYT miscites). Look these up once by
     hand or with `match_by_keyword` below and pin them here.
  2. `match_keyword` — a substring that uniquely identifies the post
     in the cached realDonaldTrump.jsonl file. Used as a fallback when
     `post_id` is None, so you can bootstrap this file without
     manually copying IDs.

Add new posts by editing this file — never hardcode IDs in scripts.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedPost:
    slug: str                   # short identifier used in filenames / CLI
    label: str                  # human-readable label for plots
    event_date: date            # date of the post (for event-window analysis)
    category: str               # escalation | rhetoric | ceasefire | epstein
    description: str
    match_keyword: str | None = None  # substring to find in cached Trump posts
    post_id: str | None = None        # explicit Truth Social status ID (preferred)


# ── The posts that drive the NYT thesis ────────────────────────────
# Dates and phrasing come from the NYT April 8 2026 piece and the
# project timeline (config/timeline.py). Fill in `post_id` as you
# confirm them against the cached realDonaldTrump.jsonl.

TRACKED_POSTS: list[TrackedPost] = [
    TrackedPost(
        slug="armada",
        label="'Armada heading to Iran'",
        event_date=date(2026, 1, 28),
        category="escalation",
        description="Trump declares a 'massive Armada is heading to Iran'.",
        match_keyword="Armada",
    ),
    TrackedPost(
        slug="power_plant_day",
        label="'Power Plant Day' / 'Praise be to Allah'",
        event_date=date(2026, 4, 5),
        category="rhetoric",
        description="Easter Sunday expletive-filled post threatening power "
                    "plants and bridges and ending with the mocking "
                    "'Praise be to Allah' signoff. A single post — both "
                    "phrases the NYT highlighted appear in the same status. "
                    "Verified id=116351998782539414 in the collected cache.",
        match_keyword="Power Plant Day",
    ),
    TrackedPost(
        slug="civilisation_dies",
        label="'Whole civilisation will die'",
        event_date=date(2026, 4, 7),
        category="rhetoric",
        description="Pre-ceasefire threat: 'A whole civilisation will die "
                    "tonight'. NYT analysed >40k replies to this post.",
        match_keyword="whole civili",  # matches civilisation or civilization
    ),
    TrackedPost(
        slug="ceasefire",
        label="Two-week ceasefire announcement",
        event_date=date(2026, 4, 7),
        category="ceasefire",
        description="Hours after the 'civilisation' post, Trump announces a "
                    "two-week ceasefire — the climb-down the NYT flagged.",
        match_keyword="ceasefire",
    ),
    # Control post used by the NYT for comparison. Not about Iran — lets us
    # check whether base anger is Iran-specific or a baseline grumble.
    TrackedPost(
        slug="epstein_hoax",
        label="'Jeffrey Epstein Hoax' (control)",
        event_date=date(2025, 7, 1),  # approximate; update on real match
        category="epstein",
        description="Control post: Trump calling the Epstein files a "
                    "'Jeffrey Epstein Hoax'. NYT reports this split ~1/3 "
                    "critical, ~1/3 support, ~1/3 neutral — much less "
                    "critical than the Iran posts. Used as a baseline.",
        match_keyword="Jeffrey Epstein Hoax",
    ),
]


def match_by_keyword(
    keyword: str,
    cache_path: Path,
) -> list[dict]:
    """
    Return all posts in a cached Trump JSONL whose text contains `keyword`
    (case-insensitive). Useful for resolving `post_id` when only a phrase
    is known.
    """
    if not cache_path.exists():
        logger.warning("Cache file not found: %s", cache_path)
        return []

    kw = keyword.lower()
    hits: list[dict] = []
    with open(cache_path) as f:
        for line in f:
            try:
                post = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kw in post.get("text", "").lower():
                hits.append(post)
    return hits


def resolve_post_ids(
    tracked: list[TrackedPost] | None = None,
    cache_path: Path | None = None,
) -> dict[str, str | None]:
    """
    For each TrackedPost without an explicit post_id, try to find a
    matching post in the cached realDonaldTrump.jsonl. Returns a
    {slug: post_id or None} dict. Logs ambiguity (multiple matches).
    """
    from config import settings

    tracked = tracked or TRACKED_POSTS
    cache_path = cache_path or (settings.TRUTH_SOCIAL_RAW_DIR / "realDonaldTrump.jsonl")

    resolved: dict[str, str | None] = {}
    for tp in tracked:
        if tp.post_id:
            resolved[tp.slug] = tp.post_id
            continue
        if not tp.match_keyword:
            resolved[tp.slug] = None
            continue

        hits = match_by_keyword(tp.match_keyword, cache_path)
        if not hits:
            logger.warning("No match for %s (keyword=%r)", tp.slug, tp.match_keyword)
            resolved[tp.slug] = None
        elif len(hits) > 1:
            # Pick the one closest to event_date; log the ambiguity
            from datetime import datetime
            def _dist(post: dict) -> int:
                try:
                    d = datetime.fromisoformat(
                        post["created_at"].replace("Z", "+00:00")
                    ).date()
                    return abs((d - tp.event_date).days)
                except Exception:
                    return 10**9
            best = min(hits, key=_dist)
            logger.info(
                "%s: %d matches for %r, picking id=%s (closest to %s)",
                tp.slug, len(hits), tp.match_keyword, best["id"], tp.event_date,
            )
            resolved[tp.slug] = best["id"]
        else:
            resolved[tp.slug] = hits[0]["id"]

    return resolved
