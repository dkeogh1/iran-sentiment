"""
What counts as a post or reply with text to judge, shared by the collectors
and the analysis (the user's rule, 2026-10-04): fewer than
settings.MIN_TEXT_CHARS characters once links and a leading retweet prefix
are removed means no text. Such rows are not scored and leave every share
and mean rather than counting as off-topic or neutral.

Links go because an X image / video tweet is a bare t.co link and a link
share is a URL the scorers can't follow; "RT @x: " (and Truth Social's bare
"RT: <link>" quote fallback) goes because a retweet or quote of a no-text
post would otherwise carry the prefix past the cut.
"""

from __future__ import annotations

import re

from config import settings

_NOISE_RE = re.compile(r"^\s*RT(?: @\w+)?:\s*|https?://\S+")


def has_text(text) -> bool:
    return isinstance(text, str) and len(_NOISE_RE.sub("", text).strip()) >= settings.MIN_TEXT_CHARS
