"""
Per-model Claude API prices and score-column tags, from config/settings.py
(MODEL_PRICES, MODEL_TAGS). Every estimator and spend tally prices a model
by its own row, and every merge names a model's columns by its own tag, so
one model is never priced, or written, as another.
"""

from __future__ import annotations

import re

from config import settings

_DATED = re.compile(r"-\d{8}$")


def undated(model: str) -> str:
    """claude-haiku-4-5-20251001 -> claude-haiku-4-5 (a pinned snapshot id)."""
    return _DATED.sub("", model)


def model_price(model: str) -> dict[str, float]:
    """The model's MODEL_PRICES row (by its id, or its id without a date
    suffix). Raises for a model with no row rather than borrowing another's."""
    row = settings.MODEL_PRICES.get(model) or settings.MODEL_PRICES.get(undated(model))
    if row is None:
        raise KeyError(
            f"no price for {model} in settings.MODEL_PRICES: add its row "
            "(platform.claude.com pricing) before pricing a run"
        )
    return row


def token_usd(tokens: dict, model: str, batch: bool = False) -> float:
    """Dollars for billed tokens keyed like the API's usage (input_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, output_tokens;
    missing keys count 0), at the model's list or batch prices."""
    p = model_price(model)
    billed_in = (
        tokens.get("input_tokens", 0)
        + tokens.get("cache_creation_input_tokens", 0) * p["cache_write_mult"]
        + tokens.get("cache_read_input_tokens", 0) * p["cache_read_mult"]
    )
    usd = (billed_in * p["input"] + tokens.get("output_tokens", 0) * p["output"]) / 1e6
    return usd * p["batch_discount"] if batch else usd


def model_tag(model: str) -> str:
    """Column suffix for a model's labels: MODEL_TAGS for the two historical
    ones (claude-opus-5 -> opus, Haiku 4.5 -> haiku), else family + version
    (claude-opus-5-5 -> opus55, claude-fable-5-1 -> fable51)."""
    base = undated(model)
    if base in settings.MODEL_TAGS:
        return settings.MODEL_TAGS[base]
    parts = base.split("-")
    if parts[0] == "claude" and len(parts) >= 3:
        return parts[1] + "".join(parts[2:])
    return re.sub(r"[^0-9A-Za-z]", "", base) or base
