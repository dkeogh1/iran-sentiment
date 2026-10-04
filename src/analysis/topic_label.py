"""
Label every broadcaster post (X + Trump's Truth Social feed) as about the
Iran war or not, through the Message Batches API.

Why: the stance prompt asks for a stance "about the Iran war" on every
post, so Opus scores "Deport them!" or "Amen" from the author's leanings.
Those scores are real signal about the author but not stance on the war,
and a tier's all-post mean moves with them. `phases` uses these labels to
split each change into topic share and on-war stance. The keyword filter
(settings.WAR_TOPIC_PATTERN) is the free fallback but misses ~half the
stance-bearing war posts (no "Iran" in "Never has a modern military been
so obliterated", or in Arabic / Hebrew posts).

  estimate  posts still unlabelled and the batch cost
  submit    one request per unlabelled post; state -> topic_batch.json
  status    poll the batch
  collect   parse results into topic_labels.parquet (id, about_war)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import settings
from src.analysis.sentiment import parse_llm_json
from src.superseded import drop_superseded, stamp
from src.text_rules import has_text

logger = logging.getLogger(__name__)

# Haiku 4.5 $/MTok; ~225 input tokens per post (prompt + text, counted 2026-09-23), ~12 output.
PRICE_IN, PRICE_OUT = 1.0, 5.0
EST_IN_TOKENS, EST_OUT_TOKENS = 225, 12
BATCH_DISCOUNT = 0.5
MAX_TOKENS = 32


def topic_prompt(text: str, user: str = "") -> str:
    return (
        "Is this social media post about the 2026 Iran war? Count it as about the war "
        "if it concerns US or Israeli military action against Iran, Iranian attacks or "
        "retaliation, the ceasefire, negotiations or a deal with Iran, the Strait of "
        "Hormuz or the blockade, Iran's nuclear program, or the US political fight over "
        "the war (including attacks on people for their position on it). Posts on other "
        "topics (immigration, domestic politics, other conflicts, religion, general "
        "praise or insults) are not about the war unless they tie back to Iran. In 2026, "
        "posts about US strikes, a military victory or \"the war\" that name no country "
        "are almost always about Iran; count them.\n"
        f"The post is by @{user}.\n\n"
        f'Post: """{text}"""\n\n'
        'Respond with ONLY valid JSON: {"about_war": true} or {"about_war": false}'
    )


def state_path() -> Path:
    return settings.PROCESSED_DIR / "topic_batch.json"


def labels_path() -> Path:
    return settings.PROCESSED_DIR / "topic_labels.parquet"


def estimate_cost(n: int) -> float:
    return n * (EST_IN_TOKENS * PRICE_IN + EST_OUT_TOKENS * PRICE_OUT) / 1e6 * BATCH_DISCOUNT


def load_labels() -> pd.DataFrame | None:
    if not labels_path().exists():
        return None
    lab = pd.read_parquet(labels_path())
    lab["id"] = lab["id"].astype(str)
    return lab


def all_posts() -> pd.DataFrame:
    """Every broadcaster post with text: the X frame plus Trump's TS feed."""
    frames = [pd.read_parquet(settings.SENTIMENT_OUTPUT, columns=["id", "user", "text"])]
    if settings.TRUMP_FEED_STANCE.exists():
        frames.append(pd.read_parquet(settings.TRUMP_FEED_STANCE, columns=["id", "user", "text"]))
    d = pd.concat(frames, ignore_index=True)
    d["id"] = d["id"].astype(str)
    d = d[d["text"].map(has_text).astype(bool)]
    return d.drop_duplicates("id").reset_index(drop=True)


def posts_to_label() -> pd.DataFrame:
    d = all_posts()
    lab = load_labels()
    if lab is not None:
        d = d[~d["id"].isin(set(lab["id"]))]
    return d.reset_index(drop=True)


def _client():
    from src.analysis.relabel import _client as relabel_client
    return relabel_client()


def submit(model: str = settings.LLM_MODEL) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    st = json.loads(state_path().read_text()) if state_path().exists() else {}
    if st.get("batch_id") and st.get("status") not in (None, "collected"):
        raise RuntimeError(f"batch {st['batch_id']} is still {st['status']} -- collect it first")
    posts = posts_to_label()
    if posts.empty:
        logger.info("topic: nothing to submit")
        return st
    reqs = [Request(custom_id=r.id, params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": topic_prompt(r.text, r.user)}]))
            for r in posts.itertuples()]
    batch = _client().messages.batches.create(requests=reqs)
    st = {"batch_id": batch.id, "model": model, "n_submitted": int(len(posts)),
          "status": batch.processing_status, "submitted_at": stamp()}
    state_path().write_text(json.dumps(st))
    logger.info("topic: submitted batch %s with %d requests", batch.id, len(posts))
    return st


def status() -> dict:
    st = json.loads(state_path().read_text())
    b = _client().messages.batches.retrieve(st["batch_id"])
    if st.get("status") != "collected":
        st["status"] = b.processing_status
    st["counts"] = {k: getattr(b.request_counts, k) for k in
                    ("processing", "succeeded", "errored", "canceled", "expired")}
    state_path().write_text(json.dumps(st))
    return st


def parse_result(result) -> dict:
    out = {"id": str(result.custom_id), "about_war": None}
    if result.result.type != "succeeded":
        return out
    text = next((b.text for b in result.result.message.content if getattr(b, "type", "") == "text"), "")
    val = (parse_llm_json(text) or {}).get("about_war")
    if isinstance(val, bool):
        out["about_war"] = val
    return out


def _file_results(path: Path):
    """Results from a downloaded results JSONL, shaped like the SDK objects
    parse_result reads. The SDK stream broke mid-body on 2026-09-23 (httpx
    ReadError, twice); `curl -C -` on the batch's results_url resumed fine."""
    from types import SimpleNamespace as NS
    for line in path.read_text().splitlines():
        r = json.loads(line)
        res = r["result"]
        msg = res.get("message") or {}
        content = [NS(type=b.get("type"), text=b.get("text", "")) for b in msg.get("content", [])]
        yield NS(custom_id=r["custom_id"], result=NS(type=res["type"], message=NS(content=content)))


def _batch_time(st: dict, results_file: Path | None) -> str | None:
    """When the collected batch was submitted, at the latest: the recorded
    submit time, or a results file's mtime when that is earlier (a download
    of an older batch)."""
    t = st.get("submitted_at")
    if results_file is not None and t:
        mtime = datetime.fromtimestamp(results_file.stat().st_mtime, UTC)
        t = min(t, mtime.isoformat(timespec="seconds"))
    return t


def collect(results_file: Path | None = None) -> dict:
    st = status()
    if st["status"] not in ("ended", "collected"):
        return st
    results = (_file_results(results_file) if results_file
               else _client().messages.batches.results(st["batch_id"]))
    rows = [parse_result(r) for r in results]
    new = pd.DataFrame(rows)
    good = new[new["about_war"].notna()].copy()
    good["about_war"] = good["about_war"].astype(bool)
    good = drop_superseded(good, labels_path(), _batch_time(st, results_file))
    prev = load_labels()
    if prev is not None:
        good = pd.concat([prev[~prev["id"].isin(good["id"])], good], ignore_index=True)
    good.to_parquet(labels_path(), index=False)
    st["status"] = "collected"
    st["collected"] = {"labelled": int(new["about_war"].notna().sum()),
                       "failed": int(new["about_war"].isna().sum())}
    state_path().write_text(json.dumps(st))
    logger.info("topic: %s -> %s", st["collected"], labels_path())
    return st
