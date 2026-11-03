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

A post Haiku answers without a verdict (it starts explaining that it can't
open a shared link and runs out of tokens: 692 of 22,197 on 2026-09-23, 96
of 2,666 on 2026-11-02) has that attempt recorded (topic_unjudgeable.parquet:
one row per no-verdict answer, with the batch or results file it came from
and the length of the text it was asked about). Many get a verdict when
sent again (of the four batches to 2026-11-02, 30% of the posts after one
no-verdict, 25% after two, 20% after three or more), so a post counts as
unjudgeable, and is not sent again while its text keeps that length, only
after settings.TOPIC_UNJUDGEABLE_MIN_ATTEMPTS no-verdict answers on it. It
keeps no Haiku label, so war_flag counts it by the keyword pattern as
before. An errored or expired request is not an attempt: the next submit
sends it again. A submit records each post's text length, so an answer is
put down to the text it was given, never to a text changed since.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import settings
from src.analysis.sentiment import parse_llm_json
from src.superseded import drop_superseded, stamp
from src.text_rules import has_text

logger = logging.getLogger(__name__)

# ~225 input tokens per post (prompt + text, counted 2026-09-23 on Haiku 4.5), ~12 output;
# priced at the model's row of settings.MODEL_PRICES.
EST_IN_TOKENS, EST_OUT_TOKENS = 225, 12
MAX_TOKENS = 32
# The cap check's ceiling reads each prompt at this many characters a token,
# under English's ~4, so it errs high; output at MAX_TOKENS.
CEILING_CHARS_PER_TOKEN = 2.5


class OverCap(RuntimeError):
    """A submit over TOPIC_LABEL_MAX_CALLS or TOPIC_LABEL_MAX_USD."""


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


def estimate_cost(n: int, model: str = settings.LLM_MODEL) -> float:
    """Batch dollars for n posts at `model`'s prices (settings.MODEL_PRICES)."""
    from src.analysis.pricing import model_price

    p = model_price(model)
    return (
        n * (EST_IN_TOKENS * p["input"] + EST_OUT_TOKENS * p["output"]) / 1e6 * p["batch_discount"]
    )


def ceiling_cost(posts: pd.DataFrame, model: str = settings.LLM_MODEL) -> float:
    """Batch dollars no submit of `posts` can pass: each prompt at
    CEILING_CHARS_PER_TOKEN characters a token, every answer at MAX_TOKENS."""
    from src.analysis.pricing import model_price

    p = model_price(model)
    chars = sum(len(topic_prompt(r.text, r.user)) for r in posts.itertuples())
    tokens_in = chars / CEILING_CHARS_PER_TOKEN
    tokens_out = len(posts) * MAX_TOKENS
    return (tokens_in * p["input"] + tokens_out * p["output"]) / 1e6 * p["batch_discount"]


def cap_problem(posts: pd.DataFrame, model: str = settings.LLM_MODEL) -> str | None:
    """Why a submit of `posts` is over the caps, or None."""
    n, ceiling = len(posts), ceiling_cost(posts, model)
    if n > settings.TOPIC_LABEL_MAX_CALLS:
        return (
            f"{n:,} posts is over TOPIC_LABEL_MAX_CALLS ({settings.TOPIC_LABEL_MAX_CALLS:,}): "
            "raise it deliberately in config/settings.py after the user approves the estimate"
        )
    if ceiling > settings.TOPIC_LABEL_MAX_USD + 1e-9:
        return (
            f"the ceiling ${ceiling:.2f} is over TOPIC_LABEL_MAX_USD "
            f"${settings.TOPIC_LABEL_MAX_USD:.2f}: raise it deliberately in config/settings.py "
            "after the user approves the estimate"
        )
    return None


def unjudgeable_path() -> Path:
    return settings.PROCESSED_DIR / "topic_unjudgeable.parquet"


UNJUDGEABLE_COLS = ["id", "model", "input_chars", "source", "reason", "recorded_at"]


def load_unjudgeable() -> pd.DataFrame:
    """The no-verdict answers recorded so far, one row per post and source
    (id, model, input_chars, source: the batch id or results file, reason,
    recorded_at); empty if none."""
    if not unjudgeable_path().exists():
        return pd.DataFrame(columns=UNJUDGEABLE_COLS)
    u = pd.read_parquet(unjudgeable_path())
    u["id"] = u["id"].astype(str)
    if "source" not in u:  # one row per post, before attempts were counted
        u["source"] = u.get("batch_id", pd.Series(None, index=u.index)).fillna(u["reason"])
    return u


def no_verdict_attempts(d: pd.DataFrame, model: str = settings.LLM_MODEL) -> pd.Series:
    """For the posts of d (id, text): how many recorded no-verdict answers
    `model` (dated or not) gave on the text they have now (its length)."""
    from src.analysis.pricing import undated

    u = load_unjudgeable()
    if u.empty:
        return pd.Series(0, index=d.index)
    u = u[u["model"].astype(str).map(undated) == undated(model)]
    n = u.drop_duplicates(["id", "input_chars", "source"]).groupby(["id", "input_chars"]).size()
    key = pd.MultiIndex.from_arrays([d["id"].astype(str), d["text"].str.len()])
    return pd.Series(n.reindex(key).fillna(0).astype(int).to_numpy(), index=d.index)


def skipped_unjudgeable(d: pd.DataFrame, model: str = settings.LLM_MODEL) -> pd.Series:
    """True for the posts of d (id, text) that `model` answered with no
    verdict at least TOPIC_UNJUDGEABLE_MIN_ATTEMPTS times on the text they
    have now: unjudgeable, not sent again."""
    return no_verdict_attempts(d, model) >= settings.TOPIC_UNJUDGEABLE_MIN_ATTEMPTS


def record_unjudgeable(
    ids, model: str, source: str, reason: str, submitted: dict | None = None
) -> dict:
    """Record one no-verdict answer from `source` (a batch id or a results
    file) for each post of `ids` still without a label, at the length of
    the text it was asked about: `submitted` (id -> length, recorded by the
    submit) where it has the id, else the text's length now. A post whose
    text has changed since the submit is left out (Haiku never saw this
    text). A source already recorded for a post is not counted twice.
    Returns {"recorded": answers added, "changed": left out for a changed
    text, "unjudgeable": of these posts, how many are now skipped}."""
    lab = load_labels()
    have = set(lab["id"]) if lab is not None else set()
    ids = {str(i) for i in ids} - have
    out = {"recorded": 0, "changed": 0, "unjudgeable": 0}
    if not ids:
        return out
    posts = all_posts()
    posts = posts[posts["id"].isin(ids)]
    now = posts["text"].str.len()
    asked = posts["id"].map(submitted or {})
    changed = asked.notna() & (asked != now)
    out["changed"] = int(changed.sum())
    posts, now = posts[~changed], now[~changed]
    new = pd.DataFrame(
        {
            "id": posts["id"],
            "model": model,
            "input_chars": now,
            "source": source,
            "reason": reason,
            "recorded_at": stamp(),
        }
    )
    prev = load_unjudgeable()
    both = pd.concat([prev, new], ignore_index=True) if len(prev) else new
    both = both.drop_duplicates(["id", "model", "input_chars", "source"], keep="first")
    out["recorded"] = len(both) - len(prev)
    if out["recorded"]:  # temp file, then rename: a killed write leaves the old file whole
        from src.analysis.stance_local import write_parquet_atomic

        write_parquet_atomic(both[UNJUDGEABLE_COLS], unjudgeable_path())
    out["unjudgeable"] = int(skipped_unjudgeable(posts, model).sum())
    return out


def unjudgeable_from_file(path: Path, model: str = settings.LLM_MODEL) -> dict:
    """Record the no-verdict answers of an already collected batch's
    results JSONL (`topic-label unjudgeable --results-file`): free, no API.
    The source is the batch id when the file name carries one (a collect's
    own download), else the file name; the text the batch saw is unknown,
    so each answer is put down to the post's text now. Posts labelled since
    are left out. Returns record_unjudgeable's counts."""
    path = Path(path)
    rows = pd.DataFrame(
        [parse_result(r) for r in _file_results(path)], columns=["id", "about_war", "outcome"]
    )
    m = re.search(r"(msgbatch_[A-Za-z0-9]+)", path.name)
    return record_unjudgeable(
        rows.loc[rows["outcome"] == "no_verdict", "id"],
        model,
        m.group(1) if m else path.name,
        f"answered with no about_war verdict ({path.name})",
    )


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


def posts_to_label(
    model: str = settings.LLM_MODEL, retry_unjudgeable: bool = False
) -> pd.DataFrame:
    """Posts with text and no topic label, less those recorded unjudgeable
    by `model` on the same text (unless retry_unjudgeable)."""
    d = all_posts()
    lab = load_labels()
    if lab is not None:
        d = d[~d["id"].isin(set(lab["id"]))]
    if not retry_unjudgeable:
        d = d[~skipped_unjudgeable(d, model)]
    return d.reset_index(drop=True)


def _client():
    from src.analysis.relabel import _client as relabel_client
    return relabel_client()


def submit(model: str = settings.LLM_MODEL, retry_unjudgeable: bool = False) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from src.analysis.batch_io import guarded_create, pending_message

    st = json.loads(state_path().read_text()) if state_path().exists() else {}
    if st.get("status") == "submitting":
        raise RuntimeError(f"topic: {pending_message(state_path(), st)}")
    if st.get("batch_id") and st.get("status") not in (None, "collected"):
        raise RuntimeError(f"batch {st['batch_id']} is still {st['status']} -- collect it first")
    posts = posts_to_label(model, retry_unjudgeable)
    if posts.empty:
        logger.info("topic: nothing to submit")
        return st
    problem = cap_problem(posts, model)  # before any client is built
    if problem:
        raise OverCap(problem)
    reqs = [Request(custom_id=r.id, params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": topic_prompt(r.text, r.user)}]))
            for r in posts.itertuples()]
    st = {"model": model, "n_submitted": int(len(posts)), "submitted_at": stamp()}
    # Each post's text length as sent: a no-verdict answer is recorded on it.
    st["input_chars"] = {str(i): int(n) for i, n in zip(posts["id"], posts["text"].str.len())}
    # Sent once, behind a "submitting" record (an SDK retry could make a second batch).
    batch = guarded_create(_client(), reqs, state_path(), st)
    st.update(batch_id=batch.id, status=batch.processing_status)
    state_path().write_text(json.dumps(st))
    logger.info("topic: submitted batch %s with %d requests", batch.id, len(posts))
    return st


def status() -> dict:
    from src.analysis.batch_io import pending_message

    st = json.loads(state_path().read_text())
    if not st.get("batch_id"):
        raise RuntimeError(f"topic: {pending_message(state_path(), st)}")
    b = _client().messages.batches.retrieve(st["batch_id"])
    if st.get("status") != "collected":
        st["status"] = b.processing_status
    st["counts"] = {k: getattr(b.request_counts, k) for k in
                    ("processing", "succeeded", "errored", "canceled", "expired")}
    state_path().write_text(json.dumps(st))
    return st


def parse_result(result) -> dict:
    """{id, about_war (None when not judged), outcome}: outcome is the API's
    for a request that did not succeed, "no_verdict" for an answer with no
    true / false (an unjudgeable post), "succeeded" otherwise."""
    out = {"id": str(result.custom_id), "about_war": None, "outcome": result.result.type}
    if result.result.type != "succeeded":
        return out
    text = next((b.text for b in result.result.message.content if getattr(b, "type", "") == "text"), "")
    val = (parse_llm_json(text) or {}).get("about_war")
    if isinstance(val, bool):
        out["about_war"] = val
    else:
        out["outcome"] = "no_verdict"
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
    from src.analysis.batch_io import stream_or_download
    from src.analysis.stance_local import write_parquet_atomic

    st = status()
    if st["status"] not in ("ended", "collected"):
        return st
    if results_file:
        results = list(_file_results(results_file))
    else:  # the stream, or results_url downloaded when it breaks
        dest = state_path().with_name(f"topic_batch_{st['batch_id']}_results.jsonl")
        results, _ = stream_or_download(_client(), st["batch_id"], dest, _file_results)
    rows = [parse_result(r) for r in results]
    new = pd.DataFrame(rows, columns=["id", "about_war", "outcome"])
    good = new[new["about_war"].notna()].drop(columns="outcome").copy()
    good["about_war"] = good["about_war"].astype(bool)
    good = drop_superseded(good, labels_path(), _batch_time(st, results_file))
    prev = load_labels()
    if prev is not None:
        good = pd.concat([prev[~prev["id"].isin(good["id"])], good], ignore_index=True)
    write_parquet_atomic(good, labels_path())  # a killed write leaves the old labels whole
    unjudged = record_unjudgeable(
        new.loc[new["outcome"] == "no_verdict", "id"],
        st.get("model", settings.LLM_MODEL),
        str(st.get("batch_id")),
        "answered with no about_war verdict",
        submitted=st.get("input_chars"),
    )
    st["status"] = "collected"
    st["collected"] = {
        "labelled": int(new["about_war"].notna().sum()),
        "failed": int(new["about_war"].isna().sum()),
        "no_verdict": int((new["outcome"] == "no_verdict").sum()),
        "unjudgeable": unjudged["unjudgeable"],
    }
    state_path().write_text(json.dumps(st))
    logger.info("topic: %s -> %s", st["collected"], labels_path())
    return st
