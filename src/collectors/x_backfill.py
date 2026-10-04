"""
Backfill the full text of cached X posts that were stored cut at 280
characters.

Until 2026-10-04 the collector did not ask for `note_tweet`, so a post over
280 characters was cached as X's `text`: its first 280 characters, cut at a
word, then its links. Its scores, Opus label and topic label were made from
that cut.

  candidates  likely-cut originals, from the cache alone (no API call)
  plan        the candidates not yet checked, priced at X_READ_COST_USD
  lookup      GET /2/tweets by id, 100 per request; each batch is journalled
              (atomically) before the next request and before any cache is
              touched, so a killed run never buys a read twice and a checked
              id is never read again
  apply       journal -> raw cache text. First the labels made from the cut
              text move to *_superseded.parquet archives (append; paid data
              is never deleted) and the post's sentiment_all row loses its
              scores, then its raw records are rewritten, so `analyze`,
              `relabel` and `topic-label` redo exactly those posts
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import settings
from src.collectors import x_collector as xc

logger = logging.getLogger(__name__)

X_CUT_CHARS = 280
SUPERSEDED_REASON = "x-backfill-text: labelled from the text cut at 280 characters"

_REPLY_MENTIONS = re.compile(r"^(?:@\w+\s+)+")
_TRAILING_LINKS = re.compile(r"(?:\s*https?://\S+)+\s*$")
_LINK = re.compile(r"https?://\S+")
_MID_SENTENCE = ",;:-–—&/(“‘"


# ── Candidates (free) ──────────────────────────────────────────────


def _body(text: str) -> str:
    """The text X's cut counts: HTML entities unescaped (the API sends &amp;),
    without a reply's leading @mentions or the links after the last word."""
    t = _REPLY_MENTIONS.sub("", html.unescape(text))
    return _TRAILING_LINKS.sub("", t).rstrip()


def cut_length(text: str) -> int:
    """
    Length of a post as the 280-character cut measured it, with every other
    link counted as X counts it (23, a t.co link). Code points, not X's
    weighting (emoji and CJK count 2): in the cache the cut sits at 280 code
    points, where only 1 of the 12,154 originals runs over, while ~70 run
    over 280 by the weighting. A post that opens with a mention but is not a
    reply is measured short by its mentions (the cache can't tell).
    """
    return len(_LINK.sub("x" * 23, _body(text or "")))


def ends_mid_sentence(text: str) -> bool:
    """The last character before the trailing links is a letter, a digit or
    mid-sentence punctuation."""
    b = _body(text or "")
    return bool(b) and (b[-1].isalnum() or b[-1] in _MID_SENTENCE)


def likely_cut(text) -> bool:
    """An original (not "RT @") whose cut length is in the pile-up below 280
    (settings.X_BACKFILL_MIN_CHARS / X_BACKFILL_OPEN_MIN_CHARS)."""
    if not isinstance(text, str) or text.startswith("RT @"):
        return False
    n = cut_length(text)
    if n > X_CUT_CHARS:
        return False
    if n >= settings.X_BACKFILL_MIN_CHARS:
        return True
    return n >= settings.X_BACKFILL_OPEN_MIN_CHARS and ends_mid_sentence(text)


def _created(rec: dict) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(rec.get("created_at")))  # 3.11+ reads "Z"
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def candidates() -> list[dict]:
    """Account posts (keyword-search rows are not) created before
    settings.X_CUT_TEXT_BEFORE whose cached text is likely_cut, one per id,
    oldest first."""
    out: dict[str, dict] = {}
    for path in sorted(settings.X_RAW_DIR.glob("*.jsonl")):
        if path.stem.startswith("search_"):
            continue
        for r in xc._load_jsonl(path):
            pid = str(r.get("id"))
            ts = _created(r)
            if r.get("tier") == "search" or pid in out or ts is None:
                continue
            if ts >= settings.X_CUT_TEXT_BEFORE or not likely_cut(r.get("text")):
                continue
            out[pid] = {
                "id": pid,
                "user": r.get("user"),
                "tier": r.get("tier"),
                "created_at": r.get("created_at"),
                "text": r["text"],
            }
    return sorted(out.values(), key=lambda c: int(c["id"]))


def plan(limit: int | None = None) -> dict:
    """What a run would read, from the cache and the journal alone (no API
    call): {candidates, checked, todo, reads, max_cost_usd}. `limit` keeps the
    first N still to read (a pilot)."""
    cands = candidates()
    done = load_journal()
    todo = [c for c in cands if c["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    return {
        "candidates": cands,
        "checked": sum(c["id"] in done for c in cands),
        "todo": todo,
        "reads": len(todo),
        "max_cost_usd": len(todo) * settings.X_READ_COST_USD,
    }


def tier_table(p: dict, topic_source: str = settings.TOPIC_SOURCE) -> pd.DataFrame:
    """Per tier and account: candidates, how many are war posts (inference.
    war_flag on the cached text, so the Haiku label where there is one) and
    how many this run reads."""
    from src.analysis.inference import war_flag

    c = pd.DataFrame(p["candidates"], columns=["id", "user", "tier", "created_at", "text"])
    if c.empty:
        return pd.DataFrame(columns=["tier", "user", "candidates", "war", "to_read"])
    c["war"] = war_flag(c, topic_source).astype(int).to_numpy()
    c["to_read"] = c["id"].isin({t["id"] for t in p["todo"]}).astype(int)
    return (
        c.groupby(["tier", "user"])
        .agg(candidates=("id", "size"), war=("war", "sum"), to_read=("to_read", "sum"))
        .reset_index()
    )


# ── Journal ─────────────────────────────────────────────────────────


def journal_path() -> Path:
    return settings.PROCESSED_DIR / "x_backfill_journal.jsonl"


def load_journal() -> dict[str, dict]:
    """{id: entry} for every post looked up so far, found or not."""
    path = journal_path()
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            out[str(e["id"])] = e
    return out


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Temp file, fsync, rename: a run killed mid-write leaves the old file whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_journal(entries: list[dict]) -> None:
    j = load_journal()
    j.update({e["id"]: e for e in entries})
    _write_jsonl_atomic(journal_path(), list(j.values()))


# ── Lookup (paid) ───────────────────────────────────────────────────


def lookup(client, todo: list[dict], batch_size: int | None = None) -> dict:
    """
    Read `todo` (plan()["todo"]) by id with the timeline's tweet fields
    (note_tweet among them) and journal each batch before the next request.
    An entry records the cached text it was checked against and, for a long
    post, the full text (x_collector._full_text). A post X does not return
    (deleted, protected) is journalled as missing with X's error title, so it
    is not asked for again either. Returns counts; posts returned are the
    reads X bills.
    """
    size = batch_size or settings.X_LOOKUP_BATCH
    stats = {"requests": 0, "returned": 0, "long": 0, "missing": 0}
    for k in range(0, len(todo), size):
        chunk = todo[k : k + size]
        resp = client.get_tweets(ids=[c["id"] for c in chunk], tweet_fields=xc._TWEET_FIELDS)
        stats["requests"] += 1
        got = {str(t.id): t for t in (resp.data or [])}
        errors = {str(e.get("resource_id") or e.get("value")): e for e in (resp.errors or [])}
        now = datetime.now(UTC).isoformat(timespec="seconds")
        entries = []
        for c in chunk:
            e = {"id": c["id"], "user": c["user"], "checked_at": now, "cached_text": c["text"]}
            t = got.get(c["id"])
            if t is None:
                err = errors.get(c["id"], {})
                e.update(status="missing", error=err.get("title") or "not returned")
                stats["missing"] += 1
            else:
                note = ((t.data or {}).get("note_tweet") or {}).get("text")
                e.update(status="found", full_text=xc._full_text(t) if note else None)
                stats["returned"] += 1
                stats["long"] += bool(note)
            entries.append(e)
        append_journal(entries)
        logger.info(
            "x-backfill: batch %d: %d returned, %d long, %d missing (run so far: %d read)",
            stats["requests"],
            len(got),
            sum(1 for e in entries if e.get("full_text")),
            sum(1 for e in entries if e["status"] == "missing"),
            stats["returned"],
        )
    return stats


# ── Apply (free) ────────────────────────────────────────────────────


def superseded_path(path: Path) -> Path:
    """teacher_labels_<model>.parquet -> teacher_labels_<model>_superseded.parquet"""
    return path.with_name(path.stem + "_superseded.parquet")


def post_label_files() -> list[Path]:
    """The paid label caches keyed by X post id: each teacher model's batch
    relabel (not the reply labels) and the Haiku topic labels."""
    from src.analysis.topic_label import labels_path as topic_labels_path

    teacher = [
        p
        for p in sorted(settings.PROCESSED_DIR.glob("teacher_labels_*.parquet"))
        if not p.name.startswith("teacher_labels_replies") and not p.stem.endswith("_superseded")
    ]
    return [*teacher, topic_labels_path()]


def supersede_labels(ids: set[str], reason: str = SUPERSEDED_REASON) -> dict[str, int]:
    """
    Move the rows for `ids` out of every post label cache into its
    *_superseded.parquet archive (appended, with superseded_at and
    superseded_reason), so `relabel` and `topic-label` see those posts as
    unlabelled. The archive is written first; rows a killed run left in both
    files are not archived twice. Returns {file name: rows moved}.
    """
    from src.analysis.stance_local import write_parquet_atomic

    moved: dict[str, int] = {}
    for path in post_label_files():
        if not path.exists():
            continue
        lab = pd.read_parquet(path)
        hit = lab["id"].astype(str).isin(ids)
        moved[path.name] = int(hit.sum())
        if not hit.any():
            continue
        out = lab[hit].copy()
        out["superseded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        out["superseded_reason"] = reason
        arch = superseded_path(path)
        if arch.exists():
            prev = pd.read_parquet(arch)
            again = set(prev.loc[prev["superseded_reason"] == reason, "id"].astype(str))
            out = pd.concat([prev, out[~out["id"].astype(str).isin(again)]], ignore_index=True)
        write_parquet_atomic(out, arch)
        write_parquet_atomic(lab[~hit].reset_index(drop=True), path)
        logger.info("x-backfill: %d rows %s -> %s", moved[path.name], path.name, arch.name)
    return moved


def clear_scores(changes: dict[str, str]) -> int:
    """Give the sentiment_all rows of the changed posts their full text and
    drop every score_* / label_* made from the cut, so a `relabel` or
    `topic-label` run before `analyze` sends the full text, and `phases` skips
    them until they are relabelled. Returns rows changed."""
    from src.analysis.stance_local import write_parquet_atomic

    path = settings.SENTIMENT_OUTPUT
    if not path.exists() or not changes:
        return 0
    df = pd.read_parquet(path)
    hit = df["id"].astype(str).isin(changes)
    if not hit.any():
        return 0
    df.loc[hit, "text"] = df.loc[hit, "id"].astype(str).map(changes)
    for col in df.columns:
        if col.startswith(("score_", "label_")):
            df.loc[hit, col] = None
    write_parquet_atomic(df, path)
    return int(hit.sum())


def apply() -> dict:
    """
    Put the journalled full text into the raw cache. A post changes when the
    journal has its full text and a cached record still holds the text the
    journal checked; then, in order, its labels move to the archives
    (supersede_labels), its sentiment_all row is cleared (clear_scores) and
    every raw X file holding it is rewritten with only `text` changed. Each
    write is atomic and a rerun redoes only what is left, so a run killed
    between steps finishes on the next. A record whose text matches neither
    is left alone and reported (`drifted`).
    """
    journal = load_journal()
    full = {i: e["full_text"] for i, e in journal.items() if e.get("full_text")}
    files: dict[Path, list[dict]] = {}
    changes: dict[str, str] = {}
    done: set[str] = set()
    drifted: set[str] = set()
    for path in sorted(settings.X_RAW_DIR.glob("*.jsonl")):
        recs = xc._load_jsonl(path)
        for r in recs:
            pid = str(r.get("id"))
            if pid not in full:
                continue
            if r.get("text") == full[pid]:
                done.add(pid)
            elif r.get("text") == journal[pid].get("cached_text"):
                changes[pid] = full[pid]
                files[path] = recs
            else:
                drifted.add(pid)

    moved = supersede_labels(set(changes)) if changes else {}
    cleared = clear_scores(changes)
    for path, recs in files.items():
        for r in recs:
            pid = str(r.get("id"))
            if pid in changes and r.get("text") == journal[pid].get("cached_text"):
                r["text"] = changes[pid]
        _write_jsonl_atomic(path, recs)
    if drifted:
        logger.warning(
            "x-backfill: %d cached posts hold neither the checked nor the full text; "
            "left alone: %s",
            len(drifted),
            sorted(drifted)[:10],
        )
    return {
        "checked": len(journal),
        "missing": sum(1 for e in journal.values() if e.get("status") == "missing"),
        "long": len(full),
        "changed": len(changes),
        "already_applied": len(done - set(changes)),
        "drifted": sorted(drifted),
        "files_rewritten": len(files),
        "labels_moved": moved,
        "sentiment_rows_cleared": cleared,
    }


def relabel_cost(model: str = settings.TEACHER_CHECK_MODEL) -> tuple[int, float]:
    """(posts, batch $) for the Opus relabel still owed on backfilled posts
    (full text journalled, no label in the cache), priced by prompt length:
    input tokens = characters / settings.TEACHER_EST_CHARS_PER_TOKEN, which
    errs high. `relabel estimate` assumes the average cached post, and a
    backfilled post is longer. No API call."""
    from src.analysis import relabel as rl
    from src.analysis.sentiment import llm_prompt

    lp = rl.labels_path(model)
    have = set(pd.read_parquet(lp, columns=["id"])["id"].astype(str)) if lp.exists() else set()
    todo = [e for i, e in load_journal().items() if e.get("full_text") and i not in have]
    chars = sum(len(llm_prompt(e["full_text"], e["user"])) for e in todo)
    tokens_in = chars / settings.TEACHER_EST_CHARS_PER_TOKEN
    usd = (tokens_in * rl.PRICE_IN + len(todo) * rl.EST_OUT_TOKENS * rl.PRICE_OUT) / 1e6
    return len(todo), usd * rl.BATCH_DISCOUNT
