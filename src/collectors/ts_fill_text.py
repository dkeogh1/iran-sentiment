"""
Re-read by id the cached Truth Social posts stored with no words of their
own, and put the words they carry into the raw cache.

1,622 of Trump's 4,360 cached posts had no text (src/text_rules.has_text)
on 2026-10-05, and 198 more held only Truth Social's quote fallback, "RT:
<quoted status uri>", which has_text counts as text (the link goes, "RT:"
is MIN_TEXT_CHARS long): 1,820 to read. Read by id, a spread sample of 12
of the 1,622 held 11 image / video posts and 1 quote post with no words of
its own, stored empty because the collector ignored `quote` until
2026-10-05. A quote post with no text of its own counts like a ReTruth
(docs/decisions.md 2026-10-04), so it now carries the quoted text as
"RT @acct: ..." (truthsocial_collector._status_content), and one that quotes
a post with no text either is stored with no text, fallback dropped.

  plan    the cached posts with no words of their own (has_text, after a
          bare quote fallback is dropped) that the journal has no answer
          for (free, no request): never-read ids first, oldest first, then
          the failed reads, longest-untried first, so a few ids that always
          fail can't stop every run before it reaches the rest
  fetch   GET /api/v1/statuses/<id>, anonymous, through the collector's
          paced transport (TS_PAGE_DELAY_S before every request, 429
          backoff). Each answer is appended to the journal (fsync) before
          the next request and before the cache is touched, so a killed run
          resumes without reading a post twice. A failed read is journalled
          as `error` and retried by the next run; a run stops after
          TS_FILL_MAX_CONSECUTIVE_ERRORS in a row, or at once on a 429 that
          outlasts the backoff
  apply   journal -> raw cache: a record still holding the text that was
          checked takes the text the collector stores now, and the
          quote_of / reblog_of / media keys; every other field stays. Where
          the text changes, what was made from the old one goes first, as
          x-backfill-text does: the topic and Trump-check labels move to
          their *_superseded archives (src/superseded.py), then the post's
          row in the Trump-feed stance parquet is archived and takes the new
          text with no score, so `score-posts` rescores exactly those posts
          and `topic-label` relabels them. Each write is atomic; a rerun
          redoes only what is left

Never uses the user's account or token: Truth Social serves /statuses/:id
anonymously, and AGENTS.md lets anonymous collection run without asking.
Not for the reply caches (replies_<slug>): the journal keeps the text it
reads and goes to S3 with data/processed. Not to be run alongside
`collect-truth` on the same handle: both write its cache.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import settings
from src.collectors import truthsocial_collector as ts
from src.superseded import archive, score_columns
from src.text_rules import has_text

logger = logging.getLogger(__name__)

# What a read found: the first four are a 200 answer (classify), `gone` a
# 404 / 410 (deleted), `error` anything else, retried by the next run.
KINDS = ("media_only", "quote", "retruth", "other", "gone", "error")
_READ = {"media_only", "quote", "retruth", "other"}
_FIELDS = ("quote_of", "reblog_of", "media")
SUPERSEDED_REASON = "ts-fill-text: labelled from the text cached before the re-read"


def cache_path(handle: str) -> Path:
    if handle.startswith("replies_"):
        raise ValueError(
            f"{handle} is a reply cache: ts-fill-text reads only an account's own feed "
            "(the journal would keep private repliers' posts)"
        )
    return settings.TRUTH_SOCIAL_RAW_DIR / f"{handle}.jsonl"


def no_words(text) -> bool:
    """No words of its own to judge: has_text, after a bare quote fallback
    ("RT: <uri>" and nothing else) is dropped."""
    return not has_text(ts._drop_bare_fallback(text or ""))


# ── Journal ─────────────────────────────────────────────────────────


def journal_path() -> Path:
    return settings.PROCESSED_DIR / "ts_fill_text_journal.jsonl"


def load_journal() -> dict[str, dict]:
    """{id: latest entry}. A line a killed write cut short is skipped, so
    that read is made again."""
    path = journal_path()
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("ts-fill-text: skipping a torn journal line (that read is redone)")
            continue
        out[str(e["id"])] = e
    return out


def _ends_mid_line(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def _append_journal(entry: dict) -> None:
    """One line, flushed and fsynced before the next request. After a killed
    write the torn line is closed first, so it never swallows this one."""
    path = journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = _ends_mid_line(path)
    with open(path, "a") as f:
        f.write(("\n" if torn else "") + json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── Plan (free) ─────────────────────────────────────────────────────


def _spread(items: list, limit: int | None) -> list:
    """`limit` items evenly spaced over `items` (plan's order: oldest to
    newest, then the retries), so a pilot sees the whole window."""
    if limit is None or limit >= len(items):
        return items
    return [items[i * len(items) // limit] for i in range(limit)]


def plan(handle: str, limit: int | None = None) -> dict:
    """What a run would read, from the cache and the journal alone (no
    request): the cached posts with no words of their own (no_words) the
    journal has no answer for (an `error` is no answer). Never-read ids come
    first, oldest first, then the failed reads, longest-untried first: a
    failed read is journalled again with a new checked_at, so ids that keep
    failing go to the back and can't hold up the rest. `limit` keeps N of
    them spread over that order (a pilot)."""
    recs = ts._load_jsonl(cache_path(handle))
    bare: dict[str, dict] = {}
    for r in recs:
        pid = str(r.get("id") or "")
        if pid and pid not in bare and no_words(r.get("text")):
            bare[pid] = {"id": pid, "text": r.get("text")}
    journal = load_journal()
    never = sorted((c for c in bare.values() if c["id"] not in journal), key=lambda c: int(c["id"]))
    failed = sorted(
        (c for c in bare.values() if journal.get(c["id"], {}).get("kind") == "error"),
        key=lambda c: (journal[c["id"]].get("checked_at") or "", int(c["id"])),
    )
    left = never + failed
    todo = _spread(left, limit)
    per_read = settings.TS_PAGE_DELAY_S + settings.TS_FILL_EST_REQUEST_S
    return {
        "handle": handle,
        "cached": len(recs),
        "no_text": len(bare),
        "quote_fallback": sum(
            (c["text"] or "") != ts._drop_bare_fallback(c["text"] or "") for c in bare.values()
        ),
        "answered": len(bare) - len(left),
        "todo": todo,
        "reads": len(todo),
        "est_minutes": len(todo) * per_read / 60,
    }


# ── Fetch (free, anonymous, paced) ──────────────────────────────────


def classify(status: dict) -> str:
    """retruth (a reblog), quote (quotes a post), media_only (attachments, no
    text of its own) or other (e.g. text it has gained since, a link card)."""
    if isinstance(status.get("reblog"), dict):
        return "retruth"
    if status.get("quote_id") or isinstance(status.get("quote"), dict):
        return "quote"
    if status.get("media_attachments") and not has_text(ts._text_of(status)):
        return "media_only"
    return "other"


def _answer(pid: str, resp) -> dict:
    """The journal fields for one response: the record keys the collector
    would store now (text, quote_of / reblog_of / media) for a 200."""
    code = resp.status_code
    if code in (404, 410):
        return {"kind": "gone", "http_status": code}
    if code != 200:
        return {"kind": "error", "http_status": code}
    try:
        status = resp.json()
    except ValueError:
        return {"kind": "error", "http_status": code, "error": "not JSON"}
    if not isinstance(status, dict) or str(status.get("id")) != pid:
        return {"kind": "error", "http_status": code, "error": "not the status asked for"}
    return {"kind": classify(status), **ts._status_content(status)}


def fetch(handle: str, todo: list[dict]) -> dict:
    """
    Read each of `todo` (plan()["todo"]) by id and journal the answer before
    the next request. Returns {"kinds": Counter of this run, "requests",
    "stopped": why the run stopped early, or None}. Stops after
    settings.TS_FILL_MAX_CONSECUTIVE_ERRORS failed reads in a row, or at once
    when a 429 outlasts _ts_get_paced's TS_MAX_RETRIES backoffs.
    """
    kinds: Counter = Counter()
    streak, stopped = 0, None
    for n, c in enumerate(todo, 1):
        entry = {
            "id": c["id"],
            "user": handle,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cached_text": c["text"],
        }
        try:
            resp = ts._ts_get_paced(f"{ts.TS_API_BASE}/statuses/{c['id']}")
        except OSError as e:  # curl_cffi's RequestsError: DNS, TLS, timeout
            entry.update(kind="error", error=type(e).__name__)
        else:
            entry.update(_answer(c["id"], resp))
        _append_journal(entry)
        kinds[entry["kind"]] += 1
        if entry["kind"] != "error":
            streak = 0
        else:
            streak += 1
            why = entry.get("http_status") or entry.get("error")
            if entry.get("http_status") == 429:
                stopped = f"429 on all {settings.TS_MAX_RETRIES + 1} tries"
                break
            if streak >= settings.TS_FILL_MAX_CONSECUTIVE_ERRORS:
                stopped = f"{streak} failed reads in a row (last: {why})"
                break
        if n % 100 == 0:
            logger.info("ts-fill-text: %d/%d read: %s", n, len(todo), dict(kinds))
    return {"kinds": kinds, "requests": sum(kinds.values()), "stopped": stopped}


# ── Apply (free) ────────────────────────────────────────────────────


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Temp file, fsync, rename: a run killed mid-write leaves the old cache
    whole (as x_backfill does; not imported, to keep the X client out)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _target(rec: dict, entry: dict) -> dict:
    """The record with the journalled answer in it: the text the collector
    stores now, only where the record has no words of its own (the quoted
    text, or no text in place of a bare quote fallback); quote_of /
    reblog_of / media where the answer has them."""
    upd = {k: entry[k] for k in _FIELDS if entry.get(k) is not None}
    if no_words(rec.get("text")) and isinstance(entry.get("text"), str):
        upd["text"] = ts._drop_bare_fallback(entry["text"])
    return {**rec, **upd}


def label_files() -> list[Path]:
    """The paid label caches that hold Truth Social post ids: the Haiku topic
    labels and each Trump-feed teacher check's Opus labels."""
    from src.analysis.topic_label import labels_path

    checks = sorted(settings.PROCESSED_DIR.glob("teacher_check_trump_*.parquet"))
    return [labels_path(), *(p for p in checks if not p.stem.endswith("_superseded"))]


def supersede_labels(ids: set[str]) -> dict[str, int]:
    """Move the rows for `ids` out of every label cache (label_files) into
    its *_superseded.parquet archive, archive first, so `topic-label` and
    the Trump check see those posts as unlabelled. Returns {file: rows
    moved} for the files that had any."""
    from src.analysis.stance_local import write_parquet_atomic

    moved: dict[str, int] = {}
    for path in label_files():
        if not path.exists():
            continue
        lab = pd.read_parquet(path)
        hit = lab["id"].astype(str).isin(ids)
        if hit.any():
            archive(lab[hit], path, SUPERSEDED_REASON)
            write_parquet_atomic(lab[~hit].reset_index(drop=True), path)
            moved[path.name] = int(hit.sum())
    return moved


def clear_feed_scores(changes: dict[str, str]) -> tuple[int, int]:
    """Give the Trump-feed stance rows (settings.TRUMP_FEED_STANCE) of the
    changed posts their new text and no score, so `score-posts` rescores
    exactly those with text, `phases` skips them until it has, and a
    `topic-label` run before it sends the new text. The rows as scored go
    to its *_superseded.parquet first. Returns (rows cleared, rows
    archived)."""
    from src.analysis.stance_local import write_parquet_atomic

    path = settings.TRUMP_FEED_STANCE
    if not changes or not path.exists():
        return 0, 0
    df = pd.read_parquet(path)
    hit = df["id"].astype(str).isin(changes)
    if not hit.any():
        return 0, 0
    cols = score_columns(df)
    old = df.loc[hit, ["id", "text", *cols]]
    scored = old[cols].notna().any(axis=1) if cols else pd.Series(False, index=old.index)
    archived = archive(old[scored], path, SUPERSEDED_REASON)  # a rerun has none
    df.loc[hit, "text"] = df.loc[hit, "id"].astype(str).map(changes)
    for col in cols:
        df.loc[hit, col] = None
    write_parquet_atomic(df, path)
    return int(hit.sum()), archived


def apply(handle: str) -> dict:
    """
    Put the journalled answers for `handle` into its raw cache. A record
    changes when the journal read it (not gone, not error) and it still holds
    the text that was checked; one that already matches is counted as
    applied, and one whose text changed since (e.g. a forced collect-truth)
    is left alone and reported as drifted. Where the text changes, first the
    labels move to their archives (supersede_labels) and the feed rows are
    archived and cleared (clear_feed_scores); then one atomic rewrite of the
    cache, record order and every other field kept. A run killed between
    steps finishes on the next, which redoes only what is left.

    Also counts the quote posts read that quote a post in this same cache:
    their words count twice, once as the original and once as "RT @...",
    as a ReTruth of the account's own post does (decisions.md 2026-10-04).
    """
    journal = {i: e for i, e in load_journal().items() if e.get("user") == handle}
    read = {i: e for i, e in journal.items() if e.get("kind") in _READ}
    path = cache_path(handle)
    recs = ts._load_jsonl(path)
    ids = {str(r.get("id")) for r in recs}
    targets: dict[int, dict] = {}
    texts: dict[str, str] = {}
    already, gained, dropped, drifted = 0, 0, 0, []
    for k, r in enumerate(recs):
        e = read.get(str(r.get("id")))
        if e is None:
            continue
        t = _target(r, e)
        if t == r:
            already += 1
        elif r.get("text") == e.get("cached_text"):
            targets[k] = t
            if t.get("text") != r.get("text"):
                texts[str(r["id"])] = t["text"]
                gained += has_text(t["text"])
                dropped += not has_text(t["text"])  # a bare fallback, now no text
        else:
            drifted.append(str(r.get("id")))
    moved = supersede_labels(set(texts)) if texts else {}
    cleared, archived = clear_feed_scores(texts)
    if targets:
        for k, t in targets.items():
            recs[k] = t
        _write_jsonl_atomic(path, recs)
    if drifted:
        logger.warning(
            "ts-fill-text: %d cached posts changed since they were read; left alone: %s",
            len(drifted),
            drifted[:10],
        )
    quotes = [e for e in read.values() if e.get("quote_of")]
    own = [e for e in quotes if str(e["quote_of"]) in ids]
    return {
        "kinds": Counter(e.get("kind") for e in journal.values()),
        "with_text": sum(has_text(e.get("text")) for e in read.values()),
        "changed": len(targets),
        "text_gained": gained,
        "text_dropped": dropped,
        "already_applied": already,
        "drifted": drifted,
        "labels_moved": moved,
        "feed_rows_cleared": cleared,
        "feed_rows_archived": archived,
        "quotes": len(quotes),
        "quotes_with_text": sum(has_text(e.get("text")) for e in quotes),
        "quotes_of_own": len(own),
        "quotes_of_own_with_text": sum(has_text(e.get("text")) for e in own),
    }
