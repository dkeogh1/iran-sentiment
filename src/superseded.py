"""
Archives of the paid labels and scores made from a post text that has since
changed.

When a post's text changes (x-backfill-text puts back the full text of a
post cached cut at 280 characters; a re-fetch that `analyze` then
rescores), what was made from the old text is no longer the post's label.
It was paid for, and other results are paired with it (the teacher check,
`teacher-retest`), so it moves to <cache>_superseded.parquet next to its
cache: appended with superseded_at and superseded_reason, never deleted.

No tweepy or model imports: the GPU image reads these archives too
(stance_local.training_frame).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# Scores that are free to redo (local models); every other score_* column
# was paid for.
FREE_SCORES = ("score_vader", "score_transformer")


def superseded_path(path: Path) -> Path:
    """teacher_labels_<model>.parquet -> teacher_labels_<model>_superseded.parquet"""
    return path.with_name(path.stem + "_superseded.parquet")


def stamp() -> str:
    """UTC to the second, the format of superseded_at and a batch's
    submitted_at, so the two compare as strings."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def score_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("score_", "label_"))]


def archive(rows: pd.DataFrame, path: Path, reason: str) -> int:
    """
    Append `rows` to the archive of the cache at `path`, with superseded_at
    and superseded_reason; written atomically. A row archived before with
    the same reason and the same values (a killed run, rerun) is not added
    again; any other row is, even for an id archived before, so a later
    label is never dropped. Returns the rows added.
    """
    from src.analysis.stance_local import write_parquet_atomic

    if rows.empty:
        return 0
    new = rows.assign(id=rows["id"].astype(str), superseded_at=stamp(), superseded_reason=reason)
    arch = superseded_path(path)
    prev = pd.read_parquet(arch) if arch.exists() else new.iloc[0:0]
    both = pd.concat([prev, new], ignore_index=True)
    dup = both.duplicated(subset=[c for c in both.columns if c != "superseded_at"])
    keep = ~dup | (both.index < len(prev))  # the archive itself is never trimmed
    added = int(keep.sum()) - len(prev)
    if added:
        write_parquet_atomic(both[keep].reset_index(drop=True), arch)
    return added


def drop_superseded(rows: pd.DataFrame, path: Path, submitted_at: str | None) -> pd.DataFrame:
    """
    Batch results minus the posts whose label in the cache at `path` was
    superseded after the batch was submitted: the batch saw the old text,
    and collecting it again (`relabel status` + `collect`, or `topic-label
    collect --results-file` on an old download) must not put that label
    back. `submitted_at` None is a batch from before submit times were
    recorded (2026-10-04), older than every archive.
    """
    arch = superseded_path(path)
    if rows.empty or not arch.exists():
        return rows
    a = pd.read_parquet(arch, columns=["id", "superseded_at"])
    if submitted_at:
        a = a[a["superseded_at"] > submitted_at]
    stale = rows["id"].astype(str).isin(set(a["id"].astype(str)))
    if stale.any():
        logger.warning(
            "%d results are for posts whose text changed after this batch was submitted; "
            "not collected (their labels from the old text are in %s)",
            int(stale.sum()),
            arch.name,
        )
    return rows[~stale]


def with_archived_scores(df: pd.DataFrame, col: str = "score_llm") -> pd.DataFrame:
    """
    `df` (sentiment_all) with each row that has no `col` but an archived
    version that has one put back as it was scored: its text and every
    score_* / label_* column from its first archived row (the text the
    earliest paid labels saw). Keeps a frame of Haiku-labelled posts, and the
    teacher check's seeded sample drawn from it, the same after
    x-backfill-text, with each label next to the text it was made from.
    """
    arch_path = superseded_path(settings.SENTIMENT_OUTPUT)
    if col not in df or not df[col].isna().any() or not arch_path.exists():
        return df
    arch = pd.read_parquet(arch_path)
    if col not in arch:
        return df
    arch = arch[arch[col].notna()].assign(id=lambda a: a["id"].astype(str))
    arch = arch.drop_duplicates("id").set_index("id")
    ids = df["id"].astype(str)
    need = df[col].isna() & ids.isin(arch.index)
    if not need.any():
        return df
    out = df.copy()
    old = arch.reindex(ids[need])
    for c in ["text", *score_columns(df)]:
        if c not in old:  # a column added since: nothing was made from this text
            out.loc[need, c] = None
            continue
        if old[c].dtype == object and out[c].dtype != object:  # labels into an all-NaN column
            out[c] = out[c].astype(object)
        out.loc[need, c] = old[c].to_numpy()
    return out
