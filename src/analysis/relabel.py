"""
Relabel the broadcaster dataset with a stronger teacher through the Message
Batches API (50% of standard pricing, results within ~1 h, kept 29 days).

  submit   build one request per labelled post -> batch; state -> RELABEL_STATE
  status   poll processing_status / request_counts
  collect  read results, parse the JSON stance, write teacher_labels parquet;
           errored / expired ids are listed for a resubmit
  merge    add score_<tag> / label_<tag> columns to sentiment_all.parquet by id

The prompt is llm_prompt() -- byte-identical to the Haiku scorer's -- so the
only thing that changes between the two label sets is the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from config import settings
from src.analysis.sentiment import llm_prompt, parse_llm_json

logger = logging.getLogger(__name__)

# Measured 2026-09-18 on 14 posts at effort=low: 217 in / 86 out per post.
EST_IN_TOKENS, EST_OUT_TOKENS = 217, 86
PRICE_IN, PRICE_OUT = 5.0, 25.0        # Opus 5 $/MTok (claude-api reference table)
BATCH_DISCOUNT = 0.5


def tag_for(model: str) -> str:
    """Column suffix for a teacher model: claude-opus-5 -> opus."""
    parts = model.split("-")
    return parts[1] if len(parts) > 1 else model


def state_path() -> Path:
    return settings.PROCESSED_DIR / "relabel_batch.json"


def labels_path(model: str) -> Path:
    return settings.PROCESSED_DIR / f"teacher_labels_{model.replace('/', '_')}.parquet"


def estimate_cost(n: int) -> float:
    return n * (EST_IN_TOKENS * PRICE_IN + EST_OUT_TOKENS * PRICE_OUT) / 1e6 * BATCH_DISCOUNT


def posts_to_label(df: pd.DataFrame, model: str, only_ids: set[str] | None = None) -> pd.DataFrame:
    """Labelled, on-topic, non-empty posts not already in the teacher parquet."""
    d = df[df["score_llm"].notna()]
    if "label_llm" in d:
        d = d[d["label_llm"] != "off_topic"]
    d = d[d["text"].fillna("").str.strip().str.len() > 0]
    lp = labels_path(model)
    if lp.exists():
        have = set(pd.read_parquet(lp)["id"].astype(str))
        d = d[~d["id"].astype(str).isin(have)]
    if only_ids is not None:
        d = d[d["id"].astype(str).isin(only_ids)]
    return d[["id", "text", "user", "tier"]].reset_index(drop=True)


def build_requests(posts: pd.DataFrame, model: str, max_tokens: int, effort: str) -> list:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs = []
    for r in posts.itertuples():
        params = MessageCreateParamsNonStreaming(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": llm_prompt(r.text, r.user)}],
        )
        params["output_config"] = {"effort": effort}
        reqs.append(Request(custom_id=str(r.id), params=params))
    return reqs


def submit(df: pd.DataFrame, *, model: str = settings.TEACHER_CHECK_MODEL,
           max_tokens: int = settings.TEACHER_MAX_TOKENS, effort: str = settings.TEACHER_EFFORT,
           only_ids: set[str] | None = None) -> dict:
    """Create the batch and record its id. Refuses if a batch is still open."""
    import anthropic
    st = json.loads(state_path().read_text()) if state_path().exists() else {}
    if st.get("batch_id") and st.get("status") not in (None, "ended", "collected"):
        raise RuntimeError(f"batch {st['batch_id']} is still {st['status']} -- collect it first")
    posts = posts_to_label(df, model, only_ids)
    if posts.empty:
        logger.info("relabel: nothing to submit (all posts already labelled by %s)", model)
        return st
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=build_requests(posts, model, max_tokens, effort))
    st = {"batch_id": batch.id, "model": model, "effort": effort, "max_tokens": max_tokens,
          "n_submitted": int(len(posts)), "status": batch.processing_status,
          "ids": posts["id"].astype(str).tolist()}
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(st))
    logger.info("relabel: submitted batch %s with %d requests", batch.id, len(posts))
    return st


def status() -> dict:
    import anthropic
    st = json.loads(state_path().read_text())
    b = anthropic.Anthropic().messages.batches.retrieve(st["batch_id"])
    st["status"] = b.processing_status
    st["counts"] = {k: getattr(b.request_counts, k) for k in
                    ("processing", "succeeded", "errored", "canceled", "expired")}
    state_path().write_text(json.dumps(st))
    return st


def parse_result(result) -> dict:
    """One batch result -> {id, score_teacher, label_teacher, outcome}."""
    out = {"id": str(result.custom_id), "score_teacher": None, "label_teacher": None,
           "outcome": result.result.type}
    if result.result.type != "succeeded":
        return out
    msg = result.result.message
    text = next((b.text for b in msg.content if getattr(b, "type", "") == "text"), "")
    data = parse_llm_json(text) or {}
    sc = data.get("score")
    if isinstance(sc, (int, float)):
        out["score_teacher"] = float(sc)
        lab = data.get("label") or ("positive" if sc > 0.05 else "negative" if sc < -0.05 else "neutral")
        out["label_teacher"] = lab
    else:
        out["outcome"] = "unparseable"
    return out


def collect() -> dict:
    """Pull results of the recorded batch into the teacher parquet (append)."""
    import anthropic
    st = status()
    if st["status"] != "ended":
        logger.info("relabel: batch %s still %s (%s)", st["batch_id"], st["status"], st.get("counts"))
        return st
    client = anthropic.Anthropic()
    rows = [parse_result(r) for r in client.messages.batches.results(st["batch_id"])]
    new = pd.DataFrame(rows)
    good = new[new["score_teacher"].notna()].drop(columns=["outcome"])
    lp = labels_path(st["model"])
    if lp.exists():
        prev = pd.read_parquet(lp)
        good = pd.concat([prev[~prev["id"].isin(good["id"])], good], ignore_index=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    good.to_parquet(lp, index=False)
    bad = new[new["score_teacher"].isna()]
    st["status"] = "collected"
    st["collected"] = {"succeeded": int(len(new) - len(bad)), "failed": int(len(bad)),
                       "failed_by_outcome": bad["outcome"].value_counts().to_dict(),
                       "failed_ids": bad["id"].tolist()}
    state_path().write_text(json.dumps(st))
    logger.info("relabel: %d labels collected, %d failed -> %s", len(new) - len(bad), len(bad), lp)
    return st


def merge(df: pd.DataFrame, model: str = settings.TEACHER_CHECK_MODEL) -> pd.DataFrame:
    """Join teacher labels onto the frame as score_<tag> / label_<tag>."""
    tag = tag_for(model)
    lab = pd.read_parquet(labels_path(model)).rename(
        columns={"score_teacher": f"score_{tag}", "label_teacher": f"label_{tag}"})
    lab["id"] = lab["id"].astype(str)
    out = df.drop(columns=[c for c in (f"score_{tag}", f"label_{tag}") if c in df])
    out = out.merge(lab, on="id", how="left")
    logger.info("relabel: %d/%d posts carry score_%s", int(out[f"score_{tag}"].notna().sum()), len(out), tag)
    return out
