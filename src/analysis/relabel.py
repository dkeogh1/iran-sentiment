"""
Relabel the broadcaster dataset with a stronger teacher through the Message
Batches API (50% of standard pricing, results within ~1 h, kept 29 days).

  submit   build one request per post with text -> batch; state -> RELABEL_STATE
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
from src.analysis.pricing import model_price, model_tag, undated
from src.analysis.sentiment import llm_prompt, parse_llm_json
from src.superseded import drop_superseded, stamp
from src.text_rules import has_text

logger = logging.getLogger(__name__)

# Measured 2026-09-18 on 14 posts at effort=low: 217 in / 86 out per post.
EST_IN_TOKENS, EST_OUT_TOKENS = 217, 86
# Opus 5's row of settings.MODEL_PRICES; estimate_cost prices any model by its own.
PRICE_IN = settings.MODEL_PRICES["claude-opus-5"]["input"]
PRICE_OUT = settings.MODEL_PRICES["claude-opus-5"]["output"]
BATCH_DISCOUNT = settings.MODEL_PRICES["claude-opus-5"]["batch_discount"]


def _client():
    """Anthropic client with the project's .env loaded (ANTHROPIC_API_KEY)."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)
    return anthropic.Anthropic()


def tag_for(model: str) -> str:
    """Column suffix for a teacher model: claude-opus-5 -> opus,
    claude-opus-5-5 -> opus55, claude-fable-5-1 -> fable51 (pricing.model_tag:
    no two models share a column)."""
    return model_tag(model)


def columns_path() -> Path:
    """The record of the model behind each score_<tag> column merge writes,
    next to sentiment_all.parquet."""
    return settings.PROCESSED_DIR / "sentiment_all_columns.json"


def claim_column(col: str, model: str, path: Path | None = None) -> None:
    """Record `col` as `model`'s before merge writes it. Refuses a column
    recorded (or known, settings.SCORE_COLUMN_MODELS) as another model's:
    never two scorers in one series (AGENTS.md). `path` is the record of
    another frame (default: sentiment_all's, columns_path)."""
    path = path or columns_path()
    reg = json.loads(path.read_text()) if path.exists() else {}
    held = reg.get(col) or settings.SCORE_COLUMN_MODELS.get(col)
    if held is not None and undated(held) != undated(model):
        raise ValueError(
            f"{col} holds labels from {held}, not {model}: a merge never writes one "
            "model's labels into another's column (settings.MODEL_TAGS)"
        )
    if reg.get(col) != model:
        from src.analysis.stance_local import _write_json

        reg[col] = model
        _write_json(path, dict(sorted(reg.items())))  # temp file, then rename


def state_path() -> Path:
    return settings.PROCESSED_DIR / "relabel_batch.json"


def labels_path(model: str) -> Path:
    return settings.PROCESSED_DIR / f"teacher_labels_{model.replace('/', '_')}.parquet"


def estimate_cost(n: int, model: str = settings.TEACHER_CHECK_MODEL) -> float:
    """Batch dollars for n posts at `model`'s prices (settings.MODEL_PRICES),
    on the token counts measured for Opus 5 (another model's are unmeasured)."""
    p = model_price(model)
    return (
        n * (EST_IN_TOKENS * p["input"] + EST_OUT_TOKENS * p["output"]) / 1e6 * p["batch_discount"]
    )


def posts_to_label(df: pd.DataFrame, model: str, only_ids: set[str] | None = None) -> pd.DataFrame:
    """Posts with text (src/text_rules.py) not already in the teacher
    parquet, less those Haiku marked off_topic. A Haiku score is not
    required: the refresh order runs `analyze` without --llm, and a post with
    no Opus label silently drops out of `phases`. Bare links (X's image /
    video tweets) have no text: the model can't follow the link, so a label
    would be a paid 0.0."""
    d = df
    if "label_llm" in d:
        d = d[d["label_llm"] != "off_topic"]
    d = d[d["text"].map(has_text).astype(bool)]
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


def refuse_model(model: str) -> None:
    """relabel writes the stance of record's own model only
    (settings.TEACHER_CHECK_MODEL). Its estimate and its caps' ceiling rest
    on that model's measured output (settings.POST_PROMPT_TOKEN_FIT), so
    another model (a second rater) goes through `post-label --job rater`,
    which measures it with a pilot (docs/decisions.md 2026-11-02, Rejected)."""
    if undated(model) != undated(settings.TEACHER_CHECK_MODEL):
        raise ValueError(
            f"relabel labels with {settings.TEACHER_CHECK_MODEL} only (its estimate and caps "
            f"rest on that model's measured output length): label posts with {model} through "
            f"`post-label estimate|pilot|submit --job rater --model {model}`"
        )


class OverCap(ValueError):
    """A relabel submit over RELABEL_MAX_CALLS or RELABEL_MAX_USD."""


def first_by_id(posts: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """The first `limit` posts by id (X ids are numeric, in time order), so
    `--limit` parts of a larger set never overlap; all of them without one."""
    if limit is None:
        return posts
    key = posts["id"].astype(str).str.zfill(32)
    return posts.loc[key.sort_values(kind="stable").index[:limit]].reset_index(drop=True)


def estimate(posts: pd.DataFrame, model: str = settings.TEACHER_CHECK_MODEL) -> dict:
    """What a batch of these posts costs, no call: llm_prompt's usage fit
    on X posts (post_labels.llm_prompt_cost) at `model`'s batch prices.
    usd_batch at the fit's mean output a call (84 tokens), usd_batch_ceiling
    at its p95 (113) with no cache (relabel caches nothing): the figure the
    dollar cap is checked against. Also the caps and whether it is over
    them (cap_problem)."""
    from src.analysis.post_labels import llm_prompt_cost

    chars = [len(llm_prompt(r.text, r.user)) for r in posts.itertuples()]
    c = llm_prompt_cost(chars, "x", model)
    est = {
        "model": model,
        "calls": int(c["calls"]),
        "usd_batch": float(c["usd_batch"]),
        "usd_batch_ceiling": float(c["usd_batch_ceiling"]),
        "output_basis": c["output_basis"],
        "max_calls": settings.RELABEL_MAX_CALLS,
        "max_usd": settings.RELABEL_MAX_USD,
    }
    est["problem"] = cap_problem(est)
    return est


def cap_problem(est: dict) -> str | None:
    """Why a submit of `est` must not go (over RELABEL_MAX_CALLS, or its
    ceiling over RELABEL_MAX_USD), or None."""
    over = []
    if est["calls"] > settings.RELABEL_MAX_CALLS:
        over.append(
            f"{est['calls']:,} calls is over RELABEL_MAX_CALLS ({settings.RELABEL_MAX_CALLS:,})"
        )
    if est["usd_batch_ceiling"] > settings.RELABEL_MAX_USD:
        over.append(
            f"the ceiling ${est['usd_batch_ceiling']:.2f} is over RELABEL_MAX_USD "
            f"(${settings.RELABEL_MAX_USD:.2f})"
        )
    if not over:
        return None
    return (
        "; ".join(over) + ": send it in parts with --limit N, or raise the caps "
        "deliberately in config/settings.py once the user approves the estimate"
    )


def estimate_line(est: dict) -> str:
    """One line: calls, expected and ceiling dollars, the caps."""
    verdict = f"OVER CAP: {est['problem']}" if est["problem"] else "within the caps"
    return (
        f"{est['calls']:,} posts -> {est['model']}: batch ≈ ${est['usd_batch']:.2f} expected, "
        f"ceiling ${est['usd_batch_ceiling']:.2f} (output at its p95, no cache; "
        f"{est['output_basis']}); caps RELABEL_MAX_CALLS {est['max_calls']:,} / "
        f"RELABEL_MAX_USD ${est['max_usd']:.2f}: {verdict}"
    )


def submit(df: pd.DataFrame, *, model: str = settings.TEACHER_CHECK_MODEL,
           max_tokens: int = settings.TEACHER_MAX_TOKENS, effort: str = settings.TEACHER_EFFORT,
           only_ids: set[str] | None = None, limit: int | None = None) -> dict:
    """Create the batch and record its id. Refuses a model other than
    settings.TEACHER_CHECK_MODEL (refuse_model), a batch still open, a
    submit that stopped before it recorded a batch id (batch_io.guarded_create:
    the create is sent once, behind a "submitting" record), and, before any
    client is built, a set over RELABEL_MAX_CALLS or whose ceiling is over
    RELABEL_MAX_USD (OverCap). `limit` sends the first N by id (first_by_id);
    resubmit passes the failed ids as only_ids and goes through the same
    caps."""
    from src.analysis.batch_io import guarded_create, pending_message

    refuse_model(model)
    st = json.loads(state_path().read_text()) if state_path().exists() else {}
    if st.get("status") == "submitting":
        raise RuntimeError(f"relabel: {pending_message(state_path(), st)}")
    if st.get("batch_id") and st.get("status") not in (None, "ended", "collected"):
        raise RuntimeError(f"batch {st['batch_id']} is still {st['status']} -- collect it first")
    posts = first_by_id(posts_to_label(df, model, only_ids), limit)
    if posts.empty:
        logger.info("relabel: nothing to submit (all posts already labelled by %s)", model)
        return st
    est = estimate(posts, model)
    if est["problem"]:
        raise OverCap(f"relabel: {est['problem']}")
    client = _client()
    ids = posts["id"].astype(str).tolist()
    st = {"model": model, "effort": effort, "max_tokens": max_tokens,
          "n_submitted": int(len(posts)), "ids": ids, "submitted_at": stamp(),
          "limit": limit, "usd_expected_batch": est["usd_batch"],
          "usd_ceiling_batch": est["usd_batch_ceiling"],
          "caps": {"calls": settings.RELABEL_MAX_CALLS, "usd": settings.RELABEL_MAX_USD}}
    batch = guarded_create(
        client, build_requests(posts, model, max_tokens, effort), state_path(), st
    )
    st.update(batch_id=batch.id, status=batch.processing_status)
    state_path().write_text(json.dumps(st))
    logger.info("relabel: submitted batch %s with %d requests", batch.id, len(posts))
    return st


def status() -> dict:
    from src.analysis.batch_io import pending_message

    st = json.loads(state_path().read_text())
    if not st.get("batch_id"):
        raise RuntimeError(f"relabel: {pending_message(state_path(), st)}")
    b = _client().messages.batches.retrieve(st["batch_id"])
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


def collect(results_file: Path | None = None) -> dict:
    """Pull results of the recorded batch into the teacher parquet (append).
    results_file: a downloaded results JSONL. Without one the results are
    streamed, and when the stream breaks (it did on 2026-09-23 and
    2026-10-05) the batch's results_url is downloaded and read instead."""
    from src.analysis.batch_io import stream_or_download
    from src.analysis.topic_label import _batch_time, _file_results

    st = status()
    if st["status"] != "ended":
        logger.info("relabel: batch %s still %s (%s)", st["batch_id"], st["status"], st.get("counts"))
        return st
    if results_file:
        results = list(_file_results(results_file))
    else:   # the stream, or results_url downloaded when it breaks
        dest = state_path().with_name(f"relabel_batch_{st['batch_id']}_results.jsonl")
        results, _ = stream_or_download(_client(), st["batch_id"], dest, _file_results)
    rows = [parse_result(r) for r in results]
    new = pd.DataFrame(rows)
    good = new[new["score_teacher"].notna()].drop(columns=["outcome"])
    lp = labels_path(st["model"])
    good = drop_superseded(good, lp, _batch_time(st, results_file))  # text changed since submit
    if lp.exists():
        prev = pd.read_parquet(lp)
        good = pd.concat([prev[~prev["id"].isin(good["id"])], good], ignore_index=True)
    from src.analysis.stance_local import write_parquet_atomic

    write_parquet_atomic(good, lp)  # temp file, then rename: a killed write keeps the old labels
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
    claim_column(f"score_{tag}", model)
    lab = pd.read_parquet(labels_path(model)).rename(
        columns={"score_teacher": f"score_{tag}", "label_teacher": f"label_{tag}"})
    lab["id"] = lab["id"].astype(str)
    out = df.drop(columns=[c for c in (f"score_{tag}", f"label_{tag}") if c in df])
    out = out.merge(lab, on="id", how="left")
    logger.info("relabel: %d/%d posts carry score_%s", int(out[f"score_{tag}"].notna().sum()), len(out), tag)
    return out
