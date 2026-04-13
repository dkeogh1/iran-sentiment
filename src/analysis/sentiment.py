"""
Sentiment analysis pipeline.

Three scoring strategies, run in order of cost/accuracy tradeoff:
  1. VADER       - fast rule-based baseline (always on)
  2. RoBERTa     - HuggingFace cardiffnlp/twitter-roberta-base-sentiment-latest
  3. Claude LLM  - optional, context-aware scoring

Guardrails (the mini PC OOM'd once already):
  - VADER checkpoints to disk before RoBERTa loads, so a crash in Phase 2
    doesn't lose Phase 1 work
  - RoBERTa runs in small batches (see settings.ROBERTA_BATCH_SIZE)
  - Model is explicitly released after use
  - Memory usage is printed between phases
"""

import gc
import json
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import settings

logger = logging.getLogger(__name__)


# ── Memory monitoring ──────────────────────────────────────────────

def mem_usage_mb() -> float:
    """Current process RSS in MB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0.0
    return 0.0


def _log_mem(label: str) -> None:
    logger.info("[mem] %s: %.0f MB RSS", label, mem_usage_mb())


# ── Phase 1: VADER ─────────────────────────────────────────────────

def score_vader_inplace(posts: list[dict]) -> None:
    """Add score_vader and label_vader fields to each post in place."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    for post in tqdm(posts, desc="VADER"):
        text = post.get("text", "")
        scores = analyzer.polarity_scores(text)
        post["score_vader"] = scores["compound"]
        if scores["compound"] >= 0.05:
            post["label_vader"] = "positive"
        elif scores["compound"] <= -0.05:
            post["label_vader"] = "negative"
        else:
            post["label_vader"] = "neutral"

    del analyzer
    gc.collect()


# ── Phase 2: RoBERTa ───────────────────────────────────────────────

_LABEL_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def score_roberta_inplace(
    posts: list[dict],
    batch_size: int = settings.ROBERTA_BATCH_SIZE,
) -> None:
    """
    Add score_transformer and label_transformer fields to each post.

    Runs in batches to bound peak memory. Inference is CPU-only.

    Note: score_transformer is a weighted average across all label
    probabilities (neg=-1, neu=0, pos=+1) while label_transformer is
    the argmax (highest-confidence label). These can disagree — e.g.
    a post scored [pos:0.4, neu:0.6] yields score +0.4 but label
    "neutral". The weighted score is more informative for aggregation;
    the label is used for categorical breakdowns.
    """
    from transformers import pipeline

    _log_mem("before model load")
    pipe = pipeline(
        "sentiment-analysis",
        model=settings.ROBERTA_MODEL,
        tokenizer=settings.ROBERTA_MODEL,
        top_k=None,
        truncation=True,
        max_length=512,
        device="cpu",
    )
    _log_mem("after model load")

    # Empty texts get an empty string — scored separately below
    texts = [p.get("text", "") or "" for p in posts]
    empty_mask = [t == "" for t in texts]

    num_batches = (len(texts) + batch_size - 1) // batch_size
    for bi in tqdm(range(num_batches), desc="RoBERTa"):
        start = bi * batch_size
        end = min(start + batch_size, len(texts))

        try:
            results = pipe(texts[start:end])
            for i, result in enumerate(results):
                weighted = sum(
                    r["score"] * _LABEL_MAP.get(r["label"], 0) for r in result
                )
                top = max(result, key=lambda r: r["score"])
                posts[start + i]["score_transformer"] = round(weighted, 4)
                posts[start + i]["label_transformer"] = top["label"]
        except Exception as e:
            logger.warning("RoBERTa batch %d failed: %s", bi, e)
            for i in range(start, end):
                posts[i]["score_transformer"] = 0.0
                posts[i]["label_transformer"] = "neutral"

        if bi % settings.GC_EVERY_N_BATCHES == 0:
            gc.collect()

    # Posts with no text are genuinely missing data, not neutral sentiment
    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            posts[i]["score_transformer"] = 0.0
            posts[i]["label_transformer"] = "neutral"
            posts[i]["empty_text"] = True

    del pipe
    gc.collect()
    _log_mem("after model release")


# ── Phase 3: Claude LLM (optional) ─────────────────────────────────

def score_llm(text: str, user: str = "", context: str = "Iran war") -> tuple[float | None, str | None]:
    """
    Use Claude for context-aware scoring. Returns (None, None) if the
    Anthropic SDK isn't installed or ANTHROPIC_API_KEY isn't set.
    """
    try:
        import anthropic
    except ImportError:
        return (None, None)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (None, None)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"Score the sentiment of this social media post about the {context}. "
        f"The post is by @{user}.\n\n"
        f'Post: """{text}"""\n\n'
        "Respond with ONLY valid JSON: "
        '{"score": <float from -1.0 (very negative/anti-war) to 1.0 (very positive/pro-war)>, '
        '"label": "<negative|neutral|positive>", '
        '"reasoning": "<one sentence>"}'
    )

    resp = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        data = json.loads(resp.content[0].text)
        return data["score"], data["label"]
    except (json.JSONDecodeError, KeyError):
        logger.warning("LLM unparseable response: %s", resp.content[0].text)
        return (None, None)


def score_llm_inplace(posts: list[dict]) -> None:
    """Add score_llm and label_llm fields to each post (slow, costs tokens)."""
    for post in tqdm(posts, desc="Claude LLM"):
        text = post.get("text", "")
        score, label = score_llm(text, user=post.get("user", ""))
        post["score_llm"] = score
        post["label_llm"] = label


# ── End-to-end pipeline ────────────────────────────────────────────

def analyze(
    posts: list[dict],
    use_vader: bool = True,
    use_transformer: bool = True,
    use_llm: bool = False,
    checkpoint: bool = True,
) -> pd.DataFrame:
    """
    Run the full sentiment pipeline on a list of post dicts.

    Phases are checkpointed to disk: if a crash occurs in RoBERTa, the
    next run picks up from the VADER checkpoint instead of re-scoring.
    """
    _log_mem("start")

    # Phase 1: VADER (cheap, always run if requested)
    if use_vader:
        if checkpoint and settings.VADER_CHECKPOINT.exists():
            logger.info("=== Phase 1: VADER (loading checkpoint) ===")
            ckpt = pd.read_parquet(settings.VADER_CHECKPOINT)
            # Merge VADER scores back into the post dicts by id
            vader_lookup = {
                row["id"]: (row["score_vader"], row["label_vader"])
                for _, row in ckpt.iterrows()
                if "score_vader" in row.index
            }
            restored = 0
            for post in posts:
                hit = vader_lookup.get(post.get("id"))
                if hit:
                    post["score_vader"], post["label_vader"] = hit
                    restored += 1
            if restored == len(posts):
                logger.info("Restored VADER scores for all %d posts from checkpoint", restored)
            else:
                # Checkpoint is stale or partial — re-score everything
                logger.info("Checkpoint covers %d/%d posts — re-scoring all", restored, len(posts))
                score_vader_inplace(posts)
        else:
            logger.info("=== Phase 1: VADER ===")
            score_vader_inplace(posts)

        if checkpoint:
            df = pd.DataFrame(posts)
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
            df = df.sort_values("created_at").reset_index(drop=True)
            settings.VADER_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(settings.VADER_CHECKPOINT, index=False)
            logger.info("VADER checkpoint saved to %s", settings.VADER_CHECKPOINT)
        _log_mem("after VADER")

    # Phase 2: RoBERTa (expensive, checkpointed)
    if use_transformer:
        logger.info("=== Phase 2: RoBERTa ===")
        score_roberta_inplace(posts)
        _log_mem("after RoBERTa")

    # Phase 3: LLM (opt-in, can be slow)
    if use_llm:
        logger.info("=== Phase 3: Claude LLM ===")
        score_llm_inplace(posts)
        _log_mem("after LLM")

    # Final dataframe
    df = pd.DataFrame(posts)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def save(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist scored results as parquet."""
    out = path or settings.SENTIMENT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved %d scored posts to %s", len(df), out)

    # Clean up stale VADER checkpoint now that we have the full file
    if settings.VADER_CHECKPOINT.exists():
        settings.VADER_CHECKPOINT.unlink()

    return out


def load_scored(path: Path | None = None) -> pd.DataFrame:
    """Load a previously-scored parquet."""
    return pd.read_parquet(path or settings.SENTIMENT_OUTPUT)
