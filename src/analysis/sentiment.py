"""
Sentiment analysis pipeline.

Three scoring strategies, run in order of cost/accuracy tradeoff:
  1. VADER       - fast rule-based baseline (always on)
  2. RoBERTa     - HuggingFace cardiffnlp/twitter-roberta-base-sentiment-latest
  3. Claude LLM  - optional, context-aware scoring

Guardrails (the mini PC has crashed twice — once OOM, once thermal/power):
  - VADER checkpoints to disk before RoBERTa loads (Phase 1 work survives
    a Phase 2 crash)
  - RoBERTa runs in small batches (see settings.ROBERTA_BATCH_SIZE)
  - PyTorch thread count is capped so inference doesn't saturate all CPU
    cores and trip thermal protection on small fanless boxes
  - RoBERTa results are checkpointed every N batches so a mid-phase
    crash loses at most N batches of work
  - Per-batch RSS check aborts cleanly before the kernel OOM-killer
    fires, leaving the partial checkpoint intact
  - Model is explicitly released after use
  - Memory usage is printed between phases
"""

import gc
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import settings

# Matches text consisting only of one or more URLs separated by whitespace.
# Pre-filtering these avoids wasted LLM calls on posts whose "content" is
# just a media link (Claude can't follow URLs and returns a refusal).
_URL_ONLY_RE = re.compile(r"^\s*(https?://\S+\s*)+$")

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
    """Add score_vader and label_vader fields to each post in place.

    Posts that already have score_vader (e.g. restored from a prior run)
    are skipped — only un-scored posts are processed.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    pending = [p for p in posts if "score_vader" not in p]
    if not pending:
        logger.info("VADER: all %d posts already scored — skipping", len(posts))
        return
    if len(pending) < len(posts):
        logger.info("VADER: %d posts pending (out of %d)", len(pending), len(posts))

    analyzer = SentimentIntensityAnalyzer()
    for post in tqdm(pending, desc="VADER"):
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


class RobertaAborted(RuntimeError):
    """Raised when the RSS ceiling is hit. Partial results are checkpointed."""


def _save_roberta_checkpoint(posts: list[dict]) -> None:
    """Persist current state of `posts` so a future run can resume."""
    df = pd.DataFrame(posts)
    settings.ROBERTA_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.ROBERTA_CHECKPOINT, index=False)


def _restore_roberta_checkpoint(posts: list[dict]) -> int:
    """
    Populate score_transformer/label_transformer on `posts` from the
    checkpoint, matched by post id. Returns the number of posts restored.
    """
    if not settings.ROBERTA_CHECKPOINT.exists():
        return 0
    ckpt = pd.read_parquet(settings.ROBERTA_CHECKPOINT)
    if "score_transformer" not in ckpt.columns:
        return 0
    lookup = {
        row["id"]: (row["score_transformer"], row["label_transformer"])
        for _, row in ckpt.iterrows()
        if pd.notna(row.get("score_transformer"))
    }
    restored = 0
    for post in posts:
        hit = lookup.get(post.get("id"))
        if hit:
            post["score_transformer"], post["label_transformer"] = hit
            restored += 1
    return restored


def _restore_prior_scores(posts: list[dict]) -> tuple[int, int]:
    """
    Populate scores from the previous sentiment_all.parquet (the canonical
    output of the last successful run). This is the cheap-refresh path:
    when new posts are added to an existing dataset, only the new ones
    need scoring.

    Returns (vader_restored, transformer_restored). Posts that already
    have a score field are not overwritten.
    """
    if not settings.SENTIMENT_OUTPUT.exists():
        return (0, 0)
    prior = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    has_vader = "score_vader" in prior.columns
    has_transformer = "score_transformer" in prior.columns
    if not (has_vader or has_transformer):
        return (0, 0)

    has_llm = "score_llm" in prior.columns

    lookup: dict[str, dict] = {}
    for _, row in prior.iterrows():
        pid = row.get("id")
        if not pid:
            continue
        entry: dict = {}
        if has_vader and pd.notna(row.get("score_vader")):
            entry["score_vader"] = row["score_vader"]
            entry["label_vader"] = row["label_vader"]
        if has_transformer and pd.notna(row.get("score_transformer")):
            entry["score_transformer"] = row["score_transformer"]
            entry["label_transformer"] = row["label_transformer"]
        if has_llm and pd.notna(row.get("score_llm")):
            entry["score_llm"] = row["score_llm"]
            entry["label_llm"] = row.get("label_llm")
        if entry:
            lookup[pid] = entry

    v_count = t_count = 0
    for post in posts:
        hit = lookup.get(post.get("id"))
        if not hit:
            continue
        if "score_vader" in hit and "score_vader" not in post:
            post["score_vader"] = hit["score_vader"]
            post["label_vader"] = hit["label_vader"]
            v_count += 1
        if "score_transformer" in hit and "score_transformer" not in post:
            post["score_transformer"] = hit["score_transformer"]
            post["label_transformer"] = hit["label_transformer"]
            t_count += 1
        if "score_llm" in hit and post.get("score_llm") is None:
            post["score_llm"] = hit["score_llm"]
            post["label_llm"] = hit["label_llm"]
    return (v_count, t_count)


def score_roberta_inplace(
    posts: list[dict],
    batch_size: int = settings.ROBERTA_BATCH_SIZE,
) -> None:
    """
    Add score_transformer and label_transformer fields to each post.

    Runs in batches to bound peak memory. Inference is CPU-only and
    thread-capped to keep small fanless boxes from thermal-tripping.

    Resumability: if a previous run wrote checkpoint_roberta.parquet,
    posts already scored there are restored and skipped. New scoring
    progress is checkpointed every N batches so a crash mid-phase
    doesn't waste hours of inference.

    Aborts cleanly (saving checkpoint) if process RSS exceeds
    ROBERTA_MAX_RSS_MB, so the kernel OOM-killer doesn't get to it.

    Note: score_transformer is a weighted average across all label
    probabilities (neg=-1, neu=0, pos=+1) while label_transformer is
    the argmax (highest-confidence label). These can disagree — e.g.
    a post scored [pos:0.4, neu:0.6] yields score +0.4 but label
    "neutral". The weighted score is more informative for aggregation;
    the label is used for categorical breakdowns.
    """
    # Cap torch threads BEFORE importing the pipeline so the runtime
    # picks up the limit. Without this, PyTorch grabs all cores and
    # sustained 100% CPU has powered off this machine.
    import torch
    torch.set_num_threads(settings.TORCH_NUM_THREADS)
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.TORCH_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(settings.TORCH_NUM_THREADS))
    logger.info("torch threads capped to %d", settings.TORCH_NUM_THREADS)

    from transformers import pipeline

    # Resume from prior partial checkpoint if present
    restored = _restore_roberta_checkpoint(posts)
    if restored:
        logger.info("Resumed RoBERTa from checkpoint: %d/%d posts already scored",
                    restored, len(posts))

    # Identify posts that still need scoring (no transformer score yet
    # AND have text — empty texts handled separately at the end)
    pending_indices = [
        i for i, p in enumerate(posts)
        if "score_transformer" not in p and (p.get("text") or "").strip()
    ]

    if not pending_indices:
        logger.info("All posts already have RoBERTa scores — nothing to do")
    else:
        logger.info("RoBERTa: %d posts pending (out of %d)",
                    len(pending_indices), len(posts))
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

        num_batches = (len(pending_indices) + batch_size - 1) // batch_size
        aborted = False

        try:
            for bi in tqdm(range(num_batches), desc="RoBERTa"):
                start = bi * batch_size
                end = min(start + batch_size, len(pending_indices))
                batch_idx = pending_indices[start:end]
                batch_texts = [posts[i].get("text", "") or "" for i in batch_idx]

                try:
                    results = pipe(batch_texts)
                    for j, result in enumerate(results):
                        weighted = sum(
                            r["score"] * _LABEL_MAP.get(r["label"], 0) for r in result
                        )
                        top = max(result, key=lambda r: r["score"])
                        posts[batch_idx[j]]["score_transformer"] = round(weighted, 4)
                        posts[batch_idx[j]]["label_transformer"] = top["label"]
                except Exception as e:
                    logger.warning("RoBERTa batch %d failed: %s", bi, e)
                    for i in batch_idx:
                        posts[i]["score_transformer"] = 0.0
                        posts[i]["label_transformer"] = "neutral"

                if bi % settings.GC_EVERY_N_BATCHES == 0:
                    gc.collect()

                # Periodic checkpoint — caps blast radius of any future crash
                if (bi + 1) % settings.ROBERTA_CHECKPOINT_EVERY_N_BATCHES == 0:
                    _save_roberta_checkpoint(posts)
                    logger.info("RoBERTa checkpoint @ batch %d/%d (mem %.0f MB)",
                                bi + 1, num_batches, mem_usage_mb())

                # Memory ceiling guard — abort cleanly before kernel OOM-kills us
                rss = mem_usage_mb()
                if rss > settings.ROBERTA_MAX_RSS_MB:
                    logger.error(
                        "RSS %.0f MB exceeded ceiling %d MB at batch %d — "
                        "aborting and checkpointing",
                        rss, settings.ROBERTA_MAX_RSS_MB, bi,
                    )
                    aborted = True
                    break
        finally:
            # Always checkpoint progress before releasing the model so a crash
            # in del/gc still leaves a recoverable state.
            _save_roberta_checkpoint(posts)
            del pipe
            gc.collect()
            _log_mem("after model release")

        if aborted:
            raise RobertaAborted(
                f"RoBERTa aborted at memory ceiling. "
                f"Checkpoint at {settings.ROBERTA_CHECKPOINT}. "
                f"Re-run `analyze` to resume."
            )

    # Posts with no text are genuinely missing data, not neutral sentiment
    for i, post in enumerate(posts):
        if not (post.get("text") or "").strip():
            post["score_transformer"] = 0.0
            post["label_transformer"] = "neutral"
            post["empty_text"] = True


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

    raw = resp.content[0].text.strip()
    # Claude commonly wraps JSON in ```json ... ``` despite the prompt
    # asking for raw JSON. Strip the fence if present before parsing.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM unparseable response: %s", resp.content[0].text)
        return (None, None)

    score = data.get("score")
    if not isinstance(score, (int, float)):
        logger.warning("LLM response missing valid score: %s", resp.content[0].text)
        return (None, None)

    # Graceful fallback: Claude sometimes returns {score, reasoning}
    # without a label for off-topic posts. Derive the label from the
    # score using the same ±0.05 threshold as VADER.
    label = data.get("label")
    if not label:
        label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    return score, label


def score_llm_inplace(
    posts: list[dict],
    tiers: list[str] | None = None,
    accounts: list[str] | None = None,
) -> None:
    """Add score_llm and label_llm fields to each post (costs tokens).

    Filters:
      tiers    - if set, only score posts whose tier is in this list
      accounts - if set, only score posts whose user is in this list
                 (combined with tiers via OR — any matching post is scored)

    Posts that already have a score_llm value are skipped (resume-safe
    and avoids re-spending tokens on a re-run).

    Runs LLM_CONCURRENCY calls in parallel via a thread pool (calls are
    I/O-bound so threads are effective). Saves progress to
    SENTIMENT_OUTPUT every LLM_SAVE_EVERY_N completions so a crash or
    interruption mid-run doesn't lose the spent tokens.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def matches(post: dict) -> bool:
        if tiers is None and accounts is None:
            return True
        if tiers and post.get("tier") in tiers:
            return True
        if accounts and post.get("user") in accounts:
            return True
        return False

    # Pre-filter URL-only posts (media-only tweets) — Claude can't see
    # linked content and would return a plain-text refusal, wasting a
    # call. Label them off_topic with score 0 directly.
    url_only_count = 0
    pending: list[dict] = []
    for p in posts:
        if not matches(p):
            continue
        if p.get("score_llm") is not None:
            continue
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if _URL_ONLY_RE.match(text):
            p["score_llm"] = 0.0
            p["label_llm"] = "off_topic"
            url_only_count += 1
        else:
            pending.append(p)

    if url_only_count:
        logger.info("LLM: pre-labeled %d URL-only posts as off_topic "
                    "(no API call)", url_only_count)

    if not pending:
        logger.info("LLM: no posts to score (filter matched nothing or all already scored)")
        return

    filter_desc = []
    if tiers:
        filter_desc.append(f"tiers={tiers}")
    if accounts:
        filter_desc.append(f"accounts={accounts}")
    logger.info("LLM: %d posts to score (%s), concurrency=%d",
                len(pending), ", ".join(filter_desc) or "no filter",
                settings.LLM_CONCURRENCY)

    def _score_one(post: dict) -> None:
        text = post.get("text", "")
        score, label = score_llm(text, user=post.get("user", ""))
        post["score_llm"] = score
        post["label_llm"] = label

    completed = 0
    with ThreadPoolExecutor(max_workers=settings.LLM_CONCURRENCY) as executor:
        futures = [executor.submit(_score_one, p) for p in pending]
        for future in tqdm(as_completed(futures), total=len(pending), desc="Claude LLM"):
            try:
                future.result()
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
            completed += 1

            # Periodic save — bounds the blast radius of a mid-run crash
            if completed % settings.LLM_SAVE_EVERY_N == 0:
                df = pd.DataFrame(posts)
                settings.SENTIMENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
                logger.info("LLM progress save @ %d/%d", completed, len(pending))


# ── End-to-end pipeline ────────────────────────────────────────────

def analyze(
    posts: list[dict],
    use_vader: bool = True,
    use_transformer: bool = True,
    use_llm: bool = False,
    checkpoint: bool = True,
    llm_tiers: list[str] | None = None,
    llm_accounts: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the full sentiment pipeline on a list of post dicts.

    Phases are checkpointed to disk: if a crash occurs in RoBERTa, the
    next run picks up from the VADER checkpoint instead of re-scoring.
    """
    _log_mem("start")

    # Cheap-refresh path: if the previous successful run wrote
    # sentiment_all.parquet, prepopulate scores for posts whose ids
    # match. This makes incremental refreshes (a few hundred new posts
    # added to thousands of existing ones) skip both phases entirely
    # for the already-scored set.
    prior_v, prior_t = _restore_prior_scores(posts)
    if prior_v or prior_t:
        logger.info("Restored from prior run: %d VADER, %d RoBERTa "
                    "(out of %d posts)", prior_v, prior_t, len(posts))

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
        score_llm_inplace(posts, tiers=llm_tiers, accounts=llm_accounts)
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

    # Clean up stale checkpoints now that the full file is on disk
    for ckpt in (settings.VADER_CHECKPOINT, settings.ROBERTA_CHECKPOINT):
        if ckpt.exists():
            ckpt.unlink()

    return out


def load_scored(path: Path | None = None) -> pd.DataFrame:
    """Load a previously-scored parquet."""
    return pd.read_parquet(path or settings.SENTIMENT_OUTPUT)
