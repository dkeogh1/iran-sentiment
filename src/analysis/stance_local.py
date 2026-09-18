"""
Stance-model experiments that run on the GPU node (k8s Jobs, see k8s/README.md):

  teacher_check   relabel a stratified sample with a stronger Claude model and
                  measure how much it disagrees with the Haiku labels the
                  dataset carries -- decides whether Haiku is a fit teacher.
  distill         fine-tune an encoder to regress score_llm; evaluate on a
                  held-out split against the teacher and against RoBERTa
                  valence (the current cheap scorer) on the same rows.
  local_llm_eval  run an open instruct model with the same prompt as the
                  Claude scorer on a held-out sample; same metrics.
  score_replies   score every cached reply with the distilled model so the
                  reply analysis has population-level stance, not a sample.

All heavy imports are local to the functions so the CPU-only CLI on dkbl1
imports this module without torch/transformers being present.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

NEUTRAL_BAND = 0.05  # same as VADER / the Haiku label fallback


# ── Frames and splits ──────────────────────────────────────────────

def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as teacher labels: a Haiku score, on-topic, non-empty text."""
    d = df[df["score_llm"].notna()].copy()
    if "label_llm" in d:
        d = d[d["label_llm"] != "off_topic"]
    d = d[d["text"].fillna("").str.strip().str.len() > 0]
    keep = [c for c in ["id", "text", "user", "tier", "score_llm", "score_transformer"] if c in d]
    return d[keep].reset_index(drop=True)


def stratified_split(df: pd.DataFrame, holdout: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-tier holdout so every tier is represented in the eval."""
    parts = [g.sample(frac=holdout, random_state=seed) for _, g in df.groupby("tier")]
    test = pd.concat(parts) if parts else df.iloc[0:0]
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """~n rows spread evenly across tiers (each tier capped at its size)."""
    tiers = df["tier"].dropna().unique()
    per = max(1, n // max(1, len(tiers)))
    parts = [g.sample(n=min(per, len(g)), random_state=seed) for _, g in df.groupby("tier")]
    return (pd.concat(parts) if parts else df.iloc[0:0]).reset_index(drop=True)


# ── Metrics ────────────────────────────────────────────────────────

def sign3(x: np.ndarray, band: float = NEUTRAL_BAND) -> np.ndarray:
    return np.where(x > band, 1, np.where(x < -band, -1, 0))


def agreement(y_ref: np.ndarray, y_new: np.ndarray) -> dict:
    """How well y_new reproduces y_ref (both in [-1, 1])."""
    y_ref = np.asarray(y_ref, dtype=float)
    y_new = np.asarray(y_new, dtype=float)
    ok = ~(np.isnan(y_ref) | np.isnan(y_new))
    y_ref, y_new = y_ref[ok], y_new[ok]
    if len(y_ref) < 2:
        return {"n": int(len(y_ref)), "pearson": float("nan"), "mae": float("nan"),
                "sign_agreement": float("nan"), "sign_flip_rate": float("nan")}
    s_ref, s_new = sign3(y_ref), sign3(y_new)
    return {
        "n": int(len(y_ref)),
        "pearson": float(np.corrcoef(y_ref, y_new)[0, 1]),
        "mae": float(np.mean(np.abs(y_ref - y_new))),
        "sign_agreement": float(np.mean(s_ref == s_new)),
        "sign_flip_rate": float(np.mean((s_ref * s_new) < 0)),  # opposite non-neutral signs
    }


def agreement_by_tier(df: pd.DataFrame, ref: str, new: str) -> pd.DataFrame:
    rows = []
    for tier, g in df.groupby("tier"):
        rows.append({"tier": tier, **agreement(g[ref].values, g[new].values)})
    rows.append({"tier": "ALL", **agreement(df[ref].values, df[new].values)})
    return pd.DataFrame(rows).set_index("tier")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


# ── 1. Teacher check (API, no GPU) ─────────────────────────────────

def teacher_check(df: pd.DataFrame, *, n: int = settings.TEACHER_CHECK_N,
                  model: str = settings.TEACHER_CHECK_MODEL,
                  seed: int = settings.DISTILL_SEED, concurrency: int = settings.LLM_CONCURRENCY,
                  out_dir: Path | None = None) -> pd.DataFrame:
    """Relabel a stratified sample with `model`; cache to
    teacher_check_<model>.parquet (incremental); return the joined frame with
    score_llm (Haiku) and score_teacher."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.analysis.sentiment import score_llm

    out_dir = out_dir or settings.PROCESSED_DIR
    out = out_dir / f"teacher_check_{model.replace('/', '_')}.parquet"
    base = training_frame(df)
    sample = stratified_sample(base, n, seed)
    cached = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    have = set(cached["id"].astype(str)) if not cached.empty else set()
    todo = sample[~sample["id"].astype(str).isin(have)]
    logger.info("teacher check: %d sampled, %d cached, %d to score with %s",
                len(sample), len(have), len(todo), model)

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(score_llm, r.text, r.user, "Iran war", model,
                          settings.TEACHER_MAX_TOKENS, settings.TEACHER_EFFORT): r.id
                for r in todo.itertuples()}
        for f in as_completed(futs):
            try:
                score, label = f.result()
            except Exception as e:  # one bad call must not discard the batch
                logger.warning("teacher call failed for %s: %s", futs[f], str(e)[:120])
                score, label = None, None
            results.append({"id": futs[f], "score_teacher": score, "label_teacher": label})
    new = pd.DataFrame(results)
    scored = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
    if not scored.empty:
        out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(out, index=False)
    joined = sample.merge(scored, on="id", how="inner")
    return joined


def teacher_report(joined: pd.DataFrame, model: str, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or settings.PROCESSED_DIR
    by = agreement_by_tier(joined, "score_llm", "score_teacher")
    d = joined.assign(diff=(joined["score_teacher"] - joined["score_llm"]).abs())
    worst = d.sort_values("diff", ascending=False).head(15)[
        ["tier", "user", "score_llm", "score_teacher", "text"]]
    worst["text"] = worst["text"].str[:110]
    report = {"model": model, "by_tier": by.reset_index().to_dict("records"),
              "worst": worst.to_dict("records")}
    _write_json(out_dir / f"teacher_check_{model.replace('/', '_')}.json", report)
    return report


# ── 2. Distillation (GPU) ──────────────────────────────────────────

def distill(df: pd.DataFrame, *, base_model: str = settings.DISTILL_BASE_MODEL,
            epochs: int = settings.DISTILL_EPOCHS, holdout: float = settings.DISTILL_HOLDOUT,
            batch_size: int = settings.DISTILL_BATCH_SIZE, lr: float = settings.DISTILL_LR,
            max_len: int = settings.DISTILL_MAX_LEN, seed: int = settings.DISTILL_SEED,
            label_col: str = "score_llm", out_dir: Path | None = None) -> dict:
    """Fine-tune `base_model` to regress `label_col` in [-1, 1]. Saves the model
    to MODELS_DIR/stance_distilled and metrics to distill_metrics.json."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    out_dir = out_dir or (settings.MODELS_DIR / "stance_distilled")
    base = training_frame(df)
    if label_col != "score_llm":
        base = base[base[label_col].notna()]
    train, test = stratified_split(base, holdout, seed)
    logger.info("distill: %d train / %d holdout rows, base=%s, device=%s",
                len(train), len(test), base_model, "cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)

    def to_ds(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(frame[["text", label_col]].rename(columns={label_col: "labels"}))
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)
        return ds.map(lambda b: {"labels": [float(x) for x in b["labels"]]}, batched=True)

    ds_train, ds_test = to_ds(train), to_ds(test)
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.06,
        fp16=torch.cuda.is_available(), eval_strategy="epoch", save_strategy="no",
        logging_steps=50, report_to=[], seed=seed, dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_test,
                      processing_class=tok)
    trainer.train()

    pred = trainer.predict(ds_test).predictions.reshape(-1)
    test = test.assign(score_distilled=np.clip(pred, -1, 1))
    metrics = {
        "base_model": base_model, "epochs": epochs, "n_train": int(len(train)),
        "n_holdout": int(len(test)), "label_col": label_col,
        "distilled_vs_teacher": agreement(test[label_col].values, test["score_distilled"].values),
        "distilled_by_tier": agreement_by_tier(test, label_col, "score_distilled").reset_index().to_dict("records"),
    }
    if "score_transformer" in test:
        metrics["roberta_valence_vs_teacher"] = agreement(test[label_col].values, test["score_transformer"].values)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    test.to_parquet(out_dir / "holdout_predictions.parquet", index=False)
    _write_json(out_dir / "distill_metrics.json", metrics)
    logger.info("distill: %s", json.dumps(metrics["distilled_vs_teacher"]))
    return metrics


def score_with_distilled(texts: list[str], model_dir: Path | None = None,
                         batch_size: int = 64, max_len: int = settings.DISTILL_MAX_LEN) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or (settings.MODELS_DIR / "stance_distilled")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = np.zeros(len(texts), dtype=float)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [t if isinstance(t, str) and t.strip() else "." for t in texts[i:i + batch_size]]
            enc = tok(chunk, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            out[i:i + batch_size] = model(**enc).logits.reshape(-1).float().cpu().numpy()
    return np.clip(out, -1, 1)


def score_replies(model_dir: Path | None = None) -> pd.DataFrame:
    """Add score_distilled to reply_sentiment.parquet (all cached replies)."""
    out = settings.REPLY_SENTIMENT_OUTPUT
    df = pd.read_parquet(out)
    todo = df["score_distilled"].isna() if "score_distilled" in df else pd.Series(True, index=df.index)
    if todo.any():
        logger.info("scoring %d replies with the distilled model", int(todo.sum()))
        df.loc[todo, "score_distilled"] = score_with_distilled(df.loc[todo, "text"].tolist(), model_dir)
        df.to_parquet(out, index=False)
    return df


# ── 3. Local LLM (GPU) ─────────────────────────────────────────────

def local_llm_eval(df: pd.DataFrame, *, model_name: str = settings.LOCAL_LLM_MODEL,
                   n: int = settings.LOCAL_LLM_EVAL_N, batch_size: int = settings.LOCAL_LLM_BATCH,
                   max_new_tokens: int = settings.LOCAL_LLM_MAX_NEW_TOKENS,
                   seed: int = settings.DISTILL_SEED, holdout: float = settings.DISTILL_HOLDOUT,
                   out_dir: Path | None = None) -> dict:
    """Score a sample of the SAME holdout split the distillation uses with an
    open instruct model in 4-bit, using the Claude prompt verbatim."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.analysis.sentiment import llm_prompt, parse_llm_json

    out_dir = out_dir or settings.PROCESSED_DIR
    base = training_frame(df)
    _, test = stratified_split(base, holdout, seed)
    sample = stratified_sample(test, n, seed)

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant,
                                                 device_map="auto").eval()
    scores, labels, raws = [], [], []
    prompts = [tok.apply_chat_template([{"role": "user", "content": llm_prompt(r.text, r.user)}],
                                       tokenize=False, add_generation_prompt=True)
               for r in sample.itertuples()]
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, seq in enumerate(gen):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            data = parse_llm_json(text) or {}
            sc = data.get("score")
            scores.append(float(sc) if isinstance(sc, (int, float)) else np.nan)
            labels.append(data.get("label"))
            raws.append(text[:200])
        logger.info("local llm: %d/%d", min(i + batch_size, len(prompts)), len(prompts))
    sample = sample.assign(score_local=scores, label_local=labels, raw_local=raws)
    tag = model_name.replace("/", "_")
    sample.to_parquet(out_dir / f"local_llm_{tag}.parquet", index=False)
    metrics = {
        "model": model_name, "n": int(len(sample)),
        "unparseable_rate": float(np.mean(np.isnan(scores))),
        "local_vs_teacher": agreement(sample["score_llm"].values, sample["score_local"].values),
        "local_by_tier": agreement_by_tier(sample, "score_llm", "score_local").reset_index().to_dict("records"),
    }
    if "score_transformer" in sample:
        metrics["roberta_valence_vs_teacher"] = agreement(sample["score_llm"].values, sample["score_transformer"].values)
    _write_json(out_dir / f"local_llm_{tag}.json", metrics)
    logger.info("local llm: %s", json.dumps(metrics["local_vs_teacher"]))
    return metrics
