"""
Stance-model experiments that run on the GPU node (k8s Jobs, see k8s/README.md):

  teacher_check   relabel a stratified sample with a stronger Claude model and
                  measure how much it disagrees with the Haiku labels the
                  dataset carries -- decides whether Haiku is a fit teacher.
  distill         fine-tune an encoder to regress score_llm; evaluate on a
                  held-out split against the teacher and against RoBERTa
                  valence (the current cheap scorer) on the same rows.
  sweep           run every recipe in settings.DISTILL_SWEEP on that split,
                  cross-validate the best, fit it on all labels.
  local_llm_eval  run an open instruct model with the same prompt as the
                  Claude scorer on a held-out sample; same metrics.
  score_replies   score every cached reply with the distilled model so the
                  reply analysis has population-level stance, not a sample.

Teacher checks on the host (direct API, no GPU): reply_teacher_check (v1,
broadcaster prompt) and reply_teacher_check_v2 (the full reply with the
Trump post it answers), and trump_teacher_check (Opus on a sample of
Trump's feed against the distilled scorer used there). Each has an
estimate that makes no call. The v2 and Trump checks also run through the
Message Batches API at half price (teacher_batch_submit / _status /
_collect): same requests, same label cache, same report.

All heavy imports are local to the functions so the CPU-only CLI on dkbl1
imports this module without torch/transformers being present.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.text_rules import has_text

logger = logging.getLogger(__name__)

NEUTRAL_BAND = 0.05  # same as VADER / the Haiku label fallback


# ── Frames and splits ──────────────────────────────────────────────

def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable as teacher labels: a Haiku score, on-topic, non-empty text.
    A post whose text changed after Haiku scored it (x-backfill-text) counts
    as it was scored, from the archive (superseded.with_archived_scores), so
    the seeded teacher-check sample and the training pairs stay the same."""
    from src.superseded import with_archived_scores
    df = with_archived_scores(df, "score_llm")
    d = df[df["score_llm"].notna()].copy()
    if "label_llm" in d:
        d = d[d["label_llm"] != "off_topic"]
    d = d[d["text"].fillna("").str.strip().str.len() > 0]
    keep = [c for c in ["id", "text", "user", "tier", "score_llm", "score_transformer"] if c in d]
    keep += [c for c in d.columns if c.startswith("score_") and c not in keep]  # alt teachers (score_opus)
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
    """Temp file, then rename: a run killed mid-write leaves the old file whole
    (reports, and the batch state that holds a paid batch's id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


# ── 1. Teacher check (API, no GPU) ─────────────────────────────────

def teacher_check_todo(df: pd.DataFrame, *, n: int = settings.TEACHER_CHECK_N,
                       model: str = settings.TEACHER_CHECK_MODEL, seed: int = settings.DISTILL_SEED,
                       out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sample, cached labels, posts still to label) for teacher_check, so
    --estimate can price the run without making it."""
    out_dir = out_dir or settings.PROCESSED_DIR
    out = out_dir / f"teacher_check_{model.replace('/', '_')}.parquet"
    base = training_frame(df)
    sample = stratified_sample(base, n, seed)
    cached = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    if not cached.empty:
        cached = cached[cached["score_teacher"].notna()]  # failed calls are retried
    have = set(cached["id"].astype(str)) if not cached.empty else set()
    todo = sample[~sample["id"].astype(str).isin(have)]
    return sample, cached, todo


def teacher_check(df: pd.DataFrame, *, n: int = settings.TEACHER_CHECK_N,
                  model: str = settings.TEACHER_CHECK_MODEL,
                  seed: int = settings.DISTILL_SEED, concurrency: int = settings.LLM_CONCURRENCY,
                  out_dir: Path | None = None) -> pd.DataFrame:
    """Relabel a stratified sample with `model`; cache to
    teacher_check_<model>.parquet (incremental); return the joined frame with
    score_llm (Haiku) and score_teacher."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv
    from src.analysis.sentiment import score_llm
    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)  # host runs need the key

    out_dir = out_dir or settings.PROCESSED_DIR
    out = out_dir / f"teacher_check_{model.replace('/', '_')}.parquet"
    sample, cached, todo = teacher_check_todo(df, n=n, model=model, seed=seed, out_dir=out_dir)
    logger.info("teacher check: %d sampled, %d cached, %d to score with %s",
                len(sample), len(cached), len(todo), model)

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

def _fit_eval(train: pd.DataFrame, test: pd.DataFrame | None, *, base_model: str, epochs: int,
              batch_size: int, lr: float, max_len: int, seed: int, label_col: str,
              work_dir: Path, save_to: Path | None = None,
              grad_accum: int = 1, optim: str = "adamw_torch",
              gradient_checkpointing: bool = False) -> tuple[dict, np.ndarray | None]:
    """Fine-tune `base_model` (regression head) on `train`; predict `test` if
    given. Returns (info, predictions). Frees the GPU afterwards so a sweep
    can chain recipes in one process."""
    import gc
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, ignore_mismatched_sizes=True)

    def to_ds(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(frame[["text", label_col]].rename(columns={label_col: "labels"}))
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)
        return ds.map(lambda b: {"labels": [float(x) for x in b["labels"]]}, batched=True)

    ds_train = to_ds(train)
    ds_test = to_ds(test) if test is not None else None
    # bf16 on Ampere+ (the 3080) is numerically safer than fp16 for DeBERTa-v3.
    bf16 = cuda and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(work_dir / "trainer"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum, optim=optim,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.06,
        bf16=bf16, fp16=cuda and not bf16,
        eval_strategy="epoch" if ds_test is not None else "no", save_strategy="no",
        logging_steps=50, report_to=[], seed=seed, dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_test,
                      processing_class=tok)
    out = trainer.train()
    pred = None
    if ds_test is not None:
        pred = np.clip(trainer.predict(ds_test).predictions.reshape(-1), -1, 1)
    if save_to is not None:
        model.save_pretrained(save_to)
        tok.save_pretrained(save_to)
    info = {"base_model": base_model, "epochs": epochs, "lr": lr, "max_len": max_len,
            "batch_size": batch_size, "grad_accum": grad_accum, "optim": optim,
            "gradient_checkpointing": gradient_checkpointing, "seed": seed, "n_train": int(len(train)),
            "train_runtime_s": float(out.metrics.get("train_runtime", 0)),
            "train_loss": float(out.metrics.get("train_loss", float("nan")))}
    del trainer, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return info, pred


def recipe_by_name(name: str) -> dict:
    rc = next((r for r in settings.DISTILL_SWEEP if r["name"] == name), None)
    if rc is None:
        raise KeyError(f"no recipe {name!r} in settings.DISTILL_SWEEP")
    return rc


def model_max_len(model_dir: Path) -> int:
    """The token length the model at model_dir was trained at: its recipe.txt
    names a DISTILL_SWEEP recipe, else it was fit at DISTILL_MAX_LEN."""
    marker = model_dir / "recipe.txt"
    name = marker.read_text().strip() if marker.exists() else ""
    rc = next((r for r in settings.DISTILL_SWEEP if r["name"] == name), None)
    return rc["max_len"] if rc else settings.DISTILL_MAX_LEN


def final_dir_free(final_dir: Path, recipe: str) -> bool:
    """True when final_dir is empty and may take a fit of `recipe`; False when
    it already holds that recipe (nothing to do). Raises when it holds any
    other model: a production model's scores live in the data, so a final dir
    is never overwritten. On 2026-09-19 the sweep's final fit replaced the
    distill-opus fit in place and the reply scores stopped being reproducible."""
    if not (final_dir / "config.json").exists():
        return True
    marker = final_dir / "recipe.txt"
    held = marker.read_text().strip() if marker.exists() else "an unmarked model"
    if held == recipe:
        return False
    raise FileExistsError(f"{final_dir} already holds {held}; move it aside to fit {recipe} there")


def distill(df: pd.DataFrame, *, base_model: str = settings.DISTILL_BASE_MODEL,
            epochs: int = settings.DISTILL_EPOCHS, holdout: float = settings.DISTILL_HOLDOUT,
            batch_size: int = settings.DISTILL_BATCH_SIZE, lr: float = settings.DISTILL_LR,
            max_len: int = settings.DISTILL_MAX_LEN, seed: int = settings.DISTILL_SEED,
            label_col: str = "score_llm", out_dir: Path | None = None,
            recipe: str | None = None, fit_all: bool = False,
            extra: pd.DataFrame | None = None) -> dict:
    """Fine-tune to regress `label_col` in [-1, 1]. With `recipe`, every
    hyper-parameter (incl. optimizer / checkpointing) comes from that
    DISTILL_SWEEP entry. Evaluates on the per-tier holdout; with `fit_all`
    also refits on every label and saves that as the production model."""
    rc = recipe_by_name(recipe) if recipe else {}
    base_model = rc.get("base_model", base_model)
    epochs, lr, max_len = rc.get("epochs", epochs), rc.get("lr", lr), rc.get("max_len", max_len)
    batch_size = rc.get("batch_size", batch_size)
    fit_kw = {"grad_accum": rc.get("grad_accum", 1), "optim": rc.get("optim", "adamw_torch"),
              "gradient_checkpointing": rc.get("gradient_checkpointing", False)}
    suffix = "" if label_col == "score_llm" else f"_{label_col}"
    if extra is not None and len(extra):
        suffix += "_mixed"
    final_dir = settings.MODELS_DIR / f"stance_distilled_final{suffix}"
    refit = fit_all and final_dir_free(final_dir, recipe or base_model)  # fail before training
    out_dir = out_dir or (settings.MODELS_DIR / f"stance_distilled{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = training_frame(df)
    if label_col != "score_llm":
        base = base[base[label_col].notna()]
    if extra is not None and len(extra):
        # Extra labelled rows (e.g. Opus-labelled replies, tier="reply_<post>")
        # join the pool; the per-tier split holds 20% of each of their tiers
        # out, so the metrics report the new domain separately.
        cols = ["id", "text", "user", "tier", label_col]
        base = pd.concat([base, extra[cols].assign(id=extra["id"].astype(str))], ignore_index=True)
        base = base.drop_duplicates("id")
        logger.info("distill: +%d extra labelled rows", len(extra))
    train, test = stratified_split(base, holdout, seed)
    logger.info("distill: %d train / %d holdout rows, base=%s, label=%s, recipe=%s",
                len(train), len(test), base_model, label_col, recipe)
    info, pred = _fit_eval(train, test, base_model=base_model, epochs=epochs, batch_size=batch_size,
                           lr=lr, max_len=max_len, seed=seed, label_col=label_col,
                           work_dir=out_dir, save_to=out_dir, **fit_kw)
    test = test.assign(score_distilled=pred)
    metrics = {
        **info, "n_holdout": int(len(test)), "label_col": label_col,
        "distilled_vs_teacher": agreement(test[label_col].values, test["score_distilled"].values),
        "distilled_by_tier": agreement_by_tier(test, label_col, "score_distilled").reset_index().to_dict("records"),
    }
    if "score_transformer" in test:
        metrics["roberta_valence_vs_teacher"] = agreement(test[label_col].values, test["score_transformer"].values)
    test.to_parquet(out_dir / "holdout_predictions.parquet", index=False)
    if fit_all and not refit:
        logger.info("distill: %s already holds %s; not refitting", final_dir, recipe or base_model)
    if refit:
        finfo, _ = _fit_eval(base, None, base_model=base_model, epochs=epochs, batch_size=batch_size,
                             lr=lr, max_len=max_len, seed=seed, label_col=label_col,
                             work_dir=out_dir / "final", save_to=final_dir, **fit_kw)
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "recipe.txt").write_text(recipe or base_model)
        metrics["final"] = {**finfo, "path": str(final_dir)}
    _write_json(out_dir / "distill_metrics.json", metrics)
    logger.info("distill: %s", json.dumps(metrics["distilled_vs_teacher"]))
    return metrics


def kfold_indices(n: int, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::folds] for i in range(folds)]


def pick_best(results: list[dict], key: str = "pearson") -> dict:
    """Best sweep entry by holdout `key`; entries missing it (failed fits) lose."""
    def score(r):
        h = r.get("holdout")
        v = h.get(key) if isinstance(h, dict) else None
        return v if isinstance(v, (int, float)) and v == v else None  # drop missing / NaN
    ok = [r for r in results if score(r) is not None]
    return max(ok, key=score) if ok else {}


def sweep(df: pd.DataFrame, *, recipes: list[dict] | None = None, folds: int = settings.DISTILL_CV_FOLDS,
          holdout: float = settings.DISTILL_HOLDOUT, seed: int = settings.DISTILL_SEED,
          batch_size: int = settings.DISTILL_BATCH_SIZE, label_col: str = "score_llm",
          out_dir: Path | None = None) -> dict:
    """Run every recipe on the shared split, cross-validate the best, fit it on
    all labels. Results accumulate in sweep_results.json so a killed run resumes."""
    recipes = recipes or settings.DISTILL_SWEEP
    out_dir = out_dir or (settings.MODELS_DIR / ("sweep" if label_col == "score_llm" else f"sweep_{label_col}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "sweep_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    # Failed recipes (OOM, missing dependency) are retried on resume.
    results = [r for r in results if "error" not in r]
    done = {r["name"] for r in results}

    base = training_frame(df)
    if label_col != "score_llm":
        base = base[base[label_col].notna()]
    train, test = stratified_split(base, holdout, seed)
    logger.info("sweep: %d recipes (%d done), %d train / %d holdout", len(recipes), len(done),
                len(train), len(test))

    for rc in recipes:
        if rc["name"] in done:
            continue
        logger.info("sweep: %s", rc)
        try:
            info, pred = _fit_eval(train, test, base_model=rc["base_model"], epochs=rc["epochs"],
                                   batch_size=rc.get("batch_size", batch_size), lr=rc["lr"],
                                   max_len=rc["max_len"], seed=seed, label_col=label_col,
                                   work_dir=out_dir / rc["name"], grad_accum=rc.get("grad_accum", 1), optim=rc.get("optim", "adamw_torch"),
                                   gradient_checkpointing=rc.get("gradient_checkpointing", False))
            t = test.assign(pred=pred)
            entry = {"name": rc["name"], **info,
                     "holdout": agreement(t[label_col].values, t["pred"].values),
                     "by_tier": agreement_by_tier(t, label_col, "pred").reset_index().to_dict("records")}
        except Exception as e:  # a recipe that OOMs or fails to load must not sink the sweep
            logger.exception("sweep: %s failed", rc["name"])
            entry = {"name": rc["name"], **rc, "error": str(e)[:300]}
        results.append(entry)
        results_path.write_text(json.dumps(results, indent=2, default=float))
        logger.info("sweep: %s -> %s", rc["name"], json.dumps(entry.get("holdout", entry.get("error"))))

    best = pick_best(results)
    summary = {"results": results, "best": best.get("name")}
    if not best:
        _write_json(out_dir / "sweep_summary.json", summary)
        return summary
    rc = next(r for r in recipes if r["name"] == best["name"])

    # Cross-validation on the best recipe: error bars on the holdout number.
    # Tied to the recipe name: if a resumed sweep finds a new best, CV and
    # the final fit are redone.
    cv_path = out_dir / "cv_results.json"
    cv = json.loads(cv_path.read_text()) if cv_path.exists() else []
    if cv and cv[0].get("recipe", best["name"]) != best["name"]:
        cv = []
    idx = kfold_indices(len(base), folds, seed)
    for k in range(len(cv), folds):
        te = base.iloc[idx[k]].reset_index(drop=True)
        tr = base.drop(base.index[idx[k]]).reset_index(drop=True)
        _, pred = _fit_eval(tr, te, base_model=rc["base_model"], epochs=rc["epochs"],
                            batch_size=rc.get("batch_size", batch_size), lr=rc["lr"],
                            max_len=rc["max_len"], seed=seed + k, label_col=label_col,
                            work_dir=out_dir / f"cv{k}", grad_accum=rc.get("grad_accum", 1), optim=rc.get("optim", "adamw_torch"),
                                   gradient_checkpointing=rc.get("gradient_checkpointing", False))
        cv.append({"fold": k, "recipe": best["name"], **agreement(te[label_col].values, pred)})
        cv_path.write_text(json.dumps(cv, indent=2, default=float))
        logger.info("sweep: cv fold %d -> %s", k, json.dumps(cv[-1]))
    arr = np.array([[c["pearson"], c["sign_agreement"], c["sign_flip_rate"]] for c in cv])
    summary["cv"] = {"folds": folds, "pearson_mean": float(arr[:, 0].mean()), "pearson_std": float(arr[:, 0].std()),
                     "sign_agreement_mean": float(arr[:, 1].mean()), "sign_agreement_std": float(arr[:, 1].std()),
                     "sign_flip_mean": float(arr[:, 2].mean()), "sign_flip_std": float(arr[:, 2].std())}

    # Final model on ALL labels.
    final_dir = settings.MODELS_DIR / ("stance_distilled_final" if label_col == "score_llm"
                                        else f"stance_distilled_final_{label_col}")
    marker = final_dir / "recipe.txt"
    if final_dir_free(final_dir, best["name"]):
        info, _ = _fit_eval(base, None, base_model=rc["base_model"], epochs=rc["epochs"],
                            batch_size=rc.get("batch_size", batch_size), lr=rc["lr"],
                            max_len=rc["max_len"], seed=seed, label_col=label_col,
                            work_dir=out_dir / "final", save_to=final_dir, grad_accum=rc.get("grad_accum", 1), optim=rc.get("optim", "adamw_torch"),
                                   gradient_checkpointing=rc.get("gradient_checkpointing", False))
        final_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(best["name"])
        summary["final"] = {**info, "path": str(final_dir)}
    _write_json(out_dir / "sweep_summary.json", summary)
    logger.info("sweep: best=%s cv=%s", best["name"], json.dumps(summary.get("cv")))
    return summary


def score_with_distilled(texts: list[str], model_dir: Path | None = None,
                         batch_size: int = 64, max_len: int | None = None) -> np.ndarray:
    """Scores in [-1, 1]; NaN for texts without text to judge
    (src/text_rules.py), which never reach the model (the old "." stand-in scored
    every image-only post a constant +0.111, i.e. pro-war). max_len defaults
    to the model's training length."""
    out = np.full(len(texts), np.nan)
    keep = [i for i, t in enumerate(texts) if has_text(t)]
    if not keep:
        return out
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or (settings.MODELS_DIR / "stance_distilled")
    max_len = max_len or model_max_len(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    with torch.no_grad():
        for i in range(0, len(keep), batch_size):
            idx = keep[i:i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            out[idx] = model(**enc).logits.reshape(-1).float().cpu().numpy()
    return np.clip(out, -1, 1)


def _to_score(df: pd.DataFrame, col: str) -> pd.Series:
    """Rows still to score: no `col` yet, and text to score (adds `col` as
    NaN when the frame lacks it). Textless rows stay NaN, so incremental
    runs do not resend them forever."""
    if col not in df:
        df[col] = np.nan
    return df[col].isna() & df["text"].map(has_text)


def score_replies(model_dir: Path | None = None, col: str = "score_distilled",
                  max_len: int | None = None) -> pd.DataFrame:
    """Add `col` (scores from the model at model_dir) to reply_sentiment.parquet."""
    model_dir = model_dir or (settings.MODELS_DIR / "stance_distilled")
    max_len = max_len or model_max_len(model_dir)
    out = settings.REPLY_SENTIMENT_OUTPUT
    df = pd.read_parquet(out)
    todo = _to_score(df, col)
    if todo.any():
        logger.info("scoring %d replies with %s -> %s (max_len %d)", int(todo.sum()),
                    model_dir, col, max_len)
        df.loc[todo, col] = score_with_distilled(df.loc[todo, "text"].tolist(), model_dir,
                                                 max_len=max_len)
        df.to_parquet(out, index=False)
    return df


# ── 3. Local LLM (GPU) ─────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Drop a leading <think>...</think> block (Qwen3-style reasoning)."""
    return re.sub(r"<think>.*?</think>", "", text, count=1, flags=re.S).strip()


def local_llm_eval(df: pd.DataFrame, *, model_name: str = settings.LOCAL_LLM_MODEL,
                   n: int = settings.LOCAL_LLM_EVAL_N, batch_size: int = settings.LOCAL_LLM_BATCH,
                   max_new_tokens: int = settings.LOCAL_LLM_MAX_NEW_TOKENS,
                   thinking: bool = False,
                   seed: int = settings.DISTILL_SEED, holdout: float = settings.DISTILL_HOLDOUT,
                   out_dir: Path | None = None) -> dict:
    """Score a sample of the SAME holdout split the distillation uses with an
    open instruct model in 4-bit, using the Claude prompt verbatim.
    `thinking` toggles the chat template's reasoning mode where the model
    supports it (Qwen3 `enable_thinking`); the <think> block is stripped
    before parsing. Give thinking runs a max_new_tokens in the hundreds."""
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
    def render(text, user):
        msgs = [{"role": "user", "content": llm_prompt(text, user)}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking)
        except TypeError:  # template without a thinking switch
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompts = [render(r.text, r.user) for r in sample.itertuples()]
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, seq in enumerate(gen):
            text = strip_thinking(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            data = parse_llm_json(text) or {}
            sc = data.get("score")
            scores.append(float(sc) if isinstance(sc, (int, float)) else np.nan)
            labels.append(data.get("label"))
            raws.append(text[:200])
        logger.info("local llm: %d/%d", min(i + batch_size, len(prompts)), len(prompts))
    sample = sample.assign(score_local=scores, label_local=labels, raw_local=raws)
    tag = model_name.replace("/", "_") + ("_think" if thinking else "")
    sample.to_parquet(out_dir / f"local_llm_{tag}.parquet", index=False)
    metrics = {
        "model": model_name, "thinking": thinking, "max_new_tokens": max_new_tokens, "n": int(len(sample)),
        "unparseable_rate": float(np.mean(np.isnan(scores))),
        "local_vs_teacher": agreement(sample["score_llm"].values, sample["score_local"].values),
        "local_by_tier": agreement_by_tier(sample, "score_llm", "score_local").reset_index().to_dict("records"),
    }
    if "score_transformer" in sample:
        metrics["roberta_valence_vs_teacher"] = agreement(sample["score_llm"].values, sample["score_transformer"].values)
    _write_json(out_dir / f"local_llm_{tag}.json", metrics)
    logger.info("local llm: %s", json.dumps(metrics["local_vs_teacher"]))
    return metrics


# ── Scoring arbitrary post files (e.g. Trump's Truth Social feed) ──

def score_post_file(inputs: list[Path], out: Path, model_dir: Path | None = None,
                    col: str = "score_opus_distilled", batch_size: int = 64,
                    max_len: int | None = None) -> pd.DataFrame:
    """Score every record in the given JSONL files with a distilled model and
    write id / user / tier / platform / created_at / text / <col> to `out`.
    Incremental: new ids are appended, and only rows without a `col` score
    are scored, so a new column scores every row and existing scores stay.
    A stored row without text whose record now has some (a ReTruth, once the
    feed is re-collected with the reblogged text) takes it and is rescored.
    max_len defaults to the model's training length."""
    model_dir = model_dir or (settings.MODELS_DIR / "stance_distilled")
    max_len = max_len or model_max_len(model_dir)
    rows = []
    for path in inputs:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({k: r.get(k) for k in ("id", "user", "tier", "platform", "created_at", "text")})
    df = pd.DataFrame(rows).drop_duplicates("id")
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    if not have.empty:
        fresh = df.set_index(df["id"].astype(str))["text"]
        fresh = fresh[~fresh.index.duplicated()]
        now = have["id"].astype(str).map(fresh)
        gained = ~have["text"].map(has_text) & now.map(has_text)
        if gained.any():
            # Its old scores were of no text (the "." stand-in): drop them all.
            have.loc[gained, "text"] = now[gained]
            have.loc[gained, [c for c in have if c.startswith("score_")]] = np.nan
            logger.info("%d stored posts without text now have some: rescoring them",
                        int(gained.sum()))
    new = df[~df["id"].astype(str).isin(set(have["id"].astype(str)))] if not have.empty else df
    result = pd.concat([have, new], ignore_index=True) if not have.empty else new
    todo = _to_score(result, col)
    if todo.any():
        logger.info("scoring %d posts from %d file(s) -> %s (max_len %d)", int(todo.sum()),
                    len(inputs), col, max_len)
        result.loc[todo, col] = score_with_distilled(result.loc[todo, "text"].fillna("").tolist(),
                                                     model_dir, batch_size, max_len)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    return result


# ── Reply-domain check: Opus on the sampled replies vs the distilled model ──

def reply_teacher_check(*, model: str = settings.TEACHER_CHECK_MODEL, max_tokens: int = settings.TEACHER_MAX_TOKENS,
                        effort: str = settings.TEACHER_EFFORT, concurrency: int = settings.LLM_CONCURRENCY,
                        col: str = "score_opus_distilled") -> dict:
    """Score the stratified reply sample (stance_sample.parquet) with the
    teacher using the broadcaster prompt, then measure how well the reply
    column `col` in reply_sentiment.parquet reproduces it -- the domain-shift
    number for a model trained on broadcaster posts. Cached per reply id."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv
    from src.analysis.sentiment import score_llm
    from src.analysis.event_study import STANCE_OUTPUT
    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)  # host runs need the key

    sample = pd.read_parquet(STANCE_OUTPUT)
    sample = sample[sample["stance"] != "media_only"][["id", "tracked_slug", "user", "text"]]
    out = settings.PROCESSED_DIR / f"teacher_labels_replies_{model.replace('/', '_')}.parquet"
    cached = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    have = set(cached["id"].astype(str)) if not cached.empty else set()
    todo = sample[~sample["id"].astype(str).isin(have)]
    logger.info("reply teacher check: %d sampled, %d cached, %d to score with %s",
                len(sample), len(have), len(todo), model)
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(score_llm, r.text, r.user, "Iran war", model, max_tokens, effort): r.id
                for r in todo.itertuples()}
        for f in as_completed(futs):
            try:
                sc, lab = f.result()
            except Exception as e:
                logger.warning("teacher call failed for %s: %s", futs[f], str(e)[:120])
                sc, lab = None, None
            results.append({"id": futs[f], "score_teacher": sc, "label_teacher": lab})
    new = pd.DataFrame(results)
    labels = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
    labels = labels[labels["score_teacher"].notna()]
    labels.to_parquet(out, index=False)

    replies = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    replies["id"] = replies["id"].astype(str)
    labels["id"] = labels["id"].astype(str)
    j = sample.assign(id=sample["id"].astype(str)).merge(labels, on="id").merge(
        replies[["id", col, "score_transformer"]], on="id", how="left")
    j["tier"] = j["tracked_slug"]  # per-post breakdown reuses the per-tier helper
    report = {"model": model, "n": int(len(j)),
              "distilled_vs_teacher": agreement(j["score_teacher"].values, j[col].values),
              "roberta_valence_vs_teacher": agreement(j["score_teacher"].values, j["score_transformer"].values),
              "by_post": agreement_by_tier(j, "score_teacher", col).reset_index().to_dict("records")}
    _write_json(settings.PROCESSED_DIR / f"reply_teacher_check_{model.replace('/', '_')}.json", report)
    return report


def reply_label_frame(model: str = settings.TEACHER_CHECK_MODEL, label_col: str = "score_opus") -> pd.DataFrame:
    """The teacher-labelled reply sample as extra training rows:
    id / text / user / tier=reply_<post> / <label_col>."""
    from src.analysis.event_study import STANCE_OUTPUT
    labels = pd.read_parquet(settings.PROCESSED_DIR / f"teacher_labels_replies_{model.replace('/', '_')}.parquet")
    labels["id"] = labels["id"].astype(str)
    smp = pd.read_parquet(STANCE_OUTPUT)
    smp["id"] = smp["id"].astype(str)
    j = smp[["id", "tracked_slug", "user", "text"]].merge(labels, on="id")
    return pd.DataFrame({"id": j["id"], "text": j["text"], "user": j["user"],
                         "tier": "reply_" + j["tracked_slug"].astype(str), label_col: j["score_teacher"]})



# ── Teacher-label runs on the host: cache, cap, estimate ───────────

TEACHER_LABELS = ("negative", "neutral", "positive")


def valid_score(sc) -> bool:
    """A finite number in [-1, 1]; a bool is not one (True would parse as 1)."""
    return (isinstance(sc, (int, float)) and not isinstance(sc, bool)
            and math.isfinite(sc) and -1.0 <= sc <= 1.0)


def _label(sc: float, lab) -> str:
    """The model's label when it gave one of ours, else the score's sign."""
    if lab in TEACHER_LABELS:
        return lab
    return "positive" if sc > NEUTRAL_BAND else "negative" if sc < -NEUTRAL_BAND else "neutral"


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Temp file, then rename: a run killed mid-write leaves the old cache whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# Prompt caching on the Claude API: a 5-minute cache write costs 1.25x the
# input price, a read 0.1x; Opus 5 caches prefixes of 512 tokens or more.
CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.1
CACHE_MIN_TOKENS = 512


def _est_tokens(chars: int) -> int:
    return math.ceil(chars / settings.TEACHER_EST_CHARS_PER_TOKEN)


def estimate_direct_cost(prompts: list[str], cache_chars: list[int] | None = None,
                         concurrency: int = settings.LLM_CONCURRENCY) -> dict:
    """Dollars for one direct-API call per prompt, making none: input tokens
    from the prompt's characters (settings.TEACHER_EST_CHARS_PER_TOKEN),
    output tokens per call and $/MTok from relabel (Opus 5 list prices,
    without relabel's batch discount). usd counts a prompt's first
    cache_chars characters as a cached prefix when it reaches
    CACHE_MIN_TOKENS, in the order given: a call reads it once a call sharing
    it started `concurrency` places earlier (done by then), else writes it.
    usd_no_cache is the ceiling."""
    from src.analysis.relabel import EST_OUT_TOKENS, PRICE_IN, PRICE_OUT

    cache_chars = cache_chars if cache_chars is not None else [0] * len(prompts)
    tin = sum(_est_tokens(len(p)) for p in prompts)
    tout = EST_OUT_TOKENS * len(prompts)
    first: dict[str, int] = {}         # prefix -> place of the first call that sent it
    billed = 0.0                       # input tokens at the full input price
    cached_calls = 0
    for i, (p, c) in enumerate(zip(prompts, cache_chars)):
        pre = _est_tokens(c) if c else 0
        if pre < CACHE_MIN_TOKENS:
            billed += _est_tokens(len(p))
            continue
        start = first.setdefault(p[:c], i)
        mult = CACHE_READ_MULT if i - start >= concurrency else CACHE_WRITE_MULT
        billed += pre * mult + _est_tokens(len(p) - c)
        cached_calls += 1
    return {"calls": len(prompts), "input_tokens": tin, "output_tokens": tout,
            "prompt_chars": sum(len(p) for p in prompts), "cached_prefix_calls": cached_calls,
            "usd": (billed * PRICE_IN + tout * PRICE_OUT) / 1e6,
            "usd_no_cache": (tin * PRICE_IN + tout * PRICE_OUT) / 1e6,
            "price_in_per_mtok": PRICE_IN, "price_out_per_mtok": PRICE_OUT}


def price_tokens(t: dict) -> float:
    """Direct-API dollars for billed tokens keyed like Spend.FIELDS (missing
    keys count 0): cache writes and reads at their multiples of the input
    price, Opus 5 list prices from relabel."""
    from src.analysis.relabel import PRICE_IN, PRICE_OUT
    billed_in = (t.get("input_tokens", 0)
                 + t.get("cache_creation_input_tokens", 0) * CACHE_WRITE_MULT
                 + t.get("cache_read_input_tokens", 0) * CACHE_READ_MULT)
    return (billed_in * PRICE_IN + t.get("output_tokens", 0) * PRICE_OUT) / 1e6


def with_pilot_and_batch_prices(est: dict) -> dict:
    """estimate_direct_cost's figures plus two more, still making no call:
    the same token estimate at batch prices (relabel.BATCH_DISCOUNT), and a
    pilot-calibrated figure at what settings.TEACHER_V2_PILOT was really
    billed: input dollars per prompt character, output dollars per call, so
    it follows the prompts' length rather than the characters-per-token
    guess. Each direct and at batch prices."""
    from src.analysis.relabel import BATCH_DISCOUNT

    p = settings.TEACHER_V2_PILOT
    tokens = {f: p[f] for f in Spend.FIELDS}
    usd_in_per_char = price_tokens({**tokens, "output_tokens": 0}) / p["prompt_chars"]
    usd_out_per_call = price_tokens({"output_tokens": p["output_tokens"]}) / p["calls"]
    pilot = est["prompt_chars"] * usd_in_per_char + est["calls"] * usd_out_per_call
    return {**est, "batch_discount": BATCH_DISCOUNT,
            "usd_batch": est["usd"] * BATCH_DISCOUNT,
            "usd_batch_no_cache": est["usd_no_cache"] * BATCH_DISCOUNT,
            "pilot_usd_per_call": price_tokens(tokens) / p["calls"],
            "usd_pilot": pilot, "usd_pilot_batch": pilot * BATCH_DISCOUNT}


def _refuse_over_cap(n: int, cap: int) -> None:
    if n > cap:
        raise RuntimeError(f"{n} calls to make, over the cap of {cap} in config/settings.py: "
                           "raise it deliberately, or run a pilot with --limit")


class Spend:
    """Tokens the API billed across a run's calls (added from worker
    threads), priced like estimate_direct_cost, so a pilot shows what a call
    really costs and whether the prefix cache engaged."""

    FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
              "output_tokens")

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = dict.fromkeys(self.FIELDS, 0)

    def add(self, usage) -> None:
        with self._lock:
            self.calls += 1
            for f in self.FIELDS:
                self.tokens[f] += int(getattr(usage, f, 0) or 0)

    def usd(self) -> float:
        return price_tokens(self.tokens)

    def summary(self) -> str:
        t = self.tokens
        return (f"{self.calls} calls billed: {t['input_tokens']:,} input, "
                f"{t['cache_creation_input_tokens']:,} cache-write, "
                f"{t['cache_read_input_tokens']:,} cache-read, {t['output_tokens']:,} output tokens "
                f"≈ ${self.usd():.2f}")


def _teacher_scorer(model: str, max_tokens: int, effort: str):
    """(prompt, cache_chars) -> (score, label) through the API, the first
    cache_chars characters sent as a cached block; `call.spend` tallies the
    tokens billed. Without a key every call would come back empty and the
    run would look like a run of failures, so it stops here instead."""
    from dotenv import load_dotenv

    from src.analysis.sentiment import score_prompt
    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)  # host runs need the key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (.env)")
    spend = Spend()

    def call(prompt: str, cache_chars: int = 0):
        return score_prompt(prompt_content(prompt, cache_chars), model, max_tokens, effort,
                            on_usage=spend.add)
    call.spend = spend
    return call


def prompt_content(prompt: str, cache_chars: int = 0):
    """The user message content: the plain prompt, or two text blocks with a
    cache breakpoint after the shared prefix. The model reads the same text."""
    if not cache_chars:
        return prompt
    return [{"type": "text", "text": prompt[:cache_chars], "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt[cache_chars:]}]


def read_label_cache(path: Path, prompt_version: str | None = None) -> pd.DataFrame:
    """A label cache (empty frame if none yet). Refuses a cache written with
    another prompt: a new prompt gets a new file, never a mixed one."""
    if not path.exists():
        return pd.DataFrame(columns=["id", "score_teacher", "label_teacher"])
    cached = pd.read_parquet(path)
    cached["id"] = cached["id"].astype(str)
    if prompt_version and "prompt_version" in cached:
        other = sorted(set(cached["prompt_version"].dropna()) - {prompt_version})
        if other:
            raise ValueError(f"{path.name} holds labels from prompt {', '.join(other)}, "
                             f"not {prompt_version}: give the new prompt its own file")
    return cached


def run_labels(todo: pd.DataFrame, cached: pd.DataFrame, out: Path, scorer=None, *, cap: int,
               model: str = settings.TEACHER_CHECK_MODEL, max_tokens: int = settings.TEACHER_MAX_TOKENS,
               effort: str = settings.TEACHER_EFFORT, concurrency: int = settings.LLM_CONCURRENCY,
               save_every: int = settings.LLM_SAVE_EVERY_N) -> pd.DataFrame:
    """Send todo["prompt"] (with todo["cache_chars"] when present) to `scorer`
    (default: `model` through the API) and add the good labels to the cache
    at `out`: todo's columns less the prompt, plus score_teacher /
    label_teacher. Refuses before any call when todo holds more than `cap`
    rows. Saves every `save_every` completions and on the way out (Ctrl-C
    included, queued calls cancelled), so a killed run keeps what it paid
    for. Failed, unparseable or out-of-range answers are not saved: the next
    run retries them. Logs the tokens billed when the scorer tallies them
    (the API scorer does). Returns the whole cache."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _refuse_over_cap(len(todo), cap)
    if todo.empty:
        return cached
    scorer = scorer or _teacher_scorer(model, max_tokens, effort)
    sent = ["prompt", "cache_chars"]
    records = todo.drop(columns=[c for c in sent if c in todo]).to_dict("records")
    cols = [c for c in todo.columns if c not in sent] + ["score_teacher", "label_teacher"]
    cache_chars = todo["cache_chars"].tolist() if "cache_chars" in todo else [0] * len(todo)
    rows: list[dict] = []
    failed = 0

    def save() -> pd.DataFrame:
        new = pd.DataFrame(rows, columns=cols)
        frame = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        if len(frame):
            write_parquet_atomic(frame, out)
        return frame

    ex = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futs = {ex.submit(scorer, p, int(c)): i
                for i, (p, c) in enumerate(zip(todo["prompt"], cache_chars))}
        for k, f in enumerate(as_completed(futs), 1):
            rec = records[futs[f]]
            try:
                sc, lab = f.result()
            except Exception as e:  # noqa: BLE001 -- one bad call must not sink the run
                logger.warning("teacher call failed for %s: %s", rec["id"], str(e)[:120])
                sc, lab = None, None
            if valid_score(sc):
                rows.append({**rec, "score_teacher": float(sc), "label_teacher": _label(sc, lab)})
            else:
                failed += 1
                if sc is not None:
                    logger.warning("teacher score out of range for %s: %r", rec["id"], sc)
            if k % save_every == 0:
                save()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        frame = save()
        if getattr(scorer, "spend", None) is not None:
            logger.info("teacher spend: %s", scorer.spend.summary())
    logger.info("teacher labels: %d new, %d failed (retried next run) -> %s", len(rows), failed, out)
    return frame


def stratified_order(df: pd.DataFrame, by: str, seed: int) -> pd.DataFrame:
    """Rows in a seeded random order that deals one row from each group in
    turn, so the first N rows of a pilot cover every group."""
    rng = np.random.default_rng(seed)
    d = df.iloc[rng.permutation(len(df))]
    rank = d.groupby(by, sort=False).cumcount()
    return d.assign(_round=rank.values).sort_values("_round", kind="stable").drop(columns="_round")


def _vs_teacher(g: pd.DataFrame, col: str) -> dict:
    """agreement() of `col` against the teacher, plus both means and the
    mean difference (model minus teacher) on the rows that have both."""
    ok = g[["score_teacher", col]].dropna()
    return {**agreement(g["score_teacher"].values, g[col].values),
            "mean_teacher": float(ok["score_teacher"].mean()) if len(ok) else float("nan"),
            "mean_model": float(ok[col].mean()) if len(ok) else float("nan"),
            "mean_diff": float((ok[col] - ok["score_teacher"]).mean()) if len(ok) else float("nan")}


def _shares(y: pd.Series) -> dict:
    """Mean and pro / anti shares (beyond +-NEUTRAL_BAND) of a label column."""
    return {"mean": float(y.mean()), "pro": float((y > NEUTRAL_BAND).mean()),
            "anti": float((y < -NEUTRAL_BAND).mean())}


# ── Reply relabel v2: the full reply, with the post it answers ─────

# v1 (reply_teacher_check) sent Opus the broadcaster prompt with the
# reply's first 100 characters (stance_sample keeps only text[:100]) and
# never showed it the post replied to: replies that only cheer under the
# ceasefire, hold-off and deal posts came back pro-war about 75% of the
# time. v2 sends the post and the full reply, asks for the author's
# position on the war rather than sentiment, and leaves out the author's
# handle (private individuals; a handle invites a prior about the author).
# The JSON is the broadcaster prompt's, so parse_llm_json reads it.
# The reply comes last: everything before it is the same for every reply to
# a post (~90% of the characters), so it is sent as a cached prefix.
REPLY_TEACHER_V2_PROMPT_VERSION = "reply-v2-2026-10-04"
REPLY_TEACHER_V2_PREFIX = (
    "Below is a post President Trump made on Truth Social, followed by a reply to it "
    "from another user.\n\n"
    'Trump\'s post: """{parent}"""\n\n'
    "What is the reply author's position on the war, meaning U.S. military action "
    "against Iran? Score it from -1.0 (against: wants peace, de-escalation, a deal, "
    "or holding off) to +1.0 (for: wants strikes, escalation, or regime change). "
    "Score 0 when the reply takes no position on the war or its position can't be "
    "told.\n\n"
    "Judge the position, not the tone: anger can be for the war, and warmth can be "
    "against it. A reply that only agrees with or praises the post takes the post's "
    "position on the war, read in context: praising a decision to strike leans for; "
    "praising a ceasefire, a deal or holding off leans against. Praise of Trump with "
    "no bearing on the war is 0.\n\n"
    "Respond with ONLY valid JSON: "
    '{{"score": <float from -1.0 (against the war) to 1.0 (for the war)>, '
    '"label": "<negative|neutral|positive>", '
    '"reasoning": "<one sentence>"}}\n\n'
)
REPLY_TEACHER_V2_SUFFIX = 'Reply: """{reply}"""'
REPLY_LABEL_VERSIONS = ("v1", "v2")


def reply_teacher_v2_prompt(parent: str, reply: str) -> tuple[str, int]:
    """(prompt, length of its cacheable prefix: everything before the reply)."""
    prefix = REPLY_TEACHER_V2_PREFIX.format(parent=parent)
    return prefix + REPLY_TEACHER_V2_SUFFIX.format(reply=reply), len(prefix)


def reply_labels_path(model: str = settings.TEACHER_CHECK_MODEL, version: str = "v1") -> Path:
    """The reply teacher labels: v1 teacher_labels_replies_<model>.parquet,
    v2 teacher_labels_replies_v2_<model>.parquet."""
    if version not in REPLY_LABEL_VERSIONS:
        raise ValueError(f"reply label version {version!r} is not one of {REPLY_LABEL_VERSIONS}")
    tag = model.replace("/", "_")
    infix = "" if version == "v1" else f"{version}_"
    return settings.PROCESSED_DIR / f"teacher_labels_replies_{infix}{tag}.parquet"


def tracked_post_texts(slugs, cache_path: Path | None = None) -> dict[str, tuple[str, str]]:
    """{slug: (post id, text)} for the Trump posts behind tracked slugs,
    resolved with config/tracked_posts.py against the cached feed. Raises,
    naming them, for slugs whose post or text can't be found: a reply judged
    against the wrong post would be a paid wrong label."""
    from config.tracked_posts import TRACKED_POSTS, resolve_post_ids

    cache_path = cache_path or settings.TRUTH_SOCIAL_RAW_DIR / "realDonaldTrump.jsonl"
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} is missing: run collect-truth")
    slugs = sorted(set(slugs))
    wanted = [tp for tp in TRACKED_POSTS if tp.slug in slugs]
    ids = {k: str(v) for k, v in resolve_post_ids(wanted, cache_path).items() if v} if wanted else {}
    texts: dict[str, str] = {}
    with open(cache_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("id")) in ids.values():
                texts[str(r["id"])] = r.get("text")
    missing = [s for s in slugs if s not in ids or not has_text(texts.get(ids[s]))]
    if missing:
        raise LookupError(f"no Trump post text for tracked post(s) {', '.join(missing)} in "
                          f"{cache_path.name}: pin post_id in config/tracked_posts.py or re-collect")
    return {s: (ids[s], texts[ids[s]]) for s in slugs}


def reply_v2_inputs() -> pd.DataFrame:
    """Every reply in the stance sample with its full text from
    reply_sentiment.parquet (the sample keeps text[:100]), a has_text flag
    on that full text, and for those with text the parent post's text and
    the v2 prompt (with cache_chars, its cacheable prefix length).
    input_chars is the reply's length as sent."""
    from src.analysis.event_study import STANCE_OUTPUT

    smp = pd.read_parquet(STANCE_OUTPUT, columns=["id", "tracked_slug"])
    smp["id"] = smp["id"].astype(str)
    smp = smp.drop_duplicates("id")
    reps = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    reps["id"] = reps["id"].astype(str)
    keep = ["id", "text"] + (["parent_id"] if "parent_id" in reps else [])
    reps = reps.loc[reps["id"].isin(set(smp["id"])), keep]
    reps = reps.drop_duplicates()                    # a few ids were collected twice, identically
    clash = reps["id"].duplicated(keep=False)
    if clash.any():
        raise ValueError(f"{reps.loc[clash, 'id'].nunique()} sampled reply ids have more than one "
                         f"row with different contents in {settings.REPLY_SENTIMENT_OUTPUT.name}")
    j = smp.merge(reps, on="id", how="left", indicator=True)
    lost = j["_merge"] != "both"
    if lost.any():
        raise LookupError(f"{int(lost.sum())} sampled replies are not in "
                          f"{settings.REPLY_SENTIMENT_OUTPUT.name}: their full text is unknown")
    j = j.drop(columns="_merge")
    j["has_text"] = j["text"].map(has_text).astype(bool)
    posts = tracked_post_texts(j.loc[j["has_text"], "tracked_slug"].unique())
    if "parent_id" in j:
        pid = j["tracked_slug"].map({s: p for s, (p, _) in posts.items()})
        off = j["has_text"] & j["parent_id"].notna() & (j["parent_id"].astype(str) != pid)
        if off.any():
            raise ValueError("replies whose parent_id is not the tracked post resolved for their "
                             f"slug: {', '.join(sorted(j.loc[off, 'tracked_slug'].unique()))}")
    j["parent_text"] = j["tracked_slug"].map({s: t for s, (_, t) in posts.items()})
    built = [reply_teacher_v2_prompt(p, r) if ok else (None, 0)
             for p, r, ok in zip(j["parent_text"], j["text"], j["has_text"])]
    j["prompt"] = [b[0] for b in built]
    j["cache_chars"] = [b[1] for b in built]
    j["input_chars"] = j["text"].fillna("").str.len()
    j["prompt_version"] = REPLY_TEACHER_V2_PROMPT_VERSION
    return j


def reply_v2_todo(model: str = settings.TEACHER_CHECK_MODEL, limit: int | None = None,
                  seed: int = settings.TEACHER_SAMPLE_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(inputs, cached v2 labels, rows still to label). The to-do rows come
    in stratified_order across posts; `limit` keeps the first N (a pilot)."""
    inputs = reply_v2_inputs()
    cached = read_label_cache(reply_labels_path(model, "v2"), REPLY_TEACHER_V2_PROMPT_VERSION)
    todo = inputs[inputs["has_text"] & ~inputs["id"].isin(set(cached["id"]))]
    todo = stratified_order(todo, "tracked_slug", seed)
    if limit is not None:
        todo = todo.head(limit)
    return inputs, cached, todo[["id", "tracked_slug", "prompt", "cache_chars", "input_chars",
                                 "prompt_version"]]


def reply_teacher_v2_estimate(model: str = settings.TEACHER_CHECK_MODEL, limit: int | None = None,
                              seed: int = settings.TEACHER_SAMPLE_SEED) -> dict:
    """What `reply-teacher-check --v2` would send, priced direct, at batch
    prices and at the pilot's measured rates. No API call."""
    inputs, cached, todo = reply_v2_todo(model, limit, seed)
    eligible = inputs[inputs["has_text"]]
    est = estimate_direct_cost(todo["prompt"].tolist(), todo["cache_chars"].tolist())
    return {"model": model, "sampled": len(inputs), "no_text": int((~inputs["has_text"]).sum()),
            "with_text": len(eligible), "cached": int(eligible["id"].isin(set(cached["id"])).sum()),
            "todo": len(todo), "cap": settings.REPLY_TEACHER_V2_MAX_CALLS,
            "todo_by_post": {str(k): int(v) for k, v in todo.groupby("tracked_slug").size().items()},
            **with_pilot_and_batch_prices(est)}


def reply_teacher_check_v2(*, model: str = settings.TEACHER_CHECK_MODEL,
                           max_tokens: int = settings.TEACHER_MAX_TOKENS,
                           effort: str = settings.TEACHER_EFFORT,
                           concurrency: int = settings.LLM_CONCURRENCY,
                           col: str = "score_opus_distilled", limit: int | None = None,
                           seed: int = settings.TEACHER_SAMPLE_SEED, scorer=None) -> dict:
    """Label the stance sample's replies with text using the v2 prompt (same
    model, effort and max_tokens as v1) into
    teacher_labels_replies_v2_<model>.parquet, then report. Incremental and
    capped (settings.REPLY_TEACHER_V2_MAX_CALLS); the v1 file is not touched.
    `scorer` ((prompt, cache_chars) -> (score, label)) replaces the API in tests.
    Refuses while a v2 batch is open: it would pay again for the batch's ids."""
    refuse_open_batch("reply_v2", "a direct run")
    _, cached, todo = reply_v2_todo(model, limit, seed)
    logger.info("reply teacher v2: %d cached, %d to label with %s", len(cached), len(todo), model)
    run_labels(todo, cached, reply_labels_path(model, "v2"), scorer,
               cap=settings.REPLY_TEACHER_V2_MAX_CALLS, model=model, max_tokens=max_tokens,
               effort=effort, concurrency=concurrency)
    return reply_teacher_v2_report(model, col)


def reply_teacher_v2_report(model: str = settings.TEACHER_CHECK_MODEL,
                            col: str = "score_opus_distilled") -> dict:
    """v1's agreement metrics for the v2 labels, plus v1 against v2 on the
    ids both label: overall, by whether v1 saw the whole reply (100
    characters or fewer), and per post the mean and pro / anti shares under
    each. Written to reply_teacher_check_v2_<model>.json."""
    out = settings.PROCESSED_DIR / f"reply_teacher_check_v2_{model.replace('/', '_')}.json"
    v2 = read_label_cache(reply_labels_path(model, "v2"))
    if v2.empty:
        report = {"model": model, "prompt_version": REPLY_TEACHER_V2_PROMPT_VERSION, "n": 0}
        _write_json(out, report)
        return report
    replies = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    replies["id"] = replies["id"].astype(str)
    replies = replies.drop_duplicates("id")[["id", col, "score_transformer"]]
    j = v2.merge(replies, on="id", how="left")
    j["tier"] = j["tracked_slug"]          # per-post breakdown reuses the per-tier helper
    report = {"model": model, "prompt_version": REPLY_TEACHER_V2_PROMPT_VERSION, "n": len(j),
              "distilled_vs_teacher": agreement(j["score_teacher"].values, j[col].values),
              "roberta_valence_vs_teacher": agreement(j["score_teacher"].values,
                                                      j["score_transformer"].values),
              "by_post": agreement_by_tier(j, "score_teacher", col).reset_index().to_dict("records")}
    v1_path = reply_labels_path(model, "v1")
    if v1_path.exists():
        v1 = read_label_cache(v1_path).dropna(subset=["score_teacher"]).drop_duplicates("id")
        both = j.merge(v1[["id", "score_teacher"]].rename(columns={"score_teacher": "v1"}), on="id")
        both = both.rename(columns={"score_teacher": "v2"})
        report["v1_vs_v2"] = agreement(both["v1"].values, both["v2"].values)
        report["v1_vs_v2_by_input"] = [
            {"v1_input": name, **agreement(g["v1"].values, g["v2"].values)}
            for name, g in (("whole reply (<=100 chars)", both[both["input_chars"] <= 100]),
                            ("first 100 chars only", both[both["input_chars"] > 100]))]
        rows = []
        for post, g in [*both.groupby("tracked_slug"), ("ALL", both)]:
            s1, s2 = _shares(g["v1"]), _shares(g["v2"])
            rows.append({"post": post, "n": len(g), **{f"v1_{k}": v for k, v in s1.items()},
                         **{f"v2_{k}": v for k, v in s2.items()}})
        report["v1_vs_v2_by_post"] = rows
    _write_json(out, report)
    return report


# ── Trump-feed check: Opus on his own posts vs the distilled scorer ──

TRUMP_PROMPT_VERSION = "llm_prompt"   # the broadcaster prompt the distilled model learnt


def trump_labels_path(model: str = settings.TEACHER_CHECK_MODEL) -> Path:
    return settings.PROCESSED_DIR / f"teacher_check_trump_{model.replace('/', '_')}.parquet"


def trump_sample(n: int = settings.TRUMP_TEACHER_CHECK_N, seed: int = settings.TEACHER_SAMPLE_SEED,
                 col: str = "score_opus_distilled") -> pd.DataFrame:
    """n of Trump's feed posts with text and a `col` score: those with the
    smallest seeded hash of their id. A fixed random sample that a growing
    feed moves only where a new post hashes in, so a rerun relabels little."""
    df = pd.read_parquet(settings.TRUMP_FEED_STANCE)
    df["id"] = df["id"].astype(str)
    d = df[df["text"].map(has_text).astype(bool) & df[col].notna()].drop_duplicates("id")
    key = d["id"].map(lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return d.assign(_key=key).sort_values("_key").head(n).drop(columns="_key").reset_index(drop=True)


def trump_check_todo(n: int = settings.TRUMP_TEACHER_CHECK_N, model: str = settings.TEACHER_CHECK_MODEL,
                     seed: int = settings.TEACHER_SAMPLE_SEED,
                     col: str = "score_opus_distilled") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sample, cached labels, posts still to label), each to-do post with the
    broadcaster prompt relabel sends (llm_prompt with the feed's handle)."""
    from src.analysis.sentiment import llm_prompt

    sample = trump_sample(n, seed, col)
    cached = read_label_cache(trump_labels_path(model), TRUMP_PROMPT_VERSION)
    todo = sample[~sample["id"].isin(set(cached["id"]))]
    todo = pd.DataFrame({"id": todo["id"],
                         "prompt": [llm_prompt(t, u) for t, u in zip(todo["text"], todo["user"])],
                         "input_chars": todo["text"].str.len(),
                         "prompt_version": TRUMP_PROMPT_VERSION})
    return sample, cached, todo


def trump_check_estimate(n: int = settings.TRUMP_TEACHER_CHECK_N, model: str = settings.TEACHER_CHECK_MODEL,
                         seed: int = settings.TEACHER_SAMPLE_SEED, col: str = "score_opus_distilled") -> dict:
    """What `teacher-check --source trump` would send, priced direct, at batch
    prices and at the v2 pilot's measured rates. No API call."""
    sample, cached, todo = trump_check_todo(n, model, seed, col)
    return {"model": model, "sample": len(sample),
            "cached": int(sample["id"].isin(set(cached["id"])).sum()), "todo": len(todo),
            "cap": settings.TRUMP_TEACHER_CHECK_MAX_CALLS,
            **with_pilot_and_batch_prices(estimate_direct_cost(todo["prompt"].tolist()))}


def trump_teacher_check(*, n: int = settings.TRUMP_TEACHER_CHECK_N, model: str = settings.TEACHER_CHECK_MODEL,
                        max_tokens: int = settings.TEACHER_MAX_TOKENS, effort: str = settings.TEACHER_EFFORT,
                        concurrency: int = settings.LLM_CONCURRENCY, seed: int = settings.TEACHER_SAMPLE_SEED,
                        col: str = "score_opus_distilled", scorer=None) -> dict:
    """Opus-label the trump_sample with the X teacher's prompt and settings
    into teacher_check_trump_<model>.parquet (incremental, capped by
    settings.TRUMP_TEACHER_CHECK_MAX_CALLS), then report how well `col`
    reproduces it. `scorer` ((prompt, cache_chars) -> (score, label))
    replaces the API in tests. Refuses while a Trump batch is open."""
    refuse_open_batch("trump", "a direct run")
    _, cached, todo = trump_check_todo(n, model, seed, col)
    logger.info("trump teacher check: %d cached, %d to label with %s", len(cached), len(todo), model)
    run_labels(todo, cached, trump_labels_path(model), scorer, cap=settings.TRUMP_TEACHER_CHECK_MAX_CALLS,
               model=model, max_tokens=max_tokens, effort=effort, concurrency=concurrency)
    return trump_teacher_report(model, col)


def trump_teacher_report(model: str = settings.TEACHER_CHECK_MODEL, col: str = "score_opus_distilled",
                         topic_source: str | None = None) -> dict:
    """`col` against the Opus labels on Trump's posts: Pearson, MAE, sign
    flips, mean difference (model minus Opus), overall and by phase, then the
    same on war posts only (inference.war_flag). Written to
    teacher_check_trump_<model>.json."""
    from src.analysis.inference import PHASE_ORDER, assign_phase, war_flag

    topic_source = topic_source or settings.TOPIC_SOURCE
    labels = read_label_cache(trump_labels_path(model)).drop_duplicates("id")
    feed = pd.read_parquet(settings.TRUMP_FEED_STANCE)
    feed["id"] = feed["id"].astype(str)
    j = feed.drop_duplicates("id").merge(labels[["id", "score_teacher"]], on="id")
    j["phase"] = assign_phase(j["created_at"]).fillna("outside phases")
    j["war"] = war_flag(j, topic_source).values if len(j) else []

    def by_phase(d: pd.DataFrame) -> list[dict]:
        return [{"phase": p, **_vs_teacher(g, col)} for p, g in d.groupby("phase", sort=False)]

    order = {p: i for i, p in enumerate([*PHASE_ORDER, "outside phases"])}
    j = j.sort_values("phase", key=lambda s: s.map(order), kind="stable")
    war = j[j["war"]]
    report = {"model": model, "col": col, "topic_source": topic_source, "n": len(j),
              "all": _vs_teacher(j, col), "by_phase": by_phase(j),
              "war_posts": _vs_teacher(war, col), "war_by_phase": by_phase(war)}
    _write_json(settings.PROCESSED_DIR / f"teacher_check_trump_{model.replace('/', '_')}.json", report)
    return report


# ── Message Batches transport for the v2 reply and Trump-feed checks ──

# The direct runs' requests (prompt, cached-prefix blocks, model, max_tokens,
# effort) through the Message Batches API at half price, results within ~1 h.
# The 2026-10-04 v2 pilot found the prefix cache engaging for one parent post
# of eight, which put the direct runs at about twice their approval.
#   submit   one request per id still to label (custom_id = the id), within
#            the check's cap; state -> teacher_batch_<check>.json
#   status   poll the batch
#   collect  judge each answer as run_labels would, add the good ones to the
#            check's label cache, then write the check's report
# Each check keeps its own state file, open from just before the create call
# (status "submitting": a submit killed before it records the batch id still
# blocks a resend) until a collect has read a result for every submitted id.
# While it is open a new submit and the direct run refuse, so no id is paid
# for twice. A short read (a results file cut short, or another batch's) adds
# its good labels and leaves the batch open for a collect that reads it all.
# Errored, expired, unparseable and out-of-range answers, and answers to a
# prompt that has changed since the submit, are not cached: the next submit
# sends those ids again. A new submit archives the last state as
# teacher_batch_<check>_<batch id>.json, and a streamed collect keeps the raw
# results in teacher_batch_<check>_<batch id>_results.jsonl, so answers that
# are paid for but not kept as labels stay on disk.

BATCH_CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")   # the Batches API's custom_id rule


@dataclass(frozen=True)
class BatchCheck:
    """What the batch transport needs from a check: its prompt version, the
    settings name of its cap (read at call time), its label cache, its to-do
    rows ((model, params, limit) -> (cached labels, rows to label, each with
    its prompt)) and its report ((model, params) -> report, written)."""
    prompt_version: str
    cap_setting: str
    labels_path: Callable[[str], Path]
    todo: Callable[[str, dict, int | None], tuple[pd.DataFrame, pd.DataFrame]]
    report: Callable[[str, dict], dict]


def _reply_v2_batch_todo(model: str, params: dict, limit: int | None):
    _, cached, todo = reply_v2_todo(model, limit, params["seed"])
    return cached, todo


def _trump_batch_todo(model: str, params: dict, limit: int | None):
    _, cached, todo = trump_check_todo(params["n"], model, params["seed"], params["col"])
    return cached, todo if limit is None else todo.head(limit)


BATCH_CHECKS = {
    "reply_v2": BatchCheck(REPLY_TEACHER_V2_PROMPT_VERSION, "REPLY_TEACHER_V2_MAX_CALLS",
                           lambda m: reply_labels_path(m, "v2"), _reply_v2_batch_todo,
                           lambda m, p: reply_teacher_v2_report(m, p["col"])),
    "trump": BatchCheck(TRUMP_PROMPT_VERSION, "TRUMP_TEACHER_CHECK_MAX_CALLS",
                        lambda m: trump_labels_path(m), _trump_batch_todo,
                        lambda m, p: trump_teacher_report(m, p["col"])),
}


def _batch_client():
    """Anthropic client with the project's .env loaded (relabel's)."""
    from src.analysis.relabel import _client
    return _client()


def batch_state_path(check: str) -> Path:
    if check not in BATCH_CHECKS:
        raise ValueError(f"no batch check {check!r}: one of {', '.join(BATCH_CHECKS)}")
    return settings.PROCESSED_DIR / f"teacher_batch_{check}.json"


def batch_archive_path(check: str, batch_id: str, suffix: str = ".json") -> Path:
    """teacher_batch_<check>_<batch id>.json, a past batch's state, or with
    suffix "_results.jsonl" its raw results."""
    return batch_state_path(check).with_name(f"teacher_batch_{check}_{batch_id}{suffix}")


def read_batch_state(check: str) -> dict | None:
    path = batch_state_path(check)
    return json.loads(path.read_text()) if path.exists() else None


def open_batch(check: str) -> dict | None:
    """The check's batch state from just before its create call until a
    collect has read every submitted id's result, else None."""
    st = read_batch_state(check)
    return st if st and st.get("status") != "collected" else None


def open_batch_message(check: str, st: dict, doing: str) -> str:
    """Why `doing` has to wait for the open batch, and what to do."""
    name = batch_state_path(check).name
    if not st.get("batch_id"):
        return (f"{check}: the submit at {st.get('submitted_at')} of {st.get('n_submitted')} "
                f"requests stopped before it recorded a batch id, so {doing} could pay again for "
                f"them. If the Console's Batches page lists a batch created then with that many "
                f'requests, set "batch_id" to its id and "status" to "in_progress" in {name}, '
                f"then --batch collect; if it lists none, delete {name}.")
    short = st.get("short_read")
    left = f" ({short['missing']} submitted ids without a result yet)" if short else ""
    return (f"{check} batch {st['batch_id']} is {st['status']} and not collected{left}: "
            f"{doing} would pay again for its ids. Collect it first ({name})")


def refuse_open_batch(check: str, doing: str) -> None:
    st = open_batch(check)
    if st:
        raise RuntimeError(open_batch_message(check, st, doing))


def _submitted_state(check: str) -> dict:
    """The check's state with its batch id, read before any client is made,
    so 'no batch' needs no API key to be said."""
    st = read_batch_state(check)
    if not st:
        raise FileNotFoundError(f"no {check} batch submitted ({batch_state_path(check).name})")
    if not st.get("batch_id"):
        raise RuntimeError(open_batch_message(check, st, "a new submit"))
    return st


def prompt_sha(prompt: str) -> str:
    """A short fingerprint of a prompt: collect checks that the row's prompt
    is still the one its answer was given to."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def build_batch_requests(todo: pd.DataFrame, model: str, max_tokens: int, effort: str) -> list:
    """One request per to-do row, custom_id = its id, with the user content
    (prompt_content: the cached-prefix blocks when cache_chars is set),
    model, max_tokens and effort that score_prompt sends for it directly."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    ids = todo["id"].astype(str)
    bad = ~ids.map(lambda i: bool(BATCH_CUSTOM_ID.match(i))).astype(bool)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} ids are not valid batch custom_ids "
                         f"({BATCH_CUSTOM_ID.pattern})")
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} ids appear twice in the to-do rows")
    cache_chars = todo["cache_chars"].tolist() if "cache_chars" in todo else [0] * len(todo)
    reqs = []
    for i, prompt, c in zip(ids, todo["prompt"], cache_chars):
        params = MessageCreateParamsNonStreaming(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt_content(prompt, int(c))}])
        if effort:
            params["output_config"] = {"effort": effort}
        reqs.append(Request(custom_id=i, params=params))
    return reqs


def _api_refused(e: Exception) -> bool:
    """The API answered the create call with a client error (4xx): no batch
    was made. A timeout, a dropped connection or a 5xx leaves that unknown."""
    import anthropic
    return isinstance(e, anthropic.APIStatusError) and e.status_code < 500


def teacher_batch_submit(check: str, *, params: dict, model: str = settings.TEACHER_CHECK_MODEL,
                         max_tokens: int = settings.TEACHER_MAX_TOKENS,
                         effort: str = settings.TEACHER_EFFORT, limit: int | None = None,
                         client=None) -> dict | None:
    """Submit the check's rows still to label (cached ids are never sent)
    as one batch and record it. Refuses while the check's last batch is
    open, and over the check's cap, before any call. `params` fixes what
    collect recomputes the to-do rows and the report from (seed, col, and n
    for the Trump sample). The state is written as "submitting" before the
    create call and kept if the call dies without an answer, so a batch the
    API made is never silently orphaned; the last collected state is
    archived first. Returns the new state, or None when every row is
    labelled and nothing was sent."""
    spec = BATCH_CHECKS[check]
    refuse_open_batch(check, "a new batch")
    _, todo = spec.todo(model, params, limit)
    _refuse_over_cap(len(todo), getattr(settings, spec.cap_setting))
    if todo.empty:
        logger.info("teacher batch %s: nothing to submit, every row is labelled", check)
        return None
    reqs = build_batch_requests(todo, model, max_tokens, effort)
    client = client or _batch_client()
    path, prev = batch_state_path(check), read_batch_state(check)
    if prev and prev.get("batch_id"):
        _write_json(batch_archive_path(check, prev["batch_id"]), prev)
    ids = todo["id"].astype(str).tolist()
    st = {"check": check, "batch_id": None, "status": "submitting",
          "submitted_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
          "model": model, "max_tokens": max_tokens, "effort": effort,
          "prompt_version": spec.prompt_version, "params": params, "limit": limit,
          "labels_file": spec.labels_path(model).name, "n_submitted": len(reqs), "ids": ids,
          "prompt_sha256": dict(zip(ids, todo["prompt"].map(prompt_sha)))}
    _write_json(path, st)
    try:
        batch = client.messages.batches.create(requests=reqs)
    except Exception as e:
        if _api_refused(e):
            if prev:
                _write_json(path, prev)
            else:
                path.unlink(missing_ok=True)
        else:
            logger.error("teacher batch %s: the create call failed without an answer (%s); %s",
                         check, type(e).__name__, open_batch_message(check, st, "a new submit"))
        raise
    st.update(batch_id=batch.id, status=batch.processing_status)
    _write_json(path, st)
    logger.info("teacher batch %s: submitted %s with %d requests", check, batch.id, len(reqs))
    return st


def teacher_batch_status(check: str, client=None) -> dict:
    """Poll the check's batch into its state file (no call once collected)."""
    st = _submitted_state(check)
    if st["status"] == "collected":
        return st
    b = (client or _batch_client()).messages.batches.retrieve(st["batch_id"])
    st["status"] = b.processing_status
    st["counts"] = {k: int(getattr(b.request_counts, k)) for k in
                    ("processing", "succeeded", "errored", "canceled", "expired")}
    _write_json(batch_state_path(check), st)
    return st


def parse_batch_result(result) -> tuple[str, str, float | None, str | None]:
    """(id, outcome, score, label) for one batch result, judged as run_labels
    judges a direct answer: a finite score in [-1, 1], with the model's label
    when it is one of TEACHER_LABELS, else the score's sign. outcome is the
    API's (succeeded, errored, canceled, expired), or unparseable / out_of_range
    for an answer that is not kept."""
    from src.analysis.sentiment import parse_llm_json

    rid, outcome = str(result.custom_id), result.result.type
    if outcome != "succeeded":
        return rid, outcome, None, None
    text = next((b.text for b in result.result.message.content
                 if getattr(b, "type", "") == "text"), "")
    data = parse_llm_json(text)
    sc = data.get("score") if isinstance(data, dict) else None
    if not isinstance(sc, (int, float)):
        return rid, "unparseable", None, None
    if not valid_score(sc):
        return rid, "out_of_range", None, None
    return rid, outcome, float(sc), _label(sc, data.get("label"))


def _result_record(result) -> dict:
    """One streamed result as a results-JSONL line, keeping what this module
    reads (the text blocks and the usage), so read_batch_results reads it back."""
    res = result.result
    out = {"custom_id": str(result.custom_id), "result": {"type": res.type}}
    if res.type == "succeeded":
        msg = res.message
        usage = getattr(msg, "usage", None)
        out["result"]["message"] = {
            "content": [{"type": "text", "text": b.text} for b in msg.content
                        if getattr(b, "type", "") == "text"],
            "usage": ({f: int(getattr(usage, f, 0) or 0) for f in Spend.FIELDS}
                      if usage is not None else None)}
    return out


def write_batch_results(results: list, path: Path) -> None:
    """The results as a results JSONL (temp file, then rename)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(_result_record(r)) + "\n" for r in results))
    os.replace(tmp, path)


def read_batch_results(path: Path) -> list:
    """Results from a results JSONL (downloaded with `curl -C -` on the
    batch's results_url when the SDK stream breaks, or kept by a streamed
    collect), shaped like the SDK objects, usage included so the spend is
    counted. A line that does not parse (a download cut mid-line) is skipped
    with a warning: its id counts as missing, which keeps the batch open."""
    from types import SimpleNamespace as NS

    out, bad = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            cid, res = r["custom_id"], r["result"]
            kind = res["type"]
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
            continue
        msg = res.get("message") or {}
        content = [NS(type=b.get("type"), text=b.get("text", "")) for b in msg.get("content", [])]
        usage = NS(**msg["usage"]) if msg.get("usage") else None
        out.append(NS(custom_id=cid, result=NS(type=kind, message=NS(content=content, usage=usage))))
    if bad:
        logger.warning("%s: %d lines that do not parse were skipped", path.name, bad)
    return out


def teacher_batch_collect(check: str, *, client=None,
                          results_file: Path | None = None) -> tuple[dict, dict | None]:
    """Once the check's batch has ended, add its good answers to the check's
    label cache (the direct run's file, columns and prompt version) and,
    when the read held a result for every submitted id, mark the batch
    collected and write the check's report from the whole cache. A short
    read adds its good labels, records what is missing under "short_read"
    and leaves the batch open (report None): collect again from the stream
    or the whole file. Results for ids not submitted, already cached, no
    longer to label, or whose prompt changed since the submit are ignored
    and logged. A collected batch only gets its report rebuilt, with no
    call. Returns (state, report), report None while the batch is open.
    `results_file` reads a results JSONL instead of the stream; a streamed
    read is kept as one (batch_archive_path)."""
    from src.analysis.relabel import BATCH_DISCOUNT

    spec = BATCH_CHECKS[check]
    st = _submitted_state(check)
    if st["status"] == "collected":
        logger.info("teacher batch %s: %s already collected; rebuilding the report",
                    check, st["batch_id"])
        return st, spec.report(st["model"], st["params"])
    client = client or _batch_client()
    st = teacher_batch_status(check, client)
    if st["status"] != "ended":
        logger.info("teacher batch %s: %s still %s (%s)",
                    check, st["batch_id"], st["status"], st.get("counts"))
        return st, None
    if st["prompt_version"] != spec.prompt_version:
        raise ValueError(f"batch {st['batch_id']} was built with prompt {st['prompt_version']}, "
                         f"not {spec.prompt_version}: its labels do not belong in this cache")
    model, out = st["model"], spec.labels_path(st["model"])
    cached, todo = spec.todo(model, st["params"], None)
    now_sha = dict(zip(todo["id"].astype(str), todo["prompt"].map(prompt_sha)))
    sent_sha = st.get("prompt_sha256", {})
    keep = todo.drop(columns=[c for c in ("prompt", "cache_chars") if c in todo])
    rows_by_id = {str(r["id"]): r for r in keep.to_dict("records")}   # in to-do order
    submitted, have = set(st["ids"]), set(cached["id"].astype(str))
    if results_file is not None:
        results = read_batch_results(Path(results_file))
    else:
        results = list(client.messages.batches.results(st["batch_id"]))
        write_batch_results(results, batch_archive_path(check, st["batch_id"], "_results.jsonl"))
    spend, no_usage = Spend(), 0
    answered: set[str] = set()
    new: dict[str, dict] = {}
    failed: dict[str, int] = {}
    ignored = {"not_submitted": 0, "already_cached": 0, "repeated": 0, "no_longer_to_label": 0,
               "prompt_changed": 0}
    for r in results:
        rid, outcome, sc, lab = parse_batch_result(r)
        if rid not in submitted:
            ignored["not_submitted"] += 1       # another batch's result: not billed to this one
            continue
        if rid in answered:
            ignored["repeated"] += 1
            continue
        answered.add(rid)
        if r.result.type == "succeeded":        # billed, kept or not
            usage = getattr(r.result.message, "usage", None)
            if usage is None:
                no_usage += 1
            else:
                spend.add(usage)
        if rid in have:
            ignored["already_cached"] += 1      # e.g. an earlier short read
        elif rid not in rows_by_id:
            ignored["no_longer_to_label"] += 1  # left the sample since the submit
        elif rid in sent_sha and sent_sha[rid] != now_sha[rid]:
            ignored["prompt_changed"] += 1      # its text changed since the submit
        elif sc is None:
            failed[outcome] = failed.get(outcome, 0) + 1
        else:
            new[rid] = {**rows_by_id[rid], "score_teacher": sc, "label_teacher": lab}
    if any(ignored.values()):
        logger.info("teacher batch %s: ignored results %s",
                    check, {k: v for k, v in ignored.items() if v})
    if new:
        cols = list(keep.columns) + ["score_teacher", "label_teacher"]   # run_labels' columns
        add = pd.DataFrame([new[i] for i in rows_by_id if i in new], columns=cols)
        write_parquet_atomic(pd.concat([cached, add], ignore_index=True) if len(cached) else add,
                             out)
    read = {"labelled": len(new), "failed": sum(failed.values()), "failed_by_outcome": failed,
            "ignored": ignored}
    missing = len(submitted - answered)
    if missing:
        st["short_read"] = {**read, "missing": missing, "results_read": len(results),
                            "source": str(results_file) if results_file is not None else "stream"}
        _write_json(batch_state_path(check), st)
        logger.warning("teacher batch %s: %d of %d submitted ids have no result in this read "
                       "(a results file cut short, or another batch's?); %d labels added -> %s. "
                       "The batch stays open: collect again from the stream or the whole file",
                       check, missing, len(submitted), len(new), out)
        return st, None
    usd = spend.usd()
    st.pop("short_read", None)
    st.update(status="collected", collected=read,
              spend={**spend.tokens, "calls": spend.calls, "calls_without_usage": no_usage,
                     "usd_direct": usd, "usd_batch": usd * BATCH_DISCOUNT})
    _write_json(batch_state_path(check), st)
    unknown = f"; usage missing for {no_usage} calls (not counted)" if no_usage else ""
    logger.info("teacher batch %s: %d labels added, %d failed (the next submit sends them "
                "again) -> %s; %s at direct prices, ≈ $%.2f at batch prices%s", check, len(new),
                sum(failed.values()), out, spend.summary(), st["spend"]["usd_batch"], unknown)
    return st, spec.report(model, st["params"])
