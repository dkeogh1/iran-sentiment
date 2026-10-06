"""Reply-domain scorer: training rows, pair inputs, leave-one-post-out, the
final fit, its column and the cross-fit column, with stub tokenizers and
models (no torch weights, no GPU)."""

import json
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings
from src.analysis import stance_local as sl

# Tracked posts with a pinned post_id in config/tracked_posts.py.
POSTS = {
    "hold_off_attack": (
        "116597121700043134",
        "I have been asked to hold off on our planned attack.",
    ),
    "deal_complete": (
        "116750587569914985",
        "The Deal with Iran is now complete. Start your engines!",
    ),
    "strikes_resume_sep": (
        "117196950497702512",
        "We are striking Iranian targets near Hormuz tonight.",
    ),
}
LONG = "Thank you Mr President, " + "peace is the right call " * 10 + "END-OF-REPLY"


def _replies(n=6):
    rows = []
    for slug, (pid, _) in POSTS.items():
        for i in range(n):
            rows.append(
                {
                    "id": f"{slug}-{i}",
                    "tracked_slug": slug,
                    "parent_id": pid,
                    "user": f"handle_{i}",
                    "text": f"reply {i} to {slug}",
                    "score_transformer": 0.1 * i,
                    "score_opus_distilled": 0.05 * i - 0.1,
                    "metrics": {"likes": i},
                }
            )
    rows[0]["text"] = LONG  # longer than the stance sample's 100 characters
    rows[1]["text"] = ""  # image-only: never labelled, never scored
    return pd.DataFrame(rows)


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Replies, the Trump feed, v2 labels on every reply with text, and a base
    model dir that is only its recipe.txt."""
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    monkeypatch.setattr(settings, "TRUTH_SOCIAL_RAW_DIR", tmp_path / "ts")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "REPLY_TEACHER_LABELS_VERSION", "v2")
    (tmp_path / "ts").mkdir()
    base = tmp_path / "models" / settings.REPLY_DISTILL_BASE
    base.mkdir(parents=True)
    (base / "recipe.txt").write_text("deb-256-1e5-3")
    with open(tmp_path / "ts" / "realDonaldTrump.jsonl", "w") as f:
        f.writelines(json.dumps({"id": pid, "text": text}) + "\n" for pid, text in POSTS.values())

    def write(replies=None, labels=None):
        reps = _replies() if replies is None else replies
        reps.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
        if labels is None:
            lab = reps[reps["text"].map(sl.has_text)].drop_duplicates("id")
            labels = pd.DataFrame(
                {
                    "id": lab["id"],
                    "tracked_slug": lab["tracked_slug"],
                    "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
                    "score_teacher": np.linspace(-0.9, 0.9, len(lab)),
                    "label_teacher": "neutral",
                }
            )
        labels.to_parquet(sl.reply_labels_path(version="v2"), index=False)
        return reps

    return write


# ── 1. Training rows ───────────────────────────────────────────────


def test_training_rows_full_text_parent_and_one_row_per_id(files):
    reps = files()
    dup = reps.iloc[[3]]  # collected twice, identically
    files(pd.concat([reps, dup], ignore_index=True))
    rows = sl.reply_training_rows()
    assert rows["id"].is_unique and len(rows) == len(reps) - 1  # the image-only reply is out
    assert list(rows.columns) == [
        "id",
        "tracked_slug",
        "text",
        "parent_text",
        "score_teacher",
        settings.REPLY_DISTILL_BASE_COL,
    ]
    long = rows.set_index("id").loc["hold_off_attack-0"]
    assert long["text"] == LONG and long["text"].endswith("END-OF-REPLY")  # not the sample's [:100]
    for slug, (_, text) in POSTS.items():
        assert (rows.loc[rows["tracked_slug"] == slug, "parent_text"] == text).all()


def test_training_rows_refuse_clashing_missing_or_moved_replies(files):
    reps = files()
    clash = reps.iloc[[3]].assign(text="a different reply")
    labels = pd.read_parquet(sl.reply_labels_path(version="v2"))
    files(pd.concat([reps, clash], ignore_index=True), labels)
    with pytest.raises(ValueError, match="more than one"):
        sl.reply_training_rows()
    files(reps[reps["id"] != "deal_complete-2"], labels)  # a labelled reply went missing
    with pytest.raises(LookupError):
        sl.reply_training_rows()
    files(
        reps,
        labels.assign(
            tracked_slug=lambda d: d["tracked_slug"].where(
                d["id"] != "deal_complete-2", "hold_off_attack"
            )
        ),
    )
    with pytest.raises(ValueError, match="another post"):
        sl.reply_training_rows()


# ── 2. Pair inputs ─────────────────────────────────────────────────


class WordTok:
    """Whitespace tokenizer with pair specials [CLS] a [SEP] b [SEP], whose
    only_second truncation raises when the first text alone overflows, as
    the fast tokenizers do."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def num_special_tokens_to_add(pair=False):
        return 3 if pair else 2

    @staticmethod
    def _spans(t):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]

    def __call__(
        self,
        texts,
        pairs=None,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        truncation=False,
        max_length=None,
        **kw,
    ):
        self.calls.append({"pairs": pairs is not None, "truncation": truncation, **kw})
        out = {"input_ids": [], "offset_mapping": []}
        for i, t in enumerate(texts):
            spans = self._spans(t)
            a = [t[s:e] for s, e in spans]
            if not add_special_tokens:
                out["input_ids"].append(a)
                out["offset_mapping"].append(spans)
                continue
            if pairs is None:
                out["input_ids"].append(
                    ["[CLS]", *(a[: max_length - 2] if truncation else a), "[SEP]"]
                )
                continue
            b = pairs[i].split()
            over = len(a) + len(b) + 3 - max_length
            if over > 0:
                assert truncation == "only_second"
                if over > len(b):
                    raise RuntimeError("Truncation error: Sequence to truncate too short")
                b = b[: len(b) - over]
            out["input_ids"].append(["[CLS]", *a, "[SEP]", *b, "[SEP]"])
        if not return_offsets_mapping:
            out.pop("offset_mapping")
        return out


def _segments(ids):
    first = ids.index("[SEP]")
    return ids[1:first], ids[first + 1 : -1]


def test_ctx_pair_keeps_the_reply_and_cuts_the_post(monkeypatch):
    monkeypatch.setattr(settings, "REPLY_CTX_MIN_PARENT_TOKENS", 4)
    tok = WordTok()
    parent = " ".join(f"p{i}" for i in range(30))
    short, long = "short reply here", " ".join(f"r{i}" for i in range(40))
    enc = sl.reply_inputs(tok, [short, long], [parent, parent], "ctx", 20)
    r0, p0 = _segments(enc["input_ids"][0])
    assert r0 == short.split() and p0 == parent.split()[:14]  # the post is cut, not the reply
    assert len(enc["input_ids"][0]) == 20
    r1, p1 = _segments(enc["input_ids"][1])
    assert r1 == long.split()[:13] and p1 == parent.split()[:4]  # overlong reply cut to leave 4
    assert tok.calls[-1]["pairs"] and tok.calls[-1]["truncation"] == "only_second"
    with pytest.raises(RuntimeError, match="too short"):  # what the cut prevents
        tok([long], [parent], truncation="only_second", max_length=20)
    t = sl.reply_inputs(tok, [long], None, "text", 10, padding=True)
    assert t["input_ids"][0] == ["[CLS]", *long.split()[:8], "[SEP]"]
    assert tok.calls[-1] == {"pairs": False, "truncation": True, "padding": True}
    with pytest.raises(ValueError):
        sl.reply_inputs(tok, [short], [parent], "both", 20)


# ── Stub torch / transformers / datasets for _fit_eval ─────────────


def _stub_training(monkeypatch, tok):
    """Fake the training stack; returns the event log. The model "predicts"
    each row's token count / 100, so order and inputs can be checked."""
    import random

    events = []
    monkeypatch.setattr(random, "seed", lambda s: events.append(("random.seed", s)))
    monkeypatch.setattr(np.random, "seed", lambda s: events.append(("np.seed", s)))

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None, is_bf16_supported=lambda: False
    )
    torch.manual_seed = lambda s: events.append(("torch.manual_seed", s))
    tf = types.ModuleType("transformers")
    tf.set_seed = lambda s: events.append(("transformers.set_seed", s))
    tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda d: events.append(("tokenizer", str(d))) or tok
    )
    tf.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda d, **kw: events.append(("model", str(d))) or object()
    )
    tf.TrainingArguments = lambda **kw: events.append(("args", kw)) or types.SimpleNamespace(**kw)

    class Trainer:
        def __init__(self, model, args, train_dataset, eval_dataset, processing_class):
            events.append(("trainer", {"train": train_dataset, "eval": eval_dataset}))

        def train(self):
            return types.SimpleNamespace(metrics={"train_runtime": 1.0, "train_loss": 0.1})

        def predict(self, ds):
            events.append(("predict", ds))
            return types.SimpleNamespace(
                predictions=np.array([[len(x) / 100] for x in ds.cols["input_ids"]])
            )

    tf.Trainer = Trainer

    class Dataset:
        def __init__(self, cols):
            self.cols = cols

        @classmethod
        def from_pandas(cls, frame):
            return cls({c: frame[c].tolist() for c in frame.columns})

        def map(self, fn, batched=True):
            return Dataset({**self.cols, **dict(fn(dict(self.cols)))})

    ds = types.ModuleType("datasets")
    ds.Dataset = Dataset
    for name, mod in (("torch", torch), ("transformers", tf), ("datasets", ds)):
        monkeypatch.setitem(sys.modules, name, mod)
    return events


def test_seeds_are_set_before_the_model_is_built(tmp_path, monkeypatch):
    tok = WordTok()
    events = _stub_training(monkeypatch, tok)
    monkeypatch.setattr(settings, "REPLY_CTX_MIN_PARENT_TOKENS", 2)
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": ["one two three four", "one", "one two"],
            "parent_text": ["p q r s t u v w"] * 3,
            "score_teacher": [0.1, -0.2, 0.3],
        }
    )
    info, pred = sl._reply_fit(
        frame,
        frame.drop(columns="score_teacher"),
        variant="ctx",
        seed=7,
        base_dir=tmp_path,
        max_len=14,
        work_dir=tmp_path,
    )
    names = [e[0] for e in events]
    first_load = min(names.index("tokenizer"), names.index("model"))
    seeded = {"random.seed", "np.seed", "torch.manual_seed", "transformers.set_seed"}
    assert seeded <= set(names[:first_load])  # every seed, before any load
    assert all(e[1] == 7 for e in events if e[0] in seeded)
    trainer = next(e[1] for e in events if e[0] == "trainer")
    assert trainer["eval"] is None  # nothing evaluated mid-training
    args = next(e[1] for e in events if e[0] == "args")
    assert args["eval_strategy"] == "no" and args["seed"] == 7
    assert args["learning_rate"] == settings.REPLY_DISTILL["lr"]
    assert args["gradient_accumulation_steps"] == settings.REPLY_DISTILL["grad_accum"]
    train_ids = trainer["train"].cols["input_ids"]
    assert train_ids[0] == [
        "[CLS]",
        "one",
        "two",
        "three",
        "four",
        "[SEP]",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "[SEP]",
    ]  # post cut to max_len 14
    # predicted shortest reply first, returned in the frame's own order
    pred_ds = next(e[1] for e in events if e[0] == "predict")
    assert pred_ds.cols["text"] == ["one", "one two", "one two three four"]
    assert pred.tolist() == pytest.approx([0.14, 0.12, 0.13])  # token counts, frame order
    assert info["variant"] == "ctx"


# ── 3. Leave-one-post-out ──────────────────────────────────────────


def test_lopo_split_holds_the_post_out(files):
    files()
    rows = sl.reply_training_rows()
    for post in POSTS:
        train, test = sl.lopo_split(rows, post)
        assert post not in set(train["tracked_slug"]) and set(test["tracked_slug"]) == {post}
        assert not set(train["id"]) & set(test["id"]) and len(train) + len(test) == len(rows)
    with pytest.raises(ValueError):
        sl.lopo_split(rows, "no_such_post")


@pytest.fixture
def stub_fit(monkeypatch):
    calls = []

    def fit(train, test, *, variant, seed, base_dir, max_len, work_dir, save_to=None):
        calls.append(
            {
                "variant": variant,
                "seed": seed,
                "max_len": max_len,
                "train": train,
                "test": test,
                "save_to": save_to,
            }
        )
        if save_to is not None:
            save_to.mkdir(parents=True, exist_ok=True)
            (save_to / "config.json").write_text("{}")
        if test is None:
            return {"train_runtime_s": 1.0, "train_loss": 0.1}, None
        lean = 0.9 if variant == "ctx" else 0.5
        y = (
            test["score_teacher"]
            if "score_teacher" in test
            else pd.Series(np.linspace(-0.5, 0.5, len(test)), index=test.index)
        )
        return {"train_runtime_s": 1.0, "train_loss": 0.1}, (y * lean + 0.01 * seed).to_numpy()

    monkeypatch.setattr(sl, "_reply_fit", fit)
    return calls


def test_lopo_runs_every_fold_and_resumes(files, stub_fit):
    files()
    res = sl.reply_lopo(seeds=1)
    assert len(stub_fit) == 6 and len(res["folds"]) == 6  # 2 variants x 3 posts
    for c in stub_fit:
        held = set(c["test"]["tracked_slug"])
        assert len(held) == 1 and not held & set(c["train"]["tracked_slug"])
        assert c["max_len"] == (384 if c["variant"] == "ctx" else 256)
    summary = {(r["model"], r["post"]): r for r in res["summary"]}
    assert {m for m, _ in summary} == {"base", "text", "ctx"}
    assert summary[("ctx", "ALL")]["n"] == 17 and summary[("ctx", "ALL")]["seeds"] == 1
    assert summary[("ctx", "ALL")]["pearson"] == pytest.approx(1.0)
    out = settings.MODELS_DIR / settings.REPLY_LOPO_DIR
    preds = pd.read_parquet(out / "lopo_predictions.parquet")
    assert len(preds) == 34 and set(preds.columns) == {
        "id",
        "tracked_slug",
        "variant",
        "seed",
        "pred",
    }

    stub_fit.clear()
    sl.reply_lopo(seeds=1)
    assert stub_fit == []  # every fold is on disk
    saved = json.loads((out / "lopo_results.json").read_text())
    saved["folds"] = saved["folds"][:-1]  # killed between the two writes
    (out / "lopo_results.json").write_text(json.dumps(saved))
    sl.reply_lopo(seeds=1)
    assert [(c["variant"], c["seed"]) for c in stub_fit] == [("ctx", 42)]
    stub_fit.clear()
    res = sl.reply_lopo(seeds=2)  # a second seed adds its folds
    assert len(stub_fit) == 6 and {c["seed"] for c in stub_fit} == {43}
    ctx_all = next(r for r in res["summary"] if r["model"] == "ctx" and r["post"] == "ALL")
    assert ctx_all["seeds"] == 2 and ctx_all["mae_min"] < ctx_all["mae_max"]


def test_lopo_refuses_another_config_unless_forced(files, stub_fit, monkeypatch):
    files()
    sl.reply_lopo(variants=("ctx",), seeds=1)
    monkeypatch.setattr(settings, "REPLY_DISTILL", {**settings.REPLY_DISTILL, "lr": 2e-5})
    stub_fit.clear()
    with pytest.raises(FileExistsError, match="lr"):
        sl.reply_lopo(variants=("ctx",), seeds=1)
    assert stub_fit == []
    sl.reply_lopo(variants=("ctx",), seeds=1, force=True)
    assert len(stub_fit) == 3


# ── 4. Final fit ───────────────────────────────────────────────────


def test_final_fit_records_its_recipe_and_is_never_overwritten(files, stub_fit):
    files()
    meta = sl.reply_fit_all("ctx")
    final = settings.MODELS_DIR / "stance_distilled_final_reply_ctx"
    assert stub_fit[0]["test"] is None and len(stub_fit[0]["train"]) == 17  # every labelled reply
    assert (final / "recipe.txt").read_text() == "reply-ctx-384"
    held = sl.reply_model_meta(final)
    assert held == meta and held["variant"] == "ctx" and held["max_len"] == 384
    assert held["base_model_dir"] == settings.REPLY_DISTILL_BASE and held["label_version"] == "v2"
    assert held["seed"] == settings.DISTILL_SEED and held["n_rows"] == 17
    assert sl.model_max_len(final) == 384
    assert not any(p.name.startswith("partial_") for p in settings.MODELS_DIR.iterdir())
    assert sl.reply_fit_all("ctx") == held and len(stub_fit) == 1  # same fit: nothing to do
    with pytest.raises(FileExistsError):
        sl.reply_fit_all("ctx", seed=7)  # another fit: refused
    other = settings.MODELS_DIR / "stance_distilled_final_reply_text"
    other.mkdir()
    (other / "config.json").write_text("{}")  # a model with no record
    with pytest.raises(FileExistsError):
        sl.reply_fit_all("text")
    assert len(stub_fit) == 1
    with pytest.raises(ValueError, match="parent post"):  # ctx scores only with parents
        sl.score_with_distilled(["a reply"], final)
    with pytest.raises(ValueError, match="reply model"):
        sl.score_replies(final, col="score_x")


# ── 5. Scoring a column ────────────────────────────────────────────


def test_scorer_writes_only_its_column_and_nan_for_textless(files, stub_fit, monkeypatch):
    reps = files()
    sl.reply_fit_all("ctx")
    final = settings.MODELS_DIR / "stance_distilled_final_reply_ctx"
    calls = []

    def fake_score(texts, model_dir, batch_size=64, max_len=None, parents=None):
        calls.append({"texts": list(texts), "parents": parents, "max_len": max_len})
        return np.full(len(texts), -0.25)

    monkeypatch.setattr(sl, "score_with_distilled", fake_score)
    before = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    df = sl.score_replies_with(final, "score_ctx_distilled")
    after = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    pd.testing.assert_frame_equal(after.drop(columns="score_ctx_distilled"), before)
    textless = ~after["text"].map(sl.has_text)
    assert after.loc[textless, "score_ctx_distilled"].isna().all()
    assert (after.loc[~textless, "score_ctx_distilled"] == -0.25).all()
    assert calls[0]["max_len"] == 384 and len(calls[0]["texts"]) == len(reps) - 1
    parent_of = {t: POSTS[s][1] for t, s in zip(reps["text"], reps["tracked_slug"])}
    assert calls[0]["parents"] == [parent_of[t] for t in calls[0]["texts"]]
    assert df["score_ctx_distilled"].notna().sum() == len(reps) - 1
    entry = sl.reply_column_entry("score_ctx_distilled")
    assert entry["model_dir"] == "stance_distilled_final_reply_ctx" and not entry["crossfit"]
    sl.score_replies_with(final, "score_ctx_distilled")
    assert len(calls) == 1  # nothing left to score

    other = settings.MODELS_DIR / "elsewhere"
    other.mkdir()
    (other / sl.REPLY_META).write_text(json.dumps({**sl.reply_model_meta(final), "seed": 7}))
    with pytest.raises(ValueError, match="another fit"):  # never two models in one column
        sl.score_replies_with(other, "score_ctx_distilled")
    with pytest.raises(ValueError, match="does not record"):  # nor into a column of the old kind
        sl.score_replies_with(final, "score_opus_distilled")
    from src.analysis import inference as inf

    with pytest.raises(ValueError, match="cross-fit"):  # in-sample for reply-population
        inf.reply_population(col="score_ctx_distilled", n_boot=5)


def test_crossfit_scores_each_post_with_a_model_blind_to_it(files, stub_fit):
    files()
    before = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    rep = sl.reply_crossfit("ctx")
    assert rep["col"] == "score_ctx_crossfit" and len(stub_fit) == 3
    for c in stub_fit:
        post = set(c["test"]["tracked_slug"])
        assert len(post) == 1 and not post & set(c["train"]["tracked_slug"])
        assert (c["test"]["parent_text"] == POSTS[post.pop()][1]).all()
    after = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    pd.testing.assert_frame_equal(after.drop(columns="score_ctx_crossfit"), before)
    textless = ~after["text"].map(sl.has_text)
    assert after.loc[textless, "score_ctx_crossfit"].isna().all()
    assert after.loc[~textless, "score_ctx_crossfit"].notna().all()
    assert set(rep["posts"]) == set(POSTS) and rep["posts"]["deal_complete"]["held_out"]["n"] == 6
    entry = sl.reply_column_entry("score_ctx_crossfit")
    assert entry["crossfit"] and entry["trained_on_reply_labels"] == "v2"
    assert (
        settings.MODELS_DIR / settings.REPLY_LOPO_DIR / "crossfit_score_ctx_crossfit.json"
    ).exists()
    stub_fit.clear()
    sl.reply_crossfit("ctx")
    assert stub_fit == []  # every post is scored

    # A new reply under one post (no score yet): that post is refit and
    # scored whole, so one post's scores never come from two fits.
    df = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    new = df.loc[df["id"] == "deal_complete-3"].assign(
        id="deal_complete-new", score_ctx_crossfit=np.nan
    )
    pd.concat([df, new], ignore_index=True).to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
    sl.reply_crossfit("ctx")
    assert len(stub_fit) == 1 and len(stub_fit[0]["test"]) == 7  # its 6 + the new one
    assert set(stub_fit[0]["test"]["tracked_slug"]) == {"deal_complete"}
    final = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT)

    def others(d):
        return d.loc[d["tracked_slug"] != "deal_complete", "score_ctx_crossfit"]

    pd.testing.assert_series_equal(others(final), others(after))


# ── CLI ────────────────────────────────────────────────────────────


def test_cli_reply_distill_needs_a_mode_and_a_variant(files, stub_fit):
    from src.cli import main

    files()
    r = CliRunner().invoke(main, ["reply-distill"])
    assert r.exit_code == 2 and "--lopo" in r.output
    r = CliRunner().invoke(main, ["reply-distill", "--fit-all"])
    assert r.exit_code == 2 and "--variant" in r.output
    r = CliRunner().invoke(main, ["reply-distill", "--lopo", "--variants", "ctx", "--seeds", "1"])
    assert r.exit_code == 0, r.output
    assert "base" in r.output and "ctx" in r.output and "ALL" in r.output
