"""Pure-python parts of the stance-model experiments (no torch, no API)."""
import numpy as np
import pandas as pd
import pytest

from config import settings
from src.analysis import stance_local as sl
from src.analysis.sentiment import parse_llm_json


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    tiers = ["admin", "maga_prowar", "religious_authority"]
    return pd.DataFrame({
        "id": [str(i) for i in range(n)],
        "text": ["post %d" % i for i in range(n)],
        "user": ["u%d" % (i % 7) for i in range(n)],
        "tier": [tiers[i % 3] for i in range(n)],
        "score_llm": rng.uniform(-1, 1, n),
        "score_transformer": rng.uniform(-1, 1, n),
        "label_llm": ["positive"] * (n - 5) + ["off_topic"] * 5,
    })


def _fake_model(model_dir, weights=b"w"):
    """A stand-in fit (config.json and weights) for tests that stub the scoring."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(weights)
    return model_dir


def test_training_frame_drops_off_topic_and_empty():
    df = _frame()
    df.loc[0, "text"] = "   "
    df.loc[1, "score_llm"] = np.nan
    out = sl.training_frame(df)
    assert len(out) == 120 - 5 - 2
    assert set(out.columns) == {"id", "text", "user", "tier", "score_llm", "score_transformer"}
    assert "score_opus" in sl.training_frame(df.assign(score_opus=0.0)).columns


def test_split_is_per_tier_and_disjoint():
    df = sl.training_frame(_frame())
    train, test = sl.stratified_split(df, 0.25, 42)
    assert set(train["id"]).isdisjoint(test["id"])
    assert len(train) + len(test) == len(df)
    for tier, g in df.groupby("tier"):
        assert abs(len(test[test["tier"] == tier]) - round(0.25 * len(g))) <= 1


def test_stratified_sample_spreads_across_tiers():
    df = sl.training_frame(_frame())
    s = sl.stratified_sample(df, 30, 1)
    assert s["tier"].value_counts().to_dict() == {"admin": 10, "maga_prowar": 10, "religious_authority": 10}


def test_agreement_metrics():
    y = np.array([-0.8, -0.2, 0.0, 0.3, 0.9])
    same = sl.agreement(y, y)
    assert same["pearson"] == pytest.approx(1.0) and same["mae"] == 0 and same["sign_agreement"] == 1
    flipped = sl.agreement(y, -y)
    assert flipped["sign_flip_rate"] == pytest.approx(4 / 5)   # the 0.0 row is neutral, not a flip
    with_nan = sl.agreement(y, np.array([-0.8, np.nan, 0.0, 0.3, 0.9]))
    assert with_nan["n"] == 4


def test_agreement_by_tier_has_all_row():
    df = sl.training_frame(_frame())
    by = sl.agreement_by_tier(df, "score_llm", "score_transformer")
    assert "ALL" in by.index and by.loc["ALL", "n"] == len(df)


def test_parse_llm_json_tolerates_fences_and_prose():
    assert parse_llm_json('```json\n{"score": 0.5, "label": "positive"}\n```')["score"] == 0.5
    assert parse_llm_json('Sure! {"score": -0.25, "label": "negative", "reasoning": "x"} done')["score"] == -0.25
    assert parse_llm_json("no json here") is None


def test_score_llm_takes_first_text_block(monkeypatch):
    import types
    from src.analysis import sentiment as sent
    seen = {}
    class _Msgs:
        def create(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(content=[
                types.SimpleNamespace(type="thinking", thinking="..."),
                types.SimpleNamespace(type="text", text='{"score": -0.4, "label": "negative"}'),
            ])
    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Msgs()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert sent.score_llm("post", "u", model="claude-opus-5", max_tokens=1024, effort="low") == (-0.4, "negative")
    assert seen["max_tokens"] == 1024 and seen["output_config"] == {"effort": "low"}
    assert sent.score_llm("post", "u") == (-0.4, "negative") and seen["max_tokens"] == 200


def test_kfold_indices_partition():
    idx = sl.kfold_indices(103, 5, 0)
    assert sorted(np.concatenate(idx)) == list(range(103))
    assert max(len(i) for i in idx) - min(len(i) for i in idx) <= 1


def test_pick_best_ignores_failed_and_nan():
    res = [{"name": "a", "holdout": {"pearson": 0.5}},
           {"name": "b", "holdout": {"pearson": float("nan")}},
           {"name": "c", "error": "oom"},
           {"name": "d", "holdout": {"pearson": 0.7}}]
    assert sl.pick_best(res)["name"] == "d"
    assert sl.pick_best([{"name": "c", "error": "oom"}]) == {}


def test_sweep_resumes_and_reports(tmp_path, monkeypatch):
    """Stub the GPU fit: sweep must skip done recipes, record failures, pick
    the best, run CV, and fit the final model."""
    calls = []
    def fake_fit(train, test, **kw):
        calls.append((kw["base_model"], kw.get("save_to") is not None, test is None))
        if kw["base_model"] == "bad":
            raise RuntimeError("CUDA OOM")
        n = len(test) if test is not None else 0
        pred = train[kw["label_col"]].mean() + np.zeros(n) if n else None
        if test is not None:
            pred = test[kw["label_col"]].values * (0.9 if kw["base_model"] == "good" else 0.1)
        return {"base_model": kw["base_model"], "n_train": len(train)}, pred
    monkeypatch.setattr(sl, "_fit_eval", fake_fit)
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    recipes = [{"name": "g", "base_model": "good", "max_len": 8, "lr": 1e-5, "epochs": 1},
               {"name": "m", "base_model": "meh", "max_len": 8, "lr": 1e-5, "epochs": 1},
               {"name": "x", "base_model": "bad", "max_len": 8, "lr": 1e-5, "epochs": 1}]
    out = tmp_path / "sweep"
    s = sl.sweep(_frame(), recipes=recipes, folds=2, out_dir=out)
    assert s["best"] == "g" and "error" in s["results"][2]
    assert s["cv"]["folds"] == 2 and "final" in s
    assert [c[0] for c in calls] == ["good", "meh", "bad", "good", "good", "good"]  # 3 recipes + 2 folds + final
    assert calls[-1][1] and calls[-1][2]                                         # final: saved, no test set
    calls.clear()
    s2 = sl.sweep(_frame(), recipes=recipes, folds=2, out_dir=out)                # resume: the failed recipe is
    assert [c[0] for c in calls] == ["bad", "good"] and s2["best"] == "g"        # retried; final stub refits (no config.json)


def test_strip_thinking_then_parse():
    raw = '<think>\nThe post praises the strikes... {"score": 0.1}\n</think>\n{"score": 0.8, "label": "positive"}'
    assert parse_llm_json(sl.strip_thinking(raw))["score"] == 0.8
    assert sl.strip_thinking('{"score": -0.2}') == '{"score": -0.2}'


def test_recipe_by_name():
    rc = sl.recipe_by_name("deb-128-1e5-3")
    assert rc["base_model"].startswith("microsoft/deberta") and rc["optim"] == "adamw_bnb_8bit"
    with pytest.raises(KeyError):
        sl.recipe_by_name("nope")


def test_distill_uses_recipe_and_fits_all(tmp_path, monkeypatch):
    calls = []
    def fake_fit(train, test, **kw):
        calls.append(kw)
        pred = test["score_opus"].values * 0.9 if test is not None else None
        return {"base_model": kw["base_model"], "n_train": len(train)}, pred
    monkeypatch.setattr(sl, "_fit_eval", fake_fit)
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    df = _frame().assign(score_opus=lambda d: d["score_llm"] * 0.8)
    m = sl.distill(df, label_col="score_opus", recipe="deb-128-1e5-3", fit_all=True)
    assert calls[0]["base_model"].startswith("microsoft/deberta") and calls[0]["optim"] == "adamw_bnb_8bit"
    assert calls[0]["grad_accum"] == 2 and calls[0]["gradient_checkpointing"] is True
    assert calls[1]["save_to"] == tmp_path / "stance_distilled_final_score_opus"
    assert (tmp_path / "stance_distilled_final_score_opus" / "recipe.txt").read_text() == "deb-128-1e5-3"
    assert m["label_col"] == "score_opus" and "final" in m


def test_score_post_file_incremental(tmp_path, monkeypatch):
    import json as _j
    f = tmp_path / "posts.jsonl"
    f.write_text("".join(_j.dumps({"id": str(i), "user": "t", "tier": "admin", "platform": "truthsocial",
                                   "created_at": "2026-05-01T00:00:00Z", "text": f"post {i}"}) + "\n" for i in range(5)))
    calls = []
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)          # no recipe.txt -> DISTILL_MAX_LEN
    _fake_model(sl.distilled_scorer_dir())
    monkeypatch.setattr(sl, "score_with_distilled",
                        lambda texts, md, bs, ml: (calls.append((len(texts), ml)) or np.full(len(texts), ml / 1000)))
    out = tmp_path / "scored.parquet"
    df = sl.score_post_file([f], out)
    assert len(df) == 5 and calls == [(5, 128)] and "score_opus_distilled" in df
    f.write_text(f.read_text() + _j.dumps({"id": "9", "user": "t", "text": "new"}) + "\n")
    df2 = sl.score_post_file([f], out)
    assert len(df2) == 6 and calls == [(5, 128), (1, 128)]         # only the new id scored
    df3 = sl.score_post_file([f], out, col="score_256", max_len=256)
    assert calls[-1] == (6, 256)                                   # a new column scores every row
    assert (df3["score_opus_distilled"] == 0.128).all() and (df3["score_256"] == 0.256).all()


def test_model_max_len_reads_recipe(tmp_path):
    assert sl.model_max_len(tmp_path) == settings.DISTILL_MAX_LEN      # no recipe.txt
    (tmp_path / "recipe.txt").write_text("deb-256-1e5-3\n")
    assert sl.model_max_len(tmp_path) == 256
    (tmp_path / "recipe.txt").write_text("roberta-large")               # a plain distill fit
    assert sl.model_max_len(tmp_path) == settings.DISTILL_MAX_LEN


def test_final_dir_never_overwritten(tmp_path):
    d = tmp_path / "stance_distilled_final_score_opus"
    assert sl.final_dir_free(d, "deb-128-1e5-3")                        # empty: fit away
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "recipe.txt").write_text("deb-128-1e5-3")
    assert not sl.final_dir_free(d, "deb-128-1e5-3")                    # same model: nothing to do
    with pytest.raises(FileExistsError):
        sl.final_dir_free(d, "deb-256-1e5-3")                           # the 2026-09-19 overwrite


def test_distill_extra_rows_join_pool_and_split(tmp_path, monkeypatch):
    seen = {}
    def fake_fit(train, test, **kw):
        seen["train_tiers"] = set(train["tier"])
        seen["test_tiers"] = set(test["tier"]) if test is not None else set()
        return {"base_model": kw["base_model"], "n_train": len(train)}, (test["score_opus"].values if test is not None else None)
    monkeypatch.setattr(sl, "_fit_eval", fake_fit)
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    df = _frame().assign(score_opus=lambda d: d["score_llm"])
    extra = pd.DataFrame({"id": [f"r{i}" for i in range(40)], "text": ["reply"] * 40, "user": ["x"] * 40,
                          "tier": ["reply_a"] * 20 + ["reply_b"] * 20, "score_opus": np.linspace(-1, 1, 40)})
    m = sl.distill(df, label_col="score_opus", recipe="deb-128-1e5-3", extra=extra)
    assert {"reply_a", "reply_b"} <= seen["train_tiers"] and {"reply_a", "reply_b"} <= seen["test_tiers"]
    assert any(r["tier"] == "reply_a" for r in m["distilled_by_tier"])
    assert (tmp_path / "stance_distilled_score_opus_mixed" / "distill_metrics.json").exists()


class _Tensor:
    """Just enough of a torch tensor for score_with_distilled."""
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def reshape(self, *shape):
        return _Tensor(self.a.reshape(*shape))

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def _stub_model(monkeypatch):
    """Fake torch / transformers: the model scores len(text) / 100. Returns
    (texts sent to the tokenizer per batch, model dirs loaded)."""
    import contextlib
    import sys
    import types
    seen, loaded = [], []

    class _Enc(dict):
        def to(self, device):
            return self

    def tok(texts, **kw):
        seen.append(list(texts))
        return _Enc(n=_Tensor([len(t) for t in texts]))

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, n):
            return types.SimpleNamespace(logits=_Tensor(n.a / 100).reshape(-1, 1))

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = contextlib.nullcontext
    torch.set_num_threads = lambda n: loaded.append(("threads", n))
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda d: loaded.append(d) or tok)
    tf.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda d: _Model())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    return seen, loaded


def test_score_with_distilled_leaves_textless_nan(tmp_path, monkeypatch):
    # The old "." stand-in scored every image-only post the same +0.111.
    seen, loaded = _stub_model(monkeypatch)
    texts = ["", "   ", None, " ok ", "yes", "a real post"]
    out = sl.score_with_distilled(texts, tmp_path, batch_size=1)
    assert np.isnan(out[:4]).all()
    assert out[4] == pytest.approx(0.03) and out[5] == pytest.approx(0.11)
    assert seen == [["yes"], ["a real post"]]                       # only texts reach the model
    assert ("threads", settings.TORCH_NUM_THREADS) in loaded       # CPU: thread cap applied
    loaded.clear()
    assert np.isnan(sl.score_with_distilled(["", "  "], tmp_path)).all()
    assert loaded == []                                     # nothing to score: no model load


def test_score_with_distilled_batches_by_length_and_keeps_each_score_in_place(
        tmp_path, monkeypatch):
    # A batch pads to its longest input: batching by length cuts the padding,
    # and each score must still land on its own text.
    seen, _ = _stub_model(monkeypatch)
    texts = ["a much longer reply here", "abc", "", "a medium one", "abcd"]
    out = sl.score_with_distilled(texts, tmp_path, batch_size=2)
    assert seen == [["abc", "abcd"], ["a medium one", "a much longer reply here"]]
    assert out[0] == pytest.approx(0.24) and out[1] == pytest.approx(0.03)
    assert np.isnan(out[2])
    assert out[3] == pytest.approx(0.12) and out[4] == pytest.approx(0.04)


def test_score_post_file_and_replies_never_resend_textless(tmp_path, monkeypatch):
    import json as _j
    f = tmp_path / "posts.jsonl"
    recs = (("1", "strikes on Iran"), ("2", ""), ("3", None), ("4", "ok"))
    f.write_text("".join(_j.dumps({"id": i, "user": "t", "text": t}) + "\n" for i, t in recs))
    calls = []
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    _fake_model(sl.distilled_scorer_dir())
    _fake_model(tmp_path)
    def fake_score(texts, md, bs=64, max_len=None):
        calls.append(list(texts))
        return np.full(len(texts), 0.5)
    monkeypatch.setattr(sl, "score_with_distilled", fake_score)
    out = tmp_path / "scored.parquet"
    df = sl.score_post_file([f], out)
    assert calls == [["strikes on Iran"]]
    unscored = df.set_index("id")["score_opus_distilled"].isna().to_dict()
    assert unscored == {"1": False, "2": True, "3": True, "4": True}
    sl.score_post_file([f], out)
    assert len(calls) == 1                                  # the NaN rows are not queued again

    calls.clear()
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    replies = pd.DataFrame({"id": ["a", "b"], "text": ["I support this", " "]})
    replies.to_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    r = sl.score_replies(tmp_path, col="score_x")
    assert calls == [["I support this"]] and r["score_x"].isna().tolist() == [False, True]
    sl.score_replies(tmp_path, col="score_x")
    assert len(calls) == 1


def test_score_post_file_rescores_rows_that_gain_text(tmp_path, monkeypatch):
    # A ReTruth stored with no text (and the old "." score) gets the reblogged
    # text from a re-collected feed: it takes the text and is scored again.
    # Posts that had text keep their score, and image-only posts stay NaN.
    import json as _j
    out = tmp_path / "scored.parquet"
    pd.DataFrame({"id": ["1", "2", "3"], "user": "t", "text": ["", "strikes on Iran", ""],
                  "score_opus_distilled": [0.111, 0.7, 0.111],
                  "score_other": [0.111, 0.2, 0.111]}).to_parquet(out, index=False)
    f = tmp_path / "posts.jsonl"
    recs = (("1", "RT @someone: no war with Iran"), ("2", "strikes on Iran, edited"), ("3", ""))
    f.write_text("".join(_j.dumps({"id": i, "user": "t", "text": t}) + "\n" for i, t in recs))
    calls = []
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    _fake_model(sl.distilled_scorer_dir())
    def fake_score(texts, md, bs=64, max_len=None):
        calls.append(list(texts))
        return np.full(len(texts), -0.5)
    monkeypatch.setattr(sl, "score_with_distilled", fake_score)
    df = sl.score_post_file([f], out, max_len=8).set_index("id")
    assert calls == [["RT @someone: no war with Iran"]]
    assert df.loc["1", "text"] == "RT @someone: no war with Iran"
    assert df.loc["1", "score_opus_distilled"] == -0.5 and np.isnan(df.loc["1", "score_other"])
    assert df.loc["2", "text"] == "strikes on Iran" and df.loc["2", "score_opus_distilled"] == 0.7
    assert df.loc["3", "score_opus_distilled"] == 0.111       # untouched: still no text
    sl.score_post_file([f], out, max_len=8)
    assert len(calls) == 1


# ── score-distilled / score-posts: the scorer of record, one model a column ──


def test_scoring_defaults_to_the_scorer_of_record_and_one_model_per_column(tmp_path, monkeypatch):
    import json as _j
    from pathlib import Path

    from click.testing import CliRunner

    from src.cli import main

    models = tmp_path / "models"
    feed = tmp_path / "truthsocial_trump_stance.parquet"
    replies = tmp_path / "reply_sentiment.parquet"
    monkeypatch.setattr(settings, "MODELS_DIR", models)
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", feed)
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", replies)
    record, reply_record = sl.trump_columns_path(), sl.reply_columns_path()
    assert sl.columns_path_for(feed) == record and sl.columns_path_for(replies) == reply_record
    used = []

    def fake_score(texts, md, bs=64, max_len=None):
        used.append(Path(md))
        return np.full(len(texts), -0.5)

    monkeypatch.setattr(sl, "score_with_distilled", fake_score)
    of_record = _fake_model(models / settings.DISTILLED_SCORER_DIR)
    old = _fake_model(models / "stance_distilled", b"haiku-taught fit")

    # The feed as it stands: the Job's score_opus_distilled, which no record
    # names, and the merge's score_opus / label_opus, whose claim it records.
    # Post 1 had no text (and a label of an older text); a re-collect gives it some.
    pd.DataFrame(
        {
            "id": ["1", "2"],
            "user": "t",
            "text": ["", "strikes on Iran"],
            "score_opus_distilled": [0.111, 0.7],
            "score_opus": [0.2, 0.5],
            "label_opus": ["neutral", "positive"],
        }
    ).to_parquet(feed, index=False)
    record.write_text(_j.dumps({"score_opus": "claude-opus-5"}))
    f = tmp_path / "feed.jsonl"
    recs = (("1", "RT @x: no war with Iran"), ("2", "strikes on Iran"), ("3", "A new post"))
    f.write_text("".join(_j.dumps({"id": i, "user": "t", "text": t}) + "\n" for i, t in recs))
    res = CliRunner().invoke(main, ["score-posts", str(f), "--out", str(feed)])
    assert res.exit_code == 0, res.output
    assert used == [of_record]
    df = pd.read_parquet(feed).set_index("id")
    assert df["score_opus_distilled"].tolist() == [-0.5, 0.7, -0.5]
    assert df.loc["2", "score_opus"] == 0.5 and df.loc["2", "label_opus"] == "positive"  # kept
    assert df.loc[["1", "3"], ["score_opus", "label_opus"]].isna().all().all()
    reg = _j.loads(record.read_text())
    sha = sl.model_weights_sha256(of_record)
    assert reg == {
        "score_opus": "claude-opus-5",
        "score_opus_distilled": {
            "source": "distill",
            "model_dir": settings.DISTILLED_SCORER_DIR,
            "weights_sha256": sha,
        },
    }
    before, held = pd.read_parquet(feed), record.read_text()
    # Another model into the column of record, distilled scores into an LLM's
    # label column, or a dir with no model (a typo): refused, nothing recorded.
    res = CliRunner().invoke(
        main, ["score-posts", str(f), "--out", str(feed), "--model-dir", str(old)]
    )
    assert res.exit_code == 1 and "another fit" in res.output, res.output
    assert "Traceback" not in res.output
    for col in ("score_opus", "score_haiku", "label_opus55"):
        with pytest.raises(sl.ScoringRefused, match="labels"):
            sl.score_post_file([f], feed, col=col)
    typo = models / "stance_distiled_final_x"
    res = CliRunner().invoke(
        main,
        ["score-posts", str(f), "--out", str(feed), "--model-dir", str(typo), "--col", "score_x"],
    )
    assert res.exit_code == 1 and "no model at" in res.output, res.output
    pd.testing.assert_frame_equal(pd.read_parquet(feed), before)
    assert record.read_text() == held
    assert len(used) == 1 and not sl.scoring_marker(feed).exists()
    # A new column takes one model, recorded at its first run.
    sl.score_post_file([f], feed, old, col="score_haiku_distilled")
    assert _j.loads(record.read_text())["score_haiku_distilled"]["model_dir"] == "stance_distilled"
    with pytest.raises(sl.ScoringRefused, match="another fit"):
        sl.score_post_file([f], feed, col="score_haiku_distilled")
    # The same dir refit in place (as every stance-distill run rewrites
    # stance_distilled) is another model too.
    (old / "model.safetensors").write_bytes(b"a later fit in the same dir")
    with pytest.raises(sl.ScoringRefused, match="weights_sha256 differs"):
        sl.score_post_file([f], feed, old, col="score_haiku_distilled")
    (old / "model.safetensors").write_bytes(b"haiku-taught fit")

    # Replies: score_opus_distilled (scorer of record), score_distilled (the
    # Haiku-taught model, the score-distilled Job's), the v0 scores of an
    # overwritten fit; none recorded yet.
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "tracked_slug": "p",
            "text": ["I support this", "No war"],
            "score_opus_distilled": [0.4, np.nan],
            "score_distilled": [0.1, np.nan],
            "score_opus_distilled_v0": [0.3, 0.2],
        }
    ).to_parquet(replies, index=False)
    used.clear()
    res = CliRunner().invoke(main, ["score-distilled"])
    assert res.exit_code == 0, res.output
    assert used == [of_record]
    assert pd.read_parquet(replies)["score_opus_distilled"].tolist() == [0.4, -0.5]
    res = CliRunner().invoke(main, ["score-distilled", "--model-dir", str(old)])
    assert res.exit_code == 1 and "another fit" in res.output, res.output
    sl.score_replies(old, col="score_distilled")  # the Job's own pair still runs
    assert used == [of_record, old]
    with pytest.raises(sl.ScoringRefused, match="does not record"):
        sl.score_replies(col="score_opus_distilled_v0")
    assert {
        k: (v["model_dir"], v["weights_sha256"])
        for k, v in _j.loads(reply_record.read_text()).items()
    } == {
        "score_opus_distilled": (settings.DISTILLED_SCORER_DIR, sha),
        "score_distilled": ("stance_distilled", sl.model_weights_sha256(old)),
    }
    # Any other error keeps its traceback: only the guards' refusals are one line.
    def broken(*a, **k):
        raise ValueError("Unrecognized model")

    monkeypatch.setattr(sl, "score_with_distilled", broken)
    pd.read_parquet(replies).assign(score_opus_distilled=np.nan).to_parquet(replies, index=False)
    res = CliRunner().invoke(main, ["score-distilled"])
    assert res.exit_code == 1 and isinstance(res.exception, ValueError)
    assert not isinstance(res.exception, sl.ScoringRefused)


def test_score_jobs_write_each_column_from_its_model(monkeypatch):
    """What the GPU Jobs run: each names the model its column is recorded for,
    so the scorer-of-record defaults change none of them."""
    import ast
    import re
    from pathlib import Path

    from src.cli import main

    default_col = {
        c: next(p.default for p in main.commands[c].params if p.name == "col")
        for c in ("score-distilled", "score-posts")
    }
    runs = {}
    # The image runs with IRAN_DATA_DIR=/data: the record names a model dir
    # as _model_ref does there, relative to /data/models.
    monkeypatch.setattr(settings, "MODELS_DIR", Path("/data/models"))
    for job in sorted((settings.PROJECT_ROOT / "k8s" / "jobs").glob("*/job.yaml")):
        line = re.search(r"^\s*args:\s*(\[.*\])\s*$", job.read_text(), re.MULTILINE)
        assert line, f"{job}: no one-line args: [...] to read"
        args = ast.literal_eval(line.group(1))
        if args[0] not in default_col:
            continue
        opts = {a: args[i + 1] for i, a in enumerate(args[:-1]) if a.startswith("--")}
        model = Path(opts.get("--model-dir", settings.MODELS_DIR / settings.DISTILLED_SCORER_DIR))
        runs[job.parent.name] = (sl._model_ref(model), opts.get("--col", default_col[args[0]]))
    assert runs == {
        "score-distilled": ("stance_distilled", "score_distilled"),
        "score-distilled-deb": ("stance_distilled_final", "score_deb_distilled"),
        "score-distilled-mixed": (
            "stance_distilled_final_score_opus_mixed",
            "score_mixed_distilled",
        ),
        "score-distilled-opus": (settings.DISTILLED_SCORER_DIR, settings.DISTILLED_SCORER_COL),
        "score-trump-feed": (settings.DISTILLED_SCORER_DIR, settings.DISTILLED_SCORER_COL),
    }
    for model, col in runs.values():
        assert settings.DISTILLED_COLUMN_MODELS[col] == model
