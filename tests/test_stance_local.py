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
    monkeypatch.setattr(sl, "score_with_distilled", lambda texts, md, bs: (calls.append(len(texts)) or np.zeros(len(texts))))
    out = tmp_path / "scored.parquet"
    df = sl.score_post_file([f], out)
    assert len(df) == 5 and calls == [5] and "score_opus_distilled" in df
    f.write_text(f.read_text() + _j.dumps({"id": "9", "user": "t", "text": "new"}) + "\n")
    df2 = sl.score_post_file([f], out)
    assert len(df2) == 6 and calls == [5, 1]                       # only the new id scored


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
