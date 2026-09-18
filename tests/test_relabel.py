"""Batch relabel plumbing without the API: request building, result parsing, merge."""
import types

import pandas as pd

from config import settings
from src.analysis import relabel as rl


def _df():
    return pd.DataFrame({"id": ["1", "2", "3", "4"], "text": ["a post", "b post", "   ", "d post"],
                         "user": ["u"] * 4, "tier": ["admin"] * 4,
                         "score_llm": [0.1, -0.2, 0.0, None], "label_llm": ["positive", "negative", "neutral", None]})


def test_posts_to_label_skips_empty_unlabelled_and_done(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    assert rl.posts_to_label(_df(), "claude-opus-5")["id"].tolist() == ["1", "2"]
    pd.DataFrame({"id": ["1"], "score_teacher": [0.5], "label_teacher": ["positive"]}).to_parquet(
        rl.labels_path("claude-opus-5"), index=False)
    assert rl.posts_to_label(_df(), "claude-opus-5")["id"].tolist() == ["2"]
    assert rl.posts_to_label(_df(), "claude-opus-5", only_ids={"1", "2"})["id"].tolist() == ["2"]


def test_build_requests_shape():
    reqs = rl.build_requests(rl.posts_to_label(_df(), "x"), "claude-opus-5", 1024, "low")
    assert [r["custom_id"] for r in reqs] == ["1", "2"]
    p = reqs[0]["params"]
    assert p["model"] == "claude-opus-5" and p["max_tokens"] == 1024
    assert p["output_config"] == {"effort": "low"} and "a post" in p["messages"][0]["content"]


def test_parse_result_variants():
    ok = types.SimpleNamespace(custom_id="7", result=types.SimpleNamespace(type="succeeded", message=types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""),
                 types.SimpleNamespace(type="text", text='{"score": -0.6, "label": "negative"}')])))
    assert rl.parse_result(ok) == {"id": "7", "score_teacher": -0.6, "label_teacher": "negative", "outcome": "succeeded"}
    bad = types.SimpleNamespace(custom_id="8", result=types.SimpleNamespace(type="succeeded", message=types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text="no json")])))
    assert rl.parse_result(bad)["outcome"] == "unparseable"
    err = types.SimpleNamespace(custom_id="9", result=types.SimpleNamespace(type="errored"))
    assert rl.parse_result(err)["outcome"] == "errored" and rl.parse_result(err)["score_teacher"] is None


def test_merge_adds_tag_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    pd.DataFrame({"id": ["1", "2"], "score_teacher": [0.5, -0.5], "label_teacher": ["positive", "negative"]}).to_parquet(
        rl.labels_path("claude-opus-5"), index=False)
    out = rl.merge(_df(), "claude-opus-5")
    assert out["score_opus"].tolist()[:2] == [0.5, -0.5] and pd.isna(out["score_opus"].iloc[3])
    assert rl.estimate_cost(19457) < 40


def test_estimate_matches_measured_rate():
    assert round(rl.estimate_cost(1) / rl.BATCH_DISCOUNT * 1e6, 0) == 217 * 5 + 86 * 25
