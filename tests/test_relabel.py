"""Batch relabel plumbing without the API: request building, result parsing, merge."""
import json
import types

import pandas as pd
import pytest

from config import settings
from src.analysis import relabel as rl


def _df():
    return pd.DataFrame({"id": ["1", "2", "3", "4"], "text": ["a post", "b post", "   ", "d post"],
                         "user": ["u"] * 4, "tier": ["admin"] * 4,
                         "score_llm": [0.1, -0.2, 0.0, None], "label_llm": ["positive", "negative", "neutral", None]})


def test_posts_to_label_skips_textless_off_topic_and_done(tmp_path, monkeypatch):
    # Post 4 has no Haiku score (analyze ran without --llm): it still needs an
    # Opus label, or `phases` drops it. Off-topic, textless and bare-link
    # posts (7, never seen by Haiku either) do not.
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    more = pd.DataFrame({"id": ["5", "6", "7"], "text": ["off topic post", "ok",
                                                      " https://t.co/abc https://t.co/d "],
                         "user": "u", "tier": "admin", "score_llm": [0.0, None, None],
                         "label_llm": ["off_topic", None, None]})
    df = pd.concat([_df(), more], ignore_index=True)
    assert rl.posts_to_label(df, "claude-opus-5")["id"].tolist() == ["1", "2", "4"]
    pd.DataFrame({"id": ["1"], "score_teacher": [0.5], "label_teacher": ["positive"]}).to_parquet(
        rl.labels_path("claude-opus-5"), index=False)
    assert rl.posts_to_label(df, "claude-opus-5")["id"].tolist() == ["2", "4"]
    assert rl.posts_to_label(df, "claude-opus-5", only_ids={"1", "2"})["id"].tolist() == ["2"]
    no_haiku = df.drop(columns="label_llm")                     # no Haiku labels at all
    assert rl.posts_to_label(no_haiku, "claude-opus-5")["id"].tolist() == ["2", "4", "5"]


def test_build_requests_shape():
    reqs = rl.build_requests(rl.posts_to_label(_df(), "x"), "claude-opus-5", 1024, "low")
    assert [r["custom_id"] for r in reqs] == ["1", "2", "4"]
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


def test_collect_reads_a_downloaded_results_file(tmp_path, monkeypatch):
    # The SDK's results stream broke on this host; collect reads the JSONL
    # downloaded from the batch's results_url instead, and never streams.
    import json

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    rl.state_path().write_text(json.dumps({"batch_id": "b", "model": "m", "status": "ended",
                                           "submitted_at": "2026-10-05T00:00:00+00:00"}))
    monkeypatch.setattr(rl, "status", lambda: json.loads(rl.state_path().read_text()))
    monkeypatch.setattr(rl, "_client", lambda: (_ for _ in ()).throw(AssertionError("streamed")))

    def line(cid, text, kind="succeeded"):
        msg = {"content": [{"type": "text", "text": text}]}
        return json.dumps({"custom_id": cid, "result": {"type": kind, "message": msg}})

    f = tmp_path / "results.jsonl"
    f.write_text("\n".join([line("1", '{"score": 0.4, "label": "positive"}'),
                            line("2", "no json"), line("3", "", kind="errored")]) + "\n")
    st = rl.collect(f)
    lab = pd.read_parquet(rl.labels_path("m"))
    assert lab["id"].tolist() == ["1"] and lab["score_teacher"].tolist() == [0.4]
    assert st["status"] == "collected" and sorted(st["collected"]["failed_ids"]) == ["2", "3"]


def test_relabel_refuses_another_model(tmp_path, monkeypatch):
    """relabel prices output, and checks its caps, at Opus 5's measured
    length: a second rater's model goes through `post-label --job rater`,
    before any call."""
    from click.testing import CliRunner

    from src.cli import main

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(rl, "_client", lambda: (_ for _ in ()).throw(AssertionError("a client")))
    for model in ("claude-opus-5-5", "claude-fable-5-1"):
        with pytest.raises(ValueError, match="post-label"):
            rl.submit(_df(), model=model)
        for action in ("estimate", "submit", "resubmit"):
            res = CliRunner().invoke(main, ["relabel", action, "--model", model, "--yes"])
            assert res.exit_code == 1 and "--job rater" in res.output, res.output
    assert not rl.state_path().exists()


def test_relabel_and_topic_submits_send_the_create_once(tmp_path, monkeypatch):
    """No SDK retry of the create (it has no idempotency key): a create whose
    answer is lost leaves a "submitting" record the next submit refuses on;
    a refusal puts the previous state back."""
    import anthropic
    import anthropic._base_client as bc
    import pytest

    from src.analysis import topic_label as tl
    from tests.test_teacher_checks import _lost_first_answer, _refusing, _sdk_client

    monkeypatch.setattr(bc.BaseClient, "_calculate_retry_timeout", lambda *a, **k: 0.0)
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", tmp_path / "no_feed.parquet")
    _df().to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    for mod, submit in ((rl, lambda: rl.submit(_df())), (tl, lambda: tl.submit())):
        posts: list = []
        monkeypatch.setattr(mod, "_client", lambda p=posts: _sdk_client(_lost_first_answer(p)))
        with pytest.raises(anthropic.APITimeoutError):
            submit()
        assert len(posts) == 1  # one create, not a retry that makes a second batch
        st = json.loads(mod.state_path().read_text())
        assert st["status"] == "submitting" and st["batch_id"] is None and st["n_submitted"] == 3
        with pytest.raises(RuntimeError, match="stopped before it recorded a batch id"):
            submit()
        with pytest.raises(RuntimeError, match="stopped before it recorded a batch id"):
            mod.status()
        assert len(posts) == 1
        prev = {"batch_id": "msgbatch_0", "status": "collected", "model": "m"}
        mod.state_path().write_text(json.dumps(prev))
        refused: list = []
        monkeypatch.setattr(mod, "_client", lambda r=refused: _sdk_client(_refusing(r, 400)))
        with pytest.raises(anthropic.BadRequestError):
            submit()
        assert len(refused) == 1 and json.loads(mod.state_path().read_text()) == prev


def test_claim_column_and_unjudgeable_writes_are_atomic(tmp_path, monkeypatch):
    """A write killed part-way leaves the old file whole: a torn column
    record would break every merge, a torn unjudgeable file every topic-label
    estimate, submit and collect."""
    from pathlib import Path

    import pytest

    from src.analysis import topic_label as tl

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", tmp_path / "no_feed.parquet")
    _df().to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    rl.claim_column("score_opus55", "claude-opus-5-5")
    assert tl.record_unjudgeable(["1"], settings.LLM_MODEL, "msgbatch_a", "no verdict")["recorded"]
    real_text, real_parquet = Path.write_text, pd.DataFrame.to_parquet

    def torn_text(self, data, *a, **k):
        real_text(self, data[:5])
        raise OSError("power lost mid-write")

    def torn_parquet(self, path, *a, **k):
        Path(path).write_bytes(b"PAR1")
        raise OSError("power lost mid-write")

    monkeypatch.setattr(Path, "write_text", torn_text)
    with pytest.raises(OSError):
        rl.claim_column("score_fable51", "claude-fable-5-1")
    monkeypatch.setattr(Path, "write_text", real_text)
    assert json.loads(rl.columns_path().read_text()) == {"score_opus55": "claude-opus-5-5"}
    monkeypatch.setattr(pd.DataFrame, "to_parquet", torn_parquet)
    with pytest.raises(OSError):
        tl.record_unjudgeable(["2"], settings.LLM_MODEL, "msgbatch_b", "no verdict")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real_parquet)
    assert tl.load_unjudgeable()["id"].tolist() == ["1"]


def test_relabel_estimate_is_the_usage_fit_with_its_ceiling(tmp_path, monkeypatch):
    """The estimate prices llm_prompt on the usage fit (post_labels): input
    0.3365 tokens a character + 51, output 84 a call, at its p95 (113) for
    the ceiling the dollar cap is checked against; Opus 5 at batch prices."""
    from src.analysis.post_labels import llm_prompt_cost
    from src.analysis.sentiment import llm_prompt

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    posts = rl.posts_to_label(_df(), "claude-opus-5")
    est = rl.estimate(posts)
    chars = [len(llm_prompt(t, u)) for t, u in zip(posts["text"], posts["user"], strict=True)]
    assert est["usd_batch_ceiling"] == llm_prompt_cost(chars, "x")["usd_batch_ceiling"]
    tin = 0.3365 * sum(chars) + 51 * 3
    assert est["calls"] == 3 and est["problem"] is None
    assert est["usd_batch"] == pytest.approx((tin * 5 + 3 * 84 * 25) / 1e6 * 0.5)
    assert est["usd_batch_ceiling"] == pytest.approx((tin * 5 + 3 * 113 * 25) / 1e6 * 0.5)


def test_relabel_caps_refuse_submit_and_resubmit_before_any_client(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from src.cli import main

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(rl, "_client", lambda: (_ for _ in ()).throw(AssertionError("a client")))
    df = _df()  # 3 posts to label
    df.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    monkeypatch.setattr(settings, "RELABEL_MAX_CALLS", 2)
    with pytest.raises(rl.OverCap, match="RELABEL_MAX_CALLS"):
        rl.submit(df)
    res = CliRunner().invoke(main, ["relabel", "estimate"])
    assert res.exit_code == 0 and "OVER CAP" in res.output and "no API call" in res.output
    assert "RELABEL_MAX_CALLS 2 / RELABEL_MAX_USD $12.00" in res.output
    res = CliRunner().invoke(main, ["relabel", "estimate", "--limit", "2"])
    assert "within the caps" in res.output and "the first 2 of 3 by id" in res.output
    res = CliRunner().invoke(main, ["relabel", "submit"])  # refused before the prompt
    assert res.exit_code == 1 and "Refusing" in res.output and "--limit" in res.output
    monkeypatch.setattr(settings, "RELABEL_MAX_CALLS", 6000)
    monkeypatch.setattr(settings, "RELABEL_MAX_USD", 0.0001)
    with pytest.raises(rl.OverCap, match="RELABEL_MAX_USD"):
        rl.submit(df)
    failed = {"batch_id": "b", "status": "collected", "collected": {"failed_ids": ["1", "2"]}}
    rl.state_path().write_text(json.dumps(failed))
    res = CliRunner().invoke(main, ["relabel", "resubmit", "--yes"])
    assert res.exit_code == 1 and "RELABEL_MAX_USD" in res.output
    with pytest.raises(rl.OverCap):
        rl.submit(df, only_ids={"1", "2"})
    assert json.loads(rl.state_path().read_text()) == failed  # nothing was sent


def test_relabel_limit_sends_the_first_n_by_id_and_records_the_caps(tmp_path, monkeypatch):
    from src.analysis import batch_io

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "RELABEL_MAX_CALLS", 2)
    monkeypatch.setattr(rl, "_client", lambda: object())
    sent: list = []

    def create(client, requests, path, pending):
        sent.append([r["custom_id"] for r in requests])
        return types.SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    monkeypatch.setattr(batch_io, "guarded_create", create)
    df = pd.DataFrame({"id": ["30", "4", "200", "1000"], "text": ["a post", "b post", "c post", "d post"],
                       "user": "u", "tier": "admin"})
    st = rl.submit(df, limit=2)
    assert sent == [["4", "30"]] and st["ids"] == ["4", "30"] and st["limit"] == 2
    est = rl.estimate(rl.first_by_id(rl.posts_to_label(df, "claude-opus-5"), 2))
    assert st["usd_ceiling_batch"] == pytest.approx(est["usd_batch_ceiling"])
    assert st["usd_expected_batch"] == pytest.approx(est["usd_batch"])
    assert st["caps"] == {"calls": 2, "usd": settings.RELABEL_MAX_USD}
    rl.state_path().write_text(json.dumps({**st, "status": "collected"}))
    with pytest.raises(rl.OverCap):
        rl.submit(df)  # all four: over the 2-call cap
    assert len(sent) == 1
