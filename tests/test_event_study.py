"""Incremental reply scoring in load_or_score_replies (stubbed scorer)."""
import pandas as pd

from config import settings
from src.analysis import event_study as es


def test_load_or_score_replies_scores_only_new(tmp_path, monkeypatch):
    out = tmp_path / "reply_sentiment.parquet"
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", out)
    pd.DataFrame({"id": ["1", "2"], "score_transformer": [0.1, -0.2]}).to_parquet(out, index=False)

    raw = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}, {"id": "3", "text": "c"}]
    scored_calls = []
    def fake_analyze(posts, **kw):
        scored_calls.append([p["id"] for p in posts])
        return pd.DataFrame({"id": [p["id"] for p in posts], "score_transformer": [0.5] * len(posts)})
    import src.collectors.truthsocial_collector as tsc
    import src.analysis.sentiment as sent
    monkeypatch.setattr(tsc, "load_all_cached_replies", lambda: raw)
    monkeypatch.setattr(sent, "analyze", fake_analyze)

    df = es.load_or_score_replies()
    assert scored_calls == [["3"]]                       # only the new id was scored
    assert sorted(df["id"]) == ["1", "2", "3"]
    assert sorted(pd.read_parquet(out)["id"]) == ["1", "2", "3"]

    scored_calls.clear()
    df2 = es.load_or_score_replies()
    assert scored_calls == [] and len(df2) == 3          # second run is a no-op


def test_score_stance_scores_only_new(tmp_path, monkeypatch):
    import types
    out = tmp_path / "stance_sample.parquet"
    monkeypatch.setattr(es, "STANCE_OUTPUT", out)
    pd.DataFrame({"id": ["1"], "tracked_slug": ["old"], "stance": ["neutral_other"]}).to_parquet(out, index=False)

    sent = []
    class _Msgs:
        def create(self, **kw):
            sent.append(kw["messages"][0]["content"])
            return types.SimpleNamespace(content=[types.SimpleNamespace(
                text='{"stance": "antiwar_betrayal", "confidence": 0.8, "reason": "r"}')])
    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Msgs()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    sample = pd.DataFrame({"id": ["1", "2"], "tracked_slug": ["old", "new"],
                           "user": ["u1", "u2"], "text": ["old reply text", "new reply text"],
                           "score_transformer": [0.0, -0.5]})
    res = es.score_stance(sample)
    assert len(sent) == 1 and "new reply text" in sent[0]  # only the uncached row hit the API
    assert sorted(res["id"]) == ["1", "2"]
    assert res.set_index("id").loc["2", "stance"] == "antiwar_betrayal"
    assert len(pd.read_parquet(out)) == 2


def test_score_stance_media_only_follows_min_text_chars(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(es, "STANCE_OUTPUT", tmp_path / "stance_sample.parquet")
    monkeypatch.setattr(settings, "MIN_TEXT_CHARS", 5)
    sent = []
    class _Msgs:
        def create(self, **kw):
            sent.append(kw["messages"][0]["content"])
            return types.SimpleNamespace(content=[types.SimpleNamespace(
                text='{"stance": "neutral_other", "confidence": 0.5, "reason": "r"}')])
    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Msgs()
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    sample = pd.DataFrame({"id": ["1", "2"], "tracked_slug": ["p", "p"], "user": ["u1", "u2"],
                           "text": [" abcd ", "long enough"], "score_transformer": [0.0, 0.0]})
    res = es.score_stance(sample).set_index("id")
    assert res.loc["1", "stance"] == "media_only" and len(sent) == 1   # 4 chars < 5: no API call
    assert res.loc["2", "stance"] == "neutral_other"


def test_compare_events_keeps_the_analysis_window_and_flags_cut_windows():
    """With start / end (the CLI's ANALYSIS_START / ANALYSIS_END), posts past
    the end (a pull's partial last day) are left out, and an event whose
    +-48 h window runs past either edge is flagged cut, not shown as full."""
    from datetime import date

    from config.timeline import Event

    def post(t, s):
        return {"created_at": pd.Timestamp(t, tz="UTC"), "score_opus": s}

    df = pd.DataFrame(
        [
            post("2026-11-19 12:00", -0.3),
            post("2026-11-20 10:00", 0.1),
            post("2026-11-21 10:00", 0.2),
            post("2026-11-22 01:00", 0.9),  # past ANALYSIS_END (Nov 21)
        ]
    )

    def ev(d):
        return Event(date=d, label=str(d), description="", category="c", impact="i")

    events = [ev(date(2026, 11, 20)), ev(date(2026, 11, 21))]
    res = es.compare_events(
        df, events, score_col="score_opus", start=date(2026, 11, 2), end=date(2026, 11, 21)
    ).set_index("event_date")
    # Nov 20: post side [Nov 20, Nov 22) ends at the window's edge: full
    assert not res.loc["2026-11-20", "cut"]
    assert res.loc["2026-11-20", "n_post"] == 2 and res.loc["2026-11-20", "n_pre"] == 1
    # Nov 21: post side runs to Nov 23, past the end; the Nov 22 post stays out
    assert res.loc["2026-11-21", "cut"]
    assert res.loc["2026-11-21", "n_post"] == 1
    assert res.loc["2026-11-21", "mean_post"] == 0.2
    # an event whose pre side starts before `start` is cut too
    early = es.compare_events(
        df, [ev(date(2026, 11, 20))], score_col="score_opus", start=date(2026, 11, 19)
    )
    assert early["cut"].tolist() == [True] and early["n_post"].tolist() == [2]
    # without bounds nothing is clipped or flagged (the call as it was)
    raw = es.compare_events(df, events, score_col="score_opus").set_index("event_date")
    assert raw.loc["2026-11-21", "n_post"] == 2 and not raw["cut"].any()
