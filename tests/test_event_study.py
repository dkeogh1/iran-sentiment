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
