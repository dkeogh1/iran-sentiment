"""dkweb chart JSON: Trump-feed phase ticks and the reply export (no API, stub data)."""

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.visualization import web_export as we


def _feed(tmp_path, monkeypatch, **cols):
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)  # no topic labels: keyword flag
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "either")
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", tmp_path / "trump.parquet")
    days = ["2026-04-04", "2026-04-10", "2026-04-12", "2026-05-02", "2026-05-03"]
    pd.DataFrame(
        {
            "id": [str(i) for i in range(5)],
            "user": "realDonaldTrump",
            "created_at": pd.to_datetime(days, utc=True),
            "text": ["Iran strikes now", "", "Great golf", "Iran deal", "Tariffs"],
            "score_opus_distilled": [0.6, 0.111, 0.0, 0.4, 0.0],
            **cols,
        }
    ).to_parquet(settings.TRUMP_FEED_STANCE)


def test_trump_phase_tick_starts_with_the_feed(tmp_path, monkeypatch):
    # The feed was collected from Apr 4: the first phase's tick must not
    # claim Feb 1. Image-only posts leave the counts and the share.
    _feed(tmp_path, monkeypatch, score_opus=[0.5, np.nan, 0.0, 0.2, 0.0])
    rows = {r["phase"]: r for r in we.trump_phases()}
    assert rows["strikes, ceasefire"]["tick"] == "Apr 4–Apr 21\nstrikes, ceasefire"
    assert rows["talks, MOU"]["tick"].startswith("Apr 22–")
    assert rows["strikes, ceasefire"]["n"] == 2 and rows["strikes, ceasefire"]["share"] == 0.5


def test_trump_phases_take_score_opus_and_leave_out_unscored_posts(tmp_path, monkeypatch, caplog):
    """Trump's stance of record is score_opus (settings.TRUMP_STANCE_SCORE_COL):
    a post with text but no score_opus is left out with a count, not imputed
    from the distilled model; a feed without the column fails."""
    assert settings.TRUMP_STANCE_SCORE_COL == "score_opus"
    _feed(tmp_path, monkeypatch, score_opus=[0.5, np.nan, np.nan, 0.2, 0.0])
    with caplog.at_level("WARNING", logger="src.visualization.web_export"):
        rows = {r["phase"]: r for r in we.trump_phases()}
    assert "1 of Trump's 4 posts with text in the phases have no score_opus" in caplog.text
    first = rows["strikes, ceasefire"]
    assert first["n"] == 1 and first["stance"] == 0.5  # golf left out, not 0.0 from distilled
    assert rows["talks, MOU"]["stance"] == 0.2
    assert {r["phase"]: r for r in we.trump_phases("score_opus_distilled")}["strikes, ceasefire"][
        "stance"
    ] == 0.6  # still there through the argument
    _feed(tmp_path, monkeypatch)
    with pytest.raises(LookupError, match="--merge"):
        we.trump_phases()


def test_replies_skips_uncovered_posts(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    pd.DataFrame(
        {
            "id": ["1", "2", "2", "3", "4"],
            "tracked_slug": ["ceasefire"] * 4 + ["deal_complete"],
            "text": ["no war", "yes", "yes", "", "fine"],
            "score_transformer": [0.2, 0.4, 0.4, 0.9, 0.0],
        }
    ).to_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    est = {
        f"assisted_{k}{s}": v
        for k, v in (("mean", 0.1), ("pro", 0.5), ("anti", 0.3))
        for s in ("", "_lo", "_hi")
    }
    n = {"n_labelled": 120, "n_labelled_fresh": 100}
    rp = pd.DataFrame(
        [
            {"post": "ceasefire", "n_replies": 2, "coverage": 1.0, **n, **est},
            {
                "post": "deal_complete",
                "n_replies": 1,
                "coverage": 0.6,
                **n,
                **{k: np.nan for k in est},
            },
            {"post": "ALL", "n_replies": 3, "coverage": 0.8, **n, **{k: np.nan for k in est}},
        ]
    )
    monkeypatch.setattr(we.inf, "reply_population", lambda: rp)
    with caplog.at_level("WARNING", logger="src.visualization.web_export"):
        rows = we.replies()
    assert [r["slug"] for r in rows] == ["ceasefire"]  # NaN post left out, not nulls
    assert rows[0]["valence"] == 0.3  # mean of 0.2 and 0.4: no dupe, no textless
    assert rows[0]["labelled"] == 120 and rows[0]["labelled_fresh"] == 100  # the draws behind it
    assert "deal_complete left out" in caplog.text


def test_replies_says_when_fresh_draws_are_pooled(tmp_path, monkeypatch, caplog):
    """The pooled first + fresh draws are the published basis (2026-11-02):
    an export says how many draws of each it rests on, each record carries
    them, and an export without fresh draws is warned about."""
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    pd.DataFrame(
        {"id": ["1"], "tracked_slug": ["ceasefire"], "text": ["no war"], "score_transformer": [0.2]}
    ).to_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    est = {
        f"assisted_{k}{s}": v
        for k, v in (("mean", 0.1), ("pro", 0.5), ("anti", 0.3))
        for s in ("", "_lo", "_hi")
    }

    def rp(fresh):
        return pd.DataFrame(
            [
                {
                    "post": "ceasefire",
                    "n_replies": 2,
                    "coverage": 1.0,
                    "n_labelled": 130 + fresh,
                    "n_labelled_fresh": fresh,
                    **est,
                },
                {
                    "post": "ALL",
                    "n_replies": 2,
                    "coverage": 1.0,
                    "n_labelled": 130 + fresh,
                    "n_labelled_fresh": fresh,
                    **est,
                },
            ]
        )

    for fresh, flag in ((998, True), (0, False), (0, True)):
        caplog.clear()
        monkeypatch.setattr(settings, "REPLY_POPULATION_FRESH_DRAWS", flag)
        monkeypatch.setattr(we.inf, "reply_population", lambda f=fresh: rp(f))
        with caplog.at_level("INFO", logger="src.visualization.web_export"):
            rows = we.replies()
        assert rows[0]["labelled"] == 130 + fresh and rows[0]["labelled_fresh"] == fresh
        if fresh:
            assert "1128 labelled random draws pooled: 130 first draws + 998 fresh" in caplog.text
            assert "WARNING" not in caplog.text
        else:
            assert "rests on the 130 first draws only" in caplog.text
            assert "WARNING" in caplog.text
            assert ("is False" in caplog.text) is (not flag)
