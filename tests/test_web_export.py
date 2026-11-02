"""dkweb chart JSON: Trump-feed phase ticks and the reply export (no API, stub data)."""

import numpy as np
import pandas as pd

from config import settings
from src.visualization import web_export as we


def test_trump_phase_tick_starts_with_the_feed(tmp_path, monkeypatch):
    # The feed was collected from Apr 4: the first phase's tick must not
    # claim Feb 1. Image-only posts leave the counts and the share.
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)  # no topic labels: keyword flag
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", tmp_path / "trump.parquet")
    days = ["2026-04-04", "2026-04-10", "2026-04-12", "2026-05-02", "2026-05-03"]
    pd.DataFrame(
        {
            "id": [str(i) for i in range(5)],
            "user": "realDonaldTrump",
            "created_at": pd.to_datetime(days, utc=True),
            "text": ["Iran strikes now", "", "Great golf", "Iran deal", "Tariffs"],
            "score_opus_distilled": [0.6, 0.111, 0.0, 0.4, 0.0],
        }
    ).to_parquet(settings.TRUMP_FEED_STANCE)
    rows = {r["phase"]: r for r in we.trump_phases()}
    assert rows["strikes, ceasefire"]["tick"] == "Apr 4–Apr 21\nstrikes, ceasefire"
    assert rows["talks, MOU"]["tick"].startswith("Apr 22–")
    assert rows["strikes, ceasefire"]["n"] == 2 and rows["strikes, ceasefire"]["share"] == 0.5


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
    rp = pd.DataFrame(
        [
            {"post": "ceasefire", "n_replies": 2, "coverage": 1.0, **est},
            {"post": "deal_complete", "n_replies": 1, "coverage": 0.6, **{k: np.nan for k in est}},
            {"post": "ALL", "n_replies": 3, "coverage": 0.8, **{k: np.nan for k in est}},
        ]
    )
    monkeypatch.setattr(we.inf, "reply_population", lambda: rp)
    with caplog.at_level("WARNING", logger="src.visualization.web_export"):
        rows = we.replies()
    assert [r["slug"] for r in rows] == ["ceasefire"]  # NaN post left out, not nulls
    assert rows[0]["valence"] == 0.3  # mean of 0.2 and 0.4: no dupe, no textless
    assert "deal_complete left out" in caplog.text


def test_replies_says_when_fresh_draws_are_pooled(tmp_path, monkeypatch, caplog):
    """The published prose cites the 1,071-draw figures: an export that pools
    the fresh draws says so, and one that does not stays quiet."""
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

    for fresh, warned in ((0, False), (998, True)):
        caplog.clear()
        monkeypatch.setattr(we.inf, "reply_population", lambda f=fresh: rp(f))
        with caplog.at_level("INFO", logger="src.visualization.web_export"):
            we.replies()
        assert f"{130 + fresh} labelled random draws, {fresh} of them fresh" in caplog.text
        assert ("1,071-draw figures" in caplog.text) is warned
        assert ("REPLY_POPULATION_FRESH_DRAWS = False" in caplog.text) is warned
