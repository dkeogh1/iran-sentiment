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
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "assisted")  # the stub's columns
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


def test_reply_shares_keep_four_decimals(tmp_path, monkeypatch):
    """The chart labels shares in whole percent: 0.4548 must not reach it as
    0.455, which Math.round labels 46%."""
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "assisted")  # the stub's columns
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    pd.DataFrame(
        {"id": ["1"], "tracked_slug": ["ceasefire"], "text": ["no war"], "score_transformer": [0.2]}
    ).to_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    est = {"assisted_mean": 0.1, "assisted_mean_lo": 0.0, "assisted_mean_hi": 0.2}
    est |= {"assisted_pro": 0.35, "assisted_anti": 0.4548}
    rp = pd.DataFrame(
        [{"post": "ceasefire", "n_replies": 1, "coverage": 1.0, "n_labelled": 1, **est}]
    )
    monkeypatch.setattr(we.inf, "reply_population", lambda: rp)
    row = we.replies()[0]
    assert (row["anti"], row["pro"], row["neutral"]) == (0.4548, 0.35, 0.1952)


def test_replies_says_when_fresh_draws_are_pooled(tmp_path, monkeypatch, caplog):
    """The pooled first + fresh draws are the published basis (2026-11-02):
    an export says how many draws of each it rests on, each record carries
    them, and an export without fresh draws is warned about."""
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "assisted")  # the stub's columns
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


def _two_estimator_rows() -> pd.DataFrame:
    row = {"post": "ceasefire", "n_replies": 2, "coverage": 1.0}
    row |= {"n_labelled": 130, "n_labelled_fresh": 100}
    for est, d in (("assisted", 0.0), ("greg", 0.01)):
        for k, v in (("mean", 0.1234), ("pro", 0.35), ("anti", 0.4548), ("neutral", 0.1952)):
            row |= {
                f"{est}_{k}": v + d,
                f"{est}_{k}_lo": v + d - 0.02,
                f"{est}_{k}_hi": v + d + 0.03,
            }
    return pd.DataFrame([row])


def test_replies_json_takes_the_estimator_of_record(tmp_path, monkeypatch, caplog):
    """settings.REPLY_POPULATION_ESTIMATOR names the published columns. Its
    default, "assisted", writes the same replies.json as before greg_* was
    there, byte for byte; "greg" switches every corrected figure; any other
    name is refused before export-web writes a file."""
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    pd.DataFrame(
        {"id": ["1"], "tracked_slug": ["ceasefire"], "text": ["no war"], "score_transformer": [0.2]}
    ).to_parquet(settings.REPLY_SENTIMENT_OUTPUT)
    rp = _two_estimator_rows()
    monkeypatch.setattr(we.inf, "reply_population", lambda: rp)

    def written(name) -> str:
        return we._write(tmp_path, name, we.replies()).read_text()

    assert settings.REPLY_POPULATION_ESTIMATOR == "assisted"
    default = written("default.json")
    assert we.replies() == [
        {
            "post": settings.WEB_POST_LABELS["ceasefire"],
            "slug": "ceasefire",
            "date": "2026-04-07",
            "replies": 2,
            "anti": 0.4548,
            "neutral": 0.1952,
            "pro": 0.35,
            "stance": 0.123,
            "lo": 0.103,
            "hi": 0.153,
            "valence": 0.2,
            "labelled": 130,
            "labelled_fresh": 100,
        }
    ]
    full, rp = rp, rp[[c for c in rp if not c.startswith("greg_")]]
    assert written("without.json") == default  # no greg_* column is read by default
    rp = full
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "greg")
    row = we.replies()[0]
    assert (row["anti"], row["pro"], row["neutral"]) == (0.4648, 0.36, 0.1752)
    assert (row["stance"], row["lo"], row["hi"]) == (0.133, 0.113, 0.163)
    rp = full.assign(greg_mean=np.nan)
    assert we.replies() == [] and "ceasefire left out" in caplog.text
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "direct")
    with pytest.raises(ValueError, match="REPLY_POPULATION_ESTIMATOR"):
        we.export(tmp_path / "web")
    assert not (tmp_path / "web").exists()


def _tier_posts() -> pd.DataFrame:
    """The administration on one war post and one off-topic post a day: war
    posts at +0.4 in the strike phase, +0.2 in the midterm hold and -0.5 in
    phase 6, off-topic posts at 0."""
    rows = []
    for phase, days, war in (
        ("strikes, ceasefire", pd.date_range("2026-03-02", periods=6), 0.4),
        ("midterm hold", pd.date_range("2026-10-05", periods=6), 0.2),
        ("midterms, standoff", pd.date_range("2026-11-09", periods=6), -0.5),
    ):
        for day in days:
            for text, score in (("Strikes on Iran tonight", war), ("Great golf", 0.0)):
                rows.append(
                    {
                        "id": str(len(rows)),
                        "user": "aide",
                        "tier": "admin",
                        "created_at": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12),
                        "text": text,
                        "score_opus": score,
                    }
                )
    return pd.DataFrame(rows)


def test_decomposition_keeps_its_phase_as_phases_are_added(tmp_path, monkeypatch):
    """Phase 6 is the last phase and grows with ANALYSIS_END: decomposition.json
    stays the post's contrast, strikes -> settings.WEB_DECOMPOSITION_PHASE
    ("midterm hold"), until the setting is switched, and a label that is no
    phase is refused before export-web writes a file."""
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)  # no topic labels: keyword flag
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "either")
    assert settings.WEB_DECOMPOSITION_PHASE == "midterm hold"
    assert we.inf.PHASE_ORDER[-1] == "midterms, standoff"
    d = we.inf.prepare(_tier_posts(), "score_opus")

    def total() -> float:
        rows = we.decomposition(d, "score_opus")
        assert {r["tier"] for r in rows} == {"Administration"} and len(rows) == 3
        return rows[0]["total"]

    assert total() == pytest.approx(0.1 - 0.2)  # all-post means: 0.2 -> 0.1
    monkeypatch.setattr(settings, "WEB_DECOMPOSITION_PHASE", "midterms, standoff")
    assert total() == pytest.approx(-0.25 - 0.2)  # the owner's switch takes effect

    monkeypatch.setattr(settings, "WEB_DECOMPOSITION_PHASE", "midterm holds")
    with pytest.raises(ValueError, match="'midterm holds' is not a phase"):
        we.decomposition(d, "score_opus")
    # the first phase is every contrast's base: phase_contrasts has no row for
    # it, so it would export an empty chart
    monkeypatch.setattr(settings, "WEB_DECOMPOSITION_PHASE", we.inf.PHASE_ORDER[0])
    with pytest.raises(ValueError, match="is the first phase"):
        we.decomposition(d, "score_opus")
    with pytest.raises(ValueError, match="WEB_DECOMPOSITION_PHASE"):
        we.export(tmp_path / "web_base")
    assert not (tmp_path / "web_base").exists()
    monkeypatch.setattr(settings, "WEB_DECOMPOSITION_PHASE", "midterm holds")
    out = tmp_path / "web"
    with pytest.raises(ValueError, match="WEB_DECOMPOSITION_PHASE"):
        we.export(out)
    assert not out.exists()  # refused before any file
