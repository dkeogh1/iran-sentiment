"""War-post figures: weekly tier panels and the account x phase heatmap (stub data, no API)."""

import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize, to_hex, to_rgba

from config import settings
from src.analysis import inference as inf
from src.visualization import plots
from src.visualization import web_export as we

BLOG_CSS = Path.home() / "repos" / "dkweb" / "src" / "styles" / "global.css"


@pytest.fixture(autouse=True)
def _no_topic_labels(tmp_path, monkeypatch):
    # No topic labels in PROCESSED_DIR: the war flag falls back to the keyword pattern.
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "FIGURES_DIR", tmp_path / "figures")
    yield
    plt.close("all")


def _posts() -> pd.DataFrame:
    """Daily war posts for a hawk (+0.6) and a dove (-0.6), an aide with war
    and off-topic posts, a sparse opposition account no 3-week window keeps,
    a media account with 3 war posts in phase 1 and 10 in phase 2, plus
    link-only rows (no text) and search rows that must stay out."""
    rows = []

    def add(user, tier, day, text, score):
        rows.append(
            {
                "id": f"{user}-{len(rows)}",
                "user": user,
                "tier": tier,
                "created_at": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12),
                "text": text,
                "score_opus": score,
            }
        )

    days = pd.date_range("2026-02-02", "2026-09-17", freq="D")
    for i, day in enumerate(days):
        add("hawk", "maga_prowar", day, "Bomb Iran now", 0.6)
        add("dove", "maga_antiwar", day, "No war with Iran", -0.6)
        add("aide", "admin", day, "Strikes on Iran tonight" if i % 2 else "Great golf", 0.4)
        add("dove", "maga_antiwar", day, "https://t.co/abc", 0.9)  # no text: leaves every mean
        add("search:iran", "search", day, "Iran war", 1.0)
        if i % 14 == 0:
            add("lone", "opposition", day, "Stop the war", -0.8)
    for k in range(3):
        add("rare", "media", f"2026-03-0{k + 1}", "Iran ceasefire talks", -0.9)
    for k in range(10):
        add("rare", "media", f"2026-05-{k + 10}", "Iran ceasefire talks", 0.1)
    return pd.DataFrame(rows)


def test_stance_cmap_matches_blog_semantics():
    cmap, norm = plots.stance_cmap(), Normalize(-1, 1)
    assert cmap(norm(-1)) == pytest.approx(to_rgba(plots.STANCE_COLORS["anti"]), abs=0.01)
    assert cmap(norm(1)) == pytest.approx(to_rgba(plots.STANCE_COLORS["pro"]), abs=0.01)
    assert cmap(norm(0)) == pytest.approx(to_rgba(plots.STANCE_COLORS["neutral"]), abs=0.01)
    r, _, b, _ = cmap(norm(-0.5))
    assert b > r  # anti-war half is blue
    r, _, b, _ = cmap(norm(0.5))
    assert r > b  # pro-war half is red (RdYlGn had it the other way round)


def test_stance_colors_match_blog_tokens():
    if not BLOG_CSS.exists():
        pytest.skip("dkweb checkout not present")
    css = BLOG_CSS.read_text()
    for key in ("anti", "pro", "neutral"):
        first = re.search(rf"--viz-{key}:\s*(#[0-9a-fA-F]{{6}})", css)  # :root, the light theme
        assert first and first.group(1).lower() == plots.STANCE_COLORS[key]


def test_text_color_picks_contrast():
    assert plots._text_color(to_rgba(plots.STANCE_COLORS["neutral"])) == plots._INK
    assert plots._text_color(to_rgba("#0b2a5a")) == "white"


def test_weekly_matches_blog_export_and_drops_thin_windows():
    d = inf.prepare(_posts(), "score_opus")
    wk = plots.weekly_war_stance(d, "score_opus")
    assert set(wk["tier"]) == {"admin", "maga_prowar", "maga_antiwar", "opposition", "media"}
    assert wk.loc[wk["tier"] == "opposition", "smooth"].isna().all()  # 1 post a fortnight
    dove = wk[wk["tier"] == "maga_antiwar"]
    assert dove["smooth"].dropna().to_numpy() == pytest.approx(-0.6)  # link-only +0.9 left out

    blog = pd.DataFrame(we.maga_weekly(d, "score_opus"))
    ours = wk[wk["tier"].isin(settings.WEB_WEEKLY_TIERS)].dropna(subset=["smooth"])
    ours = ours.assign(
        tier=ours["tier"].map(settings.WEB_TIER_LABELS), week=ours["week"].dt.date.astype(str)
    )
    m = blog.merge(ours, on=["tier", "week"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all()
    assert m["smooth_x"].to_numpy() == pytest.approx(m["smooth_y"].to_numpy(), abs=5e-4)
    assert (m["n_window_x"] == m["n_window_y"]).all()


def test_tier_war_weekly_writes_png(tmp_path):
    out = tmp_path / "tier.png"
    fig = plots.plot_tier_war_weekly(_posts(), path=out)
    assert out.exists() and out.stat().st_size > 0
    titles = {ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()}
    assert titles == {
        settings.WEB_TIER_LABELS[t]
        for t in ("admin", "maga_prowar", "maga_antiwar", "opposition", "media")
    }
    assert "war-post" in fig.texts[0].get_text().lower()
    assert "score_opus" in fig.texts[0].get_text()
    assert fig.axes[0].get_ylabel() == "war-post stance (score_opus)"


def test_lone_points_marks_kept_weeks_with_no_kept_neighbour():
    nan = float("nan")
    y = np.array([nan, 0.4, nan, 0.2, 0.3, nan, 0.5])
    assert plots._lone_points(y) == [False, True, False, False, False, False, True]
    assert plots._lone_points(np.array([0.1])) == [True]
    assert plots._lone_points(np.array([])) == []


def _marked_x(ax, color: str) -> list[float]:
    """x (matplotlib date numbers) of every marker drawn by lines of this colour."""
    out = []
    for ln in ax.get_lines():
        if to_hex(ln.get_color()) != color or ln.get_marker() in (None, "", "None"):
            continue
        x = np.asarray(ln.get_xdata(orig=False), dtype=float)
        me = ln.get_markevery()
        out.extend(x if me is None else x[np.asarray(me, dtype=bool)])
    return out


def test_tier_war_weekly_draws_lone_kept_weeks(tmp_path):
    # 8 war posts in the weeks of Mar 2 and Mar 16, none between: only the
    # Mar 9 window reaches 15, so it is kept with both neighbours dropped and
    # a line alone would draw nothing for it.
    rows = [
        {
            "id": f"v{k}-{day}",
            "user": "vatican",
            "tier": "religious_authority",
            "created_at": pd.Timestamp(day, tz="UTC"),
            "text": "Pray for peace in Iran, no war",
            "score_opus": -0.5,
        }
        for day in ("2026-03-03", "2026-03-17")
        for k in range(8)
    ]
    df = pd.concat([_posts(), pd.DataFrame(rows)], ignore_index=True)
    wk = plots.weekly_war_stance(inf.prepare(df, "score_opus"), "score_opus")
    rel = wk[wk["tier"] == "religious_authority"].set_index("week")["smooth"]
    assert rel.notna().tolist() == [False, True, False]
    lone = mdates.date2num(pd.Timestamp("2026-03-09") + pd.Timedelta(days=3.5))

    fig = plots.plot_tier_war_weekly(df, path=tmp_path / "tier.png")
    panels = {ax.get_title(loc="left"): ax for ax in fig.axes if ax.get_visible()}
    own = panels[settings.WEB_TIER_LABELS["religious_authority"]]
    assert _marked_x(own, plots._INK) == pytest.approx([lone])
    hawk = panels[settings.WEB_TIER_LABELS["maga_prowar"]]
    assert _marked_x(hawk, plots._INK) == []  # an unbroken run gets no dots
    assert _marked_x(hawk, plots._CONTEXT) == pytest.approx([lone])  # nor is it lost in grey


def test_account_war_heatmap_blanks_thin_cells(tmp_path):
    t = plots.account_war_phase_table(_posts())
    rare = t[t["group"] == "rare"].set_index("phase")
    assert rare.loc["strikes, ceasefire", "n_on"] == 3
    assert rare.loc["whole war", "mean_on"] == pytest.approx((3 * -0.9 + 10 * 0.1) / 13)
    assert "search:iran" not in set(t["group"])

    out = tmp_path / "heat.png"
    fig = plots.plot_account_war_heatmap(_posts(), path=out)
    assert out.exists() and out.stat().st_size > 0
    cells = [x.get_text() for x in fig.axes[0].texts]
    assert "-0.90" not in cells  # 3 war posts: blank
    assert "+0.10" in cells and "-0.13" in cells
    title = fig.axes[0].get_title(loc="left")
    assert "war-post" in title.lower() and "score_opus" in title


def test_generate_all_writes_war_figures():
    written = {p.name for p in plots.generate_all(_posts(), score_cols=["score_opus"])}
    assert {"tier_war_weekly_score_opus.png", "account_war_phases_score_opus.png"} <= written
    assert all((settings.FIGURES_DIR / n).stat().st_size > 0 for n in written)
