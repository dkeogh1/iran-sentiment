"""
Window weights for the X series (src/analysis/window_weights.py): window
assignment on the fill's grid, edge buckets, take-all and search rows at 1,
windows without cached counts, the weighted bootstrap and weekly series, and
the raw path left exactly as it was. Data lives under tmp_path; no network.
"""

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings, timeline
from src.analysis import inference as inf
from src.analysis import window_weights as ww
from src.collectors import x_counts as xcn
from src.visualization import plots, web_export

H = timedelta(hours=1)


def _t(s: str) -> datetime:
    """Naive UTC, as the collector and the counts cache use."""
    return datetime.fromisoformat(s)


F = _t("2026-03-01")  # COLLECTION_START here
B0 = (F, _t("2026-03-02T13:00"))  # the backward window (a 2 h head merged in)
W1 = (_t("2026-03-02T13:00"), _t("2026-03-04"))  # collect, truncated
W2 = (_t("2026-03-04"), _t("2026-03-05T11:00"))  # collect, read whole
W3 = (_t("2026-03-05T11:00"), _t("2026-03-05T23:00"))  # collect, truncated, uncounted


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(settings, name, tmp_path / name.lower())
    monkeypatch.setattr(settings, "X_RAW_DIR", tmp_path / "x")
    monkeypatch.setattr(settings, "X_SAMPLING_DIR", tmp_path / "x_sampling")
    monkeypatch.setattr(settings, "X_COUNTS_DIR", tmp_path / "x_counts")
    monkeypatch.setattr(settings, "X_ARCHIVE_DIR", tmp_path / "x_archive")
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(settings, "COLLECTION_START", F)
    monkeypatch.setattr(timeline, "ANALYSIS_END", date(2026, 3, 5))
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"Ss": 5})
    return tmp_path


def _mrow(s, e, truncated, got, cap=5, handle="Ss"):
    return {
        "run_end": "2026-03-05T23:00:00Z",
        "handle": handle,
        "tier": "t",
        "mode": "sample",
        "hours": 35,
        "window_start": xcn.z(s),
        "window_end": xcn.z(e),
        "cap": cap,
        "got": got,
        "new": got,
        "truncated": truncated,
    }


def _manifest(rows) -> None:
    p = settings.X_SAMPLING_DIR / "windows_2026-03-05.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _counts(handle: str, by_hour: dict[datetime, int], start=F, end=W2[0]) -> None:
    """X's hour counts over [start, end): by_hour, 0 elsewhere."""
    c = xcn.CountsCache(f"from:{handle}", "hour")
    rows, t = [], start
    while t < end:
        rows.append({"start": xcn.z(t), "end": xcn.z(t + H), "tweet_count": by_hour.get(t, 0)})
        t += H
    c.merge(rows, (start, end))
    c.save()


def _post(i, user, t, war=True, score=0.5, tier="t"):
    return {
        "id": f"{user}-{i}",
        "user": user,
        "tier": tier,
        "text": "strikes on Iran" if war else "lunch today",
        "created_at": pd.Timestamp(t, tz="UTC"),
        "score_opus": score,
    }


def _grid_env():
    """Ss (sampled, cap 5): B0 read whole (20 posts, X counts 20), W1 at the
    cap (5 held, X counts 30), W2 read whole but past the counts (2 held),
    W3 truncated past the counts (3 held). Tt (take-all): 4 posts."""
    _manifest([_mrow(*W1, True, 5), _mrow(*W2, False, 2), _mrow(*W3, True, 3)])
    by_hour = {F + H * k: 1 for k in range(1, 21)}  # B0's 20 posts, one an hour
    by_hour.update({W1[0] + H * (10 + k): 6 for k in range(5)})  # 30 in W1
    _counts("Ss", by_hour)
    posts = [_post(k, "Ss", F + H * (k + 1), score=0.8) for k in range(20)]
    posts += [_post(20 + k, "Ss", W1[0] + H * (10 + k), score=0.2) for k in range(5)]
    posts += [_post(30 + k, "Ss", W2[0] + H * (2 + k), score=0.4) for k in range(2)]
    posts += [_post(40 + k, "Ss", W3[0] + H * (1 + k), score=0.4) for k in range(3)]
    posts += [_post(50 + k, "Tt", F + H * (30 + k), score=-0.5) for k in range(4)]
    return pd.DataFrame(posts)


def _w(df, scheme):
    w, table = ww.post_weights(df, scheme)
    return dict(zip(df["id"], w)), table


# ── Grid and weights ──────────────────────────────────────────────────


def test_grid_is_the_fills_with_the_manifest_flags(env):
    _grid_env()
    t = ww.account_windows("Ss", xcn.load_manifests())
    got = list(zip(t["s"], t["e"], t["origin"], t["truncated"]))
    assert got == [
        (*B0, "backward", None),
        (*W1, "manifest", True),
        (*W2, "manifest", False),
        (*W3, "manifest", True),
    ]
    assert t["x_count"].iloc[0] == 20 and t["x_count"].iloc[1] == 30
    assert t["x_count"].iloc[2:].isna().all()  # past the counts' coverage


def test_volume_and_design_weights(env):
    df = _grid_env()
    vol, table = _w(df, "volume")
    des, _ = _w(df, "design")
    assert vol["Ss-0"] == 1.0 and des["Ss-0"] == pytest.approx(5 / 20)  # read whole
    assert vol["Ss-20"] == pytest.approx(30 / 5) and des["Ss-20"] == 1.0  # at the cap
    assert vol["Ss-30"] == 1.0 and des["Ss-30"] == 1.0  # uncounted, read whole
    assert all(vol[f"Tt-{50 + k}"] == 1.0 and des[f"Tt-{50 + k}"] == 1.0 for k in range(4))
    assert table.set_index("s").loc[W1[0], "held"] == 5
    # volume weights add up to X's count wherever it is cached
    w = pd.Series(vol)
    assert w[[f"Ss-{k}" for k in range(25)]].sum() == pytest.approx(50)


def test_a_post_belongs_to_the_window_its_time_starts(env):
    """[s, e): a post at a window's edge second is the later window's; one
    past the grid weighs 1 and is counted as outside."""
    _manifest([_mrow(*W1, True, 5)])
    _counts("Ss", {W1[0]: 4, W1[1]: 8}, end=W1[1] + 2 * H)
    df = pd.DataFrame(
        [
            _post(1, "Ss", W1[0]),  # W1's first second
            _post(2, "Ss", W1[0] + H),
            _post(3, "Ss", W1[1]),  # W1's end: past the grid (W1 is the last window)
            _post(4, "Ss", W1[0] - timedelta(seconds=1)),  # the backward window's last second
        ]
    )
    w, table = _w(df, "volume")
    assert table.set_index("s")["held"].to_dict() == {B0[0]: 1, W1[0]: 2}
    assert table.attrs["outside"] == {"Ss": 1}
    assert w["Ss-3"] == 1.0
    assert w["Ss-1"] == w["Ss-2"] == pytest.approx(4 / 2)  # X counts 4 in W1, 2 held
    assert w["Ss-4"] == 1.0  # B0: X counts 0 there, so its one post weighs held / held


def test_edge_buckets_are_apportioned_by_overlap(env):
    c = xcn.CountsCache("from:Ss", "hour")
    t0 = _t("2026-03-02T10:00")
    c.merge(
        [
            {"start": xcn.z(t0 + H * k), "end": xcn.z(t0 + H * (k + 1)), "tweet_count": n}
            for k, n in enumerate([8, 6, 4])
        ],
        (t0, t0 + 3 * H),
    )
    s, e = t0 + H / 2, t0 + 2 * H + H / 4  # 10:30 to 12:15
    assert ww.apportioned_count(c, s, e) == pytest.approx(8 * 0.5 + 6 + 4 * 0.25)
    assert ww.apportioned_count(c, s, t0 + 4 * H) is None  # 13:00 is not covered
    # With the posts held: an edge bucket counts its held posts where they
    # are held, and only its posts held nowhere are pro-rated. 10:00 holds 3
    # (1 inside), X counts 8: 1 + 5 x 0.5. 11:00 is inside: X's 6. 12:00
    # holds 5 (1 inside), X counts 4: 1, none unplaced.
    mins = [10, 20, 40] + [125, 140, 150, 160, 170]  # minutes after 10:00
    held = np.sort(np.array([t0 + timedelta(minutes=m) for m in mins], dtype="datetime64[us]"))
    assert ww.apportioned_count(c, s, e, held) == pytest.approx(1 + 5 * 0.5 + 6 + 1)


def test_a_post_held_at_an_edge_counts_only_in_the_window_that_holds_it(env):
    """Two windows share an edge hour: its posts held in the later window are
    not pro-rated into the earlier one as well (which weighed a lone
    read-whole post 2.09 on the real data); only the posts held nowhere are
    split, by overlap."""
    wa = (_t("2026-03-02T13:30"), _t("2026-03-04T00:30"))  # collect, read whole
    b0 = (F, wa[0])  # the backward window, its 2.5 h head merged in
    _manifest([_mrow(*wa, False, 3)])
    edge, five = _t("2026-03-02T13:00"), _t("2026-03-02T05:00")
    posts = [_post(0, "Ss", five)]  # b0's one post
    posts += [_post(1 + k, "Ss", edge + timedelta(minutes=40 + 5 * k)) for k in range(3)]
    df = pd.DataFrame(posts)  # 13:40, 13:45, 13:50: in wa
    _counts("Ss", {five: 1, edge: 3}, end=wa[1] + H)
    vol, table = _w(df, "volume")
    des, _ = _w(df, "design")
    assert list(zip(table["s"], table["e"])) == [b0, wa]
    assert table["x_count"].tolist() == pytest.approx([1.0, 3.0])  # b0 was 1 + 3 x 0.5
    assert set(vol.values()) == {1.0} and set(des.values()) == {1.0}
    # two posts in the edge hour that the cache does not hold: half each side
    _counts("Ss", {five: 1, edge: 5}, end=wa[1] + H)
    vol, table = _w(df, "volume")
    assert table["x_count"].tolist() == pytest.approx([1 + 2 * 0.5, 3 + 2 * 0.5])
    assert vol["Ss-0"] == pytest.approx(2.0) and vol["Ss-1"] == pytest.approx(4 / 3)
    assert sum(vol.values()) == pytest.approx(1 + 5)  # X's count, as before


def test_a_manifest_window_without_its_truncated_flag_is_not_read_whole(env):
    """Only an explicit `truncated: false` says the run read a window whole: a
    row without the flag, or with null, past the counts gets no weight (it
    was coerced to false, so weight 1)."""
    df = _grid_env()
    for drop in (True, False):
        row = _mrow(*W2, False, 2)
        if drop:
            del row["truncated"]
        else:
            row["truncated"] = None
        _manifest([_mrow(*W1, True, 5), row, _mrow(*W3, True, 3)])
        vol, table = _w(df, "volume")
        des, _ = _w(df, "design")
        assert table.set_index("s").loc[W2[0], "truncated"] is None
        assert np.isnan(vol["Ss-30"]) and np.isnan(des["Ss-30"])  # was 1.0


def test_a_window_holding_more_than_x_counts_weighs_one(env):
    """X counts the posts there now (one deleted since a pull read it is held
    but not counted): N_w is never under h_w."""
    _manifest([_mrow(*W1, True, 5)])
    _counts("Ss", {W1[0]: 2}, end=W1[1])
    df = pd.DataFrame([_post(k, "Ss", W1[0] + H * k) for k in range(4)])
    vol, _ = _w(df, "volume")
    des, _ = _w(df, "design")
    assert set(vol.values()) == {1.0} and set(des.values()) == {1.0}


def test_take_all_search_and_trump_rows_weigh_one(env):
    _grid_env()
    df = pd.DataFrame(
        [
            _post(1, "Tt", F + H),
            _post(2, "search:Iran war", F + H, tier="search"),
            _post(3, "realDonaldTrump", F + H, tier="trump"),
        ]
    )
    for scheme in ("design", "volume"):
        w, table = ww.post_weights(df, scheme)
        assert (w == 1.0).all() and table.empty


# ── Windows past the counts ────────────────────────────────────────────


def test_uncounted_windows_get_a_weight_only_where_it_needs_no_count(env):
    df = _grid_env()
    df = pd.concat(
        [df, pd.DataFrame([_post(60 + k, "Ss", W3[0] + H * (5 + k)) for k in range(2)])],
        ignore_index=True,
    )  # W3 now holds 5, its cap
    vol, _ = _w(df, "volume")
    des, _ = _w(df, "design")
    assert np.isnan(vol["Ss-40"])  # truncated, uncounted: X's count unknown
    assert des["Ss-40"] == 1.0  # at its cap: cap / held whatever X counts
    df3 = _grid_env()  # W3 holds 3 of its cap 5, truncated: design needs the count too
    des3, _ = _w(df3, "design")
    assert np.isnan(des3["Ss-40"])


def test_a_post_without_a_weight_is_refused_never_weighted_one(env):
    df, _ = ww.attach(_grid_env(), "volume")
    d = inf.prepare(df, "score_opus", topic_source="keyword")
    with pytest.raises(ww.WeightsUnavailable, match=r"@Ss 3.*x-counts --only Ss --estimate"):
        inf.bootstrap_cells(d, "score_opus", n_boot=5, seed=1)
    with pytest.raises(ww.WeightsUnavailable):
        plots.weekly_war_stance(d, "score_opus", ["t"])
    with pytest.raises(ww.WeightsUnavailable):
        web_export.maga_weekly(d.assign(tier="maga_prowar"), "score_opus")
    # without the uncounted posts the same frame goes through
    ok = d[~d["id"].isin([f"Ss-{40 + k}" for k in range(3)])]
    inf.bootstrap_cells(ok, "score_opus", n_boot=5, seed=1)


# ── The weighted tables ────────────────────────────────────────────────


def _counted_frame(env_df: pd.DataFrame) -> pd.DataFrame:
    """The grid fixture without W2 / W3 (past the counts)."""
    return env_df[~env_df["id"].isin([f"Ss-{k}" for k in (30, 31, 40, 41, 42)])]


def test_weighted_cells_are_weighted_ratios(env):
    df, _ = ww.attach(_counted_frame(_grid_env()), "design")
    d = inf.prepare(df, "score_opus", topic_source="keyword")
    boot = inf.bootstrap_cells(d, "score_opus", n_boot=50, seed=3)
    st = inf.phase_stats(boot).set_index("phase").loc["strikes, ceasefire"]
    # design: B0's 20 posts at 0.25 (0.8), W1's 5 at 1 (0.2), Tt's 4 at 1 (-0.5)
    want = (20 * 0.25 * 0.8 + 5 * 0.2 + 4 * -0.5) / (20 * 0.25 + 5 + 4)
    assert st["mean_on"] == pytest.approx(want)
    assert st["n"] == 29 and st["n_on"] == 29  # post counts
    assert st["w_on"] == pytest.approx(14.0)
    assert st["neff_on"] == pytest.approx(14.0**2 / (20 * 0.25**2 + 5 + 4))
    assert st["mean_on_lo"] <= st["mean_on"] <= st["mean_on_hi"]
    vol, _ = ww.attach(_counted_frame(_grid_env()), "volume")
    dv = inf.prepare(vol, "score_opus", topic_source="keyword")
    sv = inf.phase_stats(inf.bootstrap_cells(dv, "score_opus", n_boot=5, seed=3)).iloc[0]
    assert sv["mean_on"] == pytest.approx((20 * 0.8 + 30 * 0.2 + 4 * -0.5) / 54)


def test_the_weighted_bootstrap_keeps_the_account_day_blocks(env):
    """Replicate r of a weighted cell is the ratio of the account-days'
    weighted sums, each times its resampling count in _block_weights."""
    df, _ = ww.attach(_counted_frame(_grid_env()), "volume")
    d = inf.prepare(df, "score_opus", topic_source="keyword")
    boot = inf.bootstrap_cells(d, "score_opus", n_boot=30, seed=7)
    c = inf._clusters(d, "score_opus")
    w = inf._block_weights(c["user"], 30, 7)
    want = (w.T @ c["s_on"].to_numpy()) / (w.T @ c["n_on"].to_numpy())
    got = boot["cells"][("t", "strikes, ceasefire")]["mean_on"]
    np.testing.assert_allclose(got, want)


def test_weighted_weekly_series(env, monkeypatch):
    # the fixture's window is worth 14.3 posts (next test): under the gate
    monkeypatch.setattr(settings, "WEB_MIN_WINDOW_POSTS", 10)
    df, _ = ww.attach(_counted_frame(_grid_env()), "volume")
    d = inf.prepare(df, "score_opus", topic_source="keyword").assign(tier="maga_prowar")
    wk = plots.weekly_war_stance(d, "score_opus", ["maga_prowar"])
    raw = plots.weekly_war_stance(d.drop(columns=inf.WEIGHT_COL), "score_opus", ["maga_prowar"])
    assert wk["n"].tolist() == raw["n"].tolist()  # post counts either way
    week = wk.set_index("week")
    first = pd.Timestamp("2026-02-23")  # Mar 1 is a Sunday: B0's first posts
    second = pd.Timestamp("2026-03-02")
    in_second = d[pd.to_datetime(d["created_at"]).dt.tz_localize(None) >= second]
    want = (in_second[inf.WEIGHT_COL] * in_second["score_opus"]).sum() / in_second[
        inf.WEIGHT_COL
    ].sum()
    assert week.loc[second, "stance"] == pytest.approx(want)
    assert first in week.index
    rows = {r["week"]: r for r in web_export.maga_weekly(d, "score_opus")}
    assert rows["2026-03-02"]["stance"] == pytest.approx(round(want, 3), abs=1e-3)
    assert rows["2026-03-02"]["n"] == 9  # Tt's 4 and W1's 5: posts, not weights


def test_the_weighted_weekly_gate_is_on_effective_posts(env, monkeypatch):
    """A weighted window is kept on its Kish effective number of posts, not
    its post count: under volume the fixture's window holds 29 war posts
    (B0's 20 at 1, W1's 5 at 6, Tt's 4 at 1) but is worth
    54^2 / (20 + 5 x 36 + 4) = 14.3, under WEB_MIN_WINDOW_POSTS (15)."""
    assert settings.WEB_MIN_WINDOW_POSTS == 15
    df, _ = ww.attach(_counted_frame(_grid_env()), "volume")
    d = inf.prepare(df, "score_opus", topic_source="keyword").assign(tier="maga_prowar")
    neff = 54**2 / (20 + 5 * 36 + 4)
    wk = plots.weekly_war_stance(d, "score_opus", ["maga_prowar"])
    assert (wk["n_window"] == 29).all() and wk["smooth"].isna().all()
    assert web_export.maga_weekly(d, "score_opus") == []
    raw = d.drop(columns=inf.WEIGHT_COL)  # 29 posts at weight 1: kept
    assert plots.weekly_war_stance(raw, "score_opus", ["maga_prowar"])["smooth"].notna().all()
    assert len(web_export.maga_weekly(raw, "score_opus")) == 2
    monkeypatch.setattr(settings, "WEB_MIN_WINDOW_POSTS", int(neff))  # 14: kept
    wk = plots.weekly_war_stance(d, "score_opus", ["maga_prowar"])
    assert wk["smooth"].notna().all()
    assert len(web_export.maga_weekly(d, "score_opus")) == 2
    np.testing.assert_allclose(inf.kish_n([54.0, 3.0, 0.0], [204.0, 3.0, 0.0]), [neff, 3, 0])


# ── "none" is today's path, bit for bit ────────────────────────────────


def _clusters_before(d, score_col):
    """inference._clusters as it was before the weights (ebf98fa)."""
    x = d.assign(
        s_on=np.where(d["on_topic"], d[score_col], 0.0),
        n_on=d["on_topic"].astype(int),
        s_off=np.where(d["on_topic"], 0.0, d[score_col]),
        n_off=(~d["on_topic"]).astype(int),
    )
    return (
        x.groupby(["user", "day", "group", "phase"], observed=True)[
            ["s_on", "n_on", "s_off", "n_off"]
        ]
        .sum()
        .reset_index()
    )


def _bootstrap_before(d, score_col, n_boot, seed):
    """inference.bootstrap_cells as it was before the weights (ebf98fa)."""
    c = _clusters_before(d, score_col)
    w = inf._block_weights(c["user"], n_boot, seed)
    cells, counts = {}, {}
    for (g, p), idx in c.groupby(["group", "phase"], observed=True).indices.items():
        cells[(g, p)] = inf._cell(c.iloc[idx], w[idx])
        sub = c.iloc[idx]
        counts[(g, p)] = {
            "n": int(sub["n_on"].sum() + sub["n_off"].sum()),
            "n_on": int(sub["n_on"].sum()),
            "n_accounts": int(sub["user"].nunique()),
            "n_days": len(sub),
        }
    return {"cells": cells, "counts": counts}


def _mixed_posts() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k in range(400):
        t = pd.Timestamp("2026-02-01", tz="UTC") + pd.Timedelta(
            hours=int(rng.integers(0, 24 * 150))
        )
        user = ["a", "b", "c", "d", "e"][k % 5]
        rows.append(
            {
                "id": str(k),
                "user": user,
                "tier": "t1" if user in "abc" else "t2",
                "created_at": t,
                "text": "Iran strikes" if rng.random() < 0.4 else "lunch",
                "score_opus": float(np.round(rng.uniform(-1, 1), 3)),
            }
        )
    return pd.DataFrame(rows)


def test_none_is_the_path_before_the_weights_bit_for_bit(monkeypatch):
    df = _mixed_posts()
    monkeypatch.setattr(ww, "post_weights", lambda *a, **k: pytest.fail("none reads no grid"))
    out, table = ww.attach(df, "none")
    assert out is df and table.empty  # nothing added, nothing read
    d = inf.prepare(out, "score_opus", topic_source="keyword")
    assert inf.WEIGHT_COL not in d
    now = inf.bootstrap_cells(d, "score_opus", n_boot=200, seed=11)
    before = _bootstrap_before(d, "score_opus", n_boot=200, seed=11)
    pairs = [("t1", "t2")]
    for f in (inf.phase_stats, inf.phase_contrasts):
        pd.testing.assert_frame_equal(f(now), f(before), check_exact=True)
    pd.testing.assert_frame_equal(
        inf.phase_gaps(now, pairs), inf.phase_gaps(before, pairs), check_exact=True
    )
    d["tier"] = np.where(d["tier"] == "t1", "maga_prowar", "maga_antiwar")
    assert web_export.maga_weekly(d, "score_opus")  # the raw weekly path runs as before


def test_unit_weights_give_the_raw_numbers(monkeypatch):
    df = _mixed_posts()
    d = inf.prepare(df, "score_opus", topic_source="keyword")
    raw = inf.phase_stats(inf.bootstrap_cells(d, "score_opus", n_boot=100, seed=5))
    ones = d.assign(**{inf.WEIGHT_COL: 1.0})
    st = inf.phase_stats(inf.bootstrap_cells(ones, "score_opus", n_boot=100, seed=5))
    pd.testing.assert_frame_equal(st[raw.columns], raw, check_exact=False, rtol=1e-12)
    assert (st["w_on"] == st["n_on"]).all() and (st["neff_on"] == st["n_on"]).all()
    wk_raw = plots.weekly_war_stance(d.assign(tier="t"), "score_opus", ["t"])
    wk_one = plots.weekly_war_stance(ones.assign(tier="t"), "score_opus", ["t"])
    pd.testing.assert_frame_equal(wk_one, wk_raw, check_exact=False, rtol=1e-12)


# ── CLI ────────────────────────────────────────────────────────────────


def test_phases_weights_write_suffixed_files_and_refuse_missing_weights(env):
    _counted_frame(_grid_env()).to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    from src import cli

    runner = CliRunner()
    args = ["phases", "--topic", "keyword", "--n-boot", "20"]
    res = runner.invoke(cli.main, [*args, "--weights", "design"])
    assert res.exit_code == 0, res.output
    out = settings.PROCESSED_DIR
    assert (out / "phases_x_tier_score_opus_keyword_design-weighted.csv").exists()
    assert not (out / "phases_x_tier_score_opus_keyword.csv").exists()
    assert "design-weighted (raw is the record)" in res.output
    # the all-post table's war column is the weighted share, not a post count
    assert "war is the weighted war share" in res.output
    st = pd.read_csv(out / "phases_x_tier_score_opus_keyword_design-weighted.csv")
    assert {"w_on", "neff_on"} <= set(st.columns)
    res = runner.invoke(cli.main, args)  # raw: the file of record, no weight columns
    assert res.exit_code == 0, res.output
    raw = pd.read_csv(out / "phases_x_tier_score_opus_keyword.csv")
    assert "w_on" not in raw.columns
    _grid_env().to_parquet(settings.SENTIMENT_OUTPUT, index=False)  # W3: no count, truncated
    res = runner.invoke(cli.main, [*args, "--weights", "volume"])
    assert res.exit_code == 1 and "x-counts --only Ss --estimate" in res.output
    assert not (out / "phases_x_tier_score_opus_keyword_volume-weighted.csv").exists()


def test_phase_weights_setting_is_the_record():
    assert settings.PHASE_WEIGHTS == "none"
    assert settings.PHASE_WEIGHTS in ww.SCHEMES
