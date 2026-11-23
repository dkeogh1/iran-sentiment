"""Same accounts or a different crowd (`reply-crowd`, src/analysis/reply_crowd.py):
groups from every reply keyed on the account id, contrasts from the settings'
late phase, replicate weights, the post-stratified and GREG domain means, the
decomposition's identity, pairs, the panel and the CLI. Stub data, no API.

The late phase is "expiry, strikes" here: hold_off_attack and deal_complete
(phase 2) are the base, strikes_resume_sep the late post."""

import inspect
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings, timeline
from src.analysis import inference as inf
from src.analysis import reply_crowd as rc
from src.analysis import stance_local as sl
from tests.test_reply_labels import MODEL, _no_api, _replies, env  # noqa: F401 -- the fixture
from tests.test_reply_loyalty import HIGH, LOW, MEDIUM

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

EARLY = ["hold_off_attack", "deal_complete"]
LATE = ["strikes_resume_sep"]


def _acct(aid: str, base=MEDIUM) -> dict:
    return {**base, "id": aid, "username": f"user_{aid}"}


def _crowd_replies(n_per_post=40) -> pd.DataFrame:
    """Even replies come from ten accounts that reply to every post (both
    periods), odd ones from an account seen on that post only."""
    reps = _replies(n_per_post)
    i = reps["id"].str.rsplit("-", n=1).str[1].astype(int)
    aid = [
        f"shared{(k // 2) % 10}" if k % 2 == 0 else f"{s}-u{k}"
        for s, k in zip(reps["tracked_slug"], i)
    ]
    tiers = (HIGH, MEDIUM, LOW)
    return reps.assign(account=[_acct(a, tiers[n % 3]) for n, a in enumerate(aid)])


def _label(reps: pd.DataFrame, drawn: pd.DataFrame, fn) -> pd.Series:
    """Write v2 labels on the draws: fn(row of reps) -> score."""
    by = reps.set_index("id")
    score = [fn(by.loc[i]) for i in drawn["id"]]
    pd.DataFrame(
        {
            "id": drawn["id"],
            "tracked_slug": drawn["tracked_slug"],
            "input_chars": 20,
            "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
            "score_teacher": score,
            "label_teacher": "neutral",
        }
    ).to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)
    return pd.Series(score, index=drawn["id"].to_numpy())


def _shared(row) -> bool:
    return row["account"]["id"].startswith("shared")


@pytest.fixture
def crowd(env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(settings, "REPLY_CROWD_LATE_PHASE", "expiry, strikes")
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_ACCOUNTS", 1)

    def make(reps=None, n_per_bucket=1000, fn=None):
        reps = _crowd_replies() if reps is None else reps
        reps, drawn = env(reps, n_per_bucket=n_per_bucket)
        y = _label(reps, drawn, fn or (lambda r: 0.0))
        return reps, drawn, y

    return make


# ── 1. groups ──────────────────────────────────────────────────────


def test_groups_use_every_reply_and_the_account_id():
    reps = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6"],
            "tracked_slug": ["a", "b", "a", "b", "a", "b"],
            "text": ["words here", "", "words here", "more words", "words", "words here"],
            "account": [
                _acct("x"),
                _acct("x"),  # x's reply to b has no text
                {**_acct("r"), "username": "old"},
                {**_acct("r"), "username": "new"},  # renamed: one account
                None,  # no account record
                _acct("z"),
            ],
        }
    )
    part = rc.participation(reps, "all")
    assert part["replier"].tolist()[:4] == ["x", "x", "r", "r"]
    assert pd.isna(part["replier"].iloc[4])
    g = rc.r1_groups(part)
    assert g["1"] == "returner" and g["3"] == "returner" and g["6"] == "one_off"
    assert "5" not in g.index  # counted in part, in no group
    text = rc.r1_groups(rc.participation(reps, "text"))
    assert text["1"] == "one_off"  # its other reply had no text
    assert part.loc[part["id"] == "1", "m"].item() == 1
    c = rc.crowd_groups(part, ["a"], ["b"])
    assert c["1"] == "both" and c["3"] == "both" and c["6"] == "late_only"
    p = rc.pair_groups(part, "a", "b")
    assert set(p.unique()) == {"both_qp", "p_only"}
    with pytest.raises(ValueError):
        rc.participation(reps, "some")


# ── 2. contrasts ───────────────────────────────────────────────────


def test_contrast_posts_from_the_setting_never_the_last_phase(monkeypatch, caplog):
    slugs = ["power_plant_day", "hold_off_attack", "deal_complete", "strikes_resume_sep"]
    with pytest.raises(ValueError, match="not a phase"):
        rc.contrast_posts(("no such phase",), ("talks, MOU",), slugs)
    with pytest.raises(ValueError, match="first phase"):
        rc.contrast_posts(("talks, MOU",), ("strikes, ceasefire",), slugs)
    with pytest.raises(ValueError, match="empty side"):
        rc.contrast_posts(("collapse, blockade",), ("expiry, strikes",), slugs)
    with pytest.raises(ValueError, match="TRACKED_POSTS"):
        rc.contrast_posts(("talks, MOU",), ("expiry, strikes",), [*slugs, "nope"])
    with pytest.raises(ValueError, match="both sides"):
        rc.contrast_posts(("talks, MOU",), ("talks, MOU", "expiry, strikes"), slugs)
    with pytest.raises(ValueError, match="precede"):  # a contrast backwards in time
        rc.contrast_posts(("collapse, blockade", "expiry, strikes"), ("talks, MOU",), slugs)
    with pytest.raises(ValueError, match="precede"):
        rc.contrast_posts(("strikes, ceasefire", "expiry, strikes"), ("talks, MOU",), slugs)
    near = [c for c in rc.crowd_contrasts(slugs, "talks, MOU") if c.name == "near"]
    assert near == []  # the near base comes after "talks, MOU": skipped, not reversed
    with caplog.at_level("INFO", logger="src.analysis.reply_crowd"):
        early, late = rc.contrast_posts(("strikes, ceasefire",), ("talks, MOU",), slugs)
    assert early == ["power_plant_day"] and late == ["hold_off_attack", "deal_complete"]
    assert "strikes_resume_sep" in caplog.text  # dated after the late phase
    monkeypatch.setattr(settings, "REPLY_CROWD_LATE_PHASE", "expiry, strikes")
    c = rc.default_contrast(slugs)
    assert c.early == ["power_plant_day", "hold_off_attack", "deal_complete"]
    assert c.late == ["strikes_resume_sep"]
    more = [*timeline.PHASES, ("later", date(2027, 1, 3), date(2027, 2, 1))]
    monkeypatch.setattr(timeline, "PHASES", more)
    assert rc.default_contrast(slugs) == c


# ── 3. replicate weights ───────────────────────────────────────────


def test_replicate_weights_reproduce_weighted_with_its_rng():
    rng = np.random.default_rng(3)
    pop = pd.DataFrame(
        {
            "tracked_slug": ["deal_complete"] * 60 + ["hold_off_attack"] * 40,
            "bucket": rng.integers(0, 3, 100),
        }
    )
    lab = pop.groupby(["tracked_slug", "bucket"]).sample(frac=0.4, random_state=1)
    lab = lab.reset_index(drop=True).assign(id=lambda d: d.index.astype(str))
    y = rng.normal(size=len(lab))
    _, reps, _ = inf._weighted(pop, lab, {"y": y}, 300, np.random.default_rng(7))
    W = rc.replicate_weights(lab, pop, 300, 0, rng=np.random.default_rng(7))
    assert np.allclose((W * y[:, None]).sum(0) / len(pop), reps["y"], atol=1e-5)
    assert (W > 0).all()
    # Per-post streams: a post's replicates do not depend on the others.
    W1 = rc.replicate_weights(lab, pop, 50, 42)
    only = lab[lab["tracked_slug"] == "deal_complete"].reset_index(drop=True)
    W2 = rc.replicate_weights(only, pop, 50, 42)
    assert np.allclose(W1[(lab["tracked_slug"] == "deal_complete").to_numpy()], W2)


# ── 4, 5, 10. the decomposition ────────────────────────────────────


def _varying_census_column(reps: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A census score across every stance band, so GREG's slopes are fit."""
    rng = np.random.default_rng(seed)
    return reps.assign(score_opus_distilled=np.round(rng.uniform(-1, 1, len(reps)), 4))


def test_identity_holds_in_every_replicate(crowd):
    rng = np.random.default_rng(0)
    reps = _varying_census_column(_crowd_replies())
    crowd(reps, n_per_bucket=4, fn=lambda r: float(rng.choice([-0.5, 0.0, 0.3, 0.8])))
    fr = rc.crowd_frames()
    assert fr.pop["f"].nunique() > 10
    W = rc.replicate_weights(fr.lab, fr.pop, 40, 1)
    Wa = rc.account_weights(fr.lab, fr.pop, 40, 1)
    g = rc.crowd_groups(fr.part, EARLY, LATE)
    for WW in (W, Wa):
        for per_account in (False, True):
            for assist in (False, True):
                ds = rc.domain_means(fr, g, WW, per_account=per_account, assist=assist)
                terms = {k: rc.decompose(ds, EARLY, LATE, k) for k in rc.OUTCOMES}
                for t in terms.values():  # an account draw can empty a domain: NaN
                    assert np.isfinite(t["change"][0])
                    assert np.allclose(
                        t["change"], t["same_accounts"] + t["turnover"], equal_nan=True
                    )
                for term in ("change", "same_accounts", "turnover", "did"):
                    total = sum(terms[k][term] for k in ("pro", "anti", "neutral"))
                    assert np.allclose(total[np.isfinite(total)], 0, atol=1e-9)
                    assert np.isfinite(total[0])
                for p in fr.posts:  # the shares sum to 1, assisted too
                    for grp in rc.R2_GROUPS:
                        tot = sum(ds.mean(p, grp, k) for k in ("pro", "anti", "neutral"))
                        assert np.allclose(tot[np.isfinite(tot)], 1, atol=1e-9)
            direct = rc.domain_means(fr, g, WW)
            greg = rc.domain_means(fr, g, WW, assist=True)
            diff = [
                abs(greg.mean(p, grp, k)[0] - direct.mean(p, grp, k)[0])
                for p in fr.posts
                for grp in rc.R2_GROUPS
                for k in ("pro", "anti")
            ]
            assert np.nanmax(diff) > 1e-4  # the slopes are fit: GREG is not direct here
    pairs = rc.reply_crowd_pairs(fr, W, rc.Contrast("x", EARLY, LATE))
    rows = pairs[~pairs["kind"].str.startswith("summary")]
    assert len(rows) == 3 * len(rc.OUTCOMES)  # 3 ordered pairs of 3 posts
    assert np.allclose(rows["pair_change"], rows["same_accounts"] + rows["composition"])


def test_composition_alone_and_a_shift_alone(crowd):
    # Labels by group only: the accounts seen in both periods say 0.2 on
    # every post, the others -0.5 early and +0.5 late. Census sample.
    crowd(fn=lambda r: 0.2 if _shared(r) else (0.5 if r["tracked_slug"] in LATE else -0.5))
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 20, 1)
    ds = rc.domain_means(fr, rc.crowd_groups(fr.part, EARLY, LATE), W)
    t = rc.decompose(ds, EARLY, LATE, "mean")
    assert np.allclose(t["same_accounts"], 0)
    assert t["turnover"][0] == pytest.approx(0.5 * 0.5 - (-0.5 * 0.5))  # half each side
    assert np.allclose(t["change"], t["turnover"])
    # Every account moves by +0.5 from the earlier posts to the late one.
    crowd(fn=lambda r: 0.5 if r["tracked_slug"] in LATE else 0.0)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 20, 1)
    ds = rc.domain_means(fr, rc.crowd_groups(fr.part, EARLY, LATE), W)
    t = rc.decompose(ds, EARLY, LATE, "mean")
    assert np.allclose(t["turnover"], 0) and np.allclose(t["same_accounts"], 0.5)
    assert np.allclose(t["did"], 0)


def test_hajek_reproduces_reply_population(crowd):
    rng = np.random.default_rng(5)
    crowd(n_per_bucket=4, fn=lambda r: float(rng.choice([-0.6, 0.0, 0.4])))
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 30, 1)
    ds = rc.domain_means(fr, rc.r1_groups(fr.part), W, estimator="hajek")
    rp = inf.reply_population(n_boot=10).set_index("post")
    for p in fr.posts:
        for k in rc.OUTCOMES:
            assert ds.post_mean(p, k)[0] == pytest.approx(rp.loc[p, f"opus_{k}"])
    assert rc._close_to_population(fr, W) == []


# ── 6, 7. weights, census and the cell rule ────────────────────────


def test_a_census_sample_gives_the_truth_with_no_width(crowd):
    reps, _, _ = crowd(fn=lambda r: r["score_transformer"] + (0.3 if _shared(r) else 0.0))
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 30, 1)
    ds = rc.domain_means(fr, rc.r1_groups(fr.part), W)
    for p in fr.posts:
        sub = reps[reps["tracked_slug"] == p]
        ret = sub[sub["account"].map(lambda a: a["id"].startswith("shared"))]
        truth = (ret["score_transformer"] + 0.3).mean()
        v = ds.mean(p, "returner", "mean")
        assert np.allclose(v, truth)


def test_oversampled_buckets_weigh_back_to_the_population(crowd, monkeypatch):
    # Half of each post is critical (-1), the rest 0; 6 draws a bucket, every
    # cell post-stratified on its census total (no collapse).
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_CELL_DRAWS", 1)
    crowd(n_per_bucket=6, fn=lambda r: -1.0 if r["score_transformer"] < -0.3 else 0)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 30, 1)
    ds = rc.domain_means(fr, rc.r1_groups(fr.part), W)
    for p in fr.posts:
        assert ds.post_mean(p, "mean")[0] == pytest.approx(-0.5)
        assert ds.post_mean(p, "anti")[0] == pytest.approx(0.5)


def test_small_cells_collapse_and_an_empty_cell_is_nan_and_named(crowd):
    crowd(n_per_bucket=6, fn=lambda r: 0.1)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 10, 1)
    g = rc.r1_groups(fr.part)
    ds = rc.domain_means(fr, g, W)
    assert len(ds.collapsed) == 2 * len(fr.posts) and not ds.nan_cells  # 6 draws < 15
    assert ds.mean("deal_complete", "returner", "mean")[0] == pytest.approx(0.1)
    # No draw left in deal_complete's critical bucket of one-off repliers.
    drop = (
        (fr.lab["tracked_slug"] == "deal_complete")
        & (fr.lab["bucket"] == 0)
        & (fr.lab["id"].map(g) == "one_off")
    )
    f2 = rc.replace(fr, lab=fr.lab[~drop].reset_index(drop=True))
    ds = rc.domain_means(f2, g, rc.replicate_weights(f2.lab, f2.pop, 10, 1))
    assert ds.nan_names() == ["deal_complete/critical/one_off"]
    assert np.isnan(ds.mean("deal_complete", "one_off", "mean")).all()
    assert np.isnan(ds.post_mean("deal_complete", "mean")[0])  # not renormalised
    assert ds.n_flags(["deal_complete"]) == (1, 1)


def test_a_nan_post_makes_the_within_row_nan_and_is_counted(crowd, caplog):
    crowd(n_per_bucket=6, fn=lambda r: 0.5 if _shared(r) else 0.0)
    fr = rc.crowd_frames()
    g = rc.r1_groups(fr.part)
    drop = (
        (fr.lab["tracked_slug"] == "deal_complete")
        & (fr.lab["bucket"] == 0)
        & (fr.lab["id"].map(g) == "one_off")
    )
    f2 = rc.replace(fr, lab=fr.lab[~drop].reset_index(drop=True))
    W = rc.replicate_weights(f2.lab, f2.pop, 10, 1)
    ds = rc.domain_means(f2, g, W)
    assert np.isnan(rc._within(ds, f2.posts, "mean", *rc.R1_GROUPS)).all()  # not renormalised
    assert np.isfinite(rc._within(ds, ["hold_off_attack"], "mean", *rc.R1_GROUPS)[0])
    with caplog.at_level("WARNING", logger="src.analysis.reply_crowd"):
        rt = rc.reply_returners(f2, W, rc.Contrast("x", EARLY, LATE), assist=False)
    assert "deal_complete/critical/one_off" in caplog.text  # named in the log
    rt = rt.set_index("post")
    assert np.isnan(rt.loc["within posts", "direct_mean_gap"])
    assert rt.loc["within posts", "nan_domains"] == 1
    assert rt.loc["within earlier", "nan_domains"] == 1
    assert rt.loc["within late", "nan_domains"] == 0
    assert np.isfinite(rt.loc["within late", "direct_mean_gap"])
    assert rt.loc["ALL pooled (post mix differs)", "nan_domains"] == 1


def test_suppressed_domains_are_counted_and_named(crowd, monkeypatch, caplog):
    crowd(n_per_bucket=6, fn=lambda r: 0.3 if _shared(r) else -0.2)
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_ACCOUNTS", 20)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 10, 1)
    with caplog.at_level("WARNING", logger="src.analysis.reply_crowd"):
        rt = rc.reply_returners(fr, W, rc.Contrast("x", EARLY, LATE), assist=False)
    assert "suppressed domains" in caplog.text and "deal_complete/returner" in caplog.text
    rt = rt.set_index("post")
    assert (rt.loc[fr.posts, "suppressed_domains"] == 1).all()
    assert rt.loc["within posts", "suppressed_domains"] == len(fr.posts)
    assert np.isnan(rt.loc["within posts", "direct_mean_gap"])
    ch = rc.reply_crowd_change(fr, W, [rc.Contrast("x", EARLY, LATE)], assist=False)
    assert (ch["suppressed_domains"] > 0).all()


# ── 8. GREG ────────────────────────────────────────────────────────


def test_greg_is_exact_on_a_perfect_column_and_direct_with_none(crowd, monkeypatch):
    def truth(r):  # varies within every cell, so the slope is fit (1)
        i = int(str(r.get("id", r.name)).rsplit("-", 1)[1])
        return round(
            r["score_transformer"] * 0.5 + (0.2 if _shared(r) else -0.1) + 0.01 * (i % 7), 6
        )

    reps = _crowd_replies()
    reps["score_opus_distilled"] = reps.apply(truth, axis=1)
    crowd(reps, n_per_bucket=6, fn=truth)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 30, 1)
    g = rc.r1_groups(fr.part)
    for min_cell in (15, 1):  # collapsed (6 draws a bucket) and post-stratified domains
        monkeypatch.setattr(settings, "REPLY_CROWD_MIN_CELL_DRAWS", min_cell)
        ds = rc.domain_means(fr, g, W, assist=True)
        assert bool(ds.collapsed) == (min_cell == 15)
        for p in fr.posts:
            sub = reps[reps["tracked_slug"] == p]
            assert np.allclose(ds.post_mean(p, "mean"), sub["score_opus_distilled"].mean())
            ret = sub[sub["account"].map(lambda a: a["id"].startswith("shared"))]
            assert np.allclose(ds.mean(p, "returner", "mean"), ret["score_opus_distilled"].mean())
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_CELL_DRAWS", 15)
    # A constant column carries nothing: the slope is 0 and GREG is direct.
    reps["score_opus_distilled"] = 0.1
    reps.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
    fr = rc.crowd_frames()
    a = rc.domain_means(fr, g, W, assist=True)
    d = rc.domain_means(fr, g, W)
    for p in fr.posts:
        assert np.allclose(a.post_mean(p, "mean"), d.post_mean(p, "mean"))
    sl.reply_columns_path().write_text(
        json.dumps({"score_opus_distilled": {"trained_on_reply_labels": "v2"}})
    )
    with pytest.raises(ValueError, match="cross-fit"):
        rc.crowd_frames()
    assert "col" not in inspect.signature(rc.reply_crowd_panel).parameters


def test_greg_shares_add_up_on_post_stratified_domains(crowd, monkeypatch):
    rng = np.random.default_rng(4)
    reps = _varying_census_column(_crowd_replies(80), seed=4)
    crowd(reps, n_per_bucket=12, fn=lambda r: float(rng.choice([-0.6, 0.0, 0.5])))
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_CELL_DRAWS", 1)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 30, 1)
    g = rc.r1_groups(fr.part)
    ds = rc.domain_means(fr, g, W, assist=True)
    d0 = rc.domain_means(fr, g, W)
    checked = 0
    for p in fr.posts:
        for grp in rc.R1_GROUPS:
            if (p, grp) in ds.collapsed:
                continue
            tot = sum(ds.mean(p, grp, k) for k in ("pro", "anti", "neutral"))
            if np.isfinite(tot[0]):
                assert np.allclose(tot, 1, atol=1e-9)
                assert not np.allclose(ds.mean(p, grp, "pro"), d0.mean(p, grp, "pro"))
                checked += 1
    assert checked >= 2


# ── 9. per account ─────────────────────────────────────────────────


def test_a_heavy_replier_moves_the_reply_mean_not_the_account_mean(crowd):
    reps = _crowd_replies()
    heavy = reps["tracked_slug"].eq("deal_complete") & reps["id"].str[-1].isin(list("13579"))
    reps.loc[heavy, "account"] = [_acct("heavy")] * int(heavy.sum())
    crowd(reps, fn=lambda r: 1.0 if r["account"]["id"] == "heavy" else 0.0)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 10, 1)
    g = rc.r1_groups(fr.part)
    reply = rc.domain_means(fr, g, W).post_mean("deal_complete", "mean")[0]
    acct = rc.domain_means(fr, g, W, per_account=True).post_mean("deal_complete", "mean")[0]
    sub = fr.pop[fr.pop["tracked_slug"] == "deal_complete"]
    assert reply == pytest.approx(heavy.sum() / len(sub))
    assert acct == pytest.approx(1 / sub["replier"].nunique())


# ── 12. the reference ──────────────────────────────────────────────


def test_noise_corrected_rms_is_about_zero_with_no_composition():
    rng = np.random.default_rng(0)
    sd = 0.05
    ts = [np.r_[rng.normal(0, sd), rng.normal(0, sd, 400)] for _ in range(300)]
    out = rc.reference_rms(ts)
    assert out["rms_raw"] == pytest.approx(sd, rel=0.15)
    assert out["rms_corrected"] < 0.4 * out["rms_raw"]
    big = [np.r_[t[0] + 0.2, t[1:] + 0.2] for t in ts]  # a real 0.2 composition term
    assert rc.reference_rms(big)["rms_corrected"] == pytest.approx(0.2, abs=0.02)


# ── 13. suppression and the output whitelist ───────────────────────

BANNED = {"id", "replier", "user", "username", "handle", "text", "account", "account_id"}


def test_small_domains_are_suppressed_and_outputs_hold_aggregates(crowd, monkeypatch):
    reps, _, _ = crowd(n_per_bucket=6, fn=lambda r: 0.3 if _shared(r) else -0.2)
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_ACCOUNTS", 20)
    fr = rc.crowd_frames()
    W = rc.replicate_weights(fr.lab, fr.pop, 10, 1)
    ds = rc.domain_means(fr, rc.r1_groups(fr.part), W)
    # Ten shared accounts a post: returners suppressed; 20 one-off accounts kept.
    assert {(p, g) for p, g, _ in ds.suppressed} == {(p, "returner") for p in fr.posts}
    assert np.isnan(ds.mean("deal_complete", "returner", "mean")).all()
    assert np.isfinite(ds.mean("deal_complete", "one_off", "mean")).all()
    monkeypatch.setattr(settings, "REPLY_CROWD_MIN_ACCOUNTS", 1)
    tables, _ = rc.reply_crowd(n_boot=10)
    handles = set(reps["user"]) | {a["id"] for a in reps["account"]}
    handles |= {a["username"] for a in reps["account"]} | set(reps["id"]) | set(reps["text"])
    for name, df in tables.items():
        assert not BANNED & set(df.columns), name
        cells = {str(v) for v in df.astype(str).to_numpy().ravel()}
        assert not handles & cells, name


def test_r2_account_shares_count_each_period_s_own_accounts(crowd):
    # "ghost" replies with text to an earlier post and without text to the
    # late one: a both-period account (groups count every reply) that is not
    # among the late accounts with text.
    reps = _crowd_replies()
    early_one = reps.index[
        (reps["tracked_slug"] == "deal_complete") & (reps["id"] == "deal_complete-1")
    ]
    late_one = reps.index[reps["id"] == "strikes_resume_sep-1"]
    reps.loc[early_one, "account"] = [_acct("ghost")]
    reps.loc[late_one, "account"] = [_acct("ghost")]
    reps.loc[late_one, "text"] = ""
    crowd(reps)
    fr = rc.crowd_frames()
    c = rc.Contrast("all earlier", EARLY, LATE)
    out = rc._group_counts(fr, rc.crowd_groups(fr.part, EARLY, LATE), c)
    assert out["accounts_both"] == 11  # the ten shared accounts and ghost
    assert out["accounts_both_late"] == 10 and out["accounts_both_earlier"] == 11
    late = fr.pop.loc[fr.pop["tracked_slug"].isin(LATE), "replier"].nunique()
    early = fr.pop.loc[fr.pop["tracked_slug"].isin(EARLY), "replier"].nunique()
    assert out["share_both_of_late_accounts"] == pytest.approx(10 / late)
    assert out["share_both_of_earlier_accounts"] == pytest.approx(11 / early)
    assert out["share_both_of_late_accounts"] <= 1


def test_a_missing_or_partly_written_column_is_refused(crowd, monkeypatch):
    from src.cli import main

    reps = _crowd_replies()
    reps["score_ctx_crossfit"] = np.where(reps["tracked_slug"] == "hold_off_attack", np.nan, 0.2)
    crowd(reps)
    _no_api(monkeypatch)
    # inference._reply_frames refuses first (every reply with text needs a score)
    with pytest.raises(ValueError, match="partly written|have no score_ctx_crossfit"):
        rc.crowd_frames("score_ctx_crossfit")
    res = CliRunner().invoke(main, ["reply-crowd", "--n-boot", "5", "--col", "score_nope"])
    assert res.exit_code == 1 and "score_nope is not in" in res.output
    assert not list(settings.PROCESSED_DIR.glob("reply_crowd_*"))
    reps["score_ctx_crossfit"] = 0.2  # whole: the same base as the record
    crowd(reps)
    assert rc.crowd_frames("score_ctx_crossfit").posts == rc.crowd_frames().posts


def test_variant_runs_never_write_over_the_record(crowd, monkeypatch):
    from src.cli import main

    rng = np.random.default_rng(3)
    crowd(n_per_bucket=8, fn=lambda r: float(rng.choice([-0.5, 0.0, 0.6])))
    _no_api(monkeypatch)
    names = rc.output_names("score_opus_distilled")
    assert names == rc.output_names("score_opus_distilled", late_phase="expiry, strikes")
    pa = rc.output_names("score_opus_distilled", per_account=True)
    assert {k for k in names if names[k] != pa[k]} == {"returners", "change"}
    other = rc.output_names("score_ctx_crossfit")
    assert not set(names.values()) & set(other.values())  # panel and census name the column
    # The robustness table does not depend on --per-account.
    rb = rc.reply_crowd(n_boot=10)[0]["robustness"]
    rb_pa = rc.reply_crowd(n_boot=10, per_account=True)[0]["robustness"]
    pd.testing.assert_frame_equal(rb, rb_pa)
    assert {"no_collapse", "loyalty_standardised"} <= set(rb["variant"])
    res = CliRunner().invoke(main, ["reply-crowd", "--n-boot", "10"])
    assert res.exit_code == 0, res.output
    before = {n: (settings.PROCESSED_DIR / n).read_bytes() for n in names.values()}
    res = CliRunner().invoke(main, ["reply-crowd", "--n-boot", "10", "--per-account"])
    assert res.exit_code == 0, res.output
    assert (settings.PROCESSED_DIR / pa["returners"]).exists()
    # Another late phase suffixes every file.
    monkeypatch.setattr(settings, "REPLY_CROWD_LATE_PHASE", "midterms, standoff")
    late = rc.output_names("score_opus_distilled", late_phase="expiry, strikes")
    assert all(v.endswith("_late-expiry-strikes.csv") for v in late.values())
    res = CliRunner().invoke(
        main, ["reply-crowd", "--n-boot", "10", "--late-phase", "expiry, strikes"]
    )
    assert res.exit_code == 0, res.output
    for k, n in names.items():
        assert (settings.PROCESSED_DIR / late[k]).exists(), k
        assert (settings.PROCESSED_DIR / n).read_bytes() == before[n], k  # the record as it was


# ── 14. the CLI ────────────────────────────────────────────────────


def test_reply_crowd_tables_and_cli(crowd, monkeypatch):
    from src.cli import main

    rng = np.random.default_rng(2)
    reps, _, _ = crowd(n_per_bucket=8, fn=lambda r: float(rng.choice([-0.5, 0.0, 0.6])))
    _no_api(monkeypatch)
    tables, n = rc.reply_crowd(n_boot=20)
    rt = tables["returners"].set_index("post")
    assert {
        "within posts",
        "within earlier",
        "within late",
        "ALL pooled (post mix differs)",
        "within posts, loyalty-standardised",
    } <= set(rt.index)
    assert rt.loc["strikes_resume_sep", "period"] == "late"
    ch = tables["change"]
    assert set(ch["contrast"]) == {"all earlier", "base: talks, MOU"}
    rec = ch[(ch["contrast"] == "all earlier") & (ch["estimator"] == "direct")]
    assert len(rec) == 4 and rec["same_accounts_acct_lo"].notna().all()
    assert rec["share_both_of_late_replies"].iloc[0] == pytest.approx(0.5)
    rb = tables["robustness"]
    assert {
        "hajek",
        "per_account",
        "no_fpc",
        "draws_first",
        "leave_out_hold_off_attack",
        "side_balance",
        "loyalty_standardised",
    } <= set(rb["variant"])
    assert n["late"] == [("strikes_resume_sep", "expiry, strikes")]
    res = CliRunner().invoke(main, ["reply-crowd", "--n-boot", "10"])
    assert res.exit_code == 0, res.output
    assert "same accounts = post effect + change of mind: not separable" in res.output
    for name in rc.output_names("score_opus_distilled").values():
        assert (settings.PROCESSED_DIR / name).exists(), name
    for h in [*reps["user"], *reps["text"], *(a["username"] for a in reps["account"])]:
        assert h not in res.output
    bad = CliRunner().invoke(main, ["reply-crowd", "--late-phase", "strikes, ceasefire"])
    assert bad.exit_code == 1 and "first phase" in bad.output
