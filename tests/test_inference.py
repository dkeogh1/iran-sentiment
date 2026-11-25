"""Phase / topic decomposition, block bootstrap, reply weighting, topic parsing, backup commands."""
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.analysis import inference as inf
from src.analysis import topic_label as tl


def _posts():
    # admin: phase 1 half war posts at +0.6, phase 2 a quarter at +0.6 -> the
    # all-post mean drops purely through topic share.
    rows = []
    for phase_day, n_war in ((date(2026, 3, 1), 5), (date(2026, 5, 1), 2)):
        for i in range(8):
            war = i < n_war
            rows.append({"id": f"{phase_day}-{i}", "user": f"acct{i % 2}", "tier": "admin",
                         "created_at": pd.Timestamp(phase_day, tz="UTC") + pd.Timedelta(days=i),
                         "text": "strikes on Iran" if war else "lunch",
                         "score_opus": 0.6 if war else 0.0})
    return pd.DataFrame(rows)


def test_assign_phase_inclusive_edges():
    s = pd.Series(["2026-04-21T23:59:00Z", "2026-04-22T00:00:00Z", "2026-01-31T12:00:00Z"])
    out = inf.assign_phase(s)
    assert out[0] == "strikes, ceasefire" and out[1] == "talks, MOU" and pd.isna(out[2])


def test_decomposition_attributes_pure_share_change():
    d = inf.prepare(_posts(), "score_opus", topic_source="keyword")
    boot = inf.bootstrap_cells(d, "score_opus", n_boot=50, seed=1)
    c = inf.phase_contrasts(boot).iloc[0]
    assert c["all"] == pytest.approx(0.6 * 2 / 8 - 0.6 * 5 / 8)
    assert c["share_effect"] == pytest.approx(c["all"])
    assert c["on_topic_change"] == pytest.approx(0.0)
    parts = c["share_effect"] + c["on_topic_effect"] + c["off_topic_effect"]
    assert parts == pytest.approx(c["all"])


def test_block_weights_keep_each_accounts_day_count():
    users = pd.Series(["a", "a", "a", "b", "b"])
    w = inf._block_weights(users, n_boot=200, seed=0)
    assert (w[:, 0] == 1).all()
    assert (w[:3, 1:].sum(axis=0) == 3).all() and (w[3:, 1:].sum(axis=0) == 2).all()


def test_war_flag_prefers_llm_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    pd.DataFrame({"id": ["1"], "about_war": [True]}).to_parquet(tl.labels_path(), index=False)
    d = pd.DataFrame({"id": ["1", "2"], "text": ["Deport them!", "Iran strikes"]})
    assert inf.war_flag(d, "llm").tolist() == [True, True]      # 1 from the label, 2 from keywords
    assert inf.war_flag(d, "keyword").tolist() == [False, True]
    pd.DataFrame({"id": ["1", "2"], "about_war": [False, False]}).to_parquet(tl.labels_path(), index=False)
    assert inf.war_flag(d, "llm").tolist() == [False, False]    # the label wins where there is one
    assert inf.war_flag(d, "either").tolist() == [False, True]  # ... unless the keyword fires


def test_prepare_drops_rows_without_text():
    # Two image-only posts (one empty, one a 2-char stub) carry a score like the
    # distilled model's old constant; they must leave the share and the mean.
    posts = _posts()
    extra = pd.DataFrame({"id": ["img1", "img2"], "user": "acct0", "tier": "admin",
                          "created_at": pd.Timestamp("2026-03-02", tz="UTC"),
                          "text": ["", " \U0001f1fa\U0001f1f8 "], "score_opus": 0.111})
    both = pd.concat([posts, extra], ignore_index=True)
    d = inf.prepare(both, "score_opus", topic_source="keyword")
    assert not d["id"].isin(["img1", "img2"]).any()
    boot = inf.bootstrap_cells(d, "score_opus", n_boot=10, seed=1)
    cell = inf.phase_stats(boot).set_index("phase").loc["strikes, ceasefire"]
    assert cell["n"] == 8 and cell["share"] == pytest.approx(5 / 8)
    assert cell["mean_all"] == pytest.approx(0.6 * 5 / 8)


def _replies(spec, text="a reply with text"):
    """Reply frame from {slug: [score_transformer, ...]}; ids <slug>-<i>."""
    rows = [{"id": f"{slug}-{i}", "tracked_slug": slug, "text": text, "score_transformer": v,
             "score_opus_distilled": 0.0}
            for slug, vals in spec.items() for i, v in enumerate(vals)]
    return pd.DataFrame(rows)


@pytest.fixture
def reply_files(tmp_path, monkeypatch):
    from src.analysis import event_study as es
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    monkeypatch.setattr(settings, "REPLY_DRAWS_MANIFEST", tmp_path / "draws.parquet")
    monkeypatch.setattr(es, "STANCE_OUTPUT", tmp_path / "stance_sample.parquet")
    from src.analysis.stance_local import reply_labels_path

    def write(replies: pd.DataFrame, lab: pd.DataFrame, sampled: pd.DataFrame | None = None):
        """sampled: what `stance` drew into stance_sample (default: the labelled rows).
        Labels go to the file reply_population reads: the version settings pick."""
        replies.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
        lab[["id", "score_teacher"]].to_parquet(
            reply_labels_path(version=settings.REPLY_TEACHER_LABELS_VERSION), index=False)
        smp = lab if sampled is None else sampled
        smp[["id", "tracked_slug"]].to_parquet(es.STANCE_OUTPUT, index=False)
    return write


def _opus(frame):
    """Opus says -1 in the critical bucket and +1 in the supportive one."""
    return frame.assign(score_teacher=np.where(frame["score_transformer"] < 0, -1.0, 1.0))


def test_reply_population_weights_buckets(reply_files):
    # One post: 90 replies in the critical bucket, 10 in the supportive one.
    # The model says 0 everywhere; Opus says -1 in critical, +1 in supportive.
    from src.analysis.event_study import bucket_draws
    reps = _replies({"p": [-0.9] * 90 + [0.9] * 10})
    reply_files(reps, _opus(bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")))
    r = inf.reply_population(n_boot=20).set_index("post").loc["p"]
    assert r["opus_mean"] == pytest.approx(-0.8)          # 0.9 * -1 + 0.1 * +1, not the 50/50 sample
    assert r["assisted_mean"] == pytest.approx(-0.8)
    assert r["assisted_anti"] == pytest.approx(0.9) and r["model_mean"] == 0.0
    assert r["coverage"] == 1.0


def test_reply_population_reads_manifest_and_draws_only_new_posts(reply_files, caplog):
    from src.analysis.event_study import bucket_draws
    reps = _replies({"p": [-0.9] * 60 + [0.9] * 60, "q": [-0.9] * 60 + [0.9] * 60})
    # p's recorded draws are two replies per bucket that a fresh bucket_draws
    # would not pick; labels exist for every reply, and disagree by id.
    ids = ["p-0", "p-1", "p-60", "p-61"]
    pd.DataFrame({"id": ids, "tracked_slug": "p", "bucket": [0, 0, 2, 2]}).to_parquet(
        settings.REPLY_DRAWS_MANIFEST, index=False)
    lab = _opus(reps)
    lab.loc[lab["id"].isin(ids), "score_teacher"] = 0.5   # only the recorded draws say +0.5
    reply_files(reps, lab)
    with caplog.at_level("INFO", logger="src.analysis.inference"):
        r = inf.reply_population(n_boot=20).set_index("post")
    assert r.loc["p", "opus_mean"] == pytest.approx(0.5) and r.loc["p", "n_labelled"] == 4
    man = pd.read_parquet(settings.REPLY_DRAWS_MANIFEST)
    assert man[man["tracked_slug"] == "p"]["id"].tolist() == ids        # p untouched
    # q is drawn as `stance` would draw it, and recorded
    expect = bucket_draws(reps[reps["tracked_slug"] == "q"], score_col="score_transformer")
    assert man[man["tracked_slug"] == "q"]["id"].tolist() == expect["id"].tolist()
    assert "recorded 100 draws for q (100)" in caplog.text
    inf.reply_population(n_boot=5)                                       # second run: no redraw
    assert pd.read_parquet(settings.REPLY_DRAWS_MANIFEST).equals(man)


def test_reply_population_uncovered_post_is_nan_not_zero_width(reply_files, caplog):
    # q has supportive replies but only critical draws were labelled: its
    # Opus estimate cannot be weighted back, so it is NaN, and so is ALL.
    from src.analysis.event_study import bucket_draws
    reps = _replies({"p": [-0.9] * 20 + [0.9] * 20, "q": [-0.9] * 20 + [0.9] * 20})
    drawn = bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")
    labelled = drawn[(drawn["tracked_slug"] == "p") | (drawn["score_transformer"] < 0)]
    reply_files(reps, _opus(labelled), sampled=drawn)
    with caplog.at_level("WARNING", logger="src.analysis.inference"):
        r = inf.reply_population(n_boot=20).set_index("post")
    assert r.loc["p", "coverage"] == 1.0
    assert np.isfinite(r.loc["p", ["opus_mean", "assisted_mean_lo"]].astype(float)).all()
    assert r.loc["q", "coverage"] == pytest.approx(0.5)
    assert r.loc["ALL", "coverage"] == pytest.approx(0.75)
    for post in ("q", "ALL"):
        est = r.loc[post, [c for c in r.columns if c.startswith(("opus_", "assisted_"))]]
        assert est.isna().all()                                   # NaN, never a [x, x] interval
    assert r.loc["q", "model_mean"] == 0.0                         # the census itself still stands
    assert "q/supportive" in caplog.text


def test_reply_population_reads_the_label_version_set(reply_files, monkeypatch):
    # v1 says -1 / +1 by bucket; the v2 file (its own name, extra columns)
    # says +0.5 everywhere. The setting picks the file (v2 since 2026-10-05).
    from src.analysis.event_study import bucket_draws
    from src.analysis.stance_local import reply_labels_path
    monkeypatch.setattr(settings, "REPLY_TEACHER_LABELS_VERSION", "v1")
    reps = _replies({"p": [-0.9] * 30 + [0.9] * 30})
    drawn = bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")
    reply_files(reps, _opus(drawn))
    drawn.assign(score_teacher=0.5, label_teacher="positive", prompt_version="v2",
                 input_chars=10)[["id", "tracked_slug", "score_teacher", "label_teacher",
                                  "prompt_version", "input_chars"]].to_parquet(
        reply_labels_path(version="v2"), index=False)
    v1 = inf.reply_population(n_boot=10).set_index("post").loc["p"]
    assert v1["opus_mean"] == pytest.approx(0.0) and v1["labels"] == "v1"
    monkeypatch.setattr(settings, "REPLY_TEACHER_LABELS_VERSION", "v2")
    v2 = inf.reply_population(n_boot=10).set_index("post").loc["p"]
    assert v2["opus_mean"] == pytest.approx(0.5) and v2["assisted_pro"] == pytest.approx(1.0)
    assert v2["labels"] == "v2" and v2["n_labelled"] == 10
    assert inf.reply_population(n_boot=10, labels_version="v1").set_index("post").loc["p", "opus_mean"] \
        == pytest.approx(0.0)
    with pytest.raises(ValueError, match="v3"):
        inf.reply_population(n_boot=10, labels_version="v3")


def test_reply_population_dedupes_and_drops_replies_without_text(reply_files):
    from src.analysis.event_study import bucket_draws
    reps = _replies({"p": [-0.9] * 10 + [0.9] * 10})
    reps.loc[[0, 1, 10], "text"] = ["", " ok ", None]             # image / GIF-only replies
    reps = pd.concat([reps, reps.iloc[[2, 12]]], ignore_index=True)  # two ids collected twice
    drawn = bucket_draws(reps, n_per_bucket=20, score_col="score_transformer")
    lab = _opus(drawn)
    lab.loc[~inf.has_text(lab["text"]), "score_teacher"] = 0.0    # labelled anyway, as 65 were
    reply_files(reps, lab)
    r = inf.reply_population(n_boot=20).set_index("post").loc["p"]
    assert r["n_replies"] == 17 and r["n_no_text"] == 3            # 20 unique ids, 3 without text
    assert r["n_labelled"] == 17                                   # unique ids with text only
    assert r["opus_mean"] == pytest.approx((8 * -1 + 9 * 1) / 17)  # the textless 0.0 labels are out


def test_reply_population_takes_a_crossfit_column_and_refuses_an_in_sample_one(reply_files):
    # A reply-domain column (NaN where a reply has no text) goes through the
    # same census and buckets as score_opus_distilled; one whose model was fit
    # on the labelled replies is refused.
    import json

    from src.analysis.event_study import bucket_draws
    from src.analysis.stance_local import reply_columns_path
    reps = _replies({"p": [-0.9] * 30 + [0.9] * 10})
    reps.loc[[0, 30], "text"] = ""
    reps["score_opus_distilled"] = np.where(reps["score_transformer"] < 0, -0.5, 0.25)
    reps["score_ctx_crossfit"] = reps["score_opus_distilled"].where(inf.has_text(reps["text"]))
    reps["score_ctx_distilled"] = reps["score_ctx_crossfit"]
    reply_files(reps, _opus(bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")))
    reply_columns_path().write_text(json.dumps({
        "score_ctx_crossfit": {"crossfit": True, "trained_on_reply_labels": "v2"},
        "score_ctx_distilled": {"crossfit": False, "trained_on_reply_labels": "v2",
                                "model_dir": "stance_distilled_final_reply_ctx"}}))
    old = inf.reply_population(n_boot=20).set_index("post").loc["p"]
    new = inf.reply_population(col="score_ctx_crossfit", n_boot=20).set_index("post").loc["p"]
    for k in ("n_replies", "n_labelled", "model_mean", "assisted_mean", "assisted_mean_lo",
              "assisted_pro", "coverage"):
        assert new[k] == pytest.approx(old[k])
    assert new["n_replies"] == 38 and new["model_mean"] == pytest.approx((29 * -0.5 + 9 * 0.25) / 38)
    with pytest.raises(ValueError, match="cross-fit"):
        inf.reply_population(col="score_ctx_distilled", n_boot=20)


def test_reply_population_defaults_to_the_record_column_and_refuses_unscored_replies(
        reply_files, monkeypatch):
    # settings.REPLY_POPULATION_COL is the default census; a reply with text
    # and no score in it refuses the run instead of leaving the post's other
    # replies to stand in for it.
    import json

    from src.analysis.event_study import bucket_draws
    from src.analysis.stance_local import reply_columns_path
    reps = _replies({"p": [-0.9] * 30 + [0.9] * 10})
    reps["score_opus_distilled"] = np.where(reps["score_transformer"] < 0, -0.5, 0.25)
    reps["score_ctx_crossfit"] = np.where(reps["score_transformer"] < 0, -0.4, 0.3)
    reply_files(reps, _opus(bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")))
    reply_columns_path().write_text(json.dumps({
        "score_ctx_crossfit": {"crossfit": True, "trained_on_reply_labels": "v2"}}))
    monkeypatch.setattr(settings, "REPLY_POPULATION_COL", "score_ctx_crossfit")
    default = inf.reply_population(n_boot=20).set_index("post").loc["p"]
    named = inf.reply_population(col="score_ctx_crossfit", n_boot=20).set_index("post").loc["p"]
    assert default["model_mean"] == pytest.approx(named["model_mean"])
    assert default["model_mean"] == pytest.approx((30 * -0.4 + 10 * 0.3) / 40)
    reps.loc[[3, 33], "score_ctx_crossfit"] = np.nan
    reply_files(reps, _opus(bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")))
    with pytest.raises(ValueError, match="2 replies with text have no score_ctx_crossfit"):
        inf.reply_population(n_boot=20)
    with pytest.raises(ValueError, match="score_nope is not in"):
        inf.reply_population(col="score_nope", n_boot=20)


# ── greg_*: the correction with an estimated slope ─────────────────

KS = ("mean", "pro", "anti", "neutral")
GREG_COL = "score_opus_distilled"


def _greg_stub(reply_files, model, opus, spec=None, n_per_bucket=40):
    """Population with a census column model(frame, rng) and labels
    opus(frame, rng) on n_per_bucket random draws a bucket; returns
    (replies, labelled draws)."""
    from src.analysis.event_study import bucket_draws

    reps = _replies(spec or {"p": [-0.9] * 160 + [0.9] * 120, "q": [-0.9] * 70 + [0.9] * 90})
    rng = np.random.default_rng(3)
    reps[GREG_COL] = model(reps, rng)
    reps["truth"] = opus(reps, rng)
    drawn = bucket_draws(reps, n_per_bucket=n_per_bucket, score_col="score_transformer")
    drawn = drawn.assign(bucket=inf.stance_bucket(drawn["score_transformer"]).astype(int))
    drawn[["id", "tracked_slug", "bucket"]].to_parquet(settings.REPLY_DRAWS_MANIFEST, index=False)
    lab = drawn.assign(score_teacher=drawn["truth"])
    reply_files(reps.drop(columns="truth"), lab)
    return reps, lab


def _noise(reps, rng):
    return rng.choice([-0.6, -0.2, 0.0, 0.3, 0.7], len(reps))


def _informative(reps, rng):
    return np.clip(0.5 * reps[GREG_COL] + rng.normal(0, 0.15, len(reps)), -1, 1).round(2)


def _wide(r, e, k):
    return r[f"{e}_{k}_hi"] - r[f"{e}_{k}_lo"]


def test_greg_is_the_slope_one_estimate_when_the_slope_is_one(reply_files):
    # Labels = column + a constant in the critical bucket, the same band;
    # in the supportive one the column is constant, so only its offset is
    # off. Slope 1 (and the identity for the shares) in every replicate:
    # greg = assisted, which is exact here.
    def model(reps, rng):
        v = rng.choice([-0.5, 0.0, 0.5], len(reps))
        return np.where(reps["score_transformer"] < 0, v, 0.5)

    def opus(reps, rng):
        f = reps[GREG_COL]
        return np.where(reps["score_transformer"] < 0, np.where(f != 0, f + 0.1, f), 0.0)

    def opus_mean(reps, rng):
        return np.where(reps["score_transformer"] < 0, reps[GREG_COL] + 0.1, 0.0)

    _greg_stub(reply_files, model, opus)
    rp = inf.reply_population(n_boot=40).set_index("post")
    for post in ("p", "q", "ALL"):
        for k in ("pro", "anti", "neutral"):
            assert rp.loc[post, f"greg_{k}"] == pytest.approx(rp.loc[post, f"assisted_{k}"])
            assert _wide(rp.loc[post], "greg", k) < 1e-6
    _greg_stub(reply_files, model, opus_mean)
    rp = inf.reply_population(n_boot=40).set_index("post")
    for post in ("p", "q", "ALL"):
        for s in ("", "_lo", "_hi"):
            assert rp.loc[post, f"greg_mean{s}"] == pytest.approx(rp.loc[post, f"assisted_mean{s}"])


def test_greg_with_the_slope_forced_to_one_is_assisted_replicate_by_replicate(
    reply_files, monkeypatch
):
    # Noisy labels and column, slopes replaced by 1 (the identity for pro and
    # anti, -1 -1 for neutral): point and interval are assisted's, so greg
    # runs on the same resamples and finite-population correction.
    from src.analysis import reply_crowd as rc

    _greg_stub(reply_files, _noise, _informative)
    free = inf.reply_population(n_boot=60).set_index("post")

    def ones(S, idx, aux):
        n = S["a"].shape[1]
        row = {"mean": [1.0], "mean_sq": [1.0], "pro": [1.0, 0.0], "anti": [0.0, 1.0]}
        row["neutral"] = [-1.0, -1.0]
        return {k: np.tile(row[k], (n, 1)) for k in aux}

    monkeypatch.setattr(rc, "_greg_slopes", ones)
    rp = inf.reply_population(n_boot=60).set_index("post")
    for post in ("p", "q", "ALL"):
        for k in KS:
            for s in ("", "_lo", "_hi"):
                assert rp.loc[post, f"greg_{k}{s}"] == pytest.approx(
                    rp.loc[post, f"assisted_{k}{s}"], abs=1e-6
                )
                assert rp.loc[post, f"assisted_{k}{s}"] == free.loc[post, f"assisted_{k}{s}"]
    assert abs(free.loc["p", "greg_mean"] - free.loc["p", "assisted_mean"]) > 1e-3


def test_greg_interval_collapses_when_the_column_is_the_labels(reply_files):
    _greg_stub(reply_files, _noise, lambda reps, rng: reps[GREG_COL])
    rp = inf.reply_population(n_boot=60).set_index("post")
    for post in ("p", "q", "ALL"):
        for k in KS:
            assert rp.loc[post, f"greg_{k}"] == pytest.approx(rp.loc[post, f"model_{k}"], abs=1e-6)
            assert _wide(rp.loc[post], "greg", k) < 1e-5
            assert _wide(rp.loc[post], "opus", k) > 1e-2
    # Every reply labelled (n = N in every bucket): width 0 with a noise column.
    _greg_stub(reply_files, _noise, _noise, spec={"p": [-0.9] * 30 + [0.9] * 30})
    r = inf.reply_population(n_boot=60).set_index("post").loc["p"]
    for k in KS:
        assert _wide(r, "greg", k) < 1e-6 and r[f"greg_{k}"] == pytest.approx(r[f"opus_{k}"])


def test_greg_with_a_useless_column_is_about_direct(reply_files):
    _greg_stub(reply_files, lambda reps, rng: 0.3, _noise)  # constant: exactly direct
    rp = inf.reply_population(n_boot=60).set_index("post")
    for post in ("p", "q", "ALL"):
        for k in KS:
            for s in ("", "_lo", "_hi"):
                assert rp.loc[post, f"greg_{k}{s}"] == pytest.approx(
                    rp.loc[post, f"opus_{k}{s}"], abs=1e-6
                )
    _greg_stub(  # noise: near direct, and narrower than the slope-1 correction
        reply_files, _noise, _noise, {"p": [-0.9] * 900 + [0.9] * 900}, n_per_bucket=300
    )
    r = inf.reply_population(n_boot=400).set_index("post").loc["p"]
    for k in KS:
        half = _wide(r, "opus", k) / 2
        assert abs(r[f"greg_{k}"] - r[f"opus_{k}"]) < 0.5 * half
        assert _wide(r, "greg", k) < 1.1 * _wide(r, "opus", k)
        assert _wide(r, "assisted", k) > 1.2 * _wide(r, "greg", k)


def test_greg_shares_sum_to_one_and_all_is_the_posts_weighted_by_size(reply_files):
    _greg_stub(reply_files, _noise, _informative)
    rp = inf.reply_population(n_boot=40).set_index("post")
    total = rp[["greg_pro", "greg_anti", "greg_neutral"]].sum(axis=1)
    assert np.allclose(total, 1, atol=1e-9)
    n = rp.loc[["p", "q"], "n_replies"]
    for k in KS:
        want = (rp.loc[["p", "q"], f"greg_{k}"] * n).sum() / n.sum()
        assert rp.loc["ALL", f"greg_{k}"] == pytest.approx(want, abs=1e-6)
        assert np.isfinite(rp.loc["ALL", f"greg_{k}_lo"])


def _brute_greg(reps, smp, seed, n_boot):
    """Post p's GREG from scratch: _weighted's resamples, weighted least
    squares with bucket intercepts (lstsq), the census means of the column."""
    rng = np.random.default_rng(seed)
    buckets = {b: np.flatnonzero(smp["bucket"] == b) for b in sorted(smp["bucket"].unique())}
    Nh = reps.groupby("bucket").size()
    w = np.zeros((len(smp), 1 + n_boot))
    for b, idx in buckets.items():
        n, N = len(idx), Nh[b]
        a = np.sqrt(1 - n / N)
        pick = rng.integers(0, n, (n_boot, n))
        c = np.stack([np.bincount(row, minlength=n) for row in pick], axis=1)
        w[idx, 0] = N / n
        w[idx, 1:] = N / n * (1 - a + a * c)
    s = smp["score_teacher"].to_numpy()
    fl, fp = smp[GREG_COL].to_numpy(), reps[GREG_COL].to_numpy()

    def bands(v):
        return np.column_stack([v > 0.05, v < -0.05]).astype(float)

    phi = {"mean": s, "pro": s > 0.05, "anti": s < -0.05, "neutral": abs(s) <= 0.05}
    out = {}
    for k, y in phi.items():
        y = y.astype(float)
        x, X = (fl[:, None], fp[:, None]) if k == "mean" else (bands(fl), bands(fp))
        D = np.column_stack([x] + [(smp["bucket"] == b).to_numpy(float) for b in buckets])
        est = []
        for r in range(1 + n_boot):
            v, root = w[:, r], np.sqrt(w[:, r])
            beta = np.linalg.lstsq(D * root[:, None], y * root, rcond=None)[0][: x.shape[1]]
            mu = 0.0
            for b, i in buckets.items():
                census = X[(reps["bucket"] == b).to_numpy()].mean(0)
                gap = census - np.average(x[i], axis=0, weights=v[i])
                mu += Nh[b] / len(reps) * (np.average(y[i], weights=v[i]) + beta @ gap)
            est.append(mu)
        out[k] = np.array(est)
    return out


def test_greg_matches_reply_crowd_domain_means_and_a_from_scratch_fit(reply_files):
    from src.analysis import reply_crowd as rc

    reps, lab = _greg_stub(
        reply_files, _noise, _informative, {"p": [-0.9] * 160 + [0.0] * 90 + [0.9] * 120}
    )
    r = inf.reply_population(n_boot=50, seed=11).set_index("post").loc["p"]
    reps = reps.assign(bucket=inf.stance_bucket(reps["score_transformer"]).astype(int))
    smp = reps.merge(lab[["id", "score_teacher"]], on="id")
    brute = _brute_greg(reps, smp, 11, 50)

    def frame(d):
        return pd.DataFrame(
            {
                "id": d["id"],
                "tracked_slug": "p",
                "bucket": d["bucket"],
                "f": d[GREG_COL],
                "m": 1,
                "replier": "u" + d["id"],
            }
        )

    fr = rc.CrowdFrames(
        frame(reps), frame(smp).assign(y=smp["score_teacher"]), reps.iloc[:0], GREG_COL, ["p"]
    )
    W = rc.replicate_weights(fr.lab, fr.pop, 50, 0, rng=np.random.default_rng(11))
    ds = rc.domain_means(fr, pd.Series("g", index=fr.pop["id"]), W, assist=True)
    assert not ds.collapsed and not ds.nan_cells
    for k in KS:
        for reps_k in (ds.mean("p", "g", k), brute[k]):
            for name, v in zip(("", "_lo", "_hi"), inf._ci(reps_k)):
                assert r[f"greg_{k}{name}"] == pytest.approx(v, abs=1e-6)
        assert abs(r[f"greg_{k}"] - r[f"assisted_{k}"]) > 1e-4


def _labelled_post(reply_files, reps, ids, buckets, truth):
    """The draws `ids` of one-post `reps`, labelled, as reply_population reads them."""
    pd.DataFrame({"id": ids, "tracked_slug": "p", "bucket": buckets}).to_parquet(
        settings.REPLY_DRAWS_MANIFEST, index=False
    )
    lab = reps.set_index("id").loc[ids].reset_index()
    reply_files(reps, lab.assign(score_teacher=truth(lab)))


def test_greg_leaves_a_bucket_under_the_minimum_out_of_the_slope(reply_files):
    # One draw short of the minimum in each bucket: no bucket enters the slope
    # fit, so greg is direct. At the minimum it enters.
    reps = _replies({"p": [-0.9] * 40 + [0.9] * 40})
    rng = np.random.default_rng(5)
    reps[GREG_COL] = rng.uniform(-1, 1, len(reps)).round(2)
    least = settings.REPLY_GREG_MIN_BUCKET_DRAWS
    for n in (least - 1, least):
        _labelled_post(
            reply_files,
            reps,
            [f"p-{i}" for i in (*range(n), *range(40, 40 + n))],
            [0] * n + [2] * n,
            lambda lab: (0.8 * lab[GREG_COL] + rng.normal(0, 0.1, len(lab))).round(2),
        )
        r = inf.reply_population(n_boot=30).set_index("post").loc["p"]
        gap = abs(r["greg_mean"] - r["opus_mean"])
        assert bool(gap < 1e-6) == (n < least)
        ends = [f"greg_{k}{s}" for k in KS for s in ("_lo", "_hi")]
        assert np.isfinite(r[ends].astype(float)).all()


def test_greg_does_not_interpolate_a_three_draw_bucket(reply_files):
    # Three draws covering the column's three bands: the shares' two slopes
    # and intercept pass through them exactly, in every replicate, and the
    # interval would have no width. The minimum keeps the bucket out: direct.
    reps = _replies({"p": [-0.9] * 400 + [0.9] * 400})
    rng = np.random.default_rng(8)
    reps[GREG_COL] = rng.choice([-0.6, 0.0, 0.6], len(reps), p=[0.5, 0.2, 0.3])
    f = [-0.6, 0.0, 0.6, 0.6, -0.6]
    ids = ["p-0", "p-1", "p-2", "p-400", "p-401"]
    reps.loc[reps["id"].isin(ids), GREG_COL] = f
    _labelled_post(reply_files, reps, ids, [0, 0, 0, 2, 2], lambda lab: [-0.5, 0.5, 0.0, 0.5, -0.5])
    r = inf.reply_population(n_boot=200).set_index("post").loc["p"]
    for k in KS:
        for s in ("", "_lo", "_hi"):
            assert r[f"greg_{k}{s}"] == pytest.approx(r[f"opus_{k}{s}"], abs=1e-6)
        assert _wide(r, "greg", k) > 0.1


def test_greg_leaves_the_published_columns_draws_alone(reply_files):
    # opus_* and assisted_* are _weighted's on one generator walked through the
    # posts and then ALL: greg, which runs between them, draws nothing from it.
    reps, lab = _greg_stub(reply_files, _noise, _informative)
    rp = inf.reply_population(n_boot=60, seed=11).set_index("post")
    reps = reps.assign(bucket=inf.stance_bucket(reps["score_transformer"]).astype(int))
    smp_all = reps.merge(lab[["id", "score_teacher"]], on="id")
    rng = np.random.default_rng(11)

    def phi(v):
        v = np.asarray(v, float)
        return dict(zip(KS, (v, 1.0 * (v > 0.05), 1.0 * (v < -0.05), 1.0 * (abs(v) <= 0.05))))

    for post in ("p", "q", "ALL"):
        pop, smp = (d if post == "ALL" else d[d["tracked_slug"] == post] for d in (reps, smp_all))
        y, f, census = phi(smp["score_teacher"]), phi(smp[GREG_COL]), phi(pop[GREG_COL])
        arrays = {f"opus_{k}": y[k] for k in KS}
        arrays |= {f"assisted_{k}": y[k] - f[k] for k in KS}
        _, reps_k, _ = inf._weighted(pop, smp, arrays, 60, rng)
        for name, v in reps_k.items():
            if name.startswith("assisted_"):
                v = v + census[name.removeprefix("assisted_")].mean()
            want = inf._ci(v)
            got = tuple(rp.loc[post, [name, f"{name}_lo", f"{name}_hi"]])
            assert got == pytest.approx(want, abs=1e-12), (post, name)


def test_greg_is_nan_where_the_coverage_is(reply_files):
    from src.analysis.event_study import bucket_draws

    reps = _replies({"p": [-0.9] * 20 + [0.9] * 20, "q": [-0.9] * 20 + [0.9] * 20})
    drawn = bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")
    labelled = drawn[(drawn["tracked_slug"] == "p") | (drawn["score_transformer"] < 0)]
    reply_files(reps, _opus(labelled), sampled=drawn)
    r = inf.reply_population(n_boot=20).set_index("post")
    greg = [c for c in r.columns if c.startswith("greg_")]
    assert len(greg) == 12
    assert r.loc["p", greg].notna().all() and r.loc[["q", "ALL"], greg].isna().all().all()
    assert list(r.columns[-12:]) == greg  # after every published column


def test_published_reply_estimator_is_assisted_until_the_owner_switches_it(monkeypatch):
    assert settings.REPLY_POPULATION_ESTIMATOR == "assisted"
    assert inf.published_reply_estimator() == "assisted"
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "greg")
    assert inf.published_reply_estimator() == "greg"
    monkeypatch.setattr(settings, "REPLY_POPULATION_ESTIMATOR", "opus")
    with pytest.raises(ValueError, match="REPLY_POPULATION_ESTIMATOR 'opus'"):
        inf.published_reply_estimator()


def _growing_post():
    """Post x before and after a later reply collection: 60 replies per
    bucket, then 60 more per bucket appended (same ids for the first 180)."""
    vals = ([-0.9] * 60 + [0.0] * 60 + [0.9] * 60) * 2
    return _replies({"x": vals[:180]}), _replies({"x": vals})


def test_reply_draws_recorded_only_once_stance_has_sampled(reply_files, caplog):
    # reply-population (or export-web) runs on a new post before `stance`,
    # then more replies arrive, then `stance` draws from the bigger frame and
    # its draws are labelled: the manifest must hold those draws, not the
    # small frame's.
    from src.analysis.event_study import bucket_draws
    small, big = _growing_post()
    reply_files(small, _opus(small.iloc[:0]), sampled=small.iloc[:0])
    with caplog.at_level("INFO", logger="src.analysis.inference"):
        r = inf.reply_population(n_boot=5).set_index("post")
    assert np.isnan(r.loc["x", "opus_mean"])
    assert not settings.REPLY_DRAWS_MANIFEST.exists()
    assert "not sampled by `stance` yet, not recorded: x (150)" in caplog.text

    drawn = bucket_draws(big, score_col="score_transformer")      # what `stance` draws now
    early = bucket_draws(small, score_col="score_transformer")
    assert set(drawn["id"]) != set(early["id"])
    reply_files(big, _opus(drawn))
    r = inf.reply_population(n_boot=5).set_index("post").loc["x"]
    assert sorted(pd.read_parquet(settings.REPLY_DRAWS_MANIFEST)["id"]) == sorted(drawn["id"])
    assert r["n_labelled"] == 150 and r["coverage"] == 1.0         # every paid label is used


def test_reply_draws_not_recorded_when_stance_drew_another_frame(reply_files, caplog):
    # `stance` drew and labelled x before the frame grew, and the manifest
    # never recorded it: today's draws only partly meet the labels. Say so,
    # and keep them out of the manifest.
    from src.analysis.event_study import bucket_draws
    small, big = _growing_post()
    reply_files(big, _opus(bucket_draws(small, score_col="score_transformer")))
    with caplog.at_level("WARNING", logger="src.analysis.inference"):
        r = inf.reply_population(n_boot=5).set_index("post").loc["x"]
    assert not settings.REPLY_DRAWS_MANIFEST.exists()
    assert "so not recorded" in caplog.text and "x (" in caplog.text
    assert 0 < r["n_labelled"] < 150


def test_reply_draws_warn_on_recorded_draws_stance_never_drew(reply_files, caplog):
    reps = _replies({"p": [-0.9] * 10 + [0.9] * 10})
    pd.DataFrame({"id": ["p-0", "p-10", "p-99"], "tracked_slug": "p", "bucket": [0, 2, 0]}).to_parquet(
        settings.REPLY_DRAWS_MANIFEST, index=False)
    reply_files(reps, _opus(reps))
    with caplog.at_level("WARNING", logger="src.analysis.inference"):
        inf.reply_population(n_boot=5)
    assert "1 recorded draws are not in stance_sample.parquet and carry no label: p (1)" in caplog.text


def test_reply_draws_follow_the_stance_sample_size(reply_files):
    # `stance --n 20` draws 20 per bucket: the manifest must record those
    # draws, not the default 50.
    from src.analysis.event_study import bucket_draws
    reps = _replies({"p": [-0.9] * 60 + [0.9] * 60})
    drawn = bucket_draws(reps, n_per_bucket=20, score_col="score_transformer")
    reply_files(reps, _opus(drawn))
    d = inf.reply_draws(reps, n_per_bucket=20)
    assert sorted(d["id"]) == sorted(drawn["id"])
    assert sorted(pd.read_parquet(settings.REPLY_DRAWS_MANIFEST)["id"]) == sorted(drawn["id"])


def test_stance_records_no_draws_when_scoring_fails(reply_files, monkeypatch):
    # A `stance` run that dies before saving its sample (no API key) must not
    # freeze draws that were never labelled.
    from click.testing import CliRunner

    import src.analysis.event_study as es
    from src.cli import main

    reps = _replies({"p": [-0.9] * 60 + [0.9] * 60}).assign(user="u")
    reply_files(reps, _opus(reps.iloc[:0]), sampled=reps.iloc[:0])
    es.STANCE_OUTPUT.unlink()
    monkeypatch.setattr(es, "load_or_score_replies", lambda: reps)

    def no_key(*a, **k):
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env for stance scoring")

    monkeypatch.setattr(es, "score_stance", no_key)
    res = CliRunner().invoke(main, ["stance"])
    assert isinstance(res.exception, RuntimeError)
    assert not settings.REPLY_DRAWS_MANIFEST.exists()

def _result(custom_id, text, kind="succeeded"):
    msg = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])
    return types.SimpleNamespace(custom_id=custom_id, result=types.SimpleNamespace(type=kind, message=msg))


def test_topic_parse_result():
    assert tl.parse_result(_result("1", '{"about_war": true}'))["about_war"] is True
    assert tl.parse_result(_result("2", '```json\n{"about_war": false}\n```'))["about_war"] is False
    assert tl.parse_result(_result("3", "yes"))["about_war"] is None
    assert tl.parse_result(_result("4", "", kind="errored"))["about_war"] is None


def test_backup_commands_never_delete(tmp_path, monkeypatch):
    from src import backup
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(settings, "BACKUP_SYNC", [(tmp_path / "raw", "raw/", "STANDARD"),
                                                  (tmp_path / "missing", "models/", "GLACIER_IR")])
    cmds = backup.sync_commands("s3://bucket/", dry_run=True)
    assert len(cmds) == 1 and cmds[0][4] == "s3://bucket/raw/"
    assert "--delete" not in cmds[0] and "--dryrun" in cmds[0]
    assert backup.sync_commands("bucket", dry_run=True)[0][4] == "s3://bucket/raw/"
