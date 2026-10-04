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
    labels = tmp_path / "teacher_labels_replies_claude-opus-5.parquet"

    def write(replies: pd.DataFrame, lab: pd.DataFrame, sampled: pd.DataFrame | None = None):
        """sampled: what `stance` drew into stance_sample (default: the labelled rows)."""
        replies.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
        lab[["id", "score_teacher"]].to_parquet(labels, index=False)
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
