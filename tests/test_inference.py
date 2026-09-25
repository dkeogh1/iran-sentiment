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


def test_reply_population_weights_buckets(tmp_path, monkeypatch):
    # One post: 90 replies in the critical bucket, 10 in the supportive one.
    # The model says 0 everywhere; Opus says -1 in critical, +1 in supportive.
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    n = [90, 10]
    reps = pd.DataFrame({"id": [str(i) for i in range(100)], "tracked_slug": "p",
                         "score_transformer": [-0.9] * n[0] + [0.9] * n[1],
                         "score_opus_distilled": 0.0})
    reps.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
    from src.analysis.event_study import bucket_draws
    drawn = bucket_draws(reps, n_per_bucket=5, score_col="score_transformer")
    lab = drawn.assign(score_teacher=np.where(drawn["score_transformer"] < 0, -1.0, 1.0))
    lab[["id", "score_teacher"]].to_parquet(tmp_path / "teacher_labels_replies_claude-opus-5.parquet", index=False)
    r = inf.reply_population(n_boot=20).set_index("post").loc["p"]
    assert r["opus_mean"] == pytest.approx(-0.8)          # 0.9 * -1 + 0.1 * +1, not the 50/50 sample
    assert r["assisted_mean"] == pytest.approx(-0.8)
    assert r["assisted_anti"] == pytest.approx(0.9) and r["model_mean"] == 0.0


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
