"""
The long-post backfill (x-backfill-text) against a stub tweepy client: the
candidate rule, the free estimate, the cap and budget gates, journal resume,
the raw-cache rewrite, and the invalidation that makes `analyze`, `relabel`
and `topic-label` redo exactly the changed posts. No network, no spend.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import tweepy
from click.testing import CliRunner

from config import settings
from src.analysis import inference as inf
from src.analysis import relabel as rl
from src.analysis import sentiment as sen
from src.analysis import topic_label as tl
from src.collectors import x_backfill as xb

LINK = "https://t.co/AbCdEfGhIj"  # 23 characters, as X sends every link


def _cut(n: int, end: str = "word") -> str:
    """A post body of exactly n characters ending in `end`, then a link."""
    body = ("x" * (n - len(end) - 1)) + " " + end
    assert len(body) == n
    return f"{body} {LINK}"


# ── candidate rule ─────────────────────────────────────────────────


def test_cut_length_ignores_reply_mentions_trailing_links_and_entities():
    assert xb.cut_length(_cut(278)) == 278
    assert xb.cut_length("@a @b_c " + _cut(278)) == 278  # a reply's mentions don't count
    assert xb.cut_length("Tom &amp; Jerry " + LINK + " " + LINK) == len("Tom & Jerry")
    inner = "see https://example.com/a/very/long/path/that/is/not/23/chars ok"
    assert xb.cut_length(inner) == len("see ") + 23 + len(" ok")  # any other link counts 23


def test_likely_cut_window_and_endings(monkeypatch):
    monkeypatch.setattr(settings, "X_BACKFILL_MIN_CHARS", 270)
    monkeypatch.setattr(settings, "X_BACKFILL_OPEN_MIN_CHARS", 266)
    assert xb.likely_cut(_cut(280)) and xb.likely_cut(_cut(270))
    assert xb.likely_cut(_cut(275, "end.")) is True  # >= 270: any ending
    assert xb.likely_cut(_cut(267)) is True  # 266-269: mid-sentence only
    assert xb.likely_cut(_cut(267, "end.")) is False
    assert xb.likely_cut(_cut(265)) is False
    assert xb.likely_cut("x" * 281) is False  # already whole (over 280)
    assert xb.likely_cut("RT @someone: " + "x" * 270) is False
    assert xb.likely_cut(None) is False


# ── fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A data tree in tmp_path with three account files and a search file."""
    raw, proc = tmp_path / "raw" / "x", tmp_path / "processed"
    raw.mkdir(parents=True)
    proc.mkdir()
    for name, val in [
        ("DATA_DIR", tmp_path),
        ("RAW_DIR", tmp_path / "raw"),
        ("X_RAW_DIR", raw),
        ("TRUTH_SOCIAL_RAW_DIR", tmp_path / "raw" / "truthsocial"),
        ("PROCESSED_DIR", proc),
        ("FIGURES_DIR", proc / "figures"),
        ("SENTIMENT_OUTPUT", proc / "sentiment_all.parquet"),
        ("TRUMP_FEED_STANCE", proc / "no_trump_feed.parquet"),
        ("VADER_CHECKPOINT", proc / "checkpoint_vader.parquet"),
        ("ROBERTA_CHECKPOINT", proc / "checkpoint_roberta.parquet"),
    ]:
        monkeypatch.setattr(settings, name, val)
    return tmp_path


def _rec(pid, user, text, created="2026-06-01T12:00:00+00:00", tier="maga_prowar", **kw):
    return {
        "id": str(pid),
        "user": user,
        "tier": tier,
        "text": text,
        "created_at": created,
        "metrics": {"like_count": 3},
        "lang": "en",
        "platform": "x",
        **kw,
    }


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def _cache(data):
    """ids 1-6 are cut candidates, 7 is short, 8 a retweet, 9 posted after the
    cut stopped, 10 a search row; 6 is also in a search file."""
    raw = data / "raw" / "x"
    _write(
        raw / "levin.jsonl",
        [
            _rec(1, "levin", _cut(279)),
            _rec(2, "levin", _cut(276)),
            _rec(3, "levin", _cut(280, "end.")),
            _rec(7, "levin", "short post"),
            _rec(8, "levin", "RT @x: " + "y" * 273),
            _rec(9, "levin", _cut(279), created="2026-09-20T00:00:00+00:00"),
        ],
    )
    _write(raw / "Pontifex.jsonl", [_rec(4, "Pontifex", _cut(278), tier="religious_authority")])
    _write(
        raw / "StateDept.jsonl",
        [
            _rec(5, "StateDept", _cut(271), tier="admin"),
            _rec(6, "StateDept", _cut(274), tier="admin"),
        ],
    )
    _write(
        raw / "search_Iran_war.jsonl",
        [
            _rec(10, "search:Iran war", _cut(279), tier="search", author_id="1"),
            _rec(6, "search:Iran war", _cut(274), tier="search", author_id="2"),
        ],
    )
    return raw


FULL = {  # what X returns as note_tweet; 3 turns out not to be long, 5 is gone
    "1": "the whole of post one, " + "a" * 300,
    "2": "the whole of post two, " + "b" * 300,
    "4": "the whole of post four, " + "c" * 300,
    "6": "the whole of post six, " + "d" * 300,
}


class LookupStub:
    """GET /2/tweets: returns each asked id, note_tweet only for long posts,
    an error entry for posts X no longer has. `fail_on` raises on that call."""

    def __init__(self, full=FULL, missing=("5",), fail_on=None):
        self.full, self.missing, self.fail_on = full, set(missing), fail_on
        self.calls: list[list[str]] = []

    def get_tweets(self, ids, tweet_fields):
        self.calls.append(list(ids))
        if self.fail_on == len(self.calls):
            raise tweepy.errors.TweepyException("402 Payment Required")
        assert "note_tweet" in tweet_fields
        data, errors = [], []
        for i in ids:
            if i in self.missing:
                errors.append({"resource_id": i, "title": "Not Found Error"})
                continue
            d = {"id": i, "text": "cut"}
            if i in self.full:
                d["note_tweet"] = {"text": self.full[i]}
            data.append(SimpleNamespace(id=int(i), text="cut", data=d))
        return SimpleNamespace(data=data, errors=errors)

    def asked(self):
        return [i for c in self.calls for i in c]


def test_candidates_are_cut_account_originals_only(data):
    _cache(data)
    assert [c["id"] for c in xb.candidates()] == ["1", "2", "3", "4", "5", "6"]


# ── estimate, cap, budget ──────────────────────────────────────────


def _no_client(*a, **k):
    raise AssertionError("no client may be built on this path")


def test_estimate_makes_no_calls(data, monkeypatch):
    from src.cli import main
    from src.collectors import x_collector as xc

    _cache(data)
    monkeypatch.setattr(xc, "get_client", _no_client)
    res = CliRunner().invoke(main, ["x-backfill-text", "--estimate"])
    assert res.exit_code == 0, res.output
    assert "6 cached originals likely cut" in res.output
    assert "This run: 6 reads ≈ $0.03" in res.output
    assert "  @levin" in res.output and "religious_authority" in res.output
    assert not xb.journal_path().exists()


def test_cap_and_budget_refuse_before_any_call(data, monkeypatch):
    from src.cli import main
    from src.collectors import x_collector as xc

    _cache(data)
    monkeypatch.setattr(xc, "get_client", _no_client)
    monkeypatch.setattr(settings, "X_BACKFILL_MAX_READS", 5)
    res = CliRunner().invoke(main, ["x-backfill-text", "--yes"])
    assert res.exit_code == 1 and "over X_BACKFILL_MAX_READS" in res.output
    monkeypatch.setattr(settings, "X_BACKFILL_MAX_READS", 100)
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 0.02)
    res = CliRunner().invoke(main, ["x-backfill-text", "--yes"])
    assert res.exit_code == 1 and "exceeds X_RUN_BUDGET_USD" in res.output
    assert not xb.journal_path().exists()


def test_cli_pilot_then_full_run(data, monkeypatch):
    from src.cli import main
    from src.collectors import x_collector as xc

    _cache(data)
    stub = LookupStub()
    monkeypatch.setattr(xc, "get_client", lambda: stub)
    monkeypatch.setattr(settings, "X_LOOKUP_BATCH", 2)
    res = CliRunner().invoke(main, ["x-backfill-text", "--limit", "2"], input="n\n")
    assert "Aborted." in res.output and stub.calls == []
    res = CliRunner().invoke(main, ["x-backfill-text", "--limit", "2", "--yes"])
    assert res.exit_code == 0, res.output
    assert stub.asked() == ["1", "2"] and "Applied 2 new full texts" in res.output
    res = CliRunner().invoke(main, ["x-backfill-text", "--yes"])
    assert res.exit_code == 0, res.output
    assert stub.asked() == ["1", "2", "3", "4", "5", "6"]  # nothing read twice
    assert "Applied 2 new full texts (2 were already in)" in res.output
    res = CliRunner().invoke(main, ["x-backfill-text", "--yes"])
    assert res.exit_code == 0 and "This run: 0 reads" in res.output
    assert len(stub.calls) == 3  # 1 pilot request + 2 for the rest; none on the rerun


# ── journal resume ─────────────────────────────────────────────────


def test_killed_lookup_resumes_without_rereading(data):
    _cache(data)
    stub = LookupStub(fail_on=2)
    with pytest.raises(tweepy.errors.TweepyException):
        xb.lookup(stub, xb.plan()["todo"], batch_size=2)
    j = xb.load_journal()
    assert sorted(j) == ["1", "2"]  # batch 1 journalled before batch 2 was asked for
    assert j["1"]["full_text"] == FULL["1"] and j["1"]["cached_text"] == _cut(279)
    assert not list((data / "processed").glob("*.tmp"))
    p = xb.plan()
    assert p["checked"] == 2 and [c["id"] for c in p["todo"]] == ["3", "4", "5", "6"]
    stub.fail_on = None
    s = xb.lookup(stub, p["todo"], batch_size=2)
    assert s == {"requests": 2, "returned": 3, "long": 2, "missing": 1}
    assert stub.asked() == [
        "1",
        "2",
        "3",
        "4",
        "3",
        "4",
        "5",
        "6",
    ]  # the failed call billed nothing
    j = xb.load_journal()
    assert j["3"]["status"] == "found" and j["3"]["full_text"] is None  # not long after all
    assert j["5"] == {**j["5"], "status": "missing", "error": "Not Found Error"}
    assert xb.plan()["reads"] == 0


# ── apply: raw rewrite and invalidation ────────────────────────────


def _processed(data, ids=tuple(str(i) for i in range(1, 11))):
    """sentiment_all from the cache (scores on every post), Opus labels on
    1-10 and topic labels on 1-10."""
    from src.collectors.x_collector import load_all_cached

    posts = pd.DataFrame(load_all_cached())
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    for c in ("vader", "transformer", "llm", "opus"):
        posts[f"score_{c}"] = 0.4
        posts[f"label_{c}"] = "positive"
    posts.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    pd.DataFrame({"id": list(ids), "score_teacher": 0.4, "label_teacher": "positive"}).to_parquet(
        rl.labels_path(settings.TEACHER_CHECK_MODEL), index=False
    )
    pd.DataFrame({"id": list(ids), "about_war": True}).to_parquet(tl.labels_path(), index=False)
    return posts


def _run_all(data):
    xb.lookup(LookupStub(), xb.plan()["todo"])
    return xb.apply()


def test_apply_rewrites_text_atomically_and_keeps_other_fields(data):
    raw = _cache(data)
    before = {p.name: _read(p) for p in raw.glob("*.jsonl")}
    a = _run_all(data)
    assert a["changed"] == 4 and a["long"] == 4 and a["missing"] == 1
    assert a["files_rewritten"] == 4  # 6 is in StateDept's file and a search file
    after = {p.name: _read(p) for p in raw.glob("*.jsonl")}
    for name, recs in before.items():
        assert len(after[name]) == len(recs)
        for old, new in zip(recs, after[name]):
            assert new == ({**old, "text": FULL[old["id"]]} if old["id"] in FULL else old)
    assert not list(raw.glob("*.tmp"))
    assert xb.apply()["changed"] == 0  # idempotent


def test_apply_moves_cut_labels_and_relabel_topic_redo_exactly_them(data):
    _cache(data)
    before = _processed(data)
    a = _run_all(data)
    changed = {"1", "2", "4", "6"}
    model = settings.TEACHER_CHECK_MODEL
    assert a["labels_moved"] == {rl.labels_path(model).name: 4, tl.labels_path().name: 4}
    assert a["sentiment_rows_cleared"] == 4

    arch = pd.read_parquet(xb.superseded_path(rl.labels_path(model)))
    assert set(arch["id"]) == changed and (arch["superseded_reason"] == xb.SUPERSEDED_REASON).all()
    assert set(pd.read_parquet(xb.superseded_path(tl.labels_path()))["id"]) == changed
    assert changed.isdisjoint(pd.read_parquet(rl.labels_path(model))["id"])

    n, usd = xb.relabel_cost()
    assert n == 4 and usd > rl.estimate_cost(4)  # priced at the full length

    # relabel / topic-label see exactly the changed posts, with their full text
    df = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    todo = rl.posts_to_label(df, model)
    assert set(todo["id"]) == changed
    assert dict(zip(todo["id"], todo["text"])) == {i: FULL[i] for i in changed}
    assert set(tl.posts_to_label()["id"]) == changed

    # collect + merge fill them; unchanged posts keep their labels and phases
    new = pd.DataFrame({"id": sorted(changed), "score_teacher": -0.5, "label_teacher": "negative"})
    lab = pd.concat([pd.read_parquet(rl.labels_path(model)), new], ignore_index=True)
    lab.to_parquet(rl.labels_path(model), index=False)
    merged = rl.merge(df, model)
    m = merged.set_index("id")["score_opus"]
    assert (m[sorted(changed)] == -0.5).all()
    same = before[~before["id"].isin(changed)].set_index("id")
    assert (m[same.index] == same["score_opus"]).all()
    pd.DataFrame({"id": sorted(changed), "about_war": False}).pipe(
        lambda t: pd.concat([pd.read_parquet(tl.labels_path()), t])
    ).to_parquet(tl.labels_path(), index=False)
    d0 = inf.prepare(before, "score_opus", topic_source="llm").set_index("id")
    d1 = inf.prepare(merged, "score_opus", topic_source="llm").set_index("id")
    keep = d0.index.difference(sorted(changed))
    assert list(keep) == ["3", "5", "7", "8"]  # 9 is after the phases, 10 a search row
    pd.testing.assert_frame_equal(
        d0.loc[keep, ["score_opus", "on_topic"]], d1.loc[keep, ["score_opus", "on_topic"]]
    )

    assert xb.relabel_cost() == (0, 0.0)  # nothing owed once relabelled

    # a second apply moves nothing: the new labels stay
    assert xb.apply()["changed"] == 0
    assert len(pd.read_parquet(xb.superseded_path(rl.labels_path(model)))) == 4


def test_apply_killed_between_steps_finishes_on_rerun(data, monkeypatch):
    raw = _cache(data)
    _processed(data)
    xb.lookup(LookupStub(), xb.plan()["todo"])
    real = xb._write_jsonl_atomic

    def boom(path, recs):
        if path.parent == raw:
            raise OSError("killed")
        real(path, recs)

    monkeypatch.setattr(xb, "_write_jsonl_atomic", boom)
    with pytest.raises(OSError):
        xb.apply()
    assert _read(raw / "levin.jsonl")[0]["text"] == _cut(279)  # labels moved, raw not yet
    monkeypatch.setattr(xb, "_write_jsonl_atomic", real)
    a = xb.apply()
    assert a["changed"] == 4 and set(a["labels_moved"].values()) == {0}
    assert _read(raw / "levin.jsonl")[0]["text"] == FULL["1"]
    arch = pd.read_parquet(xb.superseded_path(rl.labels_path(settings.TEACHER_CHECK_MODEL)))
    assert len(arch) == 4  # archived once


def test_apply_leaves_a_post_that_changed_since_it_was_checked(data):
    raw = _cache(data)
    xb.lookup(LookupStub(), xb.plan()["todo"])
    recs = _read(raw / "Pontifex.jsonl")
    recs[0]["text"] = "edited elsewhere"
    _write(raw / "Pontifex.jsonl", recs)
    a = xb.apply()
    assert a["drifted"] == ["4"] and _read(raw / "Pontifex.jsonl")[0]["text"] == "edited elsewhere"


# ── analyze restores scores only onto the same text ────────────────


def test_restore_skips_posts_whose_text_changed(data):
    prior = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["the cut text", "same text"],
            "score_vader": [0.5, 0.2],
            "label_vader": ["positive", "positive"],
            "score_transformer": [0.1, 0.3],
            "label_transformer": ["neutral", "positive"],
            "score_llm": [0.9, 0.8],
            "label_llm": ["positive", "positive"],
        }
    )
    prior.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    prior.to_parquet(settings.ROBERTA_CHECKPOINT, index=False)
    posts = [{"id": "1", "text": "the whole text"}, {"id": "2", "text": "same text"}]
    assert sen._restore_prior_scores(posts) == (1, 1)
    assert "score_vader" not in posts[0] and posts[0].get("score_llm") is None
    assert posts[1]["score_vader"] == 0.2 and posts[1]["score_llm"] == 0.8
    posts = [{"id": "1", "text": "the whole text"}, {"id": "2", "text": "same text"}]
    assert sen._restore_roberta_checkpoint(posts) == 1 and "score_transformer" not in posts[0]


def test_teacher_retest_pairs_the_direct_label_with_the_cut_text_label(data):
    model = settings.TEACHER_CHECK_MODEL
    pd.DataFrame({"id": ["1", "2"], "tier": "admin", "text": "t"}).to_parquet(
        settings.SENTIMENT_OUTPUT, index=False
    )
    pd.DataFrame({"id": ["1", "2"], "score_teacher": [0.5, 0.2]}).to_parquet(
        data / "processed" / f"teacher_check_{model}.parquet", index=False
    )
    pd.DataFrame({"id": ["1", "2"], "score_teacher": [0.5, 0.2], "label_teacher": "x"}).to_parquet(
        rl.labels_path(model), index=False
    )
    xb.supersede_labels({"1"})
    lab = pd.read_parquet(rl.labels_path(model))
    pd.concat(
        [lab, pd.DataFrame({"id": ["1"], "score_teacher": [-0.9], "label_teacher": "y"})]
    ).to_parquet(rl.labels_path(model), index=False)
    _, s = inf.teacher_retest(model)
    assert s["n"] == 2 and s["identical"] == 1.0  # the full-text relabel is not compared
