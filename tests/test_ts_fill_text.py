"""`ts-fill-text`: re-reading cached Truth Social posts with no text by id
(stubbed HTTP; no request leaves the process)."""

import json

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings
from src.analysis import stance_local as sl
from src.analysis import topic_label as tl
from src.collectors import truthsocial_collector as ts
from src.collectors import ts_fill_text as tf

H = "realDonaldTrump"


@pytest.fixture
def data(tmp_path, monkeypatch):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    (raw / "truthsocial").mkdir(parents=True)
    proc.mkdir()
    for name, val in [
        ("DATA_DIR", tmp_path),
        ("RAW_DIR", raw),
        ("X_RAW_DIR", raw / "x"),
        ("TRUTH_SOCIAL_RAW_DIR", raw / "truthsocial"),
        ("PROCESSED_DIR", proc),
        ("FIGURES_DIR", proc / "figures"),
        ("MODELS_DIR", tmp_path / "models"),
        ("SENTIMENT_OUTPUT", proc / "sentiment_all.parquet"),
        ("TRUMP_FEED_STANCE", proc / "truthsocial_trump_stance.parquet"),
        ("TS_PAGE_DELAY_S", 0),
        ("TS_MAX_RETRIES", 2),
        ("TS_FILL_MAX_CONSECUTIVE_ERRORS", 3),
    ]:
        monkeypatch.setattr(settings, name, val)
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    # never the user's account or token on this path
    for fn in ("_get_token", "_ts_get_auth_paced", "_get_truthbrush_api", "_auth_post"):
        monkeypatch.setattr(ts, fn, _forbidden)
    _write_cache()
    return tmp_path


def _forbidden(*a, **k):
    raise AssertionError("ts-fill-text must stay anonymous")


def _rec(pid, text):
    return {
        "id": pid,
        "user": H,
        "text": text,
        "created_at": f"2026-05-{int(pid) - 90:02d}T12:00:00.000Z",
        "metrics": {"reblogs": 1, "favourites": 2, "replies": 3},
        "platform": "truthsocial",
        "tier": "admin",
    }


# 101 image only, 102 quote, 103 ReTruth stored empty, 104 deleted since,
# 105 has text (not read), 106 a link card, 107 quote of a post with no text
CACHED = [
    _rec("101", ""),
    _rec("102", ""),
    _rec("103", ""),
    _rec("104", ""),
    _rec("105", "Strikes on Iran tonight"),
    _rec("106", ""),
    _rec("107", ""),
]
NO_TEXT = ["101", "102", "103", "104", "106", "107"]
QUOTE_TEXT = "RT @SecWar: Iran will never have a nuclear weapon"
RT_TEXT = "RT @WhiteHouse: Peace through strength"


def _write_cache(recs=CACHED):
    path = tf.cache_path(H)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def _status(pid):
    base = {"id": pid, "content": "<p></p>", "media_attachments": [], "reblog": None}
    if pid == "101":
        return {**base, "media_attachments": [{"type": "image"}]}
    if pid in ("102", "107"):
        inner = "<p>Iran will never have a nuclear weapon</p>" if pid == "102" else "<p>!!</p>"
        quoted = {"id": "777", "content": inner, "account": {"acct": "SecWar"}}
        return {**base, "quote_id": "777", "quote": quoted, "media_attachments": [{}]}
    if pid == "103":
        reblog = {
            "id": "555",
            "content": "<p>Peace through strength</p>",
            "account": {"acct": "WhiteHouse"},
            "media_attachments": [],
        }
        return {**base, "content": "", "reblog": reblog}
    return {**base, "card": {"url": "https://example.com"}}


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.headers, self.text = status, body, {}, ""

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class Server:
    """GET /api/v1/statuses/<id>: 404 for 104; `script` maps an id to a list
    of responses (or exceptions) served first, one per request."""

    def __init__(self, script=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.asked: list[str] = []

    def __call__(self, url, params=None, timeout=None):
        assert url.startswith(f"{ts.TS_API_BASE}/statuses/") and params is None
        pid = url.rsplit("/", 1)[-1]
        self.asked.append(pid)
        if self.script.get(pid):
            r = self.script[pid].pop(0)
            if isinstance(r, BaseException):
                raise r
            return r
        if pid == "104":
            return _Resp(404, {"error": "Record not found"})
        return _Resp(200, _status(pid))


def _run(*args, input=None):
    from src.cli import main

    return CliRunner().invoke(main, ["ts-fill-text", *args], input=input)


def _cache():
    return {r["id"]: r for r in ts._load_jsonl(tf.cache_path(H))}


# ── classification and the plan ────────────────────────────────────


def test_classify():
    assert [tf.classify(_status(p)) for p in ("101", "102", "103", "106", "107")] == [
        "media_only",
        "quote",
        "retruth",
        "other",
        "quote",
    ]
    # a post that gained words of its own since it was cached
    assert tf.classify({"content": "<p>Big win!</p>", "media_attachments": [{}]}) == "other"


def test_estimate_makes_no_request(data, monkeypatch):
    monkeypatch.setattr(ts, "_ts_get", _forbidden)
    res = _run("--estimate")
    assert res.exit_code == 0, res.output
    assert (
        "7 cached posts, 6 with no words of their own (0 of them only a quote fallback "
        '"RT: <link>"), 0 of those already read'
    ) in res.output
    assert "This run: 6 reads" in res.output
    assert not tf.journal_path().exists()


def test_pilot_spreads_over_the_window(data):
    assert [c["id"] for c in tf.plan(H, limit=2)["todo"]] == ["101", "104"]
    assert [c["id"] for c in tf.plan(H, limit=3)["todo"]] == ["101", "103", "106"]
    assert [c["id"] for c in tf.plan(H)["todo"]] == NO_TEXT


# ── a run, end to end ──────────────────────────────────────────────


def test_pilot_then_full_run_reads_each_post_once(data, monkeypatch):
    server = Server()
    monkeypatch.setattr(ts, "_ts_get", server)
    res = _run("--limit", "2")
    assert res.exit_code == 0, res.output
    assert server.asked == ["101", "104"]
    assert "media_only 1, quote 0, retruth 0, other 0, gone 1, error 0" in res.output

    res = _run()
    assert res.exit_code == 0, res.output
    assert "6 with no words of their own (0 of them" in res.output
    assert "2 of those already read" in res.output
    assert sorted(server.asked) == NO_TEXT  # nothing read twice
    assert "Journal for @realDonaldTrump: media_only 1, quote 2, retruth 1, other 1, gone 1" in (
        res.output
    )
    assert (
        "Applied: 3 records rewritten, 2 of them gained text and 0 lost a bare quote fallback "
        "(2 were already in"
    ) in res.output
    assert "Quote posts read: 2 (1 carry the quoted text); 0 quote a post in this cache" in (
        res.output
    )
    assert "Next: score-posts" in res.output

    c = _cache()
    assert c["102"] == {**_rec("102", QUOTE_TEXT), "quote_of": "777", "media": 1}
    assert c["103"] == {**_rec("103", RT_TEXT), "reblog_of": "555"}
    assert c["101"] == {**_rec("101", ""), "media": 1}
    assert c["107"] == {**_rec("107", ""), "quote_of": "777", "media": 1}
    assert c["104"] == _rec("104", "") and c["106"] == _rec("106", "")
    assert c["105"] == _rec("105", "Strikes on Iran tonight")
    assert list(c) == [r["id"] for r in CACHED]  # record order kept

    res = _run()
    assert res.exit_code == 0 and "This run: 0 reads" in res.output
    assert "Applied: 0 records rewritten" in res.output and "5 were already in" in res.output
    assert len(server.asked) == 6


def test_killed_run_journals_first_and_resumes(data, monkeypatch):
    server = Server(script={"103": [KeyboardInterrupt()]})
    monkeypatch.setattr(ts, "_ts_get", server)
    before = tf.cache_path(H).read_text()
    with pytest.raises(KeyboardInterrupt):
        tf.fetch(H, tf.plan(H)["todo"])
    assert tf.cache_path(H).read_text() == before  # the cache waits for apply
    assert sorted(tf.load_journal()) == ["101", "102"]
    # a write killed mid-line: the torn line is skipped and its read redone
    with open(tf.journal_path(), "a") as f:
        f.write('{"id": "103", "kin')
    assert [c["id"] for c in tf.plan(H)["todo"]] == ["103", "104", "106", "107"]
    res = _run()
    assert res.exit_code == 0, res.output
    assert server.asked == ["101", "102", "103", "103", "104", "106", "107"]
    assert sorted(tf.load_journal()) == NO_TEXT
    assert _cache()["102"]["text"] == QUOTE_TEXT


def test_429_backs_off_then_stops_cleanly(data, monkeypatch):
    waits = []
    monkeypatch.setattr(ts.time, "sleep", waits.append)
    r429 = _Resp(429)
    # 101 recovers after one 429; 103 outlasts the backoff (1 + TS_MAX_RETRIES tries)
    server = Server(script={"101": [r429], "103": [r429, r429, r429]})
    monkeypatch.setattr(ts, "_ts_get", server)
    res = _run()
    assert res.exit_code == 1 and "Stopped early: 429 on all 3 tries" in res.output
    assert server.asked == ["101", "101", "102", "103", "103", "103"]
    # the pacing sleep is 0 here: one backoff for 101, one after each of 103's
    # tries but the last (the run stops on it, so a wait would only idle)
    assert [w for w in waits if w] == [settings.TS_DEFAULT_BACKOFF_S] * 3  # no Retry-After
    assert {i: e["kind"] for i, e in tf.load_journal().items()} == {
        "101": "media_only",
        "102": "quote",
        "103": "error",
    }
    assert _cache()["102"]["text"] == QUOTE_TEXT  # what was read is applied
    # the next run retries the failed read, after the ones never read
    assert [c["id"] for c in tf.plan(H)["todo"]] == ["104", "106", "107", "103"]


def test_repeated_failures_stop_and_are_retried(data, monkeypatch):
    err = ts.cffi_requests.RequestsError("connection reset")
    server = Server(script={"101": [_Resp(403)], "102": [_Resp(503)], "103": [err]})
    monkeypatch.setattr(ts, "_ts_get", server)
    res = _run()
    assert res.exit_code == 1 and "3 failed reads in a row (last: RequestException)" in res.output
    assert server.asked == ["101", "102", "103"]
    assert {e["kind"] for e in tf.load_journal().values()} == {"error"}
    assert tf.cache_path(H).read_text() == "".join(json.dumps(r) + "\n" for r in CACHED)

    # a failure streak broken by an answer does not stop the run
    server = Server(
        script={
            "101": [_Resp(503)],
            "102": [_Resp(503)],
            "106": [_Resp(200, ValueError("Expecting value"))],
        }
    )
    monkeypatch.setattr(ts, "_ts_get", server)
    res = _run()
    assert res.exit_code == 0, res.output
    # never-read ids first, then the failed reads
    assert server.asked == ["104", "106", "107", "101", "102", "103"]
    assert [i for i, e in tf.load_journal().items() if e["kind"] == "error"] == [
        "101",
        "102",
        "106",
    ]
    assert tf.load_journal()["106"]["error"] == "not JSON"


def test_answer_for_another_status_is_an_error(data, monkeypatch):
    server = Server(script={"101": [_Resp(200, _status("999"))]})
    monkeypatch.setattr(ts, "_ts_get", server)
    tf.fetch(H, tf.plan(H, limit=1)["todo"])
    assert tf.load_journal()["101"]["kind"] == "error"


def test_apply_leaves_records_that_changed_since(data, monkeypatch):
    monkeypatch.setattr(ts, "_ts_get", Server())
    tf.fetch(H, tf.plan(H)["todo"])
    # a forced collect-truth rewrote 102 with other text before the apply
    recs = [dict(r) for r in CACHED]
    recs[1]["text"] = "Something else entirely"
    _write_cache(recs)
    a = tf.apply(H)
    assert a["drifted"] == ["102"] and a["text_gained"] == 1  # 103 only
    assert _cache()["102"]["text"] == "Something else entirely"


# ── downstream: score-posts and topic-label pick up exactly those posts ──


def test_score_posts_rescores_only_filled_posts_and_topic_label_sees_them(data, monkeypatch):
    # the feed as it stands: every row scored, the no-text ones by the old
    # "." stand-in (+0.111)
    feed = pd.DataFrame(CACHED)[["id", "user", "tier", "platform", "created_at", "text"]]
    feed["score_opus_distilled"] = np.where(feed["text"] == "", 0.111, 0.4)
    feed.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    pd.DataFrame({"id": ["9"], "user": ["x"], "text": ["an X post about Iran"]}).to_parquet(
        settings.SENTIMENT_OUTPUT, index=False
    )
    assert set(tl.posts_to_label()["id"]) == {"9", "105"}

    monkeypatch.setattr(ts, "_ts_get", Server())
    assert _run().exit_code == 0

    sent = []

    def fake_score(texts, md, bs=64, max_len=None):
        sent.append(list(texts))
        return np.full(len(texts), -0.6)

    monkeypatch.setattr(sl, "score_with_distilled", fake_score)
    out = sl.score_post_file([tf.cache_path(H)], settings.TRUMP_FEED_STANCE, max_len=256)
    assert sent == [[QUOTE_TEXT, RT_TEXT]]
    s = out.set_index("id")["score_opus_distilled"]
    assert s["102"] == s["103"] == -0.6 and s["105"] == 0.4
    assert (s[["101", "104", "106", "107"]] == 0.111).all()  # still no text: dropped downstream
    assert set(tl.posts_to_label()["id"]) == {"9", "105", "102", "103"}


# ── quote posts cached as Truth Social's bare fallback, "RT: <uri>" ──

URI = "https://truthsocial.com/users/realDonaldTrump/statuses/{}"
# 108 quotes his own 105 (text), 109 his own 101 (image only), both cached as
# the fallback alone; 110 is a quote with words of its own after the fallback
FALLBACK = [
    _rec("108", f"RT: {URI.format(105)}"),
    _rec("109", f"RT: {URI.format(101)}"),
    _rec("110", f"RT: {URI.format(105)} Great going, Freedom Caucus!"),
]


def _fallback_status(pid, quoted_id, own=""):
    """A quote post as Truth Social sends it: the fallback in a quote-inline
    span, then any words of its own."""
    link = URI.format(quoted_id)
    span = f'<span class="quote-inline"><br/>RT: <a href="{link}">{link}</a></span>'
    quoted = {
        "id": quoted_id,
        "content": "<p>Strikes on Iran tonight</p>" if quoted_id == "105" else "",
        "account": {"acct": H},
        "media_attachments": [] if quoted_id == "105" else [{"type": "image"}],
    }
    return {
        "id": pid,
        "content": f"<p>{span}{own}</p>",
        "media_attachments": [],
        "reblog": None,
        "quote_id": quoted_id,
        "quote": quoted,
    }


class FallbackServer(Server):
    def __call__(self, url, params=None, timeout=None):
        pid = url.rsplit("/", 1)[-1]
        if pid in ("108", "109"):
            self.asked.append(pid)
            return _Resp(200, _fallback_status(pid, {"108": "105", "109": "101"}[pid]))
        return super().__call__(url, params, timeout)


def test_plan_reads_bare_quote_fallbacks(data, monkeypatch):
    _write_cache(CACHED + FALLBACK)
    p = tf.plan(H)
    assert [c["id"] for c in p["todo"]] == [*NO_TEXT, "108", "109"]  # not 110: words of its own
    assert p["no_text"] == 8 and p["quote_fallback"] == 2
    monkeypatch.setattr(ts, "_ts_get", _forbidden)
    res = _run("--estimate")
    assert "8 with no words of their own (2 of them only a quote fallback" in res.output


def _labels(ids, **cols):
    return pd.DataFrame({"id": ids, **{k: [v] * len(ids) for k, v in cols.items()}})


def test_bare_fallback_quotes_take_the_quoted_text_or_leave(data, monkeypatch):
    recs = CACHED + FALLBACK
    _write_cache(recs)
    feed = pd.DataFrame(recs)[["id", "user", "tier", "platform", "created_at", "text"]]
    feed["score_opus_distilled"] = 0.2
    feed.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    pd.DataFrame({"id": ["9"], "user": ["x"], "text": ["an X post about Iran"]}).to_parquet(
        settings.SENTIMENT_OUTPUT, index=False
    )
    # labels made from the fallback (108, 109) and from real text (105, 110)
    _labels(["105", "108", "109", "110"], about_war=False).to_parquet(tl.labels_path())
    check = settings.PROCESSED_DIR / "teacher_check_trump_claude-opus-5.parquet"
    _labels(["105", "108"], score_teacher=0.5).to_parquet(check)

    monkeypatch.setattr(ts, "_ts_get", FallbackServer())
    res = _run()
    assert res.exit_code == 0, res.output
    assert "6 records rewritten, 3 of them gained text and 1 lost a bare quote fallback" in (
        res.output
    )
    assert "4 feed rows cleared (4 archived)" in res.output  # 102, 103, 108, 109
    # 108 and 109 quote his own posts; only 108's words count twice
    assert "Quote posts read: 4 (2 carry the quoted text); 2 quote a post in this cache" in (
        res.output
    )
    assert "(1 of them with text, which counts twice)" in res.output
    assert "teacher-check --source trump (paid" in res.output

    c = _cache()
    assert c["108"] == {
        **_rec("108", "RT @realDonaldTrump: Strikes on Iran tonight"),
        "quote_of": "105",
    }
    assert c["109"] == {**_rec("109", ""), "quote_of": "101"}  # leaves every denominator
    assert c["110"] == FALLBACK[2]  # words of its own: not read, not changed

    # what was made from the fallback moved to the archives; the rest stayed
    assert sorted(pd.read_parquet(tl.labels_path())["id"]) == ["105", "110"]
    arch = pd.read_parquet(settings.PROCESSED_DIR / "topic_labels_superseded.parquet")
    assert sorted(arch["id"]) == ["108", "109"]
    assert set(arch["superseded_reason"]) == {tf.SUPERSEDED_REASON}
    assert list(pd.read_parquet(check)["id"]) == ["105"]
    feed_arch = pd.read_parquet(
        settings.PROCESSED_DIR / "truthsocial_trump_stance_superseded.parquet"
    )
    old = feed_arch.set_index("id")
    assert old.loc["108", "text"] == f"RT: {URI.format(105)}"
    assert sorted(old.index) == ["102", "103", "108", "109"]  # every row whose text changed
    assert (old["score_opus_distilled"] == 0.2).all()

    f = pd.read_parquet(settings.TRUMP_FEED_STANCE).set_index("id")
    assert f.loc["108", "text"] == c["108"]["text"] and f.loc["109", "text"] == ""
    assert f.loc[["108", "109"], "score_opus_distilled"].isna().all()

    # a topic-label run before score-posts already sends the new text
    todo = tl.posts_to_label().set_index("id")["text"]
    assert todo["108"] == c["108"]["text"] and "109" not in todo
    # score-posts rescores exactly the cleared row with text (102 and 103
    # gained text from empty rows the old way)
    sent = []

    def fake_score(texts, md, bs=64, max_len=None):
        sent.append(list(texts))
        return np.full(len(texts), -0.6)

    monkeypatch.setattr(sl, "score_with_distilled", fake_score)
    out = sl.score_post_file([tf.cache_path(H)], settings.TRUMP_FEED_STANCE, max_len=256)
    assert sorted(sent[0]) == sorted([QUOTE_TEXT, RT_TEXT, c["108"]["text"]])
    s = out.set_index("id")["score_opus_distilled"]
    assert s["108"] == -0.6 and np.isnan(s["109"]) and s["110"] == 0.2

    # a rerun changes nothing and archives nothing twice
    n_arch = len(pd.read_parquet(settings.PROCESSED_DIR / "topic_labels_superseded.parquet"))
    a = tf.apply(H)
    assert a["changed"] == 0 and a["labels_moved"] == {} and a["feed_rows_cleared"] == 0
    assert (
        len(pd.read_parquet(settings.PROCESSED_DIR / "topic_labels_superseded.parquet")) == n_arch
    )


def test_apply_killed_before_the_cache_write_finishes_on_rerun(data, monkeypatch):
    recs = CACHED + FALLBACK
    _write_cache(recs)
    feed = pd.DataFrame(recs)[["id", "user", "tier", "platform", "created_at", "text"]]
    feed.assign(score_opus_distilled=0.2).to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    _labels(["108"], about_war=False).to_parquet(tl.labels_path())
    monkeypatch.setattr(ts, "_ts_get", FallbackServer())
    tf.fetch(H, tf.plan(H)["todo"])

    real = tf._write_jsonl_atomic

    def killed(path, records):
        raise KeyboardInterrupt

    monkeypatch.setattr(tf, "_write_jsonl_atomic", killed)
    with pytest.raises(KeyboardInterrupt):
        tf.apply(H)
    assert _cache()["108"]["text"] == FALLBACK[0]["text"]  # the cache is written last
    monkeypatch.setattr(tf, "_write_jsonl_atomic", real)
    a = tf.apply(H)
    assert a["text_gained"] == 3 and a["labels_moved"] == {}  # moved by the killed run
    assert _cache()["108"]["text"].startswith("RT @realDonaldTrump: ")
    feed_arch = pd.read_parquet(
        settings.PROCESSED_DIR / "truthsocial_trump_stance_superseded.parquet"
    )
    assert sorted(feed_arch["id"]) == ["102", "103", "108", "109"]  # each archived once


def test_failed_reads_that_never_clear_do_not_stall_the_rest(data, monkeypatch):
    # three neighbouring ids that always fail, first in id order
    bad = {i: [_Resp(503)] * 10 for i in ("101", "102", "103")}
    server = Server(script=bad)
    monkeypatch.setattr(ts, "_ts_get", server)
    assert _run().exit_code == 1  # stops on them before reading anything else
    assert server.asked == ["101", "102", "103"]
    # the next run reads the never-read ids first, then stops on the three again
    assert _run().exit_code == 1
    assert server.asked[3:] == ["104", "106", "107", "101", "102", "103"]
    assert {i for i, e in tf.load_journal().items() if e["kind"] != "error"} == {
        "104",
        "106",
        "107",
    }
    # the failed reads rotate: the longest untried goes first
    with open(tf.journal_path(), "a") as f:
        f.write(json.dumps({"id": "101", "user": H, "kind": "error", "checked_at": "2099-01-01"}))
        f.write("\n")
    assert [c["id"] for c in tf.plan(H)["todo"]] == ["102", "103", "101"]


def test_reply_caches_are_refused(data, monkeypatch):
    monkeypatch.setattr(ts, "_ts_get", _forbidden)
    with pytest.raises(ValueError, match="reply cache"):
        tf.plan("replies_ceasefire")
    res = _run("--handle", "replies_ceasefire", "--estimate")
    assert res.exit_code == 1 and "is a reply cache" in res.output
