"""Incremental refresh of a Truth Social account cache (stubbed API)."""

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from src.collectors import truthsocial_collector as ts


def _post(i, when):
    return {
        "id": str(i),
        "user": "t",
        "text": f"p{i}",
        "created_at": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "metrics": {"replies": 1},
        "platform": "truthsocial",
    }


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ts, "_account_cache_path", lambda h: tmp_path / f"{h}.jsonl")
    return tmp_path / "t.jsonl"


def test_incremental_appends_only_new_and_dedupes_overlap(cache, monkeypatch):
    t0 = datetime(2026, 4, 1, tzinfo=UTC)
    # 1 post/day for 10 days cached (newest first, like the API)
    old = [_post(i, t0 + timedelta(days=i, hours=1)) for i in range(10)][::-1]
    cache.write_text("".join(json.dumps(p) + "\n" for p in old))

    calls = []

    def fake_api(username, start_date, end_date, on_batch=None, **kw):
        calls.append((start_date, end_date))
        # API returns everything from start_date's day onward, newest first
        allp = [_post(i, t0 + timedelta(days=i, hours=1)) for i in range(20)]
        out = [
            p
            for p in allp[::-1]
            if datetime.fromisoformat(p["created_at"].replace("Z", "+00:00")).date() >= start_date
        ]
        if on_batch:
            on_batch(out)
        return out

    monkeypatch.setattr(ts, "collect_via_public_api", fake_api)
    monkeypatch.setattr(ts, "_has_truthbrush_creds", lambda: False)

    got = ts.collect_user("t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30))

    assert calls == [(date(2026, 4, 10), date(2026, 4, 30))]  # starts on the latest cached DAY
    assert len(got) == 20 and len({p["id"] for p in got}) == 20  # overlap day deduped
    assert sum(1 for _ in open(cache)) == 20  # appended
    assert all(p["tier"] == "admin" for p in got[:10])


def test_current_cache_skips_api(cache, monkeypatch):
    cache.write_text(json.dumps(_post(1, datetime(2026, 9, 15, tzinfo=UTC))) + "\n")
    monkeypatch.setattr(
        ts, "collect_via_public_api", lambda *a, **k: pytest.fail("API should not be called")
    )
    monkeypatch.setattr(ts, "_has_truthbrush_creds", lambda: False)
    got = ts.collect_user("t", "admin", start=date(2026, 2, 1), end=date(2026, 9, 14))
    assert len(got) == 1


def _fake_walk(posts, complete):
    def walk(*a, **k):
        return [dict(p) for p in posts]

    walk.last_complete = complete
    return walk


def test_force_complete_walk_merges(cache, monkeypatch):
    # A complete forced walk over a narrower window (--since) must keep the
    # cached posts outside it.
    cache.write_text(json.dumps(_post(1, datetime(2026, 4, 1, tzinfo=UTC))) + "\n")
    monkeypatch.setattr(
        ts, "collect_via_public_api", _fake_walk([_post(7, datetime(2026, 9, 2, tzinfo=UTC))], True)
    )
    got = ts.collect_user("t", "admin", start=date(2026, 9, 1), end=date(2026, 9, 14), force=True)
    assert [p["id"] for p in got] == ["7", "1"]
    assert len(cache.read_text().splitlines()) == 2


def test_force_incomplete_walk_merges(cache, monkeypatch, caplog):
    """A forced walk cut short keeps the cached posts it never reached and
    takes the fetched version of the ones it did."""
    t0 = datetime(2026, 4, 1, tzinfo=UTC)
    old = [_post(i, t0 + timedelta(days=i)) for i in range(5)]
    cache.write_text("".join(json.dumps(p) + "\n" for p in old))
    refreshed = {**_post(4, t0 + timedelta(days=4)), "metrics": {"replies": 99}}
    walk = [_post(6, t0 + timedelta(days=6)), _post(5, t0 + timedelta(days=5)), refreshed]
    monkeypatch.setattr(ts, "collect_via_public_api", _fake_walk(walk, False))
    with caplog.at_level("WARNING", logger=ts.logger.name):
        got = ts.collect_user(
            "t", "admin", start=date(2026, 2, 1), end=date(2026, 9, 14), force=True
        )
    assert [p["id"] for p in got] == ["6", "5", "4", "3", "2", "1", "0"]  # newest first
    on_disk = {json.loads(x)["id"]: json.loads(x) for x in cache.read_text().splitlines()}
    assert sorted(on_disk, key=int) == [str(i) for i in range(7)]
    assert on_disk["4"]["metrics"]["replies"] == 99  # new version wins
    assert any("INCOMPLETE" in r.getMessage() for r in caplog.records)


def test_force_incomplete_empty_walk_leaves_cache(cache, monkeypatch):
    t0 = datetime(2026, 4, 1, tzinfo=UTC)
    # incremental runs append, so the file is oldest first
    cache.write_text("".join(json.dumps(_post(i, t0 + timedelta(days=i))) + "\n" for i in (1, 2)))
    before = cache.read_text()
    monkeypatch.setattr(ts, "collect_via_public_api", _fake_walk([], False))
    got = ts.collect_user("t", "admin", start=date(2026, 2, 1), end=date(2026, 9, 14), force=True)
    assert [p["id"] for p in got] == ["2", "1"] and cache.read_text() == before


def test_force_walk_cut_by_429_merges(cache, monkeypatch):
    """End to end over the stubbed HTTP layer: the server 429s on the 3rd
    statuses page, so the forced walk is incomplete and the older cached
    posts survive."""
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    cache.write_text(
        "".join(
            json.dumps(_post(i, t0 + timedelta(hours=i - 1000))) + "\n" for i in range(1000, 1200)
        )
    )
    server, _ = _fake_server(_statuses(200, t0), fail_429_on_call=4)  # lookup + 2 pages
    monkeypatch.setattr(ts, "_ts_get_paced", server)
    got = ts.collect_user("t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), force=True)
    assert len(got) == 200 and len(cache.read_text().splitlines()) == 200
    fetched = {p["id"]: p for p in got if p["text"].startswith("s")}  # server's versions
    assert len(fetched) == 80 and min(fetched, key=int) == "1120"
    assert all(p["tier"] == "admin" for p in fetched.values())


# ── page walker against a stubbed HTTP layer ───────────────────────


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, ""

    def json(self):
        return self._body


def _statuses(n, t0):
    """n statuses one hour apart, ids ascending with time (snowflake-like)."""
    return [
        {
            "id": str(1000 + i),
            "content": f"<p>s{i}</p>",
            "created_at": (t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "reblogs_count": 0,
            "favourites_count": 0,
            "replies_count": i,
        }
        for i in range(n)
    ]


def _fake_server(all_statuses, fail_429_on_call=None):
    """Mastodon-ish /statuses with min_id (forward) and max_id (backward)."""
    state = {"calls": 0}

    def get(url, params=None):
        state["calls"] += 1
        if url.endswith("/accounts/lookup"):
            return _Resp(200, {"id": "7"})
        if fail_429_on_call and state["calls"] == fail_429_on_call:
            return _Resp(429, headers={"retry-after": "1"})
        limit = params.get("limit", 40)
        newest_first = sorted(all_statuses, key=lambda s: int(s["id"]), reverse=True)
        if "min_id" in params:
            page = [s for s in newest_first if int(s["id"]) > int(params["min_id"])][-limit:]
        elif "max_id" in params:
            page = [s for s in newest_first if int(s["id"]) < int(params["max_id"])][:limit]
        else:
            page = newest_first[:limit]
        return _Resp(200, page)

    return get, state


def test_backward_walk_stops_at_cache_edge(monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    server, state = _fake_server(_statuses(100, t0))
    monkeypatch.setattr(ts, "_ts_get_paced", server)
    got = ts.collect_via_public_api("t", date(2026, 4, 1), date(2026, 4, 30), stop_at_id="1049")
    assert [int(p["id"]) for p in got] == list(range(1099, 1049, -1))  # newest first, > edge
    assert ts.collect_via_public_api.last_complete is True


def test_backward_walk_no_progress_guard(monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    same_page = sorted(_statuses(20, t0), key=lambda s: -int(s["id"]))

    def ignores_cursor(url, params=None):
        if url.endswith("/accounts/lookup"):
            return _Resp(200, {"id": "7"})
        return _Resp(200, same_page)

    monkeypatch.setattr(ts, "_ts_get_paced", ignores_cursor)
    got = ts.collect_via_public_api("t", date(2026, 4, 1), date(2026, 4, 30))
    assert len(got) == 20 and ts.collect_via_public_api.last_complete is False


# ── ReTruths and media ─────────────────────────────────────────────


def _retruth(sid, when, inner="<p>Peace through <b>strength</b></p>", media=0):
    return {
        "id": sid,
        "content": "",
        "created_at": when,
        "media_attachments": [],
        "reblog": {
            "id": "555",
            "content": inner,
            "account": {"acct": "WhiteHouse"},
            "media_attachments": [{"type": "image"}] * media,
        },
    }


def test_status_content_retruth_and_media():
    when = "2026-04-10T00:00:00.000Z"
    rt = ts._status_content(_retruth("1", when))
    assert rt == {"text": "RT @WhiteHouse: Peace through strength", "reblog_of": "555"}
    # a ReTruth of a post with no text stays a no-text post, media counted
    bare = ts._status_content(_retruth("2", when, inner="", media=2))
    assert bare == {"text": "", "reblog_of": "555", "media": 2}
    # ... and so does one under MIN_TEXT_CHARS: the prefix must not lift it over
    for inner in ("<p>!!</p>", "<p>\U0001f64f</p>"):
        short = ts._status_content(_retruth("3", when, inner=inner, media=1))
        assert short == {"text": "", "reblog_of": "555", "media": 1}
    assert ts._status_content(_retruth("4", when, inner="<p>No!</p>"))["text"] == (
        "RT @WhiteHouse: No!"
    )
    pic = ts._status_content({"content": "", "media_attachments": [{}, {}, {}]})
    assert pic == {"text": "", "media": 3}
    assert ts._status_content({"content": "<p>hi</p>", "media_attachments": []}) == {"text": "hi"}
    assert ts._status_content({"content": None}) == {"text": ""}


def test_public_api_records_retruth_and_media(monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    statuses = _statuses(3, t0)
    statuses[1] = {**statuses[1], **_retruth(statuses[1]["id"], statuses[1]["created_at"])}
    statuses[2] = {**statuses[2], "content": "", "media_attachments": [{"type": "video"}]}
    server, _ = _fake_server(statuses)
    monkeypatch.setattr(ts, "_ts_get_paced", server)
    got = {p["id"]: p for p in ts.collect_via_public_api("t", date(2026, 4, 1), date(2026, 4, 30))}
    assert got["1001"]["text"] == "RT @WhiteHouse: Peace through strength"
    assert got["1001"]["reblog_of"] == "555" and "media" not in got["1001"]
    assert got["1002"]["text"] == "" and got["1002"]["media"] == 1
    assert "reblog_of" not in got["1000"] and "media" not in got["1000"]  # old shape


def test_truthbrush_records_retruth(monkeypatch):
    when = "2026-04-10T00:00:00.000Z"
    api = type("Api", (), {"pull_statuses": lambda self, u, **kw: iter([_retruth("9", when)])})
    monkeypatch.setattr(ts, "_get_truthbrush_api", lambda: api())
    got = ts.collect_via_truthbrush("t", date(2026, 4, 1), date(2026, 4, 30))
    assert got[0]["text"].startswith("RT @WhiteHouse: ") and got[0]["reblog_of"] == "555"


def test_reply_record_counts_media():
    status = {
        "id": "3",
        "content": "",
        "created_at": "2026-04-10T00:00:00.000Z",
        "media_attachments": [{"type": "image"}],
        "account": {"username": "u"},
    }
    rec = ts._reply_record(status, parent_id="900")
    assert rec["text"] == "" and rec["media"] == 1 and "reblog_of" not in rec
    assert rec["source"] == "reply" and rec["account"]["username"] == "u"


def test_paced_get_retries_429(monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    monkeypatch.setattr(ts.settings, "TS_MIN_BACKOFF_S", 0)
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    seq = [_Resp(429, headers={"retry-after": "0"}), _Resp(200, {"ok": 1})]
    monkeypatch.setattr(ts, "_ts_get", lambda url, params=None, timeout=None: seq.pop(0))
    assert ts._ts_get_paced("u").status_code == 200


def test_anonymous_incremental_interrupt_then_resume(cache, monkeypatch):
    """Edge at id 1009, 200 newer posts on the server (5 pages of 40).
    Run 1 dies on the 3rd page: cache untouched, partial holds 2 pages.
    Run 2 resumes from the partial's oldest id, reaches the edge, merges."""
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    cache.write_text(
        "".join(
            json.dumps(_post(i, t0 + timedelta(hours=i - 1000))) + "\n" for i in range(1000, 1010)
        )
    )
    server, state = _fake_server(_statuses(210, t0))  # ids 1000..1209

    def flaky(url, params=None):
        r = server(url, params)
        return _Resp(429, headers={"retry-after": "1"}) if state["calls"] == 4 else r

    monkeypatch.setattr(ts, "_ts_get_paced", flaky)
    got = ts.collect_user(
        "t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), use_auth=False
    )
    partial = cache.with_suffix(".partial.jsonl")
    assert len(got) == 10  # nothing merged yet
    assert sum(1 for _ in open(cache)) == 10
    assert [json.loads(x)["id"] for x in open(partial)] == [str(i) for i in range(1209, 1129, -1)]

    monkeypatch.setattr(ts, "_ts_get_paced", server)  # healthy again
    got = ts.collect_user(
        "t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), use_auth=False
    )
    assert not partial.exists()
    on_disk = sorted(int(json.loads(x)["id"]) for x in open(cache))
    assert on_disk == list(range(1000, 1210))  # contiguous, complete
    assert len(got) == 210 and got[0]["id"] == "1209"


def test_stale_partial_after_forced_walk_adds_no_duplicates(cache, monkeypatch):
    """An interrupted anonymous run leaves a partial; a forced walk then
    refetches the whole window, those posts included. The next anonymous run
    must not append the partial's posts a second time."""
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    window = {"start": date(2026, 4, 1), "end": date(2026, 4, 30)}
    cache.write_text(
        "".join(
            json.dumps(_post(i, t0 + timedelta(hours=i - 1000))) + "\n" for i in range(1000, 1010)
        )
    )
    server, state = _fake_server(_statuses(210, t0))

    def flaky(url, params=None):
        r = server(url, params)
        return _Resp(429, headers={"retry-after": "1"}) if state["calls"] == 4 else r

    monkeypatch.setattr(ts, "_ts_get_paced", flaky)
    ts.collect_user("t", "admin", **window, use_auth=False)
    partial = cache.with_suffix(".partial.jsonl")
    assert len(partial.read_text().splitlines()) == 80

    monkeypatch.setattr(ts, "_ts_get_paced", server)
    ts.collect_user("t", "admin", **window, force=True)
    assert len(cache.read_text().splitlines()) == 210 and partial.exists()

    newer, _ = _fake_server(_statuses(212, t0))  # two posts since the forced walk
    monkeypatch.setattr(ts, "_ts_get_paced", newer)
    got = ts.collect_user("t", "admin", **window, use_auth=False)
    ids = [json.loads(x)["id"] for x in cache.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 212 and len(got) == 212
    assert not partial.exists()


def test_anonymous_incremental_skips_when_current(cache, monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=UTC)
    cache.write_text(json.dumps(_post(1099, t0 + timedelta(hours=99))) + "\n")
    server, state = _fake_server(_statuses(100, t0))
    monkeypatch.setattr(ts, "_ts_get_paced", server)
    got = ts.collect_user(
        "t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), use_auth=False
    )
    assert len(got) == 1 and not cache.with_suffix(".partial.jsonl").exists()


# ── login / security-code flow (stubbed HTTP) ──────────────────────


def test_login_flow_challenge_then_verify(monkeypatch, tmp_path):
    seen = []

    def fake_post(path, body):
        seen.append((path, body))
        if path == "/oauth/v2/token" and "challenge_id" not in body:
            return _Resp(
                403,
                {
                    "error": "security_code_required",
                    "challenge_id": "ch1",
                    "supported_delivery_methods": [{"kind": "email", "value": "d***@g***"}],
                },
            )
        if path == "/oauth/v2/choose_delivery_method" and body == {
            "username": "u",
            "challenge_id": "ch1",
            "delivery_method": "email",
        }:
            return _Resp(200, {"sent": True})
        if path == "/oauth/v2/verify_security_code" and body.get("security_code") == "654321":
            return _Resp(200, {"access_token": "tok_abc"})
        return _Resp(400, {"error": "bad"})

    monkeypatch.setattr(ts, "_auth_post", fake_post)

    with pytest.raises(ts.SecurityCodeRequired) as ex:
        ts.request_token("u", "p")
    assert ex.value.challenge_id == "ch1" and ex.value.delivery_methods[0]["kind"] == "email"

    r = ts.request_security_code_delivery("u", "p", "ch1", "email")
    assert r.status_code == 200
    assert seen[-1][0] == "/oauth/v2/choose_delivery_method"
    assert "password" not in seen[-1][1] and "client_id" not in seen[-1][1]

    tok = ts.verify_security_code("u", "p", "ch1", "654321")
    assert tok == "tok_abc"
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")
    ts.save_token_to_env(tok, env_path=env)
    assert "TRUTHSOCIAL_TOKEN=tok_abc" in env.read_text() and "OTHER=1" in env.read_text()


def test_login_flow_direct_token(monkeypatch):
    monkeypatch.setattr(ts, "_auth_post", lambda path, body: _Resp(200, {"access_token": "t1"}))
    assert ts.request_token("u", "p") == "t1"


# ── v2 descendants walker (stubbed HTTP) ───────────────────────────


class _HResp(_Resp):
    pass


def test_iter_descendants_follows_link_and_filters_direct(monkeypatch):
    pid = "900"
    pages = {
        None: (
            [{"id": "1", "in_reply_to_id": pid}, {"id": "2", "in_reply_to_id": "1"}],
            '<https://truthsocial.com/api/v2/statuses/900/context/descendants?offset=2&sort=oldest>; rel="next"',
        ),
        "https://truthsocial.com/api/v2/statuses/900/context/descendants?offset=2&sort=oldest": (
            [{"id": "3", "in_reply_to_id": pid}],
            "",
        ),
    }
    calls = []

    def fake_get(url, params, token):
        calls.append((url, params))
        key = None if params else url
        body, link = pages[key]
        return _HResp(200, body, headers={"link": link})

    monkeypatch.setattr(ts, "_ts_get_auth_paced", fake_get)
    got = list(ts.iter_descendants(pid, token="t"))
    assert [g["id"] for g in got] == ["1", "3"]  # sub-thread reply "2" dropped
    assert calls[0][1] == {"sort": "oldest"} and calls[1][1] is None
    assert calls[0][0].endswith(ts.settings.TS_DESCENDANTS_PATH.format(id=pid))
    assert list(ts.iter_descendants(pid, only_direct=False, token="t")) and [
        g["id"] for g in ts.iter_descendants(pid, only_direct=False, token="t")
    ] == ["1", "2", "3"]


def test_iter_descendants_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(ts, "_ts_get_auth_paced", lambda url, params, token: _HResp(404, None))
    with pytest.raises(RuntimeError):
        list(ts.iter_descendants("900", token="t"))
