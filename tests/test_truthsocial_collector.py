"""Incremental refresh of a Truth Social account cache (stubbed API)."""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from src.collectors import truthsocial_collector as ts


def _post(i, when):
    return {"id": str(i), "user": "t", "text": f"p{i}",
            "created_at": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "metrics": {"replies": 1}, "platform": "truthsocial"}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ts, "_account_cache_path", lambda h: tmp_path / f"{h}.jsonl")
    return tmp_path / "t.jsonl"


def test_incremental_appends_only_new_and_dedupes_overlap(cache, monkeypatch):
    t0 = datetime(2026, 4, 1, tzinfo=timezone.utc)
    # 1 post/day for 10 days cached (newest first, like the API)
    old = [_post(i, t0 + timedelta(days=i, hours=1)) for i in range(10)][::-1]
    cache.write_text("".join(json.dumps(p) + "\n" for p in old))

    calls = []
    def fake_api(username, start_date, end_date, on_batch=None, **kw):
        calls.append((start_date, end_date))
        # API returns everything from start_date's day onward, newest first
        allp = [_post(i, t0 + timedelta(days=i, hours=1)) for i in range(20)]
        out = [p for p in allp[::-1]
               if datetime.fromisoformat(p["created_at"].replace("Z", "+00:00")).date() >= start_date]
        if on_batch:
            on_batch(out)
        return out
    monkeypatch.setattr(ts, "collect_via_public_api", fake_api)
    monkeypatch.setattr(ts, "_has_truthbrush_creds", lambda: False)

    got = ts.collect_user("t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30))

    assert calls == [(date(2026, 4, 10), date(2026, 4, 30))]   # starts on the latest cached DAY
    assert len(got) == 20 and len({p["id"] for p in got}) == 20  # overlap day deduped
    assert sum(1 for _ in open(cache)) == 20                      # appended
    assert all(p["tier"] == "admin" for p in got[:10])


def test_current_cache_skips_api(cache, monkeypatch):
    cache.write_text(json.dumps(_post(1, datetime(2026, 9, 15, tzinfo=timezone.utc))) + "\n")
    monkeypatch.setattr(ts, "collect_via_public_api",
                        lambda *a, **k: pytest.fail("API should not be called"))
    monkeypatch.setattr(ts, "_has_truthbrush_creds", lambda: False)
    got = ts.collect_user("t", "admin", start=date(2026, 2, 1), end=date(2026, 9, 14))
    assert len(got) == 1


def test_force_overwrites(cache, monkeypatch):
    cache.write_text(json.dumps(_post(1, datetime(2026, 4, 1, tzinfo=timezone.utc))) + "\n")
    monkeypatch.setattr(ts, "collect_via_public_api",
                        lambda *a, **k: [_post(7, datetime(2026, 4, 2, tzinfo=timezone.utc))])
    got = ts.collect_user("t", "admin", start=date(2026, 2, 1), end=date(2026, 9, 14), force=True)
    assert [p["id"] for p in got] == ["7"]
    assert sum(1 for _ in open(cache)) == 1


# ── page walker against a stubbed HTTP layer ───────────────────────

class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, ""
    def json(self):
        return self._body


def _statuses(n, t0):
    """n statuses one hour apart, ids ascending with time (snowflake-like)."""
    return [{"id": str(1000 + i), "content": f"<p>s{i}</p>",
             "created_at": (t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
             "reblogs_count": 0, "favourites_count": 0, "replies_count": i}
            for i in range(n)]


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
    t0 = datetime(2026, 4, 10, tzinfo=timezone.utc)
    server, state = _fake_server(_statuses(100, t0))
    monkeypatch.setattr(ts, "_ts_get_paced", server)
    got = ts.collect_via_public_api("t", date(2026, 4, 1), date(2026, 4, 30), stop_at_id="1049")
    assert [int(p["id"]) for p in got] == list(range(1099, 1049, -1))   # newest first, > edge
    assert ts.collect_via_public_api.last_complete is True


def test_backward_walk_no_progress_guard(monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=timezone.utc)
    same_page = sorted(_statuses(20, t0), key=lambda s: -int(s["id"]))
    def ignores_cursor(url, params=None):
        if url.endswith("/accounts/lookup"):
            return _Resp(200, {"id": "7"})
        return _Resp(200, same_page)
    monkeypatch.setattr(ts, "_ts_get_paced", ignores_cursor)
    got = ts.collect_via_public_api("t", date(2026, 4, 1), date(2026, 4, 30))
    assert len(got) == 20 and ts.collect_via_public_api.last_complete is False


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
    t0 = datetime(2026, 4, 10, tzinfo=timezone.utc)
    cache.write_text("".join(json.dumps(_post(i, t0 + timedelta(hours=i - 1000))) + "\n"
                             for i in range(1000, 1010)))
    server, state = _fake_server(_statuses(210, t0))          # ids 1000..1209

    def flaky(url, params=None):
        r = server(url, params)
        return _Resp(429, headers={"retry-after": "1"}) if state["calls"] == 4 else r
    monkeypatch.setattr(ts, "_ts_get_paced", flaky)
    got = ts.collect_user("t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), use_auth=False)
    partial = cache.with_suffix(".partial.jsonl")
    assert len(got) == 10                                       # nothing merged yet
    assert sum(1 for _ in open(cache)) == 10
    assert [json.loads(x)["id"] for x in open(partial)] == [str(i) for i in range(1209, 1129, -1)]

    monkeypatch.setattr(ts, "_ts_get_paced", server)           # healthy again
    got = ts.collect_user("t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), use_auth=False)
    assert not partial.exists()
    on_disk = sorted(int(json.loads(x)["id"]) for x in open(cache))
    assert on_disk == list(range(1000, 1210))                    # contiguous, complete
    assert len(got) == 210 and got[0]["id"] == "1209"


def test_anonymous_incremental_skips_when_current(cache, monkeypatch):
    monkeypatch.setattr(ts.settings, "TS_PAGE_DELAY_S", 0)
    t0 = datetime(2026, 4, 10, tzinfo=timezone.utc)
    cache.write_text(json.dumps(_post(1099, t0 + timedelta(hours=99))) + "\n")
    server, state = _fake_server(_statuses(100, t0))
    monkeypatch.setattr(ts, "_ts_get_paced", server)
    got = ts.collect_user("t", "admin", start=date(2026, 4, 1), end=date(2026, 4, 30), use_auth=False)
    assert len(got) == 1 and not cache.with_suffix(".partial.jsonl").exists()


# ── login / security-code flow (stubbed HTTP) ──────────────────────

def test_login_flow_challenge_then_verify(monkeypatch, tmp_path):
    seen = []
    def fake_post(path, body):
        seen.append((path, body))
        if path == "/oauth/v2/token" and "challenge_id" not in body:
            return _Resp(403, {"error": "security_code_required", "challenge_id": "ch1",
                               "supported_delivery_methods": [{"kind": "email", "value": "d***@g***"}]})
        if path == "/oauth/v2/choose_delivery_method" and \
                body == {"username": "u", "challenge_id": "ch1", "delivery_method": "email"}:
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
