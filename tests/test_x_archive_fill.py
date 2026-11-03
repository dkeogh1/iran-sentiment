"""
x-archive fill (GET /2/tweets/search/all on collect's window grid), tested
against a stub search client that honours the time and id bounds: no
network, no spend. Data lives under tmp_path.
"""

import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
import requests
import tweepy
from click.testing import CliRunner

from config import settings, timeline
from src.collectors import x_archive as xa
from src.collectors import x_archive_fill as xf
from src.collectors import x_backfill as xb
from src.collectors import x_collector as xc
from src.collectors import x_counts as xcn
from tests.test_x_counts_archive import (
    ENROLL,
    UNAUTHORIZED,
    USAGE_CAPPED,
    _http_error,
    _ms,
)

H = timedelta(hours=1)


def _t(s: str) -> datetime:
    """Naive UTC, as the collector and the counts cache use."""
    return datetime.fromisoformat(s)


F = _t("2026-03-01")  # COLLECTION_START here
T = _t("2026-03-04")  # the end of ANALYSIS_END (Mar 3)
M0 = _t("2026-03-02T11:00")  # the collect window: 35 h, ending 2 h before T
M1 = _t("2026-03-03T22:00")
ACCOUNTS = {"t": ["Aa", "Tt"]}


def _pid(ts: datetime, k: int) -> int:
    """A snowflake id, as X's are (created at ts, plus k ms to keep accounts
    apart)."""
    return xf.id_at(ts) + (k << 22)


def _at(i: int) -> datetime:
    """The time a snowflake id was made, to the ms (naive UTC)."""
    ms = (i >> 22) + xf.SNOWFLAKE_EPOCH_MS
    return datetime.fromtimestamp(ms / 1000, UTC).replace(tzinfo=None)


def _lower(call: dict) -> str:
    """A request's lower edge as a time (since_id is exclusive)."""
    return xcn.z(_at(int(call["since_id"]) + 1))


def _upper(call: dict) -> str:
    """A request's upper edge as a time."""
    return xcn.z(_at(int(call["until_id"])))


def _posts(times: list[datetime], k: int, text="post") -> list[tuple[int, datetime, str]]:
    return [(_pid(t, k), t, f"{text} {i}") for i, t in enumerate(times)]


def _rec(post, user, **kw) -> dict:
    pid, ts, text = post
    return {
        "id": str(pid),
        "user": user,
        "tier": "t",
        "text": text,
        "created_at": ts.isoformat() + "+00:00",
        "metrics": {},
        "lang": "en",
        "platform": "x",
        **kw,
    }


def _jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _counts(handle: str, posts, start=F, end=T) -> None:
    """X's hour counts of these posts over [start, end)."""
    c = xcn.CountsCache(f"from:{handle}", "hour")
    rows, t = [], start
    while t < end:
        n = sum(1 for _, ts, _ in posts if t <= ts < t + H)
        rows.append({"start": _ms(t), "end": _ms(t + H), "tweet_count": n})
        t += H
    c.merge(rows, (start, end))
    c.save()


def _mrow(handle, s, e, cap, got, mode="sample", run_end="2026-03-04T00:00:00Z", hours=35):
    return {
        "run_end": run_end,
        "handle": handle,
        "tier": "t",
        "mode": mode,
        "hours": hours,
        "window_start": xcn.z(s),
        "window_end": xcn.z(e),
        "cap": cap,
        "got": got,
        "new": got,
        "truncated": False,
    }


def _tweet(post, extra: dict | None = None):
    pid, ts, text = post
    data = {"id": str(pid), "text": text, **(extra or {})}
    return SimpleNamespace(
        id=pid,
        text=text,
        created_at=ts.replace(tzinfo=UTC),
        public_metrics={"like_count": 1},
        lang="en",
        data=data,
    )


class SearchStub:
    """
    search/all over `posts` {handle: [(id, created_at, text)]} between
    since_id and until_id (both exclusive; X answers 400 to any mix of ids
    with start_time / end_time: the 2026-11-03 pilot); newest first, up to
    max_results, next_token an offset. It fails the test when a request's
    bounds hold a post the cache or the extras already hold. `fail` maps an
    attempt number to the error it raises; `limited` answers that many
    rate-limit 429s first; `adversarial` always returns max_results posts it
    makes up, with a next_token. `data` adds tweet data (note_tweet, refs).
    """

    def __init__(self, posts, fail=None, limited=0, adversarial=False, data=None):
        self.posts, self.fail, self.limited = posts, fail or {}, limited
        self.adversarial, self.data = adversarial, data or {}
        self.calls: list[dict] = []
        self.attempts = 0
        self._made = 0

    def _known(self, handle: str) -> set[str]:
        ids = {str(r["id"]) for r in xc._load_jsonl(xc._account_cache_path(handle))}
        return ids | {str(r["id"]) for r in xf._load_lines(xf.extras_path(handle))}

    def search_all_tweets(self, query, **p):
        self.attempts += 1
        err = self.fail.get(self.attempts)
        if err is not None:
            raise err
        if self.limited:
            self.limited -= 1
            reset = {"x-rate-limit-reset": str(int(datetime.now(UTC).timestamp()) + 7)}
            raise _http_error(tweepy.errors.TooManyRequests, 429, {"title": "Too"}, reset)
        self.calls.append({"query": query, **p})
        assert p["sort_order"] == "recency" and p["tweet_fields"] == xc._TWEET_FIELDS
        assert 10 <= p["max_results"] <= 100 and "expansions" not in p
        handle = query.split(":", 1)[1]
        lo, hi = p.get("since_id"), p.get("until_id")
        if ("start_time" in p or "end_time" in p) and (lo or hi):
            raise _http_error(
                tweepy.errors.BadRequest,
                400,
                {"title": "Invalid use of 'since_id' or 'until_id' with 'start_time'"},
            )
        assert lo and hi, p
        if self.adversarial:
            n, out = p["max_results"], []
            for _ in range(n):
                self._made += 1
                pid = int(hi) - self._made
                out.append(_tweet((pid, _at(pid).replace(microsecond=0), "made up")))
            return SimpleNamespace(data=out, meta={"next_token": "more"})
        hits = [post for post in self.posts.get(handle, []) if int(lo) < post[0] < int(hi)]
        known = self._known(handle) & {str(h[0]) for h in hits}
        assert not known, f"the request's bounds hold known posts {known}: {p}"
        hits.sort(key=lambda h: -h[0])
        off = int(p.get("next_token") or 0)
        page = hits[off : off + p["max_results"]]
        more = off + p["max_results"] < len(hits)
        return SimpleNamespace(
            data=[_tweet(h, self.data.get(h[0])) for h in page] or None,
            meta={"next_token": str(off + p["max_results"])} if more else {"result_count": 0},
        )


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "FIGURES_DIR", "TRUTH_SOCIAL_RAW_DIR"):
        monkeypatch.setattr(settings, name, tmp_path / name.lower())
    monkeypatch.setattr(settings, "X_RAW_DIR", tmp_path / "x")
    monkeypatch.setattr(settings, "X_SAMPLING_DIR", tmp_path / "x_sampling")
    monkeypatch.setattr(settings, "X_COUNTS_DIR", tmp_path / "x_counts")
    monkeypatch.setattr(settings, "X_ARCHIVE_DIR", tmp_path / "x_archive")
    monkeypatch.setattr(settings, "COLLECTION_START", F)
    monkeypatch.setattr(timeline, "ANALYSIS_END", date(2026, 3, 3))
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"Aa": 5, "Nn": 5})
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_MAX_USD", 5.0)
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 45.0)
    sleeps: list[float] = []
    monkeypatch.setattr(xcn, "_sleep", sleeps.append)
    monkeypatch.setattr(xcn, "_clock", lambda: 0.0)
    from src import cli

    monkeypatch.setattr(cli, "X_ACCOUNTS", ACCOUNTS)
    return SimpleNamespace(path=tmp_path, sleeps=sleeps)


# Aa (sampled, 5 a window): 12 posts in the backward window, 8 in the collect
# window, of which collect got the newest 2. Tt (take-all): 3 and 4, 1 held.
AA_X = _posts([F + H + 2 * i * H for i in range(12)], 1) + _posts(
    [M0 + 2 * H + 3 * i * H for i in range(8)], 1, "late"
)
TT_X = _posts([_t("2026-03-01T05:30"), _t("2026-03-01T12:30"), _t("2026-03-02T02:30")], 2) + _posts(
    [
        _t("2026-03-02T14:30"),
        _t("2026-03-02T20:30"),
        _t("2026-03-03T03:30"),
        _t("2026-03-03T09:30"),
    ],
    2,
    "late",
)
AA_HELD = AA_X[-2:]
TT_HELD = [TT_X[4]]


def _base(aa_held=AA_HELD, tt_held=TT_HELD, aa_x=AA_X, tt_x=TT_X) -> SearchStub:
    _jsonl(xc._account_cache_path("Aa"), [_rec(p, "Aa") for p in aa_held])
    _jsonl(xc._account_cache_path("Tt"), [_rec(p, "Tt") for p in tt_held])
    _counts("Aa", aa_x)
    _counts("Tt", tt_x)
    _jsonl(
        settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl",
        [
            _mrow("Aa", M0, M1, 5, len(aa_held)),
            _mrow("Tt", M0, M1, 500, len(tt_held), mode="take_all"),
        ],
    )
    return SearchStub({"Aa": aa_x, "Tt": tt_x})


def _cli(args, input=None):
    from src import cli

    return CliRunner().invoke(cli.main, args, input=input)


def _no_client(*a, **k):
    raise AssertionError("no client may be built on this path")


def _log() -> list[dict]:
    """The request log's answered lines (each request also has a pending line,
    written before it is sent)."""
    return [line for line in _read(xf.log_path()) if not line.get("pending")]


def _ids(handle: str) -> list[str]:
    return [r["id"] for r in xc._load_jsonl(xc._account_cache_path(handle))]


# ── grid ───────────────────────────────────────────────────────────


def test_grid_steps_back_from_the_collect_windows_and_keeps_them():
    a = _t("2026-03-06T04:30:07")
    step = timedelta(hours=35)
    rows = [_mrow("Aa", a, a + step, 5, 1), _mrow("Aa", a + step, a + step + 30 * H, 6, 1)]
    end = _t("2026-03-09")
    grid = xf.account_grid(xf.collect_rows("Aa", rows, end), 5, F, end)
    back = [w for w in grid if w.origin == "backward"]
    # a - 3 x 35 h is Mar 1 19:30:07: a 19.5 h head stands alone, cap w
    assert [w.s for w in back] == [F, a - 3 * step, a - 2 * step, a - step]
    assert (
        all(w.e - w.s == step for w in back[1:])
        and back[0].e - back[0].s == 19.5 * H + 7 * H / 3600
    )
    assert all(w.cap == 5 for w in back) and back[-1].e == a
    man = [w for w in grid if w.origin == "manifest"]
    assert [(w.s, w.e, w.cap) for w in man] == [(a, a + step, 5), (a + step, a + step + 30 * H, 6)]
    assert grid[-1].e == a + step + 30 * H  # 2.5 h before the end: nothing forward
    # a head under half a window merges: 50 h, cap 7 (_window_cap)
    a2 = _t("2026-03-06")
    grid = xf.account_grid(
        xf.collect_rows("Aa", [_mrow("Aa", a2, a2 + step, 5, 1)], end), 5, F, end
    )
    assert (grid[0].s, grid[0].e, grid[0].cap) == (F, _t("2026-03-03T02:00"), 7)
    assert all(w.e - w.s == step for w in grid[1:-1])
    # half a window or more left at the end: plan_windows from the last edge
    far = _t("2026-03-10")
    grid = xf.account_grid(xf.collect_rows("Aa", rows, far), 5, F, far)
    fwd = [w for w in grid if w.origin == "forward"]
    assert [(w.s, w.e) for w in fwd] == xc.plan_windows(a + step + 30 * H, far, 35)
    # no manifest: plan_windows over the whole range; take-all has no caps
    grid = xf.account_grid([], None, F, end)
    assert [(w.s, w.e) for w in grid] == xc.plan_windows(F, end, 35)
    assert all(w.cap is None for w in grid)


def test_a_later_runs_window_cuts_the_earlier_one_where_its_cursor_began():
    a, step = _t("2026-03-02"), timedelta(hours=35)
    c = a + step + 20 * H  # the next run's cursor: inside the first run's last window
    rows = [
        _mrow("Aa", a, a + step, 5, 5),
        _mrow("Aa", a + step, a + 2 * step, 5, 5),
        _mrow("Aa", c, c + step, 5, 5, run_end="2026-03-08T00:00:00Z"),
        {**_mrow("Aa", a, a + step, 5, 5), "source": "archive"},  # the fill's rows never count
        _mrow("Bb", a, a + step, 5, 5),
    ]
    got = xf.collect_rows("Aa", rows, _t("2026-03-09"))
    assert [(w["s"], w["e"]) for w in got] == [(a, a + step), (a + step, c), (c, c + step)]


def test_a_later_run_owns_its_span_past_the_earlier_runs_trailing_empty_windows():
    # run 1's last two windows got nothing, so run 2's cursor (its newest post
    # + 1 s) sits in run 1's second window, two windows before run 1's end
    a, step = _t("2026-03-02"), timedelta(hours=35)
    run1 = [_mrow("Aa", a + i * step, a + (i + 1) * step, 5, n) for i, n in enumerate((5, 5, 0, 0))]
    c = a + step + 10 * H
    late = "2026-03-20T00:00:00Z"
    run2 = [_mrow("Aa", c + i * step, c + (i + 1) * step, 5, 5, run_end=late) for i in range(4)]
    got = xf.collect_rows("Aa", run1 + run2, _t("2026-04-01"))
    assert [(w["s"], w["e"]) for w in got] == [(a, a + step), (a + step, c)] + [
        (c + i * step, c + (i + 1) * step) for i in range(4)
    ]


def test_a_forced_run_owns_its_whole_span_and_a_later_incremental_run_its_tail():
    # an incremental run's 35 h windows, then `collect --force` over the whole
    # range with 67 h windows, then an incremental run from the forced run's cursor
    step, long = timedelta(hours=35), timedelta(hours=67)
    a = _t("2026-03-05")
    inc = [
        _mrow("Aa", a + i * step, a + (i + 1) * step, 5, 5, run_end="2026-03-22T12:00:00Z")
        for i in range(12)
    ]
    forced = [
        _mrow("Aa", F + i * long, F + (i + 1) * long, 5, 5, "sample", "2026-03-26T03:00:00Z", 67)
        for i in range(9)
    ]
    c = F + 8 * long + 20 * H  # the later run's cursor, in the forced run's last window
    later = [
        _mrow("Aa", c + i * step, c + (i + 1) * step, 5, 5, run_end="2026-03-28T13:00:00Z")
        for i in range(3)
    ]
    end = _t("2026-04-01")
    rows = xf.collect_rows("Aa", inc + later + forced, end)
    assert [(w["s"], w["e"]) for w in rows] == [
        (F + i * long, F + (i + 1) * long) for i in range(8)
    ] + [(F + 8 * long, c)] + [(c + i * step, c + (i + 1) * step) for i in range(3)]
    # collect's density: one cap of 5 a forced window, no short pieces in between
    grid = xf.account_grid(rows, 5, F, end)
    inside = [w for w in grid if a <= w.s < a + 12 * step]
    assert all(w.e - w.s == long and w.cap == 5 for w in inside) and len(inside) == 6


# ── planning ───────────────────────────────────────────────────────


def test_plan_targets_budgets_kinds_and_scope(env):
    _base()
    plan = xf.plan_fill(ACCOUNTS)
    got = {(w["handle"], w["s"]): w for w in plan["windows"]}
    assert list(got) == [("Aa", F), ("Tt", F), ("Aa", M0), ("Tt", M0)]  # by start, then handle
    aa0, tt0, aa1, tt1 = got.values()
    assert (aa0["x_hi"], aa0["target"], aa0["held"], aa0["need_hi"], aa0["budget"]) == (
        12,
        5,
        0,
        5,
        14,
    )
    assert (aa1["x_hi"], aa1["held"], aa1["need_hi"], aa1["budget"]) == (8, 2, 3, 12)
    assert (tt0["target"], tt0["need_hi"], tt1["held"], tt1["need_hi"]) == (3, 3, 1, 3)
    assert [w["kind"] for w in got.values()] == ["late_start", "late_start", "short", "short"]
    assert plan["ceiling"] == 50 and plan["run_max"] == 50 and plan["refusal"] is None
    assert plan["new"] == pytest.approx(14) and plan["expected"] == pytest.approx(14)
    assert plan["requests"] == 5
    empty = xf.plan_fill(ACCOUNTS, scope="empty")
    assert [(w["handle"], w["s"]) for w in empty["windows"]] == [("Aa", F), ("Tt", F)]
    assert [(w["handle"], w["s"]) for w in xf.plan_fill(ACCOUNTS, since=M0)["windows"]] == [
        ("Aa", M0),
        ("Tt", M0),
    ]
    assert len(xf.plan_fill(ACCOUNTS, until=M0)["windows"]) == 2
    assert len(xf.plan_fill(ACCOUNTS, limit=3)["windows"]) == 3


def test_full_windows_and_windows_x_counts_no_more_in_are_left(env):
    # Aa's collect window holds 5 (its cap) of 8; Tt's holds all 4 X counts,
    # a duplicate line counted once, and a post exactly at M0 belongs to it.
    tt_x = [*TT_X[:3], (_pid(M0, 2), M0, "edge"), *TT_X[3:]]
    _base(aa_held=AA_X[-5:], tt_held=[*tt_x[3:], tt_x[4]], tt_x=tt_x)
    plan = xf.plan_fill(ACCOUNTS)
    assert [(w["handle"], w["s"]) for w in plan["windows"]] == [("Aa", F), ("Tt", F)]
    acct = plan["accts"]["Tt"]
    assert acct.held_in(F, M0) == 0 and acct.held_in(M0, M1) == 5


def test_estimate_makes_no_call_and_writes_nothing(env, monkeypatch):
    _base()
    monkeypatch.setattr(xcn, "get_client", _no_client)
    monkeypatch.setattr(xc, "get_client", _no_client)
    monkeypatch.setattr(xc.tweepy, "Client", _no_client)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    before = sorted(p.name for p in env.path.rglob("*") if p.is_file())
    cache = xc._account_cache_path("Aa").read_bytes()
    for args in (["--estimate"], ["--estimate", "--scope", "empty"], ["--estimate", "--force"]):
        res = _cli(["x-archive", "fill", *args])
        assert res.exit_code == 0, res.output
    assert "Expected reads: 14 ≈ $0.07" in _cli(["x-archive", "fill", "--estimate"]).output
    out = res.output
    assert "Ceiling (each window's need + 9): 50" in out and "Run maximum: 50 posts" in out
    assert "Requests ≈ 5" in out and "TOTAL" in out
    assert sorted(p.name for p in env.path.rglob("*") if p.is_file()) == before
    assert xc._account_cache_path("Aa").read_bytes() == cache


def test_missing_counts_refuse_with_the_paid_step_named(env, monkeypatch):
    _base()
    xcn.cache_path("from:Tt", "hour").unlink()
    with pytest.raises(xf.CountsMissing, match="x-counts --only Tt --estimate"):
        xf.plan_fill(ACCOUNTS)
    res = _cli(["x-archive", "fill", "--estimate"])
    assert res.exit_code == 1 and "@Tt" in res.output


# ── a run ──────────────────────────────────────────────────────────


def test_run_fills_the_windows_bounded_by_held_ids(env):
    stub = _base()
    stub.data = {  # a long post (note_tweet) and a retweet (refs)
        AA_X[11][0]: {"note_tweet": {"text": "the whole long post " + "x" * 300}},
        AA_X[10][0]: {"referenced_tweets": [{"type": "retweeted", "id": "77"}]},
    }
    before = xc.plan_account("Aa", start=F, end=_t("2026-03-10"))["window"][0]
    res = xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert res["error"] is None and len(res["done"]) == 4 and not res["stopped"]
    assert res["read"] == 14 and res["kept"] == 14 and res["extra"] == 0 and res["reread"] == 0
    assert res["requests"] == 5 == len(stub.calls)
    # Aa: the newest 5 of the backward window, the newest 3 it lacked in the collect window
    assert set(_ids("Aa")) == {str(p[0]) for p in AA_X[7:12] + AA_X[15:]}
    # Tt's collect window: the gap above its held post by since_id, the one below by until_id
    held = str(TT_HELD[0][0])
    tt1 = [c for c in stub.calls if c["query"] == "from:Tt" and _upper(c) > xcn.z(M0)]
    assert [_lower(tt1[0]), _upper(tt1[1])] == ["2026-03-02T20:30:00Z"] * 2
    # ids only: the held post's, and the window's edges as the ids they map to
    assert [(c["since_id"], c["until_id"]) for c in tt1] == [
        (held, str(xf.id_at(M1))),
        (str(xf.id_at(M0) - 1), held),
    ]
    assert not any("start_time" in c or "end_time" in c for c in stub.calls)
    # records: the collector's fields, the whole text, refs, via
    recs = {r["id"]: r for r in xc._load_jsonl(xc._account_cache_path("Aa"))}
    long = recs[str(AA_X[11][0])]
    assert long["text"].startswith("the whole long post") and long["via"] == "search_all"
    assert recs[str(AA_X[10][0])]["ref"] == [{"type": "retweeted", "id": "77"}]
    assert {"id", "user", "tier", "text", "created_at", "metrics", "lang", "platform"} <= set(long)
    # older posts appended out of order never move collect's cursor; none at or after M1
    after = xc.plan_account("Aa", start=F, end=_t("2026-03-10"))["window"][0]
    assert after == before
    filled = [r for r in recs.values() if r.get("via")]
    assert filled and all(xcn.parse_ts(r["created_at"]) < M1 for r in filled)
    # state, manifest rows (got = what the window holds after), the log
    st = json.loads(xf.state_path("Aa").read_text())["windows"]
    assert {k: v["status"] for k, v in st.items()} == {xcn.z(F): "done", xcn.z(M0): "done"}
    rows = [r for p in settings.X_SAMPLING_DIR.glob("windows_archive_*.jsonl") for r in _read(p)]
    assert {(r["handle"], r["window_start"], r["got"], r["source"]) for r in rows} == {
        ("Aa", xcn.z(F), 5, "archive"),
        ("Aa", xcn.z(M0), 5, "archive"),
        ("Tt", xcn.z(F), 3, "archive"),
        ("Tt", xcn.z(M0), 4, "archive"),
    }
    assert sum(line["returned"] for line in _log()) == 14 == xf.billed_so_far()["posts"]
    # a rerun plans nothing and asks nothing
    assert xf.plan_fill(ACCOUNTS)["windows"] == []
    # the report shows the fill's rows with their source, counted apart
    lines = "\n".join(xcn.report_lines(ACCOUNTS, None, F, T, "hour"))
    assert "1 archive windows" not in lines and "2 archive windows, 1 collect windows" in lines
    assert "@Aa archive" in lines and "@Tt archive" in lines
    assert xa.complete_until("Tt", F, T) == T  # take-all rows read whole
    assert env.sleeps == [settings.X_ARCHIVE_FILL_DELAY_S] * 4  # paced between requests


def test_narrowed_walk_reads_the_crossing_hour_in_slices_newest_first(env, monkeypatch):
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"Nn": 5})
    times = [
        _t(f"2026-03-03T{h}")
        for h in ("10:05", "10:20", "10:35", "10:50", "20:10", "20:20", "20:50")
    ]
    nn = _posts(times, 3)
    _jsonl(xc._account_cache_path("Nn"), [])
    _counts("Nn", nn)
    _jsonl(settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl", [_mrow("Nn", M0, M1, 5, 0)])
    stub = SearchStub({"Nn": nn})
    res = xf.run_fill(stub, xf.plan_fill({"t": ["Nn"]}))
    assert [(_lower(c)[11:16], _upper(c)[11:16]) for c in stub.calls] == [
        ("11:00", "22:00"),  # the whole hours after the crossing hour
        ("10:45", "11:00"),  # then the crossing hour, newest slice first
        ("10:30", "10:45"),
    ]
    assert res["read"] == 5 and res["extra"] == 0
    assert set(_ids("Nn")) == {str(p[0]) for p in nn[2:]}  # the newest 5


def test_whole_hours_keep_the_newest_and_set_the_rest_aside_then_promote_them(env, monkeypatch):
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"Nn": 5})
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_SLICE_MINUTES", 0)
    times = [
        _t(f"2026-03-03T{h}")
        for h in ("10:05", "10:20", "10:35", "10:50", "20:10", "20:20", "20:50")
    ]
    nn = _posts(times, 3)
    _jsonl(xc._account_cache_path("Nn"), [])
    _counts("Nn", nn)
    _jsonl(settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl", [_mrow("Nn", M0, M1, 5, 0)])
    stub = SearchStub({"Nn": nn})
    res = xf.run_fill(stub, xf.plan_fill({"t": ["Nn"]}))
    assert [(_lower(c)[11:16], _upper(c)[11:16]) for c in stub.calls] == [("10:00", "22:00")]
    assert res["read"] == 7 and res["kept"] == 5 and res["extra"] == 2
    assert set(_ids("Nn")) == {str(p[0]) for p in nn[2:]}  # the window holds its cap
    extras = _read(xf.extras_path("Nn"))
    assert [r["id"] for r in extras] == [str(nn[1][0]), str(nn[0][0])]
    assert all(r["fill_window"] == xcn.z(M0) and r["via"] == "search_all" for r in extras)
    # the extras stay out of everything that reads the cache, and are billed
    assert not {r["id"] for r in extras} & {p["id"] for p in xc.load_all_cached()}
    assert not {r["id"] for r in extras} & {c["id"] for c in xb.candidates()}
    assert xf.billed_so_far()["posts"] == 7
    # Nn turns take-all: the window wants X's 7, and takes the 2 set aside for free
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {})
    plan = xf.plan_fill({"t": ["Nn"]}, force=True)
    res = xf.run_fill(stub, plan)
    assert len(stub.calls) == 1 and res["read"] == 0 and res["promoted"] == 2
    recs = xc._load_jsonl(xc._account_cache_path("Nn"))
    assert len(recs) == 7 and not any("fill_window" in r for r in recs)
    assert list(env.path.glob("x_archive/fill_Nn.*.json"))  # the old state set aside


def test_take_all_counts_low_by_the_slack_complete_and_lower_stops_then_resumes(env):
    # X counts 1 post in Tt's backward window; search/all has 5 there, then 15
    xs = _posts([F + 2 * i * H for i in range(15)], 2)
    for n, status in ((5, "done"), (15, "stopped")):
        _jsonl(xc._account_cache_path("Tt"), [])
        if xf.state_path("Tt").exists():
            xf.state_path("Tt").unlink()
        _counts("Tt", xs[:1])
        _jsonl(
            settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl",
            [_mrow("Tt", M0, M1, 500, 0, mode="take_all")],
        )
        stub = SearchStub({"Tt": xs[:n]})
        res = xf.run_fill(stub, xf.plan_fill({"t": ["Tt"]}))
        row = (res["done"] + res["stopped"])[0]
        assert row["status"] == status and row["read"] == min(n, 10) <= row["budget"] == 10
    assert row["stop"] == "window budget reached"
    # the next run plans the stopped window again and reads the rest, none twice
    res = xf.run_fill(stub, xf.plan_fill({"t": ["Tt"]}))
    assert res["done"] and res["read"] == 5 and len(set(_ids("Tt"))) == 15 == len(_ids("Tt"))


def test_spend_never_passes_a_window_budget_or_the_run_maximum(env, monkeypatch):
    stub = _base()
    stub.adversarial = True  # every answer is max_results posts, with more to come
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_MAX_USD", 0.105)  # 21 posts
    plan = xf.plan_fill(ACCOUNTS)
    assert plan["run_max"] == 21
    res = xf.run_fill(stub, plan)
    budgets = {(w["handle"], xcn.z(w["s"])): w["budget"] for w in plan["windows"]}
    run_read, by_window = 0, {}
    for line in _log():
        key = (line["handle"], line["window_start"])
        before = by_window.get(key, 0)
        assert 10 <= line["max_results"] <= min(budgets[key] - before, 21 - run_read)
        by_window[key] = before + line["returned"]
        run_read += line["returned"]
    assert run_read == res["read"] == 21  # Aa 10 (5 kept, 5 set aside), Tt 11 by the run cap
    assert all(n <= budgets[k] for k, n in by_window.items())
    assert res["stop"] and "run maximum" in res["stop"] and res["left"] == 2
    assert [(r["handle"], r["read"]) for r in res["stopped"]] == [("Tt", 11)]
    assert xf.billed_so_far()["posts"] == 21 and xf.plan_fill(ACCOUNTS)["refusal"]


@pytest.mark.parametrize(
    "err, name",
    [
        (_http_error(tweepy.errors.Forbidden, 403, ENROLL), "403"),
        (_http_error(tweepy.errors.Unauthorized, 401, UNAUTHORIZED), "401"),
        (_http_error(tweepy.errors.TooManyRequests, 429, USAGE_CAPPED), "UsageCapExceeded"),
        (tweepy.errors.TweepyException("network"), "network"),
    ],
)
def test_an_error_stops_the_run_with_no_further_request_and_a_rerun_resumes(env, err, name):
    stub = _base()
    stub.fail = {2: err}  # Tt's backward window
    res = xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert res["error"] and name in res["error"] and stub.attempts == 2 and len(stub.calls) == 1
    assert [r["handle"] for r in res["done"]] == ["Aa"] and res["stopped"][0]["handle"] == "Tt"
    assert json.loads(xf.state_path("Tt").read_text())["windows"][xcn.z(F)]["status"] == "stopped"
    assert env.sleeps == [settings.X_ARCHIVE_FILL_DELAY_S]  # the pacing, no 429 wait
    # an X error status served no posts: its pending line is closed at 0; a
    # failure with no answer from X stays counted at its max_results (12)
    b = xf.billed_so_far()
    assert (b["unanswered"], b["posts"]) == ((1, 5 + 12) if name == "network" else (0, 5))
    stub.fail = {}
    res = xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert res["error"] is None and len(res["done"]) == 3  # Aa's backward window is done
    ids = [i for line in _log() for i in line["ids"]]
    assert len(ids) == len(set(ids)) == 14


def test_a_killed_run_resumes_from_the_cache_without_rebuying(env):
    class Killed(BaseException):
        pass

    stub = _base()
    stub.fail = {3: Killed()}  # Aa's collect window, after both backward windows
    with pytest.raises(Killed):
        xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert xcn.z(M0) not in json.loads(xf.state_path("Aa").read_text())["windows"]
    plan = xf.plan_fill(ACCOUNTS)
    assert [(w["handle"], w["s"]) for w in plan["windows"]] == [("Aa", M0), ("Tt", M0)]
    stub.fail = {}
    res = xf.run_fill(stub, plan)
    assert res["error"] is None and res["read"] == 6
    ids = [i for line in _log() for i in line["ids"]]
    assert len(ids) == len(set(ids)) == 14


def test_rate_limit_429_waits_per_header(env):
    stub = _base()
    stub.limited = 2
    res = xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert res["error"] is None and stub.attempts == 7
    waits = [s for s in env.sleeps if s != settings.X_ARCHIVE_FILL_DELAY_S]
    assert len(waits) == 2 and all(5 <= s <= 9 for s in waits)


def test_rate_limit_429_past_the_retries_stops_the_run(env, monkeypatch):
    monkeypatch.setattr(settings, "X_429_MAX_RETRIES", 1)
    stub = _base()
    stub.limited = 5
    res = xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert res["error"] and "TooManyRequests" in res["error"] and stub.attempts == 2


def test_a_done_window_is_not_walked_again_unless_forced(env):
    # search/all no longer has one of Tt's posts (deleted): the window is done short
    stub = _base()
    stub.posts["Tt"] = [p for p in TT_X if p != TT_X[6]]
    xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    n = len(stub.calls)
    plan = xf.plan_fill(ACCOUNTS)
    assert plan["windows"] == []  # Tt's collect window is still under X's count, but done
    plan = xf.plan_fill(ACCOUNTS, force=True)
    assert [(w["handle"], w["s"]) for w in plan["windows"]] == [("Tt", M0)]
    res = xf.run_fill(stub, plan)
    assert len(stub.calls) > n and res["read"] == 0 and res["done"]
    # the old state is kept aside; the windows not planned keep their mark
    assert len(list(env.path.glob("x_archive/fill_Tt.*.json"))) == 1
    st = json.loads(xf.state_path("Tt").read_text())["windows"]
    assert st[xcn.z(F)]["status"] == "done" and st[xcn.z(M0)]["run"] == res["run"]


def test_billed_reconciles_a_stored_post_with_no_log_line(env, monkeypatch):
    _base()
    _jsonl(
        xf.log_path(),
        [{"returned": 7, "ids": [str(p[0]) for p in AA_X[:7]]}, {"returned": 3, "ids": []}],
    )
    recs = [_rec(p, "Aa") for p in AA_HELD] + [_rec(AA_X[0], "Aa", via="search_all")]
    recs.append(_rec(AA_X[12], "Aa", via="search_all"))  # stored, then killed before the log
    _jsonl(xc._account_cache_path("Aa"), recs)
    _jsonl(xf.extras_path("Tt"), [_rec(TT_X[0], "Tt", via="search_all")])
    b = xf.billed_so_far()
    assert (b["posts"], b["requests"], b["unlogged"]) == (12, 2, 2)
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_MAX_USD", 0.10)  # 20 posts: 8 left
    plan = xf.plan_fill(ACCOUNTS)
    assert plan["cap_left"] == 8 and "used up" in plan["refusal"]


def test_a_request_whose_answer_is_lost_is_tallied_at_its_max_results(env):
    # X serves the first page (billed), then the connection drops before the
    # answer arrives: nothing is stored, but the pending line counts it
    stub = _base()
    real, sent = stub.search_all_tweets, []

    def lossy(query, **p):
        resp = real(query, **p)
        sent.append(p["max_results"])
        if len(sent) == 1:
            raise requests.exceptions.ChunkedEncodingError("connection reset mid-body")
        return resp

    stub.search_all_tweets = lossy
    left = xf.plan_fill(ACCOUNTS)["cap_left"]
    res = xf.run_fill(stub, xf.plan_fill(ACCOUNTS))
    assert res["error"] and "ChunkedEncodingError" in res["error"] and sent == [10]
    (line,) = _read(xf.log_path())
    assert line["pending"] and line["max_results"] == 10 and _log() == []
    b = xf.billed_so_far()
    assert (b["posts"], b["requests"], b["unanswered"]) == (10, 1, 1)
    plan = xf.plan_fill(ACCOUNTS)
    assert plan["cap_left"] == left - 10
    assert "1 sent with no answer logged" in "\n".join(xf.estimate_lines(plan))
    # the rerun resumes; the lost request stays counted at its max_results
    res = xf.run_fill(stub, plan)
    assert res["error"] is None and res["read"] == 14
    assert xf.billed_so_far()["posts"] == 24


def test_a_cut_line_costs_only_itself(env, monkeypatch, tmp_path):
    p = tmp_path / "cut.jsonl"
    p.write_text('{"a": 1}\n{"run": "x", "retu')  # a write cut by a power-off
    xc._append_jsonl(p, [{"b": 2}])
    assert xf._load_lines(p) == [{"a": 1}, {"b": 2}]
    assert p.read_text().splitlines()[1:] == ['{"run": "x", "retu', '{"b": 2}']
    # in a run: the request log and the extras file both end in a cut line
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"Nn": 5})
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_SLICE_MINUTES", 0)
    times = [
        _t(f"2026-03-03T{h}")
        for h in ("10:05", "10:20", "10:35", "10:50", "20:10", "20:20", "20:50")
    ]
    nn = _posts(times, 3)
    _jsonl(xc._account_cache_path("Nn"), [])
    _counts("Nn", nn)
    _jsonl(settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl", [_mrow("Nn", M0, M1, 5, 0)])
    xf.log_path().parent.mkdir(parents=True, exist_ok=True)
    xf.log_path().write_text('{"run": "old", "returned": 7, "ids": ["1"')
    xf.extras_path("Nn").write_text('{"id": "5", "created_at": "2026-03-0')
    res = xf.run_fill(SearchStub({"Nn": nn}), xf.plan_fill({"t": ["Nn"]}))
    assert res["read"] == 7 and res["extra"] == 2
    assert [bool(x.get("pending")) for x in xf._load_lines(xf.log_path())] == [True, False]
    assert [r["id"] for r in xf._load_lines(xf.extras_path("Nn"))] == [
        str(nn[1][0]),
        str(nn[0][0]),
    ]
    assert xf.billed_so_far()["posts"] == 7


@pytest.mark.parametrize("handle, mode, cap", [("Tt", "take_all", 500), ("Nn", "sample", 5)])
def test_a_gap_whose_hours_another_gap_filled_is_not_read(env, handle, mode, cap):
    # X has 14:05, 14:20 and 14:40; the cache holds 14:05 and 14:40. All three
    # gaps touch the 14:00 bucket: once the middle gap's read stores 14:20,
    # the oldest gap has nothing left to find (the budget, 1 + 9, is no
    # longer enough for a request, so reading it would stop the window).
    xs = _posts([_t("2026-03-02T14:05"), _t("2026-03-02T14:20"), _t("2026-03-02T14:40")], 2)
    _jsonl(xc._account_cache_path(handle), [_rec(p, handle) for p in (xs[0], xs[2])])
    _counts(handle, xs)
    _jsonl(
        settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl",
        [_mrow(handle, M0, M1, cap, 2, mode=mode)],
    )
    stub = SearchStub({handle: xs})
    plan = xf.plan_fill({"t": [handle]})
    (wp,) = plan["windows"]
    assert (wp["x_hi"], wp["held"], wp["need_hi"], wp["budget"]) == (3, 2, 1, 10)
    res = xf.run_fill(stub, plan)
    assert not res["stopped"]
    (row,) = res["done"]
    assert (row["status"], row["got"], row["truncated"], row["requests"]) == ("done", 3, False, 2)
    since, until = str(xs[0][0]), str(xs[2][0])
    assert [(c["since_id"], c["until_id"]) for c in stub.calls] == [
        (until, str(xf.id_at(M1))),  # newest gap first, above 14:40: nothing
        (since, until),  # the middle gap, between the held posts: 14:20
    ]
    assert xf.plan_fill({"t": [handle]})["windows"] == []


# ── CLI ────────────────────────────────────────────────────────────


def test_cli_refusals_before_any_call(env, monkeypatch):
    _base()
    monkeypatch.setattr(xcn, "get_client", _no_client)
    res = _cli(["x-archive", "fill", "--yes", "--only", ""])
    assert res.exit_code == 2 and "no handles given" in res.output
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 0.1)  # the run maximum is $0.25
    res = _cli(["x-archive", "fill", "--yes"])
    assert res.exit_code == 1 and "exceeds X_RUN_BUDGET_USD" in res.output, res.output
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 45.0)
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_MAX_USD", 0.04)  # 8 posts
    res = _cli(["x-archive", "fill", "--yes"])
    assert res.exit_code == 1 and "X_ARCHIVE_FILL_MAX_USD $0.04 is used up" in res.output
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_MAX_USD", 5.0)
    monkeypatch.setattr(settings, "X_ARCHIVE_FILL_MAX_REQUESTS", 4)
    res = _cli(["x-archive", "fill", "--yes"])
    assert res.exit_code == 1 and "over X_ARCHIVE_FILL_MAX_REQUESTS" in res.output
    res = _cli(["x-archive", "fill", "--yes", "--since", "2026-03-03", "--until", "2026-03-02"])
    assert res.exit_code == 2
    assert not xf.log_path().exists()


def test_cli_confirm_lock_only_since_and_limit(env, monkeypatch):
    stub = _base()
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-archive", "fill"], input="n\n")
    assert "Aborted." in res.output and stub.calls == []
    with xc.cache_lock():
        res = _cli(["x-archive", "fill", "--yes"])
        assert res.exit_code == 1 and "Refusing" in res.output and stub.calls == []
        assert _cli(["x-archive", "fill", "--estimate"]).exit_code == 0  # an estimate only reads
    stub.fail = {1: _http_error(tweepy.errors.Forbidden, 403, ENROLL)}
    res = _cli(["x-archive", "fill", "--yes", "--only", "Aa"])
    assert res.exit_code == 1 and "answered 403" in res.output and stub.calls == []
    assert "No further requests were made" in res.output
    stub.fail = {}
    res = _cli(
        ["x-archive", "fill", "--yes", "--only", "tt", "--since", "2026-03-02", "--limit", "1"]
    )
    assert res.exit_code == 0, res.output
    assert "1 windows done" in res.output and len(stub.calls) == 2
    assert {c["query"] for c in stub.calls} == {"from:Tt"} and all(
        _upper(c) > xcn.z(M0) for c in stub.calls
    )
    res = _cli(["x-archive", "fill", "--yes"])
    assert res.exit_code == 0 and "3 windows done" in res.output, res.output
    assert len(stub.calls) == 5


def test_an_edge_bucket_the_neighbour_holds_costs_no_request(env):
    # Tt's collect window starts at 11:30; X counts 1 post in the 11:00 bucket,
    # and the cache holds it (11:10, in the window before). The window is
    # picked (edge buckets count whole), but no gap can hold a missing post.
    s, e = _t("2026-03-02T11:30"), _t("2026-03-03T22:30")
    tt = _posts([_t("2026-03-02T11:10")], 2)
    _jsonl(xc._account_cache_path("Tt"), [_rec(tt[0], "Tt")])
    _counts("Tt", tt)
    _jsonl(
        settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl",
        [_mrow("Tt", s, e, 500, 0, "take_all")],
    )
    plan = xf.plan_fill({"t": ["Tt"]})
    (wp,) = plan["windows"]
    assert (wp["s"], wp["x_lo"], wp["x_hi"], wp["held"], wp["kind"]) == (s, 0, 1, 0, "empty")
    assert wp["exp_slice"] == 0 and wp["req"] == 0  # evenly spread, half a post would be expected
    stub = SearchStub({"Tt": tt})
    res = xf.run_fill(stub, plan)
    assert stub.calls == [] and [r["status"] for r in res["done"]] == ["done"]
