"""
x-counts (GET /2/tweets/counts/all) and x-archive probe (GET
/2/tweets/search/all), tested against a stub tweepy client: no network, no
spend.
"""

import json
import random
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import tweepy
from click.testing import CliRunner

from config import settings
from src.collectors import x_archive as xa
from src.collectors import x_collector as xc
from src.collectors import x_counts as xcn

H = timedelta(hours=1)
D = timedelta(days=1)


def _t(ts: str) -> datetime:
    """Naive UTC, as the collector and the counts cache use."""
    return datetime.fromisoformat(ts)


def _ms(dt: datetime) -> str:
    """X's bucket time format."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _http_error(cls, status: int, body: dict, headers: dict | None = None):
    resp = SimpleNamespace(
        status_code=status,
        reason={401: "Unauthorized", 403: "Forbidden", 429: "Too Many Requests"}[status],
        headers=headers or {},
        json=lambda: body,
    )
    return cls(resp)


ENROLL = {
    "title": "Client Forbidden",
    "reason": "client-not-enrolled",
    "detail": "This request must be made using an approved developer account.",
}
UNAUTHORIZED = {"title": "Unauthorized", "type": "about:blank", "status": 401}
USAGE_CAPPED = {
    "title": "UsageCapExceeded",
    "detail": "Usage cap exceeded: Monthly product cap",
    "type": "https://api.x.com/2/problems/usage-capped",
}


class Stub:
    """
    counts/all: serves `posts[query]` (naive-UTC datetimes) as zero-filled
    buckets, one page per `page_days` (31, as X documents), pages newest
    first unless `oldest_first`, next_token = the page index. search/all:
    `tweets` [(id, created_at, text)] in [start, end), newest first, up to
    max_results; `drop_rt` leaves out "RT @" posts, `since` posts before it,
    `drop_ids` those ids. `short` answers counts/all with its first page
    only and no next_token; `omit_zero` sends no row for an empty bucket.
    `deny` (a set of "counts"/"search") answers 403 client-not-enrolled,
    `unauth` 401; `usage_capped` answers that many usage-capped 429s first,
    `limited` that many rate-limit 429s; `fail_on` raises a generic error
    on that call number. Records every call and every sleep in `events`.
    """

    def __init__(
        self,
        posts=None,
        tweets=(),
        page_days=31,
        oldest_first=False,
        deny=(),
        limited=0,
        fail_on=None,
        drop_rt=False,
        since=None,
        count_key="tweet_count",
        unauth=(),
        usage_capped=0,
        short=False,
        omit_zero=False,
        drop_ids=(),
    ):
        self.posts = posts or {}
        self.unauth, self.usage_capped = set(unauth), usage_capped
        self.short, self.omit_zero, self.drop_ids = short, omit_zero, set(drop_ids)
        self.tweets = sorted(tweets, key=lambda t: t[1], reverse=True)
        self.page_days, self.oldest_first = page_days, oldest_first
        self.deny, self.limited, self.fail_on = set(deny), limited, fail_on
        self.drop_rt, self.since, self.count_key = drop_rt, since, count_key
        self.events: list[tuple] = []
        self.attempts = 0

    @property
    def calls(self):
        return [e for e in self.events if e[0] in ("counts", "search")]

    def _gate(self, kind: str):
        self.attempts += 1
        if kind in self.unauth:
            raise _http_error(tweepy.errors.Unauthorized, 401, UNAUTHORIZED)
        if kind in self.deny:
            raise _http_error(tweepy.errors.Forbidden, 403, ENROLL)
        if self.usage_capped:
            self.usage_capped -= 1
            raise _http_error(tweepy.errors.TooManyRequests, 429, USAGE_CAPPED)
        if self.limited:
            self.limited -= 1
            reset = str(int(time.time()) + 7)
            raise _http_error(
                tweepy.errors.TooManyRequests,
                429,
                {"title": "Too Many Requests"},
                {"x-rate-limit-reset": reset},
            )
        if self.attempts == self.fail_on:
            raise tweepy.errors.TweepyException(f"stub failure on attempt {self.attempts}")

    def get_all_tweets_count(self, query, start_time, end_time, granularity, next_token=None):
        self._gate("counts")
        self.events.append(("counts", query, start_time, end_time, granularity, next_token))
        s, e = xcn.parse_ts(start_time), xcn.parse_ts(end_time)
        step, page = xcn.GRANULARITIES[granularity], timedelta(days=self.page_days)
        chunks = []
        if self.oldest_first:
            t = s
            while t < e:
                chunks.append((t, min(e, t + page)))
                t += page
        else:
            t = e
            while t > s:
                chunks.append((max(s, t - page), t))
                t -= page
        i = int(next_token or 0)
        a, b = chunks[i]
        times = self.posts.get(query, [])
        data, t = [], a
        while t < b:
            n = sum(1 for p in times if t <= p < t + step)
            if n or not self.omit_zero:
                data.append({"start": _ms(t), "end": _ms(t + step), self.count_key: n})
            t += step
        more = i + 1 < len(chunks) and not self.short
        return SimpleNamespace(data=data, meta={"next_token": str(i + 1)} if more else {})

    def search_all_tweets(self, query, start_time, end_time, max_results, tweet_fields):
        self._gate("search")
        self.events.append(("search", query, start_time, end_time, max_results))
        assert 10 <= max_results <= 500 and tweet_fields == xc._TWEET_FIELDS
        s, e = xcn.parse_ts(start_time), xcn.parse_ts(end_time)
        hits = [
            t
            for t in self.tweets
            if s <= t[1] < e
            and not (self.drop_rt and t[2].startswith("RT @"))
            and not (self.since and t[1] < self.since)
            and t[0] not in self.drop_ids
        ]
        page = hits[:max_results]
        data = [
            SimpleNamespace(
                id=i,
                text=text,
                created_at=ts.replace(tzinfo=UTC),
                public_metrics={"like_count": 1},
                lang="en",
                data={"id": str(i), "text": text},
            )
            for i, ts, text in page
        ]
        meta = {"next_token": "more"} if len(hits) > max_results else {}
        return SimpleNamespace(data=data, meta=meta)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Every data path in tmp_path; sleeps recorded, the clock frozen."""
    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "FIGURES_DIR", "TRUTH_SOCIAL_RAW_DIR"):
        monkeypatch.setattr(settings, name, tmp_path / name.lower())
    monkeypatch.setattr(settings, "X_RAW_DIR", tmp_path / "x")
    monkeypatch.setattr(settings, "X_SAMPLING_DIR", tmp_path / "x_sampling")
    monkeypatch.setattr(settings, "X_COUNTS_DIR", tmp_path / "x_counts")
    monkeypatch.setattr(settings, "X_ARCHIVE_DIR", tmp_path / "x_archive")
    sleeps: list[float] = []
    monkeypatch.setattr(xcn, "_sleep", sleeps.append)
    monkeypatch.setattr(xcn, "_clock", lambda: 0.0)
    return SimpleNamespace(path=tmp_path, sleeps=sleeps)


def _no_client(*a, **k):
    raise AssertionError("no client may be built on this path")


def _cli(args, input=None):
    from src import cli

    return CliRunner().invoke(cli.main, args, input=input)


def _accounts(monkeypatch, accounts):
    from src import cli

    monkeypatch.setattr(cli, "X_ACCOUNTS", accounts)


def _every(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    out, t = [], start
    while t < end:
        out.append(t)
        t += step
    return out


FEB, MAR5 = _t("2026-02-01"), _t("2026-03-05")  # 32 days: 2 requests


# ── estimate, budget, caps ─────────────────────────────────────────


def test_estimate_makes_no_call_and_needs_no_credentials(data, monkeypatch):
    _accounts(monkeypatch, {"t1": ["Aa", "Bb"], "t2": ["Cc"]})
    monkeypatch.setattr(xcn, "get_client", _no_client)
    monkeypatch.setattr(xc, "get_client", _no_client)
    monkeypatch.setattr(xc.tweepy, "Client", _no_client)
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    res = _cli(["x-counts", "--estimate", "--start", "2026-02-01", "--end", "2026-03-05"])
    assert res.exit_code == 0, res.output
    for q in ("from:Aa", "from:Bb", "from:Cc"):
        assert q in res.output
    assert "Planned: 6 requests ≈ $0.06" in res.output
    assert "Maximum 9" in res.output  # one slack request per query
    assert not settings.X_COUNTS_DIR.exists()

    res = _cli(
        [
            "x-counts",
            "--estimate",
            "--no-accounts",
            "--terms",
            "--query",
            "from:Aa is:retweet",
            "--start",
            "2026-02-01",
            "--end",
            "2026-03-05",
        ]
    )
    assert res.exit_code == 0, res.output
    assert "from:Aa is:retweet" in res.output and "from:Bb" not in res.output
    assert "Planned: 10 requests" in res.output  # 4 terms + the query, 2 each


def test_pages_are_31_days_per_request():
    assert xcn.pages_for(FEB, FEB + 31 * D) == 1
    assert xcn.pages_for(FEB, FEB + 31 * D + H) == 2
    assert xcn.pages_for(FEB, FEB + H) == 1
    assert xcn.pages_for(_t("2026-02-01"), _t("2026-11-01T23:00")) == 9


def test_budget_and_request_cap_refuse_before_any_call(data, monkeypatch):
    _accounts(monkeypatch, {"t": ["Aa", "Bb"]})
    monkeypatch.setattr(xcn, "get_client", _no_client)
    args = ["x-counts", "--yes", "--start", "2026-02-01", "--end", "2026-03-05"]
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 0.05)  # maximum is 6 x $0.010
    res = _cli(args)
    assert res.exit_code == 1 and "exceeds X_RUN_BUDGET_USD" in res.output, res.output
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 45.0)
    monkeypatch.setattr(settings, "X_COUNTS_MAX_REQUESTS", 3)
    res = _cli(args)
    assert res.exit_code == 1 and "over X_COUNTS_MAX_REQUESTS" in res.output, res.output
    assert not settings.X_COUNTS_DIR.exists()


def test_confirmation_and_cache_lock(data, monkeypatch):
    _accounts(monkeypatch, {"t": ["Aa"]})
    stub = Stub()
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    args = ["x-counts", "--start", "2026-02-01", "--end", "2026-03-05"]
    res = _cli(args, input="n\n")
    assert "Aborted." in res.output and stub.calls == []
    with xc.cache_lock():
        res = _cli([*args, "--yes"])
        assert res.exit_code == 1 and "x-counts" in res.output and stub.calls == []
        res = _cli(["x-archive", "probe", "--yes"])
        assert res.exit_code == 1 and stub.calls == []
        assert _cli([*args, "--estimate"]).exit_code == 0  # an estimate only reads
    res = _cli([*args, "--yes"])
    assert res.exit_code == 0, res.output
    assert len(stub.calls) == 2 and "2 requests ≈ $0.02" in res.output


def test_only_parsing_matches_collect(data, monkeypatch):
    _accounts(monkeypatch, {"t1": ["PeteHegseth", "VP"], "t2": ["Pontifex"]})
    res = _cli(["x-counts", "--estimate", "--only", " @petehegseth, PONTIFEX "])
    assert res.exit_code == 0, res.output
    assert "from:PeteHegseth" in res.output and "from:Pontifex" in res.output
    assert "from:VP" not in res.output
    for empty in ("", " , ", "@"):
        res = _cli(["x-counts", "--estimate", "--only", empty])
        assert res.exit_code == 2 and "no handles" in res.output and "Planned" not in res.output
    res = _cli(["x-counts", "--estimate", "--only", "nosuch"])
    assert res.exit_code == 2 and "nosuch" in res.output
    res = _cli(["x-counts", "--report", "--force"])
    assert res.exit_code == 2 and "drop --force" in res.output
    res = _cli(["x-counts", "--estimate", "--no-accounts"])
    assert res.exit_code == 2 and "nothing to count" in res.output


def test_page_cap_stops_a_query_that_pages_more_finely(data):
    stub = Stub({"from:a": _every(FEB, MAR5, 5 * H)}, page_days=10)  # X pages every 10 days
    plan = xcn.plan_query("from:a", FEB, MAR5, "hour")
    assert (plan["pages"], plan["page_cap"]) == (2, 3)
    res = xcn.fetch_query(stub, plan, xcn.RequestCap(100, 1.0))
    assert res["requests"] == 3 and not res["complete"] and "page cap" in res["stopped"]
    # the three newest 10-day pages are cached; the next run plans only the rest
    left = xcn.plan_query("from:a", FEB, MAR5, "hour")
    assert left["todo"] == [(FEB, MAR5 - 30 * D)] and left["pages"] == 1


def test_run_cap_stops_the_run(data):
    stub = Stub({"from:a": []})
    plans = [xcn.plan_query(q, FEB, MAR5, "hour") for q in ("from:a", "from:b")]
    cap = xcn.RequestCap(3, 1.0)
    xcn.fetch_query(stub, plans[0], cap)
    with pytest.raises(xcn.RunCapReached):
        xcn.fetch_query(stub, plans[1], cap)
    assert len(stub.calls) == 3 and cap.used == 3
    assert data.sleeps == [1.0, 1.0]  # spaced; none before the first request


# ── cache: skip, partial range, resume ─────────────────────────────


@pytest.mark.parametrize("oldest_first", [False, True])
def test_covered_range_is_skipped_and_new_range_fetched_alone(data, oldest_first):
    posts = _every(FEB, _t("2026-04-01"), 7 * H)
    stub = Stub({"from:a": posts}, oldest_first=oldest_first, count_key="post_count")
    plan = xcn.plan_query("from:a", FEB, MAR5, "hour")
    res = xcn.fetch_query(stub, plan, xcn.RequestCap(10, 1.0))
    assert res["complete"] and res["requests"] == 2
    assert res["total"] == sum(1 for p in posts if p < MAR5)
    cache = json.loads(xcn.cache_path("from:a", "hour").read_text())
    assert cache["covered"] == [["2026-02-01T00:00:00Z", "2026-03-05T00:00:00Z"]]
    assert len(cache["buckets"]) == 32 * 24 and cache["fetches"][0]["complete"]

    again = xcn.plan_query("from:a", FEB, MAR5, "hour")
    assert again["pages"] == 0 and again["skip_reason"] == "cached (hour)"
    # end floored to the hour: a few minutes later plans nothing either
    assert xcn.plan_query("from:a", FEB, MAR5 + timedelta(minutes=50), "hour")["pages"] == 0

    later = xcn.plan_query("from:a", FEB, _t("2026-03-20T06:30"), "hour")
    assert later["todo"] == [(MAR5, _t("2026-03-20T06:00"))] and later["pages"] == 1
    n = len(stub.calls)
    xcn.fetch_query(stub, later, xcn.RequestCap(10, 1.0))
    assert [c[2:4] for c in stub.calls[n:]] == [("2026-03-05T00:00:00Z", "2026-03-20T06:00:00Z")]
    cache = xcn.CountsCache("from:a", "hour")
    assert cache.covered == [(FEB, _t("2026-03-20T06:00"))]
    assert cache.total(FEB, _t("2026-03-20T06:00"))[0] == sum(
        1 for p in posts if p < _t("2026-03-20T06:00")
    )

    # another granularity is its own file; --force refetches a covered range
    assert xcn.plan_query("from:a", FEB, MAR5, "day")["pages"] == 2
    assert xcn.plan_query("from:a", FEB, MAR5, "hour", force=True)["pages"] == 2


@pytest.mark.parametrize("oldest_first", [False, True])
def test_killed_run_resumes_with_what_is_left(data, oldest_first):
    posts = _every(FEB, MAR5, 3 * H)
    stub = Stub({"from:a": posts}, oldest_first=oldest_first, fail_on=2)
    plan = xcn.plan_query("from:a", FEB, MAR5, "hour")
    with pytest.raises(tweepy.errors.TweepyException):
        xcn.fetch_query(stub, plan, xcn.RequestCap(10, 1.0))
    first = (FEB, FEB + 31 * D) if oldest_first else (MAR5 - 31 * D, MAR5)
    assert xcn.CountsCache("from:a", "hour").covered == [first]  # page 1 was written
    left = xcn.plan_query("from:a", FEB, MAR5, "hour")
    rest = (FEB + 31 * D, MAR5) if oldest_first else (FEB, MAR5 - 31 * D)
    assert left["todo"] == [rest] and left["pages"] == 1
    stub.fail_on = None
    res = xcn.fetch_query(stub, left, xcn.RequestCap(10, 1.0))
    assert res["requests"] == 1 and res["total"] == len(posts)


# ── errors: 403, 429 ───────────────────────────────────────────────


def test_403_stops_the_run_with_no_further_request(data, monkeypatch):
    _accounts(monkeypatch, {"t": ["Aa", "Bb", "Cc"]})
    stub = Stub(deny={"counts"})
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-counts", "--yes", "--start", "2026-02-01", "--end", "2026-03-05"])
    assert res.exit_code == 1
    assert "GET /2/tweets/counts/all answered 403" in res.output
    assert "client-not-enrolled" in res.output and "No further requests were made" in res.output
    assert stub.attempts == 1 and "0 requests ≈ $0.00" in res.output
    assert not xcn.cache_path("from:Aa", "hour").exists()


def test_429_waits_per_reset_header_then_gives_up(data, monkeypatch):
    stub = Stub({"from:a": []}, limited=2)
    plan = xcn.plan_query("from:a", FEB, FEB + 10 * D, "hour")
    res = xcn.fetch_query(stub, plan, xcn.RequestCap(5, 1.0))
    assert res["complete"] and res["requests"] == 1 and stub.attempts == 3
    assert len(data.sleeps) == 2 and all(6 <= s <= 8 for s in data.sleeps)  # until the reset

    monkeypatch.setattr(settings, "X_429_MAX_RETRIES", 1)
    stub = Stub({"from:b": []}, limited=5)
    plan = xcn.plan_query("from:b", FEB, FEB + 10 * D, "hour")
    with pytest.raises(tweepy.errors.TooManyRequests):
        xcn.fetch_query(stub, plan, xcn.RequestCap(5, 1.0))
    assert stub.attempts == 2


def test_backoff_without_header_uses_default_and_is_clamped(monkeypatch):
    assert xcn._backoff_s(SimpleNamespace(headers={})) == settings.X_429_DEFAULT_BACKOFF_S
    far = SimpleNamespace(headers={"x-rate-limit-reset": str(int(time.time()) + 99_999)})
    assert xcn._backoff_s(far) == settings.X_429_MAX_BACKOFF_S
    past = SimpleNamespace(headers={"x-rate-limit-reset": str(int(time.time()) - 50)})
    assert xcn._backoff_s(past) == 1


# ── report ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("lo", "hi", "cap", "got", "want"),
    [
        (0, 0, 5, 0, "silent"),
        (0, 0, 5, 2, "complete"),  # counts may leave out retweets
        (3, 3, 5, 3, "complete"),
        (3, 3, 450, 3, "complete"),  # take-all: cap is the backstop left
        (12, 13, 5, 5, "sampled ok"),
        (12, 12, 5, 0, "FETCH SHORTFALL"),
        (12, 12, 5, 3, "FETCH SHORTFALL"),
        (2, 2, 5, 0, "FETCH SHORTFALL"),  # got nothing where X counts posts
        (4, 4, 5, 2, "short"),
        (3, 5, 5, 4, "edge"),  # inside the edge-bucket range
    ],
)
def test_verdicts(lo, hi, cap, got, want):
    assert xcn.verdict(lo, hi, cap, got) == want


def _rec(pid, ts: datetime, text: str, user="a") -> dict:
    return {
        "id": str(pid),
        "user": user,
        "tier": "t",
        "text": text,
        "created_at": ts.isoformat() + "+00:00",
        "platform": "x",
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_joins_counts_cache_and_manifest(data, monkeypatch):
    t0 = _t("2026-03-01")
    w = [(t0 + i * 10 * H, t0 + (i + 1) * 10 * H) for i in range(6)]
    # X's view: retweets left out of the count
    x_posts = (
        _every(w[1][0] + H, w[1][0] + 9 * H, H)  # 8: the run got nothing
        + _every(w[2][0] + H, w[2][0] + 9 * H, H)  # 8: sampled 5
        + _every(w[3][0] + H, w[3][0] + 4 * H, H)  # 3: complete
        + _every(w[4][0] + H, w[4][0] + 3 * H, H)  # 2 originals (+ 2 retweets held)
        + _every(w[5][0] + H, w[5][0] + 5 * H, H)  # 4: got 2
    )
    held = (
        [_rec(i, t, "post") for i, t in enumerate(x_posts[11:16])]  # w2: newest 5 of 8
        + [_rec(100 + i, t, "post") for i, t in enumerate(x_posts[16:19])]  # w3
        + [_rec(200 + i, t, "post") for i, t in enumerate(x_posts[19:21])]  # w4 originals
        + [_rec(300 + i, w[4][0] + (5 + i) * H, "RT @b: hi") for i in range(2)]  # w4 RTs
        + [_rec(400 + i, t, "post") for i, t in enumerate(x_posts[21:23])]  # w5: 2 of 4
    )
    held = list({r["id"]: r for r in held}.values())
    _write_jsonl(settings.X_RAW_DIR / "a.jsonl", held)
    got = [0, 0, 5, 3, 4, 2]
    _write_jsonl(
        settings.X_SAMPLING_DIR / "windows_2026-03-04.jsonl",
        [
            {
                "run_end": "2026-03-04T00:00:00Z",
                "handle": "a",
                "tier": "t",
                "mode": "sample",
                "hours": 10,
                "window_start": xcn.z(s),
                "window_end": xcn.z(e),
                "cap": 5,
                "got": g,
                "new": g,
                "truncated": g == 5,
            }
            for (s, e), g in zip(w, got, strict=True)
        ]
        + [
            {
                "handle": "zz",
                "window_start": xcn.z(t0),
                "window_end": xcn.z(t0 + H),
                "cap": 5,
                "got": 0,
                "mode": "sample",
            }
        ],
    )
    stub = Stub({"from:a": x_posts})
    xcn.fetch_query(stub, xcn.plan_query("from:a", t0, t0 + 3 * D, "hour"), xcn.RequestCap(9, 1))

    rows = xcn.window_rows(xcn.load_manifests(), {"a"}, "hour")
    assert [r["verdict"] for r in rows] == [
        "silent",
        "FETCH SHORTFALL",
        "sampled ok",
        "complete",
        "complete",
        "short",
    ]
    assert [r["lo"] for r in rows] == [0, 8, 8, 3, 2, 4]
    assert [r["over"] for r in rows] == [False, False, False, False, True, False]
    assert rows[4]["held"] == 4 and rows[4]["held_rt"] == 2

    # a window whose edges cut hour buckets gets a range; past the counts, none
    cache = xcn.CountsCache("from:a", "hour")
    assert cache.span_counts(w[1][0] + timedelta(hours=1, minutes=30), w[1][1]) == (7, 8)
    assert cache.span_counts(t0 + 3 * D - H, t0 + 3 * D + H) is None

    lines = "\n".join(xcn.report_lines({"t": ["a"]}, None, t0, t0 + 3 * D, "hour"))
    assert "FETCH SHORTFALL" in lines and "@zz" not in lines
    assert "may leave out native retweets" in lines
    month = xcn.monthly_rows("a", t0, t0 + 3 * D, "hour")
    assert month == [
        {
            "month": "2026-03",
            "x": len(x_posts),
            "share": 1.0,
            "x_rt": None,
            "rt_share": 0.0,
            "held": len(held),
            "held_rt": 2,
        },
    ]
    # a month only partly counted is marked with its covered share
    part = xcn.monthly_rows("a", _t("2026-02-27"), t0 + 3 * D, "hour")
    assert part[0]["x"] is None and part[1]["share"] == 1.0


def test_report_needs_no_counts_and_no_credentials(data, monkeypatch):
    _accounts(monkeypatch, {"t": ["a"]})
    monkeypatch.setattr(xcn, "get_client", _no_client)
    _write_jsonl(settings.X_RAW_DIR / "a.jsonl", [_rec(1, _t("2026-03-01T05:00"), "x")])
    path = settings.X_SAMPLING_DIR / "windows_2026-03-02.jsonl"
    _write_jsonl(
        path,
        [
            {
                "handle": "a",
                "mode": "sample",
                "window_start": "2026-03-01T00:00:00Z",
                "window_end": "2026-03-02T11:00:00Z",
                "cap": 5,
                "got": 1,
            }
        ],
    )
    res = _cli(
        [
            "x-counts",
            "--report",
            "--manifest",
            str(path),
            "--start",
            "2026-03-01",
            "--end",
            "2026-03-03",
        ]
    )
    assert res.exit_code == 0, res.output
    assert "not counted" in res.output and "no hour counts cached" in res.output


# ── archive probe ──────────────────────────────────────────────────

HEG = "PeteHegseth"


def _day(d: datetime, n_rt: int, n_orig: int, first_id: int) -> list[tuple]:
    """n_rt retweets then n_orig originals an hour apart from 01:00."""
    out = []
    for k in range(n_rt + n_orig):
        text = f"RT @x: shared {k}" if k < n_rt else f"own post {k}"
        out.append((first_id + k, d + (k + 1) * H, text))
    return out


SEP_CUT = _t("2026-09-17T20:59:50")  # Hegseth's first collect window on 2026-11-01


def _probe_cache(manifest_cut=SEP_CUT):
    """Feb: the 5th holds 9 (4 RT / 5 orig), the 12th 7 (6/1), the 19th 12
    (too many), the 22nd 6 originals (no mix). Sep: the 15th 6 (4/2) and the
    17th 6 (3/3), which a sampled collect window cuts. Returns the posts."""
    posts = (
        _day(_t("2026-02-05"), 4, 5, 1000)
        + _day(_t("2026-02-12"), 6, 1, 2000)
        + _day(_t("2026-02-19"), 6, 6, 3000)
        + _day(_t("2026-02-22"), 0, 6, 4000)
        + _day(_t("2026-09-15"), 4, 2, 5000)
        + _day(_t("2026-09-17"), 3, 3, 6000)
    )
    recs = [_rec(i, ts, text, user=HEG) for i, ts, text in posts]
    random.Random(7).shuffle(recs)  # the choice must not depend on file order
    _write_jsonl(settings.X_RAW_DIR / f"{HEG}.jsonl", recs)
    if manifest_cut:
        _write_jsonl(
            settings.X_SAMPLING_DIR / "windows_2026-11-01.jsonl",
            [
                {
                    "handle": HEG,
                    "mode": "sample",
                    "window_start": xcn.z(manifest_cut),
                    "window_end": xcn.z(manifest_cut + 35 * H),
                    "cap": 5,
                    "got": 5,
                    "truncated": True,
                }
            ],
        )
    return posts


def test_probe_window_choice_is_deterministic_and_closed(data):
    _probe_cache()
    wins = xa.choose_windows(HEG)
    assert [(w["label"], w["start"], w["end"]) for w in wins] == [
        ("feb", "2026-02-05T00:00:00Z", "2026-02-06T00:00:00Z"),
        ("sep", "2026-09-15T00:00:00Z", "2026-09-16T00:00:00Z"),  # not the 17th: not closed
    ]
    assert (wins[0]["cached"], wins[0]["cached_rt"], wins[0]["cached_orig"]) == (9, 4, 5)
    assert xa.choose_windows(HEG) == wins

    # without the sampled window the 17th is whole, and its 3/3 mix wins
    (settings.X_SAMPLING_DIR / "windows_2026-11-01.jsonl").unlink()
    assert xa.choose_windows(HEG)[1]["start"] == "2026-09-17T00:00:00Z"


def test_probe_refuses_without_a_fitting_day(data, monkeypatch):
    _probe_cache()
    monkeypatch.setattr(settings, "X_ARCHIVE_PROBE_MAX_POSTS", 5)
    with pytest.raises(xa.NoProbeWindow, match="feb"):
        xa.choose_windows(HEG)
    monkeypatch.setattr(xcn, "get_client", _no_client)
    res = _cli(["x-archive", "probe", "--estimate"])
    assert res.exit_code == 1 and "Refusing" in res.output


def _probe_stub(posts, **kw) -> Stub:
    all_t = [ts for _, ts, _ in posts]
    rt_t = [ts for _, ts, text in posts if text.startswith("RT @")]
    counted = {f"from:{HEG}": all_t, f"from:{HEG} is:retweet": rt_t}
    if kw.pop("counts_drop_rt", False):
        counted = {f"from:{HEG}": sorted(set(all_t) - set(rt_t)), f"from:{HEG} is:retweet": []}
    return Stub(counted, tweets=posts, **kw)


def test_probe_estimate_makes_no_call(data, monkeypatch):
    _probe_cache()
    monkeypatch.setattr(xcn, "get_client", _no_client)
    monkeypatch.setattr(xc.tweepy, "Client", _no_client)
    res = _cli(["x-archive", "probe", "--estimate"])
    assert res.exit_code == 0, res.output
    assert "feb: 2026-02-05T00:00 .. 2026-02-06T00:00 UTC, 9 cached posts" in res.output
    assert "at most $0.12" in res.output
    assert not settings.X_ARCHIVE_DIR.exists() and not settings.X_COUNTS_DIR.exists()
    monkeypatch.setattr(settings, "X_RUN_BUDGET_USD", 0.10)
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 1 and "exceeds X_RUN_BUDGET_USD" in res.output


@pytest.mark.parametrize(
    ("kw", "want"),
    [
        ({}, "archive works incl. retweets"),
        ({"drop_rt": True}, "archive works, retweets dropped"),
        ({"since": _t("2026-06-01")}, "archive misses older posts"),
        (
            {"since": _t("2026-06-01"), "drop_rt": True},
            "archive misses older posts; retweets dropped",
        ),
    ],
)
def test_probe_verdicts(data, monkeypatch, kw, want):
    posts = _probe_cache()
    stub = _probe_stub(posts, **kw)
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    monkeypatch.setattr(xcn, "_sleep", lambda s: stub.events.append(("sleep", s)))
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 0, res.output
    assert f"Verdict: {want}" in res.output
    kinds = [e[0] for e in stub.events]
    assert kinds == ["search", "sleep", "search", "sleep", "counts", "sleep", "counts"]
    assert all(e[1] >= 1.1 for e in stub.events if e[0] == "sleep")
    searches = [e for e in stub.events if e[0] == "search"]
    assert [(e[1], e[2], e[4]) for e in searches] == [
        (f"from:{HEG}", "2026-02-05T00:00:00Z", 10),
        (f"from:{HEG}", "2026-09-15T00:00:00Z", 10),
    ]
    counts = [e for e in stub.events if e[0] == "counts"]
    assert [(e[1], e[2], e[3], e[4]) for e in counts] == [
        (f"from:{HEG}", "2026-02-01T00:00:00Z", "2026-03-03T00:00:00Z", "day"),
        (f"from:{HEG} is:retweet", "2026-02-01T00:00:00Z", "2026-03-03T00:00:00Z", "day"),
    ]
    state = xa.load_state(HEG)
    assert state["complete"] and state["verdict"] == want
    assert "counts include retweets" in state["counts_reading"]
    saved = [json.loads(x) for x in xa.posts_path(HEG).read_text().splitlines()]
    assert {r["probe_window"] for r in saved} <= {"feb", "sep"}
    assert len(saved) == sum(state["searches"][k]["returned"] for k in ("feb", "sep"))

    # a finished probe is reused, free; the counts went to the shared cache
    n = len(stub.calls)
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 0 and "A finished probe is saved" in res.output
    assert f"Verdict: {want}" in res.output and len(stub.calls) == n
    assert xcn.CountsCache(f"from:{HEG}", "day").exists


def test_probe_reads_counts_that_leave_out_retweets(data, monkeypatch):
    posts = _probe_cache()
    stub = _probe_stub(posts, counts_drop_rt=True)
    state = xa.run_probe(stub, xa.plan_probe())
    assert state["counts_reading"].startswith("counts leave out retweets")
    lines = "\n".join(xa.report_lines(state))
    assert "2026-02-05" in lines and "Verdict: archive works incl. retweets" in lines


def test_probe_403_is_a_saved_result(data, monkeypatch):
    _probe_cache()
    stub = Stub(deny={"search"})
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 0, res.output
    assert "GET /2/tweets/search/all answered 403" in res.output
    assert "No further requests were made" in res.output
    assert "Verdict: no archive access" in res.output and stub.attempts == 1
    assert xa.load_state(HEG)["verdict"] == "no archive access"
    res = _cli(["x-archive", "probe", "--yes"])  # reused: no request
    assert "no archive access" in res.output and stub.attempts == 1

    # --force runs it again and sets the old files aside
    stub.deny = set()
    stub.posts = {}
    res = _cli(["x-archive", "probe", "--yes", "--force"])
    assert res.exit_code == 0, res.output
    assert "no archive access" not in res.output.split("Verdict:")[-1]
    aside = [p.name for p in settings.X_ARCHIVE_DIR.glob(f"probe_{HEG}.*.json")]
    assert len(aside) == 1


def test_probe_resumes_without_rebuying_a_step(data, monkeypatch):
    posts = _probe_cache()
    stub = _probe_stub(posts, fail_on=2)  # the Sep search fails
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 1 and "rerun to resume" in res.output
    state = xa.load_state(HEG)
    assert list(state["searches"]) == ["feb"] and not state["complete"]
    stub.fail_on = None
    res = _cli(["x-archive", "probe", "--estimate"])
    assert "1 search/all requests" in res.output and "at most $0.07" in res.output
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 0, res.output
    assert [e[2] for e in stub.calls if e[0] == "search"] == [
        "2026-02-05T00:00:00Z",
        "2026-09-15T00:00:00Z",
    ]
    assert len(xa.posts_path(HEG).read_text().splitlines()) == 15  # 9 + 6, none twice


# ── short answers, query keys, refusals that no wait fixes ─────────


def test_short_answer_without_token_records_only_what_it_covers(data):
    """X answers a 62-day stretch with one 31-day page and no next_token:
    only that page is counted, not two months of zeros."""
    end = FEB + 62 * D
    posts = _every(FEB, end, 5 * H)
    stub = Stub({"from:a": posts}, short=True)
    plan = xcn.plan_query("from:a", FEB, end, "hour")
    assert plan["pages"] == 2
    res = xcn.fetch_query(stub, plan, xcn.RequestCap(10, 1.0))
    assert res["requests"] == 1 and not res["complete"] and "no next_token" in res["stopped"]
    cache = xcn.CountsCache("from:a", "hour")
    assert cache.covered == [(FEB + 31 * D, end)]
    assert cache.total(FEB, FEB + 31 * D) == (0, 0.0)  # not counted, not "0 posts"
    assert cache.span_counts(FEB + 2 * D, FEB + 3 * D) is None
    assert not cache.fetches[0]["complete"]
    left = xcn.plan_query("from:a", FEB, end, "hour")
    assert left["todo"] == [(FEB, FEB + 31 * D)] and left["pages"] == 1
    assert xcn.fetch_query(stub, left, xcn.RequestCap(10, 1.0))["complete"]
    assert xcn.CountsCache("from:a", "hour").total(FEB, end) == (len(posts), 1.0)


def test_every_expected_page_answered_counts_missing_buckets_as_zero(data):
    """X may send no row for an empty bucket: when it answered every page
    the paging implies, the stretch is covered and the silence counts 0."""
    posts = _every(FEB + 7 * D, FEB + 10 * D, 6 * H)  # nothing in the first week
    stub = Stub({"from:a": posts}, omit_zero=True)
    res = xcn.fetch_query(
        stub, xcn.plan_query("from:a", FEB, FEB + 10 * D, "hour"), xcn.RequestCap(5, 1.0)
    )
    assert res["complete"] and res["total"] == len(posts)
    cache = xcn.CountsCache("from:a", "hour")
    assert cache.covered == [(FEB, FEB + 10 * D)]
    assert cache.span_counts(FEB, FEB + 7 * D) == (0, 0)


def test_query_key_ignores_username_case_and_spacing(data, monkeypatch):
    k = xcn.query_key
    assert (
        k("from:whitehouse  is:retweet")
        == k(" from:WhiteHouse is:retweet")
        == ("from:whitehouse is:retweet")
    )
    assert k("-FROM:VP (from:Pontifex OR x)") == "-from:vp (from:pontifex OR x)"
    assert k("xfrom:AbC Iran") == "xfrom:AbC Iran"  # not an operator; words keep their case
    assert xcn.cache_path("from:WhiteHouse is:retweet", "hour") == xcn.cache_path(
        "from:whitehouse is:retweet", "hour"
    )
    # one query per key; the account's spelling comes first and stays
    assert xcn.build_queries({"t": ["WhiteHouse"]}, None, "  from:whitehouse ") == [
        "from:WhiteHouse"
    ]
    _accounts(monkeypatch, {"t": ["Pontifex"]})
    res = _cli(
        [
            "x-counts",
            "--estimate",
            "--query",
            "from:PONTIFEX",
            "--start",
            "2026-02-01",
            "--end",
            "2026-03-05",
        ]
    )
    assert res.exit_code == 0 and "Planned: 2 requests" in res.output, res.output

    # an is:retweet count bought in lower case is the one the report reads
    stub = Stub({"from:whitehouse is:retweet": _every(FEB, MAR5, 9 * H)})
    xcn.fetch_query(
        stub,
        xcn.plan_query("from:whitehouse is:retweet", FEB, MAR5, "hour"),
        xcn.RequestCap(5, 1.0),
    )
    again = xcn.plan_query("from:WhiteHouse is:retweet", FEB, MAR5, "hour")
    assert again["pages"] == 0 and again["skip_reason"] == "cached (hour)"
    _write_jsonl(settings.X_RAW_DIR / "WhiteHouse.jsonl", [])
    month = xcn.monthly_rows("WhiteHouse", FEB, _t("2026-03-01"), "hour")
    assert month[0]["x_rt"] == len(_every(FEB, _t("2026-03-01"), 9 * H))


def test_401_is_a_credential_failure_not_access(data, monkeypatch):
    def unauthorized(**_):
        raise _http_error(tweepy.errors.Unauthorized, 401, UNAUTHORIZED)

    with pytest.raises(xcn.CredentialsRejected) as err:
        xcn.call_with_backoff(unauthorized, xcn.COUNTS_ENDPOINT)
    assert not isinstance(err.value, xcn.AccessDenied) and err.value.status == 401

    _accounts(monkeypatch, {"t": ["Aa", "Bb"]})
    stub = Stub(unauth={"counts"})
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-counts", "--yes", "--start", "2026-02-01", "--end", "2026-03-05"])
    assert res.exit_code == 1 and "answered 401" in res.output
    assert "No further requests were made; check X_BEARER_TOKEN" in res.output
    assert stub.attempts == 1 and data.sleeps == []


def test_usage_capped_429_stops_at_once(data, monkeypatch):
    stub = Stub({"from:a": []}, usage_capped=1)
    plan = xcn.plan_query("from:a", FEB, FEB + 10 * D, "hour")
    with pytest.raises(xcn.UsageCapped, match="UsageCapExceeded"):
        xcn.fetch_query(stub, plan, xcn.RequestCap(5, 1.0))
    assert stub.attempts == 1 and data.sleeps == []  # no wait, no retry
    assert not xcn.cache_path("from:a", "hour").exists()

    _accounts(monkeypatch, {"t": ["Aa"]})
    stub = Stub(usage_capped=1)
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-counts", "--yes", "--start", "2026-02-01", "--end", "2026-03-05"])
    assert res.exit_code == 1 and "usage cap is reached" in res.output, res.output
    assert stub.attempts == 1


def test_span_counts_needs_the_last_partial_bucket_covered(data):
    t0 = _t("2026-03-01")
    cache = xcn.CountsCache("from:a", "hour")
    cache.merge([], (t0, t0 + 3 * D))
    end = t0 + 3 * D
    assert cache.span_counts(end - 2 * H, end) == (0, 0)
    # a window ending mid-bucket past the counted range is not counted
    assert cache.span_counts(end - 2 * H, end + timedelta(minutes=38)) is None


# ── probe: stops that leave it resumable, verdict edges ────────────


def test_complete_until_stops_at_sampled_or_truncated_windows(data):
    def row(mode, start, hours, truncated, handle=HEG):
        s = _t(start)
        return {
            "handle": handle,
            "mode": mode,
            "window_start": xcn.z(s),
            "window_end": xcn.z(s + hours * H),
            "cap": 450,
            "got": 3,
            "truncated": truncated,
        }

    _write_jsonl(
        settings.X_SAMPLING_DIR / "windows_2026-03-02.jsonl",
        [
            row("take_all", "2026-02-03T00:00", 35, False),  # read whole: no cut
            row("take_all", "2026-02-10T05:00", 35, True),  # older posts left unread
            row("sample", "2026-02-07T00:00", 35, True, handle="other"),
        ],
    )
    feb, mar = _t("2026-02-01"), _t("2026-03-01")
    assert xa.complete_until(HEG, feb, mar) == _t("2026-02-10T05:00")
    assert xa.complete_until(HEG, _t("2026-02-11"), mar) == _t("2026-02-11")  # cut before it


def test_probe_401_leaves_it_resumable(data, monkeypatch):
    posts = _probe_cache()
    stub = _probe_stub(posts, unauth={"search"})
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 1 and "answered 401" in res.output
    assert "check X_BEARER_TOKEN" in res.output and "rerun to resume" in res.output
    assert "no archive access" not in res.output and stub.attempts == 1
    state = xa.load_state(HEG)
    assert not state["complete"] and state["verdict"] is None and state["denied"] is None

    stub.unauth = set()  # the token is fixed
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 0, res.output
    assert "Verdict: archive works incl. retweets" in res.output
    assert len([c for c in stub.calls if c[0] == "search"]) == 2


def test_probe_counts_step_at_its_page_cap_is_resumed(data, monkeypatch):
    posts = _probe_cache()
    stub = _probe_stub(posts, page_days=20)  # X pages more finely than planned
    monkeypatch.setattr(xcn, "get_client", lambda: stub)
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 1 and "Probe unfinished" in res.output, res.output
    assert "UNFINISHED" in res.output and "counts step left for a rerun" in res.output
    state = xa.load_state(HEG)
    assert not state["complete"] and state["counts"] == {} and len(state["pending"]) == 2
    assert len(stub.calls) == 4  # 2 searches + 1 page per counts query, at its cap

    res = _cli(["x-archive", "probe", "--estimate"])
    assert "0 search/all requests" in res.output and "2 counts/all" in res.output
    assert "at most $0.02" in res.output
    res = _cli(["x-archive", "probe", "--yes"])
    assert res.exit_code == 0, res.output
    state = xa.load_state(HEG)
    assert state["complete"] and state["pending"] == []
    assert state["counts_reading"].startswith("counts include retweets")
    assert [c[0] for c in stub.calls].count("search") == 2  # no search bought twice
    assert len(stub.calls) == 6


def test_probe_some_retweets_missing_is_inconclusive(data, monkeypatch):
    posts = _probe_cache()
    stub = _probe_stub(posts, drop_ids={1000, 1001, 1002})  # 3 of Feb's 4 retweets
    state = xa.run_probe(stub, xa.plan_probe())
    # 12 of 15 back overall (0.8), but only 5 of 8 retweets
    assert state["verdict"] == (
        "inconclusive: 7/7 cached originals and 5/8 cached retweets came back"
    )


@pytest.mark.parametrize(
    ("x_rt", "held_rt", "want"),
    [
        (8, 10, "counts include retweets"),
        (5, 10, "counts include some retweets"),
        (3, 10, "counts include some retweets"),
        (2, 10, "counts leave out retweets"),  # the 1 - 0.8 edge, not a float miss
        (0, 10, "counts leave out retweets"),
        (0, 0, "counts: no cached retweets"),
    ],
)
def test_counts_reading_bands(x_rt, held_rt, want):
    q_all, q_rt = xa.counts_queries(HEG)
    state = {
        "handle": HEG,
        "counts": {q_all: {"days": {"2026-02-05": 12}}, q_rt: {"days": {"2026-02-05": x_rt}}},
        "held_by_day": {"2026-02-05": [12, held_rt]},
    }
    assert xa.counts_reading(state).startswith(want)
