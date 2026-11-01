"""
Gap-fill windows and run-planning for the X collector, tested against a
stub tweepy client (no network, no spend).
"""

import itertools
import json
import math
import os
import random
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from config import settings
from src.collectors import x_collector as xc

# ── stub client ────────────────────────────────────────────────────

_EDGE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


class StubClient:
    """Serves synthetic tweets newest-first with 100-per-page pagination,
    exactly like GET /2/users/:id/tweets: [start_time, end_time] inclusive,
    as X documents, whole-second edges and 5 <= max_results <= 100. `extra`
    maps a tweet id to raw API fields (note_tweet, referenced_tweets) that
    tweepy leaves in .data; `texts` overrides a tweet's truncated `text`.
    `short_by=k` returns k fewer posts than asked while more remain, and
    `over_by=k` k more; `empty_pages=True` returns nothing but a next_token
    (and gives up after 60 requests, so a loop with no page guard fails
    instead of hanging); `fail_on_call=n` raises on the n-th timeline
    request. Search pages the same way (10 <= max_results <= 100). Counts
    what it returned (every post in every response: what X bills) and the
    user lookups."""

    def __init__(
        self,
        tweets,
        extra=None,
        texts=None,
        short_by=0,
        over_by=0,
        empty_pages=False,
        fail_on_call=None,
    ):
        # tweets: list of (id, created_at naive-UTC datetime)
        self.tweets = sorted(tweets, key=lambda t: t[1], reverse=True)
        self.extra, self.texts = extra or {}, texts or {}
        self.short_by, self.over_by, self.empty_pages = short_by, over_by, empty_pages
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.lookups = 0
        self.returned = 0
        self.returned_ids: list[str] = []
        self.max_results: list[int] = []
        self.windows: list[tuple[str, str]] = []
        self.fields = []

    def get_user(self, username):
        self.lookups += 1
        return SimpleNamespace(data=SimpleNamespace(id=42))

    def _tweet(self, i, ts):
        text = self.texts.get(i, f"t{i}")
        return SimpleNamespace(
            id=i,
            text=text,
            created_at=ts.replace(tzinfo=UTC),
            public_metrics={"like_count": 1},
            lang="en",
            author_id=7,
            data={"id": str(i), "text": text, **self.extra.get(i, {})},
        )

    def get_users_tweets(
        self, id, start_time, end_time, max_results, pagination_token, tweet_fields
    ):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError(f"stub failure on call {self.calls}")
        assert _EDGE.match(start_time) and _EDGE.match(end_time), (start_time, end_time)
        assert 5 <= max_results <= 100, max_results
        self.fields.append(tweet_fields)
        self.max_results.append(max_results)
        self.windows.append((start_time, end_time))
        s, e = _parse(start_time), _parse(end_time)
        return self._page([t for t in self.tweets if s <= t[1] <= e], max_results, pagination_token)

    def _page(self, hits, max_results, token):
        if self.empty_pages:
            assert self.calls <= 60, "no page guard"
            return SimpleNamespace(data=None, meta={"next_token": "more"})
        offset = int(token or 0)
        page = hits[offset : offset + max_results - self.short_by + self.over_by]
        self.returned += len(page)
        self.returned_ids.extend(str(i) for i, _ in page)
        nxt = offset + len(page)
        meta = {"next_token": str(nxt)} if nxt < len(hits) else {}
        return SimpleNamespace(data=[self._tweet(i, ts) for i, ts in page], meta=meta)

    def search_recent_tweets(self, query, max_results, next_token, tweet_fields, start_time=None):
        self.calls += 1
        assert 10 <= max_results <= 100, max_results
        self.fields.append(tweet_fields)
        self.max_results.append(max_results)
        return self._page(self.tweets, max_results, next_token)


def _stream(start: datetime, days: int, per_day: int):
    """`per_day` tweets on every day of [start, start+days)."""
    out, i = [], 0
    for d in range(days):
        for k in range(per_day):
            out.append((i, start + timedelta(days=d, hours=k)))
            i += 1
    return out


def _every(start: datetime, end: datetime, step: timedelta, first_id: int = 0):
    """A post every `step` from `start` while before `end`."""
    out, t, i = [], start, first_id
    while t < end:
        out.append((i, t))
        t, i = t + step, i + 1
    return out


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """The X cache in tmp_path; the processed tree too, which a forced re-read
    that changes a post's text rewrites (labels and sentiment_all). Every
    account is take-all unless a test samples it."""
    monkeypatch.setattr(settings, "X_RAW_DIR", tmp_path)
    monkeypatch.setattr(settings, "X_SAMPLING_DIR", tmp_path / "x_sampling")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        settings, "SENTIMENT_OUTPUT", tmp_path / "processed" / "sentiment_all.parquet"
    )
    monkeypatch.setattr(settings, "ACCOUNT_CAP_OVERRIDES", {})
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {})
    return tmp_path


def _manifest_rows() -> list[dict]:
    rows = []
    for path in sorted(settings.X_SAMPLING_DIR.glob("*.jsonl")):
        rows.extend(json.loads(x) for x in path.read_text().splitlines())
    return rows


def _how(d: datetime) -> int:
    """Hour of the week, Monday 00:00 = 0."""
    return d.weekday() * 24 + d.hour


# ── pure planning ──────────────────────────────────────────────────


def test_plan_windows_fixed_length_whole_seconds():
    raw = _parse("2026-06-01T00:00:00Z").replace(microsecond=250_000)
    s = _parse("2026-06-01T00:00:01Z")  # rounded up to the next second
    full = timedelta(hours=35)

    merged = xc.plan_windows(raw, s + 3 * full + timedelta(hours=10))
    assert merged[0][0] == s and merged[-1][1] == s + 3 * full + timedelta(hours=10)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(merged))
    assert [e - a for a, e in merged] == [full, full, full + timedelta(hours=10)]

    apart = xc.plan_windows(raw, s + 3 * full + timedelta(hours=20))
    assert [e - a for a, e in apart] == [full, full, full, timedelta(hours=20)]

    end = (s + 2 * full).replace(microsecond=900_000)  # floored
    assert xc.plan_windows(raw, end)[-1][1] == s + 2 * full
    assert all(a.microsecond == 0 == e.microsecond for a, e in apart)
    assert xc.plan_windows(s, s) == [] and xc.plan_windows(s, s - full) == []
    short = xc.plan_windows(s, s + timedelta(hours=3))  # one window, however short
    assert short == [(s, s + timedelta(hours=3))]


def test_plan_windows_contiguous_oldest_first():
    s, e = _parse("2026-05-10T00:00:00Z"), _parse("2026-09-15T00:00:00Z")
    windows = xc.plan_windows(s, e, hours=336)
    assert windows[0][0] == s and windows[-1][1] == e
    assert all(a[1] == b[0] for a, b in itertools.pairwise(windows))
    assert all(b - a == timedelta(days=14) for a, b in windows[:-1])
    assert windows[-1][1] - windows[-1][0] <= timedelta(days=21)
    assert windows == sorted(windows)
    assert xc.plan_windows(e, s) == []


def test_window_ends_step_through_clock_and_week():
    s0 = _parse("2026-09-18T01:38:57Z")
    windows = xc.plan_windows(s0, s0 + timedelta(hours=35 * 24))
    ends = [e for _, e in windows]
    assert len(ends) == 24
    assert len({e.hour for e in ends}) == 24
    assert {(_how(e) - _how(s0)) % 168 for e in ends} == {7 * j for j in range(24)}
    assert math.gcd(settings.GAP_FILL_WINDOW_HOURS % 24, 24) == 1


def test_window_cap_tail_rule():
    """w for a full window, never under w, in proportion rounded half up for
    a longer merged tail: 38.5 h is 5.5 posts (6) and 40 h 5.71 (6), where
    a floor would give 5 and change the printed maximum."""
    s = _parse("2026-06-01T00:00:00Z")
    for hours, cap in [(35, 5), (20, 5), (38, 5), (38.5, 6), (40, 6), (45, 6), (52, 7)]:
        assert xc._window_cap(5, s, s + timedelta(hours=hours), 35) == cap, hours


def _no_client(*a, **k):
    raise AssertionError("no client may be built on this path")


def test_sampled_plan_for_the_oct_refresh(raw_dir, monkeypatch):
    monkeypatch.setattr(xc.tweepy, "Client", _no_client)
    monkeypatch.setattr(xc, "get_client", _no_client)
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    _write_cache(
        raw_dir / "acct.jsonl", [_cached(1, _parse("2026-09-18T01:38:56Z"), "newest cached")]
    )
    plan = xc.plan_account("acct", end=_parse("2026-11-01T14:00:00Z"))
    assert plan["mode"] == "sample" and plan["hours"] == 35 and plan["window_cap"] == 5
    assert len(plan["slices"]) == 31 and plan["caps"] == [5] * 31
    assert plan["max_reads"] == 155 and plan["lookups"] == 1
    assert plan["max_cost_usd"] == pytest.approx(
        155 * settings.X_READ_COST_USD + settings.X_USER_READ_COST_USD
    )
    assert plan["slices"][0][0] == _parse("2026-09-18T01:38:57Z")


def test_sampled_plan_lengthens_to_fit_backstop(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    s = _parse("2026-05-10T00:00:00Z")
    plan = xc.plan_account("acct", start=s, end=s + timedelta(days=120), cap=100)
    assert sum(plan["caps"]) <= 100 and plan["max_reads"] == sum(plan["caps"])
    assert min(plan["caps"]) >= 5 and len(plan["caps"]) == len(plan["slices"])
    assert plan["hours"] >= 35 and math.gcd(plan["hours"] % 24, 24) == 1


def test_sampled_single_window_clamped_to_backstop(raw_dir, monkeypatch):
    """7 per window, backstop 5, a 72 h gap: lengthening leaves one window
    whose cap (7) is still over the backstop, so the cap is the backstop."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 7})
    s = _parse("2026-06-01T00:00:00Z")
    plan = xc.plan_account("acct", start=s, end=s + timedelta(hours=72), cap=5)
    assert len(plan["slices"]) == 1 and plan["caps"] == [5] and plan["max_reads"] == 5


def test_estimate_run_counts_lookups(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "MAX_TWEETS_PER_USER", 100)
    monkeypatch.setattr(settings, "MAX_TWEETS_PER_SEARCH", 50)
    monkeypatch.setattr(settings, "ACCOUNT_CAP_OVERRIDES", {"heavy": 20})
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"sampled": 5})
    _write_cache(
        raw_dir / "sampled.jsonl",
        [_cached(1, settings.COLLECTION_END - timedelta(days=10), "x")],
    )
    _write_cache(
        raw_dir / "done.jsonl", [_cached(2, settings.COLLECTION_END + timedelta(days=1), "x")]
    )
    plan = xc.estimate_run({"t": ["a", "heavy", "sampled", "done"]}, ["q"])
    by = {p["handle"]: p for p in plan["accounts"]}
    assert by["heavy"]["cap"] == 20 and by["heavy"]["max_reads"] == 20
    assert by["heavy"]["caps"] is None and by["heavy"]["mode"] == "take_all"
    assert by["sampled"]["max_reads"] == sum(by["sampled"]["caps"]) == 35  # 7 windows x 5
    assert by["done"]["skip_reason"] and by["done"]["lookups"] == 0
    assert by["done"]["max_cost_usd"] == 0
    assert plan["max_reads"] == 100 + 20 + 35 + 50
    assert plan["lookups"] == 3
    assert plan["max_cost_usd"] == pytest.approx(
        205 * settings.X_READ_COST_USD + 3 * settings.X_USER_READ_COST_USD
    )


def test_cap_under_request_minimum_skips(raw_dir, monkeypatch):
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(days=5)
    cache = raw_dir / "acct.jsonl"
    _write_cache(cache, [_cached("c0", t0 - timedelta(seconds=1), "old")])
    before = cache.read_bytes()

    plan = xc.plan_account("acct", start=t0, end=end, cap=4)
    assert "5-post" in plan["skip_reason"]
    assert plan["max_reads"] == 0 and plan["max_cost_usd"] == 0 and plan["lookups"] == 0
    stub = StubClient(_stream(t0, 5, per_day=3))
    got = xc.collect_user(stub, "acct", "tier", start=t0, end=end, max_tweets=4)
    assert len(got) == 1 and stub.calls == 0 and stub.lookups == 0
    assert cache.read_bytes() == before

    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 4})
    with pytest.raises(ValueError, match="ACCOUNT_WINDOW_CAPS"):
        xc.plan_account("acct", start=t0, end=end)


def test_sampled_gap_under_half_window_is_left(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    end = _parse("2026-06-10T12:00:00Z")
    cache = raw_dir / "acct.jsonl"
    _write_cache(cache, [_cached("c0", end - timedelta(hours=3), "old")])
    before = cache.read_bytes()
    plan = xc.plan_account("acct", end=end)
    assert "half a window" in plan["skip_reason"] and plan["lookups"] == 0
    stub = StubClient([(1, end - timedelta(hours=1))])
    xc.collect_user(stub, "acct", "tier", end=end)
    assert stub.calls == 0 and stub.lookups == 0
    assert cache.read_bytes() == before  # the cursor did not move
    assert not settings.X_SAMPLING_DIR.exists()

    # the threshold is half a window (17.5 h): 17 h waits, 18 h is one window of 5
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"x": 5})
    s = _parse("2026-06-01T00:00:00Z")
    short = xc.plan_account("x", start=s, end=s + timedelta(hours=17))
    assert "half a window" in short["skip_reason"]
    plan = xc.plan_account("x", start=s, end=s + timedelta(hours=18))
    assert plan["skip_reason"] is None
    assert plan["slices"] == [(s, s + timedelta(hours=18))] and plan["caps"] == [5]


def test_force_on_stale_sampled_cache_is_refused(raw_dir, monkeypatch):
    """A forced plan covers COLLECTION_START to now at one density: with a
    450 backstop the windows grow to 73 h, so a cache that stopped on Sep 18
    would get its uncached tail at half density, and the cursor would jump
    past it for good. The planner skips it until an incremental run has
    caught the cache up; a current cache can be forced."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    monkeypatch.setattr(settings, "ACCOUNT_CAP_OVERRIDES", {"acct": 450})
    start, end = _parse("2026-02-01T00:00:00Z"), _parse("2026-10-31T00:00:00Z")
    cache = raw_dir / "acct.jsonl"
    _write_cache(cache, [_cached(1, _parse("2026-09-18T01:38:56Z"), "newest cached")])
    before = cache.read_bytes()

    plan = xc.plan_account("acct", start=start, end=end, force=True)
    assert plan["skip_reason"] and "incremental" in plan["skip_reason"]
    assert plan["max_reads"] == 0 and plan["lookups"] == 0
    stub = StubClient(_every(start, end, timedelta(minutes=30)))
    xc.collect_user(stub, "acct", "tier", start=start, end=end, force=True)
    assert stub.calls == 0 and stub.lookups == 0 and cache.read_bytes() == before

    incremental = xc.plan_account("acct", start=start, end=end)
    assert incremental["hours"] == 35 and incremental["skip_reason"] is None

    _write_cache(cache, [_cached(1, end - timedelta(hours=3), "current")])
    current = xc.plan_account("acct", start=start, end=end, force=True)
    assert current["skip_reason"] is None and current["window"] == (start, end)

    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {})  # take-all: no density to lose
    _write_cache(cache, [_cached(1, _parse("2026-09-18T01:38:56Z"), "newest cached")])
    assert xc.plan_account("acct", start=start, end=end, force=True)["skip_reason"] is None


# ── fetching against the stub ──────────────────────────────────────


def test_gap_fill_has_no_hole(raw_dir, monkeypatch):
    """The failure this exists to prevent: a 120-day gap, 10 posts/day,
    backstop 100. The old newest-first fetch kept the last 10 days only.
    Sampled at 5 per window, the planner lengthens the windows to fit."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    gap_start = _parse("2026-05-10T00:00:00Z")
    end = gap_start + timedelta(days=120)
    client = StubClient(_stream(gap_start, 120, per_day=10))
    plan = xc.plan_account("acct", start=gap_start, end=end, cap=100)

    got = xc.collect_user(client, "acct", "tier", start=gap_start, end=end, max_tweets=100)

    assert len(got) == 100 == client.returned
    dates = sorted(_parse(t["created_at"]) for t in got)
    gaps = [b - a for a, b in itertools.pairwise(dates)]
    assert max(gaps) < timedelta(hours=2 * plan["hours"])
    assert dates[0] < gap_start + timedelta(hours=plan["hours"])  # oldest window present
    assert dates[-1] > end - timedelta(hours=plan["hours"])  # newest window present


def test_incremental_appends_only_new(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "GAP_FILL_WINDOW_HOURS", 336)
    t0 = _parse("2026-05-01T00:00:00Z")
    cache = raw_dir / "acct.jsonl"
    cache.write_text(
        json.dumps(
            {
                "id": "old",
                "user": "acct",
                "tier": "tier",
                "text": "x",
                "created_at": (t0 + timedelta(days=9)).replace(tzinfo=UTC).isoformat(),
            }
        )
        + "\n"
    )
    # 1/day for 40 days; only days 10..39 are newer than the cache
    client = StubClient(_stream(t0, 40, per_day=1))

    got = xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=40))

    assert got[0]["id"] == "old"
    new = got[1:]
    assert len(new) == 30
    assert min(_parse(t["created_at"]) for t in new) > t0 + timedelta(days=9)
    assert len(cache.read_text().splitlines()) == 31  # appended, not overwritten
    plan = xc.plan_account("acct", start=t0, end=t0 + timedelta(days=40))
    assert plan["incremental"] and plan["window"][0] > t0 + timedelta(days=9)


def test_uncapped_slice_fetches_everything(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "GAP_FILL_WINDOW_HOURS", 168)
    t0 = _parse("2026-06-01T00:00:00Z")
    client = StubClient(_stream(t0, 21, per_day=3))  # 63 tweets, backstop 500
    got = xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=21))
    assert len(got) == 63
    assert len({t["id"] for t in got}) == 63  # a post on a shared edge is stored once
    assert client.calls == 3  # the patched 7-day windows applied


def test_take_all_reads_bursts_whole(raw_dir):
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(days=44)
    _write_cache(raw_dir / "acct.jsonl", [_cached("c0", t0 - timedelta(seconds=1), "old")])
    daily = [(i, t0 + timedelta(days=i, hours=12, minutes=7)) for i in range(44)]
    burst_at = t0 + timedelta(hours=35 * 5 + 3)  # inside the sixth window
    burst = [(1000 + k, burst_at + timedelta(minutes=10 * k)) for k in range(40)]
    stub = StubClient(daily + burst)
    plan = xc.plan_account("acct", start=t0, end=end)
    assert plan["mode"] == "take_all" and plan["caps"] is None and plan["max_reads"] == 500

    got = xc.collect_user(stub, "acct", "tier", start=t0, end=end)

    new = got[1:]
    assert len(new) == 84 and len({t["id"] for t in new}) == 84
    assert stub.calls == len(plan["slices"])


def test_take_all_backstop_defers_rest(raw_dir, caplog):
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(days=10)
    cache = raw_dir / "acct.jsonl"
    _write_cache(cache, [_cached("c0", t0 - timedelta(seconds=1), "old")])
    posts = _stream(t0, 10, per_day=10)
    windows = xc.plan_windows(t0, end)

    stub = StubClient(posts)
    with caplog.at_level("INFO", logger=xc.logger.name):
        got = xc.collect_user(stub, "acct", "tier", start=t0, end=end, max_tweets=20)
    new = got[1:]
    assert len(new) == 20 and stub.returned <= 20
    assert all(_parse(t["created_at"]) <= windows[1][1] for t in new)  # the oldest windows
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "backstop" in errors[0]

    newest = max(_parse(r["created_at"]) for r in _read_cache(cache))
    again = StubClient(posts)
    xc.collect_user(again, "acct", "tier", start=t0, end=end)
    recs = _read_cache(cache)
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))
    later = recs[len(got) :]
    assert later and min(_parse(r["created_at"]) for r in later) > newest


@pytest.mark.parametrize("short_by", [0, 2])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("mode", ["sample", "take_all"])
@pytest.mark.parametrize("pattern", ["quiet", "bursty", "dense"])
def test_returned_never_exceeds_plan_maximum(raw_dir, monkeypatch, pattern, mode, force, short_by):
    """Over 20 seeds per case: X never bills more than the plan's maximum,
    one lookup per planned account, and every post paid for is stored (or
    was already)."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)  # thousands of appends
    cache = raw_dir / "acct.jsonl"
    for seed in range(20):
        rng = random.Random(f"{seed}-{pattern}-{mode}-{force}-{short_by}")
        cache.unlink(missing_ok=True)
        for f in settings.X_SAMPLING_DIR.glob("*"):
            f.unlink()
        t0 = _parse("2026-06-01T00:00:00Z") + timedelta(seconds=rng.randrange(86_400))
        end = t0 + timedelta(days=rng.randint(3, 15), seconds=rng.randrange(86_400))
        span = int((end - t0).total_seconds())
        if pattern == "dense":
            times = [
                t0 + timedelta(seconds=rng.randrange(span)) for _ in range(span * 60 // 86_400)
            ]
        else:
            times = [t0 + timedelta(seconds=rng.randrange(span)) for _ in range(span // 30_000)]
        if pattern == "bursty":
            for _ in range(rng.randint(1, 2)):
                b = t0 + timedelta(seconds=rng.randrange(span - 3600))
                times += [b + timedelta(seconds=rng.randrange(3600)) for _ in range(100)]
        posts = list(enumerate(times))

        monkeypatch.setattr(
            settings,
            "ACCOUNT_WINDOW_CAPS",
            {"acct": rng.choice([5, 7])} if mode == "sample" else {},
        )
        max_tweets = rng.choice([None, 5, 20, 60])
        if force:  # a current cache holding some of the window's posts, re-read and merged
            tail = _cached("tail", end - timedelta(hours=1), "old")
            _write_cache(cache, [_cached(i, ts, "old") for i, ts in posts[::7]] + [tail])
        else:
            _write_cache(cache, [_cached("c0", t0 - timedelta(seconds=1), "old")])
        before = {r["id"] for r in _read_cache(cache)}
        plan = xc.plan_account("acct", start=t0, end=end, force=force, cap=max_tweets)
        assert plan["skip_reason"] is None, plan["skip_reason"]
        if plan["mode"] == "sample":
            assert plan["max_reads"] <= (max_tweets or xc.account_cap("acct"))
        # posts exactly on shared window edges, which two windows can both return
        posts += [(10_000 + k, s) for k, (s, _) in enumerate(plan["slices"][1:4])]

        stub = StubClient(posts, short_by=short_by)
        xc.collect_user(stub, "acct", "tier", start=t0, end=end, max_tweets=max_tweets, force=force)

        assert stub.returned <= plan["max_reads"], (seed, plan["mode"], plan["caps"])
        assert stub.lookups == plan["lookups"]
        rows = _manifest_rows()
        assert sum(r["got"] for r in rows) == stub.returned
        ids = [r["id"] for r in _read_cache(cache)]
        assert len(ids) == len(set(ids))
        assert set(stub.returned_ids) <= set(ids)  # nothing paid for is dropped
        seen = set() if force else set(before)
        duplicates = 0
        for pid in stub.returned_ids:
            if pid in seen:
                duplicates += 1
            seen.add(pid)
        new_ids = set(stub.returned_ids) - (set() if force else before)
        assert sum(r["new"] for r in rows) == len(new_ids)
        assert len(new_ids) + duplicates == stub.returned
        if not force:
            assert set(ids) - before == new_ids


def test_short_page_stops_at_floor():
    t0 = _parse("2026-06-01T00:00:00Z")
    stub = StubClient(_stream(t0, 2, per_day=10), short_by=2)
    got, truncated = xc._fetch_window(stub, 42, "acct", "tier", t0, t0 + timedelta(days=2), 5)
    assert stub.max_results == [5] and stub.calls == 1
    assert len(got) == 3 and truncated


def test_window_page_guard(caplog):
    """Empty pages that keep a next_token: the client waits out rate limits
    (wait_on_rate_limit), so without the guard this would never end."""
    t0 = _parse("2026-06-01T00:00:00Z")
    stub = StubClient([], empty_pages=True)
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got, truncated = xc._fetch_window(stub, 42, "acct", "tier", t0, t0 + timedelta(days=1), 500)
    assert got == [] and truncated
    assert stub.calls == settings.X_MAX_PAGES_PER_WINDOW == 20
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1 and "20-page guard" in warns[0]


@pytest.mark.parametrize("mode", ["sample", "take_all"])
def test_over_returned_posts_are_all_kept(raw_dir, monkeypatch, caplog, mode):
    """X returning more than max_results bills them all: every one is stored."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5} if mode == "sample" else {})
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(hours=70)
    stub = StubClient(_every(t0 + timedelta(minutes=7), end, timedelta(minutes=20)), over_by=3)
    with caplog.at_level("WARNING", logger=xc.logger.name):
        xc.collect_user(stub, "acct", "tier", start=t0, end=end, max_tweets=20)
    ids = [r["id"] for r in _read_cache(raw_dir / "acct.jsonl")]
    assert stub.returned > sum(stub.max_results)  # X sent more than was asked
    assert sorted(ids) == sorted(stub.returned_ids) and len(ids) == len(set(ids))
    assert any("kept them all" in r.getMessage() for r in caplog.records)


# ── keyword search: a strict maximum, nothing billed is dropped ────


def test_search_stays_within_its_cap_and_keeps_every_post(raw_dir):
    """/search/recent pages often come back short with a next_token. Asking
    for its 10-post minimum when fewer are left went over the plan's 150 per
    term, and the extra posts, billed, were dropped."""
    now = datetime.now(UTC).replace(tzinfo=None)
    stub = StubClient(_every(now - timedelta(days=3), now, timedelta(minutes=5)), short_by=3)
    got = xc.collect_search(stub, "iran war", max_total=150)
    assert stub.max_results == [100, 53]  # 3 left: under the 10-post minimum, so stop
    assert stub.returned == 147  # within the 150 the plan counts
    assert sorted(t["id"] for t in got) == sorted(stub.returned_ids)
    assert len(_read_cache(xc._search_cache_path("iran war"))) == 147


def test_search_page_guard(raw_dir, caplog):
    stub = StubClient([], empty_pages=True)
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_search(stub, "iran war", max_total=150)
    assert got == [] and stub.calls == settings.X_MAX_PAGES_PER_SEARCH
    assert any("page guard" in r.getMessage() for r in caplog.records)


def test_no_rollover(raw_dir, monkeypatch):
    """A quiet first half leaves reads unused; they are not passed on."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    quiet = [(i, t0 + i * full + timedelta(hours=10)) for i in range(5)]
    dense = _every(t0 + 5 * full + timedelta(minutes=7), t0 + 10 * full, timedelta(minutes=20), 100)
    got = xc.collect_user(StubClient(quiet + dense), "acct", "tier", start=t0, end=t0 + 10 * full)
    rows = _manifest_rows()
    assert len(rows) == 10
    assert [r["got"] for r in rows] == [1] * 5 + [5] * 5
    assert len(got) == len(quiet) + 5 * 5


def _day_night(start: datetime, end: datetime):
    """60 posts a day: 56 over the 16 US-waking hours (12:00-04:00 UTC), 4 over
    the other 8."""
    out, day = [], start.replace(hour=0, minute=0, second=0)
    while day < end:
        out += [day + timedelta(hours=12) + timedelta(minutes=16 * 60 * k / 56) for k in range(56)]
        out += [day + timedelta(hours=5 + 2 * k, minutes=3) for k in range(4)]
        day += timedelta(days=1)
    return list(enumerate(t.replace(microsecond=0) for t in out))


def test_heavy_account_spread(raw_dir, monkeypatch):
    """The fault of the 14-day split: a heavy account kept each fortnight's
    newest day or so, mostly a weekend. 5 per 35 h spreads it out."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    newest = _parse("2026-09-18T01:38:56Z")
    _write_cache(raw_dir / "acct.jsonl", [_cached("c0", newest, "newest cached")])
    start = newest + timedelta(seconds=1)
    end = start + timedelta(days=44.5)
    stub = StubClient(_day_night(newest - timedelta(days=2), end + timedelta(days=2)))

    kept = sorted(
        _parse(t["created_at"])
        for t in xc.collect_user(stub, "acct", "tier", end=end)[1:]  # the cached post
    )

    assert len(kept) == 155
    monday = (start + timedelta(days=7 - start.weekday())).replace(hour=0, minute=0, second=0)
    weeks = 0
    while monday + timedelta(days=7) <= end:
        n = sum(monday <= t < monday + timedelta(days=7) for t in kept)
        assert 20 <= n <= 25, (monday, n)
        monday, weeks = monday + timedelta(days=7), weeks + 1
    assert weeks == 5
    assert len({t.date() for t in kept}) >= 30
    assert {t.weekday() for t in kept} == set(range(7))
    assert {t.hour // 6 for t in kept} == {0, 1, 2, 3}
    assert max(b - a for a, b in itertools.pairwise(kept)) <= timedelta(hours=70)


@pytest.mark.parametrize("incremental", [True, False])
def test_boundary_second_stored_once(raw_dir, incremental):
    """Windows share their edge second ([s0, s1] then [s1, s2]): a post at s1
    can come back twice and is stored once; nothing at s1 - 1 s or s1 is
    missed, whichever way X reads end_time."""
    t0 = _parse("2026-06-01T00:00:00Z")
    s1 = t0 + timedelta(hours=35)
    end = t0 + timedelta(hours=60)  # a 25 h second window
    cache = raw_dir / "acct.jsonl"
    if incremental:
        _write_cache(cache, [_cached("c0", t0 - timedelta(seconds=1), "old")])
    posts = [
        (1, s1 - timedelta(seconds=1)),
        (2, s1),
        (3, t0 + timedelta(hours=2)),
        (4, s1 + timedelta(hours=2)),
    ]
    stub = StubClient(posts)
    xc.collect_user(stub, "acct", "tier", start=t0, end=end)

    assert stub.windows == [(xc._iso(t0), xc._iso(s1)), (xc._iso(s1), xc._iso(end))]
    ids = [r["id"] for r in _read_cache(cache)]
    assert len(ids) == len(set(ids)) == 4 + incremental
    assert {"1", "2", "3", "4"} <= set(ids)
    assert stub.returned == 5  # post 2 was bought twice, and stored once
    assert [r["new"] for r in _manifest_rows()] == [3, 1]


def test_appends_after_each_window_and_resumes(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5, "ref": 5})
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(hours=35 * 14 + 25)  # 15 windows
    posts = _every(
        t0 - timedelta(days=1, minutes=-7), end + timedelta(days=1), timedelta(minutes=20)
    )
    old = [_cached("c0", t0 - timedelta(seconds=1), "old")]
    for handle in ("acct", "ref"):
        _write_cache(raw_dir / f"{handle}.jsonl", old)
    windows = xc.plan_account("acct", start=t0, end=end)["slices"]
    assert len(windows) == 15

    ref = StubClient(posts)  # the same run, never interrupted
    xc.collect_user(ref, "ref", "tier", start=t0, end=end)

    cache = raw_dir / "acct.jsonl"
    first = StubClient(posts, fail_on_call=10)
    with pytest.raises(RuntimeError, match="stub failure"):
        xc.collect_user(first, "acct", "tier", start=t0, end=end)
    stored = [_parse(r["created_at"]) for r in _read_cache(cache)[1:]]
    assert len(stored) == 9 * 5 and max(stored) <= windows[8][1]
    for s, e in windows[:9]:  # whole windows only
        assert sum(s <= t <= e for t in stored) == 5

    second = StubClient(posts)
    xc.collect_user(second, "acct", "tier", start=t0, end=end)
    recs = _read_cache(cache)
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))
    stored = [_parse(r["created_at"]) for r in recs[1:]]
    for s, e in windows:
        assert any(s <= t <= e for t in stored), (s, e)
    assert first.returned + second.returned <= ref.returned + 5


def test_first_pull_appends_each_window_and_resumes(raw_dir, monkeypatch):
    """An account newly added to config/accounts.py has no cache. Its first
    pull used to write once, at the end: a failure part-way lost every read,
    left manifest rows for posts that never reached the cache, and the rerun
    bought them all again. It now appends each window as it goes, so the
    rerun is incremental from the newest stored post."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5, "ref": 5})
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(hours=35 * 14 + 25)  # 15 windows
    posts = _every(t0 + timedelta(minutes=7), end, timedelta(minutes=20))
    ref = StubClient(posts)  # the same first pull, never interrupted
    xc.collect_user(ref, "ref", "tier", start=t0, end=end)
    assert ref.returned == 15 * 5

    cache = raw_dir / "acct.jsonl"
    first = StubClient(posts, fail_on_call=8)
    with pytest.raises(RuntimeError, match="stub failure"):
        xc.collect_user(first, "acct", "tier", start=t0, end=end)
    stored = _read_cache(cache)
    assert len(stored) == 7 * 5 == first.returned
    rows = [r for r in _manifest_rows() if r["handle"] == "acct"]
    assert len(rows) == 7 and sum(r["new"] for r in rows) == len(stored)

    second = StubClient(posts)
    xc.collect_user(second, "acct", "tier", start=t0, end=end)
    recs = _read_cache(cache)
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))
    rows = [r for r in _manifest_rows() if r["handle"] == "acct"]
    assert sum(r["new"] for r in rows) == len(recs)  # every row's posts are in the cache
    assert first.returned + second.returned <= ref.returned + 5


def test_first_pull_with_nothing_still_writes_the_cache(raw_dir):
    """`status` tells a collected account (even with 0 posts) from one never
    collected by its cache file."""
    t0 = _parse("2026-06-01T00:00:00Z")
    xc.collect_user(StubClient([]), "acct", "tier", start=t0, end=t0 + timedelta(days=3))
    assert (raw_dir / "acct.jsonl").read_text() == ""


def test_forced_run_that_fails_keeps_what_it_read(raw_dir, caplog):
    """A forced run merges into the cache once, at the end. A failure
    part-way used to drop every post it had billed; now what it read is
    merged in before the error propagates, and the manifest gets rows only
    for windows whose posts reached the cache."""
    t0 = _parse("2026-06-01T00:00:00Z")
    end = t0 + timedelta(days=10)  # 7 windows
    cache = raw_dir / "acct.jsonl"
    old = [
        _cached(1, t0 + timedelta(hours=1), "old text"),
        _cached("kept", t0 + timedelta(hours=2, minutes=30), "beyond the horizon"),
        _cached("tail", end - timedelta(hours=1), "current"),
    ]
    _write_cache(cache, old)
    stub = StubClient(_stream(t0, 10, per_day=3), fail_on_call=4)
    with caplog.at_level("INFO", logger=xc.logger.name), pytest.raises(RuntimeError):
        xc.collect_user(stub, "acct", "tier", start=t0, end=end, force=True)

    by = {r["id"]: r for r in _read_cache(cache)}
    assert stub.returned > 0 and set(stub.returned_ids) <= set(by)
    assert {"kept", "tail"} <= set(by) and by["1"]["text"] == "t1"  # merged, refreshed
    assert not list(raw_dir.glob("*.tmp"))
    rows = _manifest_rows()
    assert len(rows) == 3 and sum(r["got"] for r in rows) == stub.returned
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any("3 of 7 windows" in m for m in errors), errors


# ── forced runs merge into the cache ───────────────────────────────


def _cached(pid, ts: datetime, text: str, likes: int = 0) -> dict:
    return {
        "id": str(pid),
        "user": "acct",
        "tier": "tier",
        "text": text,
        "created_at": ts.replace(tzinfo=UTC).isoformat(),
        "metrics": {"like_count": likes},
        "lang": "en",
        "platform": "x",
    }


def _write_cache(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _read_cache(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_force_merges_into_cache_by_id(raw_dir, caplog):
    """Until 2026-10-05 a forced run replaced the cache with what it fetched.
    The timeline reaches back only ~3,200 posts and the run is capped, so a
    heavy account lost paid posts it could not read again. Post 9 stands for
    those: in the window, but X no longer returns it."""
    t0 = _parse("2026-06-01T00:00:00Z")
    cache = raw_dir / "acct.jsonl"
    _write_cache(
        cache,
        [
            _cached(1, t0, "t1"),  # re-read: same text, new metrics
            _cached(9, t0 + timedelta(minutes=30), "beyond the horizon"),
            _cached(2, t0 + timedelta(hours=1), "cut at 280"),  # re-read: new text
        ],
    )
    client = StubClient([(1, t0), (2, t0 + timedelta(hours=1)), (3, t0 + timedelta(hours=2))])

    with caplog.at_level("INFO", logger=xc.logger.name):
        got = xc.collect_user(
            client, "acct", "tier", start=t0, end=t0 + timedelta(days=2), force=True
        )

    # cached posts keep their place, the new one is appended
    assert [t["id"] for t in got] == ["1", "9", "2", "3"]
    assert _read_cache(cache) == got
    by = {t["id"]: t for t in got}
    assert by["1"]["metrics"] == {"like_count": 1}  # the re-read version wins
    assert by["2"]["text"] == "t2"
    assert by["9"]["text"] == "beyond the horizon"  # kept
    assert not list(raw_dir.glob("*.tmp"))  # written atomically
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "2 refreshed (1 with new text), 1 added, 1 kept" in msgs


def test_force_text_change_moves_paid_labels(raw_dir):
    """The Opus and topic labels are keyed by id: a re-read that changes a
    post's text moves them to the *_superseded archives (as x-backfill-text
    does), and its sentiment_all row is archived and cleared onto the new
    text. A post re-read with the same text keeps everything."""
    import pandas as pd

    from src.analysis.topic_label import labels_path
    from src.superseded import superseded_path

    t0 = _parse("2026-06-01T00:00:00Z")
    _write_cache(
        raw_dir / "acct.jsonl", [_cached(1, t0, "t1"), _cached(2, t0 + timedelta(hours=1), "cut")]
    )
    proc = settings.PROCESSED_DIR
    proc.mkdir()
    teacher = proc / "teacher_labels_claude-opus-5.parquet"
    pd.DataFrame({"id": ["1", "2"], "score_opus": [0.5, -0.5]}).to_parquet(teacher)
    pd.DataFrame({"id": ["1", "2"], "about_war": [True, False]}).to_parquet(labels_path())
    pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["t1", "cut"],
            "score_vader": [0.1, 0.2],
            "score_llm": [0.3, 0.4],
        }
    ).to_parquet(settings.SENTIMENT_OUTPUT)

    client = StubClient([(1, t0), (2, t0 + timedelta(hours=1))])
    xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=1), force=True)

    for path in (teacher, labels_path()):
        assert pd.read_parquet(path)["id"].tolist() == ["1"]
        arch = pd.read_parquet(superseded_path(path))
        assert arch["id"].tolist() == ["2"]
        assert (arch["superseded_reason"] == xc.FORCED_SUPERSEDED_REASON).all()
    sent = pd.read_parquet(settings.SENTIMENT_OUTPUT).set_index("id")
    assert sent.loc["2", "text"] == "t2" and pd.isna(sent.loc["2", "score_llm"])
    assert sent.loc["1", "score_llm"] == 0.3
    arch = pd.read_parquet(superseded_path(settings.SENTIMENT_OUTPUT))
    assert arch["id"].tolist() == ["2"] and arch["text"].tolist() == ["cut"]
    assert arch["score_llm"].tolist() == [0.4]


def test_force_that_fetches_nothing_keeps_the_cache(raw_dir):
    """Before the merge, a forced run that got nothing back (X returned no
    posts in the window) wrote an empty cache."""
    t0 = _parse("2026-06-01T00:00:00Z")
    cache = raw_dir / "acct.jsonl"
    recs = [_cached(1, t0, "kept"), _cached(2, t0 + timedelta(hours=1), "kept too")]
    _write_cache(cache, recs)
    got = xc.collect_user(
        StubClient([]), "acct", "tier", start=t0, end=t0 + timedelta(days=1), force=True
    )
    assert got == recs and _read_cache(cache) == recs


def test_forced_search_keeps_posts_older_than_its_reach(raw_dir):
    """/search/recent reaches back ~7 days: a forced search used to drop
    every cached result older than that."""
    now = datetime.now(UTC).replace(tzinfo=None)
    cache = xc._search_cache_path("iran war")
    _write_cache(cache, [_cached(50, now - timedelta(days=30), "old"), _cached(5, now, "x")])
    client = StubClient([(5, now - timedelta(days=1)), (6, now - timedelta(hours=1))])
    got = xc.collect_search(client, "iran war", max_total=10, force=True)
    assert [t["id"] for t in got] == ["50", "5", "6"]
    assert got[0]["text"] == "old" and got[1]["text"] == "t5"
    assert _read_cache(cache) == got


def test_skip_when_cache_is_current(raw_dir):
    end = datetime(2026, 9, 15)
    cache = raw_dir / "acct.jsonl"
    cache.write_text(
        json.dumps({"id": "1", "created_at": end.replace(tzinfo=UTC).isoformat()}) + "\n"
    )
    client = StubClient([])
    got = xc.collect_user(client, "acct", "tier", start=datetime(2026, 2, 1), end=end)
    assert len(got) == 1 and client.calls == 0


# ── record fields: note_tweet text, referenced tweets ───────────────

_LONG = "word " * 80  # a post over 280 characters, whole only in note_tweet


def test_timeline_records_note_text_and_refs(raw_dir):
    t0 = _parse("2026-06-01T00:00:00Z")
    client = StubClient(
        [(1, t0), (2, t0 + timedelta(hours=1)), (3, t0 + timedelta(hours=2))],
        extra={
            1: {"note_tweet": {"text": _LONG.strip()}},
            2: {"referenced_tweets": [{"type": "retweeted", "id": "99"}]},
        },
        texts={1: _LONG[:270] + "…", 2: "RT @other: short"},
    )
    got = {
        t["id"]: t
        for t in xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=1))
    }

    assert got["1"]["text"] == _LONG.strip()  # full text, not the 280 cut
    assert got["2"]["ref"] == [{"type": "retweeted", "id": "99"}]
    assert got["2"]["text"] == "RT @other: short"
    assert "ref" not in got["1"] and "ref" not in got["3"]  # only when present
    assert set(got["3"]) == {
        "id",
        "user",
        "tier",
        "text",
        "created_at",
        "metrics",
        "lang",
        "platform",
    }  # old shape otherwise
    lines = (raw_dir / "acct.jsonl").read_text().splitlines()
    on_disk = {json.loads(x)["id"]: json.loads(x) for x in lines}
    assert on_disk["2"]["ref"][0]["id"] == "99"
    # same-post fields only: no expansions argument (the stub would reject one)
    assert all({"note_tweet", "referenced_tweets"} <= set(f) for f in client.fields)


def test_retweet_note_text_keeps_rt_prefix():
    tweet = SimpleNamespace(
        text="RT @other: start of a long…", data={"note_tweet": {"text": "start of a long post"}}
    )
    assert xc._full_text(tweet) == "RT @other: start of a long post"
    plain = SimpleNamespace(text="short", data={"id": "1", "text": "short"})
    assert xc._full_text(plain) == "short"


def test_search_records_note_text_and_refs(raw_dir):
    t0 = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
    client = StubClient(
        [(5, t0), (6, t0 + timedelta(hours=1))],
        extra={
            5: {
                "note_tweet": {"text": "the whole post"},
                "referenced_tweets": [{"type": "quoted", "id": "8"}],
            }
        },
    )
    got = {t["id"]: t for t in xc.collect_search(client, "iran war", max_total=10)}
    assert got["5"]["text"] == "the whole post"
    assert got["5"]["ref"] == [{"type": "quoted", "id": "8"}]
    assert "ref" not in got["6"] and got["6"]["author_id"] == "7"
    assert {"note_tweet", "referenced_tweets", "author_id"} <= set(client.fields[0])


# ── empty windows: horizon and possible fetch failures ─────────────


def test_horizon_warning(raw_dir, monkeypatch, caplog):
    """The first windows come back empty and the rest are dense (a post every
    2 minutes: ~1,050 per window, extrapolated from the 5 kept): the gap
    started past the ~3,200-post horizon. One warning, not one per window."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    end = t0 + 8 * full
    posts = _every(t0 + 3 * full + timedelta(minutes=7), end, timedelta(minutes=2))
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_user(StubClient(posts), "acct", "tier", start=t0, end=end)
    assert len(got) == 5 * 5
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1
    assert "horizon" in warns[0] and "first 3 window(s)" in warns[0]
    assert f"{xc._hm(t0)}..{xc._hm(t0 + 3 * full)}" in warns[0]


def test_quiet_start_is_not_the_horizon(raw_dir, monkeypatch, caplog):
    """A sampled account silent for the first window after its cursor, then
    a post every 40 minutes (truncated windows, ~50 posts each): far too few
    after it for the horizon, so no 'lost' warning. The empty window sits
    next to a dense truncated one, which is flagged as a possible failed
    fetch."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 5})
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    end = t0 + 6 * full
    _write_cache(raw_dir / "acct.jsonl", [_cached("c0", t0 - timedelta(seconds=1), "old")])
    posts = _every(t0 + timedelta(hours=36), end, timedelta(minutes=40))
    with caplog.at_level("WARNING", logger=xc.logger.name):
        xc.collect_user(StubClient(posts), "acct", "tier", start=t0, end=end)
    assert [r["truncated"] for r in _manifest_rows()] == [False] + [True] * 5
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not [m for m in warns if "horizon" in m]
    assert len(warns) == 1 and "possible fetch failure" in warns[0]
    assert f"{xc._hm(t0)}..{xc._hm(t0 + full)}" in warns[0]


def test_quiet_start_then_steady_has_no_warning(raw_dir, caplog):
    """A take-all account with its first three windows empty, then one post
    per window, none truncated: nothing to warn about."""
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    posts = [(k, t0 + k * full + timedelta(hours=10)) for k in range(3, 7)]
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_user(StubClient(posts), "acct", "tier", start=t0, end=t0 + 7 * full)
    assert len(got) == 4
    assert [r["got"] for r in _manifest_rows()] == [0, 0, 0, 1, 1, 1, 1]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_empty_window_next_to_dense_truncated_warns(raw_dir, monkeypatch, caplog):
    """Three 35 h windows, cap 10 each: window 1 is dense (truncated, its 10
    newest posts span 90 minutes), window 2 returns nothing, window 3 is
    quiet. Window 2 is flagged, window 3 is not."""
    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"acct": 10})
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    tweets = _every(t0 + timedelta(minutes=7), t0 + full, timedelta(minutes=10)) + [
        (1000 + k, t0 + 2 * full + timedelta(hours=3 + k)) for k in range(3)
    ]
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_user(StubClient(tweets), "acct", "tier", start=t0, end=t0 + 3 * full)
    assert len(got) == 13
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1
    assert "@acct" in warns[0] and "possible fetch failure" in warns[0]
    assert f"{xc._hm(t0 + full)}..{xc._hm(t0 + 2 * full)}" in warns[0]


def test_quiet_account_empty_window_is_quiet(raw_dir, monkeypatch, caplog):
    """An empty window between quiet ones, or next to a truncated window
    whose kept posts span more than a quarter of a window (a steady
    account), is a silent account, not a failed fetch."""
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    quiet = [(1, t0 + timedelta(hours=10)), (2, t0 + 2 * full + timedelta(hours=10))]
    with caplog.at_level("WARNING", logger=xc.logger.name):
        xc.collect_user(StubClient(quiet), "acct", "tier", start=t0, end=t0 + 3 * full)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]

    monkeypatch.setattr(settings, "ACCOUNT_WINDOW_CAPS", {"steady": 5})
    steady = [(100 + k, t0 + timedelta(hours=1 + 4 * k)) for k in range(8)]  # 8 posts 4 h apart
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_user(StubClient(steady), "steady", "tier", start=t0, end=t0 + 3 * full)
    assert len(got) == 5
    assert [r["truncated"] for r in _manifest_rows() if r["handle"] == "steady"] == [
        True,
        False,
        False,
    ]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ── the per-window manifest ────────────────────────────────────────


def test_window_manifest(tmp_path, monkeypatch):
    """One row per window, in X_SAMPLING_DIR beside (never inside) the X
    cache, so nothing that globs the cache reads it."""
    assert not settings.X_SAMPLING_DIR.is_relative_to(settings.X_RAW_DIR)  # the real paths

    from src.collectors import x_backfill as xb

    raw = tmp_path / "raw"
    for name, val in [
        ("DATA_DIR", tmp_path),
        ("RAW_DIR", raw),
        ("X_RAW_DIR", raw / "x"),
        ("X_SAMPLING_DIR", raw / "x_sampling"),
        ("PROCESSED_DIR", tmp_path / "processed"),
        ("ACCOUNT_CAP_OVERRIDES", {}),
        ("ACCOUNT_WINDOW_CAPS", {"acct": 5}),
    ]:
        monkeypatch.setattr(settings, name, val)
    t0 = _parse("2026-06-01T00:00:00Z")
    full = timedelta(hours=35)
    end = t0 + 3 * full
    posts = _every(t0 + timedelta(minutes=7), t0 + full, timedelta(minutes=20)) + [
        (500, t0 + 2 * full + timedelta(hours=1))
    ]
    xc.collect_user(StubClient(posts), "acct", "maga", start=t0, end=end)

    files = list(settings.X_SAMPLING_DIR.glob("*.jsonl"))
    assert [f.name for f in files] == [f"windows_{end:%Y-%m-%d}.jsonl"]
    assert not files[0].is_relative_to(settings.X_RAW_DIR)
    rows = _manifest_rows()
    assert len(rows) == 3
    assert set(rows[0]) == {
        "run_end",
        "handle",
        "tier",
        "mode",
        "hours",
        "window_start",
        "window_end",
        "cap",
        "got",
        "new",
        "truncated",
        "oldest_kept",
        "newest_kept",
    }
    assert [(r["window_start"], r["window_end"]) for r in rows] == [
        (xc._iso(t0 + i * full), xc._iso(t0 + (i + 1) * full)) for i in range(3)
    ]
    assert {(r["run_end"], r["handle"], r["tier"], r["mode"], r["hours"]) for r in rows} == {
        (xc._iso(end), "acct", "maga", "sample", 35)
    }
    assert [(r["cap"], r["got"], r["new"], r["truncated"]) for r in rows] == [
        (5, 5, 5, True),
        (5, 0, 0, False),
        (5, 1, 1, False),
    ]
    assert rows[1]["oldest_kept"] is None and rows[1]["newest_kept"] is None
    assert _parse(rows[0]["newest_kept"]) == posts[104][1]  # the window's newest post
    assert _parse(rows[0]["oldest_kept"]) == posts[100][1]

    # A row there that would pass for a post cut at 280 if anything globbed it.
    fake = {
        "id": "999",
        "user": "acct",
        "tier": "maga",
        "text": "x " * 138 + "an",
        "created_at": "2026-05-01T00:00:00+00:00",
    }
    assert xb.likely_cut(fake["text"])
    with open(files[0], "a") as f:
        f.write(json.dumps(fake) + "\n")
    cached = xc.load_all_cached()
    assert len(cached) == 6 and "999" not in {p["id"] for p in cached}
    assert "999" not in {c["id"] for c in xb.candidates()}


# ── the CLI ────────────────────────────────────────────────────────


def test_collect_estimate_makes_no_calls_and_only_filters(raw_dir, monkeypatch):
    from click.testing import CliRunner

    from src import cli

    for name in ("RAW_DIR", "TRUTH_SOCIAL_RAW_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(settings, name, raw_dir / name.lower())
    monkeypatch.setattr(cli, "X_ACCOUNTS", {"t1": ["a", "b"], "t2": ["c"]})
    monkeypatch.setattr(xc, "get_client", _no_client)
    monkeypatch.setattr(xc.tweepy, "Client", _no_client)

    res = CliRunner().invoke(cli.main, ["collect", "--estimate", "--no-search", "--only", "a"])
    assert res.exit_code == 0, res.output
    rows = [line for line in res.output.splitlines() if line.startswith("  @")]
    assert len(rows) == 1 and rows[0].split()[0] == "@a"
    assert all(col in res.output for col in (" h ", "windows", "cap/win", "per14d"))
    reads = settings.MAX_TWEETS_PER_USER
    cost = reads * settings.X_READ_COST_USD + settings.X_USER_READ_COST_USD
    assert f"Maximum this run: {reads} reads + 1 user lookups ≈ ${cost:.2f}" in res.output

    res = CliRunner().invoke(cli.main, ["collect", "--estimate", "--only", "nosuch"])
    assert res.exit_code != 0 and "nosuch" in res.output

    # `--only "$HANDLES"` with the variable unset must not mean every account
    for empty in ("", " , ", "@"):
        res = CliRunner().invoke(
            cli.main, ["collect", "--estimate", "--no-search", "--only", empty]
        )
        assert res.exit_code == 2 and "no handles" in res.output, (empty, res.output)
        assert "Maximum this run" not in res.output
