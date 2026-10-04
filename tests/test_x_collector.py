"""
Gap-fill slicing and run-planning for the X collector, tested against a
stub tweepy client (no network, no spend).
"""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from config import settings
from src.collectors import x_collector as xc

# ── stub client ────────────────────────────────────────────────────


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


class StubClient:
    """Serves synthetic tweets newest-first with 100-per-page pagination,
    exactly like GET /2/users/:id/tweets. `extra` maps a tweet id to raw
    API fields (note_tweet, referenced_tweets) that tweepy leaves in .data;
    `texts` overrides a tweet's truncated `text`."""

    def __init__(self, tweets, extra=None, texts=None):
        # tweets: list of (id, created_at naive-UTC datetime)
        self.tweets = sorted(tweets, key=lambda t: t[1], reverse=True)
        self.extra, self.texts = extra or {}, texts or {}
        self.calls = 0
        self.fields = []

    def get_user(self, username):
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
        self.fields.append(tweet_fields)
        s, e = _parse(start_time), _parse(end_time)
        hits = [t for t in self.tweets if s <= t[1] < e]
        offset = int(pagination_token or 0)
        page = hits[offset : offset + max_results]
        data = [self._tweet(i, ts) for i, ts in page]
        nxt = offset + max_results
        meta = {"next_token": str(nxt)} if nxt < len(hits) else {}
        return SimpleNamespace(data=data, meta=meta)

    def search_recent_tweets(self, query, max_results, next_token, tweet_fields, start_time=None):
        self.calls += 1
        self.fields.append(tweet_fields)
        return SimpleNamespace(
            data=[self._tweet(i, ts) for i, ts in self.tweets[:max_results]], meta={}
        )


def _stream(start: datetime, days: int, per_day: int):
    """`per_day` tweets on every day of [start, start+days)."""
    out, i = [], 0
    for d in range(days):
        for k in range(per_day):
            out.append((i, start + timedelta(days=d, hours=k)))
            i += 1
    return out


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "X_RAW_DIR", tmp_path)
    monkeypatch.setattr(settings, "ACCOUNT_CAP_OVERRIDES", {})
    return tmp_path


# ── pure planning ──────────────────────────────────────────────────


def test_plan_slices_contiguous_oldest_first():
    s, e = datetime(2026, 5, 10), datetime(2026, 9, 15)
    slices = xc.plan_slices(s, e, slice_days=14)
    assert slices[0][0] == s and slices[-1][1] == e
    assert all(a[1] == b[0] for a, b in zip(slices, slices[1:]))
    assert all(b - a <= timedelta(days=14) for a, b in slices)
    assert slices == sorted(slices)
    assert xc.plan_slices(e, s) == []


def test_split_cap_sums_exactly_and_favours_newest():
    assert sum(xc.split_cap(500, 9)) == 500
    assert xc.split_cap(10, 3) == [3, 3, 4]
    assert xc.split_cap(0, 3) == [0, 0, 0]
    assert xc.split_cap(5, 0) == []


def test_estimate_run_is_a_hard_maximum(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "MAX_TWEETS_PER_USER", 100)
    monkeypatch.setattr(settings, "MAX_TWEETS_PER_SEARCH", 50)
    monkeypatch.setattr(settings, "ACCOUNT_CAP_OVERRIDES", {"heavy": 20})
    plan = xc.estimate_run({"t": ["a", "heavy"]}, ["q"])
    assert plan["max_reads"] == 100 + 20 + 50
    assert plan["max_cost_usd"] == pytest.approx(170 * settings.X_READ_COST_USD)
    by = {p["handle"]: p for p in plan["accounts"]}
    assert by["heavy"]["cap"] == 20 and sum(by["heavy"]["caps"]) == 20


# ── fetching against the stub ──────────────────────────────────────


def test_gap_fill_has_no_hole(raw_dir, monkeypatch):
    """The failure this exists to prevent: a 120-day gap, 10 posts/day,
    cap 100. The old newest-first fetch kept the last 10 days only."""
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 14)
    gap_start = datetime(2026, 5, 10)
    end = gap_start + timedelta(days=120)
    client = StubClient(_stream(gap_start, 120, per_day=10))

    got = xc.collect_user(client, "acct", "tier", start=gap_start, end=end, max_tweets=100)

    assert len(got) == 100
    dates = sorted(_parse(t["created_at"]) for t in got)
    # every 14-day slice got its share -> no gap wider than one slice
    gaps = [b - a for a, b in zip(dates, dates[1:])]
    assert max(gaps) < timedelta(days=14)
    assert dates[0] < gap_start + timedelta(days=14)  # oldest slice present
    assert dates[-1] > end - timedelta(days=14)  # newest slice present
    # cap shared across 9 slices (8 full + 1 short): 100 = 8*11 + 12
    per_slice = xc.split_cap(100, 9)
    assert sum(per_slice) == 100


def test_incremental_appends_only_new(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 14)
    t0 = datetime(2026, 5, 1)
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
    assert sum(1 for _ in open(cache)) == 31  # appended, not overwritten
    plan = xc.plan_account("acct", start=t0, end=t0 + timedelta(days=40))
    assert plan["incremental"] and plan["window"][0] > t0 + timedelta(days=9)


def test_uncapped_slice_fetches_everything(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 7)
    t0 = datetime(2026, 6, 1)
    client = StubClient(_stream(t0, 21, per_day=3))  # 63 tweets, cap 500
    got = xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=21))
    assert len(got) == 63
    assert len({t["id"] for t in got}) == 63  # slices don't overlap
    assert client.calls == 3  # the patched 7-day slices applied


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


# ── empty slice next to a capped one ───────────────────────────────


def test_empty_slice_next_to_capped_slice_warns(raw_dir, monkeypatch, caplog):
    """Three 7-day slices, cap 10 each: slice 1 is dense (capped), slice 2
    returns nothing, slice 3 is quiet. Slice 2 is flagged, slice 3 is not."""
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 7)
    t0 = _parse("2026-06-01T00:00:00Z")
    tweets = _stream(t0, 7, per_day=3) + [
        (100 + k, t0 + timedelta(days=15, hours=k)) for k in range(3)
    ]
    client = StubClient(tweets)
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_user(
            client, "acct", "tier", start=t0, end=t0 + timedelta(days=21), max_tweets=30
        )
    assert len(got) == 13
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 1
    assert "@acct" in warns[0] and str(t0 + timedelta(days=7)) in warns[0]


def test_empty_slice_without_capped_neighbour_is_quiet(raw_dir, monkeypatch, caplog):
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 7)
    t0 = _parse("2026-06-01T00:00:00Z")
    tweets = [(1, t0 + timedelta(days=1)), (2, t0 + timedelta(days=15))]
    with caplog.at_level("WARNING", logger=xc.logger.name):
        xc.collect_user(
            StubClient(tweets), "acct", "tier", start=t0, end=t0 + timedelta(days=21), max_tweets=30
        )
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_zero_cap_slice_is_not_flagged(raw_dir, monkeypatch, caplog):
    """max_tweets 2 over three slices gives caps [0, 1, 1]: the first slice
    makes no request, so it cannot be a failed fetch, capped neighbour or not."""
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 7)
    t0 = _parse("2026-06-01T00:00:00Z")
    tweets = [
        (1, t0 + timedelta(days=1)),
        (2, t0 + timedelta(days=8)),
        (3, t0 + timedelta(days=15)),
    ]
    client = StubClient(tweets)
    with caplog.at_level("WARNING", logger=xc.logger.name):
        got = xc.collect_user(
            client, "acct", "tier", start=t0, end=t0 + timedelta(days=21), max_tweets=2
        )
    assert sorted(t["id"] for t in got) == ["2", "3"] and client.calls == 2
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
