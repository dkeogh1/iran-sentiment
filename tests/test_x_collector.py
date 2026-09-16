"""
Gap-fill slicing and run-planning for the X collector, tested against a
stub tweepy client (no network, no spend).
"""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from config import settings
from src.collectors import x_collector as xc


# ── stub client ────────────────────────────────────────────────────

def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


class StubClient:
    """Serves synthetic tweets newest-first with 100-per-page pagination,
    exactly like GET /2/users/:id/tweets."""

    def __init__(self, tweets):
        # tweets: list of (id, created_at naive-UTC datetime)
        self.tweets = sorted(tweets, key=lambda t: t[1], reverse=True)
        self.calls = 0

    def get_user(self, username):
        return SimpleNamespace(data=SimpleNamespace(id=42))

    def get_users_tweets(self, id, start_time, end_time, max_results,
                         pagination_token, tweet_fields):
        self.calls += 1
        s, e = _parse(start_time), _parse(end_time)
        hits = [t for t in self.tweets if s <= t[1] < e]
        offset = int(pagination_token or 0)
        page = hits[offset:offset + max_results]
        data = [SimpleNamespace(id=i, text=f"t{i}", created_at=ts.replace(tzinfo=timezone.utc),
                                public_metrics={"like_count": 1}, lang="en")
                for i, ts in page]
        nxt = offset + max_results
        meta = {"next_token": str(nxt)} if nxt < len(hits) else {}
        return SimpleNamespace(data=data, meta=meta)


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
    assert dates[0] < gap_start + timedelta(days=14)          # oldest slice present
    assert dates[-1] > end - timedelta(days=14)               # newest slice present
    # cap shared across 9 slices (8 full + 1 short): 100 = 8*11 + 12
    per_slice = xc.split_cap(100, 9)
    assert sum(per_slice) == 100


def test_incremental_appends_only_new(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 14)
    t0 = datetime(2026, 5, 1)
    cache = raw_dir / "acct.jsonl"
    cache.write_text(json.dumps({"id": "old", "user": "acct", "tier": "tier", "text": "x",
                                 "created_at": (t0 + timedelta(days=9)).replace(tzinfo=timezone.utc).isoformat()}) + "\n")
    # 1/day for 40 days; only days 10..39 are newer than the cache
    client = StubClient(_stream(t0, 40, per_day=1))

    got = xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=40))

    assert got[0]["id"] == "old"
    new = got[1:]
    assert len(new) == 30
    assert min(_parse(t["created_at"]) for t in new) > t0 + timedelta(days=9)
    assert sum(1 for _ in open(cache)) == 31            # appended, not overwritten
    plan = xc.plan_account("acct", start=t0, end=t0 + timedelta(days=40))
    assert plan["incremental"] and plan["window"][0] > t0 + timedelta(days=9)


def test_uncapped_slice_fetches_everything(raw_dir, monkeypatch):
    monkeypatch.setattr(settings, "GAP_FILL_SLICE_DAYS", 7)
    t0 = datetime(2026, 6, 1)
    client = StubClient(_stream(t0, 21, per_day=3))     # 63 tweets, cap 500
    got = xc.collect_user(client, "acct", "tier", start=t0, end=t0 + timedelta(days=21))
    assert len(got) == 63
    assert len({t["id"] for t in got}) == 63            # slices don't overlap


def test_skip_when_cache_is_current(raw_dir):
    end = datetime(2026, 9, 15)
    cache = raw_dir / "acct.jsonl"
    cache.write_text(json.dumps({"id": "1", "created_at": end.replace(tzinfo=timezone.utc).isoformat()}) + "\n")
    client = StubClient([])
    got = xc.collect_user(client, "acct", "tier", start=datetime(2026, 2, 1), end=end)
    assert len(got) == 1 and client.calls == 0
