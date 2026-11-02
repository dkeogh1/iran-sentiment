"""Per-model prices and tags, the batch results download and delete, and the
reply-label jobs (v2b, cat-v2, the fresh draws), all with stubs: no API, no
network, no API key."""

import json
import types

import httpx
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings
from src.analysis import batch_io
from src.analysis import inference as inf
from src.analysis import relabel as rl
from src.analysis import reply_labels as rlab
from src.analysis import stance_local as sl
from src.analysis.pricing import model_price, model_tag, token_usd
from tests.test_teacher_checks import (
    GOOD,
    POSTS,
    _api_line,
    _content_text,
    _ended,
    _stub_client,
    files,  # noqa: F401 -- the fixture
)

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

MODEL = settings.TEACHER_CHECK_MODEL
PARAMS = {"seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
PATTERN = (-0.9, 0.0, 0.9, -0.9)  # half critical, a quarter mid, a quarter supportive


def _replies(n_per_post=40):
    rows = []
    for slug, (pid, _) in POSTS.items():
        for i in range(n_per_post):
            rows.append(
                {
                    "id": f"{slug}-{i}",
                    "tracked_slug": slug,
                    "parent_id": pid,
                    "user": f"handle_{slug}_{i}",
                    "text": f"reply number {i} to {slug}",
                    "score_transformer": PATTERN[i % len(PATTERN)],
                    "score_opus_distilled": 0.1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Replies, the first draws (2 per bucket per post) as the stance sample
    and its v2 labels, and the Trump feed, in tmp_path; no API key."""
    from src.analysis import event_study as es
    from src.analysis.event_study import bucket_draws

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    monkeypatch.setattr(settings, "REPLY_DRAWS_MANIFEST", tmp_path / "draws.parquet")
    monkeypatch.setattr(settings, "TRUTH_SOCIAL_RAW_DIR", tmp_path / "ts")
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(es, "STANCE_OUTPUT", tmp_path / "stance_sample.parquet")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "ts").mkdir()
    with open(tmp_path / "ts" / "realDonaldTrump.jsonl", "w") as f:
        f.writelines(
            json.dumps({"id": pid, "user": "realDonaldTrump", "text": text}) + "\n"
            for pid, text in POSTS.values()
        )

    def write(reps=None, n_per_bucket=2, v2_score=-0.5):
        reps = _replies() if reps is None else reps
        reps.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
        drawn = bucket_draws(reps, n_per_bucket=n_per_bucket, score_col="score_transformer")
        drawn.assign(text=drawn["text"].str[:100], stance="neutral_other")[
            ["id", "tracked_slug", "user", "text", "score_transformer", "stance"]
        ].to_parquet(es.STANCE_OUTPUT, index=False)
        pd.DataFrame(
            {
                "id": drawn["id"],
                "tracked_slug": drawn["tracked_slug"],
                "bucket": inf.stance_bucket(drawn["score_transformer"]).astype(int),
            }
        ).to_parquet(settings.REPLY_DRAWS_MANIFEST, index=False)
        pd.DataFrame(
            {
                "id": drawn["id"],
                "tracked_slug": drawn["tracked_slug"],
                "input_chars": drawn["text"].str.len(),
                "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
                "score_teacher": v2_score,
                "label_teacher": "negative",
            }
        ).to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)
        return reps, drawn

    return write


def _no_api(monkeypatch):
    import anthropic

    def boom(*a, **k):
        raise AssertionError("an API call")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(sl, "_batch_client", boom)
    monkeypatch.setattr(sl, "_teacher_scorer", boom)
    monkeypatch.setattr(batch_io, "_http", boom)


# ── prices and tags ────────────────────────────────────────────────


def test_every_estimator_prices_a_model_by_its_own_row(monkeypatch):
    from src.analysis import topic_label as tl

    o5 = rl.estimate_cost(100, "claude-opus-5")
    assert rl.estimate_cost(100, "claude-fable-5-1") == pytest.approx(2 * o5)
    assert rl.estimate_cost(100, "claude-opus-5-5") == pytest.approx(0.8 * o5)
    assert rl.estimate_cost(100) == pytest.approx(o5)
    assert tl.estimate_cost(10) == pytest.approx(10 * (225 * 1 + 12 * 5) / 1e6 * 0.5)
    assert model_price("claude-haiku-4-5-20251001") is settings.MODEL_PRICES["claude-haiku-4-5"]
    with pytest.raises(KeyError, match="claude-sonnet-9"):
        rl.estimate_cost(1, "claude-sonnet-9")
    # Cache reads: 0.1x on Opus 5, 0.05x on Opus 5.5, 0.025x on Fable 5.1.
    t = {"input_tokens": 1000, "cache_read_input_tokens": 10_000, "output_tokens": 100}
    assert token_usd(t, "claude-opus-5") == pytest.approx((1000 * 5 + 1000 * 5 + 100 * 25) / 1e6)
    assert token_usd(t, "claude-opus-5-5") == pytest.approx((1000 * 4 + 500 * 4 + 100 * 20) / 1e6)
    assert token_usd(t, "claude-fable-5-1") == pytest.approx(
        (1000 * 10 + 250 * 10 + 100 * 50) / 1e6
    )
    assert token_usd(t, "claude-opus-5-5", batch=True) == pytest.approx(
        token_usd(t, "claude-opus-5-5") / 2
    )
    spend = sl.Spend("claude-opus-5-5")
    spend.add(types.SimpleNamespace(**t, cache_creation_input_tokens=0))
    assert spend.usd() == pytest.approx(token_usd(t, "claude-opus-5-5"))
    # The direct estimate takes the model's prices and cache multipliers too.
    monkeypatch.setattr(settings, "TEACHER_EST_CHARS_PER_TOKEN", 1.0)
    prompts, cache = ["p" * 600 + "r" * 10] * 6, [600] * 6
    e5 = sl.estimate_direct_cost(prompts, cache, concurrency=2)
    e55 = sl.estimate_direct_cost(prompts, cache, concurrency=2, model="claude-opus-5-5")
    billed55 = 2 * 600 * 1.25 + 4 * 600 * 0.05 + 6 * 10
    assert e55["usd"] == pytest.approx((billed55 * 4 + e55["output_tokens"] * 20) / 1e6)
    assert e55["usd_no_cache"] == pytest.approx(0.8 * e5["usd_no_cache"])
    assert e55["model"] == "claude-opus-5-5" and e55["price_in_per_mtok"] == 4.0
    haiku = sl.estimate_direct_cost(prompts, cache, model="claude-haiku-4-5")  # 4,096 minimum
    assert haiku["cached_prefix_calls"] == 0
    p55 = sl.with_pilot_and_batch_prices(e55)
    pilot = {f: settings.TEACHER_V2_PILOT[f] for f in sl.Spend.FIELDS}
    assert p55["pilot_usd_per_call"] == pytest.approx(token_usd(pilot, "claude-opus-5-5") / 16)


def test_model_tags_never_collide():
    assert rl.tag_for("claude-opus-5") == "opus"
    assert rl.tag_for("claude-opus-5-5") == "opus55"
    assert rl.tag_for("claude-fable-5-1") == "fable51"
    assert rl.tag_for(settings.LLM_MODEL) == "haiku" == model_tag("claude-haiku-4-5")
    models = [
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-haiku-4-5",
        "claude-opus-4-8",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
    ]
    assert len({model_tag(m) for m in models}) == len(models)


def test_merge_refuses_a_column_another_model_wrote(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    df = pd.DataFrame({"id": ["1", "2"], "text": ["a post", "b post"]})
    for m, s in (("claude-opus-5", 0.5), ("claude-opus-5-5", -0.5)):
        pd.DataFrame({"id": ["1"], "score_teacher": [s], "label_teacher": ["x"]}).to_parquet(
            rl.labels_path(m), index=False
        )
    out = rl.merge(rl.merge(df, "claude-opus-5"), "claude-opus-5-5")
    assert out["score_opus"].iloc[0] == 0.5 and out["score_opus55"].iloc[0] == -0.5
    assert json.loads(rl.columns_path().read_text()) == {
        "score_opus": "claude-opus-5",
        "score_opus55": "claude-opus-5-5",
    }
    # Were the tags ever edited to collide, the merge refuses rather than overwrite.
    monkeypatch.setattr(settings, "MODEL_TAGS", {**settings.MODEL_TAGS, "claude-opus-5-5": "opus"})
    with pytest.raises(ValueError, match="score_opus holds labels from claude-opus-5"):
        rl.merge(out, "claude-opus-5-5")
    rl.columns_path().unlink()  # score_opus is Opus 5's even with no record
    with pytest.raises(ValueError, match="not claude-opus-5-5"):
        rl.merge(out, "claude-opus-5-5")


# ── results download and batch delete ──────────────────────────────


class _Resp:
    def __init__(self, http, url, status, data=b"", location=None):
        self.http, self.url, self.status_code, self.data = http, url, status, data
        self.headers = {"location": location} if location else {}
        if http.encoding and status < 300:
            self.headers["content-encoding"] = http.encoding

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("GET", self.url)
            raise httpx.HTTPStatusError(
                "err", request=req, response=httpx.Response(self.status_code, request=req)
            )

    def iter_bytes(self):
        if self.http.cut_at is not None:
            cut, self.http.cut_at = self.http.cut_at, None
            yield self.data[:cut]
            raise httpx.ReadError("connection dropped")
        if self.http.undecodable:
            self.http.undecodable = False
            yield self.data[:3]
            raise httpx.DecodingError("bad gzip stream")
        yield self.data


class StubHTTP:
    """results_url without the network: serves `body`, honours Range (416
    past its end), can drop the first connection after cut_at bytes, send
    one undecodable body, redirect once, answer a status, or mark every
    body with a content-encoding (as a server ignoring identity would)."""

    def __init__(
        self, body=b"", cut_at=None, redirect=None, status=None, encoding=None, undecodable=False
    ):
        self.body, self.cut_at, self.redirect, self.status = body, cut_at, redirect, status
        self.encoding, self.undecodable = encoding, undecodable
        self.calls = []

    def stream(self, method, url, headers):
        self.calls.append((url, dict(headers)))
        if self.status:
            return _Resp(self, url, self.status)
        if self.redirect and len(self.calls) == 1:
            return _Resp(self, url, 302, location=self.redirect)
        rng = headers.get("range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(self.body):
                return _Resp(self, url, 416)
            return _Resp(self, url, 206, self.body[start:])
        return _Resp(self, url, 200, self.body)

    def close(self):
        pass


class BrokenStream:
    """The SDK's results stream cut mid-body, as on this host (`error`: what
    the cut raises)."""

    def __init__(self, inner, error=None):
        self.inner = inner
        self.error = error or httpx.ReadError("peer closed connection")

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def results(self, batch_id):
        yield from list(self.inner.results(batch_id))[:2]
        raise self.error


def test_collect_downloads_results_url_when_the_stream_breaks(files, monkeypatch):  # noqa: F811
    files()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(batch_io.time, "sleep", lambda s: None)
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=PARAMS, client=client)
    body = "".join(_api_line(i, GOOD) for i in st["ids"]).encode()
    http = StubHTTP(body, cut_at=len(body) // 3)
    monkeypatch.setattr(batch_io, "_http", lambda: http)
    _ended(client).messages.batches = BrokenStream(client.messages.batches)
    st, rep = sl.teacher_batch_collect("reply_v2", client=client)
    assert st["status"] == "collected" and st["collected"]["labelled"] == 10 and rep["n"] == 10
    assert st["spend"]["calls"] == 10  # usage read from the downloaded lines
    (url0, h0), (url1, h1) = http.calls
    assert url0 == url1 and url0.endswith("/msgbatch_1/results")
    assert h0["x-api-key"] == "test-key" and h0["anthropic-version"] == "2023-06-01"
    assert h0["accept-encoding"] == "identity"  # bytes held = the offset Range counts
    assert "range" not in h0 and h1["range"] == f"bytes={len(body) // 3}-"  # resumed, not restarted
    kept = sl.batch_archive_path("reply_v2", "msgbatch_1", "_results.jsonl")
    assert kept.read_bytes() == body
    assert not list(settings.PROCESSED_DIR.glob("*.part"))


def test_download_redirect_drops_the_key_and_4xx_is_not_retried(tmp_path, monkeypatch):
    client = _ended(_stub_client())
    client.messages.batches.created.append([])
    client.api_key = "k"
    http = StubHTTP(b"line\n", redirect="https://storage.other.test/blob?sig=1")
    out = batch_io.download_results(client, "msgbatch_1", tmp_path / "r.jsonl", http=http)
    assert out.read_bytes() == b"line\n"
    assert http.calls[0][1]["x-api-key"] == "k" and "x-api-key" not in http.calls[1][1]
    slept = []
    bad = StubHTTP(status=403)
    with pytest.raises(httpx.HTTPStatusError):
        batch_io.download_results(
            client, "msgbatch_1", tmp_path / "s.jsonl", http=bad, sleep=slept.append
        )
    assert len(bad.calls) == 1 and slept == []
    flaky = StubHTTP(status=503)
    with pytest.raises(httpx.HTTPStatusError):
        batch_io.download_results(
            client, "msgbatch_1", tmp_path / "t.jsonl", http=flaky, retries=2, sleep=slept.append
        )
    assert len(flaky.calls) == 3 and slept == [1, 2]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client.api_key = None
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        batch_io.download_results(client, "msgbatch_1", tmp_path / "u.jsonl", http=http)


def test_download_recovery_branches(tmp_path, monkeypatch):
    client = _ended(_stub_client())
    client.messages.batches.created.append([])
    client.api_key = "k"
    body = b"".join(f"line {i}\n".encode() for i in range(50))
    # A whole .part (killed before the rename): the resume gets 416 and keeps it.
    dest = tmp_path / "a.jsonl"
    dest.with_name("a.jsonl.part").write_bytes(body)
    http = StubHTTP(body)
    assert batch_io.download_results(client, "msgbatch_1", dest, http=http).read_bytes() == body
    assert http.calls[0][1]["range"] == f"bytes={len(body)}-" and len(http.calls) == 1
    # An encoded body cut short leaves nothing held: the retry starts over.
    http = StubHTTP(body, cut_at=40, encoding="gzip")
    out = batch_io.download_results(
        client, "msgbatch_1", tmp_path / "b.jsonl", http=http, sleep=lambda s: None
    )
    assert out.read_bytes() == body and "range" not in http.calls[1][1]
    # An encoded 206 to a resume: what is held is dropped and fetched whole.
    dest = tmp_path / "c.jsonl"
    dest.with_name("c.jsonl.part").write_bytes(body[:25])
    http = StubHTTP(body, encoding="gzip")
    assert batch_io.download_results(client, "msgbatch_1", dest, http=http).read_bytes() == body
    assert [("range" in h) for _, h in http.calls] == [True, False]
    # A body that will not decode: dropped and retried from the start.
    http = StubHTTP(body, undecodable=True)
    out = batch_io.download_results(
        client, "msgbatch_1", tmp_path / "d.jsonl", http=http, sleep=lambda s: None
    )
    assert out.read_bytes() == body and [("range" in h) for _, h in http.calls] == [False, False]
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    "error",
    [
        "connection",
        "json",
        "decoding",
    ],
)
def test_every_stream_break_falls_back_to_the_download(files, monkeypatch, error):  # noqa: F811
    import anthropic

    raised = {
        "connection": anthropic.APIConnectionError(request=httpx.Request("GET", "https://x")),
        "json": json.JSONDecodeError("cut", '{"custom_id": "a', 16),
        "decoding": httpx.DecodingError("bad gzip"),
    }[error]
    files()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=PARAMS, client=client)
    body = "".join(_api_line(i, GOOD) for i in st["ids"]).encode()
    monkeypatch.setattr(batch_io, "_http", lambda: StubHTTP(body))
    _ended(client).messages.batches = BrokenStream(client.messages.batches, raised)
    st, _ = sl.teacher_batch_collect("reply_v2", client=client)
    assert st["status"] == "collected" and st["collected"]["labelled"] == 10


def test_relabel_and_topic_collect_fall_back_to_the_download(tmp_path, monkeypatch):
    from src.analysis import topic_label as tl

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(batch_io.time, "sleep", lambda s: None)
    client = _ended(_stub_client())
    client.messages.batches.created.append([])
    client.messages.batches = BrokenStream(client.messages.batches)
    st = {
        "batch_id": "msgbatch_1",
        "model": MODEL,
        "status": "ended",
        "submitted_at": "2026-11-01T00:00:00+00:00",
    }
    rl.state_path().write_text(json.dumps(st))
    monkeypatch.setattr(rl, "status", lambda: json.loads(rl.state_path().read_text()))
    monkeypatch.setattr(rl, "_client", lambda: client)
    http = StubHTTP(_api_line("1", GOOD).encode() + _api_line("2", "no json").encode())
    monkeypatch.setattr(batch_io, "_http", lambda: http)
    out = rl.collect()
    assert out["status"] == "collected" and out["collected"]["failed_ids"] == ["2"]
    assert pd.read_parquet(rl.labels_path(MODEL))["id"].tolist() == ["1"]
    assert (tmp_path / "relabel_batch_msgbatch_1_results.jsonl").exists()
    tl.state_path().write_text(json.dumps({**st, "model": settings.LLM_MODEL}))
    monkeypatch.setattr(tl, "status", lambda: json.loads(tl.state_path().read_text()))
    monkeypatch.setattr(tl, "_client", lambda: client)
    http.body = _api_line("7", '{"about_war": true}').encode()
    assert tl.collect()["collected"] == {
        "labelled": 1,
        "failed": 0,
        "no_verdict": 0,
        "unjudgeable": 0,
    }
    assert (tmp_path / "topic_batch_msgbatch_1_results.jsonl").exists()


def test_reply_batches_are_deleted_once_collected_trump_not(files, monkeypatch):  # noqa: F811
    from tests.test_teacher_checks import TRUMP_PARAMS, _feed

    files()
    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "keyword")
    client = _stub_client()
    sl.teacher_batch_submit("trump", params=TRUMP_PARAMS, client=client)
    st, _ = sl.teacher_batch_collect("trump", client=_ended(client))
    assert client.messages.batches.deleted == [] and "delete" not in st
    # --keep-batch: collected, not deleted; a later collect without it deletes it
    client = _stub_client()
    sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)
    st, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client), keep_batch=True)
    assert st["delete"] == {"status": "kept"} and client.messages.batches.deleted == []
    st, _ = sl.teacher_batch_collect("reply_v2", client=client, keep_batch=True)
    assert st["delete"] == {"status": "kept"} and client.messages.batches.deleted == []
    st, _ = sl.teacher_batch_collect("reply_v2", client=client)
    assert st["delete"]["status"] == "deleted" and client.messages.batches.deleted == ["msgbatch_1"]
    # a delete that fails is recorded and tried again by the next collect
    client = _stub_client()
    sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)
    real = client.messages.batches.delete

    def fails(batch_id):
        raise RuntimeError("overloaded")

    client.messages.batches.delete = fails
    st, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    assert st["status"] == "collected" and st["delete"]["status"] == "failed"
    assert "overloaded" in st["delete"]["error"]
    client.messages.batches.delete = real
    st, rep = sl.teacher_batch_collect("reply_v2", client=client)
    assert st["delete"]["status"] == "deleted" and client.messages.batches.deleted == ["msgbatch_1"]
    assert sl.read_batch_state("reply_v2")["delete"]["status"] == "deleted" and rep["n"] == 6
    polls = client.messages.batches.retrieved
    sl.teacher_batch_collect("reply_v2", client=client)  # deleted: no call at all
    assert client.messages.batches.deleted == ["msgbatch_1"]
    assert client.messages.batches.retrieved == polls


def test_a_reply_batch_left_on_the_api_is_deleted_before_the_next_submit(files):  # noqa: F811
    files()
    client = _stub_client()
    b = client.messages.batches
    real = b.delete

    def fails(batch_id):
        raise RuntimeError("overloaded")

    sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)
    b.delete = fails
    st, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    assert st["delete"]["status"] == "failed"
    # The next submit tries again, and refuses (nothing sent, nothing archived) while it fails.
    with pytest.raises(RuntimeError, match="msgbatch_1 could not be deleted"):
        sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)
    assert len(b.created) == 1 and sl.read_batch_state("reply_v2")["batch_id"] == "msgbatch_1"
    assert not sl.batch_archive_path("reply_v2", "msgbatch_1").exists()
    b.delete = real
    st = sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)
    assert st["batch_id"] == "msgbatch_2" and b.deleted == ["msgbatch_1"]
    arch = json.loads(sl.batch_archive_path("reply_v2", "msgbatch_1").read_text())
    assert arch["delete"]["status"] == "deleted"
    # Kept with --keep-batch: the next submit deletes it before archiving its state.
    sl.teacher_batch_collect("reply_v2", client=_ended(client), keep_batch=True)
    sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)
    assert b.deleted == ["msgbatch_1", "msgbatch_2"]

    # A batch the API no longer has (deleted in the Console: 404) counts as deleted.
    class Gone(Exception):
        status_code = 404

    def gone(batch_id):
        raise Gone("not found")

    b.delete = gone
    st, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    assert st["delete"]["status"] == "deleted" and "not found" in st["delete"]["note"]


def test_a_pending_delete_is_done_by_the_next_collect(files):  # noqa: F811
    files()
    client = _stub_client()
    b = client.messages.batches
    real = b.delete
    sl.teacher_batch_submit("reply_v2", params=PARAMS, limit=3, client=client)

    def killed(batch_id):  # the process dies between recording "pending" and the call
        raise KeyboardInterrupt

    b.delete = killed
    with pytest.raises(KeyboardInterrupt):
        sl.teacher_batch_collect("reply_v2", client=_ended(client))
    assert sl.read_batch_state("reply_v2")["delete"] == {"status": "pending"}
    b.delete = real
    st, rep = sl.teacher_batch_collect("reply_v2", client=client)
    assert st["delete"]["status"] == "deleted" and b.deleted == ["msgbatch_1"] and rep["n"] == 3


# ── v2b ────────────────────────────────────────────────────────────


def test_v2b_is_v2_without_the_praise_rule_and_nothing_else():
    v2 = sl.REPLY_TEACHER_V2_PREFIX
    assert rlab.V2_PRAISE_RULE in v2 and rlab.V2_PRAISE_RULE not in rlab.REPLY_V2B_PREFIX
    assert rlab.REPLY_V2B_PREFIX == v2.replace(rlab.V2_PRAISE_RULE, "")
    assert (
        "leans for" in rlab.V2_PRAISE_RULE and "leans against" in rlab.V2_PRAISE_RULE
    )  # both halves
    assert rlab.REPLY_V2B_PROMPT_VERSION != sl.REPLY_TEACHER_V2_PROMPT_VERSION


def test_v2b_round_trip_writes_its_own_file(env):
    reps, drawn = env()
    v2_before = sl.reply_labels_path(MODEL, "v2").read_bytes()
    client = _stub_client(lambda cid, text: '{"score": 0.3, "label": "positive"}')
    st = sl.teacher_batch_submit("reply_v2b", params=PARAMS, client=client)
    reqs = client.messages.batches.created[0]
    assert len(reqs) == len(drawn) == st["n_submitted"]
    for r in reqs:
        blocks = r["params"]["messages"][0]["content"]
        text = _content_text(blocks)
        assert not any(h in text for h in reps["user"]) and "leans against" not in text
        slug = r["custom_id"].rsplit("-", 1)[0]
        assert POSTS[slug][1] in text and slug in text
        # A cache breakpoint only where the v2b block reaches 512 tokens (hold-off's
        # 777-token v2 prefix less the rule; deal's and strikes' fall short).
        if slug == "hold_off_attack":
            assert POSTS[slug][1] in blocks[0]["text"]  # parent in the cached block
            assert blocks[0]["cache_control"] == {"type": "ephemeral"}
            assert blocks[1]["text"].startswith("Reply:")
        else:
            assert isinstance(blocks, str) and "cache_control" not in str(r["params"])
        assert r["params"]["model"] == MODEL and r["params"]["output_config"] == {"effort": "low"}
    assert st["prompt_version"] == rlab.REPLY_V2B_PROMPT_VERSION
    st, rep = sl.teacher_batch_collect("reply_v2b", client=_ended(client))
    lab = pd.read_parquet(sl.reply_labels_path(MODEL, "v2b"))
    assert set(lab["prompt_version"]) == {rlab.REPLY_V2B_PROMPT_VERSION} and len(lab) == len(drawn)
    assert sl.reply_labels_path(MODEL, "v2").read_bytes() == v2_before
    assert rep["v2_vs_v2b"]["mean_shift_v2b_minus_v2"] == pytest.approx(0.8)
    assert {r["post"] for r in rep["by_post"]} == {*POSTS, "ALL"}
    assert client.messages.batches.deleted == ["msgbatch_1"]
    assert rlab.report_path("v2b", MODEL).exists()
    again = sl.teacher_batch_submit("reply_v2b", params=PARAMS, client=client)
    assert again is None  # every reply labelled
    # reply-population weights the v2b set for free
    r = inf.reply_population(n_boot=10, labels_version="v2b").set_index("post")
    assert r.loc["ALL", "opus_mean"] == pytest.approx(0.3) and r.loc["ALL", "labels"] == "v2b"


def test_v2b_cap_is_the_v2_cap(env, monkeypatch):
    env()
    monkeypatch.setattr(settings, "REPLY_TEACHER_V2_MAX_CALLS", 5)
    client = _stub_client()
    with pytest.raises(RuntimeError, match="over the cap of 5"):
        sl.teacher_batch_submit("reply_v2b", params=PARAMS, client=client)
    assert client.messages.batches.created == []
    st = sl.teacher_batch_submit("reply_v2b", params=PARAMS, limit=3, client=client)
    assert len({sl.read_batch_state("reply_v2b")["ids"][i].split("-")[0] for i in range(3)}) == 3
    assert st["n_submitted"] == 3


# ── cat-v2 ─────────────────────────────────────────────────────────


def test_cat_v2_prompt_parse_and_collect(env):
    from src.analysis.event_study import STANCE_CATEGORIES

    reps, _ = env()
    for c in STANCE_CATEGORIES:
        assert c in rlab.REPLY_CAT_V2_PREFIX
    assert "@" not in rlab.REPLY_CAT_V2_PREFIX and "{parent}" in rlab.REPLY_CAT_V2_PREFIX
    answers = {}

    def answer(cid, text):
        i = int(cid.rsplit("-", 1)[1])
        a = [
            '{"stance": "antiwar_betrayal", "reasoning": "x"}',
            '```json\n{"stance": "pro_war_supportive"}\n```',
            '{"stance": "very_angry"}',
            "no json",
        ][i % 4]
        answers[cid] = a
        return a

    client = _stub_client(answer)
    sl.teacher_batch_submit("reply_cat_v2", params=PARAMS, client=client)
    for r in client.messages.batches.created[0]:
        text = _content_text(r["params"]["messages"][0]["content"])
        assert not any(h in text for h in reps["user"]) and "Reply by" not in text
    st, rep = sl.teacher_batch_collect("reply_cat_v2", client=_ended(client))
    lab = pd.read_parquet(sl.reply_labels_path(MODEL, "cat-v2"))
    assert list(lab.columns) == ["id", "tracked_slug", "input_chars", "prompt_version", "category"]
    good = [c for c, a in answers.items() if "antiwar" in a or "pro_war" in a]
    assert sorted(lab["id"]) == sorted(good)
    assert st["collected"]["failed_by_outcome"] == {
        "out_of_range": sum("very_angry" in a for a in answers.values()),
        "unparseable": sum(a == "no json" for a in answers.values()),
    }
    assert rep["n"] == len(good) and "population" in rep and "vs_haiku" in rep
    assert all(k.startswith("neutral_other -> ") for k in rep["vs_haiku"])
    # The cells to read on the de-escalation posts before a full run (v2 is
    # -0.5 everywhere here, so no category lands on "pro").
    assert set(rep["vs_v2_sign_deescalation"]) == {"hold_off_attack", "deal_complete"}
    assert sum(rep["vs_v2_sign_deescalation"]["deal_complete"].values()) == 2 * sum(
        1 for c in good if c.startswith("deal_complete")
    )
    lines = rlab.summary_lines(rep)
    assert any(
        x.startswith("  check hold_off_attack") and "pro_trump_antiwar -> pro 0" in x for x in lines
    )
    assert client.messages.batches.deleted == ["msgbatch_1"]
    assert not sl.reply_labels_path(MODEL, "v2b").exists()
    assert rlab.parse_category_result(
        types.SimpleNamespace(custom_id="x", result=types.SimpleNamespace(type="errored"))
    ) == ("x", "errored", None)


def test_category_population_weights_the_first_draws(env):
    # One post: 90 critical replies, 10 supportive; Opus files the critical
    # draws as antiwar_opposition, the supportive as pro_war_supportive.
    from src.analysis.event_study import STANCE_OUTPUT

    pid, _ = POSTS["deal_complete"]
    reps = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(100)],
            "tracked_slug": "deal_complete",
            "parent_id": pid,
            "user": "u",
            "text": "a reply with words",
            "score_transformer": [-0.9] * 90 + [0.9] * 10,
            "score_opus_distilled": 0.0,
        }
    )
    _, drawn = env(reps, n_per_bucket=5)
    cats = drawn.assign(
        category=np.where(
            drawn["score_transformer"] < 0, "antiwar_opposition", "pro_war_supportive"
        )
    )
    cats[["id", "tracked_slug", "category"]].to_parquet(sl.reply_labels_path(MODEL, "cat-v2"))
    smp = pd.read_parquet(STANCE_OUTPUT)
    smp.loc[smp.index[smp["score_transformer"] > 0][:2], "stance"] = "error"  # Haiku failed
    smp.to_parquet(STANCE_OUTPUT, index=False)
    pop = inf.reply_category_population(MODEL, n_boot=20).set_index(["post", "category"])
    p = pop.loc["deal_complete"]
    assert p.loc["antiwar_opposition", "opus_share"] == pytest.approx(0.9)
    assert p.loc["pro_war_supportive", "opus_share"] == pytest.approx(0.1)
    assert p.loc["neutral_other", "opus_share"] == 0.0
    assert p.loc["neutral_other", "haiku_share"] == pytest.approx(1.0)  # the error rows left out
    assert p["coverage"].eq(1.0).all() and p["n_labelled"].eq(10).all()


# ── fresh draws ────────────────────────────────────────────────────


def test_fresh_draw_is_proportional_new_seeded_and_frozen(env):
    _, drawn = env(_replies(n_per_post=200))
    draws, summary = inf.draw_fresh_sample(n_per_post=40, write=False)
    assert not inf.fresh_draws_path().exists()  # previewed, not recorded
    assert not set(draws["id"]) & set(drawn["id"])  # none of the first draws
    by = draws.groupby(["tracked_slug", "bucket"]).size()
    for slug in POSTS:
        assert by[slug].to_dict() == {0: 20, 1: 10, 2: 10}  # 100 / 50 / 50 replies with text
    again, _ = inf.draw_fresh_sample(n_per_post=40, write=False)
    assert again["id"].tolist() == draws["id"].tolist()
    other, _ = inf.draw_fresh_sample(n_per_post=40, seed=7, write=False)
    assert set(other["id"]) != set(draws["id"])
    inf.draw_fresh_sample(n_per_post=40, write=True)
    assert inf.read_fresh_draws()["id"].tolist() == draws["id"].tolist()
    # More replies arrive: recorded posts are not redrawn.
    more = _replies(n_per_post=300)
    more.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
    later, summary = inf.draw_fresh_sample(n_per_post=40, write=True)
    assert later["id"].tolist() == draws["id"].tolist() and summary["frozen"].all()
    assert set(summary.columns) >= {"post", "bucket", "with_text", "first_draws", "drawn"}


def test_allocate_redistributes_what_a_full_bucket_cannot_take():
    sizes = pd.Series({0: 50, 1: 30, 2: 20})
    assert inf._allocate(10, sizes, sizes) == {0: 5, 1: 3, 2: 2}
    assert inf._allocate(30, sizes, pd.Series({0: 50, 1: 30, 2: 2})) == {0: 18, 1: 10, 2: 2}
    assert sum(inf._allocate(7, sizes, sizes).values()) == 7
    assert inf._allocate(500, sizes, sizes) == {0: 50, 1: 30, 2: 20}  # no more than there is


def test_v2_fresh_needs_the_manifest_skips_labelled_ids_and_goes_post_by_post(env, monkeypatch):
    env(_replies(n_per_post=120))
    client = _stub_client()
    with pytest.raises(FileNotFoundError, match="reply-fresh-sample"):
        sl.teacher_batch_submit("reply_v2_fresh", params=PARAMS, client=client)
    est = rlab.job_estimate("v2-fresh")
    assert est["preview"] and est["todo"] == 3 * (120 - 6)  # all but the 6 first draws a post
    draws, _ = inf.draw_fresh_sample(n_per_post=30, write=True)
    # One drawn reply already carries a v2 label (a flipper, say): never sent again.
    reused = draws["id"].iloc[0]
    v2 = pd.read_parquet(sl.reply_labels_path(MODEL, "v2"))
    extra = v2.iloc[[0]].assign(id=reused, tracked_slug=draws["tracked_slug"].iloc[0])
    pd.concat([v2, extra]).to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)
    est = rlab.job_estimate("v2-fresh")
    assert not est["preview"] and est["todo"] == 3 * 30 - 1 and est["labelled"] == 1
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", 0.001)
    with pytest.raises(RuntimeError, match="REPLY_FRESH_MAX_USD"):
        sl.teacher_batch_submit(
            "reply_v2_fresh", params={**PARAMS, "post": "deal_complete"}, client=client
        )
    assert client.messages.batches.created == []
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", 20.0)
    post = draws.loc[draws["id"] == reused, "tracked_slug"].iloc[0]
    st = sl.teacher_batch_submit("reply_v2_fresh", params={**PARAMS, "post": post}, client=client)
    ids = [r["custom_id"] for r in client.messages.batches.created[0]]
    assert len(ids) == 29 and reused not in ids and all(i.startswith(post) for i in ids)
    assert st["usd_ceiling_batch"] > 0 and st["params"]["post"] == post
    _, rep = sl.teacher_batch_collect("reply_v2_fresh", client=_ended(client))
    fresh = pd.read_parquet(sl.reply_labels_path(MODEL, "v2-fresh"))
    assert sorted(fresh["id"]) == sorted(ids)
    assert "prefix_chars" not in fresh and "cache_chars" not in fresh
    # Plain means per post, no pooled ALL row (posts differ in size).
    by = {r["post"]: r for r in rep["by_post"]}
    assert set(by) == {post} and by[post]["drawn"] == 30 and by[post]["labelled_share"] == 1.0
    assert len(pd.read_parquet(sl.reply_labels_path(MODEL, "v2"))) == len(v2) + 1  # untouched
    _, _, rest, _ = rlab.job_todo(rlab.JOBS["v2-fresh"], MODEL, PARAMS)
    assert len(rest) == 60 and post not in set(rest["tracked_slug"])
    with pytest.raises(ValueError, match="no replies of post"):
        rlab.job_todo(rlab.JOBS["v2-fresh"], MODEL, {**PARAMS, "post": "nope"})


def test_v2_fresh_caps_bound_the_whole_run(env, monkeypatch):
    env(_replies(n_per_post=120))
    inf.draw_fresh_sample(n_per_post=30, write=True)  # 30 a post, 90 in all
    client = _stub_client()
    j = rlab.JOBS["v2-fresh"]
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_CALLS", 50)
    st = sl.teacher_batch_submit(
        "reply_v2_fresh", params={**PARAMS, "post": "deal_complete"}, client=client
    )
    assert st["n_submitted"] == 30  # one post's batch is under the cap
    sl.teacher_batch_collect("reply_v2_fresh", client=_ended(client))
    with pytest.raises(RuntimeError, match="30 calls already sent .* over the run cap of 50"):
        sl.teacher_batch_submit(
            "reply_v2_fresh", params={**PARAMS, "post": "hold_off_attack"}, client=client
        )
    assert len(client.messages.batches.created) == 1
    # Dollars: the spend billed so far plus the next batch's ceiling.
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_CALLS", 8500)
    spent = sl.batch_run_totals("reply_v2_fresh")
    assert spent["calls"] == 30 and spent["batches"] == 1 and spent["usd_batch"] > 0
    post = {**PARAMS, "post": "hold_off_attack"}
    _, _, rows, _ = rlab.job_todo(j, MODEL, post)
    nxt = rlab.calibrated_cost(j, rows, MODEL)["usd_batch_ceiling"]
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", spent["usd_batch"] + 0.99 * nxt)
    with pytest.raises(RuntimeError, match="on top of"):
        sl.teacher_batch_submit("reply_v2_fresh", params=post, client=client)
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", spent["usd_batch"] + 1.01 * nxt)
    st = sl.teacher_batch_submit("reply_v2_fresh", params=post, client=client)
    assert st["batch_id"] == "msgbatch_2" and st["usd_ceiling_batch"] == pytest.approx(nxt)
    # The estimate counts the run too (the open batch at its ceiling).
    est = rlab.job_estimate("v2-fresh", post="strikes_resume_sep")
    assert (est["prior_batches"], est["prior_calls"], est["run_calls"]) == (2, 60, 90)
    assert est["run_usd_ceiling"] == pytest.approx(
        spent["usd_batch"] + nxt + est["usd_batch_ceiling"]
    )


def test_v2_fresh_goes_post_by_post_and_pilots_an_unmeasured_model(env, monkeypatch):
    monkeypatch.delitem(settings.REPLY_OUTPUT_TOKENS, "claude-opus-5-5")  # unmeasured here
    from src.cli import main

    env(_replies(n_per_post=60))
    inf.draw_fresh_sample(n_per_post=20, write=True)
    client = _stub_client()
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    with pytest.raises(RuntimeError, match="one post at a time"):
        sl.teacher_batch_submit("reply_v2_fresh", params=PARAMS, client=client)
    res = CliRunner().invoke(main, ["reply-label", "submit", "--job", "v2-fresh", "--yes"])
    assert res.exit_code == 1 and "one post at a time" in res.output
    assert client.messages.batches.created == []
    st = sl.teacher_batch_submit("reply_v2_fresh", params=PARAMS, limit=6, client=client)
    assert st["n_submitted"] == 6  # a --limit pilot across posts
    sl.teacher_batch_collect("reply_v2_fresh", client=_ended(client))
    # Opus 5.5's output length is unmeasured: a pilot of at most the cap only.
    monkeypatch.setattr(settings, "REPLY_UNMEASURED_PILOT_MAX_CALLS", 5)
    post = {**PARAMS, "post": "deal_complete"}
    with pytest.raises(RuntimeError, match="unmeasured"):
        sl.teacher_batch_submit("reply_v2_fresh", params=post, model="claude-opus-5-5")
    est = rlab.job_estimate("v2-fresh", model="claude-opus-5-5", post="deal_complete")
    assert "unmeasured" in est["problem"]
    st = sl.teacher_batch_submit(
        "reply_v2_fresh", params=post, model="claude-opus-5-5", limit=5, client=client
    )
    assert st["n_submitted"] == 5 and st["labels_file"].endswith("claude-opus-5-5.parquet")
    sl.teacher_batch_collect("reply_v2_fresh", client=_ended(client))
    monkeypatch.setattr(
        settings, "REPLY_OUTPUT_TOKENS", {**settings.REPLY_OUTPUT_TOKENS, "claude-opus-5-5": 90.0}
    )
    assert (
        rlab.job_estimate("v2-fresh", model="claude-opus-5-5", post="deal_complete")["problem"]
        is None
    )  # measured: the cap bounds it again


def test_the_fresh_draw_is_bounded_by_the_run_caps(env, monkeypatch):
    from src.cli import main

    env(_replies(n_per_post=60))
    with pytest.raises(ValueError, match="REPLY_FRESH_PER_POST"):
        inf.draw_fresh_sample(n_per_post=settings.REPLY_FRESH_PER_POST + 1)
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_CALLS", 40)
    with pytest.raises(ValueError, match="REPLY_FRESH_MAX_CALLS"):
        inf.draw_fresh_sample(n_per_post=20, write=True)  # 60 draws
    assert not inf.fresh_draws_path().exists()
    run = CliRunner().invoke
    res = run(main, ["reply-fresh-sample", "--n", str(settings.REPLY_FRESH_PER_POST + 1)])
    assert res.exit_code == 2  # click's range
    res = run(main, ["reply-fresh-sample", "--n", "20"])
    assert res.exit_code == 1 and "nothing recorded" in res.output
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_CALLS", 8500)
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", 0.001)  # the run's no-call ceiling
    res = run(main, ["reply-fresh-sample", "--n", "20"])
    assert res.exit_code == 1 and "REPLY_FRESH_MAX_USD" in res.output
    assert not inf.fresh_draws_path().exists()
    monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", 20.0)
    res = run(main, ["reply-fresh-sample", "--n", "20"])
    assert res.exit_code == 0 and len(inf.read_fresh_draws()) == 60
    with pytest.raises(ValueError, match="REPLY_FRESH_MAX_USD"):
        monkeypatch.setattr(settings, "REPLY_FRESH_MAX_USD", 0.001)
        rlab.refuse_fresh_run(inf.read_fresh_draws())


def test_reply_draws_count_a_fresh_draw_that_is_a_first_draw_once(env):
    reps, drawn = env()
    first = inf.reply_draws(reps)
    fr = pd.DataFrame(
        {
            "id": [drawn["id"].iloc[0], "deal_complete-39"],
            "tracked_slug": [drawn["tracked_slug"].iloc[0], "deal_complete"],
            "bucket": [0, 0],
            "draw_set": "fresh-1",
        }
    )
    fr.to_parquet(inf.fresh_draws_path(), index=False)
    both = inf.reply_draws(reps, fresh=True)
    assert len(both) == len(first) + 1 and not both["id"].duplicated().any()
    assert both.set_index("id").loc[drawn["id"].iloc[0], "draw_set"] == "stance"


def test_reply_population_pools_the_fresh_draws(env, monkeypatch):
    # deal_complete: 80 critical, 20 supportive replies. First draws (5 a
    # bucket) say -1 / +1; the fresh draws say 0 in the critical bucket and
    # +1 in the supportive one.
    pid, _ = POSTS["deal_complete"]
    reps = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(100)],
            "tracked_slug": "deal_complete",
            "parent_id": pid,
            "user": "u",
            "text": "a reply with words",
            "score_transformer": [-0.9] * 80 + [0.9] * 20,
            "score_opus_distilled": 0.0,
        }
    )
    env(reps, n_per_bucket=5)
    v2 = pd.read_parquet(sl.reply_labels_path(MODEL, "v2"))
    v2["score_teacher"] = np.where(
        v2["id"].map(dict(zip(reps["id"], reps["score_transformer"]))) < 0, -1.0, 1.0
    )
    v2.to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)
    before = inf.reply_population(n_boot=10).set_index("post").loc["deal_complete"]
    assert before["opus_mean"] == pytest.approx(0.8 * -1 + 0.2 * 1)
    draws, _ = inf.draw_fresh_sample(n_per_post=20, write=True)  # 16 critical, 4 supportive
    crit = draws["bucket"] == 0
    pd.DataFrame(
        {
            "id": draws["id"],
            "tracked_slug": "deal_complete",
            "input_chars": 18,
            "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
            "score_teacher": np.where(crit, 0.0, 1.0),
            "label_teacher": "neutral",
        }
    ).to_parquet(sl.reply_labels_path(MODEL, "v2-fresh"), index=False)
    r = inf.reply_population(n_boot=10).set_index("post").loc["deal_complete"]
    assert r["n_labelled"] == 30 and r["n_labelled_fresh"] == 20
    crit_mean = (5 * -1 + 16 * 0) / 21
    assert r["opus_mean"] == pytest.approx(0.8 * crit_mean + 0.2 * 1)
    assert r["opus_neutral"] == pytest.approx(0.8 * 16 / 21)
    assert r["opus_pro"] + r["opus_anti"] + r["opus_neutral"] == pytest.approx(1.0)
    assert r["model_neutral"] == 1.0
    assert r["opus_mean_lo"] <= r["opus_mean"] <= r["opus_mean_hi"]
    monkeypatch.setattr(settings, "REPLY_POPULATION_FRESH_DRAWS", False)  # the 1,071-draw figures
    off = inf.reply_population(n_boot=10).set_index("post").loc["deal_complete"]
    assert off["opus_mean"] == pytest.approx(before["opus_mean"]) and off["n_labelled_fresh"] == 0
    monkeypatch.setattr(settings, "REPLY_POPULATION_FRESH_DRAWS", True)
    first = inf.reply_population(n_boot=10, fresh=False).set_index("post").loc["deal_complete"]
    assert first["opus_mean"] == pytest.approx(before["opus_mean"])
    # Before a post's batch is collected only reused (flipper) labels sit on its
    # fresh draws: not a random part, so they stay out until the post is whole.
    lab = pd.read_parquet(sl.reply_labels_path(MODEL, "v2-fresh"))
    lab.iloc[:2].to_parquet(sl.reply_labels_path(MODEL, "v2-fresh"), index=False)
    part = inf.reply_population(n_boot=10).set_index("post").loc["deal_complete"]
    assert part["n_labelled_fresh"] == 0 and part["opus_mean"] == pytest.approx(before["opus_mean"])
    monkeypatch.setattr(settings, "REPLY_FRESH_POOL_MIN_SHARE", 0.1)
    assert (
        inf.reply_population(n_boot=10).set_index("post").loc["deal_complete", "n_labelled_fresh"]
        == 2
    )


def test_v2_and_v2b_compare_on_the_same_draws(env, caplog):
    # v2 labels the first and the fresh draws, v2b the first draws only.
    pid, _ = POSTS["deal_complete"]
    reps = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(100)],
            "tracked_slug": "deal_complete",
            "parent_id": pid,
            "user": "u",
            "text": "a reply with words",
            "score_transformer": [-0.9] * 80 + [0.9] * 20,
            "score_opus_distilled": 0.0,
        }
    )
    _, drawn = env(reps, n_per_bucket=5, v2_score=-0.5)
    draws, _ = inf.draw_fresh_sample(n_per_post=20, write=True)
    pd.DataFrame(
        {
            "id": draws["id"],
            "tracked_slug": "deal_complete",
            "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
            "score_teacher": 0.9,  # far from the first draws': pooling would move v2
            "label_teacher": "positive",
        }
    ).to_parquet(sl.reply_labels_path(MODEL, "v2-fresh"), index=False)
    pd.DataFrame(
        {
            "id": drawn["id"],
            "tracked_slug": drawn["tracked_slug"],
            "prompt_version": rlab.REPLY_V2B_PROMPT_VERSION,
            "score_teacher": 0.0,
            "label_teacher": "neutral",
        }
    ).to_parquet(sl.reply_labels_path(MODEL, "v2b"), index=False)
    pair = inf.reply_population_paired("v2", "v2b", n_boot=50).set_index("post")
    d = pair.loc["deal_complete"]
    assert d["n_labelled"] == 10  # the first draws, not the fresh ones
    assert d["v2_mean"] == pytest.approx(-0.5) and d["v2b_mean"] == pytest.approx(0.0)
    assert d["diff_mean"] == pytest.approx(0.5) and d["diff_anti"] == pytest.approx(-1.0)
    assert d["v2_anti"] == pytest.approx(1.0) and d["v2b_neutral"] == pytest.approx(1.0)
    pooled = inf.reply_population(n_boot=10).set_index("post").loc["deal_complete"]
    assert pooled["opus_mean"] > -0.5  # v2 with the fresh draws: another sample
    with caplog.at_level("WARNING"):
        inf.reply_population(n_boot=10, labels_version="v2b")
    assert "same draws" in caplog.text
    rep = rlab.job_report("v2b", MODEL)
    row = next(r for r in rep["population"] if r["post"] == "deal_complete")
    assert row["diff_mean"] == pytest.approx(0.5)
    assert any("paired" in line for line in rlab.summary_lines(rep))


def test_bootstrap_has_the_finite_population_correction(env):
    # deal_complete's 4 supportive replies are all drawn and labelled (n = N):
    # no sampling error there, whatever the labels' spread.
    pid, _ = POSTS["deal_complete"]
    reps = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(104)],
            "tracked_slug": "deal_complete",
            "parent_id": pid,
            "user": "u",
            "text": "a reply with words",
            "score_transformer": [-0.9] * 100 + [0.9] * 4,
            "score_opus_distilled": 0.0,
        }
    )
    env(reps, n_per_bucket=5)
    v2 = pd.read_parquet(sl.reply_labels_path(MODEL, "v2"))
    v2["score_teacher"] = np.linspace(-1, 1, len(v2))
    v2.to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)
    pop = inf.reply_population(n_boot=200).set_index("post")
    assert pop.loc["deal_complete", "n_labelled"] == 9  # 5 critical, all 4 supportive
    w = inf._weighted
    p = reps.assign(bucket=inf.stance_bucket(reps["score_transformer"]).astype(int))
    whole = p[p["bucket"] == 2]
    smp = whole.assign(y=np.linspace(-1, 1, len(whole)))
    _, r, _ = w(whole, smp, {"y": smp["y"].values}, 200, np.random.default_rng(0))
    assert np.allclose(r["y"], r["y"][0])  # a census: every replicate is the sample
    half = p[p["bucket"] == 0].head(10)
    smp = half.head(5).assign(y=np.linspace(-1, 1, 5))
    _, r, _ = w(half, smp, {"y": smp["y"].values}, 2000, np.random.default_rng(0))
    _, r0, _ = w(
        pd.concat([half] * 100), smp, {"y": smp["y"].values}, 2000, np.random.default_rng(0)
    )
    assert np.std(r["y"][1:]) == pytest.approx(np.sqrt(0.5) * np.std(r0["y"][1:]), rel=0.02)


# ── calibrated estimate ────────────────────────────────────────────


def test_calibrated_cost_matches_a_hand_count(monkeypatch):
    monkeypatch.delitem(settings.REPLY_OUTPUT_TOKENS, "claude-opus-5-5")  # unmeasured here
    u = settings.REPLY_V2_BATCH_USAGE
    j = rlab.JOBS["v2-fresh"]  # the v2 prompt: no extra instructions
    rows = pd.DataFrame(
        {
            "tracked_slug": ["ceasefire", "ceasefire", "deal_complete"],
            "input_chars": [100, 300, 200],
            "cache_chars": [2132, 2132, 1291],
        }
    )
    est = rlab.calibrated_cost(j, rows, "claude-opus-5")
    pre_c, pre_d = u["prefix_tokens"]["ceasefire"], u["prefix_tokens"]["deal_complete"]
    block = pre_c - u["wrapper_tokens"]
    mult = u["cache_hit_share"] * 0.1 + (1 - u["cache_hit_share"]) * 1.25
    billed = 2 * (pre_c - block) + 2 * block * mult + 0.35 * 400 + pre_d + 0.35 * 200
    out = 3 * u["output_tokens"]
    assert est["usd_batch"] == pytest.approx((billed * 5 + out * 25) / 1e6 * 0.5)
    # The ceiling: no cache hits, so every ceasefire request writes its block
    # (1.25x); deal's requests carry no breakpoint and pay plain input.
    ceiling = 2 * (pre_c - block) + 2 * block * 1.25 + pre_d + 0.35 * 600
    assert est["usd_batch_ceiling"] == pytest.approx((ceiling * 5 + out * 25) / 1e6 * 0.5)
    assert est["usd_batch_ceiling"] > ((2 * pre_c + pre_d + 0.35 * 600) * 5 + out * 25) / 2e6
    assert est["usd_direct_ceiling"] == pytest.approx(2 * est["usd_batch_ceiling"])
    assert est["by_post"]["ceasefire"]["usd_batch_ceiling"] + est["by_post"]["deal_complete"][
        "usd_batch_ceiling"
    ] == pytest.approx(est["usd_batch_ceiling"])
    assert est["cached_posts"] == ["ceasefire"]  # deal's prefix is under 512 tokens
    assert est["output_measured"]
    # Opus 5.5 by its own row (same 1.25x write); Haiku 4.5's 4,096-token
    # minimum caches nothing.
    e55 = rlab.calibrated_cost(j, rows, "claude-opus-5-5")
    assert not e55["output_measured"]  # Opus 5's output length, unmeasured on 5.5
    # Unmeasured, the ceiling's output is twice Opus 5's length (thinking runs longer).
    assert e55["output_tokens_ceiling_per_call"] == 2 * u["output_tokens"]
    assert e55["usd_batch_ceiling"] == pytest.approx((ceiling * 4 + 2 * out * 20) / 1e6 * 0.5)
    assert rlab.calibrated_cost(j, rows, "claude-haiku-4-5")["cached_posts"] == []
    # prefix_chars (job_todo's) prices the prefix when cache_chars was zeroed.
    zeroed = rows.assign(prefix_chars=rows["cache_chars"], cache_chars=0)
    assert rlab.calibrated_cost(j, zeroed, MODEL)["usd_batch_ceiling"] == pytest.approx(
        est["usd_batch_ceiling"]
    )
    # v2b drops ~77 tokens: strikes falls under the minimum, ceasefire does not.
    strikes = pd.DataFrame(
        {
            "tracked_slug": ["strikes_resume_sep", "ceasefire"],
            "input_chars": [50, 50],
            "cache_chars": [1669, 2132],
        }
    )
    assert rlab.calibrated_cost(rlab.JOBS["v2-fresh"], strikes, MODEL)["cached_posts"] == [
        "ceasefire",
        "strikes_resume_sep",
    ]
    assert rlab.calibrated_cost(rlab.JOBS["v2b"], strikes, MODEL)["cached_posts"] == ["ceasefire"]
    # A post v2 never labelled is priced from its prefix length.
    new = pd.DataFrame({"tracked_slug": ["new_post"], "input_chars": [0], "cache_chars": [2670]})
    assert rlab.calibrated_cost(j, new, MODEL)["input_tokens"] == round(1000 + u["wrapper_tokens"])
    guess = settings.REPLY_CAT_EST_OUTPUT_TOKENS
    assert rlab.JOBS["cat-v2"].output_tokens() == (guess, 2 * guess, False)
    # A pilot-measured model records its p95, which the ceiling takes.
    monkeypatch.setitem(settings.REPLY_OUTPUT_TOKENS, "claude-opus-5-5", {"mean": 90, "p95": 160})
    assert rlab.JOBS["v2-rater"].output_tokens("claude-opus-5-5") == (90.0, 160.0, True)
    assert rlab.JOBS["v2b"].output_tokens() == (u["output_tokens"], u["output_tokens"], True)


# ── CLI ────────────────────────────────────────────────────────────


def test_cli_job_names_are_the_jobs():
    from src.cli import REPLY_LABEL_JOBS

    assert tuple(rlab.JOBS) == REPLY_LABEL_JOBS
    assert {j.check for j in rlab.JOBS.values()} == set(rlab.REPLY_BATCH_CHECKS)
    assert all(c.carries_replies for c in rlab.REPLY_BATCH_CHECKS.values())


def test_reply_label_estimates_make_no_call_without_a_key(env, monkeypatch):
    monkeypatch.delitem(settings.REPLY_OUTPUT_TOKENS, "claude-opus-5-5")  # unmeasured here
    from src.cli import main

    env(_replies(n_per_post=60))
    _no_api(monkeypatch)
    run = CliRunner().invoke
    for job in ("v2b", "cat-v2", "v2-fresh"):
        res = run(main, ["reply-label", "estimate", "--job", job])
        assert res.exit_code == 0, res.output
        assert "no API call" in res.output and "calibrated on the 2026-10-05 v2 batch" in res.output
    assert "a preview" in res.output  # v2-fresh before the draw
    res = run(main, ["reply-label", "estimate", "--job", "v2b", "--model", "claude-opus-5-5"])
    assert res.exit_code == 0 and "unmeasured" in res.output and "$4 / $20" in res.output
    res = run(main, ["reply-label", "submit", "--job", "v2-fresh", "--yes"])
    assert res.exit_code == 1 and "reply-fresh-sample" in res.output
    res = run(main, ["reply-fresh-sample", "--n", "10", "--dry-run"])
    assert res.exit_code == 0 and "nothing recorded" in res.output
    assert not inf.fresh_draws_path().exists()
    res = run(main, ["reply-fresh-sample", "--n", "10"])
    assert res.exit_code == 0 and inf.fresh_draws_path().exists()
    res = run(main, ["reply-label", "estimate", "--job", "v2-fresh", "--post", "deal_complete"])
    assert (
        res.exit_code == 0
        and "{'deal_complete': 10}" in res.output
        and "a preview" not in res.output
    )
    for bad in (
        ["reply-label", "status", "--job", "v2b", "--post", "x"],
        ["reply-label", "estimate", "--job", "v2b", "--keep-batch"],
        ["reply-label", "submit", "--job", "v2b", "--results-file", __file__],
        ["reply-teacher-check", "--v2", "--keep-batch"],
    ):
        assert run(main, bad).exit_code == 2, bad
    assert not list(settings.PROCESSED_DIR.glob("teacher_batch_*.json"))


def test_reply_label_cli_lifecycle(env, monkeypatch):
    from src.cli import main

    env()
    client = _stub_client(lambda cid, text: '{"stance": "neutral_other", "reasoning": "x"}')
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    run = CliRunner().invoke
    res = run(main, ["reply-label", "submit", "--job", "cat-v2"], input="n\n")
    assert "Aborted." in res.output and client.messages.batches.created == []
    res = run(main, ["reply-label", "submit", "--job", "cat-v2", "--yes"])
    assert res.exit_code == 0 and "msgbatch_1 in_progress" in res.output, res.output
    assert "reply-label collect --job cat-v2" in res.output and "--batch" not in res.output
    res = run(main, ["reply-label", "submit", "--job", "cat-v2", "--yes"])
    assert res.exit_code == 1 and "not collected" in res.output
    _ended(client)
    real = client.messages.batches.delete

    def fails(batch_id):
        raise RuntimeError("overloaded")

    client.messages.batches.delete = fails
    res = run(main, ["reply-label", "collect", "--job", "cat-v2"])
    assert res.exit_code == 0, res.output
    assert "delete failed" in res.output and "reply-label collect --job cat-v2" in res.output
    assert "--batch" not in res.output and "neutral_other 100%" in res.output
    client.messages.batches.delete = real
    res = run(main, ["reply-label", "collect", "--job", "cat-v2"])
    assert res.exit_code == 0 and "deleted on the API" in res.output
    res = run(main, ["reply-label", "report", "--job", "cat-v2"])
    assert res.exit_code == 0 and "cat-v2 labels from claude-opus-5: 18" in res.output


def test_reply_population_cli_prints_the_direct_estimate(env):
    from src.cli import main

    env()
    res = CliRunner().invoke(main, ["reply-population", "--labels", "v2b", "--n-boot", "10"])
    assert res.exit_code == 1 and "no v2b reply labels" in res.output
    assert "reply-label submit --job v2b" in res.output and "Traceback" not in res.output
    res = CliRunner().invoke(main, ["reply-population", "--n-boot", "10", "--first-draws"])
    assert res.exit_code == 0, res.output
    assert (
        settings.PROCESSED_DIR / "reply_population_score_opus_distilled_v2_first_draws.csv"
    ).exists()
    res = CliRunner().invoke(main, ["reply-population", "--n-boot", "10"])
    assert res.exit_code == 0, res.output
    assert "Direct: the labels alone" in res.output and "neutral" in res.output
    csv = pd.read_csv(settings.PROCESSED_DIR / "reply_population_score_opus_distilled_v2.csv")
    assert {"opus_neutral", "opus_neutral_lo", "n_labelled_fresh", "assisted_neutral"} <= set(csv)


# ── review fixes: per-post pilots, the rater's run caps, max_tokens stops ──

O55 = "claude-opus-5-5"


def test_v2_fresh_without_post_is_only_a_pilot_sized_limit(env, monkeypatch):
    env(_replies(n_per_post=60))
    inf.draw_fresh_sample(n_per_post=20, write=True)
    client = _stub_client()
    monkeypatch.setattr(settings, "REPLY_UNMEASURED_PILOT_MAX_CALLS", 5)
    with pytest.raises(RuntimeError, match="one post at a time"):
        sl.teacher_batch_submit("reply_v2_fresh", params=PARAMS, limit=6, client=client)
    assert client.messages.batches.created == []
    assert "one post at a time" in rlab.job_estimate("v2-fresh", limit=40)["problem"]
    st = sl.teacher_batch_submit("reply_v2_fresh", params=PARAMS, limit=5, client=client)
    assert st["n_submitted"] == 5


def test_v2_rater_caps_bound_the_run_and_unmeasured_batches_add_up(env, monkeypatch):
    monkeypatch.delitem(settings.REPLY_OUTPUT_TOKENS, "claude-opus-5-5")  # unmeasured here
    env()
    client = _stub_client()
    j = rlab.JOBS["v2-rater"]
    assert j.run_caps and rlab.REPLY_BATCH_CHECKS["reply_v2_rater"].run_caps
    monkeypatch.setattr(settings, "REPLY_UNMEASURED_PILOT_MAX_CALLS", 5)
    st = sl.teacher_batch_submit("reply_v2_rater", params=PARAMS, model=O55, limit=5, client=client)
    assert st["n_submitted"] == 5
    sl.teacher_batch_collect("reply_v2_rater", client=_ended(client))
    # A second unmeasured pilot would make 6: refused, however small.
    with pytest.raises(RuntimeError, match="5 sent already"):
        sl.teacher_batch_submit("reply_v2_rater", params=PARAMS, model=O55, limit=1, client=client)
    # Measured from the pilot's collect (mean and p95): the dollar cap bounds the run.
    monkeypatch.setitem(settings.REPLY_OUTPUT_TOKENS, O55, {"mean": 50.0, "p95": 80.0})
    spent = sl.batch_run_totals("reply_v2_rater")
    _, _, rows, _ = rlab.job_todo(j, O55, PARAMS)
    nxt = rlab.calibrated_cost(j, rows, O55)["usd_batch_ceiling"]
    monkeypatch.setattr(settings, "SECOND_RATER_REPLY_MAX_USD", spent["usd_batch"] + 0.99 * nxt)
    assert nxt < settings.SECOND_RATER_REPLY_MAX_USD  # per batch it would have gone
    with pytest.raises(RuntimeError, match="on top of"):
        sl.teacher_batch_submit("reply_v2_rater", params=PARAMS, model=O55, client=client)
    est = rlab.job_estimate("v2-rater")
    assert est["run_caps"] and est["prior_calls"] == 5
    assert est["output_tokens_ceiling_per_call"] == 80.0


def test_a_reply_batch_cut_at_max_tokens_blocks_the_next(env, monkeypatch):
    env()
    client = _stub_client()
    sl.teacher_batch_submit("reply_v2b", params=PARAMS, limit=4, client=client)
    batches = client.messages.batches
    orig = batches.results

    def results(bid):
        for k, r in enumerate(orig(bid)):
            r.result.message.stop_reason = "max_tokens" if k == 0 else "end_turn"
            yield r

    batches.results = results
    st, _ = sl.teacher_batch_collect("reply_v2b", client=_ended(client))
    assert st["spend"]["stop_reasons"] == {"max_tokens": 1, "end_turn": 3}
    assert st["spend"]["output_tokens_p95"] == 50
    v2b = rlab.JOBS["v2b"]
    for limit in (None, 2):
        assert "REPLY_LABEL_MAX_TOKENS" in (rlab.submit_problem(v2b, MODEL, None, limit, 9) or "")
    with pytest.raises(RuntimeError, match="stopped at max_tokens"):
        sl.teacher_batch_submit("reply_v2b", params=PARAMS, client=client)
    # Another model, or a higher max_tokens, is another request.
    assert rlab.submit_problem(rlab.JOBS["cat-v2"], MODEL, None, None, 9) is None
    monkeypatch.setattr(settings, "REPLY_LABEL_MAX_TOKENS", 2048)
    assert rlab.submit_problem(v2b, MODEL, None, None, 9) is None
