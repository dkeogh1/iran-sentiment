"""Reply relabel v2 and the Trump-feed teacher check, with stub scorers (no API)."""

import json
import logging
import math
import time
import types

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings
from src.analysis import stance_local as sl
from src.analysis.sentiment import llm_prompt

# Stub scorers answer a constant, which has no Pearson correlation (NaN).
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

# Tracked posts with a pinned post_id in config/tracked_posts.py.
POSTS = {
    "hold_off_attack": (
        "116597121700043134",
        "I have been asked to hold off on our planned attack.",
    ),
    "deal_complete": (
        "116750587569914985",
        "The Deal with Iran is now complete. Start your engines!",
    ),
    "strikes_resume_sep": (
        "117196950497702512",
        "We are striking Iranian targets near Hormuz tonight.",
    ),
}
LONG_TAIL = "TAIL-MARKER-PAST-100"
LONG = "Thank you Mr President, " + "peace is the right call " * 6 + LONG_TAIL


def _reply_rows(n_per_post=4):
    rows = []
    for slug, (pid, _) in POSTS.items():
        for i in range(n_per_post):
            rows.append(
                {
                    "id": f"{slug}-{i}",
                    "tracked_slug": slug,
                    "parent_id": pid,
                    "user": f"handle_{slug}_{i}",
                    "text": f"reply {i} to {slug}",
                    "score_transformer": 0.5,
                    "score_opus_distilled": 0.2,
                }
            )
    rows[0]["text"] = LONG  # longer than the sample's 100 chars
    rows[1]["text"] = ""  # image-only reply
    rows[2]["text"] = " ok "  # under MIN_TEXT_CHARS
    return pd.DataFrame(rows)


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Sampled replies, their full texts, the Trump feed and the v1 labels in tmp_path."""
    from src.analysis import event_study as es

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "REPLY_SENTIMENT_OUTPUT", tmp_path / "replies.parquet")
    monkeypatch.setattr(settings, "TRUTH_SOCIAL_RAW_DIR", tmp_path / "ts")
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", tmp_path / "trump.parquet")
    monkeypatch.setattr(es, "STANCE_OUTPUT", tmp_path / "stance_sample.parquet")
    (tmp_path / "ts").mkdir()

    def write(replies: pd.DataFrame | None = None, posts: dict | None = None):
        reps = _reply_rows() if replies is None else replies
        reps.to_parquet(settings.REPLY_SENTIMENT_OUTPUT, index=False)
        smp = reps.drop_duplicates("id").assign(text=lambda d: d["text"].str[:100], stance="x")
        smp[["id", "tracked_slug", "user", "text", "stance"]].to_parquet(
            es.STANCE_OUTPUT, index=False
        )
        feed = POSTS if posts is None else posts
        with open(settings.TRUTH_SOCIAL_RAW_DIR / "realDonaldTrump.jsonl", "w") as f:
            f.writelines(
                json.dumps(
                    {
                        "id": pid,
                        "user": "realDonaldTrump",
                        "text": text,
                        "created_at": "2026-06-01T00:00:00Z",
                    }
                )
                + "\n"
                for pid, text in feed.values()
            )
        v1 = smp[["id"]].assign(score_teacher=0.6, label_teacher="positive")
        v1.to_parquet(sl.reply_labels_path(version="v1"), index=False)
        return reps

    return write


class Stub:
    """A scorer that records what it was sent and answers with a fixed score."""

    def __init__(self, score=-0.5, fail_on=()):
        self.calls, self.score, self.fail_on = [], score, set(fail_on)

    def __call__(self, prompt, cache_chars=0):
        self.calls.append((prompt, cache_chars))
        if any(f in prompt for f in self.fail_on):
            raise RuntimeError("overloaded")
        return self.score, "negative"


def _v2(**kw):
    return sl.reply_teacher_check_v2(concurrency=1, **kw)


def test_v2_prompt_has_parent_and_full_reply_not_handle(files):
    reps = files()
    v1_before = sl.reply_labels_path(version="v1").read_bytes()
    stub = Stub()
    rep = _v2(scorer=stub)
    sent = {p for p, _ in stub.calls}
    assert len(sent) == len(stub.calls) == 12 - 2  # the two textless replies are skipped
    long_prompt = next(p for p in sent if LONG_TAIL in p)  # the whole reply, not text[:100]
    assert POSTS["hold_off_attack"][1] in long_prompt
    for prompt, cache_chars in stub.calls:
        assert not any(h in prompt for h in reps["user"])  # no reply author handle
        assert "Reply:" not in prompt[:cache_chars] and prompt[cache_chars:].startswith("Reply:")
        slug = next(s for s in POSTS if POSTS[s][1] in prompt)
        assert POSTS[slug][1] in prompt[:cache_chars]  # the parent post is in the cached prefix
    labels = pd.read_parquet(sl.reply_labels_path(version="v2"))
    assert len(labels) == 10 and labels["id"].is_unique
    assert set(labels["prompt_version"]) == {sl.REPLY_TEACHER_V2_PROMPT_VERSION}
    assert labels.set_index("id").loc["hold_off_attack-0", "input_chars"] == len(LONG)
    assert "prompt" not in labels and "cache_chars" not in labels
    assert sl.reply_labels_path(version="v1").read_bytes() == v1_before  # v1 untouched
    assert rep["n"] == 10 and rep["v1_vs_v2"]["n"] == 10
    by_post = {r["post"]: r for r in rep["v1_vs_v2_by_post"]}
    assert by_post["ALL"]["v1_pro"] == 1.0 and by_post["ALL"]["v2_anti"] == 1.0
    assert by_post["ALL"]["v2_mean"] == pytest.approx(-0.5)
    by_input = {r["v1_input"]: r["n"] for r in rep["v1_vs_v2_by_input"]}
    assert by_input == {"whole reply (<=100 chars)": 9, "first 100 chars only": 1}
    assert (settings.PROCESSED_DIR / "reply_teacher_check_v2_claude-opus-5.json").exists()


def test_v2_cache_skip_and_resume(files):
    files()
    first = Stub(fail_on=["reply 3 to deal_complete"])
    _v2(scorer=first)
    assert len(first.calls) == 10
    labels = pd.read_parquet(sl.reply_labels_path(version="v2"))
    assert len(labels) == 9 and "deal_complete-3" not in set(labels["id"])  # failed: not saved
    second = Stub()
    _v2(scorer=second)
    assert len(second.calls) == 1 and "reply 3 to deal_complete" in second.calls[0][0]
    third = Stub()
    _v2(scorer=third)
    assert third.calls == []
    assert len(pd.read_parquet(sl.reply_labels_path(version="v2"))) == 10


def test_v2_killed_run_keeps_labels_and_cancels_the_rest(files, monkeypatch):
    files()
    monkeypatch.setattr(settings, "LLM_SAVE_EVERY_N", 2)
    calls = []

    def dies_on_third(prompt, cache_chars=0):
        calls.append(prompt)
        if len(calls) == 3:
            raise KeyboardInterrupt
        if len(calls) > 3:
            time.sleep(0.2)  # a real call takes seconds: the queue is still full
        return 0.3, "positive"

    with pytest.raises(KeyboardInterrupt):
        _v2(scorer=dies_on_third)
    assert len(calls) in (3, 4)  # the queue is cancelled; one call may already be in flight
    assert len(pd.read_parquet(sl.reply_labels_path(version="v2"))) == 2
    rest = Stub()
    _v2(scorer=rest)
    assert len(rest.calls) == 8


def test_v2_rejects_bad_scores(files):
    files()
    answers = iter([1.5, float("nan"), True, "0.5", None, 0.4, -1.0, 1.0, 0.0, -0.2])

    def scorer(prompt, cache_chars=0):
        return next(answers), None

    _v2(scorer=scorer)
    labels = pd.read_parquet(sl.reply_labels_path(version="v2"))
    assert sorted(labels["score_teacher"]) == [-1.0, -0.2, 0.0, 0.4, 1.0]
    assert set(labels["label_teacher"]) == {"negative", "neutral", "positive"}  # from the score
    assert not sl.valid_score(float("inf")) and sl.valid_score(-1) and not sl.valid_score(False)


def test_v2_estimate_makes_no_calls(files, monkeypatch):
    import anthropic

    from src.analysis import sentiment
    from src.cli import main

    files()

    def boom(*a, **k):
        raise AssertionError("an API call")

    monkeypatch.setattr(sentiment, "score_prompt", boom)
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(sl, "_teacher_scorer", boom)
    est = sl.reply_teacher_v2_estimate()
    assert est["sampled"] == 12 and est["no_text"] == 2 and est["todo"] == est["calls"] == 10
    assert est["usd_no_cache"] > 0 and est["usd"] <= est["usd_no_cache"] + 1e-9
    assert est["todo_by_post"] == {
        "deal_complete": 4,
        "hold_off_attack": 2,
        "strikes_resume_sep": 4,
    }
    res = CliRunner().invoke(main, ["reply-teacher-check", "--v2", "--estimate"])
    assert res.exit_code == 0, res.output
    assert "10 calls to claude-opus-5" in res.output and "no API call" in res.output
    assert not sl.reply_labels_path(version="v2").exists()
    res = CliRunner().invoke(main, ["reply-teacher-check", "--estimate"])
    assert res.exit_code != 0 and "go with --v2" in res.output


def test_v2_limit_stratifies_across_posts(files):
    reps = _reply_rows(n_per_post=20)
    files(reps)
    _, _, todo = sl.reply_v2_todo(limit=3)
    assert sorted(todo["tracked_slug"]) == sorted(POSTS)  # one from each post
    _, _, todo5 = sl.reply_v2_todo(limit=5)
    assert set(todo5["tracked_slug"]) == set(POSTS)
    assert todo5["id"].tolist()[:3] == todo["id"].tolist()  # seeded: same order
    stub = Stub()
    _v2(scorer=stub, limit=3)
    assert len(stub.calls) == 3
    _, cached, rest = sl.reply_v2_todo()
    assert len(cached) == 3 and not set(rest["id"]) & set(cached["id"])
    assert len(rest) == 60 - 2 - 3


def test_v2_cap_is_enforced_before_any_call(files, monkeypatch):
    files()
    monkeypatch.setattr(settings, "REPLY_TEACHER_V2_MAX_CALLS", 4)
    stub = Stub()
    with pytest.raises(RuntimeError, match="over the cap of 4"):
        _v2(scorer=stub)
    assert stub.calls == [] and not sl.reply_labels_path(version="v2").exists()
    _v2(scorer=stub, limit=4)  # a pilot under the cap runs
    assert len(stub.calls) == 4


def test_v2_fails_loudly_on_missing_post_text(files):
    files(posts={k: v for k, v in POSTS.items() if k != "deal_complete"})
    with pytest.raises(LookupError, match="deal_complete"):
        sl.reply_v2_inputs()
    files(posts={**POSTS, "deal_complete": (POSTS["deal_complete"][0], "")})
    with pytest.raises(LookupError, match="deal_complete"):
        sl.reply_v2_inputs()


def test_v2_reply_ids_one_row_each(files):
    reps = _reply_rows()
    files(pd.concat([reps, reps.iloc[[5]]], ignore_index=True))  # identical twice: fine
    assert sl.reply_v2_inputs()["id"].is_unique
    clash = reps.iloc[[5]].assign(text="a different text")
    files(pd.concat([reps, clash], ignore_index=True))
    with pytest.raises(ValueError, match="1 sampled reply ids"):
        sl.reply_v2_inputs()
    wrong = reps.copy()
    wrong.loc[wrong["tracked_slug"] == "deal_complete", "parent_id"] = "999"
    files(wrong)
    with pytest.raises(ValueError, match="deal_complete"):
        sl.reply_v2_inputs()


def test_v2_refuses_a_cache_from_another_prompt(files):
    files()
    pd.DataFrame(
        {
            "id": ["x"],
            "score_teacher": [0.1],
            "label_teacher": ["positive"],
            "prompt_version": ["reply-v2-old"],
        }
    ).to_parquet(sl.reply_labels_path(version="v2"))
    with pytest.raises(ValueError, match="reply-v2-old"):
        _v2(scorer=Stub())


def test_v2_cli_pilot_and_confirmation(files, monkeypatch):
    from src.cli import main

    files()
    stub = Stub()
    monkeypatch.setattr(sl, "_teacher_scorer", lambda *a: stub)
    res = CliRunner().invoke(main, ["reply-teacher-check", "--v2", "--limit", "3"], input="n\n")
    assert "Aborted." in res.output and stub.calls == []
    res = CliRunner().invoke(main, ["reply-teacher-check", "--v2", "--limit", "3", "--yes"])
    assert res.exit_code == 0, res.output
    assert len(stub.calls) == 3 and "v1 vs v2 labels on 3 replies" in res.output


def test_prompt_content_and_score_prompt_send_blocks(monkeypatch):
    import anthropic

    from src.analysis import sentiment

    assert sl.prompt_content("abc") == "abc"
    blocks = sl.prompt_content("prefix|reply", 7)
    assert "".join(b["text"] for b in blocks) == "prefix|reply"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in blocks[1]
    seen = {}

    class _Client:
        def __init__(self, *a, **k):
            self.messages = types.SimpleNamespace(
                create=lambda **kw: (
                    seen.update(kw)
                    or types.SimpleNamespace(
                        content=[
                            types.SimpleNamespace(
                                type="text", text='{"score": 0.25, "label": "positive"}'
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert sentiment.score_prompt(blocks, "claude-opus-5", 1024, "low") == (0.25, "positive")
    assert seen["messages"][0]["content"] == blocks and seen["output_config"] == {"effort": "low"}


def test_estimate_prices_cached_prefixes(monkeypatch):
    from src.analysis.relabel import EST_OUT_TOKENS, PRICE_IN, PRICE_OUT

    monkeypatch.setattr(settings, "TEACHER_EST_CHARS_PER_TOKEN", 1.0)
    prefix = "p" * 600  # 600 "tokens": over CACHE_MIN_TOKENS
    prompts = [prefix + "r" * 10] * 6
    e = sl.estimate_direct_cost(prompts, [600] * 6, concurrency=2)
    assert e["input_tokens"] == 6 * 610 and e["output_tokens"] == 6 * EST_OUT_TOKENS
    billed = 2 * 600 * sl.CACHE_WRITE_MULT + 4 * 600 * sl.CACHE_READ_MULT + 6 * 10
    assert e["usd"] == pytest.approx((billed * PRICE_IN + e["output_tokens"] * PRICE_OUT) / 1e6)
    assert e["usd_no_cache"] == pytest.approx(
        (6 * 610 * PRICE_IN + e["output_tokens"] * PRICE_OUT) / 1e6
    )
    short = sl.estimate_direct_cost(["s" * 100 + "r"] * 3, [100] * 3)  # under the minimum: no cache
    assert (
        short["usd"] == pytest.approx(short["usd_no_cache"]) and short["cached_prefix_calls"] == 0
    )


def test_stratified_order_deals_round_robin():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 2 + ["c"] * 3, "v": range(10)})
    out = sl.stratified_order(df, "g", seed=1)
    assert sorted(out["v"]) == list(range(10))
    assert set(out["g"].iloc[:3]) == set(out["g"].iloc[3:6]) == {"a", "b", "c"}
    assert set(out["g"].iloc[6:8]) == {"a", "c"} and out["g"].iloc[8:].tolist() == ["a", "a"]


# ── Trump-feed check ───────────────────────────────────────────────


def _feed(n=60):
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-04-05", "2026-09-28", periods=n, tz="UTC")
    texts = [
        ("Iran must open the Strait" if i % 3 == 0 else f"Great rally number {i}") for i in range(n)
    ]
    texts[1], texts[2] = "", "https://t.co/abc"  # no text to judge
    return pd.DataFrame(
        {
            "id": [str(1000 + i) for i in range(n)],
            "user": "realDonaldTrump",
            "tier": "admin",
            "platform": "truthsocial",
            "created_at": [d.strftime("%Y-%m-%dT%H:%M:%S.000Z") for d in days],
            "text": texts,
            "score_opus_distilled": rng.uniform(-1, 1, n),
        }
    )


def test_trump_check_sends_the_broadcaster_prompt(files, monkeypatch):
    feed = _feed()
    feed.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "keyword")
    stub = Stub(score=0.4)
    rep = sl.trump_teacher_check(n=20, concurrency=1, scorer=stub)
    sample = sl.trump_sample(20)
    by_id = feed.set_index("id")
    assert len(stub.calls) == 20 and not {"1001", "1002"} & set(sample["id"])
    assert sorted(p for p, _ in stub.calls) == sorted(
        llm_prompt(by_id.loc[i, "text"], "realDonaldTrump") for i in sample["id"]
    )
    assert all(c == 0 for _, c in stub.calls)  # byte-identical to relabel's
    labels = pd.read_parquet(sl.trump_labels_path())
    assert set(labels["id"]) == set(sample["id"]) and set(labels["prompt_version"]) == {
        "llm_prompt"
    }
    assert rep["n"] == 20 and rep["all"]["n"] == 20
    assert rep["all"]["mean_diff"] == pytest.approx(sample["score_opus_distilled"].mean() - 0.4)
    assert sum(r["n"] for r in rep["by_phase"]) == 20
    war = sample["text"].str.contains("Iran")
    assert rep["war_posts"]["n"] == int(war.sum()) and sum(
        r["n"] for r in rep["war_by_phase"]
    ) == int(war.sum())
    assert (settings.PROCESSED_DIR / "teacher_check_trump_claude-opus-5.json").exists()
    again = Stub()
    sl.trump_teacher_check(n=20, concurrency=1, scorer=again)
    assert again.calls == []  # cached


def test_trump_sample_is_stable_as_the_feed_grows(files):
    feed = _feed()
    feed.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    before = set(sl.trump_sample(20)["id"])
    grown = pd.concat([feed, _feed(5).assign(id=lambda d: "new" + d["id"])], ignore_index=True)
    grown.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    after = set(sl.trump_sample(20)["id"])
    assert len(before - after) == len({i for i in after if i.startswith("new")})  # only displaced
    assert len(before & after) >= 15


def test_trump_estimate_and_cap(files, monkeypatch):
    from src.cli import main

    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)

    def no_api(*a):
        raise AssertionError("an API call")

    monkeypatch.setattr(sl, "_teacher_scorer", no_api)
    res = CliRunner().invoke(
        main, ["teacher-check", "--source", "trump", "--n", "30", "--estimate"]
    )
    assert res.exit_code == 0, res.output
    assert "30 sampled posts with text" in res.output and "30 calls to claude-opus-5" in res.output
    assert not sl.trump_labels_path().exists()
    est = sl.trump_check_estimate(n=30)
    prompts = [llm_prompt(t, "realDonaldTrump") for t in sl.trump_sample(30)["text"]]
    assert est["input_tokens"] == sum(
        math.ceil(len(p) / settings.TEACHER_EST_CHARS_PER_TOKEN) for p in prompts
    )
    monkeypatch.setattr(settings, "TRUMP_TEACHER_CHECK_MAX_CALLS", 10)
    stub = Stub()
    with pytest.raises(RuntimeError, match="over the cap of 10"):
        sl.trump_teacher_check(n=30, scorer=stub)
    assert stub.calls == []


def test_x_teacher_check_runs_unattended(tmp_path, monkeypatch):
    """The k8s teacher-check Job runs `teacher-check` with no stdin: it must
    not stop at a prompt, and --yes (a no-op there) is accepted."""
    import dotenv

    from src.analysis import sentiment
    from src.cli import main

    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(6)],
            "text": [f"post {i} about the Iran war" for i in range(6)],
            "user": "someone",
            "tier": ["admin"] * 3 + ["media"] * 3,
            "score_llm": 0.1,
            "label_llm": "positive",
        }
    ).to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    calls = []

    def stub(text, *a):
        calls.append(text)
        return 0.3, "positive"

    monkeypatch.setattr(sentiment, "score_llm", stub)
    res = CliRunner().invoke(main, ["teacher-check", "--n", "4"])
    assert res.exit_code == 0, res.output
    assert "Run?" not in res.output and len(calls) == 4
    assert "Haiku vs claude-opus-5 on 4 posts" in res.output
    res = CliRunner().invoke(main, ["teacher-check", "--n", "4", "--yes"])
    assert res.exit_code == 0, res.output
    assert len(calls) == 4  # all cached
    res = CliRunner().invoke(main, ["teacher-check", "--n", "6", "--estimate"])
    assert res.exit_code == 0, res.output
    assert "6 sampled, 4 cached; 2 calls to claude-opus-5" in res.output and len(calls) == 4


def test_trump_cli_refuses_over_cap_before_the_prompt(files, monkeypatch):
    from src.cli import main

    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    monkeypatch.setattr(settings, "TRUMP_TEACHER_CHECK_MAX_CALLS", 10)

    def no_api(*a):
        raise AssertionError("an API call")

    monkeypatch.setattr(sl, "_teacher_scorer", no_api)
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--n", "30"])
    assert res.exit_code == 1 and "30 calls is over the cap (10)" in res.output
    assert "Run?" not in res.output and isinstance(res.exception, SystemExit)  # no traceback
    assert not sl.trump_labels_path().exists()


def test_api_scorer_tallies_spend(tmp_path, monkeypatch, caplog):
    """run_labels through the real scorer path (fake client) logs the tokens
    billed, unparseable answers included."""
    import anthropic
    import dotenv

    from src.analysis.relabel import PRICE_IN, PRICE_OUT

    usage = types.SimpleNamespace(
        input_tokens=100,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=600,
        output_tokens=50,
    )

    class _Client:
        def __init__(self, *a, **k):
            def create(**kw):
                ok = "good" in kw["messages"][0]["content"]
                text = '{"score": -0.4, "label": "negative"}' if ok else "no json here"
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text=text)], usage=usage
                )

            self.messages = types.SimpleNamespace(create=create)

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    todo = pd.DataFrame({"id": ["a", "b"], "prompt": ["good prompt", "bad prompt"]})
    empty = pd.DataFrame(columns=["id", "score_teacher", "label_teacher"])
    with caplog.at_level(logging.INFO, logger=sl.logger.name):
        frame = sl.run_labels(todo, empty, tmp_path / "labels.parquet", cap=5, concurrency=1)
    assert list(frame["id"]) == ["a"]  # the unparseable answer is not saved...
    assert (
        "2 calls billed: 200 input, 0 cache-write, 1,200 cache-read, 100 output tokens"
        in caplog.text
    )  # ...but it is counted
    spend = sl.Spend()
    spend.add(usage)
    spend.add(types.SimpleNamespace(input_tokens=10, cache_creation_input_tokens=700))
    billed_in = 110 + 700 * sl.CACHE_WRITE_MULT + 600 * sl.CACHE_READ_MULT
    assert spend.usd() == pytest.approx((billed_in * PRICE_IN + 50 * PRICE_OUT) / 1e6)


# ── Batch transport (stub batch client, no API) ────────────────────

V2_PARAMS = {"seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
TRUMP_PARAMS = {"n": 20, "seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
GOOD = '{"score": -0.5, "label": "negative", "reasoning": "x"}'


def _content_text(content):
    return "".join(b["text"] for b in content) if isinstance(content, list) else content


def _result(custom_id, out):
    """A batch result shaped like the SDK's: `out` is the answer text, or an
    outcome (errored / expired / canceled)."""
    ns = types.SimpleNamespace
    if out in ("errored", "expired", "canceled"):
        return ns(custom_id=custom_id, result=ns(type=out))
    usage = ns(
        input_tokens=100,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        output_tokens=50,
    )
    content = [ns(type="thinking", thinking=""), ns(type="text", text=out)]
    return ns(
        custom_id=custom_id, result=ns(type="succeeded", message=ns(content=content, usage=usage))
    )


class StubBatches:
    """client.messages.batches without the API: records each batch's requests
    and answers each request with answer(custom_id, prompt text)."""

    def __init__(self, answer=None):
        self.created, self.retrieved, self.extra = [], 0, []
        self.status = "in_progress"
        self.answer = answer or (lambda cid, text: GOOD)

    def create(self, requests):
        self.created.append(list(requests))
        return types.SimpleNamespace(
            id=f"msgbatch_{len(self.created)}", processing_status="in_progress"
        )

    def retrieve(self, batch_id):
        self.retrieved += 1
        n, done = len(self.created[-1]), self.status == "ended"
        counts = types.SimpleNamespace(
            processing=0 if done else n,
            succeeded=n if done else 0,
            errored=0,
            canceled=0,
            expired=0,
        )
        return types.SimpleNamespace(
            id=batch_id, processing_status=self.status, request_counts=counts
        )

    def results(self, batch_id):
        assert batch_id == f"msgbatch_{len(self.created)}"
        for req in self.created[-1]:
            text = _content_text(req["params"]["messages"][0]["content"])
            yield _result(req["custom_id"], self.answer(req["custom_id"], text))
        yield from self.extra


def _stub_client(answer=None):
    return types.SimpleNamespace(messages=types.SimpleNamespace(batches=StubBatches(answer)))


def _ended(client):
    client.messages.batches.status = "ended"
    return client


def test_batch_requests_are_the_direct_calls(files, monkeypatch):
    """Only the transport changes: each request's params are the kwargs
    score_prompt sends for the same row (content blocks, model, max_tokens,
    effort), and custom_id is the row's id."""
    import anthropic
    import dotenv

    files()
    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    sent = []
    answer = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=GOOD)], usage=None
    )

    class _Client:
        def __init__(self, *a, **k):
            self.messages = types.SimpleNamespace(create=lambda **kw: sent.append(kw) or answer)

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    for check, params in (("reply_v2", V2_PARAMS), ("trump", TRUMP_PARAMS)):
        _, todo = sl.BATCH_CHECKS[check].todo("claude-opus-5", params, None)
        reqs = sl.build_batch_requests(todo, "claude-opus-5", 1024, "low")
        sent.clear()
        direct = sl._teacher_scorer("claude-opus-5", 1024, "low")
        cache_chars = todo["cache_chars"] if "cache_chars" in todo else [0] * len(todo)
        for prompt, c in zip(todo["prompt"], cache_chars):
            direct(prompt, int(c))
        assert [r["custom_id"] for r in reqs] == todo["id"].tolist()
        assert [dict(r["params"]) for r in reqs] == sent
        assert all(r["params"]["output_config"] == {"effort": "low"} for r in reqs)
    _, todo = sl.BATCH_CHECKS["reply_v2"].todo("claude-opus-5", V2_PARAMS, None)
    for r, prompt, c in zip(
        sl.build_batch_requests(todo, "claude-opus-5", 1024, "low"),
        todo["prompt"],
        todo["cache_chars"],
    ):
        blocks = r["params"]["messages"][0]["content"]
        assert blocks[0]["text"] == prompt[:c] and blocks[0]["cache_control"] == {
            "type": "ephemeral"
        }
        assert blocks[1]["text"].startswith("Reply:") and _content_text(blocks) == prompt
    bad = todo.head(2).assign(id=["ok_1", "not ok!"])
    with pytest.raises(ValueError, match="not valid batch custom_ids"):
        sl.build_batch_requests(bad, "claude-opus-5", 1024, "low")
    with pytest.raises(ValueError, match="appear twice"):
        sl.build_batch_requests(todo.head(2).assign(id="same"), "claude-opus-5", 1024, "low")


def test_batch_submit_skips_cached_ids_and_records_state(files):
    files()
    _v2(scorer=Stub(), limit=3)  # a pilot's labels are in the cache
    piloted = set(pd.read_parquet(sl.reply_labels_path(version="v2"))["id"])
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    ids = [r["custom_id"] for r in client.messages.batches.created[0]]
    assert len(ids) == 10 - 3 and not piloted & set(ids)
    saved = json.loads(sl.batch_state_path("reply_v2").read_text())
    assert saved == st and saved["ids"] == ids and saved["n_submitted"] == 7
    assert saved["batch_id"] == "msgbatch_1" and saved["status"] == "in_progress"
    assert saved["prompt_version"] == sl.REPLY_TEACHER_V2_PROMPT_VERSION
    assert (saved["model"], saved["max_tokens"], saved["effort"]) == (
        settings.TEACHER_CHECK_MODEL,
        settings.TEACHER_MAX_TOKENS,
        settings.TEACHER_EFFORT,
    )
    assert sl.batch_state_path("reply_v2").name == "teacher_batch_reply_v2.json"
    assert not list(settings.PROCESSED_DIR.glob("*.tmp"))  # written by rename
    assert not (settings.PROCESSED_DIR / "relabel_batch.json").exists()


def test_batch_submit_refuses_until_collected(files, monkeypatch):
    files()
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "keyword")
    client = _stub_client()
    sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    with pytest.raises(RuntimeError, match="in_progress and not collected"):
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    with pytest.raises(RuntimeError, match="a direct run would pay again"):
        _v2(scorer=Stub())
    sl.teacher_batch_status("reply_v2", _ended(client))
    with pytest.raises(RuntimeError, match="ended and not collected"):  # ended is not enough
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    assert len(client.messages.batches.created) == 1
    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    other = _stub_client()  # each check has its own state
    assert sl.teacher_batch_submit("trump", params=TRUMP_PARAMS, client=other)["n_submitted"] == 20
    st, rep = sl.teacher_batch_collect("reply_v2", client=client)
    assert st["status"] == "collected" and rep["n"] == 10
    again = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    assert again is None and len(client.messages.batches.created) == 1  # nothing left to send
    assert sl.read_batch_state("reply_v2") == st


def test_batch_submit_respects_the_cap(files, monkeypatch):
    files()
    monkeypatch.setattr(settings, "REPLY_TEACHER_V2_MAX_CALLS", 4)
    client = _stub_client()
    with pytest.raises(RuntimeError, match="over the cap of 4"):
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    assert client.messages.batches.created == [] and not sl.batch_state_path("reply_v2").exists()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, limit=4, client=client)
    assert st["n_submitted"] == 4
    _, _, first4 = sl.reply_v2_todo(limit=4)
    assert st["ids"] == first4["id"].tolist()  # the same stratified pilot as the direct --limit


def test_batch_collect_validates_and_is_idempotent(files):
    files()
    answers = [
        '{"score": 0.4, "label": "positive"}',
        '{"score": 1.5}',
        '{"score": NaN}',
        '{"score": true}',
        '{"score": "0.5"}',
        "no json",
        "errored",
        "expired",
        '```json\n{"score": -1.0, "label": "off_topic"}\n```',
        '{"score": 0.0, "label": "neutral"}',
    ]
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    by_id = dict(zip(st["ids"], answers, strict=True))
    client.messages.batches.answer = lambda cid, text: by_id[cid]
    st, rep = sl.teacher_batch_collect("reply_v2", client=client)  # still running
    assert st["status"] == "in_progress" and rep is None
    assert not sl.reply_labels_path(version="v2").exists()
    client.messages.batches.extra = [_result("not-ours", GOOD)]
    st, rep = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    labels = pd.read_parquet(sl.reply_labels_path(version="v2"))
    assert list(labels.columns) == [
        "id",
        "tracked_slug",
        "input_chars",
        "prompt_version",
        "score_teacher",
        "label_teacher",
    ]  # run_labels' columns
    assert sorted(labels["score_teacher"]) == [-1.0, 0.0, 0.4]
    assert dict(zip(labels["score_teacher"], labels["label_teacher"])) == {
        0.4: "positive",
        -1.0: "negative",  # not one of ours: from the score
        0.0: "neutral",
    }
    assert set(labels["prompt_version"]) == {sl.REPLY_TEACHER_V2_PROMPT_VERSION}
    assert st["status"] == "collected" and st["collected"]["labelled"] == 3
    assert st["collected"]["failed_by_outcome"] == {
        "out_of_range": 3,
        "unparseable": 2,
        "errored": 1,
        "expired": 1,
    }
    assert st["collected"]["ignored"]["not_submitted"] == 1
    # Billed: the 8 submitted ids that succeeded; another batch's result is not this one's.
    assert st["spend"]["calls"] == 8 and st["spend"]["output_tokens"] == 8 * 50
    assert st["spend"]["calls_without_usage"] == 0
    assert st["spend"]["usd_batch"] == pytest.approx(st["spend"]["usd_direct"] / 2)
    assert rep["n"] == 3 and json.loads(sl.batch_state_path("reply_v2").read_text()) == st
    report = settings.PROCESSED_DIR / "reply_teacher_check_v2_claude-opus-5.json"
    cache_bytes, polls = (
        sl.reply_labels_path(version="v2").read_bytes(),
        client.messages.batches.retrieved,
    )
    report.unlink()
    st2, rep2 = sl.teacher_batch_collect("reply_v2", client=client)  # collected: no call
    assert rep2["n"] == 3
    assert client.messages.batches.retrieved == polls and report.exists() and st2 == st
    assert sl.reply_labels_path(version="v2").read_bytes() == cache_bytes
    # A collect that died after the label write but before the state write runs again.
    state = sl.batch_state_path("reply_v2")
    state.write_text(json.dumps({**st, "status": "ended"}))
    st3, _ = sl.teacher_batch_collect("reply_v2", client=client)
    assert sl.reply_labels_path(version="v2").read_bytes() == cache_bytes
    assert st3["collected"]["labelled"] == 0 and st3["collected"]["ignored"]["already_cached"] == 3
    nxt = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    assert nxt["n_submitted"] == 7  # the failed answers are sent again, the good ones never


def test_batch_collect_ignores_ids_no_longer_to_label(files, caplog):
    reps = files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    gone, done = st["ids"][0], st["ids"][1]
    files(reps[reps["id"] != gone])  # left the sample after the submit
    pd.DataFrame(
        {
            "id": [done],
            "tracked_slug": ["deal_complete"],
            "input_chars": [5],
            "prompt_version": [sl.REPLY_TEACHER_V2_PROMPT_VERSION],
            "score_teacher": [0.9],
            "label_teacher": ["positive"],
        }
    ).to_parquet(sl.reply_labels_path(version="v2"), index=False)  # labelled meanwhile
    with caplog.at_level(logging.INFO, logger=sl.logger.name):
        st, rep = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    assert st["collected"]["ignored"] == {
        "not_submitted": 0,
        "already_cached": 1,
        "repeated": 0,
        "no_longer_to_label": 1,
        "prompt_changed": 0,
    }
    assert "ignored results" in caplog.text and "no_longer_to_label" in caplog.text
    labels = pd.read_parquet(sl.reply_labels_path(version="v2"))
    assert len(labels) == 9 and gone not in set(labels["id"]) and labels["id"].is_unique
    assert labels.set_index("id").loc[done, "score_teacher"] == 0.9  # the cached label stands
    assert rep["n"] == 9


def test_batch_collect_refuses_another_prompt_version(files):
    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    sl.batch_state_path("reply_v2").write_text(json.dumps({**st, "prompt_version": "reply-v2-old"}))
    with pytest.raises(ValueError, match="reply-v2-old"):
        sl.teacher_batch_collect("reply_v2", client=_ended(client))
    assert not sl.reply_labels_path(version="v2").exists()


@pytest.mark.parametrize("check", ["reply_v2", "trump"])
def test_batch_and_direct_runs_give_the_same_labels_and_report(files, monkeypatch, check):
    files()
    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "keyword")
    if check == "reply_v2":
        labels, report = (
            sl.reply_labels_path(version="v2"),
            settings.PROCESSED_DIR / "reply_teacher_check_v2_claude-opus-5.json",
        )
        _v2(scorer=Stub(score=-0.5))
        params = V2_PARAMS
    else:
        labels, report = (
            sl.trump_labels_path(),
            settings.PROCESSED_DIR / "teacher_check_trump_claude-opus-5.json",
        )
        sl.trump_teacher_check(n=20, concurrency=1, scorer=Stub(score=-0.5))
        params = TRUMP_PARAMS
    direct = pd.read_parquet(labels).sort_values("id").reset_index(drop=True)
    direct_report = report.read_text()
    labels.unlink()
    report.unlink()
    client = _stub_client()
    sl.teacher_batch_submit(check, params=params, client=client)
    sl.teacher_batch_collect(check, client=_ended(client))
    batch = pd.read_parquet(labels).sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(batch, direct)
    assert report.read_text() == direct_report


def test_batch_collect_from_a_results_file(files, tmp_path):
    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "custom_id": i,
            "result": {
                "type": "succeeded",
                "message": {"content": [{"type": "text", "text": GOOD}]},
            },
        }
        for i in st["ids"][:-1]
    ] + [{"custom_id": st["ids"][-1], "result": {"type": "errored", "error": {}}}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    st, rep = sl.teacher_batch_collect("reply_v2", client=_ended(client), results_file=path)
    assert st["collected"]["labelled"] == 9 and st["collected"]["failed_by_outcome"] == {
        "errored": 1
    }
    assert rep["n"] == 9
    # No usage in these lines: counted as unknown, not as calls that cost $0.
    assert st["spend"]["calls"] == 0 and st["spend"]["calls_without_usage"] == 9


def test_pilot_calibration_reproduces_the_pilot_bill():
    p = settings.TEACHER_V2_PILOT
    est = {"calls": p["calls"], "prompt_chars": p["prompt_chars"], "usd": 1.0, "usd_no_cache": 2.0}
    out = sl.with_pilot_and_batch_prices(est)
    billed = sl.price_tokens({f: p[f] for f in sl.Spend.FIELDS})
    assert out["usd_pilot"] == pytest.approx(billed) and billed == pytest.approx(0.0687, abs=1e-4)
    assert out["pilot_usd_per_call"] == pytest.approx(billed / 16)
    assert out["usd_pilot_batch"] == pytest.approx(billed / 2)
    assert (out["usd_batch"], out["usd_batch_no_cache"]) == (0.5, 1.0)
    longer = sl.with_pilot_and_batch_prices({**est, "prompt_chars": 2 * p["prompt_chars"]})
    out_part = sl.price_tokens({"output_tokens": p["output_tokens"]})
    assert longer["usd_pilot"] == pytest.approx(2 * billed - out_part)  # input scales, output not


def test_batch_estimate_cli_makes_no_calls(files, monkeypatch):
    import anthropic

    from src.cli import main

    files()
    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)

    def boom(*a, **k):
        raise AssertionError("an API call")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(sl, "_batch_client", boom)
    monkeypatch.setattr(sl, "_teacher_scorer", boom)
    res = CliRunner().invoke(
        main, ["reply-teacher-check", "--v2", "--batch", "submit", "--estimate"]
    )
    assert res.exit_code == 0, res.output
    est = sl.reply_teacher_v2_estimate()
    assert (
        f"at batch prices (0.5 x direct): ≈ ${est['usd_no_cache'] / 2:.2f} with no prefix cache "
        "(the ceiling)" in res.output
    )
    assert "if the shared prefix cached" not in res.output
    from src.cli import _fmt_estimate

    flagged = _fmt_estimate({**est, "cached_prefix_calls": 3})  # prefixes past the cache minimum
    assert "cached on 3 calls (optimistic: in the v2 pilot the cache engaged" in flagged
    assert f"pilot-calibrated: ≈ ${est['usd_pilot_batch']:.2f} at batch prices" in res.output
    assert "10 calls to claude-opus-5" in res.output and "Run?" not in res.output
    res = CliRunner().invoke(main, ["reply-teacher-check", "--v2", "--estimate"])
    assert f"pilot-calibrated: ≈ ${est['usd_pilot']:.2f} direct" in res.output
    assert "at batch prices (0.5" not in res.output
    res = CliRunner().invoke(
        main, ["teacher-check", "--source", "trump", "--n", "20", "--batch", "submit", "--estimate"]
    )
    assert res.exit_code == 0, res.output
    est = sl.trump_check_estimate(n=20)
    assert f"≈ ${est['usd_no_cache'] / 2:.2f}" in res.output and "pilot-calibrated" in res.output
    assert not list(settings.PROCESSED_DIR.glob("teacher_batch_*.json"))


def test_batch_cli_lifecycle(files, monkeypatch):
    from src.cli import main

    files()
    client = _stub_client()
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    run = CliRunner().invoke
    v2 = ["reply-teacher-check", "--v2"]
    res = run(main, [*v2, "--batch", "status"])
    assert res.exit_code == 1 and "no reply_v2 batch submitted" in res.output
    res = run(main, [*v2, "--batch", "submit"], input="n\n")
    assert "Aborted." in res.output and client.messages.batches.created == []
    res = run(main, [*v2, "--batch", "submit", "--yes"])
    assert res.exit_code == 0, res.output
    assert "batch msgbatch_1 in_progress  (10 requests)" in res.output
    for again in ([*v2, "--batch", "submit", "--yes"], [*v2, "--yes"]):  # batch, then direct
        res = run(main, again)
        assert res.exit_code == 1 and "not collected" in res.output and "Run?" not in res.output
    assert len(client.messages.batches.created) == 1
    res = run(main, [*v2, "--estimate"])
    assert res.exit_code == 0 and "a v2 batch is open" in res.output
    res = run(main, [*v2, "--batch", "collect"])
    assert res.exit_code == 0 and "in_progress" in res.output and "v1 vs v2" not in res.output
    _ended(client)
    res = run(main, [*v2, "--batch", "collect"])
    assert res.exit_code == 0, res.output
    assert "collected" in res.output and "v1 vs v2 labels on 10 replies" in res.output
    assert "at batch prices" in res.output
    res = run(main, [*v2, "--report"])
    assert res.exit_code == 0 and "v1 vs v2 labels on 10 replies" in res.output
    res = run(main, [*v2, "--batch", "submit", "--yes"])
    assert res.exit_code == 0 and "nothing submitted: every row is labelled" in res.output
    assert "msgbatch_1 collected" not in res.output and len(client.messages.batches.created) == 1
    for bad in (
        [*v2, "--batch", "status", "--limit", "3"],
        [*v2, "--batch", "collect", "--estimate"],
        [*v2, "--report", "--batch", "submit"],
        ["reply-teacher-check", "--batch", "submit"],
        ["teacher-check", "--batch", "submit"],
        ["teacher-check", "--report"],
    ):
        res = run(main, bad)
        assert res.exit_code == 2, (bad, res.output)  # click usage error


def test_trump_batch_cli(files, monkeypatch):
    from src.cli import main

    files()
    _feed().to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "keyword")
    client = _stub_client()
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    trump = ["teacher-check", "--source", "trump", "--n", "20"]
    res = CliRunner().invoke(main, [*trump, "--batch", "submit", "--yes"])
    assert res.exit_code == 0 and "(20 requests)" in res.output, res.output
    res = CliRunner().invoke(main, [*trump, "--yes"])
    assert res.exit_code == 1 and "a direct run would pay again" in res.output
    _ended(client)
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--batch", "collect"])
    assert res.exit_code == 0, res.output
    assert "score_opus_distilled vs claude-opus-5 on 20 Trump posts" in res.output
    assert "all posts" in res.output
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--report"])
    assert res.exit_code == 0 and "on 20 Trump posts" in res.output


def _api_line(custom_id, out, usage=True):
    """One line of a results JSONL as the API writes it (thinking block,
    usage with fields we don't read)."""
    if out in ("errored", "expired", "canceled"):
        res = {"type": out}
        if out == "errored":
            res["error"] = {"type": "error", "error": {"type": "overloaded_error"}}
    else:
        msg = {
            "id": "msg_x",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": out},
            ],
            "stop_reason": "end_turn",
        }
        if usage:
            msg["usage"] = {
                "input_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 20,
                "output_tokens": 50,
                "service_tier": "batch",
            }
        res = {"type": "succeeded", "message": msg}
    return json.dumps({"custom_id": custom_id, "result": res}) + "\n"


def test_batch_short_read_leaves_the_batch_open(files, tmp_path):
    """A read without a result for every submitted id (a results file cut on
    a line boundary, or another batch's file) adds its good labels but leaves
    the batch open: the missing ids can still be collected, and are not sent
    and paid for again."""
    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    _ended(client)
    half = tmp_path / "half.jsonl"
    half.write_text("".join(_api_line(i, GOOD) for i in st["ids"][:4]))
    st, rep = sl.teacher_batch_collect("reply_v2", client=client, results_file=half)
    assert rep is None and st["status"] == "ended" and "collected" not in st
    assert st["short_read"]["missing"] == 6 and st["short_read"]["labelled"] == 4
    assert json.loads(sl.batch_state_path("reply_v2").read_text())["status"] == "ended"
    assert len(pd.read_parquet(sl.reply_labels_path(version="v2"))) == 4  # the good ones kept
    with pytest.raises(RuntimeError, match="6 submitted ids without a result yet"):
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    with pytest.raises(RuntimeError, match="a direct run would pay again"):
        _v2(scorer=Stub())
    other = tmp_path / "other.jsonl"  # another batch's results
    other.write_text("".join(_api_line(f"other-{i}", GOOD) for i in range(10)))
    st, rep = sl.teacher_batch_collect("reply_v2", client=client, results_file=other)
    assert rep is None and st["status"] == "ended"
    assert st["short_read"]["missing"] == 10 and st["short_read"]["labelled"] == 0
    assert st["short_read"]["ignored"]["not_submitted"] == 10
    st, rep = sl.teacher_batch_collect("reply_v2", client=client)  # the whole stream
    assert st["status"] == "collected" and "short_read" not in st and rep["n"] == 10
    assert st["collected"]["labelled"] == 6 and st["collected"]["ignored"]["already_cached"] == 4
    assert st["spend"]["calls"] == 10  # what the batch billed, from the read that saw it all
    assert sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client) is None
    assert len(client.messages.batches.created) == 1  # no id paid for twice


def test_batch_results_file_cut_mid_line(files, tmp_path, caplog):
    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=_ended(client))
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(_api_line(i, GOOD) for i in st["ids"])[:-30])  # download stopped
    with caplog.at_level(logging.WARNING, logger=sl.logger.name):
        st, rep = sl.teacher_batch_collect("reply_v2", client=client, results_file=cut)
    assert "1 lines that do not parse were skipped" in caplog.text
    assert rep is None and st["status"] == "ended" and st["short_read"]["missing"] == 1
    assert st["short_read"]["labelled"] == 9


def test_batch_cli_short_read_exits_1(files, monkeypatch, tmp_path):
    from src.cli import main

    files()
    client = _stub_client()
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    run = CliRunner().invoke
    v2 = ["reply-teacher-check", "--v2"]
    assert run(main, [*v2, "--batch", "submit", "--yes"]).exit_code == 0
    ids = sl.read_batch_state("reply_v2")["ids"]
    _ended(client)
    half = tmp_path / "half.jsonl"
    half.write_text("".join(_api_line(i, GOOD) for i in ids[:4]))
    res = run(main, [*v2, "--batch", "collect", "--results-file", str(half)])
    assert res.exit_code == 1 and "short read" in res.output, res.output
    assert "6 of 10 submitted ids have no result" in res.output and "v1 vs v2" not in res.output
    res = run(main, [*v2, "--batch", "submit", "--yes"])
    assert res.exit_code == 1 and "6 submitted ids without a result yet" in res.output
    res = run(main, [*v2, "--batch", "status"])
    assert res.exit_code == 1 and "short read" in res.output
    res = run(main, [*v2, "--batch", "collect"])
    assert res.exit_code == 0 and "v1 vs v2 labels on 10 replies" in res.output, res.output


def test_batch_collect_counts_spend_from_a_results_file(files, tmp_path):
    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=_ended(client))
    path = tmp_path / "results.jsonl"
    path.write_text(
        "".join(_api_line(i, GOOD) for i in st["ids"][:8])
        + _api_line(st["ids"][8], GOOD, usage=False)
        + _api_line(st["ids"][9], "errored")
    )
    st, _ = sl.teacher_batch_collect("reply_v2", client=client, results_file=path)
    s = st["spend"]
    assert st["status"] == "collected" and st["collected"]["labelled"] == 9
    assert (s["calls"], s["calls_without_usage"]) == (8, 1)
    assert (s["input_tokens"], s["cache_read_input_tokens"], s["output_tokens"]) == (800, 160, 400)
    assert s["usd_direct"] == pytest.approx(
        sl.price_tokens({"input_tokens": 800, "cache_read_input_tokens": 160, "output_tokens": 400})
    )
    assert s["usd_batch"] == pytest.approx(s["usd_direct"] / 2) and s["usd_batch"] > 0


def test_streamed_results_are_kept_and_read_back(files):
    """A streamed collect keeps the raw results (paid answers, the ones not
    kept as labels included) as a results JSONL that --results-file reads."""
    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    first = st["ids"][0]
    client.messages.batches.answer = lambda cid, text: "errored" if cid == first else GOOD
    st1, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    kept = sl.batch_archive_path("reply_v2", "msgbatch_1", "_results.jsonl")
    assert kept.name == "teacher_batch_reply_v2_msgbatch_1_results.jsonl"
    assert len(kept.read_text().splitlines()) == 10
    labels = sl.reply_labels_path(version="v2").read_bytes()
    sl.reply_labels_path(version="v2").unlink()
    sl.batch_state_path("reply_v2").write_text(json.dumps({**st, "status": "ended"}))
    st2, _ = sl.teacher_batch_collect("reply_v2", client=client, results_file=kept)
    assert st2["collected"] == st1["collected"] and st2["spend"] == st1["spend"]
    assert sl.reply_labels_path(version="v2").read_bytes() == labels


def test_batch_submit_killed_mid_create_stays_open(files, monkeypatch):
    """A submit killed after the create call went out may have made a paid
    batch: its state stays open, so nothing resends those ids unasked."""
    from src.cli import main

    files()
    client = _stub_client()

    def killed(requests):
        raise KeyboardInterrupt

    client.messages.batches.create = killed
    with pytest.raises(KeyboardInterrupt):
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    st = sl.read_batch_state("reply_v2")
    assert st["status"] == "submitting" and st["batch_id"] is None and st["n_submitted"] == 10
    assert sl.open_batch("reply_v2") == st
    for call in (
        lambda: sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=_stub_client()),
        lambda: _v2(scorer=Stub()),
        lambda: sl.teacher_batch_status("reply_v2", client),
        lambda: sl.teacher_batch_collect("reply_v2", client=client),
    ):
        with pytest.raises(RuntimeError, match="stopped before it recorded a batch id"):
            call()
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    for args in (["--batch", "collect"], ["--batch", "submit", "--yes"], ["--yes"]):
        res = CliRunner().invoke(main, ["reply-teacher-check", "--v2", *args])
        assert res.exit_code == 1 and "stopped before it recorded a batch id" in res.output
        assert "Traceback" not in res.output


def test_batch_submit_refused_by_the_api_and_archives(files):
    """A create the API refuses (4xx) made no batch: the last state comes
    back. One that died without an answer may have: it stays open. A new
    submit archives the last collected batch's state."""
    import anthropic
    import httpx

    files()
    client = _stub_client()
    st = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    first = st["ids"][0]
    client.messages.batches.answer = lambda cid, text: "errored" if cid == first else GOOD
    st1, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    req = httpx.Request("POST", "https://api.anthropic.invalid/v1/messages/batches")

    def refused(requests):
        raise anthropic.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)

    def dropped(requests):
        raise anthropic.APIConnectionError(request=req)

    def failing(create):
        return types.SimpleNamespace(
            messages=types.SimpleNamespace(batches=types.SimpleNamespace(create=create))
        )

    with pytest.raises(anthropic.BadRequestError):
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=failing(refused))
    assert sl.read_batch_state("reply_v2") == st1 and sl.open_batch("reply_v2") is None
    archived = sl.batch_archive_path("reply_v2", "msgbatch_1")
    assert json.loads(archived.read_text()) == st1
    st2 = sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=client)
    assert st2["batch_id"] == "msgbatch_2" and st2["ids"] == [first]  # the failed one again
    assert json.loads(archived.read_text()) == st1
    st2, _ = sl.teacher_batch_collect("reply_v2", client=_ended(client))
    sl.reply_labels_path(version="v2").unlink()  # so there is something to submit
    with pytest.raises(anthropic.APIConnectionError):
        sl.teacher_batch_submit("reply_v2", params=V2_PARAMS, client=failing(dropped))
    assert sl.read_batch_state("reply_v2")["status"] == "submitting"
    assert json.loads(sl.batch_archive_path("reply_v2", "msgbatch_2").read_text()) == st2


def test_batch_collect_without_a_batch_needs_no_key(files, monkeypatch):
    from src.cli import main

    files()

    def boom():
        raise AssertionError("a client was built")

    monkeypatch.setattr(sl, "_batch_client", boom)
    with pytest.raises(FileNotFoundError, match="no reply_v2 batch submitted"):
        sl.teacher_batch_collect("reply_v2")
    res = CliRunner().invoke(main, ["reply-teacher-check", "--v2", "--batch", "collect"])
    assert res.exit_code == 1 and "no reply_v2 batch submitted" in res.output, res.output


def test_batch_collect_drops_answers_to_a_changed_prompt(files, monkeypatch):
    """A post whose text changed between submit and collect (a rescored feed)
    keeps no label from the old prompt; the next submit sends the new one."""
    files()
    feed = _feed()
    feed.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    monkeypatch.setattr(settings, "TOPIC_SOURCE", "keyword")
    client = _stub_client()
    st = sl.teacher_batch_submit("trump", params=TRUMP_PARAMS, client=client)
    assert set(st["prompt_sha256"]) == set(st["ids"])
    changed = st["ids"][0]
    feed.loc[feed["id"] == changed, "text"] = "Iran must open the Strait, edited"
    feed.to_parquet(settings.TRUMP_FEED_STANCE, index=False)
    st, rep = sl.teacher_batch_collect("trump", client=_ended(client))
    assert st["collected"]["ignored"]["prompt_changed"] == 1 and st["collected"]["labelled"] == 19
    assert changed not in set(pd.read_parquet(sl.trump_labels_path())["id"]) and rep["n"] == 19
    nxt = sl.teacher_batch_submit("trump", params=TRUMP_PARAMS, client=client)
    assert nxt["ids"] == [changed]
    sent = _content_text(client.messages.batches.created[-1][0]["params"]["messages"][0]["content"])
    assert "edited" in sent
