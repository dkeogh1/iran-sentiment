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
