"""The alternate post raters (stance-topic, rater, stance-topic-nohandle), the
Trump whole-feed check and its merge, the unjudgeable topic record and the
reply rater arm, all with stubs: no API, no network, no API key."""

import json
import types

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings
from src.analysis import post_labels as pl
from src.analysis import stance_local as sl
from src.analysis import topic_label as tl
from src.analysis.sentiment import llm_prompt
from tests.test_reply_labels import env  # noqa: F401 -- the fixture
from tests.test_teacher_checks import _ended, _stub_client

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

TIERS = ("admin", "maga_prowar", "maga_antiwar", "media", "opposition", "religious_authority")
O5, O55 = "claude-opus-5", "claude-opus-5-5"


def _x_rows(n=8):
    """Per tier: i % 4 == 0 names Iran (keyword), 1 only Haiku flags, 2-3 no
    filter flags; plus search rows and a bare link."""
    rows = []
    for tier in TIERS:
        for i in range(n):
            kind = ("Strike Iran now", "A quiet day", "Lunch with friends", "Lunch with friends")
            rows.append(
                {
                    "id": f"x_{tier}_{i}",
                    "user": f"acct_{tier}",
                    "tier": tier,
                    "created_at": f"2026-0{4 + i % 5}-1{i % 9}T12:00:00.000Z",
                    "text": f"{kind[i % 4]} {tier} {i}",
                    "score_opus": 0.5 if i % 4 == 0 else -0.2,
                    "label_opus": "positive",
                }
            )
    for i in range(3):
        rows.append(
            {
                "id": f"s_{i}",
                "user": "search:iran war",
                "tier": "search",
                "created_at": "2026-05-01T00:00:00.000Z",
                "text": f"Iran war search {i}",
                "score_opus": 0.1,
                "label_opus": "positive",
            }
        )
    rows.append(
        {
            "id": "x_link",
            "user": "acct_admin",
            "tier": "admin",
            "created_at": "2026-05-01T00:00:00.000Z",
            "text": "https://t.co/abc",
            "score_opus": 0.0,
            "label_opus": "neutral",
        }
    )
    return pd.DataFrame(rows)


def _trump_rows(n=6):
    return pd.DataFrame(
        {
            "id": [f"t_{i}" for i in range(n)],
            "user": "realDonaldTrump",
            "tier": "admin",
            "platform": "truthsocial",
            "created_at": [f"2026-05-0{i + 1}T00:00:00.000Z" for i in range(n)],
            "text": [
                "Iran must open the Strait of Hormuz" if i % 2 == 0 else f"Great rally {i}"
                for i in range(n)
            ],
            "score_opus_distilled": np.linspace(-0.5, 0.5, n),
        }
    )


@pytest.fixture
def posts(tmp_path, monkeypatch):
    """The X frame, Trump's feed and the Haiku topic labels in tmp_path."""
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(settings, "SENTIMENT_OUTPUT", tmp_path / "sentiment_all.parquet")
    monkeypatch.setattr(settings, "TRUMP_FEED_STANCE", tmp_path / "trump.parquet")
    monkeypatch.setattr(settings, "POST_RANDOM_N", 6)
    monkeypatch.setattr(settings, "POST_NOHANDLE_PER_TIER", 1)
    monkeypatch.setattr(settings, "POST_PILOT_MAX_CALLS", 40)
    monkeypatch.setattr(settings, "POST_PILOT_MIN_BILLED", 1)  # the stub pilots are small
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def write(x=None, trump=None):
        x = _x_rows() if x is None else x
        x.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
        (_trump_rows() if trump is None else trump).to_parquet(
            settings.TRUMP_FEED_STANCE, index=False
        )
        lab = x[x["text"].str.startswith(("A quiet day", "Lunch"))][["id", "text"]]
        lab = lab.assign(about_war=lab["text"].str.startswith("A quiet day"))[["id", "about_war"]]
        lab.to_parquet(tl.labels_path(), index=False)
        return x

    return write


def _no_api(monkeypatch):
    import anthropic

    def boom(*a, **k):
        raise AssertionError("an API call")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(sl, "_batch_client", boom)
    monkeypatch.setattr(pl, "_direct_caller", boom)


def _answer(cid, text):
    """A stance-topic or llm_prompt answer by the post's words."""
    post = text.rsplit("Post: ", 1)[-1]
    if "Score the sentiment" in text:
        return '{"score": 0.3, "label": "positive", "reasoning": "x"}'
    if "Strike Iran" in post or "Strait" in post:
        return '{"topic": "iran_war", "score": 0.6, "reasoning": "x"}'
    if "quiet day" in post:
        return '{"topic": "iran_war", "score": -0.4, "reasoning": "x"}'
    return '{"topic": "not_war", "score": null, "reasoning": "x"}'


class Caller:
    """A direct caller: records (prompt, cache_chars), answers _answer."""

    def __init__(self, out=90, stop="end_turn", answer=_answer, cache=True):
        self.calls, self.out, self.stop, self.answer, self.cache = [], out, stop, answer, cache

    def __call__(self, prompt, cache_chars=0):
        self.calls.append((prompt, cache_chars))
        first = len(self.calls) == 1
        hit = 700 if self.cache and cache_chars else 0
        usage = {
            "input_tokens": 120,
            "cache_creation_input_tokens": hit if first else 0,
            "cache_read_input_tokens": 0 if first else hit,
            "output_tokens": self.out,
        }
        return self.answer("", prompt), usage, self.stop


# ── prompts and columns ────────────────────────────────────────────


def test_stance_topic_prompt_handle_and_cached_prefix():
    with_h, c1 = pl.stance_topic_prompt("We will strike tonight", "acct_x")
    without, c2 = pl.stance_topic_prompt("We will strike tonight")
    assert c1 == c2 == len(pl.STANCE_TOPIC_INSTRUCTIONS)
    assert with_h[:c1] == without[:c2] == pl.STANCE_TOPIC_INSTRUCTIONS  # the same cached block
    # The handle sits where llm_prompt has it: just before the post.
    assert with_h[c1:] == 'The post is by @acct_x.\n\nPost: """We will strike tonight"""'
    assert without == with_h.replace("The post is by @acct_x.\n\n", "")  # nothing else differs
    assert "acct_x" not in without
    for t in pl.TOPIC_CLASSES:
        assert t in pl.STANCE_TOPIC_INSTRUCTIONS
    # Long enough to cache on Opus 5 even at prose's characters a token.
    per = settings.POST_PROMPT_TOKEN_FIT["instruction_chars_per_token"]
    assert (
        len(pl.STANCE_TOPIC_INSTRUCTIONS) / per
        > 1.2 * settings.MODEL_PRICES[O5]["cache_min_tokens"]
    )
    assert pl.JOBS["rater"].prompt("a post", "acct") == (llm_prompt("a post", "acct"), 0)
    assert pl.JOBS["stance-topic-nohandle"].prompt("p", "acct")[0] == pl.stance_topic_prompt("p")[0]


def test_columns_never_collide_with_score_opus():
    assert pl.JOBS["stance-topic"].columns(O5) == {
        "score": "score_opus_stance",
        "label": "label_opus_stance",
        "topic": "topic_opus",
    }
    assert pl.JOBS["rater"].columns(O55) == {"score": "score_opus55", "label": "label_opus55"}
    assert pl.JOBS["stance-topic-nohandle"].columns(O5)["score"] == "score_opus_stance_nohandle"
    assert pl.JOBS["stance-topic-nohandle"].columns(O5)["topic"] == "topic_opus_nohandle"
    taken = {settings.STANCE_SCORE_COL, *settings.PLOT_SCORE_COLS, "score_opus_distilled"}
    for j in pl.JOBS.values():
        for m in settings.MODEL_PRICES:
            if j.name == "rater" and m in settings.MODEL_TAGS:
                with pytest.raises(ValueError, match="second model"):
                    pl.check_model(j, m)
                continue
            assert not taken & set(j.columns(m).values()), (j.name, m)
    assert "score_opus55" not in settings.PLOT_SCORE_COLS
    assert settings.STANCE_SCORE_COL == "score_opus"
    with pytest.raises(KeyError, match="no price"):
        pl.check_model(pl.JOBS["rater"], "claude-sonnet-9")


def test_registries_match():
    from src.analysis import reply_labels as rlab
    from src.cli import POST_LABEL_JOBS, REPLY_LABEL_JOBS

    assert tuple(pl.JOBS) == POST_LABEL_JOBS
    assert tuple(rlab.JOBS) == REPLY_LABEL_JOBS
    checks = sl.batch_checks()
    assert {j.check for j in pl.JOBS.values()} <= set(checks)
    assert not any(c.carries_replies for c in pl.POST_BATCH_CHECKS.values())  # public posts
    assert "v2-rater" in sl.REPLY_LABEL_VERSIONS


# ── the posts ──────────────────────────────────────────────────────


def test_candidates_random_and_subsample(posts):
    posts()
    f = pl.post_frame()
    assert "x_link" not in set(f["id"]) and not f["id"].str.startswith("s_").any()
    cand = set(f.loc[f["candidate"], "id"])
    keyword = {f"x_{t}_{i}" for t in TIERS for i in (0, 4)}
    haiku = {f"x_{t}_{i}" for t in TIERS for i in (1, 5)}
    assert cand == keyword | haiku | {"t_0", "t_2", "t_4"}
    rnd = pl.select_random(f)
    assert len(rnd) == 6 and not set(rnd["id"]) & cand
    assert set(rnd["id"]) <= set(f.loc[f["source"] == "x", "id"])
    assert rnd["id"].tolist() == pl.select_random(f.sample(frac=1, random_state=1))["id"].tolist()
    sub = pl.select_subsample(f)
    by = sub.groupby("tier").size().to_dict()
    assert by == {
        "religious_authority": 4,
        "opposition": 4,
        "admin": 1,
        "maga_prowar": 1,
        "maga_antiwar": 1,
        "media": 1,
    }
    assert set(sub["id"]) <= cand


def test_estimate_makes_no_call_and_writes_nothing(posts, monkeypatch):
    from src.cli import main

    posts()
    _no_api(monkeypatch)
    before = sorted(p.name for p in settings.PROCESSED_DIR.iterdir())
    for job in pl.JOBS:
        res = CliRunner().invoke(main, ["post-label", "estimate", "--job", job])
        assert res.exit_code == 0, res.output
        assert "no API call" in res.output
    est = pl.job_estimate("stance-topic")
    assert est["preview"] and est["todo"] == 27 + 6  # 24 X + 3 Trump candidates, 6 random
    assert est["posts"] == {"random/x": 6, "war/trump": 3, "war/x": 24}
    assert sorted(p.name for p in settings.PROCESSED_DIR.iterdir()) == before
    rater = pl.job_estimate("rater")
    assert rater["model"] == settings.SECOND_RATER_MODEL and rater["todo"] == 27
    assert rater["problem"] is None  # 27 calls is a pilot's size
    monkeypatch.setattr(settings, "POST_PILOT_MAX_CALLS", 10)
    assert "unmeasured" in (pl.job_estimate("rater")["problem"] or "")
    res = CliRunner().invoke(main, ["post-label", "submit", "--job", "rater", "--yes"])
    assert res.exit_code == 1 and "unmeasured" in res.output
    with pytest.raises(FileNotFoundError, match="records the draw"):
        pl.job_posts(pl.JOBS["stance-topic"], pl.post_frame())


def test_cost_model_counts_the_cache_and_the_pilot(posts, monkeypatch):
    posts()
    j = pl.JOBS["stance-topic"]
    _, _, todo, _ = pl.job_todo(j, O5, {"seed": 1}, preview=True)
    assert (todo["cache_chars"] == len(pl.STANCE_TOPIC_INSTRUCTIONS)).all()
    fit, per = (
        settings.POST_PROMPT_TOKEN_FIT,
        settings.POST_PROMPT_TOKEN_FIT["instruction_chars_per_token"],
    )
    r = todo.iloc[0]
    want = (
        fit[r["source"]]["fixed_tokens"]
        + fit[r["source"]]["tokens_per_char"] * r["base_chars"]
        + (r["prompt_chars"] - r["base_chars"]) / per
    )
    assert pl.predicted_input_tokens(todo.head(1))[0] == pytest.approx(want)
    c = pl.calibrated_cost(j, todo, O5)
    assert c["cache"]["engaged"] and not c["output_measured"]
    assert c["usd_batch"] < c["usd_batch_ceiling"]
    # With the cache off, expected input is the plain price; the ceiling writes every prefix.
    monkeypatch.setitem(settings.MODEL_PRICES[O5], "cache_min_tokens", 10**6)
    plain = pl.calibrated_cost(j, todo, O5)
    assert not plain["cache"]["engaged"] and plain["usd_batch"] > c["usd_batch"]
    tin = pl.predicted_input_tokens(todo)
    pre = len(pl.STANCE_TOPIC_INSTRUCTIONS) / per
    out = 2 * settings.POST_OUTPUT_TOKENS_GUESS["stance-topic"]
    assert plain["usd_batch_ceiling"] == pytest.approx(
        ((tin.sum() + 0.25 * pre * len(todo)) * 5 + out * len(todo) * 25) / 1e6 * 0.5
    )
    # The rater on llm_prompt and Opus 5 is measured by the fit; on Opus 5.5 a guess.
    rj = pl.JOBS["rater"]
    _, _, rt, _ = pl.job_todo(rj, O55, {"seed": 1}, preview=True)
    assert pl.output_basis(rj, O5)["measured"]
    assert not pl.output_basis(rj, O55)["measured"]
    assert pl.calibrated_cost(rj, rt, O55)["output_tokens_per_call"] == pytest.approx(
        2 * np.mean([fit[s]["output_tokens"] for s in rt["source"]])
    )
    # llm_prompt_cost is the Trump check's usage-fit figure.
    t = pl.llm_prompt_cost([1000, 500], "trump", O5)
    tok = 2 * 35 + 0.3628 * 1500
    assert t["usd_batch"] == pytest.approx((tok * 5 + 2 * 77 * 25) / 1e6 * 0.5)


# ── pilot ──────────────────────────────────────────────────────────


def test_pilot_labels_measures_and_calibrates(posts, monkeypatch):
    posts()
    caller = Caller(out=95)
    got = pl.run_pilot("stance-topic", n=10, caller=caller, concurrency=1)
    assert len(caller.calls) == 10
    assert all(c == len(pl.STANCE_TOPIC_INSTRUCTIONS) for _, c in caller.calls)
    assert pl.random_manifest_path().exists()  # the draw is recorded before anything is sent
    lab = pd.read_parquet(pl.labels_path("stance-topic", O5))
    assert len(lab) == 10 and set(lab["prompt_version"]) == {pl.STANCE_TOPIC_PROMPT_VERSION}
    assert {"prompt", "cache_chars"}.isdisjoint(lab.columns)
    assert set(lab["post_set"]) == {"war", "random"}  # dealt across sets
    war = lab[lab["topic"] == "iran_war"]
    assert (
        war["score_teacher"].notna().all()
        and lab.loc[lab["topic"] != "iran_war", "score_teacher"].isna().all()
    )
    assert got["calls"] == 10 and got["output_mean"] == 95 and got["cache_engaged"]
    assert got["prefix_tokens"] == 700 and got["stop_reasons"] == {"end_turn": 10}
    est = pl.job_estimate("stance-topic")
    assert est["output_measured"] and est["todo"] == 33 - 10
    assert est["output_tokens_per_call"] == 95 and est["cache"]["basis"] == "measured by the pilot"
    assert est["problem"] is None
    pl.run_pilot("stance-topic", n=5, caller=Caller(), concurrency=1)
    lab = pd.read_parquet(pl.labels_path("stance-topic", O5))
    assert len(lab) == 15 and lab["id"].is_unique  # labelled posts are never sent again
    assert len(pl.read_pilot("stance-topic", O5)["runs"]) == 2
    with pytest.raises(ValueError, match="POST_PILOT_MAX_CALLS"):
        pl.run_pilot("stance-topic", n=settings.POST_PILOT_MAX_CALLS + 1, caller=Caller())


def test_pilot_max_tokens_stop_blocks_the_batch(posts, monkeypatch):
    posts()
    bad = Caller(stop="max_tokens", answer=lambda cid, text: "")
    got = pl.run_pilot("rater", n=4, caller=bad, concurrency=1)
    assert got["stop_reasons"] == {"max_tokens": 4} and got["kept"] == 0
    assert (
        not pl.labels_path("rater", O55).exists()
        or pd.read_parquet(pl.labels_path("rater", O55)).empty
    )
    why = pl.submit_problem(pl.JOBS["rater"], O55, None, 20)
    assert "max_tokens" in why
    # A higher max_tokens starts the measurement over (unmeasured again).
    monkeypatch.setattr(settings, "POST_LABEL_MAX_TOKENS", 2048)
    assert "unmeasured" in pl.submit_problem(pl.JOBS["rater"], O55, None, 200)
    assert pl.submit_problem(pl.JOBS["rater"], O55, 20, 20) is None  # a --limit pilot may go


def test_direct_pilot_sends_the_batch_request(posts, monkeypatch):
    import anthropic
    import dotenv

    posts()
    sent = []
    resp = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text='{"topic": "not_war", "score": null}')],
        usage=types.SimpleNamespace(
            input_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=5,
        ),
        stop_reason="end_turn",
    )

    class _Client:
        def __init__(self, *a, **k):
            self.messages = types.SimpleNamespace(create=lambda **kw: sent.append(kw) or resp)

    monkeypatch.setattr(anthropic, "Anthropic", _Client)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    _, _, todo, _ = pl.job_todo(pl.JOBS["stance-topic"], O5, {"seed": 1}, limit=3, preview=True)
    call = pl._direct_caller(O5, 1024, "low")
    for p, c in zip(todo["prompt"], todo["cache_chars"]):
        assert call(p, int(c))[2] == "end_turn"
    reqs = sl.build_batch_requests(todo, O5, 1024, "low")
    assert [dict(r["params"]) for r in reqs] == sent
    block = sent[0]["messages"][0]["content"][0]
    assert (
        block["text"] == pl.STANCE_TOPIC_INSTRUCTIONS
        and block["cache_control"]["type"] == "ephemeral"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        pl._direct_caller(O5, 1024, "low")


# ── batch ──────────────────────────────────────────────────────────


def test_batch_lifecycle_parses_topic_answers(posts, monkeypatch):
    from src.cli import main

    posts()

    def answer(cid, text):
        if cid == "x_admin_0":
            return '{"topic": "iran_war", "score": null}'  # no score: not kept
        if cid == "x_media_0":
            return '{"topic": "the weather", "score": 0.2}'  # no such topic
        if cid == "x_media_4":
            return '{"topic": "not_war", "score": 0.9}'  # a score off the war is dropped
        return _answer(cid, text)

    client = _stub_client(answer)
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    pl.run_pilot("stance-topic", n=2, caller=Caller(), concurrency=1)  # measured
    res = CliRunner().invoke(main, ["post-label", "submit", "--job", "stance-topic", "--yes"])
    assert res.exit_code == 0 and "(31 requests)" in res.output, res.output
    st = sl.read_batch_state("post_stance_topic")
    assert st["max_tokens"] == settings.POST_LABEL_MAX_TOKENS and st["effort"] == "low"
    req = client.messages.batches.created[0][0]
    assert req["params"]["messages"][0]["content"][0]["text"] == pl.STANCE_TOPIC_INSTRUCTIONS
    _ended(client)
    res = CliRunner().invoke(main, ["post-label", "collect", "--job", "stance-topic"])
    assert res.exit_code == 0, res.output
    assert "stop reasons {'None': 31}" in res.output
    lab = pd.read_parquet(pl.labels_path("stance-topic", O5)).set_index("id")
    assert "x_admin_0" not in lab.index and "x_media_0" not in lab.index
    assert lab.loc["x_media_4", "topic"] == "not_war" and np.isnan(
        lab.loc["x_media_4", "score_teacher"]
    )
    assert lab.loc["t_0", "score_teacher"] == 0.6 and lab.loc["t_0", "label_teacher"] == "positive"
    st = sl.read_batch_state("post_stance_topic")
    assert st["collected"]["failed_by_outcome"] == {"no_score": 1, "bad_topic": 1}
    assert pl.report_path("stance-topic", O5).exists()


def test_rater_batch_and_report(posts, monkeypatch):
    posts()
    pd.DataFrame(
        {
            "id": ["t_0", "t_2"],
            "input_chars": [35, 35],
            "prompt_version": "llm_prompt",
            "score_teacher": [0.1, 0.2],
            "label_teacher": "positive",
        }
    ).to_parquet(sl.trump_labels_path(O5), index=False)
    client = _stub_client(lambda cid, text: _answer(cid, text))
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    monkeypatch.setattr(settings, "POST_PILOT_MAX_CALLS", 10)
    with pytest.raises(RuntimeError, match="unmeasured"):
        sl.teacher_batch_submit("post_rater", params={"seed": 1}, model=O55)
    assert not client.messages.batches.created
    monkeypatch.setattr(settings, "POST_PILOT_MAX_CALLS", 40)
    st = sl.teacher_batch_submit("post_rater", params={"seed": 1}, model=O55, limit=30)
    assert st["n_submitted"] == 27
    prompts = [r["params"]["messages"][0]["content"] for r in client.messages.batches.created[0]]
    assert all(p.startswith("Score the sentiment") for p in prompts)  # llm_prompt, no blocks
    _ended(client)
    st, rep = sl.teacher_batch_collect("post_rater")
    assert st["status"] == "collected" and rep["n"] == 27
    by = {r["group"]: r for r in rep["vs_score_opus"]["by_group"]}
    assert by["admin"]["n"] == 4 and by["trump"]["n"] == 2  # Trump's reference: the check labels
    assert by["ALL"]["mean_new"] == pytest.approx(0.3)
    assert by["admin"]["mean_diff"] == pytest.approx(0.3 - np.mean([0.5, 0.5, -0.2, -0.2]))
    assert {"spearman", "loa_lo", "loa_hi", "sign_flips"} <= set(by["ALL"])
    assert {r["phase"] for r in rep["vs_score_opus"]["by_phase"]} >= {"ALL"}


def test_rater_stats_and_wilson():
    ref = np.array([-0.5, 0.0, 0.5, 0.8, np.nan])
    new = np.array([-0.4, 0.1, -0.6, 0.9, 0.2])
    s = pl.rater_stats(ref, new)
    assert s["n"] == 4 and s["sign_flips"] == 1
    d = new[:4] - ref[:4]
    assert s["mean_diff"] == pytest.approx(d.mean()) and s["sd_diff"] == pytest.approx(
        d.std(ddof=1)
    )
    assert s["loa_hi"] - s["loa_lo"] == pytest.approx(2 * 1.96 * d.std(ddof=1))
    assert s["spearman"] == pytest.approx(0.4)
    p, lo, hi = pl.wilson(3, 100)
    assert p == 0.03 and 0.01 < lo < 0.03 < hi < 0.09
    assert np.isnan(pl.wilson(0, 0)[0])


# ── read time: phases ──────────────────────────────────────────────


def _label_file(job, model, rows):
    j = pl.JOBS[job]
    df = pd.DataFrame(rows)
    df["prompt_version"] = j.prompt_version
    df.to_parquet(pl.labels_path(job, model), index=False)


def test_attach_joins_only_candidates_and_current_text(posts):
    x = posts()
    ln = dict(zip(x["id"], x["text"].str.len()))
    _label_file(
        "stance-topic",
        O5,
        [
            {
                "id": "x_admin_0",
                "input_chars": ln["x_admin_0"],
                "topic": "iran_war",
                "score_teacher": 0.7,
                "label_teacher": "positive",
                "post_set": "war",
            },
            {
                "id": "x_admin_1",
                "input_chars": 3,
                "topic": "iran_war",  # text since changed
                "score_teacher": 0.1,
                "label_teacher": "positive",
                "post_set": "war",
            },
            {
                "id": "x_admin_2",
                "input_chars": ln["x_admin_2"],
                "topic": "iran_war",  # random arm
                "score_teacher": 0.9,
                "label_teacher": "positive",
                "post_set": "random",
            },
        ],
    )
    df = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    out = pl.attach(df, "score_opus_stance").set_index("id")
    assert out.loc["x_admin_0", "score_opus_stance"] == 0.7
    assert out.loc["x_admin_0", "topic_opus"] == "iran_war"
    assert np.isnan(out.loc["x_admin_1", "score_opus_stance"])  # stale
    assert np.isnan(out.loc["x_admin_2", "score_opus_stance"])  # never joins a series
    assert out["score_opus_stance"].notna().sum() == 1
    with pytest.raises(ValueError, match="never overwrites"):
        pl.attach(out.reset_index(), "score_opus_stance")
    with pytest.raises(LookupError, match="no post-label job"):
        pl.attach(df, "score_nobody")
    with pytest.raises(LookupError, match="not a series"):
        pl.attach(df, "score_opus_stance_nohandle")
    with pytest.raises(LookupError, match="no rater labels"):
        pl.attach(df, "score_opus55")


def test_phases_reads_post_label_columns_and_opus_topic(posts, monkeypatch, caplog):
    from src.cli import main

    x = posts()
    ln = dict(zip(x["id"], x["text"].str.len()))
    cand = [f"x_{t}_{i}" for t in TIERS for i in (0, 1, 4, 5)]
    rows = [
        {
            "id": i,
            "input_chars": ln[i],
            "topic": "iran_war" if i.endswith(("_0", "_1")) else "not_war",
            "score_teacher": 0.4 if i.endswith(("_0", "_1")) else np.nan,
            "label_teacher": None,
            "post_set": "war",
        }
        for i in cand
    ]
    rows.append(
        {
            "id": "x_admin_2",
            "input_chars": ln["x_admin_2"],
            "topic": "iran_war",
            "score_teacher": 0.9,
            "label_teacher": None,
            "post_set": "random",
        }
    )
    _label_file("stance-topic", O5, rows)
    run = CliRunner().invoke(
        main, ["phases", "--score", "score_opus_stance", "--topic", "opus", "--n-boot", "20"]
    )
    assert run.exit_code == 0, run.output
    assert "covers 12 of" in run.output and "calls iran_war" in run.output
    assert (settings.PROCESSED_DIR / "phases_x_tier_score_opus_stance_opus.csv").exists()
    assert "AND Opus iran_war" not in run.output  # --topic opus: the war set is Opus's own
    # Under the default flag the score's nulls still cut the war set to Opus's
    # iran_war candidates: said so, with the like-for-like baseline.
    run = CliRunner().invoke(main, ["phases", "--score", "score_opus_stance", "--n-boot", "20"])
    assert run.exit_code == 0, run.output
    assert "war candidates Opus classes iran_war with score_opus_stance" in run.output
    assert "either flag AND Opus iran_war" in run.output
    assert "`phases --score score_opus --topic opus`" in run.output
    # --topic opus on score_opus: candidates Opus calls iran_war, not the random arm's.
    from src.analysis import inference as inf

    df = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    d = inf.prepare(df, "score_opus", topic_source="opus")
    flagged = set(d.loc[d["on_topic"], "id"])
    assert flagged == {f"x_{t}_{i}" for t in TIERS for i in (0, 1)}
    # A candidate with no label yet counts as the either flag does, with a warning.
    _label_file("stance-topic", O5, rows[1:])
    with caplog.at_level("WARNING"):
        d = inf.prepare(df, "score_opus", topic_source="opus")
    assert "x_admin_0" in set(d.loc[d["on_topic"], "id"]) and "no stance-topic label" in caplog.text
    pl.labels_path("stance-topic", O5).unlink()
    run = CliRunner().invoke(main, ["phases", "--topic", "opus", "--n-boot", "5"])
    assert run.exit_code == 1 and "no stance-topic labels" in run.output
    run = CliRunner().invoke(main, ["phases", "--score", "score_opus55", "--n-boot", "5"])
    assert run.exit_code == 1 and "no rater labels" in run.output


# ── reports ────────────────────────────────────────────────────────


def test_stance_topic_and_nohandle_reports(posts, monkeypatch):
    posts()
    pd.DataFrame(
        {
            "id": ["t_0"],
            "input_chars": [35],
            "prompt_version": "llm_prompt",
            "score_teacher": [0.2],
            "label_teacher": "positive",
        }
    ).to_parquet(sl.trump_labels_path(O5), index=False)
    pl.run_pilot("stance-topic", n=33, caller=Caller(), concurrency=1)
    rep = pl.job_report("stance-topic")
    flags = {r["flag"]: r for r in rep["filters_vs_topic"]}
    assert flags["keyword (no Haiku label)"]["n"] == 12 + 3  # X keyword posts + Trump's
    assert flags["Haiku only"]["n"] == 12 and flags["Haiku only"]["iran_war_share"] == 1.0
    rnd = {r["tier"]: r for r in rep["random_arm"]}
    assert rnd["ALL"]["n"] == 6 and rnd["ALL"]["iran_war"] == 0 and rnd["ALL"]["share_hi"] > 0
    assert rnd["ALL"]["pool_now"] == 24
    vs = {r["group"]: r for r in rep["vs_score_opus_on_iran_war"]["by_group"]}
    assert vs["trump"]["n"] == 1 and vs["admin"]["n"] == 4
    assert vs["admin"]["mean_diff"] == pytest.approx(
        np.mean([0.6 - 0.5, 0.6 - 0.5, -0.4 + 0.2, -0.4 + 0.2])
    )
    lines = pl.summary_lines(rep)
    assert any("random, ALL" in s for s in lines) and any("against score_opus" in s for s in lines)
    assert pl.report_path("stance-topic", O5).exists()

    # No-handle arm, paired with stance-topic's labels on the same posts.
    def nohandle(cid, text):
        post = text.rsplit("Post: ", 1)[-1]
        return '{"topic": "iran_war", "score": 0.2}' if "Strike" in post else _answer(cid, text)

    pl.run_pilot("stance-topic-nohandle", n=12, caller=Caller(answer=nohandle), concurrency=1)
    prompts = pd.read_parquet(pl.labels_path("stance-topic-nohandle", O5))
    assert len(prompts) == 12 and pl.subsample_manifest_path().exists()
    rep = pl.job_report("stance-topic-nohandle")
    by = {r["tier"]: r for r in rep["paired_by_tier"]}
    assert by["ALL"]["pairs"] == 12 and rep["unpaired"] == 0
    rel = by["religious_authority"]
    assert rel["pairs"] == 4 and rel["n_both_iran_war"] == 4 and rel["topic_moved"] == 0
    assert rel["diff"] == pytest.approx(np.mean([0.2 - 0.6, 0.2 - 0.6, 0, 0]))
    assert rel["diff_lo"] < rel["diff"] < rel["diff_hi"]


# ── Trump: the whole feed and the merge ────────────────────────────


def test_trump_whole_feed_estimate_and_merge(posts, monkeypatch):
    from src.cli import main

    posts()
    _no_api(monkeypatch)
    feed = _trump_rows()
    assert len(sl.trump_sample(None)) == 6 and set(sl.trump_sample(2)["id"]) <= set(
        sl.trump_sample(None)["id"]
    )
    pd.DataFrame(
        {
            "id": ["t_0", "t_1"],
            "input_chars": [len(feed.loc[0, "text"]), 3],
            "prompt_version": "llm_prompt",
            "score_teacher": [0.4, -0.3],
            "label_teacher": ["positive", "negative"],
        }
    ).to_parquet(sl.trump_labels_path(O5), index=False)
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--all", "--estimate"])
    assert res.exit_code == 0, res.output
    # t_1's label was made from another text: it does not count, t_1 is labelled again.
    assert "6 posts with text (the whole feed), 1 cached" in res.output
    assert "usage-fit:" in res.output and "5 calls to claude-opus-5" in res.output
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--all", "--n", "3"])
    assert res.exit_code == 2
    res = CliRunner().invoke(main, ["phases", "--source", "trump", "--score", "score_opus"])
    assert res.exit_code == 1 and "--merge" in res.output
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--merge"])
    assert res.exit_code == 0, res.output
    assert (
        "score_opus on 1 of 6" in res.output
        and "1 labels were made from another text" in res.output
    )
    merged = pd.read_parquet(settings.TRUMP_FEED_STANCE).set_index("id")
    assert merged.loc["t_0", "score_opus"] == 0.4 and merged.loc["t_0", "label_opus"] == "positive"
    assert np.isnan(merged.loc["t_1", "score_opus"])  # stale: left out
    assert merged["score_opus_distilled"].tolist() == feed["score_opus_distilled"].tolist()
    reg = json.loads(sl.trump_columns_path().read_text())
    assert reg == {"score_opus": O5}
    res = CliRunner().invoke(
        main, ["phases", "--source", "trump", "--score", "score_opus", "--n-boot", "5"]
    )
    assert res.exit_code == 0 and "covers 1 of 6" in res.output, res.output
    # Another model never writes score_opus; a merge of its own labels claims its own column.
    monkeypatch.setattr(settings, "MODEL_TAGS", {**settings.MODEL_TAGS, O55: "opus"})
    pd.read_parquet(sl.trump_labels_path(O5)).to_parquet(sl.trump_labels_path(O55), index=False)
    with pytest.raises(ValueError, match="score_opus holds labels from claude-opus-5"):
        sl.trump_merge(O55)


def test_trump_whole_feed_batch_reuses_cached_labels(posts, monkeypatch):
    posts()
    pd.DataFrame(
        {
            "id": ["t_0"],
            "input_chars": [35],
            "prompt_version": "llm_prompt",
            "score_teacher": [0.4],
            "label_teacher": "positive",
        }
    ).to_parquet(sl.trump_labels_path(O5), index=False)
    client = _stub_client(lambda cid, text: '{"score": 0.1, "label": "positive"}')
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    params = {"n": None, "seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
    st = sl.teacher_batch_submit("trump", params=params)
    assert st["n_submitted"] == 5 and "t_0" not in st["ids"]
    _ended(client)
    st, rep = sl.teacher_batch_collect("trump")
    assert st["status"] == "collected" and rep["n"] == 6


def test_batch_collect_records_stop_reasons(posts, monkeypatch):
    posts()
    client = _stub_client(lambda cid, text: '{"score": 0.1, "label": "positive"}')
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    params = {"n": None, "seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
    sl.teacher_batch_submit("trump", params=params)
    _ended(client)
    batches = client.messages.batches
    orig = batches.results

    def results(bid):
        for r in orig(bid):
            r.result.message.stop_reason = "max_tokens" if r.custom_id == "t_0" else "end_turn"
            yield r

    batches.results = results
    st, _ = sl.teacher_batch_collect("trump")
    assert st["spend"]["stop_reasons"] == {"max_tokens": 1, "end_turn": 5}
    assert st["spend"]["output_tokens_per_call"] == 50
    kept = sl.read_batch_results(sl.batch_archive_path("trump", st["batch_id"], "_results.jsonl"))
    assert {r.result.message.stop_reason for r in kept} == {"max_tokens", "end_turn"}


# ── topic labels: unjudgeable posts ────────────────────────────────


def _topic_file(path, lines):
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


NO_VERDICT = {
    "type": "succeeded",
    "message": {"content": [{"type": "text", "text": "I can't open links, so I"}]},
}


def test_unjudgeable_needs_repeated_no_verdicts(posts, monkeypatch, tmp_path):
    from src.cli import main

    x = posts()
    lab = pd.read_parquet(tl.labels_path())
    lab = lab[lab["id"] != "x_admin_2"]  # unlabelled, like a link share
    lab.to_parquet(tl.labels_path(), index=False)
    assert "x_admin_2" in set(tl.posts_to_label()["id"])
    assert settings.TOPIC_UNJUDGEABLE_MIN_ATTEMPTS == 3
    lines = [
        {"custom_id": "x_admin_2", "result": NO_VERDICT},
        {"custom_id": "x_admin_0", "result": {"type": "errored"}},
    ]
    f1 = _topic_file(tmp_path / "topic_results_1.jsonl", lines)
    res = CliRunner().invoke(main, ["topic-label", "unjudgeable", "--results-file", str(f1)])
    assert res.exit_code == 0 and "1 no-verdict answers recorded" in res.output, res.output
    u = tl.load_unjudgeable()
    assert u["id"].tolist() == ["x_admin_2"]  # an errored request is retried, not recorded
    assert u["input_chars"].iloc[0] == len(x.set_index("id").loc["x_admin_2", "text"])
    # One no-verdict is not enough: about a third get a verdict when sent again.
    assert "x_admin_2" in set(tl.posts_to_label()["id"])
    # The same file twice is one attempt, not two.
    assert tl.unjudgeable_from_file(f1)["recorded"] == 0
    assert tl.unjudgeable_from_file(_topic_file(tmp_path / "topic_results_2.jsonl", lines)) == {
        "recorded": 1,
        "changed": 0,
        "unjudgeable": 0,
    }
    assert "x_admin_2" in set(tl.posts_to_label()["id"])
    # A batch's own download is keyed by its batch id.
    f3 = _topic_file(tmp_path / "topic_batch_msgbatch_9_results.jsonl", lines)
    assert tl.unjudgeable_from_file(f3)["unjudgeable"] == 1
    assert "msgbatch_9" in set(tl.load_unjudgeable()["source"])
    assert "x_admin_2" not in set(tl.posts_to_label()["id"])
    assert "x_admin_2" in set(tl.posts_to_label(retry_unjudgeable=True)["id"])
    res = CliRunner().invoke(main, ["topic-label", "estimate"])
    assert "1 posts recorded as unjudgeable are left out" in res.output
    # A text that changes since (a backfill) is sent again: its attempts were on the old one.
    x.loc[x["id"] == "x_admin_2", "text"] = "Lunch with friends admin 2, and much more text"
    x.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    assert "x_admin_2" in set(tl.posts_to_label()["id"])
    # The war flag still counts it by the keyword pattern (no Haiku label).
    from src.analysis.inference import war_flag

    d = pl.post_frame()
    assert not war_flag(d[d["id"] == "x_admin_2"], "llm").iloc[0]


def test_topic_collect_records_no_verdict_on_the_text_sent(posts, monkeypatch):
    x = posts()
    ln = dict(zip(x["id"], x["text"].str.len()))
    lab = pd.read_parquet(tl.labels_path())
    gone = ["x_media_3", "x_media_2", "x_admin_3"]
    lab[~lab["id"].isin(gone)].to_parquet(tl.labels_path(), index=False)
    ns = types.SimpleNamespace

    def sorry(cid):
        return ns(
            custom_id=cid,
            result=ns(type="succeeded", message=ns(content=[ns(type="text", text="Sorry")])),
        )

    results = [
        sorry("x_media_3"),
        sorry("x_media_2"),  # its text changed after the submit: Haiku never saw this one
        ns(custom_id="x_admin_3", result=ns(type="errored", message=None)),
    ]
    tl.state_path().write_text(
        json.dumps(
            {
                "batch_id": "b1",
                "model": "claude-haiku-4-5-20251001",
                "status": "ended",
                "submitted_at": "2026-11-02T00:00:00+00:00",
                "input_chars": {
                    "x_media_3": ln["x_media_3"],
                    "x_media_2": ln["x_media_2"] - 5,
                    "x_admin_3": ln["x_admin_3"],
                },
            }
        )
    )
    monkeypatch.setattr(tl, "status", lambda: json.loads(tl.state_path().read_text()))
    monkeypatch.setattr(
        tl, "_client", lambda: ns(messages=ns(batches=ns(results=lambda b: iter(results))))
    )
    monkeypatch.setattr(settings, "TOPIC_UNJUDGEABLE_MIN_ATTEMPTS", 1)
    st = tl.collect()
    assert st["collected"] == {"labelled": 0, "failed": 3, "no_verdict": 2, "unjudgeable": 1}
    u = tl.load_unjudgeable()
    assert u["id"].tolist() == ["x_media_3"] and u["source"].tolist() == ["b1"]
    todo = set(tl.posts_to_label()["id"])
    assert "x_media_3" not in todo and {"x_media_2", "x_admin_3"} <= todo


# ── reply rater arm ────────────────────────────────────────────────


def test_reply_rater_job(monkeypatch):
    from src.analysis import reply_labels as rlab

    j = rlab.job("v2-rater")
    assert j.default_model() == settings.SECOND_RATER_MODEL == O55
    assert (
        j.prompt_version == sl.REPLY_TEACHER_V2_PROMPT_VERSION
        and j.prefix == sl.REPLY_TEACHER_V2_PREFIX
    )
    assert (
        sl.reply_labels_path(O55, "v2-rater").name
        == "teacher_labels_replies_v2-rater_claude-opus-5-5.parquet"
    )
    assert "v2 labels of record" in rlab.submit_problem(j, O5, None, None, 10)
    assert "unmeasured" in rlab.submit_problem(j, O55, None, None, 200)
    assert rlab.submit_problem(j, O55, None, 50, 50) is None
    monkeypatch.setitem(settings.REPLY_OUTPUT_TOKENS, O55, 120.0)
    assert rlab.submit_problem(j, O55, None, None, 1157) is None


def test_reply_rater_report_against_v2(env):  # noqa: F811
    from src.analysis import reply_labels as rlab

    env()
    lab = pd.read_parquet(sl.reply_labels_path(O5, "v2"))
    lab.assign(score_teacher=0.5, label_teacher="positive").to_parquet(
        sl.reply_labels_path(O55, "v2-rater"), index=False
    )
    est = rlab.job_estimate("v2-rater")
    assert est["model"] == O55 and est["todo"] == 0 and est["labelled"] == len(lab)
    rep = rlab.job_report("v2-rater")
    assert rep["model"] == O55 and rep["reference"] == O5 and rep["n"] == len(lab)
    assert rep["coverage"]["unlabelled"] == 0 and rep["coverage"]["with_text"] == len(lab)
    assert any(s.startswith("coverage: ") for s in rlab.summary_lines(rep))
    assert rep["vs_v2"]["n"] == len(lab) and rep["vs_v2"]["mean_diff"] == pytest.approx(1.0)
    assert rep["vs_v2"]["sign_flips"] == len(lab)
    assert rep["by_post"][-1]["tracked_slug"] == "ALL"
    assert any("vs the v2 labels of claude-opus-5" in s for s in rlab.summary_lines(rep))
    assert rlab.report_path("v2-rater", O55).exists()


def test_pilot_and_report_cli(posts, monkeypatch):
    from src.cli import main

    posts()
    caller = Caller(out=80)
    monkeypatch.setattr(pl, "_direct_caller", lambda *a, **k: caller)
    res = CliRunner().invoke(
        main, ["post-label", "pilot", "--job", "stance-topic", "--n", "6"], input="n\n"
    )
    assert res.exit_code == 0 and "Aborted" in res.output and not caller.calls  # asks first
    res = CliRunner().invoke(
        main, ["post-label", "pilot", "--job", "stance-topic", "--n", "6", "--yes"]
    )
    assert res.exit_code == 0, res.output
    assert "pilot: 6 direct calls" in res.output and len(caller.calls) == 6
    assert "output 80.0 a call" in res.output and "the run, calibrated on the pilot" in res.output
    res = CliRunner().invoke(main, ["post-label", "pilot", "--job", "stance-topic", "--limit", "3"])
    assert res.exit_code == 2
    res = CliRunner().invoke(main, ["post-label", "report", "--job", "stance-topic"])
    assert res.exit_code == 0 and "stance-topic labels from claude-opus-5" in res.output
    res = CliRunner().invoke(main, ["post-label", "estimate", "--job", "rater", "--model", O5])
    assert res.exit_code == 1 and "second model" in res.output


# ── review fixes: run caps, gates, stale labels, column owners ─────


def test_post_caps_bound_the_whole_run_pilot_included(posts, monkeypatch):
    from src.cli import main

    posts()
    client = _stub_client(lambda cid, text: _answer(cid, text))
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    assert all(c.run_caps and c.extra_spend for c in pl.POST_BATCH_CHECKS.values())
    pl.run_pilot("stance-topic", n=4, caller=Caller(), concurrency=1)  # measured
    spent = pl.pilot_spend("stance-topic")
    assert spent["calls"] == 4 and spent["usd"] > 0
    assert pl.pilot_spend("stance-topic-nohandle") == {"calls": 0, "usd": 0.0}  # its own record
    est = pl.job_estimate("stance-topic")
    assert est["prior_calls"] == 4 and est["prior_usd"] == pytest.approx(spent["usd"])
    assert est["run_calls"] == 4 + est["todo"]
    assert est["run_usd_ceiling"] == pytest.approx(spent["usd"] + est["usd_batch_ceiling"])
    params = {"seed": settings.TEACHER_SAMPLE_SEED}
    kw = {"model": O5, "max_tokens": settings.POST_LABEL_MAX_TOKENS, "effort": "low"}
    st = sl.teacher_batch_submit("post_stance_topic", params=params, limit=12, **kw)
    assert st["n_submitted"] == 12
    sl.teacher_batch_collect("post_stance_topic", client=_ended(client))
    # The rest alone is under the cap; with what the run has cost it is over.
    rest = pl.job_estimate("stance-topic")
    assert rest["prior_calls"] == 16 and rest["prior_usd"] > spent["usd"]
    cap = rest["usd_batch_ceiling"] + rest["prior_usd"] / 2
    monkeypatch.setattr(settings, "POST_STANCE_TOPIC_MAX_USD", cap)
    with pytest.raises(RuntimeError, match="the run has cost so far.*bounds the whole run"):
        sl.teacher_batch_submit("post_stance_topic", params=params, **kw)
    assert len(client.messages.batches.created) == 1
    res = CliRunner().invoke(main, ["post-label", "submit", "--job", "stance-topic", "--yes"])
    assert res.exit_code == 1 and "with the run so far" in res.output, res.output
    assert "--limit" not in res.output.split("ceiling")[-1]  # no way round a run cap
    # Calls: the pilot's and the first batch's count against the call cap too.
    monkeypatch.setattr(settings, "POST_STANCE_TOPIC_MAX_USD", 30.0)
    monkeypatch.setattr(settings, "POST_STANCE_TOPIC_MAX_CALLS", 16 + 5)
    with pytest.raises(RuntimeError, match="16 calls already sent by the run"):
        sl.teacher_batch_submit("post_stance_topic", params=params, limit=6, **kw)
    # A pilot that would pass a run cap is refused before any call.
    monkeypatch.setattr(settings, "POST_STANCE_TOPIC_MAX_CALLS", 16 + 3)
    caller = Caller()
    with pytest.raises(ValueError, match="bounds the whole run"):
        pl.run_pilot("stance-topic", n=4, caller=caller, concurrency=1)
    assert not caller.calls


def test_a_max_tokens_stop_refuses_every_batch_size(posts, monkeypatch):
    from src.cli import main

    posts()
    pl.run_pilot("rater", n=4, caller=Caller(stop="max_tokens"), concurrency=1)
    rater = pl.JOBS["rater"]
    for limit, n in ((None, 5000), (5000, 5000), (30, 23), (5, 5)):
        assert "max_tokens" in (pl.submit_problem(rater, O55, limit, n) or ""), (limit, n)
    res = CliRunner().invoke(main, ["post-label", "submit", "--job", "rater", "--limit", "30"])
    assert res.exit_code == 1 and "POST_LABEL_MAX_TOKENS" in res.output
    monkeypatch.setattr(settings, "POST_LABEL_MAX_TOKENS", 2048)  # a new request: pilot again
    assert pl.submit_problem(rater, O55, 20, 20) is None


def test_a_pilot_of_another_prompt_or_effort_measures_nothing(posts, monkeypatch):
    posts()
    pl.run_pilot("stance-topic", n=3, caller=Caller(out=90), concurrency=1)
    j = pl.JOBS["stance-topic"]
    assert pl.output_basis(j, O5)["measured"]
    rec = pl.read_pilot("stance-topic", O5)
    for key, other in (("prompt_version", "post-stance-topic-OLD"), ("effort", "medium")):
        changed = {**rec, "runs": [{**r, key: other} for r in rec["runs"]]}
        sl._write_json(pl.pilot_path("stance-topic", O5), changed)
        assert pl.pilot_summary("stance-topic", O5) is None, key
        assert not pl.output_basis(j, O5)["measured"], key
    # The spend of every pilot still counts against the run caps.
    assert pl.pilot_spend("stance-topic")["calls"] == 3


def test_unmeasured_batches_count_together_against_the_pilot_size(posts, monkeypatch):
    posts()
    client = _stub_client(lambda cid, text: _answer(cid, text))
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    monkeypatch.setattr(settings, "POST_PILOT_MAX_CALLS", 20)
    params = {"seed": settings.TEACHER_SAMPLE_SEED}
    kw = {"model": O55, "max_tokens": settings.POST_LABEL_MAX_TOKENS, "effort": "low"}
    st = sl.teacher_batch_submit("post_rater", params=params, limit=12, **kw)
    assert st["n_submitted"] == 12
    sl.teacher_batch_collect("post_rater", client=_ended(client))
    assert pl.unmeasured_batch_calls(pl.JOBS["rater"], O55) == 12
    with pytest.raises(RuntimeError, match="12 sent by earlier batches"):
        sl.teacher_batch_submit("post_rater", params=params, limit=9, **kw)
    st = sl.teacher_batch_submit("post_rater", params=params, limit=8, **kw)  # 20 in all
    assert st["n_submitted"] == 8


def test_column_owner_never_claims_score_opus(posts):
    from src.cli import main

    assert pl.column_owner("score_opus") is None and pl.column_owner("label_opus") is None
    assert pl.column_owner("score_haiku") is None
    assert pl.column_owner("score_opus55") == (pl.JOBS["rater"], O55)
    assert pl.column_owner("topic_opus") == (pl.JOBS["stance-topic"], O5)
    posts()
    res = CliRunner().invoke(main, ["phases", "--source", "trump", "--score", "score_opus"])
    assert res.exit_code == 1 and "--merge" in res.output and "post-label submit" not in res.output
    df = pd.read_parquet(settings.SENTIMENT_OUTPUT).drop(columns=["score_opus", "label_opus"])
    df.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
    res = CliRunner().invoke(main, ["phases", "--n-boot", "5"])
    assert res.exit_code == 1 and "relabel merge" in res.output
    assert "post-label submit" not in res.output


def test_phases_on_a_post_label_column_reports_war_posts_only(posts):
    from src.cli import main

    x = posts()
    ln = dict(zip(x["id"], x["text"].str.len()))
    cand = [f"x_{t}_{i}" for t in TIERS for i in (0, 1, 4, 5)]
    _label_file(
        "rater",
        O55,
        [
            {"id": i, "input_chars": ln[i], "score_teacher": 0.3, "label_teacher": "positive"}
            for i in cand
        ],
    )
    run = CliRunner().invoke(main, ["phases", "--score", "score_opus55", "--n-boot", "10"])
    assert run.exit_code == 0, run.output
    assert "left blank" in run.output and "share effect" not in run.output
    stats = pd.read_csv(settings.PROCESSED_DIR / "phases_x_tier_score_opus55_either.csv")
    assert stats[["share", "mean_all", "mean_all_lo", "mean_off"]].isna().all().all()
    assert stats["mean_on"].notna().all() and (stats["mean_on"] == 0.3).all()
    con = pd.read_csv(settings.PROCESSED_DIR / "phase_contrasts_x_tier_score_opus55_either.csv")
    assert con[["all", "share_effect", "on_topic_effect", "off_topic_effect"]].isna().all().all()
    assert con["on_topic_change"].notna().all()
    # score_opus keeps every column.
    run = CliRunner().invoke(main, ["phases", "--n-boot", "10"])
    assert run.exit_code == 0 and "share effect" in run.output
    full = pd.read_csv(settings.PROCESSED_DIR / "phases_x_tier_score_opus_either.csv")
    assert full["share"].notna().all()


def test_nohandle_all_row_weights_tiers_by_their_pool(posts):
    x = posts()
    ln = dict(zip(x["id"], x["text"].str.len()))
    rel = ["x_religious_authority_0", "x_religious_authority_1"]
    adm = ["x_admin_0", "x_admin_1"]
    pd.DataFrame(
        {
            "id": rel + adm,
            "tier": ["religious_authority"] * 2 + ["admin"] * 2,
            "pool": [2, 2, 100, 100],
        }
    ).to_parquet(pl.subsample_manifest_path(), index=False)

    def rows(scores):
        return [
            {
                "id": i,
                "input_chars": ln[i],
                "topic": "iran_war",
                "score_teacher": s,
                "label_teacher": "positive",
                "post_set": "war",
            }
            for i, s in scores.items()
        ]

    _label_file("stance-topic", O5, rows({**dict.fromkeys(rel, 0.6), **dict.fromkeys(adm, 0.5)}))
    _label_file(
        "stance-topic-nohandle", O5, rows({**dict.fromkeys(rel, 0.2), **dict.fromkeys(adm, 0.5)})
    )
    rep = pl.job_report("stance-topic-nohandle")
    by = {r["tier"]: r for r in rep["paired_by_tier"]}
    assert by["religious_authority"]["diff"] == pytest.approx(-0.4)
    assert by["admin"]["diff"] == pytest.approx(0.0)
    # Each religious pair stands for 1 post, each admin pair for 50: not -0.2 unweighted.
    assert by["ALL"]["weighted"] and by["ALL"]["pairs"] == 4
    assert by["ALL"]["diff"] == pytest.approx((2 * -0.4) / (2 + 100))
    assert by["ALL"]["mean_handle"] == pytest.approx((2 * 0.6 + 100 * 0.5) / 102)
    assert "pool" in rep["all_weights"]
    assert any("ALL weighted" in s for s in pl.summary_lines(rep))


def test_a_label_on_a_changed_text_is_redone_and_archived(posts, monkeypatch, caplog):
    x = posts()
    ln = dict(zip(x["id"], x["text"].str.len()))
    _label_file(
        "stance-topic",
        O5,
        [
            {
                "id": "x_admin_0",
                "input_chars": ln["x_admin_0"] - 3,  # made from an older text
                "topic": "not_war",
                "score_teacher": np.nan,
                "label_teacher": None,
                "post_set": "war",
                "source": "x",
                "group": "admin",
            },
            {
                "id": "x_admin_4",
                "input_chars": ln["x_admin_4"],
                "topic": "not_war",
                "score_teacher": np.nan,
                "label_teacher": None,
                "post_set": "war",
                "source": "x",
                "group": "admin",
            },
        ],
    )
    j = pl.JOBS["stance-topic"]
    _, _, todo, _ = pl.job_todo(j, O5, {"seed": 1}, preview=True)
    assert "x_admin_0" in set(todo["id"]) and "x_admin_4" not in set(todo["id"])
    assert pl.job_estimate("stance-topic")["labelled"] == 1
    # --topic opus: the stale label is dropped, so the candidate counts as either does.
    df = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    with caplog.at_level("WARNING"):
        flag = pl.opus_war_flag(df, pd.Series(True, index=df.index)).set_axis(df["id"])
    assert flag["x_admin_0"] and not flag["x_admin_4"]
    # A report reads only the current label.
    assert pl._labelled(j, O5)["id"].tolist() == ["x_admin_4"]
    # Labelled again by a pilot: one row per id, the old one in the archive.
    pl.run_pilot("stance-topic", n=40, caller=Caller(), concurrency=1)
    lab = pd.read_parquet(pl.labels_path("stance-topic", O5)).set_index("id")
    assert lab.index.is_unique and lab.loc["x_admin_0", "input_chars"] == ln["x_admin_0"]
    arch = pd.read_parquet(
        pl.labels_path("stance-topic", O5).with_name(
            "post_labels_stance-topic_claude-opus-5_superseded.parquet"
        )
    )
    assert (
        arch["id"].tolist() == ["x_admin_0"] and arch["input_chars"].iloc[0] == ln["x_admin_0"] - 3
    )


def test_a_stale_trump_label_is_redone_by_the_batch(posts, monkeypatch):
    posts()
    feed = _trump_rows()
    pd.DataFrame(
        {
            "id": ["t_0", "t_1"],
            "input_chars": [len(feed.loc[0, "text"]), 3],  # t_1's text has changed since
            "prompt_version": "llm_prompt",
            "score_teacher": [0.4, -0.3],
            "label_teacher": ["positive", "negative"],
        }
    ).to_parquet(sl.trump_labels_path(O5), index=False)
    client = _stub_client(lambda cid, text: '{"score": 0.1, "label": "positive"}')
    monkeypatch.setattr(sl, "_batch_client", lambda: client)
    params = {"n": None, "seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
    st = sl.teacher_batch_submit("trump", params=params)
    assert st["n_submitted"] == 5 and "t_1" in st["ids"] and "t_0" not in st["ids"]
    st, _ = sl.teacher_batch_collect("trump", client=_ended(client))
    assert st["collected"]["labelled"] == 5 and not st["collected"]["ignored"]["already_cached"]
    lab = pd.read_parquet(sl.trump_labels_path(O5)).set_index("id")
    assert lab.index.is_unique and lab.loc["t_1", "score_teacher"] == 0.1
    arch = pd.read_parquet(
        sl.trump_labels_path(O5).with_name("teacher_check_trump_claude-opus-5_superseded.parquet")
    )
    assert arch["id"].tolist() == ["t_1"] and arch["score_teacher"].iloc[0] == -0.3


def test_text_changes_move_post_label_caches_aside(posts):
    from src.collectors import ts_fill_text as tf
    from src.collectors import x_backfill as xb

    posts()
    _label_file("rater", O55, [{"id": "x_admin_0", "input_chars": 1, "score_teacher": 0.1}])
    path = pl.labels_path("rater", O55)
    assert path in xb.post_label_files() and path in tf.label_files()
    assert xb.supersede_labels({"x_admin_0"})[path.name] == 1
    assert pd.read_parquet(path).empty


def test_merge_waits_for_score_posts_which_keeps_score_opus(posts, monkeypatch):
    import os
    import subprocess

    from src.cli import main

    posts()
    pd.DataFrame(
        {
            "id": ["t_0"],
            "input_chars": [len(_trump_rows().loc[0, "text"])],
            "prompt_version": "llm_prompt",
            "score_teacher": [0.4],
            "label_teacher": ["positive"],
        }
    ).to_parquet(sl.trump_labels_path(O5), index=False)
    marker = sl.scoring_marker(settings.TRUMP_FEED_STANCE)
    marker.write_text(json.dumps({"pid": os.getpid(), "started_at": "2026-11-02T00:00:00+00:00"}))
    res = CliRunner().invoke(main, ["teacher-check", "--source", "trump", "--merge"])
    assert res.exit_code == 1 and "score-posts" in res.output
    assert "score_opus" not in pd.read_parquet(settings.TRUMP_FEED_STANCE)
    gone = subprocess.Popen(["true"])
    gone.wait()
    marker.write_text(json.dumps({"pid": gone.pid}))  # a killed run's marker: ignored
    sl.trump_merge(O5)
    marker.unlink()
    # score-posts keeps score_opus on the rows it is on; a new post has none until a merge.
    raw = settings.PROCESSED_DIR / "feed.jsonl"
    rows = _trump_rows().to_dict("records") + [
        {"id": "t_new", "user": "realDonaldTrump", "text": "A new post", "created_at": "2026-06-01"}
    ]
    raw.write_text(
        "".join(
            json.dumps({k: v for k, v in r.items() if k != "score_opus_distilled"}) + "\n"
            for r in rows
        )
    )
    seen = []

    def score(texts, md, bs, ml):
        seen.append(marker.exists())
        return np.zeros(len(texts))

    monkeypatch.setattr(sl, "score_with_distilled", score)
    monkeypatch.setattr(settings, "MODELS_DIR", settings.PROCESSED_DIR)
    out = sl.score_post_file([raw], settings.TRUMP_FEED_STANCE).set_index("id")
    assert seen == [True] and not marker.exists()  # the marker lives while it scores
    assert out.loc["t_0", "score_opus"] == 0.4 and np.isnan(out.loc["t_new", "score_opus"])


# ── second-look fixes ──────────────────────────────────────────────


def test_trump_merge_without_labels_says_so(posts, monkeypatch):
    from src.cli import main

    posts()
    _no_api(monkeypatch)
    before = pd.read_parquet(settings.TRUMP_FEED_STANCE)
    with pytest.raises(FileNotFoundError, match="no Trump labels from claude-haiku-4-5"):
        sl.trump_merge("claude-haiku-4-5")
    res = CliRunner().invoke(
        main, ["teacher-check", "--source", "trump", "--merge", "--model", "claude-haiku-4-5"]
    )
    assert res.exit_code == 1 and "--all --model claude-haiku-4-5" in res.output, res.output
    assert "Traceback" not in res.output and not sl.trump_columns_path().exists()
    pd.testing.assert_frame_equal(pd.read_parquet(settings.TRUMP_FEED_STANCE), before)


def test_a_small_pilot_does_not_count_as_the_measurement(posts, monkeypatch):
    """A pilot of a few calls would set the ceiling (what the dollar cap is
    checked against) at a p95 of a handful: under POST_PILOT_MIN_BILLED it
    stays a guess, the unmeasured gate holds, and its stop reasons count."""
    posts()
    monkeypatch.setattr(settings, "POST_PILOT_MIN_BILLED", 5)
    guess = pl.output_basis(pl.JOBS["rater"], O55)
    pl.run_pilot("rater", O55, n=3, caller=Caller(out=60))
    basis = pl.output_basis(pl.JOBS["rater"], O55)
    assert not basis["measured"] and basis["pilot"]["billed"] == 3
    assert "under the 5 a measurement needs" in basis["basis"]
    assert basis["ceiling"] == guess["ceiling"] and basis["mean"] == guess["mean"]
    assert not pl.job_estimate("rater", O55)["output_measured"]
    big = settings.POST_PILOT_MAX_CALLS + 1
    assert "pilot --job rater" in pl.submit_problem(pl.JOBS["rater"], O55, None, big)
    pl.run_pilot("rater", O55, n=2, caller=Caller(out=60))
    basis = pl.output_basis(pl.JOBS["rater"], O55)  # pooled: 5 billed
    assert basis["measured"] and basis["ceiling"]["x"] == 60
    assert pl.submit_problem(pl.JOBS["rater"], O55, None, big) is None
    # A small pilot's cache tokens do not stand for the run either, but its
    # max_tokens stop still refuses every batch.
    pl.run_pilot("stance-topic", n=2, caller=Caller(stop="max_tokens"), concurrency=1)
    assert not pl.output_basis(pl.JOBS["stance-topic"], O5)["measured"]
    assert pl.cache_basis(pl.JOBS["stance-topic"], O5)["basis"].startswith("estimated")
    assert "stopped at max_tokens" in pl.submit_problem(pl.JOBS["stance-topic"], O5, None, 5)


def test_a_batch_cut_at_max_tokens_refuses_the_next_submit(posts):
    """Answers cut at max_tokens in a collected batch are not kept, so the
    next submit would send them again at the same max_tokens."""
    posts()
    j = pl.JOBS["stance-topic"]
    pl.run_pilot("stance-topic", n=3, caller=Caller(), concurrency=1)  # measured, no stop
    assert pl.submit_problem(j, O5, None, 5) is None
    st = {
        "check": j.check,
        "batch_id": "msgbatch_9",
        "status": "collected",
        "model": O5,
        "prompt_version": j.prompt_version,
        "max_tokens": settings.POST_LABEL_MAX_TOKENS,
        "effort": settings.POST_LABEL_EFFORT,
        "n_submitted": 10,
        "spend": {"usd_batch": 0.01, "stop_reasons": {"end_turn": 7, "max_tokens": 3}},
    }
    sl._write_json(sl.batch_archive_path(j.check, "msgbatch_9"), st)
    why = pl.submit_problem(j, O5, None, 5)
    assert why and "3 answers in earlier stance-topic batches" in why
    assert "raise POST_LABEL_MAX_TOKENS" in why
    st["max_tokens"] = settings.POST_LABEL_MAX_TOKENS // 2  # another request: does not count
    sl._write_json(sl.batch_archive_path(j.check, "msgbatch_9"), st)
    assert pl.submit_problem(j, O5, None, 5) is None


def test_a_pilot_is_on_record_before_its_first_call(posts, monkeypatch):
    """A pilot killed outright (no finally: dkbl1 powers off) must keep the
    labels it paid for and still count against the job's caps: its run is
    recorded in_progress at its direct ceiling before the first call, and
    labels and usage are saved every POST_PILOT_SAVE_EVERY answers."""
    posts()
    monkeypatch.setattr(settings, "POST_PILOT_SAVE_EVERY", 4)
    pilot_file = pl.pilot_path("stance-topic", O5)
    writes, saved = [], []
    real_write, real_merge = sl._write_json, sl.merge_new_labels

    def spy_write(path, obj):
        if path == pilot_file:
            writes.append(json.loads(json.dumps(obj)))
        real_write(path, obj)

    def spy_merge(cached, new, out):
        saved.append(len(new))
        return real_merge(cached, new, out)

    monkeypatch.setattr(sl, "_write_json", spy_write)
    monkeypatch.setattr(sl, "merge_new_labels", spy_merge)
    ceiling = pl.job_estimate("stance-topic", limit=10)["usd_direct_ceiling"]
    pl.run_pilot("stance-topic", n=10, caller=Caller(), concurrency=1)
    first = writes[0]["runs"][-1]
    assert first["in_progress"] and first["calls"] == 10 and first["output_tokens"] == []
    assert first["usd_direct_ceiling"] == pytest.approx(ceiling)
    mid = [w["runs"][-1] for w in writes[1:-1]]
    assert [len(r["output_tokens"]) for r in mid] == [4, 8] and all(r["in_progress"] for r in mid)
    assert saved[:2] == [4, 8]  # labels saved at each checkpoint
    last = writes[-1]["runs"]
    assert len(last) == 1 and not last[0].get("in_progress") and len(last[0]["output_tokens"]) == 10
    # A run left in_progress (killed) counts at its ceiling and measures nothing.
    real_write(pilot_file, writes[1])
    spent = pl.pilot_spend("stance-topic")
    assert spent["calls"] == 10 and spent["usd"] == pytest.approx(ceiling)
    assert pl.pilot_summary("stance-topic", O5) is None


def test_reports_before_any_label(posts):
    from src.cli import main

    posts()
    for job in pl.JOBS:
        assert pl.job_report(job)["n"] == 0
        res = CliRunner().invoke(main, ["post-label", "report", "--job", job])
        assert res.exit_code == 0 and ": 0" in res.output, res.output


def test_nohandle_all_interval_is_the_ratios_linearized_one():
    """The ALL shift is a ratio over the pairs both arms call iran_war, a
    random subset: its interval counts the subset's size varying by tier
    (e = 1{both} (d - R), weighted, stratified by tier), not each tier's
    variance of d alone."""
    d_a, d_b = [0.1, 0.3, 0.2], [-0.4, -0.2, -0.3]
    g = pd.DataFrame(
        {
            "group": ["a"] * 4 + ["b"] * 3,
            "topic_handle": ["iran_war"] * 7,
            "topic_nohandle": ["iran_war"] * 3 + ["not_war"] + ["iran_war"] * 3,
            "score_teacher_handle": [0.0] * 7,
            "score_teacher_nohandle": [*d_a, 0.0, *d_b],
        }
    )
    w = pd.Series([10.0] * 4 + [2.0] * 3, index=g.index)  # pool 40 / 4 pairs, 6 / 3
    row = pl._paired_row(g, w)
    r = (10 * sum(d_a) + 2 * sum(d_b)) / (10 * 3 + 2 * 3)
    z_a = np.array([10 * (x - r) for x in d_a] + [0.0])
    z_b = np.array([2 * (x - r) for x in d_b])
    var = (4 * z_a.var(ddof=1) + 3 * z_b.var(ddof=1)) / 36**2
    assert row["diff"] == pytest.approx(r)
    assert row["diff_hi"] - row["diff"] == pytest.approx(1.96 * np.sqrt(var))
    # With every pair iran_war it is the stratified mean's interval.
    g2 = g.assign(topic_nohandle="iran_war")
    row2 = pl._paired_row(g2, w)
    sh = {"a": 40 / 46, "b": 6 / 46}
    v_a = np.var([*d_a, 0.0], ddof=1)
    v2 = sh["a"] ** 2 * v_a / 4 + sh["b"] ** 2 * np.var(d_b, ddof=1) / 3
    assert row2["diff_hi"] - row2["diff"] == pytest.approx(1.96 * np.sqrt(v2))
