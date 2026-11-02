"""The reply audience's loyalty gradient (`reply-loyalty`,
inference.reply_loyalty): tiers on a fixed reference date, design-weighted
tier means and high - low per post, pooled and within posts. Stub data,
no API."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from config import settings
from src.analysis import event_study as es
from src.analysis import inference as inf
from src.analysis import stance_local as sl
from tests.test_reply_labels import MODEL, _replies, env  # noqa: F401 -- the fixture

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

HIGH = {"id": "a_high", "bio": "MAGA patriot, Trump 2028, conservative", "created_at": "2022-02-21"}
MEDIUM = {"id": "a_mid", "bio": "maga", "created_at": "2025-10-01T00:00:00Z"}
LOW = {"id": "a_low", "bio": "", "created_at": "2026-03-01T00:00:00Z"}


def test_loyalty_score_takes_the_age_on_the_reference_date():
    ref = datetime(2026, 11, 1, tzinfo=UTC)
    assert es.loyalty_tier(es.loyalty_score({"account": HIGH}, ref)) == "high"
    # 0.2 for the bio + 0.1 for 13 months: medium on Nov 1, low on Jun 1.
    assert es.loyalty_tier(es.loyalty_score({"account": MEDIUM}, ref)) == "medium"
    early = datetime(2026, 6, 1, tzinfo=UTC)
    assert es.loyalty_tier(es.loyalty_score({"account": MEDIUM}, early)) == "low"
    assert es.loyalty_tier(es.loyalty_score({"account": LOW}, ref)) == "low"
    # No reference date: the moment of the call, as event-study has it.
    now = es.loyalty_score({"account": MEDIUM})
    assert now >= es.loyalty_score({"account": MEDIUM}, ref)


def test_replier_tiers_default_to_the_settings_date(monkeypatch):
    reps = pd.DataFrame({"id": ["1", "2", "3", "4"], "account": [HIGH, MEDIUM, LOW, None]})
    t = inf.replier_tiers(reps)
    assert t["loyalty"].tolist() == ["high", "medium", "low", "unknown"]
    assert t["replier"].tolist()[:3] == ["a_high", "a_mid", "a_low"]
    monkeypatch.setattr(settings, "LOYALTY_REFERENCE_DATE", datetime(2026, 6, 1, tzinfo=UTC))
    assert inf.replier_tiers(reps)["loyalty"].tolist()[1] == "low"


def _with_accounts(reps: pd.DataFrame) -> pd.DataFrame:
    """Each reply's account by its index: high, medium, low in turn (one
    account each per post, so a post has three repliers)."""
    acct = []
    for slug, i in zip(reps["tracked_slug"], reps["id"].str.rsplit("-", n=1).str[1].astype(int)):
        a = (HIGH, MEDIUM, LOW)[i % 3]
        acct.append({**a, "id": f"{a['id']}_{slug}"})
    return reps.assign(account=acct)


def _tier_labels(reps: pd.DataFrame, drawn: pd.DataFrame, by_tier: dict) -> None:
    """v2 labels on the drawn replies: by_tier[tier of the reply]."""
    tiers = inf.replier_tiers(reps).set_index("id")["loyalty"]
    pd.DataFrame(
        {
            "id": drawn["id"],
            "tracked_slug": drawn["tracked_slug"],
            "input_chars": drawn["text"].str.len(),
            "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
            "score_teacher": drawn["id"].map(tiers).map(by_tier),
            "label_teacher": "negative",
        }
    ).to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)


def test_reply_loyalty_gradient_per_post_pooled_and_within(env):  # noqa: F811
    reps = _with_accounts(_replies(n_per_post=60))
    reps, drawn = env(reps, n_per_bucket=6)
    _tier_labels(reps, drawn, {"high": 0.5, "medium": 0.0, "low": -0.5})
    rows, n = inf.reply_loyalty(n_boot=50)
    by = rows.set_index("post")
    posts = set(reps["tracked_slug"])
    assert set(by.index) == posts | {"ALL", "within posts"}
    # Constant labels within a tier: every tier mean is exact, high - low is 1.
    for p in [*posts, "ALL"]:
        assert by.loc[p, "mean_high"] == pytest.approx(0.5)
        assert by.loc[p, "mean_medium"] == pytest.approx(0.0)
        assert by.loc[p, "high_minus_low"] == pytest.approx(1.0)
    assert by.loc["within posts", "high_minus_low"] == pytest.approx(1.0)
    assert np.isnan(by.loc["within posts", "mean_high"])
    # Counts: replies with text, distinct repliers and labelled draws per tier.
    assert n["replies_high"] == 20 * len(posts) and n["repliers_high"] == len(posts)
    assert n["repliers"] == 3 * len(posts) and n["n_replies"] == 60 * len(posts)
    assert sum(n[f"labelled_{t}"] for t in inf.LOYALTY_TIERS) == n["n_labelled"] == len(drawn)
    assert n["replies_unknown"] == 0 and n["as_of"].startswith("2026-11-01")
    one = by.loc[next(iter(posts))]
    assert one["repliers_low"] == 1 and one["replies_low"] == 20


def test_reply_loyalty_weights_the_draws_back_to_the_population(env):  # noqa: F811
    """A tier's mean is its estimated population mean, not the draws' mean:
    the low tier's labels differ by valence bucket, and its draws
    over-represent the small buckets (6 of each)."""
    reps = _with_accounts(_replies(n_per_post=60))
    reps, drawn = env(reps, n_per_bucket=6)
    tiers = inf.replier_tiers(reps).set_index("id")["loyalty"]
    bucket = drawn.set_index("id")["score_transformer"].map(lambda s: "crit" if s < 0 else "rest")
    score = [
        (-1.0 if bucket[i] == "crit" else 0.0) if tiers[i] == "low" else 0.2 for i in drawn["id"]
    ]
    pd.DataFrame(
        {
            "id": drawn["id"],
            "tracked_slug": drawn["tracked_slug"],
            "input_chars": drawn["text"].str.len(),
            "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
            "score_teacher": score,
            "label_teacher": "negative",
        }
    ).to_parquet(sl.reply_labels_path(MODEL, "v2"), index=False)
    rows, _ = inf.reply_loyalty(n_boot=50)
    by = rows.set_index("post")
    # Half of each post's replies are critical: the low tier's population
    # mean is about -0.5, where its draws (a third critical) average -1/3.
    low = by.loc["ALL", "mean_low"]
    pop = reps.assign(t=reps["id"].map(tiers), crit=reps["score_transformer"] < 0)
    want = -pop.loc[pop["t"] == "low", "crit"].mean()
    drawn_low = [s for s, i in zip(score, drawn["id"]) if tiers[i] == "low"]
    assert abs(low - want) < abs(np.mean(drawn_low) - want)  # weighted, not the draws' mean
    assert by.loc["ALL", "mean_low_lo"] <= want <= by.loc["ALL", "mean_low_hi"]


def test_reply_loyalty_cli_is_free_and_writes_its_csv(env, monkeypatch):  # noqa: F811
    from src.cli import main

    reps = _with_accounts(_replies(n_per_post=60))
    reps, drawn = env(reps, n_per_bucket=6)
    _tier_labels(reps, drawn, {"high": 0.5, "medium": 0.0, "low": -0.5})
    import anthropic

    def boom(*a, **k):
        raise AssertionError("an API call")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    res = CliRunner().invoke(main, ["reply-loyalty", "--n-boot", "20"])
    assert res.exit_code == 0, res.output
    assert "loyalty tiers with account age on 2026-11-01" in res.output
    assert "repliers" in res.output and "within posts" in res.output
    assert "+1.000 [+1.00, +1.00]" in res.output
    assert (settings.PROCESSED_DIR / "reply_loyalty_v2.csv").exists()
    res = CliRunner().invoke(main, ["reply-loyalty", "--first-draws", "--n-boot", "5"])
    assert res.exit_code == 0 and "(first draws)" in res.output
    assert (settings.PROCESSED_DIR / "reply_loyalty_v2_first_draws.csv").exists()


def test_reply_loyalty_counts_repliers_in_two_tiers(env):  # noqa: F811
    """A tier is per reply, from that reply's account snapshot: a replier
    whose bio differs between replies sits in two tiers, and the count says
    so, so the tier counts add up to the repliers plus that count."""
    from src.cli import main

    reps = _with_accounts(_replies(n_per_post=60))
    post = reps["tracked_slug"].iloc[0]
    low_id = f"{LOW['id']}_{post}"
    first_low = reps.index[reps["account"].map(lambda a: a["id"]) == low_id][0]
    reps.at[first_low, "account"] = {**HIGH, "id": low_id}  # one later snapshot, a MAGA bio
    reps, drawn = env(reps, n_per_bucket=6)
    _tier_labels(reps, drawn, {"high": 0.5, "medium": 0.0, "low": -0.5})
    _, n = inf.reply_loyalty(n_boot=20)
    assert n["repliers_multi_tier"] == 1
    assert sum(n[f"repliers_{t}"] for t in inf.LOYALTY_TIERS) == n["repliers"] + 1
    res = CliRunner().invoke(main, ["reply-loyalty", "--n-boot", "5"])
    assert res.exit_code == 0, res.output
    assert "1 repliers fall in more than one" in res.output
