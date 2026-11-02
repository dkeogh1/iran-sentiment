"""
Opus labels on Truth Social replies beyond the v2 run: one path whose
parameters are the prompt version, the model and the id source, on the v2
run's prompt code, label cache and batch transport (stance_local).

  v2b       v2 without its praise rule (the sentence that makes praise of a
            ceasefire, a deal or a hold-off a position against the war, and
            praise of a strike one for it), on the stance sample's replies
            with text: does the June deal's anti-war reading rest on it?
  cat-v2    the six reply categories of event_study.STANCE_PROMPT, asked v2's
            way: the parent post as the cached prefix, the full reply, no
            author handle; on the stance sample's replies with text. The
            Haiku categories in stance_sample.parquet stay as history.
  v2-fresh  the v2 prompt unchanged on a second random sample of replies
            (inference.draw_fresh_sample, frozen before anything is sent):
            the same scorer as the v2 labels, so reply-population pools
            them (stance_local.REPLY_LABEL_SETS), in a file of their own.
  v2-rater  the v2 prompt unchanged on a second model
            (settings.SECOND_RATER_MODEL, claude-opus-5-5) on the stance
            sample's replies with text: the reply arm of the second rater
            (post_labels "rater"). Never the v2 labels' own model, and never
            pooled with them: its report sets it against them.

Each job has its own label file (stance_local.reply_labels_path: the model
id in its name, so another model is another file), batch state file, caps
and report; nothing here writes the v1 or v2 files. Labels are made through
the Message Batches API only (`reply-label submit|status|collect`); v2-fresh
goes one batch per post (--post); its caps, and v2-rater's, bound the whole
run (every batch it has sent plus the next). The estimate makes no call and is
calibrated on the usage the 2026-10-05 v2 batch was billed
(settings.REPLY_V2_BATCH_USAGE); a request carries a cache breakpoint only
where the post's prefix reaches the model's cache minimum, so the ceiling
(no cache hits: every such request writes the prefix) is a true bound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from src.analysis import stance_local as sl
from src.analysis.pricing import model_price, undated

logger = logging.getLogger(__name__)

# v2 less one sentence, word for word otherwise (tests check it).
V2_PRAISE_RULE = (
    "A reply that only agrees with or praises the post takes the post's position on the war, "
    "read in context: praising a decision to strike leans for; praising a ceasefire, a deal or "
    "holding off leans against. "
)
REPLY_V2B_PROMPT_VERSION = "reply-v2b-2026-11-02"
REPLY_V2B_PREFIX = (
    "Below is a post President Trump made on Truth Social, followed by a reply to it "
    "from another user.\n\n"
    'Trump\'s post: """{parent}"""\n\n'
    "What is the reply author's position on the war, meaning U.S. military action "
    "against Iran? Score it from -1.0 (against: wants peace, de-escalation, a deal, "
    "or holding off) to +1.0 (for: wants strikes, escalation, or regime change). "
    "Score 0 when the reply takes no position on the war or its position can't be "
    "told.\n\n"
    "Judge the position, not the tone: anger can be for the war, and warmth can be "
    "against it. Praise of Trump with no bearing on the war is 0.\n\n"
    "Respond with ONLY valid JSON: "
    '{{"score": <float from -1.0 (against the war) to 1.0 (for the war)>, '
    '"label": "<negative|neutral|positive>", '
    '"reasoning": "<one sentence>"}}\n\n'
)

# The categories and their definitions are event_study.STANCE_PROMPT's; the
# generic context line becomes the post itself, and the handle goes.
REPLY_CAT_V2_PROMPT_VERSION = "reply-cat-v2-2026-11-02"
REPLY_CAT_V2_PREFIX = (
    "Below is a post President Trump made on Truth Social, followed by a reply to it "
    "from another user.\n\n"
    'Trump\'s post: """{parent}"""\n\n'
    "Classify the reply's STANCE (not just sentiment) on the war, meaning U.S. military "
    "action against Iran, into exactly one category:\n"
    "- pro_war_supportive: supports Trump AND supports the war/strikes\n"
    "- pro_trump_antiwar: supports Trump generally but opposes THIS action "
    '("I love you but please stop")\n'
    "- antiwar_betrayal: feels personally betrayed, often mentions voting for Trump "
    '("voted 3x", "losing me as a supporter")\n'
    "- antiwar_opposition: opposes war on moral/political grounds (could be from left "
    "or right)\n"
    "- pro_war_critical: supports the war but criticizes Trump for not going far enough "
    "or for backing down\n"
    "- neutral_other: off-topic, unclear, or genuinely neutral\n\n"
    "Respond with ONLY valid JSON: "
    '{{"stance": "<one of the categories above>", "reasoning": "<one sentence>"}}\n\n'
)


@dataclass(frozen=True)
class ReplyJob:
    """A reply label job: prompt (prefix with {parent}, then the reply),
    id source ("stance_sample" or "fresh"), answer kind ("stance": a score;
    "category": one of event_study.STANCE_CATEGORIES), caps (settings
    names), and the stance label set whose files already hold labels from
    this prompt and model (their ids are never sent again). per_post: a
    submit names one post (--post), or is a --limit pilot. run_caps: the
    caps bound every batch of the job together, not each one."""

    name: str
    check: str
    prompt_version: str
    prefix: str
    ids: str
    kind: str
    cap_setting: str
    usd_cap_setting: str | None = None
    label_set: str | None = None
    per_post: bool = False
    run_caps: bool = False
    suffix: str = sl.REPLY_TEACHER_V2_SUFFIX
    model_setting: str = "TEACHER_CHECK_MODEL"
    second_rater: bool = False

    def default_model(self) -> str:
        return getattr(settings, self.model_setting)

    def build(self, parent: str, reply: str) -> tuple[str, int]:
        prefix = self.prefix.format(parent=parent)
        return prefix + self.suffix.format(reply=reply), len(prefix)

    def output_tokens(self, model: str = settings.TEACHER_CHECK_MODEL) -> tuple[float, float, bool]:
        """(output tokens a request expected, at the ceiling, whether
        measured for this model). A stance answer's from
        settings.REPLY_OUTPUT_TOKENS: a bare figure (a large batch's mean,
        which bounds a run's mean too) for both, or {"mean", "p95"} (a
        pilot's) with the ceiling at the p95. Unmeasured, Opus 5's mean, and
        twice it at the ceiling (thinking may run longer); a category
        answer's is a guess (REPLY_CAT_EST_OUTPUT_TOKENS), twice it at the
        ceiling, until a batch measures it."""
        mult = settings.REPLY_UNMEASURED_CEILING_MULT
        if self.kind == "category":
            guess = settings.REPLY_CAT_EST_OUTPUT_TOKENS
            return guess, guess * mult, False
        got = settings.REPLY_OUTPUT_TOKENS.get(undated(model))
        if got is None:
            base = settings.REPLY_OUTPUT_TOKENS[settings.REPLY_V2_BATCH_USAGE["model"]]
            return base, base * mult, False
        if isinstance(got, dict):
            return float(got["mean"]), max(float(got["mean"]), float(got["p95"])), True
        return float(got), float(got), True


JOBS = {
    "v2b": ReplyJob(
        "v2b",
        "reply_v2b",
        REPLY_V2B_PROMPT_VERSION,
        REPLY_V2B_PREFIX,
        "stance_sample",
        "stance",
        "REPLY_TEACHER_V2_MAX_CALLS",
        label_set="v2b",
    ),
    "cat-v2": ReplyJob(
        "cat-v2",
        "reply_cat_v2",
        REPLY_CAT_V2_PROMPT_VERSION,
        REPLY_CAT_V2_PREFIX,
        "stance_sample",
        "category",
        "REPLY_TEACHER_V2_MAX_CALLS",
    ),
    "v2-fresh": ReplyJob(
        "v2-fresh",
        "reply_v2_fresh",
        sl.REPLY_TEACHER_V2_PROMPT_VERSION,
        sl.REPLY_TEACHER_V2_PREFIX,
        "fresh",
        "stance",
        "REPLY_FRESH_MAX_CALLS",
        "REPLY_FRESH_MAX_USD",
        label_set="v2",
        per_post=True,
        run_caps=True,
    ),
    "v2-rater": ReplyJob(
        "v2-rater",
        "reply_v2_rater",
        sl.REPLY_TEACHER_V2_PROMPT_VERSION,
        sl.REPLY_TEACHER_V2_PREFIX,
        "stance_sample",
        "stance",
        "REPLY_TEACHER_V2_MAX_CALLS",
        "SECOND_RATER_REPLY_MAX_USD",
        run_caps=True,
        model_setting="SECOND_RATER_MODEL",
        second_rater=True,
    ),
}


def job(name: str) -> ReplyJob:
    if name not in JOBS:
        raise ValueError(f"no reply label job {name!r}: one of {', '.join(JOBS)}")
    return JOBS[name]


# ── Inputs and the to-do rows ──────────────────────────────────────


def job_ids(j: ReplyJob, preview: bool = False) -> tuple[pd.DataFrame, bool]:
    """(id, tracked_slug of the replies the job labels, whether they are a
    preview). The stance sample, or the fresh draws' manifest; with
    `preview` and posts the manifest lacks, those posts' draws as
    `reply-fresh-sample` would record them (nothing written), so an estimate
    can price the run before the draw. A submit never previews."""
    from src.analysis import inference as inf
    from src.analysis.event_study import STANCE_OUTPUT

    if j.ids == "stance_sample":
        return pd.read_parquet(STANCE_OUTPUT, columns=["id", "tracked_slug"]), False
    have = inf.read_fresh_draws()
    if not preview:
        if have.empty:
            raise FileNotFoundError(
                f"no fresh draws ({inf.fresh_draws_path().name}): run "
                "`reply-fresh-sample` first; it records them before any is sent"
            )
        return have[["id", "tracked_slug"]], False
    draws, _ = inf.draw_fresh_sample(write=False)
    return draws[["id", "tracked_slug"]], len(draws) > len(have)


def done_ids(j: ReplyJob, model: str) -> tuple[pd.DataFrame, set[str]]:
    """(the job's own label cache, every id already labelled by this prompt
    and model: the cache plus the other files of the job's label set)."""
    cached = sl.read_label_cache(sl.reply_labels_path(model, j.name), j.prompt_version)
    done = set(cached["id"].astype(str))
    for version in sl.REPLY_LABEL_SETS.get(j.label_set or "", ()):
        if version != j.name and sl.reply_labels_path(model, version).exists():
            other = sl.read_label_cache(sl.reply_labels_path(model, version), j.prompt_version)
            done |= set(other["id"].astype(str))
    return cached, done


def job_todo(
    j: ReplyJob, model: str, params: dict, limit: int | None = None, preview: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """(inputs, the job's cached labels, rows still to label, preview). The
    to-do rows are the inputs with text whose id no file of this prompt and
    model holds, only params["post"]'s when given, in stratified_order
    across posts; `limit` keeps the first N (a pilot). prefix_chars is the
    post's prefix length; cache_chars, where the request puts its cache
    breakpoint, is that only where the prefix reaches the model's cache
    minimum (cacheable) and 0 elsewhere, so no request pays a cache write
    the estimate does not count."""
    ids, is_preview = job_ids(j, preview)
    inputs = sl.reply_prompt_inputs(ids, j.build, j.prompt_version)
    cached, done = done_ids(j, model)
    todo = inputs[inputs["has_text"] & ~inputs["id"].isin(done)]
    post = params.get("post")
    if post is not None:
        if post not in set(inputs["tracked_slug"]):
            raise ValueError(
                f"no replies of post {post!r} in this job's ids: one of "
                f"{', '.join(sorted(inputs['tracked_slug'].unique()))}"
            )
        todo = todo[todo["tracked_slug"] == post]
    todo = sl.stratified_order(
        todo, "tracked_slug", params.get("seed", settings.TEACHER_SAMPLE_SEED)
    )
    if limit is not None:
        todo = todo.head(limit)
    todo = todo.assign(prefix_chars=todo["cache_chars"].astype(int))
    ok = {
        slug: cacheable(j, slug, int(g["prefix_chars"].iloc[0]), model)
        for slug, g in todo.groupby("tracked_slug")
    }
    keep = todo["tracked_slug"].map(ok).fillna(False).astype(bool)
    todo["cache_chars"] = todo["prefix_chars"].where(keep, 0).astype(int)
    cols = [
        "id",
        "tracked_slug",
        "prompt",
        "cache_chars",
        "prefix_chars",
        "input_chars",
        "prompt_version",
    ]
    return inputs, cached, todo[cols], is_preview


# ── Calibrated estimate (no call) ──────────────────────────────────


def prefix_tokens(j: ReplyJob, slug: str, cache_chars: int) -> float:
    """Input tokens before the reply in one of the job's requests: the
    post's measured v2 prefix (settings.REPLY_V2_BATCH_USAGE) plus the
    job's extra instruction characters at v2's characters per token. A post
    v2 never labelled is priced from its prefix length alone."""
    u = settings.REPLY_V2_BATCH_USAGE
    per = u["instruction_chars_per_token"]
    if slug not in u["prefix_tokens"]:
        return cache_chars / per + u["wrapper_tokens"]
    extra = len(j.prefix.format(parent="")) - len(sl.REPLY_TEACHER_V2_PREFIX.format(parent=""))
    return u["prefix_tokens"][slug] + extra / per


def cacheable(j: ReplyJob, slug: str, prefix_chars: int, model: str) -> bool:
    """Whether the post's cached block (its prefix less the reply wrapper)
    reaches the model's cache minimum: only then does a request carry a
    cache breakpoint (job_todo) and the estimate price cache reads and
    writes (calibrated_cost)."""
    block = prefix_tokens(j, slug, prefix_chars) - settings.REPLY_V2_BATCH_USAGE["wrapper_tokens"]
    return block >= model_price(model)["cache_min_tokens"]


def calibrated_cost(j: ReplyJob, rows: pd.DataFrame, model: str) -> dict:
    """The rows (tracked_slug, input_chars, and prefix_chars or cache_chars)
    priced at `model`'s row of settings.MODEL_PRICES with v2's measured
    token counts: input = the post's prefix + REPLY_V2_BATCH_USAGE tokens
    per reply character; where the cached block (the prefix less the reply
    wrapper) is cacheable it is a cache read on the measured share of
    requests and a write on the rest; output per request from the job.
    usd_batch is that at batch prices. usd_batch_ceiling has no cache hits:
    every request on a cacheable post writes its block (cache_write_mult),
    the rest pay plain input; it is the bound the dollar cap is checked
    against, with output at the job's ceiling a request (output_tokens:
    the measured mean, a pilot's p95, or twice the guess unmeasured).
    usd_direct_ceiling is the same off-batch. Token counts are Opus 5's;
    output_measured says whether the output length is measured for this job
    and model."""
    u = settings.REPLY_V2_BATCH_USAGE
    p = model_price(model)
    out_tok, out_ceil, measured = j.output_tokens(model)
    hit = u["cache_hit_share"]
    cached_mult = hit * p["cache_read_mult"] + (1 - hit) * p["cache_write_mult"]
    write_mult = max(1.0, p["cache_write_mult"])
    by_post, cached_posts = {}, []
    tot = {"input_tokens": 0.0, "billed_input": 0.0, "ceiling_input": 0.0}
    chars_col = "prefix_chars" if "prefix_chars" in rows else "cache_chars"
    for slug, g in rows.groupby("tracked_slug"):
        pre = prefix_tokens(j, slug, int(g[chars_col].iloc[0]))
        block = pre - u["wrapper_tokens"]
        reply = u["tokens_per_reply_char"] * g["input_chars"].to_numpy(dtype=float)
        n = len(g)
        tin = pre * n + reply.sum()
        cached = block >= p["cache_min_tokens"]
        if cached:
            billed = (pre - block) * n + block * cached_mult * n + reply.sum()
            ceiling = (pre - block) * n + block * write_mult * n + reply.sum()
            cached_posts.append(slug)
        else:
            billed = ceiling = tin
        out_usd, ceil_usd = out_tok * n * p["output"], out_ceil * n * p["output"]
        by_post[str(slug)] = {
            "calls": n,
            "prefix_tokens": round(pre, 1),
            "cached": bool(cached),
            "usd_batch": (billed * p["input"] + out_usd) / 1e6 * p["batch_discount"],
            "usd_batch_ceiling": (ceiling * p["input"] + ceil_usd) / 1e6 * p["batch_discount"],
        }
        tot["input_tokens"] += tin
        tot["billed_input"] += billed
        tot["ceiling_input"] += ceiling
    n = len(rows)
    out = out_tok * n
    direct = (tot["ceiling_input"] * p["input"] + out_ceil * n * p["output"]) / 1e6
    return {
        "model": model,
        "calls": n,
        "input_tokens": round(tot["input_tokens"]),
        "output_tokens": round(out),
        "output_tokens_per_call": out_tok,
        "output_tokens_ceiling_per_call": out_ceil,
        "output_measured": measured,
        "cache_min_tokens": int(p["cache_min_tokens"]),
        "cache_hit_share": hit,
        "cached_posts": sorted(map(str, cached_posts)),
        "by_post": by_post,
        "usd_batch": (tot["billed_input"] * p["input"] + out * p["output"])
        / 1e6
        * p["batch_discount"],
        "usd_batch_ceiling": direct * p["batch_discount"],
        "usd_direct_ceiling": direct,
        "price_in_per_mtok": p["input"],
        "price_out_per_mtok": p["output"],
        "batch_discount": p["batch_discount"],
    }


def submit_problem(
    j: ReplyJob, model: str, post: str | None, limit: int | None, n: int
) -> str | None:
    """Why a submit of n requests must not go, or None: the second rater on
    the v2 labels' own model; a per-post job sent without --post (other than
    a --limit pilot of at most REPLY_UNMEASURED_PILOT_MAX_CALLS); a job with
    a dollar cap on a model whose output length is unmeasured, past a pilot
    of that size counting the calls its batches of this prompt and model
    have sent (the cap would bound a guess); or an earlier batch of the job
    on this model at the current REPLY_LABEL_MAX_TOKENS with an answer cut
    at max_tokens (the cut ones would be sent again and cut again)."""
    if j.second_rater and undated(model) == undated(settings.TEACHER_CHECK_MODEL):
        return (
            f"{model} made the v2 labels of record (`reply-teacher-check --v2`): {j.name} is "
            f"a second model (settings.SECOND_RATER_MODEL, {settings.SECOND_RATER_MODEL})"
        )
    pilot = settings.REPLY_UNMEASURED_PILOT_MAX_CALLS
    if j.per_post and post is None and (limit is None or limit > pilot):
        return (
            f"{j.name} goes one post at a time: pass --post <slug> (or --limit {pilot} or "
            "fewer for a pilot)"
        )
    max_tokens = settings.REPLY_LABEL_MAX_TOKENS
    stops = sl.batch_run_totals(j.check, {"model": model, "max_tokens": max_tokens})
    cut = stops["stop_reasons"].get("max_tokens", 0)
    if cut:
        return (
            f"{cut} answers of earlier {j.name} batches on {model} stopped at max_tokens "
            f"({max_tokens}; thinking counts against it): raise REPLY_LABEL_MAX_TOKENS, "
            "then pilot again"
        )
    _, _, measured = j.output_tokens(model)
    sent = sl.batch_run_totals(j.check, {"model": model, "prompt_version": j.prompt_version})
    if j.usd_cap_setting and not measured and sent["calls"] + n > pilot:
        so_far = f" ({sent['calls']} sent already)" if sent["calls"] else ""
        return (
            f"{model}'s output length on {j.name} is unmeasured, so the dollar cap would bound "
            f"a guess: pilot with --limit {pilot} or fewer in all{so_far}, then record the "
            'collect\'s output tokens a call as {"mean": ..., "p95": ...} in '
            "REPLY_OUTPUT_TOKENS"
        )
    return None


def run_so_far(j: ReplyJob) -> dict:
    """What the job's batches have sent and cost so far when its caps bound
    the whole run (stance_local.batch_run_totals), else nothing."""
    if not j.run_caps:
        return {"batches": 0, "calls": 0, "usd_batch": 0.0}
    return sl.batch_run_totals(j.check)


def refuse_fresh_run(draws: pd.DataFrame, model: str = settings.TEACHER_CHECK_MODEL) -> dict:
    """The fresh run a manifest `draws` implies, priced with no call: every
    draw with text that no v2 file holds, plus what v2-fresh batches have
    sent already. Raises ValueError, before the manifest is recorded, when
    the calls or the dollar ceiling are over REPLY_FRESH_MAX_CALLS /
    REPLY_FRESH_MAX_USD."""
    j = JOBS["v2-fresh"]
    inputs = sl.reply_prompt_inputs(draws[["id", "tracked_slug"]], j.build, j.prompt_version)
    _, done = done_ids(j, model)
    todo = inputs[inputs["has_text"] & ~inputs["id"].isin(done)]
    prior = run_so_far(j)
    cost = calibrated_cost(j, todo, model)
    calls, usd = prior["calls"] + len(todo), prior["usd_batch"] + cost["usd_batch_ceiling"]
    cap, usd_cap = getattr(settings, j.cap_setting), getattr(settings, j.usd_cap_setting)
    if calls > cap:
        raise ValueError(
            f"the fresh run would make {calls} calls, over {j.cap_setting} ({cap}): "
            "raise it deliberately, or draw fewer a post"
        )
    if usd > usd_cap:
        raise ValueError(
            f"the fresh run's ceiling is ${usd:.2f} at batch prices with no cache hits, over "
            f"{j.usd_cap_setting} (${usd_cap:g}): raise it deliberately, or draw fewer a post"
        )
    return {"calls": calls, "usd_batch_ceiling": usd, "usd_batch": cost["usd_batch"]}


def job_estimate(
    name: str,
    model: str | None = None,
    post: str | None = None,
    limit: int | None = None,
    seed: int = settings.TEACHER_SAMPLE_SEED,
) -> dict:
    """What `reply-label submit --job <name>` would send and what it would
    cost, making no call (the fresh job previews draws not yet recorded).
    model None is the job's default (j.default_model)."""
    j = job(name)
    model = model or j.default_model()
    inputs, _, todo, preview = job_todo(j, model, {"seed": seed, "post": post}, limit, preview=True)
    with_text = inputs[inputs["has_text"]]
    _, done = done_ids(j, model)
    usd_cap = getattr(settings, j.usd_cap_setting) if j.usd_cap_setting else None
    cost = calibrated_cost(j, todo, model)
    prior = run_so_far(j)
    return {
        "job": j.name,
        "prompt_version": j.prompt_version,
        "ids": j.ids,
        "preview": preview,
        "inputs": len(inputs),
        "no_text": int((~inputs["has_text"]).sum()),
        "with_text": len(with_text),
        "labelled": int(with_text["id"].isin(done).sum()),
        "todo": len(todo),
        "cap": getattr(settings, j.cap_setting),
        "usd_cap": usd_cap,
        "todo_by_post": {str(k): int(v) for k, v in todo.groupby("tracked_slug").size().items()},
        "run_caps": j.run_caps,
        "prior_batches": prior["batches"],
        "prior_calls": prior["calls"],
        "prior_usd_batch": prior["usd_batch"],
        "run_calls": prior["calls"] + len(todo),
        "run_usd_ceiling": prior["usd_batch"] + cost["usd_batch_ceiling"],
        "problem": submit_problem(j, model, post, limit, len(todo)),
        **cost,
    }


# ── Batch checks ───────────────────────────────────────────────────


def parse_category_result(result) -> tuple[str, str, dict | None]:
    """(id, outcome, {"category": ...} or None): the answer's "stance" (or
    "category") must be one of event_study.STANCE_CATEGORIES."""
    from src.analysis.event_study import STANCE_CATEGORIES
    from src.analysis.sentiment import parse_llm_json

    rid, outcome = str(result.custom_id), result.result.type
    if outcome != "succeeded":
        return rid, outcome, None
    text = next(
        (b.text for b in result.result.message.content if getattr(b, "type", "") == "text"), ""
    )
    data = parse_llm_json(text)
    cat = data.get("stance", data.get("category")) if isinstance(data, dict) else None
    if not isinstance(cat, str):
        return rid, "unparseable", None
    if cat.strip() not in STANCE_CATEGORIES:
        return rid, "out_of_range", None
    return rid, outcome, {"category": cat.strip()}


def _check(j: ReplyJob) -> sl.BatchCheck:
    def todo(model, params, limit):
        _, cached, rows, _ = job_todo(j, model, params, limit)
        return cached, rows

    def guard(model, params, limit, rows):
        why = submit_problem(j, model, params.get("post"), limit, len(rows))
        if why:
            raise RuntimeError(why)

    kind = (
        {"parse": parse_category_result, "value_cols": ("category",)}
        if j.kind == "category"
        else {}
    )
    return sl.BatchCheck(
        j.prompt_version,
        j.cap_setting,
        lambda m: sl.reply_labels_path(m, j.name),
        todo,
        lambda m, p: job_report(j.name, m),
        carries_replies=True,
        usd_cap_setting=j.usd_cap_setting,
        price_rows=lambda rows, m: calibrated_cost(j, rows, m)["usd_batch_ceiling"],
        run_caps=j.run_caps,
        submit_guard=guard,
        **kind,
    )


REPLY_BATCH_CHECKS = {j.check: _check(j) for j in JOBS.values()}


# ── Reports (aggregates only) ──────────────────────────────────────


def report_path(name: str, model: str):
    return settings.PROCESSED_DIR / f"reply_teacher_check_{name}_{model.replace('/', '_')}.json"


def _with_model_col(lab: pd.DataFrame, col: str = "score_opus_distilled") -> pd.DataFrame:
    reps = pd.read_parquet(settings.REPLY_SENTIMENT_OUTPUT, columns=["id", col])
    reps = reps.assign(id=reps["id"].astype(str)).drop_duplicates("id")
    return lab.merge(reps, on="id", how="left")


def _v2b_report(model: str) -> dict:
    """v2b against v2 on the replies both label: agreement, the mean shift,
    per post the sample's (unweighted) mean and pro / anti shares under
    each, and the population estimates of both on the same first draws
    with their paired difference (inference.reply_population_paired; also
    written to reply_population_paired_v2_v2b_<model>.csv). Not
    `reply-population --labels v2` against `--labels v2b`: v2 there pools
    the fresh draws, which v2b does not label."""
    from src.analysis import inference as inf

    v2b = sl.read_label_cache(sl.reply_labels_path(model, "v2b")).dropna(subset=["score_teacher"])
    rep = {"job": "v2b", "model": model, "prompt_version": REPLY_V2B_PROMPT_VERSION, "n": len(v2b)}
    if v2b.empty:
        return rep
    v2 = sl.read_label_cache(sl.reply_labels_path(model, "v2")).dropna(subset=["score_teacher"])
    both = (
        v2b.drop_duplicates("id")
        .merge(
            v2.drop_duplicates("id")[["id", "score_teacher"]].rename(
                columns={"score_teacher": "v2"}
            ),
            on="id",
        )
        .rename(columns={"score_teacher": "v2b"})
    )
    rep["v2_vs_v2b"] = {
        **sl.agreement(both["v2"].values, both["v2b"].values),
        "mean_shift_v2b_minus_v2": float((both["v2b"] - both["v2"]).mean()),
    }
    rows = []
    for post, g in [*both.groupby("tracked_slug"), ("ALL", both)]:
        a, b = sl._shares(g["v2"]), sl._shares(g["v2b"])
        rows.append(
            {
                "post": post,
                "n": len(g),
                **{f"v2_{k}": v for k, v in a.items()},
                **{f"v2b_{k}": v for k, v in b.items()},
                "moved_sign": int((sl.sign3(g["v2"].values) != sl.sign3(g["v2b"].values)).sum()),
            }
        )
    rep["by_post"] = rows
    pop = inf.reply_population_paired("v2", "v2b", model=model)
    pop.to_csv(
        settings.PROCESSED_DIR / f"reply_population_paired_v2_v2b_{model.replace('/', '_')}.csv",
        index=False,
    )
    rep["population"] = pop.to_dict("records")
    return rep


def deescalation_posts() -> list[str]:
    """The tracked posts of category "ceasefire" (config/tracked_posts.py):
    where a category's "THIS action" is the de-escalation itself."""
    from config.tracked_posts import TRACKED_POSTS

    return [t.slug for t in TRACKED_POSTS if t.category == "ceasefire"]


# The cat-v2 cells to read before the full batch on the de-escalation posts:
# opposing "THIS action" there can mean wanting the war back.
CAT_CHECK_CELLS = (
    "pro_trump_antiwar -> pro",
    "antiwar_betrayal -> pro",
    "antiwar_opposition -> pro",
)


def _cat_report(model: str) -> dict:
    """Counts of the Opus categories by post, against the Haiku categories
    and the v2 stance sign on the same replies (counts only), and their
    population shares (inference.reply_category_population, also written to
    reply_categories_population_<model>.csv)."""
    from src.analysis import inference as inf
    from src.analysis.event_study import STANCE_OUTPUT

    cats = sl.read_label_cache(sl.reply_labels_path(model, "cat-v2"))
    rep = {
        "job": "cat-v2",
        "model": model,
        "prompt_version": REPLY_CAT_V2_PROMPT_VERSION,
        "n": int(cats["category"].notna().sum()) if "category" in cats else 0,
    }
    if not rep["n"]:
        return rep
    cats = cats.dropna(subset=["category"]).drop_duplicates("id")
    rep["by_post"] = {
        str(k): {str(c): int(n) for c, n in g["category"].value_counts().items()}
        for k, g in cats.groupby("tracked_slug")
    }
    haiku = pd.read_parquet(STANCE_OUTPUT, columns=["id", "stance"])
    j = cats.merge(
        haiku.assign(id=haiku["id"].astype(str)).drop_duplicates("id"), on="id", how="left"
    )
    rep["vs_haiku"] = {
        f"{h} -> {o}": int(n)
        for (h, o), n in j.groupby([j["stance"].fillna("none"), "category"]).size().items()
    }
    v2 = sl.reply_labels_path(model, "v2")
    if v2.exists():
        lab = sl.read_label_cache(v2).dropna(subset=["score_teacher"]).drop_duplicates("id")
        k = cats.merge(lab[["id", "score_teacher"]], on="id")
        sign = pd.Series(sl.sign3(k["score_teacher"].values), index=k.index).map(
            {-1: "anti", 0: "neutral", 1: "pro"}
        )
        rep["vs_v2_sign"] = {
            f"{c} -> {s}": int(n) for (c, s), n in k.groupby([k["category"], sign]).size().items()
        }
        rep["vs_v2_sign_deescalation"] = {
            str(post): {
                "n": len(g),
                **{
                    f"{c} -> {s}": int(n)
                    for (c, s), n in g.groupby([g["category"], sign[g.index]]).size().items()
                },
            }
            for post, g in k[k["tracked_slug"].isin(deescalation_posts())].groupby("tracked_slug")
        }
    pop = inf.reply_category_population(model)
    pop.to_csv(
        settings.PROCESSED_DIR / f"reply_categories_population_{model.replace('/', '_')}.csv",
        index=False,
    )
    rep["population"] = pop.to_dict("records")
    return rep


def _fresh_report(model: str) -> dict:
    """The fresh draws' v2 labels: per post the draws, those labelled and
    their share, and, since a post's draws are proportional to its buckets,
    the plain mean and pro / anti shares over its labelled draws; the
    distilled score against them. No ALL row: every post has about the same
    number of draws whatever its size, so a pooled plain mean matches no
    estimand (reply-population weighs the posts)."""
    from src.analysis import inference as inf

    lab = sl.read_label_cache(sl.reply_labels_path(model, "v2-fresh")).dropna(
        subset=["score_teacher"]
    )
    draws = inf.read_fresh_draws()
    rep = {
        "job": "v2-fresh",
        "model": model,
        "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
        "drawn": len(draws),
        "n": len(lab),
    }
    if lab.empty and draws.empty:
        return rep
    # Every fresh draw with a v2 label of either file: the reused ones count.
    try:
        labs = sl.read_reply_label_set(model, "v2")
    except FileNotFoundError:
        labs = lab
    d = draws[["id", "tracked_slug"]].merge(labs[["id", "score_teacher"]], on="id")
    d = _with_model_col(d)
    rep["labelled_draws"] = len(d)
    drawn = draws.groupby("tracked_slug")["id"].nunique()
    rep["by_post"] = [
        {
            "post": post,
            "drawn": int(drawn.get(post, 0)),
            "n": len(g),
            "labelled_share": len(g) / drawn[post] if drawn.get(post) else float("nan"),
            **sl._shares(g["score_teacher"]),
        }
        for post, g in d.groupby("tracked_slug")
    ]
    rep["distilled_vs_teacher"] = sl.agreement(
        d["score_teacher"].values, d["score_opus_distilled"].values
    )
    return rep


def _rater_report(model: str) -> dict:
    """The second model's v2 labels against the v2 labels of record
    (settings.TEACHER_CHECK_MODEL) on the replies both label: Pearson,
    Spearman, MAE, sign flips and Bland-Altman (rater - record), overall and
    by post, and per post the (unweighted) mean and pro / anti shares under
    each. Coverage first: the replies with text still unlabelled, and how
    the job's batches on this model failed (by outcome) and stopped, since
    agreement on the replies the model could answer may not hold on the
    rest."""
    from src.analysis.post_labels import rater_stats, rater_table

    j = JOBS["v2-rater"]
    lab = sl.read_label_cache(
        sl.reply_labels_path(model, "v2-rater"), sl.REPLY_TEACHER_V2_PROMPT_VERSION
    ).dropna(subset=["score_teacher"])
    rep = {
        "job": "v2-rater",
        "model": model,
        "prompt_version": sl.REPLY_TEACHER_V2_PROMPT_VERSION,
        "reference": settings.TEACHER_CHECK_MODEL,
        "n": len(lab),
    }
    inputs, _, todo, _ = job_todo(j, model, {"seed": settings.TEACHER_SAMPLE_SEED})
    runs = sl.batch_run_totals(j.check, {"model": model})
    rep["coverage"] = {
        "with_text": int(inputs["has_text"].sum()),
        "unlabelled": len(todo),
        "batches": runs["batches"],
        "failed_by_outcome": runs["failed_by_outcome"],
        "stop_reasons": runs["stop_reasons"],
    }
    ref_path = sl.reply_labels_path(settings.TEACHER_CHECK_MODEL, "v2")
    if lab.empty or not ref_path.exists():
        return rep
    ref = sl.read_label_cache(ref_path).dropna(subset=["score_teacher"]).drop_duplicates("id")
    both = lab.drop_duplicates("id").merge(
        ref[["id", "score_teacher", "input_chars"]].rename(
            columns={"score_teacher": "v2", "input_chars": "v2_chars"}
        ),
        on="id",
    )
    both = both[both["input_chars"] == both["v2_chars"]].rename(columns={"score_teacher": "rater"})
    rep["vs_v2"] = rater_stats(both["v2"], both["rater"])
    rep["by_post"] = rater_table(both, "v2", "rater", "tracked_slug")
    rep["shares_by_post"] = [
        {
            "post": post,
            "n": len(g),
            **{f"v2_{k}": v for k, v in sl._shares(g["v2"]).items()},
            **{f"rater_{k}": v for k, v in sl._shares(g["rater"]).items()},
        }
        for post, g in [*both.groupby("tracked_slug"), ("ALL", both)]
    ]
    return rep


def job_report(name: str, model: str | None = None) -> dict:
    """The job's report, written to reply_teacher_check_<job>_<model>.json."""
    j = job(name)
    model = model or j.default_model()
    rep = {
        "v2b": _v2b_report,
        "cat-v2": _cat_report,
        "v2-fresh": _fresh_report,
        "v2-rater": _rater_report,
    }[j.name](model)
    sl._write_json(report_path(name, model), rep)
    return rep


def summary_lines(rep: dict) -> list[str]:
    """A few printable lines of a job report (aggregates only)."""
    lines = [f"{rep['job']} labels from {rep['model']}: {rep.get('n', 0)}"]
    if rep["job"] == "v2b" and "v2_vs_v2b" in rep:
        a = rep["v2_vs_v2b"]
        lines.append(
            f"v2 vs v2b on {a['n']} replies: pearson {a['pearson']:.3f}, flips "
            f"{a['sign_flip_rate']:.1%}, mean shift {a['mean_shift_v2b_minus_v2']:+.3f}"
        )
        lines.append("sample (unweighted):")
        for r in rep["by_post"]:
            lines.append(
                f"  {r['post']:20s}{r['n']:>5d}  v2 {r['v2_mean']:+.3f} "
                f"({r['v2_pro']:.0%} pro / {r['v2_anti']:.0%} anti)  v2b {r['v2b_mean']:+.3f} "
                f"({r['v2b_pro']:.0%} / {r['v2b_anti']:.0%})  sign moved {r['moved_sign']}"
            )
        if rep.get("population"):
            lines.append("population, on the first draws both label (paired CIs):")
            for r in rep["population"]:
                lines.append(
                    f"  {r['post']:20s}{r['n_labelled']:>5d}  v2 {_ci_txt(r, 'v2_mean')}  "
                    f"v2b {_ci_txt(r, 'v2b_mean')}  diff {_ci_txt(r, 'diff_mean')}  "
                    f"anti {_ci_txt(r, 'v2_anti', '.0%')} -> {_ci_txt(r, 'v2b_anti', '.0%')}"
                )
    elif rep["job"] == "cat-v2" and "population" in rep:
        pop = pd.DataFrame(rep["population"])
        for post, g in pop.groupby("post", sort=False):
            parts = [
                f"{r.category} {r.opus_share:.0%} (haiku {r.haiku_share:.0%})"
                for r in g.itertuples()
                if np.isfinite(r.opus_share)
            ]
            lines.append(f"  {post:20s} " + ("; ".join(parts) if parts else "not covered"))
        for post, cells in (rep.get("vs_v2_sign_deescalation") or {}).items():
            got = ", ".join(f"{c} {cells.get(c, 0)}" for c in CAT_CHECK_CELLS)
            lines.append(f"  check {post} (of {cells['n']} with a v2 label): {got}")
    elif rep["job"] == "v2-rater" and "vs_v2" in rep:
        c = rep.get("coverage") or {}
        if c:
            lines.append(
                f"coverage: {c['with_text'] - c['unlabelled']} of {c['with_text']} replies with "
                f"text labelled; {c['batches']} batches, failed by outcome "
                f"{c['failed_by_outcome'] or 0}, stop reasons {c['stop_reasons'] or 'n/a'}"
            )
        a = rep["vs_v2"]
        lines.append(
            f"vs the v2 labels of {rep['reference']} on {a['n']} replies: pearson "
            f"{a['pearson']:.3f}, spearman {a['spearman']:.3f}, mae {a['mae']:.3f}, flips "
            f"{a['sign_flips']}, mean diff {a['mean_diff']:+.3f} [LoA {a['loa_lo']:+.2f}, "
            f"{a['loa_hi']:+.2f}]"
        )
        for r in rep["shares_by_post"]:
            lines.append(
                f"  {r['post']:20s}{r['n']:>5d}  v2 {r['v2_mean']:+.3f} ({r['v2_pro']:.0%} pro / "
                f"{r['v2_anti']:.0%} anti)  rater {r['rater_mean']:+.3f} ({r['rater_pro']:.0%} / "
                f"{r['rater_anti']:.0%})"
            )
    elif rep["job"] == "v2-fresh" and "by_post" in rep:
        lines.append(
            f"{rep['drawn']} drawn, {rep['labelled_draws']} with a v2 label; plain means over "
            "each post's draws (population figures: reply-population)"
        )
        for r in rep["by_post"]:
            lines.append(
                f"  {r['post']:20s}{r['n']:>6d} of {r['drawn']:<6d} mean {r['mean']:+.3f}  "
                f"pro {r['pro']:.0%}  anti {r['anti']:.0%}"
            )
    return lines


def _ci_txt(r: dict, k: str, fmt: str = "+.3f") -> str:
    if not np.isfinite(r[k]):
        return "n/a"
    return f"{r[k]:{fmt}} [{r[k + '_lo']:{fmt}}, {r[k + '_hi']:{fmt}}]"
