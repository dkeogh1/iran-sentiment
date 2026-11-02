"""
Alternate raters on broadcaster posts: Claude labels beyond score_opus, on
one path whose parameters are the model, the prompt version and the post
set (`post-label estimate|pilot|submit|status|collect|report --job <job>`).

  stance-topic           stance_topic_prompt: a topic class (iran_war /
                         war_or_peace_general / other_conflict / not_war)
                         and the author's position on U.S. military action
                         against Iran (null when the post is not about
                         Iran), with the author's handle where llm_prompt
                         has it; on the war candidates (X accounts and
                         Trump's feed) plus a seeded random POST_RANDOM_N
                         unflagged X posts (what the two filters miss)
  stance-topic-census    the same prompt, model, label file and columns on
                         every other post with text that has no current
                         stance-topic label (set "rest"), so the topic flag
                         of record (settings.TOPIC_SOURCE "opus") rests on
                         one Opus label on every post. Its own batches and
                         caps (POST_CENSUS_*); its output is measured by
                         stance-topic's pilot and collected batches, so it
                         takes no pilot
  rater                  llm_prompt, the score_opus prompt word for word, on
                         a second model (settings.SECOND_RATER_MODEL), on
                         the war candidates: do the published contrasts
                         hold under another strong rater?
  stance-topic-nohandle  stance-topic without the handle, on a seeded paired
                         subsample of the X candidates: how much of a tier's
                         level is the model's prior about the author?

The war candidates are the posts with text flagged by the keyword pattern OR
the Haiku topic label (inference.war_flag "either"), the X search rows left
out. Each job has its own label file (post_labels_<job>_<model>.parquet;
the census writes stance-topic's), batch state (teacher_batch_post_*.json),
caps (settings.POST_*_MAX_CALLS and _MAX_USD) and report. Its columns carry
the model's tag (pricing.model_tag: score_opus_stance and topic_opus on
claude-opus-5, score_opus55 and label_opus55 for the rater on
claude-opus-5-5) and are joined onto a frame when it is read (attach:
`phases --score <col>`), on every post the job labelled (any set) and only
where the post's text is the one labelled. opus_war_flag is the about-the-war
flag of record (inference.war_flag "opus"): a post is a war post when its
current stance-topic label says iran_war. Nothing here writes
sentiment_all.parquet, the Trump feed parquet, score_opus or relabel's label
file.

The random posts and the subsample are the smallest seeded hashes of their
ids, recorded in a manifest by the first pilot or submit and never redrawn.

Estimates make no call: input tokens from the llm_prompt fit on billed
usage (settings.POST_PROMPT_TOKEN_FIT) plus a longer prompt's extra
instruction characters, output tokens measured or guessed. `pilot` sends
the first N posts still to do through the direct API, records what they
were billed and how each answer stopped, and the estimate is calibrated on
it (the census: on stance-topic's pilot and collected batches pooled);
batches of a model and prompt whose output length is unmeasured are
refused past a pilot's size in all, and every batch after a pilot or batch
answer stopped at max_tokens (thinking counts against it). Each job's caps
bound its whole run: every pilot and batch it has made, on any model. Jobs
that share a label file refuse to send while another's batch is open, so
no post is paid for twice.

A label made from a text the post no longer has does not count as done: the
post is labelled again and the old label moves to the cache's *_superseded
archive (x-backfill-text, collect --force and ts-fill-text also move these
caches' labels there when a text changes).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import settings
from src.analysis import stance_local as sl
from src.analysis.pricing import model_price, model_tag, undated

logger = logging.getLogger(__name__)

TOPIC_CLASSES = ("iran_war", "war_or_peace_general", "other_conflict", "not_war")
IRAN_WAR = "iran_war"
OUTSIDE = "outside phases"

STANCE_TOPIC_PROMPT_VERSION = "post-stance-topic-2026-11-02"
STANCE_TOPIC_NOHANDLE_PROMPT_VERSION = "post-stance-topic-nohandle-2026-11-02"


# The instructions come first and are the same for every post, so they go
# as a cached prefix (stance_local.prompt_content) when they reach the
# model's cache minimum (512 tokens on Opus 5: about 710 here at v2's 2.67
# characters a token); the author's handle, where llm_prompt has it, and the
# post come after them.
STANCE_TOPIC_INSTRUCTIONS = (
    "You will read a social media post from 2026, the year of the U.S. war with Iran, and "
    "answer two questions about it.\n\n"
    "1. What is the post about? Pick exactly one topic:\n"
    "- iran_war: the 2026 Iran war: U.S. or Israeli military action against Iran, Iranian "
    "attacks or retaliation, the ceasefire, negotiations or a deal with Iran, the Strait of "
    "Hormuz or the blockade, Iran's nuclear program, or the U.S. political fight over the war "
    "(including attacks on people for their position on it). In 2026, posts about U.S. "
    'strikes, a military victory or "the war" that name no country are almost always about '
    "Iran; count them.\n"
    "- war_or_peace_general: war, peace or military force in general, tied to no particular "
    "conflict (a prayer for peace, a warning against war).\n"
    "- other_conflict: a particular war or conflict other than Iran's (Ukraine, Gaza, "
    "Venezuela), with no tie to Iran.\n"
    "- not_war: anything else (immigration, the economy, elections, religion, general praise "
    "or insults).\n"
    "A post on another subject that ties it back to the Iran war (the war's cost in a post "
    'on prices, Iran in a post on Israel) is iran_war. A repost or quote ("RT @name: ...") '
    "counts as the author's own post.\n\n"
    "2. Only if the topic is iran_war: what is the author's position on the war, meaning U.S. "
    "military action against Iran? Score it from -1.0 (against: wants peace, de-escalation, a "
    "deal, or holding off) to +1.0 (for: wants strikes, escalation, or regime change). Score "
    "0 when the post is about the war but takes no position on it (news with no slant, say), "
    "or its position can't be told. Judge the position, not the tone: anger can be for the "
    "war, and warmth can be against it. For any other topic the score is null.\n\n"
    "Respond with ONLY valid JSON: "
    '{"topic": "<iran_war|war_or_peace_general|other_conflict|not_war>", '
    '"score": <float from -1.0 (against the war) to 1.0 (for the war), or null>, '
    '"reasoning": "<one sentence>"}\n\n'
)


def stance_topic_prompt(text: str, user: str | None = None) -> tuple[str, int]:
    """(the stance-topic prompt, the length of its cacheable prefix: the
    instructions). With `user` None, the same words without the sentence
    naming the author (the no-handle arm)."""
    by = f"The post is by @{user}.\n\n" if user is not None else ""
    return STANCE_TOPIC_INSTRUCTIONS + by + f'Post: """{text}"""', len(STANCE_TOPIC_INSTRUCTIONS)


@dataclass(frozen=True)
class PostJob:
    """A post label job. kind "stance" answers a score and a label
    (llm_prompt's JSON); "stance_topic" a topic class and a score that is
    null off the Iran war. posts: "candidates", "candidates+random",
    "subsample" or "census" (every post in the frame). Columns are
    score_<tag><score_suffix> (label_ alike) and, for stance_topic,
    topic_<tag><topic_suffix>. model_setting names the settings entry of its
    default model. in_series: attach may join its columns onto a frame for
    `phases` (a paired subsample may not). shares names the job whose label
    file, columns and report this one writes into, and whose pilot and
    collected batches measure its output (it takes no pilot of its own);
    its batches, batch state and caps stay its own."""

    name: str
    check: str
    prompt_version: str
    kind: str
    posts: str
    cap_setting: str
    usd_cap_setting: str
    score_suffix: str
    topic_suffix: str | None = None
    handle: bool = True
    model_setting: str = "TEACHER_CHECK_MODEL"
    in_series: bool = True
    output_guess: str | None = None
    shares: str | None = None

    @property
    def label_name(self) -> str:
        """The job whose label file (and columns and report) this one uses."""
        return self.shares or self.name

    @property
    def measure_name(self) -> str:
        """The job whose pilot (and, for a sharing job, collected batches)
        measure this one's output."""
        return self.shares or self.name

    def prompt(self, text: str, user: str) -> tuple[str, int]:
        """(prompt, the length of its cached prefix, 0 for none)."""
        if self.kind == "stance":
            from src.analysis.sentiment import llm_prompt

            return llm_prompt(text, user), 0
        return stance_topic_prompt(text, user if self.handle else None)

    def prefix_chars(self) -> int:
        """The cached prefix's length, the same for every post (0: none)."""
        return len(STANCE_TOPIC_INSTRUCTIONS) if self.kind == "stance_topic" else 0

    def default_model(self) -> str:
        return getattr(settings, self.model_setting)

    @property
    def value_cols(self) -> tuple[str, ...]:
        if self.kind == "stance_topic":
            return ("topic", "score_teacher", "label_teacher")
        return ("score_teacher", "label_teacher")

    def columns(self, model: str) -> dict[str, str]:
        """{"score": ..., "label": ..., ["topic": ...]} for `model`."""
        tag = model_tag(model)
        cols = {
            "score": f"score_{tag}{self.score_suffix}",
            "label": f"label_{tag}{self.score_suffix}",
        }
        if self.topic_suffix is not None:
            cols["topic"] = f"topic_{tag}{self.topic_suffix}"
        return cols


JOBS = {
    "stance-topic": PostJob(
        "stance-topic",
        "post_stance_topic",
        STANCE_TOPIC_PROMPT_VERSION,
        "stance_topic",
        "candidates+random",
        "POST_STANCE_TOPIC_MAX_CALLS",
        "POST_STANCE_TOPIC_MAX_USD",
        score_suffix="_stance",
        topic_suffix="",
        output_guess="stance-topic",
    ),
    "rater": PostJob(
        "rater",
        "post_rater",
        sl.TRUMP_PROMPT_VERSION,  # "llm_prompt": relabel's and the Trump check's prompt
        "stance",
        "candidates",
        "POST_RATER_MAX_CALLS",
        "POST_RATER_MAX_USD",
        score_suffix="",
        model_setting="SECOND_RATER_MODEL",
    ),
    "stance-topic-nohandle": PostJob(
        "stance-topic-nohandle",
        "post_stance_topic_nohandle",
        STANCE_TOPIC_NOHANDLE_PROMPT_VERSION,
        "stance_topic",
        "subsample",
        "POST_NOHANDLE_MAX_CALLS",
        "POST_NOHANDLE_MAX_USD",
        score_suffix="_stance_nohandle",
        topic_suffix="_nohandle",
        handle=False,
        in_series=False,
        output_guess="stance-topic",
    ),
    "stance-topic-census": PostJob(
        "stance-topic-census",
        "post_stance_topic_census",
        STANCE_TOPIC_PROMPT_VERSION,
        "stance_topic",
        "census",
        "POST_CENSUS_MAX_CALLS",
        "POST_CENSUS_MAX_USD",
        score_suffix="_stance",
        topic_suffix="",
        output_guess="stance-topic",
        shares="stance-topic",
    ),
}


def sharing_checks(j: PostJob) -> list[str]:
    """The batch checks of the other jobs that write j's label file: while
    one of them has a batch open, j sends nothing (the same post could be
    paid for twice)."""
    return [o.check for o in JOBS.values() if o.name != j.name and o.label_name == j.label_name]


def refuse_sharing_open_batch(j: PostJob, doing: str) -> None:
    """Raises (RuntimeError) while a job sharing j's label file has a batch
    open."""
    for check in sharing_checks(j):
        st = sl.open_batch(check)
        if st:
            raise RuntimeError(sl.open_batch_message(check, st, doing))


def job(name: str) -> PostJob:
    if name not in JOBS:
        raise ValueError(f"no post label job {name!r}: one of {', '.join(JOBS)}")
    return JOBS[name]


def check_model(j: PostJob, model: str) -> None:
    """Raises for a model with no price row, or one whose columns are
    another scorer's: the rater's score_<tag> on claude-opus-5 would be
    score_opus, the stance of record (relabel), and on Haiku score_haiku
    beside score_llm."""
    model_price(model)
    if j.score_suffix == "" and undated(model) in settings.MODEL_TAGS:
        raise ValueError(
            f"{j.name} on {model} would write {j.columns(model)['score']}, which is "
            f"{undated(model)}'s own series (relabel): the rater is a second model "
            f"(settings.SECOND_RATER_MODEL, {settings.SECOND_RATER_MODEL})"
        )


def labels_path(name: str, model: str) -> Path:
    return settings.PROCESSED_DIR / f"post_labels_{name}_{model.replace('/', '_')}.parquet"


def pilot_path(name: str, model: str) -> Path:
    return settings.PROCESSED_DIR / f"post_labels_pilot_{name}_{model.replace('/', '_')}.json"


def report_path(name: str, model: str) -> Path:
    return settings.PROCESSED_DIR / f"post_labels_report_{name}_{model.replace('/', '_')}.json"


def random_manifest_path() -> Path:
    return settings.PROCESSED_DIR / "post_random_draws.parquet"


def subsample_manifest_path() -> Path:
    return settings.PROCESSED_DIR / "post_nohandle_draws.parquet"


# ── The posts ──────────────────────────────────────────────────────


def post_frame() -> pd.DataFrame:
    """Every broadcaster post with text: the X accounts (search rows out)
    and Trump's feed, with source ("x" / "trump"), group (the tier, or
    "trump"), and candidate (inference.war_flag "either"). Raises when an id
    is in both: one batch's custom_ids must not clash."""
    from src.analysis.inference import war_flag
    from src.text_rules import has_text

    x = pd.read_parquet(
        settings.SENTIMENT_OUTPUT, columns=["id", "user", "tier", "created_at", "text"]
    )
    x = x[~x["user"].astype(str).str.startswith("search:")].assign(source="x")
    x["group"] = x["tier"].astype(str)
    frames = [x]
    if settings.TRUMP_FEED_STANCE.exists():
        t = pd.read_parquet(
            settings.TRUMP_FEED_STANCE, columns=["id", "user", "created_at", "text"]
        )
        frames.append(t.assign(source="trump", tier="trump", group="trump"))
    d = pd.concat(frames, ignore_index=True)
    d["id"] = d["id"].astype(str)
    d = d[d["text"].map(has_text).astype(bool)].drop_duplicates(["source", "id"])
    twice = d["id"].duplicated(keep=False)
    if twice.any():
        raise ValueError(
            f"{d.loc[twice, 'id'].nunique()} post ids are in both the X frame and Trump's "
            "feed: one batch's custom_ids would clash"
        )
    d = d.reset_index(drop=True)
    d["candidate"] = war_flag(d, "either").to_numpy(dtype=bool)
    return d


def _smallest_hashes(ids: pd.Series, seed: int, n: int | None) -> pd.Series:
    key = ids.map(lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    order = key.sort_values(kind="stable").index
    return ids.loc[order] if n is None else ids.loc[order[:n]]


def select_random(frame: pd.DataFrame) -> pd.DataFrame:
    """POST_RANDOM_N of the X posts no filter flags: the smallest seeded
    hashes of their ids (a post entering or leaving the pool moves the draw
    only where it hashes in)."""
    pool = frame[(frame["source"] == "x") & ~frame["candidate"]]
    ids = _smallest_hashes(pool["id"], settings.POST_RANDOM_SEED, settings.POST_RANDOM_N)
    return pd.DataFrame(
        {"id": ids.to_numpy(), "tier": pool.loc[ids.index, "group"].to_numpy(), "pool": len(pool)}
    )


def select_subsample(frame: pd.DataFrame) -> pd.DataFrame:
    """The no-handle arm's paired subsample of the X candidates: every post
    of POST_NOHANDLE_ALL_TIERS, the POST_NOHANDLE_PER_TIER smallest seeded
    hashes of each of POST_NOHANDLE_TIERS."""
    pool = frame[(frame["source"] == "x") & frame["candidate"]]
    parts = []
    for tier in (*settings.POST_NOHANDLE_ALL_TIERS, *settings.POST_NOHANDLE_TIERS):
        g = pool[pool["group"] == tier]
        n = None if tier in settings.POST_NOHANDLE_ALL_TIERS else settings.POST_NOHANDLE_PER_TIER
        ids = _smallest_hashes(g["id"], settings.POST_NOHANDLE_SEED, n)
        parts.append(pd.DataFrame({"id": ids.to_numpy(), "tier": tier, "pool": len(g)}))
    return pd.concat(parts, ignore_index=True)


def _manifest(j: PostJob) -> tuple[Path, Callable[[pd.DataFrame], pd.DataFrame]] | None:
    if j.posts == "candidates+random":
        return random_manifest_path(), select_random
    if j.posts == "subsample":
        return subsample_manifest_path(), select_subsample
    return None


def read_manifest(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    m = pd.read_parquet(path)
    m["id"] = m["id"].astype(str)
    return m


def freeze_sets(j: PostJob, frame: pd.DataFrame | None = None) -> dict | None:
    """Record the job's drawn posts (the random arm, or the subsample) in
    their manifest if it is not there yet; a recorded draw is never redrawn.
    None for a job with no draw. Free: no API."""
    spec = _manifest(j)
    if spec is None:
        return None
    path, select = spec
    have = read_manifest(path)
    if have is not None:
        return {"manifest": str(path), "n": len(have), "new": False}
    drawn = select(post_frame() if frame is None else frame)
    drawn = drawn.assign(
        drawn_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        seed=settings.POST_RANDOM_SEED
        if j.posts == "candidates+random"
        else settings.POST_NOHANDLE_SEED,
    )
    sl.write_parquet_atomic(drawn, path)
    logger.info("post labels: %d drawn posts recorded in %s", len(drawn), path)
    return {"manifest": str(path), "n": len(drawn), "new": True}


def job_posts(j: PostJob, frame: pd.DataFrame, preview: bool = False) -> tuple[pd.DataFrame, bool]:
    """(the job's posts with post_set "war" / "random" / "subsample" /
    "rest", whether the drawn part is a preview). A drawn set comes from its
    manifest; with `preview` and no manifest, from the draw freeze_sets
    would record (nothing written). Without either it raises. The census
    takes every post in the frame: the candidates ("war"), the random arm's
    recorded draws ("random", when the manifest exists; it is never drawn
    here) and the rest ("rest")."""
    cand = frame[frame["candidate"]].assign(post_set="war")
    if j.posts == "candidates":
        return cand, False
    if j.posts == "census":
        man = read_manifest(random_manifest_path())
        rnd = set(man["id"]) if man is not None else set()
        sets = np.where(
            frame["candidate"], "war", np.where(frame["id"].isin(rnd), "random", "rest")
        )
        return frame.assign(post_set=sets), False
    path, select = _manifest(j)
    man = read_manifest(path)
    if man is None and not preview:
        raise FileNotFoundError(
            f"no {path.name}: `post-label pilot|submit --job {j.name}` records the draw "
            "before anything is sent"
        )
    is_preview = man is None
    ids = select(frame)["id"] if is_preview else man["id"]
    lost = ~ids.isin(set(frame["id"]))
    if lost.any():
        logger.warning(
            "%s: %d drawn posts are no longer in the frame with text: skipped",
            path.name,
            int(lost.sum()),
        )
    drawn = frame[frame["id"].isin(set(ids))]
    if j.posts == "subsample":
        return drawn.assign(post_set="subsample"), is_preview
    rnd = drawn[~drawn["id"].isin(set(cand["id"]))].assign(post_set="random")
    return pd.concat([cand, rnd], ignore_index=True), is_preview


def _done_ids(cached: pd.DataFrame, posts: pd.DataFrame) -> set[str]:
    """Ids of `posts` the cache labels on the text they have now."""
    return sl.current_label_ids(cached, posts.set_index("id")["text"].str.len())


def job_todo(
    j: PostJob, model: str, params: dict, limit: int | None = None, preview: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """(the job's posts, its cached labels, rows still to label, preview).
    The rows are the posts no label of this job and model holds on the text
    they have now (stance_local.current_label_ids), in a seeded order that
    deals across set, source and tier (a pilot's first N cover them all);
    `limit` keeps the first N. Each row carries its prompt,
    its length (prompt_chars), its cached prefix's (cache_chars: the
    instructions, sent as a cached block) and llm_prompt's on the same post
    (base_chars, for the estimate)."""
    from src.analysis.sentiment import llm_prompt

    check_model(j, model)
    frame = post_frame()
    posts, is_preview = job_posts(j, frame, preview)
    cached = sl.read_label_cache(labels_path(j.label_name, model), j.prompt_version)
    todo = posts[~posts["id"].isin(_done_ids(cached, posts))]
    todo = todo.assign(stratum=todo["post_set"] + "/" + todo["group"])
    todo = sl.stratified_order(todo, "stratum", params.get("seed", settings.TEACHER_SAMPLE_SEED))
    if limit is not None:
        todo = todo.head(limit)
    built = [j.prompt(t, u) for t, u in zip(todo["text"], todo["user"])]
    rows = pd.DataFrame(
        {
            "id": todo["id"].to_numpy(),
            "source": todo["source"].to_numpy(),
            "group": todo["group"].to_numpy(),
            "post_set": todo["post_set"].to_numpy(),
            "input_chars": todo["text"].str.len().to_numpy(),
            "base_chars": [len(llm_prompt(t, u)) for t, u in zip(todo["text"], todo["user"])],
            "prompt_chars": [len(b[0]) for b in built],
            "prompt": [b[0] for b in built],
            "cache_chars": [int(b[1]) for b in built],
            "prompt_version": j.prompt_version,
        }
    )
    return posts, cached, rows, is_preview


# ── Calibrated estimate (no call) ──────────────────────────────────


def predicted_input_tokens(rows: pd.DataFrame) -> np.ndarray:
    """Input tokens a request, before any pilot: llm_prompt's fit on billed
    usage for the post's source on its base_chars, plus the prompt's other
    characters at instruction_chars_per_token (negative for a shorter one)."""
    fit = settings.POST_PROMPT_TOKEN_FIT
    per = fit["instruction_chars_per_token"]
    src = rows["source"].to_numpy()
    tpc = np.array([fit[s]["tokens_per_char"] for s in src], dtype=float)
    fixed = np.array([fit[s]["fixed_tokens"] for s in src], dtype=float)
    base = rows["base_chars"].to_numpy(dtype=float)
    extra = rows["prompt_chars"].to_numpy(dtype=float) - base
    return fixed + tpc * base + extra / per


def read_pilot(name: str, model: str) -> dict | None:
    path = pilot_path(name, model)
    return json.loads(path.read_text()) if path.exists() else None


def pilot_spend(name: str) -> dict:
    """Every pilot run of the job, on any model: {"calls" (sent), "usd"
    (billed, direct prices)}. Counted against the job's run caps. A run
    still marked in_progress (one running, or killed before it finished:
    what it billed after its last checkpoint is unknown) counts at its
    direct ceiling."""
    calls, usd = 0, 0.0
    for path in sorted(settings.PROCESSED_DIR.glob(f"post_labels_pilot_{name}_*.json")):
        rec = json.loads(path.read_text())
        if rec.get("job") != name:
            continue
        for r in rec.get("runs", []):
            calls += int(r.get("calls") or 0)
            got = float(r.get("usd_direct") or 0.0)
            if r.get("in_progress"):
                got = max(got, float(r.get("usd_direct_ceiling") or 0.0))
            usd += got
    return {"calls": calls, "usd": usd}


def pilot_summary(name: str, model: str, max_tokens: int | None = None) -> dict | None:
    """The job's pilot runs on `model` with its current prompt version, at
    `max_tokens` (default settings.POST_LABEL_MAX_TOKENS) and
    settings.POST_LABEL_EFFORT, pooled: calls, answers kept, billed input
    over the estimate's (input_scale), output tokens a call (mean, p95),
    stop reasons. A pilot of another wording, max_tokens or effort measured
    another request and does not count, nor does a run still marked
    in_progress (run_pilot's checkpoint record). None without such a run."""
    j = job(name)
    rec = read_pilot(name, model)
    max_tokens = settings.POST_LABEL_MAX_TOKENS if max_tokens is None else max_tokens
    runs = [
        r
        for r in (rec or {}).get("runs", [])
        if r.get("max_tokens") == max_tokens
        and r.get("prompt_version") == j.prompt_version
        and r.get("effort") == settings.POST_LABEL_EFFORT
        and not r.get("in_progress")
    ]
    outs = [o for r in runs for o in r["output_tokens"]]
    if not outs:
        return None
    stops: dict[str, int] = {}
    for r in runs:
        for k, v in r["stop_reasons"].items():
            stops[k] = stops.get(k, 0) + v
    pred = sum(r["input_tokens_estimated"] for r in runs)
    writes = [w for r in runs for w in r.get("cache_write_tokens", [])]
    reads = [w for r in runs for w in r.get("cache_read_tokens", [])]
    cached = [max(w, x) for w, x in zip(writes, reads) if max(w, x) > 0]
    return {
        "runs": len(runs),
        "calls": sum(r["calls"] for r in runs),
        "billed": len(outs),
        "kept": sum(r["kept"] for r in runs),
        "failed": {
            k: sum(r["failed"].get(k, 0) for r in runs)
            for k in {k for r in runs for k in r["failed"]}
        },
        "input_scale": sum(r["input_tokens_billed"] for r in runs) / pred if pred else 1.0,
        "input_tokens_billed": sum(r["input_tokens_billed"] for r in runs),
        "input_tokens_estimated": pred,
        "input_tokens_per_call": sum(r["input_tokens_billed"] for r in runs) / len(outs),
        "output_mean": float(np.mean(outs)),
        "output_p95": float(np.quantile(outs, 0.95)),
        "output_max": int(max(outs)),
        "stop_reasons": stops,
        "max_tokens": max_tokens,
        "cache_writes": sum(1 for w in writes if w > 0),
        "cache_reads": sum(1 for x in reads if x > 0),
        "cache_engaged": bool(cached),
        "prefix_tokens": float(np.median(cached)) if cached else None,
        "usd_direct": sum(r["usd_direct"] for r in runs),
    }


def measured_pilot(pilot: dict | None) -> bool:
    """Whether a pilot_summary is large enough to stand for the run:
    settings.POST_PILOT_MIN_BILLED billed answers or more."""
    return bool(pilot) and pilot["billed"] >= settings.POST_PILOT_MIN_BILLED


def batch_usage(j: PostJob, model: str) -> dict | None:
    """For a job that shares another's labels (j.shares): that job's
    collected batches on `model` sent with the current request (prompt
    version, max_tokens, effort), pooled: billed calls, billed input tokens
    (plain, cache writes and reads), the estimate's input for the same
    posts (predicted_input_tokens on the label rows the batches wrote, per
    call, times the billed calls), output tokens, the largest batch p95 a
    call and the stop reasons. None for a job that measures itself (its
    pilot alone, as before) or with no such batch. No call."""
    if j.shares is None:
        return None
    m = JOBS[j.measure_name]
    states = [
        st
        for st in sl.batch_states(m.check, _current_request(m, model)).values()
        if (st.get("spend") or {}).get("calls")
    ]
    if not states:
        return None
    calls = sum(int(st["spend"]["calls"]) for st in states)
    billed_in = sum(
        int(st["spend"].get(k) or 0)
        for st in states
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    ids = {str(i) for st in states for i in st.get("ids", [])}
    lab = sl.read_label_cache(labels_path(m.label_name, model), m.prompt_version)
    need = ["source", "base_chars", "prompt_chars"]
    rows = lab[lab["id"].isin(ids)] if set(need) <= set(lab.columns) else lab.iloc[0:0]
    rows = rows.dropna(subset=need) if len(rows) else rows
    pred = float(predicted_input_tokens(rows).mean()) * calls if len(rows) else float("nan")
    stops: dict[str, int] = {}
    for st in states:
        for k, v in (st["spend"].get("stop_reasons") or {}).items():
            stops[k] = stops.get(k, 0) + int(v)
    p95 = [
        float(st["spend"]["output_tokens_p95"])
        for st in states
        if st["spend"].get("output_tokens_p95")
    ]
    return {
        "job": m.name,
        "batches": len(states),
        "calls": calls,
        "input_tokens_billed": billed_in,
        "input_tokens_estimated": pred,
        "input_scale": billed_in / pred if pred and np.isfinite(pred) else float("nan"),
        "cache_read_tokens": sum(
            int(st["spend"].get("cache_read_input_tokens") or 0) for st in states
        ),
        "cache_write_tokens": sum(
            int(st["spend"].get("cache_creation_input_tokens") or 0) for st in states
        ),
        "output_tokens": sum(int(st["spend"].get("output_tokens") or 0) for st in states),
        "output_mean": sum(int(st["spend"].get("output_tokens") or 0) for st in states) / calls,
        "output_p95": max(p95) if p95 else None,
        "stop_reasons": stops,
        "usd_batch": sum(float(st["spend"].get("usd_batch") or 0.0) for st in states),
        "rows_matched": len(rows),
    }


def _shared_basis(j: PostJob, model: str, pilot: dict | None) -> dict | None:
    """A sharing job's output basis: its measuring job's pilot and collected
    batches (batch_usage) pooled, measured once they hold
    POST_PILOT_MIN_BILLED billed answers. Output a call: the pooled mean;
    the ceiling's: the largest p95 (pilot or batch), at least the mean.
    Input scale: billed over the estimate's input, pooled. None when
    neither has a billed answer."""
    usage = batch_usage(j, model)
    parts = [x for x in (pilot, usage) if x]
    billed = (pilot["billed"] if pilot else 0) + (usage["calls"] if usage else 0)
    if not parts or not billed:
        return None
    out = (pilot["output_mean"] * pilot["billed"] if pilot else 0.0) + (
        usage["output_tokens"] if usage else 0.0
    )
    mean = out / billed
    ceil = max([mean, *(x["output_p95"] for x in parts if x.get("output_p95") is not None)])
    est = sum(
        x["input_tokens_estimated"] for x in parts if np.isfinite(x["input_tokens_estimated"])
    )
    got = sum(x["input_tokens_billed"] for x in parts if np.isfinite(x["input_tokens_estimated"]))
    stops: dict[str, int] = {}
    for x in parts:
        for k, v in (x.get("stop_reasons") or {}).items():
            stops[k] = stops.get(k, 0) + int(v)
    what = []
    if pilot:
        what.append(f"{j.measure_name}'s pilot of {pilot['billed']} billed calls")
    if usage:
        what.append(
            f"its {usage['batches']} collected batch{'es' if usage['batches'] > 1 else ''} of "
            f"{usage['calls']} calls"
        )
    return {
        "measured": billed >= settings.POST_PILOT_MIN_BILLED,
        "basis": " and ".join(what),
        "pilot": pilot,
        "batch": usage,
        "stops": stops,
        "input_scale": got / est if est else 1.0,
        "mean": {s: mean for s in ("x", "trump")},
        "ceiling": {s: ceil for s in ("x", "trump")},
    }


def output_basis(j: PostJob, model: str) -> dict:
    """Where the estimate's output (and input scale) comes from: the job's
    pilot on this model at the current max_tokens, once it has
    POST_PILOT_MIN_BILLED billed answers (measured_pilot); for a job that
    shares another's labels (the census), that job's pilot and collected
    batches of the current request pooled (_shared_basis); else, for
    llm_prompt on the fit's model, the fit's measured output (per source);
    else a guess (POST_OUTPUT_TOKENS_GUESS, or the fit's output for
    llm_prompt), twice it for a model other than the fit's. measured says
    whether a full batch may go without a pilot. "pilot" is the pilot
    summary even when it is too small to measure (its stop reasons still
    count); "stops" the stop reasons of every measured answer (pilot, and
    for a sharing job its measuring job's batches)."""
    fit = settings.POST_PROMPT_TOKEN_FIT
    pilot = pilot_summary(j.measure_name, model)
    shared = _shared_basis(j, model, pilot) if j.shares is not None else None
    if shared and shared["measured"]:
        return shared
    small = (
        f" (a pilot of {pilot['billed']} billed calls is under the "
        f"{settings.POST_PILOT_MIN_BILLED} a measurement needs)"
        if pilot and not measured_pilot(pilot)
        else ""
    )
    stops = dict(shared["stops"] if shared else (pilot or {}).get("stop_reasons") or {})
    if measured_pilot(pilot):
        return {
            "measured": True,
            "basis": f"pilot of {pilot['billed']} billed calls",
            "pilot": pilot,
            "stops": stops,
            "input_scale": pilot["input_scale"],
            "mean": {s: pilot["output_mean"] for s in ("x", "trump")},
            "ceiling": {s: max(pilot["output_p95"], pilot["output_mean"]) for s in ("x", "trump")},
        }
    if j.output_guess is None:
        base = {s: fit[s]["output_tokens"] for s in ("x", "trump")}
        p95 = {s: fit[s].get("output_p95_tokens", fit[s]["output_tokens"]) for s in ("x", "trump")}
    else:
        base = {s: settings.POST_OUTPUT_TOKENS_GUESS[j.output_guess] for s in ("x", "trump")}
        p95 = None
    if j.output_guess is None and undated(model) == fit["model"]:
        return {
            "measured": True,
            "basis": f"llm_prompt usage fit on {fit['model']}{small}",
            "pilot": pilot,
            "stops": stops,
            "input_scale": 1.0,
            "mean": base,
            "ceiling": p95,
        }
    mult = 1.0 if undated(model) == fit["model"] else 2.0
    guess = {s: v * mult for s, v in base.items()}
    return {
        "measured": False,
        "basis": ("a guess" + (small or " (no pilot)"))
        + (" at twice the fit model's" if mult > 1 else ""),
        "pilot": pilot,
        "stops": stops,
        "input_scale": 1.0,
        "mean": guess,
        "ceiling": {s: 2 * v for s, v in guess.items()},
    }


def cache_basis(j: PostJob, model: str) -> dict:
    """The job's cached prefix on `model`: its tokens (the pilot's billed
    cache tokens when it measured them, else its characters at v2's
    instruction_chars_per_token), whether the cache engages (the pilot saw
    cache tokens; without a pilot, the prefix reaches the model's cache
    minimum) and the share of requests expected to read it (the 2026-10-05
    v2 batch's, settings.REPLY_V2_BATCH_USAGE)."""
    chars = j.prefix_chars()
    if not chars:
        return {"prefix_tokens": 0.0, "engaged": False, "hit_share": 0.0, "basis": "no prefix"}
    pilot = pilot_summary(j.measure_name, model)
    hit = settings.REPLY_V2_BATCH_USAGE["cache_hit_share"]
    if measured_pilot(pilot):
        if pilot["cache_engaged"]:
            return {
                "prefix_tokens": pilot["prefix_tokens"],
                "engaged": True,
                "hit_share": hit,
                "basis": "measured by the pilot",
            }
        est = chars / settings.POST_PROMPT_TOKEN_FIT["instruction_chars_per_token"]
        return {
            "prefix_tokens": est,
            "engaged": False,
            "hit_share": 0.0,
            "basis": "the pilot saw no cache tokens",
        }
    est = chars / settings.POST_PROMPT_TOKEN_FIT["instruction_chars_per_token"]
    engaged = est >= model_price(model)["cache_min_tokens"]
    return {
        "prefix_tokens": est,
        "engaged": engaged,
        "hit_share": hit if engaged else 0.0,
        "basis": "estimated from its characters (no pilot)",
    }


def calibrated_cost(j: PostJob, rows: pd.DataFrame, model: str) -> dict:
    """The rows (source, base_chars, prompt_chars, post_set) priced at
    `model`'s row of settings.MODEL_PRICES: input tokens from
    predicted_input_tokens times the pilot's input scale, of them the cached
    prefix (cache_basis) read on its hit share and written on the rest where
    the cache engages; output a call from output_basis. usd_batch at batch
    prices. usd_batch_ceiling, the figure the dollar cap is checked against:
    output at the pilot's p95 (the fit's p95; twice the guess unmeasured)
    and no cache hits (every request that sends the prefix as a cached block
    writes it)."""
    p = model_price(model)
    basis = output_basis(j, model)
    cache = cache_basis(j, model)
    src = rows["source"].to_numpy() if len(rows) else np.array([], dtype=object)
    tin = predicted_input_tokens(rows) * basis["input_scale"] if len(rows) else np.zeros(0)
    pre = np.minimum(tin, cache["prefix_tokens"])
    hit = cache["hit_share"]
    mult = (
        hit * p["cache_read_mult"] + (1 - hit) * p["cache_write_mult"] if cache["engaged"] else 1.0
    )
    billed = tin - pre + pre * mult
    ceiling_in = tin - pre + pre * max(1.0, p["cache_write_mult"]) if j.prefix_chars() else tin
    out = np.array([basis["mean"][s] for s in src], dtype=float)
    ceil = np.array([basis["ceiling"][s] for s in src], dtype=float)
    disc = p["batch_discount"]

    def usd(i, o):
        return (float(np.sum(i)) * p["input"] + float(np.sum(o)) * p["output"]) / 1e6

    by_set = {}
    if len(rows):
        keys = rows["post_set"].astype(str) + "/" + rows["source"].astype(str)
        for k in sorted(keys.unique()):
            m = (keys == k).to_numpy()
            by_set[k] = {"calls": int(m.sum()), "usd_batch": usd(billed[m], out[m]) * disc}
    return {
        "model": model,
        "calls": len(rows),
        "input_tokens": round(float(np.sum(tin))),
        "output_tokens": round(float(np.sum(out))),
        "output_tokens_per_call": float(np.mean(out)) if len(out) else float("nan"),
        "output_measured": basis["measured"],
        "output_basis": basis["basis"],
        "input_scale": basis["input_scale"],
        "pilot": basis["pilot"],
        "measure_batch": basis.get("batch"),
        "cache": cache,
        "by_set": by_set,
        "usd_batch": usd(billed, out) * disc,
        "usd_batch_ceiling": usd(ceiling_in, ceil) * disc,
        "usd_direct": usd(billed, out),
        "usd_direct_ceiling": usd(ceiling_in, ceil),
        "price_in_per_mtok": p["input"],
        "price_out_per_mtok": p["output"],
        "batch_discount": disc,
    }


def llm_prompt_cost(prompt_chars, source: str, model: str = settings.TEACHER_CHECK_MODEL) -> dict:
    """llm_prompt requests of these lengths on one source's posts priced on
    the usage fit (the Trump check's and relabel's estimate beside their
    characters / 2 one). No call."""
    chars = np.asarray(list(prompt_chars), dtype=float)
    rows = pd.DataFrame(
        {
            "source": source,
            "base_chars": chars,
            "prompt_chars": chars,
            "post_set": "war",
        }
    )
    return calibrated_cost(JOBS["rater"], rows, model)


def _current_request(j: PostJob, model: str) -> dict:
    """The batch settings that make the job's current request on `model`."""
    return {
        "model": model,
        "prompt_version": j.prompt_version,
        "max_tokens": settings.POST_LABEL_MAX_TOKENS,
        "effort": settings.POST_LABEL_EFFORT,
    }


def unmeasured_batch_calls(j: PostJob, model: str) -> int:
    """Calls the job's batches have sent on `model` with the current prompt
    version, max_tokens and effort: with no pilot of those, all of them went
    out unmeasured."""
    return sl.batch_run_totals(j.check, _current_request(j, model))["calls"]


def submit_problem(j: PostJob, model: str, limit: int | None, n: int) -> str | None:
    """Why a batch of n requests must not go, or None: the model's columns
    are another scorer's; a pilot or a collected batch of the current
    request had an answer stop at max_tokens (any batch, --limit or not: a
    cut answer is not kept, so it would go again and be cut again); or the
    output length is unmeasured and the job's unmeasured batches on this
    model (unmeasured_batch_calls) plus n would pass a pilot's size.
    `limit` is the submit's --limit (kept for the callers; a limit no
    longer opens any gate)."""
    try:
        check_model(j, model)
    except (KeyError, ValueError) as e:
        return str(e)
    basis = output_basis(j, model)
    stops = (basis.get("stops") or {}).get("max_tokens", 0)
    if stops:
        where = f"answers in {basis['basis']}" if j.shares else "pilot answers"
        return (
            f"{stops} {where} stopped at max_tokens ({settings.POST_LABEL_MAX_TOKENS}; "
            "thinking counts against it): raise POST_LABEL_MAX_TOKENS and pilot again"
        )
    cut = sl.batch_run_totals(j.check, _current_request(j, model))["stop_reasons"]
    if cut.get("max_tokens", 0):
        return (
            f"{cut['max_tokens']} answers in earlier {j.name} batches on {model} stopped at "
            f"max_tokens ({settings.POST_LABEL_MAX_TOKENS}; thinking counts against it) and "
            "would be cut again: raise POST_LABEL_MAX_TOKENS and pilot again"
        )
    cap = settings.POST_PILOT_MAX_CALLS
    sent = unmeasured_batch_calls(j, model) if not basis["measured"] else 0
    if not basis["measured"] and sent + n > cap:
        so_far = f" ({sent} sent by earlier batches)" if sent else ""
        return (
            f"{model}'s output length on {j.name} is unmeasured, so the caps would bound a "
            f"guess: run `post-label pilot --job {j.name} --model {model}` first (or send "
            f"--limit batches of {cap} calls or fewer in all{so_far})"
        )
    return None


def job_estimate(
    name: str,
    model: str | None = None,
    limit: int | None = None,
    seed: int = settings.TEACHER_SAMPLE_SEED,
) -> dict:
    """What `post-label submit --job <name>` would send and cost, with no
    call and nothing written (a draw not yet recorded is previewed)."""
    j = job(name)
    model = model or j.default_model()
    posts, cached, todo, preview = job_todo(j, model, {"seed": seed}, limit, preview=True)
    cost = calibrated_cost(j, todo, model)
    sets = posts.groupby(["post_set", "source"]).size()
    batches = sl.batch_run_totals(j.check)["batches"]
    prior = sl.run_spent(j.check)  # every pilot and batch of the job so far
    return {
        "job": j.name,
        "model": model,
        "prompt_version": j.prompt_version,
        "columns": j.columns(model),
        "labels_file": labels_path(j.label_name, model).name,
        "measured_by": j.measure_name if j.shares else None,
        "posts": {f"{a}/{b}": int(v) for (a, b), v in sets.items()},
        "preview": preview,
        "labelled": int(posts["id"].isin(_done_ids(cached, posts)).sum()),
        "todo": len(todo),
        "cap": getattr(settings, j.cap_setting),
        "usd_cap": getattr(settings, j.usd_cap_setting),
        "prior_batches": batches,
        "prior_calls": prior["calls"],
        "prior_usd": prior["usd_batch"],
        "run_calls": prior["calls"] + len(todo),
        "run_usd_ceiling": prior["usd_batch"] + cost["usd_batch_ceiling"],
        "problem": submit_problem(j, model, limit, len(todo)),
        "max_tokens": settings.POST_LABEL_MAX_TOKENS,
        "effort": settings.POST_LABEL_EFFORT,
        **cost,
    }


# ── Answers ────────────────────────────────────────────────────────


def parse_answer(kind: str, text: str) -> tuple[str, dict | None]:
    """(outcome, the label's value columns or None) for one answer text.
    stance: a finite score in [-1, 1] and its label (run_labels' rule).
    stance_topic: a topic of TOPIC_CLASSES; on iran_war a finite score in
    [-1, 1] (a null or missing one is "no_score", not kept); on any other
    topic the score is dropped (NaN) whatever the answer says."""
    from src.analysis.sentiment import parse_llm_json

    data = parse_llm_json(text)
    if not isinstance(data, dict):
        return "unparseable", None
    sc = data.get("score")
    if kind == "stance":
        if not isinstance(sc, (int, float)) or isinstance(sc, bool):
            return "unparseable", None
        if not sl.valid_score(sc):
            return "out_of_range", None
        return "succeeded", {
            "score_teacher": float(sc),
            "label_teacher": sl._label(sc, data.get("label")),
        }
    topic = data.get("topic")
    topic = topic.strip().lower() if isinstance(topic, str) else None
    if topic not in TOPIC_CLASSES:
        return "bad_topic", None
    if topic != IRAN_WAR:
        return "succeeded", {"topic": topic, "score_teacher": float("nan"), "label_teacher": None}
    if sc is None or not isinstance(sc, (int, float)) or isinstance(sc, bool):
        return "no_score", None
    if not sl.valid_score(sc):
        return "out_of_range", None
    return "succeeded", {
        "topic": topic,
        "score_teacher": float(sc),
        "label_teacher": sl._label(sc, data.get("label")),
    }


def _text_of(message) -> str:
    return next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")


def _batch_parse(j: PostJob):
    def parse(result) -> tuple[str, str, dict | None]:
        rid, outcome = str(result.custom_id), result.result.type
        if outcome != "succeeded":
            return rid, outcome, None
        got, vals = parse_answer(j.kind, _text_of(result.result.message))
        return rid, got, vals

    return parse


# ── Direct pilot ───────────────────────────────────────────────────


def _direct_caller(model: str, max_tokens: int, effort: str):
    """(prompt, cache_chars) -> (answer text, billed usage, stop_reason)
    through the direct API, with the request a batch would send for it
    (stance_local.build_batch_requests: the cached-prefix blocks, model,
    max_tokens, effort). Raises with no key before any call."""
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(settings.PROJECT_ROOT / ".env", override=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (.env)")
    client = anthropic.Anthropic()

    def call(prompt: str, cache_chars: int = 0):
        kwargs = {"output_config": {"effort": effort}} if effort else {}
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": sl.prompt_content(prompt, cache_chars)}],
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        tokens = {f: int(getattr(usage, f, 0) or 0) for f in sl.Spend.FIELDS}
        return _text_of(resp), tokens, getattr(resp, "stop_reason", None)

    return call


def run_pilot(
    name: str,
    model: str | None = None,
    n: int = settings.POST_PILOT_N,
    *,
    caller=None,
    seed: int = settings.TEACHER_SAMPLE_SEED,
    concurrency: int = settings.LLM_CONCURRENCY,
) -> dict:
    """Send the first n posts still to do (dealt across set, source and
    tier) through the direct API with the batch's settings, keep the good
    answers in the job's label cache (they are labels like the batch's), and
    add the run to the pilot record: calls, billed input against the
    estimate's, output tokens each call, stop reasons, failures by outcome.
    Refuses over POST_PILOT_MAX_CALLS, while the job's batch is open, and
    when the job's run so far (stance_local.run_spent) plus the pilot would
    pass its call cap, or plus the pilot's direct ceiling its dollar cap.
    Records the job's draw first. Before the first call the run goes into
    the pilot record marked in_progress at its direct ceiling, and every
    POST_PILOT_SAVE_EVERY answers the labels and the running usage are
    saved, so a run killed outright (dkbl1 powers off) keeps what it paid
    for and still counts against the caps (pilot_spend); the finished run
    replaces the mark. `caller` replaces the API in tests. Returns
    pilot_summary."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    j = job(name)
    model = model or j.default_model()
    check_model(j, model)
    if j.shares is not None:
        raise ValueError(
            f"{j.name} takes no pilot: {j.shares}'s pilot and collected batches (the same "
            f"prompt and model) measure its output; `post-label submit --job {j.name} --limit "
            "N` sends a small batch first"
        )
    if n > settings.POST_PILOT_MAX_CALLS:
        raise ValueError(
            f"a pilot of {n} calls is over POST_PILOT_MAX_CALLS ({settings.POST_PILOT_MAX_CALLS})"
        )
    sl.refuse_open_batch(j.check, "a pilot")
    refuse_sharing_open_batch(j, "a pilot")
    freeze_sets(j)
    _, cached, todo, _ = job_todo(j, model, {"seed": seed}, limit=n)
    if todo.empty:
        logger.info("post pilot %s: nothing left to label", j.name)
        return pilot_summary(j.name, model) or {}
    so_far = sl.run_spent(j.check)
    cap, usd_cap = getattr(settings, j.cap_setting), getattr(settings, j.usd_cap_setting)
    if so_far["calls"] + len(todo) > cap:
        raise ValueError(
            f"{j.name}: {so_far['calls']} calls made so far + a pilot of {len(todo)} is over "
            f"{j.cap_setting} ({cap}), which bounds the whole run: raise it deliberately"
        )
    ceiling = calibrated_cost(j, todo, model)["usd_direct_ceiling"]
    if so_far["usd_batch"] + ceiling > usd_cap:
        raise ValueError(
            f"{j.name}: ${so_far['usd_batch']:.2f} spent so far + the pilot's ${ceiling:.2f} "
            f"direct ceiling is over {j.usd_cap_setting} (${usd_cap:g}), which bounds the "
            "whole run: raise it deliberately"
        )
    max_tokens, effort = settings.POST_LABEL_MAX_TOKENS, settings.POST_LABEL_EFFORT
    caller = caller or _direct_caller(model, max_tokens, effort)
    pred = predicted_input_tokens(todo)
    keep_cols = [c for c in todo.columns if c not in ("prompt", "cache_chars")]
    records = todo[keep_cols].to_dict("records")
    rows, outs, writes, reads, billed_in, billed_pred = [], [], [], [], 0, 0.0
    stops: dict[str, int] = {}
    failed: dict[str, int] = {}
    spend = sl.Spend(model)
    out = labels_path(j.label_name, model)
    run_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    run_id = uuid.uuid4().hex  # this run's entry in the pilot record

    def save():
        sl.merge_new_labels(cached, pd.DataFrame(rows, columns=keep_cols + list(j.value_cols)), out)

    def run_record(done: bool) -> dict:
        r = {
            "run_id": run_id,
            "at": run_at,
            "prompt_version": j.prompt_version,
            "max_tokens": max_tokens,
            "effort": effort,
            "calls": len(todo),
            "kept": len(rows),
            "failed": dict(failed),
            "stop_reasons": dict(stops),
            "input_tokens_billed": int(billed_in),
            "input_tokens_estimated": billed_pred,
            "output_tokens": list(outs),
            "cache_write_tokens": list(writes),
            "cache_read_tokens": list(reads),
            "usd_direct": spend.usd(),
        }
        if not done:
            r.update(in_progress=True, usd_direct_ceiling=ceiling)
        return r

    def write_run(r: dict | None) -> None:
        """Put this run's entry in the pilot record (None: take it out)."""
        rec = read_pilot(j.name, model) or {"job": j.name, "model": model, "runs": []}
        rec["runs"] = [x for x in rec["runs"] if x.get("run_id") != run_id]
        if r is not None:
            rec["runs"].append(r)
        sl._write_json(pilot_path(j.name, model), rec)

    write_run(run_record(done=False))  # before any call: a killed run still counts
    ex = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futs = {
            ex.submit(caller, p, int(c)): i
            for i, (p, c) in enumerate(zip(todo["prompt"], todo["cache_chars"]))
        }
        for f in as_completed(futs):
            i = futs[f]
            try:
                text, usage, stop = f.result()
            except Exception as e:  # noqa: BLE001 -- one bad call must not sink the pilot
                logger.warning("pilot call failed for %s: %s", records[i]["id"], str(e)[:120])
                failed["error"] = failed.get("error", 0) + 1
                continue
            spend.add(SimpleNamespace(**usage))
            billed_in += sum(
                usage.get(k, 0)
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            billed_pred += float(pred[i])
            outs.append(int(usage.get("output_tokens", 0)))
            writes.append(int(usage.get("cache_creation_input_tokens", 0)))
            reads.append(int(usage.get("cache_read_input_tokens", 0)))
            stops[str(stop)] = stops.get(str(stop), 0) + 1
            outcome, vals = parse_answer(j.kind, text)
            if vals is None:
                failed[outcome] = failed.get(outcome, 0) + 1
            else:
                rows.append({**records[i], **vals})
            if len(outs) % settings.POST_PILOT_SAVE_EVERY == 0:  # checkpoint
                save()
                write_run(run_record(done=False))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        save()
        write_run(run_record(done=True) if outs else None)  # nothing billed: no run
        logger.info("post pilot %s: %s", j.name, spend.summary())
    return pilot_summary(j.name, model) or {}


# ── Batch checks (stance_local's transport) ────────────────────────


def _check(j: PostJob) -> sl.BatchCheck:
    def todo(model, params, limit):
        _, cached, rows, _ = job_todo(j, model, params, limit)
        return cached, rows

    def guard(model, params, limit, rows):
        refuse_sharing_open_batch(j, "a new batch")
        why = submit_problem(j, model, limit, len(rows))
        if why:
            raise RuntimeError(why)

    return sl.BatchCheck(
        j.prompt_version,
        j.cap_setting,
        lambda m: labels_path(j.label_name, m),
        todo,
        lambda m, p: job_report(j.name, m),
        parse=_batch_parse(j),
        value_cols=j.value_cols,
        usd_cap_setting=j.usd_cap_setting,
        price_rows=lambda rows, m: calibrated_cost(j, rows, m)["usd_batch_ceiling"],
        run_caps=True,  # every batch and pilot of the job counts against its caps
        submit_guard=guard,
        extra_spend=lambda: pilot_spend(j.name),
    )


POST_BATCH_CHECKS = {j.check: _check(j) for j in JOBS.values()}


# ── Columns on a frame (read time) ─────────────────────────────────


def column_owner(col: str) -> tuple[PostJob, str] | None:
    """(job, model) whose labels are column `col`, or None. Pairs the job
    refuses (check_model: the rater on claude-opus-5 or Haiku, whose
    score_<tag> is relabel's series) own nothing: score_opus is never a
    post-label column. A job that writes another's label file (the census)
    owns nothing: its columns are that job's."""
    for j in JOBS.values():
        if j.shares is not None:
            continue
        for m in settings.MODEL_PRICES:
            try:
                check_model(j, m)
            except (KeyError, ValueError):
                continue
            if col in j.columns(m).values():
                return j, m
    return None


def _labels_by_id(j: PostJob, model: str) -> pd.DataFrame:
    path = labels_path(j.label_name, model)
    if not path.exists():
        raise LookupError(
            f"no {j.name} labels from {model} yet ({path.name}): `post-label submit --job "
            f"{j.name} --model {model}`, then collect"
        )
    lab = sl.read_label_cache(path, j.prompt_version)
    return lab.drop_duplicates("id").set_index("id")


def _current(d: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    """Per row of d: whether `lab` (by id) holds a label made from the text
    the row has now (its input_chars equal the text's length; a label with
    no input_chars counts as current, as stance_local.current_label_ids
    has it)."""
    ids = d["id"].astype(str)
    held = ids.isin(lab.index)
    if "input_chars" not in lab:
        return held
    had = ids.map(lab["input_chars"]).astype(float)
    now = d["text"].fillna("").astype(str).str.len().astype(float)
    return held & (had.isna() | (had == now))


def attach(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """df plus the columns of the job and model that `col` names
    (column_owner), from its label file, on every post the file labels (any
    set: candidates, the random arm, the census) and only where the post's
    text has the length it was labelled with. Raises LookupError for a
    column no job writes, a job whose posts are no series (the paired
    subsample) or labels not made yet; ValueError when the frame has one of
    the columns already (never overwritten)."""
    owner = column_owner(col)
    if owner is None:
        raise LookupError(f"{col} is not in the frame and no post-label job writes it")
    j, model = owner
    if not j.in_series:
        raise LookupError(
            f"{col} is {j.name}'s, a paired subsample for its report (`post-label report "
            f"--job {j.name}`), not a series"
        )
    cols = j.columns(model)
    clash = [c for c in cols.values() if c in df.columns]
    if clash:
        raise ValueError(f"the frame already has {', '.join(clash)}: attach never overwrites one")
    lab = _labels_by_id(j, model)
    d = df.copy()
    ids = d["id"].astype(str)
    keep = _current(d, lab)
    stale = ids.isin(lab.index) & ~keep
    if stale.any():
        logger.warning(
            "%s: %d labels were made from another text of the post (since changed): left out",
            col,
            int(stale.sum()),
        )
    src = {"score": "score_teacher", "label": "label_teacher", "topic": "topic"}
    for kind, c in cols.items():
        d[c] = ids.map(lab[src[kind]]).where(keep)
    return d


def opus_topic(d: pd.DataFrame, model: str | None = None) -> pd.Series:
    """Per row of d: Opus's topic class from the stance-topic label file
    (every set: candidates, random arm, census), NaN where the row has no
    label made from its text now. Raises FileNotFoundError with no label
    file: the flag of record cannot be applied at all."""
    j = JOBS["stance-topic"]
    model = model or j.default_model()
    try:
        lab = _labels_by_id(j, model)
    except LookupError as e:
        raise FileNotFoundError(
            f"{e}; the census (`post-label submit --job stance-topic-census`) labels every "
            "other post. The topic flag of record (settings.TOPIC_SOURCE opus) needs them: "
            "`--topic either` reads the posts without"
        ) from None
    return d["id"].astype(str).map(lab["topic"]).where(_current(d, lab))


def opus_war_flag(d: pd.DataFrame, model: str | None = None) -> pd.Series:
    """The about-the-war flag of record (inference.war_flag "opus",
    settings.TOPIC_SOURCE): a post is a war post when its current
    stance-topic label says iran_war (opus_topic). A post with no current
    label (not labelled yet, or labelled from another text) counts as the
    "either" flag counts it, with a warning that counts the posts with text
    (X search rows, which no job labels, and rows with no text, which leave
    every denominator, fall back silently)."""
    from src.analysis.inference import war_flag
    from src.text_rules import has_text

    model = model or JOBS["stance-topic"].default_model()
    topic = opus_topic(d, model)
    flag = topic.eq(IRAN_WAR)
    missing = topic.isna()
    if missing.any():
        flag[missing] = war_flag(d[missing], "either").to_numpy(dtype=bool)
        posts = d["text"].map(has_text).astype(bool)  # what a job labels: text, no search row
        if "user" in d:
            posts &= ~d["user"].astype(str).str.startswith("search:")
        counted = missing & posts
        if counted.any():
            logger.warning(
                "topic opus: %d of %d posts with text have no current stance-topic label from "
                "%s: counted by the either flag (%d of them flagged). The census labels them: "
                "`post-label estimate --job stance-topic-census`",
                int(counted.sum()),
                int(posts.sum()),
                model,
                int(flag[counted].sum()),
            )
    return flag.astype(bool)


# ── Reports (aggregates only) ──────────────────────────────────────


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rater_stats(ref, new) -> dict:
    """`new` against `ref` on the rows with both: stance_local.agreement
    (n, Pearson, MAE, sign agreement and flips) plus Spearman, both means
    and Bland-Altman (mean and SD of new - ref, 95% limits of agreement)."""
    r = np.asarray(ref, dtype=float)
    x = np.asarray(new, dtype=float)
    ok = ~(np.isnan(r) | np.isnan(x))
    r, x = r[ok], x[ok]
    out = sl.agreement(r, x)
    diff = x - r
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    md = float(diff.mean()) if len(diff) else float("nan")
    with np.errstate(invalid="ignore"):
        out.update(
            spearman=_spearman(r, x),
            mean_ref=float(r.mean()) if len(r) else float("nan"),
            mean_new=float(x.mean()) if len(x) else float("nan"),
            mean_diff=md,
            sd_diff=sd,
            loa_lo=md - 1.96 * sd,
            loa_hi=md + 1.96 * sd,
            sign_flips=int(np.sum(sl.sign3(r) * sl.sign3(x) < 0)),
        )
    return out


def rater_table(d: pd.DataFrame, ref: str, new: str, by: str) -> list[dict]:
    """rater_stats per value of `by` (phases in timeline order), then ALL."""
    from src.analysis.inference import PHASE_ORDER

    keys = sorted(map(str, d[by].dropna().unique()))
    if by == "phase":
        rank = {p: i for i, p in enumerate([*PHASE_ORDER, OUTSIDE])}
        keys.sort(key=lambda k: rank.get(k, len(rank)))
    col = d[by].astype(str)
    rows = [{by: k, **rater_stats(d.loc[col == k, ref], d.loc[col == k, new])} for k in keys]
    rows.append({by: "ALL", **rater_stats(d[ref], d[new])})
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(share, lo, hi): the Wilson 95% interval of k in n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def reference_scores() -> pd.Series:
    """score_opus by id: the X frame's (relabel) and, for Trump's feed, the
    Trump check's labels (the same model and prompt)."""
    parts = []
    x = pd.read_parquet(settings.SENTIMENT_OUTPUT)
    if "score_opus" in x:
        parts.append(pd.Series(x["score_opus"].to_numpy(), index=x["id"].astype(str)))
    tpath = sl.trump_labels_path(settings.TEACHER_CHECK_MODEL)
    if tpath.exists():
        t = sl.read_label_cache(tpath).dropna(subset=["score_teacher"]).drop_duplicates("id")
        parts.append(pd.Series(t["score_teacher"].to_numpy(), index=t["id"].astype(str)))
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    return s[~s.index.duplicated()]


def _labelled(j: PostJob, model: str, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The job's labels on the posts they were made from, with the post's
    source, group, phase, candidate flag and post_set as labelled; labels on
    a text since changed (or a post gone) are dropped."""
    from src.analysis.inference import assign_phase

    lab = sl.read_label_cache(labels_path(j.label_name, model), j.prompt_version)
    lab = lab.drop_duplicates("id")
    if lab.empty:  # no labels yet (no pilot, or no batch collected)
        cols = ["source", "group", "created_at", "candidate", "phase"]
        return pd.DataFrame(columns=list(dict.fromkeys([*lab.columns, *cols])))
    frame = post_frame() if frame is None else frame
    f = frame[["id", "source", "group", "created_at", "text", "candidate"]]
    d = lab.drop(columns=[c for c in ("source", "group") if c in lab]).merge(f, on="id")
    if "input_chars" in d:  # a label with no length recorded counts as current
        d = d[d["input_chars"].isna() | (d["input_chars"] == d["text"].str.len())]
    d = d.copy()
    d["phase"] = assign_phase(d["created_at"]).fillna(OUTSIDE)
    return d.drop(columns=["text"])


def _coverage(j: PostJob, model: str, frame: pd.DataFrame, d: pd.DataFrame) -> dict:
    """How much of the job's post set holds a current label, and how the
    job's batches and pilots on this model failed (by outcome) and stopped:
    agreement on the posts the model could answer may not hold on the rest."""
    posts, _ = job_posts(j, frame, preview=True)
    runs = sl.batch_run_totals(j.check, {"model": model})
    pilot = pilot_summary(j.name, model) or {}
    failed = dict(runs["failed_by_outcome"])
    for k, v in (pilot.get("failed") or {}).items():
        failed[k] = failed.get(k, 0) + v
    stops = dict(runs["stop_reasons"])
    for k, v in (pilot.get("stop_reasons") or {}).items():
        stops[k] = stops.get(k, 0) + v
    return {
        "posts": len(posts),
        "labelled": int(posts["id"].isin(set(d["id"])).sum()),
        "batches": runs["batches"],
        "failed_by_outcome": failed,
        "stop_reasons": stops,
    }


def _census_coverage(frame: pd.DataFrame, d: pd.DataFrame) -> list[dict]:
    """Per source and ALL: the posts with text in the frame (post_frame),
    how many carry a current stance-topic label (any set) and of those how
    many Opus classes iran_war; the posts the flag of record (TOPIC_SOURCE
    "opus") still reads by the either flag are the unlabelled ones."""
    from src.analysis.inference import assign_phase

    lab = d.drop_duplicates("id").set_index("id")["topic"]
    rows = []
    for src, g in [*frame.groupby("source"), ("ALL", frame)]:
        topic = g["id"].map(lab)
        inph = assign_phase(g["created_at"]).notna()
        rows.append(
            {
                "source": str(src),
                "posts": len(g),
                "in_phases": int(inph.sum()),
                "labelled": int(topic.notna().sum()),
                "unlabelled_in_phases": int((topic.isna() & inph).sum()),
                "iran_war": int(topic.eq(IRAN_WAR).sum()),
                "iran_war_not_candidate": int((topic.eq(IRAN_WAR) & ~g["candidate"]).sum()),
            }
        )
    return rows


def _stance_topic_report(j: PostJob, model: str) -> dict:
    """Topic classes by set and group; on the candidates, the topic against
    which filter flagged the post; the random arm's Iran-war share (what
    both filters miss) and what it implies for the unflagged pool; the
    score on Iran-war candidates against score_opus by group and phase."""
    from src.analysis.inference import on_topic
    from src.analysis.topic_label import load_labels

    frame = post_frame()
    d = _labelled(j, model, frame)
    rep = {"job": j.name, "model": model, "prompt_version": j.prompt_version, "n": len(d)}
    if d.empty:
        return rep
    rep["coverage"] = _coverage(j, model, frame, d)
    rep["census"] = _census_coverage(frame, d)
    rep["topics"] = {
        f"{s}/{g}": {str(t): int(c) for t, c in grp["topic"].value_counts().items()}
        for (s, g), grp in d.groupby(["post_set", "group"])
    }
    cand = d[d["post_set"] == "war"].merge(frame[["id", "text"]], on="id")
    haiku = load_labels()
    hmap = haiku.set_index("id")["about_war"] if haiku is not None else pd.Series(dtype=object)
    kw = on_topic(cand["text"]).to_numpy()
    hk = cand["id"].map(hmap)
    flag = np.where(
        hk.isna(),
        "keyword (no Haiku label)",
        np.where(kw & hk.eq(True), "both", np.where(kw, "keyword only", "Haiku only")),
    )
    cand = cand.assign(flag=flag)
    rep["filters_vs_topic"] = [
        {
            "flag": str(f),
            "n": len(g),
            **{t: int((g["topic"] == t).sum()) for t in TOPIC_CLASSES},
            "iran_war_share": float((g["topic"] == IRAN_WAR).mean()),
        }
        for f, g in cand.groupby("flag")
    ]
    rnd = d[d["post_set"] == "random"]
    if len(rnd):
        pool = frame[(frame["source"] == "x") & ~frame["candidate"]]
        rows = []
        for tier, g in [*rnd.groupby("group"), ("ALL", rnd)]:
            k = int((g["topic"] == IRAN_WAR).sum())
            share, lo, hi = wilson(k, len(g))
            size = len(pool) if tier == "ALL" else int((pool["group"] == tier).sum())
            rows.append(
                {
                    "tier": str(tier),
                    "n": len(g),
                    "iran_war": k,
                    "share": share,
                    "share_lo": lo,
                    "share_hi": hi,
                    "pool_now": size,
                    "implied_missed": share * size,
                    "implied_missed_lo": lo * size,
                    "implied_missed_hi": hi * size,
                }
            )
        rep["random_arm"] = rows
    ref = reference_scores()
    war = cand[cand["topic"] == IRAN_WAR].assign(score_opus=lambda x: x["id"].map(ref))
    rep["vs_score_opus_on_iran_war"] = {
        "by_group": rater_table(war, "score_opus", "score_teacher", "group"),
        "by_phase": rater_table(war, "score_opus", "score_teacher", "phase"),
    }
    off = cand[cand["topic"] != IRAN_WAR].assign(score_opus=lambda x: x["id"].map(ref))
    rep["score_opus_on_other_topics"] = {
        str(t): {"n": len(g), "mean_score_opus": float(g["score_opus"].mean())}
        for t, g in off.groupby("topic")
    }
    return rep


def _rater_report(j: PostJob, model: str) -> dict:
    """The rater's labels against score_opus on the same posts: by group
    and by phase (Pearson, Spearman, MAE, sign flips, Bland-Altman)."""
    frame = post_frame()
    d = _labelled(j, model, frame)
    rep = {"job": j.name, "model": model, "prompt_version": j.prompt_version, "n": len(d)}
    if d.empty:
        return rep
    rep["coverage"] = _coverage(j, model, frame, d)
    ref = reference_scores()
    d = d[d["candidate"]].assign(score_opus=lambda x: x["id"].map(ref))
    rep["vs_score_opus"] = {
        "by_group": rater_table(d, "score_opus", "score_teacher", "group"),
        "by_phase": rater_table(d, "score_opus", "score_teacher", "phase"),
    }
    return rep


def _paired_row(g: pd.DataFrame, w: pd.Series | None = None) -> dict:
    """One row of the no-handle report on the pairs g: how often the topic
    class moves, the Iran-war share under each, and on the posts both call
    iran_war the mean shift (no handle - handle) with its 95% interval,
    Pearson and sign flips. With per-post weights `w` (the ALL row: pool /
    pairs of the post's tier) the shares, means, shift and Pearson are
    weighted, and the interval is the linearized variance of that weighted
    ratio, stratified by tier: with e = 1{both iran_war} (shift - the
    weighted shift) on every pair, sum over tiers of n_t var_t(w e) / D^2,
    D the weight on posts both call iran_war. It counts that which posts
    both call iran_war is itself random (a tier whose shift differs from
    the rest widens it). Each tier's pairs are taken as a sample of that
    tier's, with no finite-population correction, as the per-tier rows do:
    Opus's answers are draws too (they vary on retest), so a tier taken
    whole still carries error. pairs, n_both_iran_war and sign_flips stay
    counts."""
    both = (g["topic_handle"] == IRAN_WAR) & (g["topic_nohandle"] == IRAN_WAR)
    b = g[both]
    diff = b["score_teacher_nohandle"] - b["score_teacher_handle"]
    n = len(diff)
    st = rater_stats(b["score_teacher_handle"], b["score_teacher_nohandle"])
    if w is None:
        sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
        md = float(diff.mean()) if n else float("nan")
        half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
        moved = float((g["topic_handle"] != g["topic_nohandle"]).mean())
        war_h = float((g["topic_handle"] == IRAN_WAR).mean())
        war_n = float((g["topic_nohandle"] == IRAN_WAR).mean())
        mean_h, mean_n, pearson = st["mean_ref"], st["mean_new"], st["pearson"]
    else:
        w = w.reindex(g.index).astype(float)
        wb = w[both]

        def wmean(x, ww):
            return float(np.average(x, weights=ww)) if len(x) and ww.sum() > 0 else float("nan")

        moved = wmean(g["topic_handle"] != g["topic_nohandle"], w)
        war_h = wmean(g["topic_handle"] == IRAN_WAR, w)
        war_n = wmean(g["topic_nohandle"] == IRAN_WAR, w)
        md = wmean(diff, wb)
        mean_h = wmean(b["score_teacher_handle"], wb)
        mean_n = wmean(b["score_teacher_nohandle"], wb)
        z = pd.Series(0.0, index=g.index)  # w e: zero on pairs not both iran_war
        z[both] = (w[both] * (diff - md)).to_numpy()
        total, var = float(wb.sum()), 0.0
        for _, zt in z.groupby(g["group"]):
            v = float(zt.var(ddof=1)) if len(zt) > 1 else float(z.var(ddof=1))
            var += len(zt) * v
        var = var / total**2 if total else float("nan")
        sd = float(np.sqrt(np.cov(diff, aweights=wb))) if n > 1 else float("nan")
        half = 1.96 * math.sqrt(var) if n > 1 else float("nan")
        pearson = float("nan")
        if n > 2:
            c = np.cov(b["score_teacher_handle"], b["score_teacher_nohandle"], aweights=wb)
            if c[0, 0] * c[1, 1] > 0:
                pearson = float(c[0, 1] / math.sqrt(c[0, 0] * c[1, 1]))
    return {
        "pairs": len(g),
        "topic_moved": moved,
        "iran_war_handle": war_h,
        "iran_war_nohandle": war_n,
        "n_both_iran_war": n,
        "mean_handle": mean_h,
        "mean_nohandle": mean_n,
        "diff": md,
        "diff_lo": md - half,
        "diff_hi": md + half,
        "sd_diff": sd,
        "pearson": pearson,
        "sign_flips": st["sign_flips"],
        "weighted": w is not None,
    }


def _nohandle_report(j: PostJob, model: str) -> dict:
    """Paired: the no-handle labels against stance-topic's (with the handle,
    same model) on the same posts, by tier (_paired_row). The ALL row
    weights each pair by its tier's pool over the tier's pairs (the
    manifest's pool: the X candidates of the tier when drawn), so it
    estimates the shift over the candidate set, not over a subsample that
    takes religious and opposition whole and 400 of each larger tier; its
    interval is the weighted ratio's linearized one (_paired_row).
    Subsample posts stance-topic has not labelled yet are counted."""
    nh = _labelled(j, model)
    rep = {"job": j.name, "model": model, "prompt_version": j.prompt_version, "n": len(nh)}
    hpath = labels_path("stance-topic", model)
    if nh.empty or not hpath.exists():
        rep["note"] = "needs stance-topic labels from the same model on the subsample"
        return rep
    h = sl.read_label_cache(hpath, STANCE_TOPIC_PROMPT_VERSION).drop_duplicates("id")
    pairs = nh.merge(
        h[["id", "topic", "score_teacher", "input_chars"]],
        on="id",
        suffixes=("_nohandle", "_handle"),
    )
    pairs = pairs[pairs["input_chars_nohandle"] == pairs["input_chars_handle"]]
    rep["unpaired"] = int(len(nh) - len(pairs))
    rows = [{"tier": str(tier), **_paired_row(g)} for tier, g in pairs.groupby("group")]
    man = read_manifest(subsample_manifest_path())
    if man is not None and len(pairs):
        pool = man.groupby("tier")["pool"].first()
        w = pairs["group"].map(pool / pairs.groupby("group").size())
        rows.append({"tier": "ALL", **_paired_row(pairs, w.fillna(0.0))})
        rep["all_weights"] = "each tier's pool / its pairs"
    else:
        rows.append({"tier": "ALL", **_paired_row(pairs)})
        rep["all_weights"] = "none (no manifest): the pairs pooled as drawn"
    rep["paired_by_tier"] = rows
    return rep


def job_report(name: str, model: str | None = None) -> dict:
    """The job's report, written to post_labels_report_<job>_<model>.json.
    A job that writes another's label file (the census) gets that job's
    report, over every label in the file, written under that job's name."""
    j = job(name)
    model = model or j.default_model()
    j = JOBS[j.label_name]
    rep = {
        "stance-topic": _stance_topic_report,
        "rater": _rater_report,
        "stance-topic-nohandle": _nohandle_report,
    }[j.name](j, model)
    sl._write_json(report_path(j.name, model), rep)
    return rep


def _agree_line(r: dict, key: str) -> str:
    return (
        f"  {r[key]:22s}{r['n']:>6d}  pearson {r['pearson']:+.3f}  spearman {r['spearman']:+.3f}  "
        f"mae {r['mae']:.3f}  flips {r['sign_flips']:>3d}  diff {r['mean_diff']:+.3f} "
        f"[LoA {r['loa_lo']:+.2f}, {r['loa_hi']:+.2f}]"
    )


def summary_lines(rep: dict) -> list[str]:
    """Printable lines of a job report (aggregates only)."""
    lines = [f"{rep['job']} labels from {rep['model']} ({rep['prompt_version']}): {rep['n']}"]
    if rep.get("note"):
        lines.append(rep["note"])
    c = rep.get("coverage")
    if c:
        lines.append(
            f"  coverage: {c['labelled']} of {c['posts']} posts labelled; {c['batches']} batches; "
            f"failed by outcome {c['failed_by_outcome'] or 0}; stop reasons "
            f"{c['stop_reasons'] or 'n/a'}"
        )
    for r in rep.get("census", []):
        lines.append(
            f"  census, {r['source']:6s} {r['labelled']:>6d} of {r['posts']} posts with text "
            f"labelled ({r['unlabelled_in_phases']} in the phases without one: the flag of "
            f"record reads those by the either flag); iran_war {r['iran_war']}, of them "
            f"{r['iran_war_not_candidate']} no filter flags"
        )
    for r in rep.get("filters_vs_topic", []):
        lines.append(
            f"  candidates, {r['flag']:26s}{r['n']:>6d}  iran_war {r['iran_war_share']:.0%}  "
            + "  ".join(f"{t} {r[t]}" for t in TOPIC_CLASSES[1:])
        )
    for r in rep.get("random_arm", []):
        lines.append(
            f"  random, {r['tier']:22s}{r['n']:>5d}  iran_war {r['iran_war']:>3d} "
            f"({r['share']:.1%} [{r['share_lo']:.1%}, {r['share_hi']:.1%}]); of {r['pool_now']} "
            f"unflagged now ≈ {r['implied_missed']:.0f} [{r['implied_missed_lo']:.0f}, "
            f"{r['implied_missed_hi']:.0f}] missed"
        )
    vs = rep.get("vs_score_opus_on_iran_war") or rep.get("vs_score_opus")
    if vs:
        lines.append("against score_opus (new - score_opus):")
        lines += [_agree_line(r, "group") for r in vs["by_group"]]
        lines += [_agree_line(r, "phase") for r in vs["by_phase"]]
    if rep.get("paired_by_tier"):
        lines.append(f"  no handle - handle by tier (ALL weighted: {rep.get('all_weights')}):")
    for r in rep.get("paired_by_tier", []):
        lines.append(
            f"  {r['tier']:22s}{r['pairs']:>5d} pairs  topic moved {r['topic_moved']:.0%}  iran_war "
            f"{r['iran_war_handle']:.0%} -> {r['iran_war_nohandle']:.0%}  on {r['n_both_iran_war']} "
            f"both: {r['mean_handle']:+.3f} -> {r['mean_nohandle']:+.3f}, diff {r['diff']:+.3f} "
            f"[{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]  flips {r['sign_flips']}"
        )
    return lines
