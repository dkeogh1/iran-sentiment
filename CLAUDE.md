# iran-sentiment

Sentiment analysis experiment on the 2026 Iran war. Tracks how Trump
administration messaging and the MAGA influencer ecosystem's sentiment
shifts over the Feb-Apr 2026 conflict window, including the split
between pro- and anti-war factions.

## Architecture

Everything runs through a single Click CLI -- no one-off scripts.
Adding accounts, search terms, caps, or colors is a config-file edit;
never hard-code them in scripts.

```
config/
  settings.py      # paths, budget caps, batch sizes, tier colors, model names
  accounts.py      # X/Truth Social handles organized by tier
  timeline.py      # 24 key events for event-overlay plots
src/
  cli.py                              # single entrypoint -- all commands live here
  collectors/x_collector.py           # per-account JSONL caching + budget caps
  collectors/truthsocial_collector.py # truthbrush + public Mastodon API
  analysis/sentiment.py               # VADER + RoBERTa + optional Claude LLM
  visualization/plots.py              # timeline / tier_comparison / heatmap / search
data/
  raw/x/<handle>.jsonl       # one file per account; skip if exists unless --force
  raw/x/search_<query>.jsonl
  processed/sentiment_all.parquet
  processed/figures/*.png
```

## Running things

Always activate the venv first: `source .venv/bin/activate`. Then:

```bash
python -m src.cli test       # verify X API credentials (free)
python -m src.cli status     # show what's cached, what's missing
python -m src.cli collect    # collect all configured accounts (respects cache)
python -m src.cli analyze    # score all cached data
python -m src.cli visualize  # regenerate all figures
python -m src.cli summary    # print stats tables
python -m src.cli run-all    # full pipeline
```

`collect` is incremental -- it skips any account that already has a
cached JSONL file. Pass `--force` to re-fetch. To add a new account,
edit `config/accounts.py` and rerun `collect` -- only the new account
will be hit.

## Budget discipline (X API)

X is pay-as-you-go at $0.005 per tweet read. A single prolific account
can blow the budget (@marklevinshow alone was 2,544 tweets = ~$13).
Always:

1. Respect `settings.MAX_TWEETS_PER_USER` (currently 500) as a hard cap
2. Estimate cost before running: `num_accounts * MAX_TWEETS_PER_USER * $0.005`
3. Check `data/raw/x/` first to see what's already cached -- never
   re-fetch cached data without `--force`
4. For keyword searches, `/search/recent` only covers ~7 days, so
   historical searches require Pro tier ($$$)

## Memory discipline (sentiment analysis)

The RoBERTa transformer will OOM a small machine if run naively on
thousands of tweets -- this has already crashed the mini PC once.
Guardrails in `src/analysis/sentiment.py`:

1. Batched inference via `ROBERTA_BATCH_SIZE` (32) -- never score
   tweets one-by-one in a Python loop
2. Checkpoint after VADER -- if RoBERTa crashes mid-run, the VADER
   checkpoint is on disk; a fresh `analyze` run will reload it and skip
   straight to RoBERTa
3. Explicit model cleanup -- `del pipe; gc.collect()` after use
4. Memory monitoring -- `_log_mem()` at phase boundaries

Peak memory for the current dataset (~5,800 tweets) is ~760 MB. Watch
this if the dataset grows substantially.

## Scoring strategies

Three scorers live in `sentiment.py`, run in order of cost/accuracy:

- VADER -- rule-based, always on. Fast but naive; flags anti-war
  rhetoric as positive because words like "peace", "diplomacy",
  "humanity" are lexically positive (see @SenSanders scoring +0.110
  despite being the loudest anti-war voice).
- RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) --
  the main signal, tuned on Twitter-style text.
- Claude Haiku (optional, opt-in with `--llm`) -- context-aware,
  best for the war-framing problem VADER can't handle. Costs tokens.

Known VADER weakness: it can't distinguish "we destroyed their nuclear
facility" (triumphant/positive admin framing) from "destroyed" (negative
lexical). Use RoBERTa as the primary signal; VADER is a baseline only.

## Python environment

Project uses a venv at `.venv/`. The system Python doesn't have pip
installed, and `python3-venv` had to be installed manually via apt.
Activate with `source .venv/bin/activate` before running anything.
