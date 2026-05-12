# iran-sentiment

Sentiment analysis experiment on the 2026 Iran war. Tracks how Trump
administration messaging, the MAGA influencer ecosystem, the
opposition, and the Pope Leo XIV / Vatican moral axis shift over the
Feb-Apr 2026 conflict window, including the MAGA split between pro-
and anti-war factions.

## Architecture

Everything runs through a single Click CLI -- no one-off scripts.
Adding accounts, search terms, caps, or colors is a config-file edit;
never hard-code them in scripts.

```
config/
  settings.py      # paths, budget caps, batch sizes, tier colors, model names
  accounts.py      # X/Truth Social handles organized by tier
  timeline.py      # ~36 key events for event-overlay plots
src/
  cli.py                              # single entrypoint -- all commands live here
  collectors/x_collector.py           # per-account JSONL caching + incremental fetch
  collectors/truthsocial_collector.py # truthbrush + public Mastodon API
  analysis/sentiment.py               # VADER + RoBERTa + optional Claude LLM stance
  visualization/plots.py              # timeline / tier_comparison / heatmap / search
data/
  raw/x/<handle>.jsonl       # one file per account; incremental-appended on refresh
  raw/x/search_<query>.jsonl
  processed/sentiment_all.parquet
  processed/figures/*.png
```

## Tiers

`config/accounts.py` organizes handles into political tiers. Current:
admin, maga_prowar, maga_antiwar, opposition, media, religious_authority,
plus `search` (keyword-query public-sentiment proxy). The
`religious_authority` tier (Pontifex / USCCB / VaticanNews) exists
because Pope Leo XIV became the single loudest anti-war voice in the
dataset — and because sentiment models sign-flip his rhetoric. See
`feedback_religious_rhetoric_stance.md` in auto-memory for the failure
mode.

## Running things

Always activate the venv first: `source .venv/bin/activate`. Then:

```bash
python -m src.cli test       # verify X API credentials (free)
python -m src.cli status     # show what's cached, what's missing
python -m src.cli collect    # incremental fetch — appends only new tweets since last run
python -m src.cli analyze    # score all cached data (VADER + RoBERTa; add --llm for stance)
python -m src.cli visualize  # regenerate all figures (incl. score_llm-based ones)
python -m src.cli summary    # print stats tables (use --score score_llm for true stance)
python -m src.cli run-all    # full pipeline
```

`collect` is incremental at the per-account level: each rerun fetches
only tweets newer than the latest cached `created_at` and appends. To
fetch a brand-new account, add it to `config/accounts.py` and rerun —
the existing accounts are cheap to bring up to date. Pass `--force` to
re-fetch a whole account's window from scratch.

For LLM stance scoring, filter by tier or handle so tokens track the
subset that actually needs it:

```bash
# Score only the religious tier (required — RoBERTa sign-flips these)
python -m src.cli analyze --llm --llm-tiers religious_authority

# Score specific handles
python -m src.cli analyze --llm --llm-accounts WhiteHouse,POTUS

# Full dataset (~$12 on Haiku 4.5, ~30 min with concurrency=5)
python -m src.cli analyze --llm
```

`analyze --llm` is restart-safe: posts that already have `score_llm`
are skipped, a partial run periodically saves progress, and a full
rerun picks up from the prior `sentiment_all.parquet`.

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

## Memory & CPU discipline (sentiment analysis)

The mini PC has powered off twice during `analyze` runs — once from
OOM on unbatched RoBERTa, once from thermal trip on unthrottled CPU
inference. Guardrails in `src/analysis/sentiment.py` + `config/settings.py`:

1. **Batched inference** via `ROBERTA_BATCH_SIZE` (16, lowered from 32
   after the second incident) — never score tweets one-by-one.
2. **Thread cap** via `TORCH_NUM_THREADS` (2) — PyTorch saturates all
   cores by default; capping prevents thermal trip on fanless hardware.
   Set before importing the pipeline.
3. **Mid-phase checkpointing** every `ROBERTA_CHECKPOINT_EVERY_N_BATCHES`
   (25) — a crash mid-RoBERTa only loses up to 25 batches of work.
4. **RSS ceiling** via `ROBERTA_MAX_RSS_MB` (6144) — in-loop check
   aborts cleanly (with checkpoint write) before the kernel OOM-kills.
5. **Resume path** — on startup, restore all scores from prior
   `sentiment_all.parquet` by post id, plus RoBERTa checkpoint if
   present. Only truly-new posts get re-scored. Re-running `analyze`
   after a crash is a no-brainer recovery.
6. **Cheap-refresh path for LLM** — `score_llm` skips posts with an
   existing score; periodic save every `LLM_SAVE_EVERY_N` (100)
   completions bounds the crash-loss of spent tokens.
7. **Explicit model cleanup** — `del pipe; gc.collect()` in a
   `try/finally` so aborts still release memory.

Baseline on this hardware: RoBERTa peaks at ~1.36 GB RSS on 9,477
posts, runtime ~10 min. If these numbers drift substantially, something
is leaking or the thread cap got removed.

## Scoring strategies

Three scorers live in `sentiment.py`, run in order of cost/accuracy:

- **VADER** — rule-based, always on. Fast but naive; flags anti-war
  rhetoric as positive because words like "peace", "diplomacy",
  "humanity" are lexically positive (see @SenSanders at VADER +0.110
  despite being a loud anti-war voice).
- **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) —
  Twitter-tuned sentiment. Better than VADER in general, but has a
  systematic positive-bias on stance: institutional positive-valenced
  language (religious, bureaucratic, victorious) reads as pro-war.
- **Claude Haiku stance** (`--llm`) — context-aware JSON scoring
  from −1.0 (anti-war) to +1.0 (pro-war). The source of truth for
  stance; VADER and RoBERTa are valence proxies.

**Known miscalibrations when stance is what matters:**
- VADER confuses "we destroyed their nuclear facility" (triumphant
  admin framing) with negative lexical.
- RoBERTa sign-flips faith-based anti-war voices: @Pontifex at RoBERTa
  +0.364 vs LLM −0.496 (full sign flip). The `religious_authority`
  tier must use `score_llm`; the RoBERTa number is actively misleading.
- RoBERTa underrates pro-war admin messaging (@WhiteHouse RoBERTa
  +0.364 vs LLM +0.067) and overrates negative-valenced pro-war
  voices (@LauraLoomer RoBERTa −0.335 vs LLM −0.007).

Rule of thumb: use `score_transformer` for quick looks and `score_llm`
for anything published or compared across tiers.

## Python environment

Project uses a venv at `.venv/`. The system Python doesn't have pip
installed, and `python3-venv` had to be installed manually via apt.
Activate with `source .venv/bin/activate` before running anything.
