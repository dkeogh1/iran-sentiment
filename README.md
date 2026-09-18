# iran-sentiment

Sentiment analysis of US political messaging during the 2026 Iran war
(Feb 1 -- May 12, 2026). Tracks the Trump administration, MAGA
influencers (both pro- and anti-war), the opposition, media, and the
Pope Leo XIV / Vatican moral axis across X and Truth Social, from a
month before the Feb 28 strikes through the Apr 8 ceasefire and three
weeks of the post-ceasefire negotiation period.

The X broadcaster data ends 2026-05-12 (accounts through May 10,
keyword searches through May 12). The Truth Social layer -- Trump's
feed and the reply-level audience analysis -- runs through 2026-09-16.
Nothing runs on a schedule.

## Findings

13,974 posts from 17 X/Twitter accounts across 6 political tiers plus
4 keyword searches (a public-sentiment proxy), plus 83,054 Truth Social
replies to 6 Trump posts (three from the April escalation, three from
the May-September deal-and-collapse cycle) and Trump's own Truth Social
feed from Apr 4 to Sep 16 (4,087 posts). Every post carries three scores: VADER
(lexicon baseline), RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`,
valence), and a Claude Haiku stance score from -1.0 (anti-war) to +1.0
(pro-war). **Stance numbers below are the LLM score**; the RoBERTa
column is shown to make its miscalibration visible.

### Tier divergence

![Tier comparison](docs/figures/tier_comparison_score_llm.png)

| Tier | LLM stance | RoBERTa | n |
|------|-----------:|--------:|--:|
| admin | +0.236 | +0.136 | 2,604 |
| maga_prowar | -0.058 | -0.206 | 4,676 |
| media | -0.098 | -0.035 | 908 |
| religious_authority | -0.207 | **+0.296** | 1,176 |
| maga_antiwar | -0.209 | -0.189 | 2,718 |
| search (public) | -0.326 | -0.346 | 1,768 |
| opposition | -0.344 | -0.293 | 124 |

The administration is the only net-pro-war tier. Everyone else is
net-negative, including the "pro-war" MAGA influencers: Levin and
Loomer are hawkish in framing but critical in tone, and land near zero
on stance. The MAGA split shows up as a 0.15-point gap between the
pro-war and anti-war influencer tiers.

### The religious sign flip

RoBERTa reads the Vatican tier as the *most positive* tier in the
dataset (+0.296) because faith-based anti-war language ("peace",
"mercy", "dialogue") is lexically positive. The LLM stance score puts
the same posts at -0.207, and @Pontifex at **-0.443 -- the single most
anti-war account in the dataset** (RoBERTa: +0.374). Of 1,176
religious-tier posts, zero were labelled pro-war. This tier is the
reason the project moved to LLM stance scoring; see the *Sentiment
scoring* section.

### Who the hawks actually are

The operational arms out-hawk the PR shop. @PeteHegseth (+0.327) and
@StateDept (+0.293) are the two most pro-war accounts; @POTUS is
+0.143 and @WhiteHouse only +0.071, even though RoBERTa had the White
House as the most positive account in the dataset (+0.380). @VP, the
lead negotiator from the Apr 11 Islamabad talks, sits at +0.044.

### After the ceasefire (Apr 22 -- May 12)

| Tier | Feb 1 -- Apr 21 | Apr 22 -- May 12 |
|------|---------------:|----------------:|
| admin | +0.235 | +0.239 |
| maga_prowar | -0.044 | -0.107 |
| maga_antiwar | -0.231 | -0.150 |
| religious_authority | -0.227 | -0.136 |
| opposition | -0.352 | -0.310 |
| search (public) | -0.307 | -0.364 |

Administration messaging did not move. The anti-war influencers and
the Vatican softened once the shooting stopped, the pro-war
influencers hardened (blockade and negotiation criticism), and the
keyword searches got more negative. The weekly series shows its most
negative week at May 11 (-0.37), but that bucket is search-only (the
accounts were last refreshed May 10) and should not be read as a
cross-tier shift.

### Per-account detail

![Account heatmap](docs/figures/account_heatmap_score_llm.png)

- @SenSanders is the most anti-war non-Vatican voice (-0.344). VADER
  scores him positive because anti-war vocabulary is lexically
  positive.
- @LauraLoomer moves from RoBERTa -0.290 to LLM -0.080: her angry
  pro-war posts read as negative valence but near-neutral stance.
- @TuckerCarlson posts rarely (104 tweets) but is consistently
  anti-war from the strikes onward (-0.264).
- @mtgreenee is the most anti-war MAGA voice (-0.317).

### Audience replies: six Trump posts, April to September

Every direct reply to six Trump Truth Social posts (83,054 replies,
97-99% of each post's reply count) scored with RoBERTa, plus a
stratified sample of 992 replies (50 per sentiment bucket per post)
classified by Claude Haiku into stance categories.

| Post | Date | Replies | RoBERTa mean | Critical | Supportive |
|---|---|--:|--:|--:|--:|
| "Power Plant Day" rant | Apr 5 | 23,656 | -0.225 | 54% | 24% |
| "Whole civilisation will die" | Apr 7 | 16,591 | -0.342 | 59% | 17% |
| Two-week ceasefire | Apr 7 | 16,808 | -0.303 | 58% | 19% |
| "Hold off on our planned Military attack" | May 18 | 8,800 | **-0.475** | **69%** | 11% |
| "The Deal with Iran is now complete" | Jun 14 | 12,609 | **+0.193** | 33% | **50%** |
| "Striking Iranian Targets near Hormuz" | Sep 1 | 4,590 | -0.246 | 53% | 21% |

- **Restraint was punished harder than escalation.** The May 18 post
  announcing a called-off attack is the most negative post in the
  dataset, and the only one where the most loyal accounts are the most
  negative (high-loyalty tier -0.51 vs low-loyalty -0.47; every other
  post has loyalists as the least negative). The Haiku sample shows why:
  it has the highest share of `pro_war_critical` replies of any post
  (12.7%), people angry that Trump did not strike.
- **The deal was the only thing the audience liked.** June 14 is the
  sole net-positive post, with the steepest loyalty gradient (+0.05 low
  to +0.40 high). Even so, a third of the sampled replies are anti-war
  in some form (betrayal 10.7%, opposition 15.3%, pro-Trump-anti-war
  10.0%).
- **Renewed strikes re-consolidated the base.** The September 1 strikes
  post has the highest `pro_war_supportive` share in the sample (56%)
  and the lowest betrayal share (4.7%), while its population-level
  RoBERTa mean matches the April "Power Plant Day" post.
- **The within-MAGA "betrayal" voice persists at 5-12% across all six
  posts** ("voted 3x for you, losing me as a supporter"), peaking on
  the April rant and the June deal, not on the strikes.

Stance shares are from a bucket-balanced sample, so they describe the
spectrum of each post's replies, not population proportions; the
RoBERTa columns are the population numbers. RoBERTa's blind spots are
the same as on the broadcaster data: `pro_war_critical` replies read as
negative (-0.55) and `pro_trump_antiwar` replies read as positive
(+0.22).

## Methodology

### Data collection

- X/Twitter: v2 API, per-account JSONL caching, incremental refresh
  (each run appends only tweets newer than the latest cached one),
  capped at 500 tweets per account per run to control cost
  ($0.005/read).
- Keyword searches use `/search/recent`, which only reaches back 7
  days. Searches were refreshed on Apr 16 and May 12, so search
  coverage has a gap from roughly Apr 20 to May 5; the collector logs
  the unfetched gap rather than silently skipping it.
- Truth Social: `curl_cffi` (Cloudflare bypass). Account feeds via the
  public API with an incremental, resume-safe refresh; replies via the
  authenticated v2 descendants endpoint after a one-time `ts-login`
  (Truth Social's new-device security-code check).
- Window: Feb 1 -- May 12, 2026, with pre-war context events back to
  Jun 2025.

### Sentiment scoring

Three scorers, in order of cost:

1. VADER -- rule-based lexicon baseline. Can't distinguish "we
   destroyed their nuclear facility" (triumphant) from "destroyed"
   (negative lexical). Included to show why lexicon-based sentiment
   fails on war rhetoric.
2. RoBERTa -- `cardiffnlp/twitter-roberta-base-sentiment-latest`,
   fine-tuned on Twitter text. A valence signal, not a stance signal:
   it has a systematic positive bias on institutional language and
   sign-flips the religious tier. Batched CPU inference with memory
   checkpointing (the pipeline runs on a fanless mini PC).
3. Claude Haiku stance -- JSON-scored stance from -1.0 (anti-war) to
   +1.0 (pro-war), with an `off_topic` pre-filter. Run over the full
   dataset (100% coverage); the source of truth for every stance
   number in this README. Also used for the 500-reply stance sample.

### Event overlay

84 events are catalogued in `config/timeline.py` (military strikes,
diplomatic moments, polling, media events) and overlaid on time-series
plots. The first 36 cover the pre-war build-up through Apr 15; 48 more,
added Sep 2026 from a web-research pass with independent fact-checking,
cover May 12 -- Sep 15 (the Islamabad MOU of Jun 17, its collapse Jul 8,
expiry Aug 17, and the renewed strikes of September). Each event carries
an importance score; `EVENT_LABEL_MIN_IMPORTANCE` in settings controls
which get labelled on plots. Source URLs for the new events are in
`docs/timeline_candidates_2026-05-12_to_2026-09-15.json`. Data currently
ends May 12, so the later events are not yet plotted.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Credentials live in `secrets.env`, committed encrypted with sops + age
(homelab-infra `docs/secrets.md`; the private key is on dkbl1). Decrypt it
into the working `.env` the code reads:

```bash
sops -d secrets.env > .env          # on dkbl1, SOPS_AGE_KEY_FILE=~/.config/homelab-infra/age.key
sops secrets.env                    # edit a value in place; then re-run the line above
sops -e .env > secrets.env          # or edit .env and re-encrypt; commit secrets.env
```

Without the key, copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
# Edit .env with your X API bearer token and (optionally)
# Truth Social credentials and Anthropic API key
```

## Usage

Everything runs through a single CLI:

```bash
python -m src.cli test          # verify X API credentials
python -m src.cli status        # show what's cached vs. missing
python -m src.cli collect       # incremental X fetch (appends new tweets only)
python -m src.cli collect-truth # incremental Truth Social feed refresh
python -m src.cli ts-login      # one-time login (security code) -> token in .env
python -m src.cli collect-replies  # fetch replies to tracked Trump posts (needs token)
python -m src.cli analyze       # score all cached data (VADER + RoBERTa)
python -m src.cli analyze --llm # add Claude stance scores (restart-safe)
python -m src.cli visualize     # regenerate all figures
python -m src.cli summary --score score_llm   # stance tables
python -m src.cli event-study   # reply sentiment + event-window analysis
python -m src.cli run-all       # full pipeline
```

`collect` is incremental at the per-account level: each rerun fetches
only tweets newer than the latest cached `created_at` and appends.
Pass `--force` to re-fetch an account's whole window. To add accounts
or search terms, edit `config/accounts.py` and rerun. `analyze --llm`
skips posts that already have a stance score, so re-running after a
collect only spends tokens on new posts.

## Project structure

```
config/
  settings.py          # paths, budget caps, batch sizes, tier colors
  accounts.py          # X/Truth Social handles organized by tier
  timeline.py          # 84 key events (Jun 2025 - Sep 2026) for plot overlays
  tracked_posts.py     # specific Trump posts for reply analysis
src/
  cli.py               # Click CLI -- single entrypoint
  collectors/
    x_collector.py     # X API v2, per-account JSONL cache, incremental fetch
    truthsocial_collector.py  # Truth Social API + curl_cffi
  analysis/
    sentiment.py       # VADER + RoBERTa + Claude stance scoring
    event_study.py     # reply sentiment + event-window comparisons
  visualization/
    plots.py           # timeline, tier comparison, heatmap, search plots
data/                     # gitignored -- not included in repo
  raw/x/*.jsonl           # per-account tweet caches
  raw/truthsocial/*.jsonl # Truth Social posts + replies
  processed/*.parquet     # scored sentiment data
  processed/figures/*.png # generated plots
docs/figures/             # the LLM-stance figures referenced above
```

## Limitations

- 17 X accounts across 6 tiers is illustrative, not representative.
  The opposition tier is a single account (124 posts).
- VADER and RoBERTa are valence proxies. Any stance comparison across
  tiers must use `score_llm`; the RoBERTa number for the religious
  tier is actively misleading.
- The 500-tweet-per-run cap was hit by @marklevinshow, @RealAlexJones
  and @LauraLoomer in the May 10 refresh, so those accounts are
  truncated within Apr 22 -- May 10.
- Search coverage has a gap in late April (see *Data collection*), and
  the final week of the series is search-only.
- Truth Social's API may truncate large reply trees. Coverage is
  validated per-post during collection.
- This is observational sentiment tracking, not causal inference. Event
  overlays show correlation, not causation.

## Cost

X API reads cost $0.005/tweet. Collection through Apr 20 (9,477
tweets) cost about $47; the May 10/12 refresh added roughly 4,500
tweets, about $22 more. LLM stance scoring of the full dataset ran
about $15 on Claude Haiku through April; the May 12 pass over the
~4,500 new posts was about $2 (672k input + 267k output tokens on
Haiku 4.5, from the console usage page). Truth Social API access is
free.
