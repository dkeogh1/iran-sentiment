"""
CLI entrypoint for the iran-sentiment pipeline.

Commands:
  test            Verify X API credentials with a tiny request
  collect         Collect tweets from all configured X accounts + searches
  collect-truth   Collect Truth Social posts from configured accounts
  collect-replies Collect Truth Social replies for tracked Trump posts
  analyze         Run sentiment scoring on cached raw data
  visualize       Generate all figures from scored data
  summary         Print stats tables (by account, tier, and weekly trend)
  event-study     Per-post reply stats + event-window broadcaster deltas
  stance          LLM stance classification on a stratified reply sample
  status          Report what's been collected and what's missing
  run-all         Full pipeline: collect → analyze → visualize → summary
"""

import logging
import sys

import click
import pandas as pd

from config import settings
from config.accounts import X_ACCOUNTS, SEARCH_TERMS, TRUTH_SOCIAL_ACCOUNTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("iran-sentiment")


@click.group()
def main():
    """Iran War Sentiment Analysis Pipeline."""
    settings.ensure_dirs()


# ── test ────────────────────────────────────────────────────────────

@main.command()
def test():
    """Sanity-check the X API credentials."""
    from src.collectors.x_collector import get_client

    client = get_client()
    user = client.get_user(username="POTUS")
    if user.data:
        click.secho(f"✓ API OK — @POTUS resolves to id={user.data.id}", fg="green")
    else:
        click.secho("✗ API reachable but @POTUS lookup failed", fg="red")
        sys.exit(1)


# ── collect ─────────────────────────────────────────────────────────

@main.command()
@click.option("--force", is_flag=True, help="Ignore cached files and re-fetch")
@click.option("--no-search", is_flag=True, help="Skip keyword searches")
@click.option("--estimate", is_flag=True,
              help="Print the per-account plan and maximum cost, then exit (no API calls)")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt")
def collect(force: bool, no_search: bool, estimate: bool, yes: bool):
    """Collect tweets from all configured X accounts (caches per-account).

    Incremental: each account fetches only tweets newer than its cache,
    walked oldest-slice-first so a cap hit thins a slice instead of
    dropping months. The plan below is a MAXIMUM -- accounts that posted
    less than their cap cost less.
    """
    from src.collectors.x_collector import collect_all, estimate_run

    plan = estimate_run(X_ACCOUNTS, None if no_search else SEARCH_TERMS, force=force)

    click.echo(f"\n{'account':<18}{'cached':>7}{'from':>12}{'slices':>7}{'cap':>6}{'max $':>8}")
    for p in plan["accounts"]:
        if p["skip_reason"]:
            click.echo(f"  @{p['handle']:<16}{p['cached']:>7}  {p['skip_reason']}")
            continue
        click.echo(f"  @{p['handle']:<16}{p['cached']:>7}{p['window'][0].date().isoformat():>12}"
                   f"{len(p['slices']):>7}{p['cap']:>6}{p['max_cost_usd']:>8.2f}")
    for s in plan["searches"]:
        click.echo(f"  search:{s['query']:<28}{'7d':>12}{1:>7}{s['max_reads']:>6}{s['max_cost_usd']:>8.2f}")
    click.echo(f"\nMaximum this run: {plan['max_reads']} reads ≈ ${plan['max_cost_usd']:.2f}"
               f"  (budget settings.X_RUN_BUDGET_USD = ${settings.X_RUN_BUDGET_USD:.2f})")

    if estimate:
        return

    if plan["max_cost_usd"] > settings.X_RUN_BUDGET_USD:
        click.secho(
            f"Refusing: plan maximum ${plan['max_cost_usd']:.2f} exceeds X_RUN_BUDGET_USD "
            f"${settings.X_RUN_BUDGET_USD:.2f}. Lower MAX_TWEETS_PER_USER / add "
            f"ACCOUNT_CAP_OVERRIDES, trim config/accounts.py, use --no-search, or raise "
            f"the budget deliberately in config/settings.py.",
            fg="red")
        raise SystemExit(1)

    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    summary = collect_all(
        X_ACCOUNTS,
        search_terms=None if no_search else SEARCH_TERMS,
        force=force,
    )

    total = sum(summary.values())
    click.echo(f"\nCollected {total} total tweets across {len(summary)} sources:")
    for name, count in sorted(summary.items(), key=lambda x: -x[1]):
        click.echo(f"  {name}: {count}")


# ── ts-login ────────────────────────────────────────────────────────

@main.command("ts-login")
@click.option("--deliver", type=click.Choice(["email", "sms"]), default=None,
              help="After a new-device challenge, ask for the code via this method")
@click.option("--challenge-id", default=None, help="challenge_id printed by an earlier run")
@click.option("--code", default=None, help="The security code you received")
def ts_login_cmd(deliver: str | None, challenge_id: str | None, code: str | None):
    """Log in to Truth Social once and save the token to .env.

    Handles the new-device security-code check:

      ts-login                      -> token saved, or prints challenge_id + methods
      ts-login --deliver email      -> asks the server to send the code (prints reply)
      ts-login --challenge-id X --code 123456 -> verifies, saves TRUTHSOCIAL_TOKEN
    """
    import os
    from dotenv import load_dotenv
    from src.collectors.truthsocial_collector import (
        SecurityCodeRequired, request_token, request_security_code_delivery,
        verify_security_code, save_token_to_env,
    )
    load_dotenv(override=True)
    username = os.environ.get("TRUTHSOCIAL_USERNAME") or os.environ.get("TRUTH_SOCIAL_USERNAME")
    password = os.environ.get("TRUTHSOCIAL_PASSWORD") or os.environ.get("TRUTH_SOCIAL_PASSWORD")
    if not username or not password:
        click.secho("Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD in .env first.", fg="red")
        sys.exit(1)

    if challenge_id and code:
        token = verify_security_code(username, password, challenge_id, code)
        path = save_token_to_env(token)
        click.secho(f"Verified. TRUTHSOCIAL_TOKEN saved to {path} (starts {token[:8]}...)", fg="green")
        click.echo("Re-encrypt secrets.env if you keep .env under sops.")
        return

    click.echo(f"Authenticating as @{username}...")
    try:
        token = request_token(username, password)
    except SecurityCodeRequired as ch:
        click.secho("New-device security code required.", fg="yellow")
        click.echo(f"  challenge_id: {ch.challenge_id}")
        for m in ch.delivery_methods:
            click.echo(f"  method: {m.get('kind')}  -> {m.get('value')}")
        if not deliver:
            click.echo("\nNext: python -m src.cli ts-login --deliver email   (or sms)")
            return
        resp = request_security_code_delivery(username, password, ch.challenge_id, deliver)
        click.echo(f"\nDelivery request -> HTTP {resp.status_code}: {resp.text[:400]}")
        if resp.status_code == 200:
            click.echo(f"\nWhen the code arrives:\n  python -m src.cli ts-login "
                       f"--challenge-id {ch.challenge_id} --code <CODE>")
        else:
            click.secho(
                "\nThe server rejected the delivery request. The shape mirrors the "
                "web app's sign-in modal (settings.py, security-code flow); if Truth "
                "Social changed it, capture the /oauth/v2/choose_delivery_method call "
                "from DevTools > Network on truthsocial.com and compare.", fg="yellow")
        return
    path = save_token_to_env(token)
    click.secho(f"Logged in. TRUTHSOCIAL_TOKEN saved to {path} (starts {token[:8]}...)", fg="green")


# ── probe-auth ──────────────────────────────────────────────────────

@main.command("probe-auth")
@click.option("--post-id", default="116363336033995961",
              help="Truth Social post ID to probe (default: 'civilisation_dies')")
def probe_auth_cmd(post_id: str):
    """One-shot test: authenticate to Truth Social and fetch replies for one post."""
    import os
    from dotenv import load_dotenv
    from curl_cffi import requests as cffi_requests

    load_dotenv(override=True)

    ts_base = "https://truthsocial.com"
    api_base = f"{ts_base}/api/v1"

    username = os.environ.get("TRUTH_SOCIAL_USERNAME")
    password = os.environ.get("TRUTH_SOCIAL_PASSWORD")
    if not username or not password:
        click.secho(
            "Set TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD in .env first.",
            fg="red",
        )
        sys.exit(1)

    saved = os.environ.get("TRUTHSOCIAL_TOKEN")
    if saved:
        click.echo("Using TRUTHSOCIAL_TOKEN from .env (run ts-login to refresh)")
    click.echo(f"Authenticating as @{username}...")

    # Use the same client creds and endpoint that truthbrush uses —
    # see settings.TRUTH_SOCIAL_WEB_CLIENT_ID / SECRET. These are public
    # web-app values (not user secrets); the v2 endpoint + JSON body +
    # the bundled client ID is the combination that actually works
    # (earlier attempts with /oauth/token + form-encoded data +
    # self-registered app all returned 403).
    r = None if saved else cffi_requests.post(
        f"{ts_base}/oauth/v2/token",
        json={
            "client_id": settings.TRUTH_SOCIAL_WEB_CLIENT_ID,
            "client_secret": settings.TRUTH_SOCIAL_WEB_CLIENT_SECRET,
            "grant_type": "password",
            "username": username,
            "password": password,
            "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
            "scope": "read",
        },
        impersonate="chrome",
    )
    if r is not None and r.status_code != 200:
        click.secho(f"Token exchange failed: {r.status_code} {r.text[:300]}", fg="red")
        if "security_code_required" in r.text:
            click.echo("Run: python -m src.cli ts-login")
        sys.exit(1)
    token = saved or r.json()["access_token"]
    click.secho(f"  Bearer token acquired (starts {token[:12]}...)", fg="green")

    # Step 3: fetch /context for the target post, authenticated
    # Match truthbrush's exact request shape: chrome136 impersonation +
    # explicit User-Agent header. Cloudflare is pickier on /context than
    # on the timeline endpoints.
    auth_headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/136.0.0.0 Safari/537.36",
    }
    # Truth Social extends the standard Mastodon API with a paginated
    # /context/descendants endpoint (the standard /context is blocked by
    # Cloudflare). truthbrush uses this with Link-header pagination.
    click.echo(f"\nFetching /context/descendants for post {post_id}...")
    descendants = []
    next_url = f"{ts_base}{settings.TS_DESCENDANTS_PATH.format(id=post_id)}"
    page = 0
    while next_url:
        r = cffi_requests.get(
            next_url,
            params={"sort": "oldest"} if page == 0 else None,
            headers=auth_headers,
            impersonate="chrome136",
        )
        click.echo(f"  Page {page}: HTTP {r.status_code}")
        if r.status_code != 200:
            click.secho(f"  Failed: {r.text[:300]}", fg="red")
            if page == 0:
                sys.exit(1)
            break

        batch = r.json()
        if not batch:
            break
        descendants.extend(batch)
        click.echo(f"    +{len(batch)} replies (total: {len(descendants)})")

        # Follow Link: <url>; rel="next" pagination
        next_url = None
        link_header = r.headers.get("Link", "")
        for link in link_header.split(","):
            parts = link.split(";")
            if len(parts) == 2 and parts[1].strip() == 'rel="next"':
                next_url = parts[0].strip().strip("<>")
                break

        page += 1
        # Safety cap for the probe — don't paginate forever
        if page >= 3:
            click.echo("  (stopping after 3 pages for the probe)")
            break

    click.secho(f"  Total descendants fetched: {len(descendants)}", fg="green")

    # Step 4: compare to the reported replies_count
    r2 = cffi_requests.get(
        f"{api_base}/statuses/{post_id}",
        headers=auth_headers,
        impersonate="chrome136",
    )
    if r2.status_code == 200:
        reported = r2.json().get("replies_count", "?")
        pct = (len(descendants) / reported * 100) if isinstance(reported, int) and reported else 0
        click.echo(f"  Post's replies_count metric: {reported}")
        click.echo(f"  Coverage: {pct:.1f}%")
        if isinstance(reported, int) and len(descendants) < reported * 0.5:
            click.secho(
                "  ⚠ Likely truncated — /context is capping the response.",
                fg="yellow",
            )
        elif len(descendants) > 0:
            click.secho("  ✓ Looks good — reply tree is intact or nearly so.", fg="green")

    # Step 5: show a sample reply
    if descendants:
        sample = descendants[0]
        acct = sample.get("account", {})
        click.echo("\n  Sample reply:")
        click.echo(f"    @{acct.get('username', '?')} ({acct.get('display_name', '')})")
        from src.collectors.truthsocial_collector import _strip_html
        click.echo(f"    {_strip_html(sample.get('content', ''))[:200]}")

    click.echo("\nDone. Token works, /context returns data.")
    click.echo("Next: wire this token into the collector for full reply fetching.")


# ── collect-truth ───────────────────────────────────────────────────

@main.command("collect-truth")
@click.option("--force", is_flag=True, help="Ignore cached files and re-fetch")
@click.option("--since", "since_s", default=None,
              help="Override start date (ISO, e.g. 2026-01-01). "
                   "Defaults to settings.COLLECTION_START.")
@click.option("--until", "until_s", default=None,
              help="Override end date (ISO). Defaults to settings.COLLECTION_END.")
@click.option("--handle", default=None,
              help="Only collect this one handle (e.g. realDonaldTrump)")
@click.option("--anonymous", is_flag=True,
              help="Force the paced public API for incremental refresh (no login)")
def collect_truth_cmd(force: bool, since_s: str | None, until_s: str | None,
                      handle: str | None, anonymous: bool):
    """Collect Truth Social posts for the configured accounts."""
    from datetime import date as _date
    from src.collectors.truthsocial_collector import collect_all, collect_user

    start = _date.fromisoformat(since_s) if since_s else settings.COLLECTION_START
    end = _date.fromisoformat(until_s) if until_s else settings.COLLECTION_END

    # Heads-up: the `armada` tracked post (Jan 28 2026) sits outside the
    # default Feb 1 window — pass --since 2026-01-01 to capture it.
    click.echo(f"Window: {start} → {end}")
    click.echo("Truth Social API is free — no budget estimate.\n")

    if handle:
        # Find which tier this handle belongs to
        tier = next(
            (t for t, hs in TRUTH_SOCIAL_ACCOUNTS.items() if handle in hs),
            "admin",
        )
        posts = collect_user(handle, tier, start=start, end=end, force=force,
                             use_auth=False if anonymous else None)
        click.echo(f"@{handle} [{tier}]: {len(posts)} posts")
        return

    if anonymous:
        from config import settings as _s
        _s.TS_PREFER_AUTH = False
    summary = collect_all(TRUTH_SOCIAL_ACCOUNTS, start=start, end=end, force=force)
    total = sum(summary.values())
    click.echo(f"\nCollected {total} total posts across {len(summary)} accounts:")
    for h, count in sorted(summary.items(), key=lambda x: -x[1]):
        click.echo(f"  @{h}: {count}")


# ── collect-replies ─────────────────────────────────────────────────

@main.command("collect-replies")
@click.option("--force", is_flag=True, help="Ignore cached reply files and re-fetch")
@click.option("--slug", default=None, help="Only collect replies for this tracked-post slug")
def collect_replies_cmd(force: bool, slug: str | None):
    """Fetch Truth Social replies for the posts in config/tracked_posts.py."""
    from config.tracked_posts import TRACKED_POSTS, resolve_post_ids
    from src.collectors.truthsocial_collector import collect_replies, RAW_DIR

    tracked = [tp for tp in TRACKED_POSTS if slug is None or tp.slug == slug]
    if not tracked:
        click.secho(f"No tracked posts match slug={slug!r}", fg="red")
        sys.exit(1)

    ids = resolve_post_ids(tracked)

    total = 0
    for tp in tracked:
        post_id = ids.get(tp.slug)
        if not post_id:
            click.secho(
                f"  ✗ {tp.slug}: no post_id (run `collect` first or pin post_id "
                f"in config/tracked_posts.py)",
                fg="yellow",
            )
            continue

        cache = RAW_DIR / f"replies_{tp.slug}.jsonl"
        if cache.exists() and not force:
            n = sum(1 for _ in open(cache))
            click.secho(f"  ⤷ {tp.slug}: cached ({n} replies) — skip", fg="cyan")
            total += n
            continue

        click.echo(f"  → {tp.slug}: fetching replies to post {post_id}")
        replies = collect_replies(post_id, label=tp.slug)
        click.secho(f"    {len(replies)} replies saved", fg="green")
        total += len(replies)

    click.echo(f"\nTotal replies across {len(tracked)} tracked posts: {total}")


# ── analyze ─────────────────────────────────────────────────────────

@main.command()
@click.option("--llm", is_flag=True, help="Also run Claude LLM scoring (slow, costs tokens)")
@click.option("--no-transformer", is_flag=True, help="Skip the RoBERTa phase")
@click.option("--llm-tiers", default=None,
              help="Comma-separated tiers to LLM-score (e.g. "
                   "'religious_authority,opposition'). Only used with --llm.")
@click.option("--llm-accounts", default=None,
              help="Comma-separated handles to LLM-score. Only used with --llm.")
def analyze(llm: bool, no_transformer: bool,
            llm_tiers: str | None, llm_accounts: str | None):
    """Run sentiment scoring on all cached X data."""
    from src.collectors.x_collector import load_all_cached
    from src.analysis.sentiment import analyze as run_analyze, save

    posts = load_all_cached()
    click.echo(f"Loaded {len(posts)} cached tweets")
    if not posts:
        click.secho("No cached data — run `collect` first.", fg="yellow")
        return

    tiers_list = [t.strip() for t in llm_tiers.split(",")] if llm_tiers else None
    accounts_list = [a.strip() for a in llm_accounts.split(",")] if llm_accounts else None

    df = run_analyze(
        posts,
        use_vader=True,
        use_transformer=not no_transformer,
        use_llm=llm,
        llm_tiers=tiers_list,
        llm_accounts=accounts_list,
    )
    out = save(df)
    click.secho(f"✓ Scored {len(df)} tweets → {out}", fg="green")


# ── visualize ───────────────────────────────────────────────────────

@main.command()
def visualize():
    """Generate all figures from the scored parquet."""
    from src.analysis.sentiment import load_scored
    from src.visualization.plots import generate_all

    if not settings.SENTIMENT_OUTPUT.exists():
        click.secho(f"No scored data at {settings.SENTIMENT_OUTPUT} — run `analyze` first.",
                    fg="yellow")
        return

    df = load_scored()
    click.echo(f"Loaded {len(df)} scored tweets")
    written = generate_all(df)
    click.secho(f"✓ Generated {len(written)} figures in {settings.FIGURES_DIR}", fg="green")
    for p in written:
        click.echo(f"  {p.name}")


# ── summary ─────────────────────────────────────────────────────────

@main.command()
@click.option("--score", default="score_vader", help="Score column to summarize")
def summary(score: str):
    """Print stats tables: by account, by tier, weekly trend."""
    from src.analysis.sentiment import load_scored

    if not settings.SENTIMENT_OUTPUT.exists():
        click.secho("No scored data — run `analyze` first.", fg="yellow")
        return

    df = load_scored()
    click.echo(f"\n{'='*60}")
    click.echo(f"RESULTS SUMMARY ({len(df)} tweets, {score})")
    click.echo(f"{'='*60}")

    click.echo("\n--- By account ---")
    stats = df.groupby("user")[score].agg(["mean", "count"]).sort_values("mean")
    for user, row in stats.iterrows():
        bar = "█" * int(abs(row["mean"]) * 20)
        direction = "+" if row["mean"] >= 0 else "-"
        click.echo(f"  @{user:22s}  {row['mean']:+.3f}  n={int(row['count']):4d}  {direction}{bar}")

    click.echo("\n--- By tier ---")
    stats = df.groupby("tier")[score].agg(["mean", "count"]).sort_values("mean")
    for tier, row in stats.iterrows():
        click.echo(f"  {tier:15s}  {row['mean']:+.3f}  n={int(row['count'])}")

    click.echo("\n--- Weekly trend ---")
    df = df.copy()
    df["week"] = df["created_at"].dt.isocalendar().week.astype(int)
    weekly = df.groupby("week")[score].mean()
    for week, val in weekly.items():
        bar = "█" * int(abs(val) * 30)
        direction = "+" if val >= 0 else "-"
        click.echo(f"  Week {week:2d}: {val:+.3f}  {direction}{bar}")


# ── event-study ─────────────────────────────────────────────────────

@main.command("event-study")
@click.option("--slug", default=None, help="Drill into a single tracked-post slug")
@click.option("--window-hours", default=48, show_default=True,
              help="Pre/post window for broadcaster event comparison")
@click.option("--score", default=None,
              help="Force a score column (default: best available)")
@click.option("--broadcaster/--no-broadcaster", default=True,
              help="Also run the pre/post timeline-event comparison on broadcaster data")
@click.option("--force-score", is_flag=True,
              help="Re-run sentiment on replies even if reply_sentiment.parquet exists")
def event_study_cmd(slug: str | None, window_hours: int, score: str | None,
                    broadcaster: bool, force_score: bool):
    """Run the per-post reply and broadcaster event-window analyses."""
    from src.analysis.event_study import (
        load_or_score_replies,
        summarize_post_replies,
        summarize_all_posts,
        segment_by_loyalty,
        compare_events,
    )

    # 1. Reply-level (audience sentiment on tracked posts)
    df_replies = load_or_score_replies(force=force_score)
    if df_replies.empty:
        click.secho("No reply data — run `collect-replies` first.", fg="yellow")
    else:
        click.echo(f"\n{'='*70}")
        click.echo(f"REPLY SENTIMENT ({len(df_replies)} replies across tracked posts)")
        click.echo(f"{'='*70}")

        if slug:
            summaries = pd.DataFrame(
                [summarize_post_replies(df_replies, slug, score_col=score).as_row()]
            )
        else:
            summaries = summarize_all_posts(df_replies, score_col=score)

        if summaries.empty:
            click.secho("(no matching replies)", fg="yellow")
        else:
            click.echo(
                f"\n{'slug':20s} {'N':>6s}  {'mean':>8s}  {'95% CI':>18s}  "
                f"{'crit%':>6s} {'neut%':>6s} {'supp%':>6s}"
            )
            click.echo("-" * 78)
            for _, row in summaries.iterrows():
                ci = f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"
                click.echo(
                    f"{row['slug']:20s} {int(row['n']):>6d}  "
                    f"{row['mean_score']:+8.3f}  {ci:>18s}  "
                    f"{row['pct_critical']:>5.1f}% {row['pct_neutral']:>5.1f}% "
                    f"{row['pct_supportive']:>5.1f}%"
                )
            click.echo(f"\nscore col: {summaries.iloc[0]['score_col']}   "
                       f"label col: {summaries.iloc[0]['label_col']}")

        # Loyalty segmentation — the within-MAGA civil-war test
        click.echo("\n--- Loyalty-tier segmentation ---")
        loyalty = segment_by_loyalty(df_replies, score_col=score)
        if loyalty.empty:
            click.secho("(no loyalty data)", fg="yellow")
        else:
            for _, row in loyalty.iterrows():
                slug_col = row.get("tracked_slug", "(all)")
                ci = f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"
                click.echo(
                    f"  {str(slug_col):20s}  {row['loyalty_tier']:7s}  "
                    f"n={int(row['n']):4d}  {row['mean']:+.3f}  {ci}"
                )

    # 2. Broadcaster-level (pre/post around timeline events)
    if broadcaster:
        if not settings.SENTIMENT_OUTPUT.exists():
            click.secho(
                f"\nNo broadcaster sentiment at {settings.SENTIMENT_OUTPUT} — "
                f"run `analyze` first.", fg="yellow"
            )
            return

        from src.analysis.sentiment import load_scored
        from config.timeline import EVENTS

        df_bcast = load_scored()
        click.echo(f"\n{'='*70}")
        click.echo(f"BROADCASTER EVENT WINDOWS (±{window_hours}h on {len(df_bcast)} posts)")
        click.echo(f"{'='*70}")

        results = compare_events(df_bcast, EVENTS, window_hours=window_hours, score_col=score)
        if results.empty:
            click.secho("(no events had any data in window)", fg="yellow")
            return

        click.echo(
            f"\n{'date':10s}  {'event':36s}  {'n_pre':>5s} {'n_post':>6s}  "
            f"{'pre':>7s} {'post':>7s}  {'Δ [95% CI]':>22s}"
        )
        click.echo("-" * 100)
        for _, row in results.iterrows():
            ci = f"[{row['diff_ci_low']:+.2f}, {row['diff_ci_high']:+.2f}]"
            diff_ci = f"{row['diff']:+.3f} {ci}"
            label = (row["event_label"][:34] + "..") if len(row["event_label"]) > 36 else row["event_label"]
            click.echo(
                f"{row['event_date']:10s}  {label:36s}  "
                f"{int(row['n_pre']):>5d} {int(row['n_post']):>6d}  "
                f"{row['mean_pre']:+7.3f} {row['mean_post']:+7.3f}  {diff_ci:>22s}"
            )
        click.echo(f"\nscore col: {results.iloc[0]['score_col']}")


# ── stance ─────────────────────────────────────────────────────────

@main.command()
@click.option("--n", "n_per_bucket", default=50, show_default=True,
              help="Replies to sample per sentiment bucket per post")
@click.option("--force", is_flag=True, help="Re-score even if cached")
@click.option("--model", default=None,
              help="Override LLM model (default: settings.LLM_MODEL)")
def stance(n_per_bucket: int, force: bool, model: str | None):
    """Run LLM stance classification on a stratified reply sample."""
    from src.analysis.event_study import (
        load_or_score_replies,
        stratified_stance_sample,
        score_stance,
        stance_summary,
    )

    df_replies = load_or_score_replies()
    if df_replies.empty:
        click.secho("No reply data — run `collect-replies` then `event-study` first.",
                     fg="yellow")
        return

    sample = stratified_stance_sample(df_replies, n_per_bucket=n_per_bucket)
    click.echo(f"Stratified sample: {len(sample)} replies "
               f"({n_per_bucket}/bucket × 3 buckets × "
               f"{sample['tracked_slug'].nunique()} posts + flippers)")

    kwargs = {"force": force}
    if model:
        kwargs["model"] = model

    result = score_stance(sample, **kwargs)
    stance_summary(result)


# ── status ──────────────────────────────────────────────────────────

@main.command()
def status():
    """Report what's been collected and what's still missing."""
    cached_files = list(settings.X_RAW_DIR.glob("*.jsonl"))
    cached_handles = {f.stem for f in cached_files if not f.stem.startswith("search_")}
    cached_searches = {f.stem.replace("search_", "").replace("_", " ")
                       for f in cached_files if f.stem.startswith("search_")}

    click.echo("\n=== Collection status ===")
    for tier, handles in X_ACCOUNTS.items():
        click.echo(f"\n{tier}:")
        for handle in handles:
            # Account caching uses lowercase-preserved handle
            matched = [h for h in cached_handles if h.lower() == handle.lower()]
            if matched:
                # Count lines to show volume
                count = sum(1 for _ in open(settings.X_RAW_DIR / f"{matched[0]}.jsonl"))
                click.secho(f"  ✓ @{handle}  ({count} tweets)", fg="green")
            else:
                click.secho(f"  ✗ @{handle}  (not collected)", fg="yellow")

    click.echo("\nSearches:")
    for term in SEARCH_TERMS:
        if term in cached_searches:
            click.secho(f"  ✓ '{term}'", fg="green")
        else:
            click.secho(f"  ✗ '{term}'", fg="yellow")

    # ── Truth Social ─────────────────────────────────────────────
    click.echo("\n=== Truth Social ===")
    ts_files = list(settings.TRUTH_SOCIAL_RAW_DIR.glob("*.jsonl"))
    ts_handle_files = {f.stem for f in ts_files if not f.stem.startswith("replies_")}
    ts_reply_files = {f.stem.removeprefix("replies_")
                      for f in ts_files if f.stem.startswith("replies_")}

    for tier, handles in TRUTH_SOCIAL_ACCOUNTS.items():
        click.echo(f"\n{tier}:")
        for handle in handles:
            if handle in ts_handle_files:
                count = sum(1 for _ in open(settings.TRUTH_SOCIAL_RAW_DIR / f"{handle}.jsonl"))
                click.secho(f"  ✓ @{handle}  ({count} posts)", fg="green")
            else:
                click.secho(f"  ✗ @{handle}  (not collected)", fg="yellow")

    # Tracked-post replies (for the NYT-style audience analysis)
    from config.tracked_posts import TRACKED_POSTS
    click.echo("\nTracked-post replies:")
    for tp in TRACKED_POSTS:
        if tp.slug in ts_reply_files:
            count = sum(1 for _ in open(
                settings.TRUTH_SOCIAL_RAW_DIR / f"replies_{tp.slug}.jsonl"
            ))
            click.secho(f"  ✓ {tp.slug:20s}  ({count} replies)", fg="green")
        else:
            click.secho(f"  ✗ {tp.slug:20s}  (not collected)", fg="yellow")

    click.echo("\n=== Analysis status ===")
    if settings.SENTIMENT_OUTPUT.exists():
        size_kb = settings.SENTIMENT_OUTPUT.stat().st_size / 1024
        click.secho(f"  ✓ broadcaster: {settings.SENTIMENT_OUTPUT} ({size_kb:.0f} KB)",
                    fg="green")
    else:
        click.secho("  ✗ broadcaster: no scored data", fg="yellow")

    if settings.REPLY_SENTIMENT_OUTPUT.exists():
        size_kb = settings.REPLY_SENTIMENT_OUTPUT.stat().st_size / 1024
        click.secho(f"  ✓ replies:     {settings.REPLY_SENTIMENT_OUTPUT} ({size_kb:.0f} KB)",
                    fg="green")
    else:
        click.secho("  ✗ replies:     no scored data", fg="yellow")

    fig_count = len(list(settings.FIGURES_DIR.glob("*.png")))
    click.echo(f"\n=== Figures: {fig_count} PNG files in {settings.FIGURES_DIR} ===")


# ── run-all ─────────────────────────────────────────────────────────

@main.command()
@click.option("--llm", is_flag=True)
@click.option("--force", is_flag=True)
@click.pass_context
def run_all(ctx, llm: bool, force: bool):
    """Full pipeline: collect → analyze → visualize → summary."""
    ctx.invoke(collect, force=force, no_search=False)
    ctx.invoke(analyze, llm=llm, no_transformer=False)
    ctx.invoke(visualize)
    ctx.invoke(summary, score="score_vader")


if __name__ == "__main__":
    main()
