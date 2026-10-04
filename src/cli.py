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

import json
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
    walked oldest-slice-first so a cap hit costs a slice its older days
    instead of dropping months. The plan below is a MAXIMUM -- accounts that posted
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


# ── x-backfill-text ─────────────────────────────────────────────────

@main.command("x-backfill-text")
@click.option("--estimate", is_flag=True, help="Count and price the reads, then exit (no API call)")
@click.option("--limit", type=click.IntRange(min=1), default=None,
              help="Pilot: read only the first N posts still to check")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt")
def x_backfill_text_cmd(estimate: bool, limit: int | None, yes: bool):
    """Re-read the cached X posts stored cut at 280 characters (collected
    before note_tweet was requested) by id, and put their full text in the
    raw cache. Their Opus and topic labels move to *_superseded.parquet and
    their sentiment_all scores are cleared, so the refresh order (analyze,
    relabel, topic-label) redoes exactly them. Paid: one X read per post
    returned. A killed run resumes from the journal; a run with nothing left
    to read only applies it (free)."""
    from src.collectors import x_backfill as xb

    p = xb.plan(limit)
    t = xb.tier_table(p)
    click.echo(f"\n{len(p['candidates'])} cached originals likely cut at 280 characters, "
               f"{p['checked']} already checked (journal {xb.journal_path().name})")
    click.echo(f"\n{'tier / account':26s}{'likely cut':>11s}{'war posts':>10s}{'to read':>9s}")
    for tier, g in t.groupby("tier"):
        click.echo(f"{tier:26s}{g['candidates'].sum():>11d}{g['war'].sum():>10d}{g['to_read'].sum():>9d}")
        for r in g.itertuples():
            click.echo(f"  @{r.user:23s}{r.candidates:>11d}{r.war:>10d}{r.to_read:>9d}")
    click.echo(f"{'all':26s}{t['candidates'].sum():>11d}{t['war'].sum():>10d}{t['to_read'].sum():>9d}")
    click.echo(f"\nThis run: {p['reads']} reads ≈ ${p['max_cost_usd']:.2f} at "
               f"${settings.X_READ_COST_USD} per post (cap X_BACKFILL_MAX_READS = "
               f"{settings.X_BACKFILL_MAX_READS}, budget X_RUN_BUDGET_USD = "
               f"${settings.X_RUN_BUDGET_USD:.2f}); war posts by topic={settings.TOPIC_SOURCE}")
    if estimate:
        return
    if p["reads"] > settings.X_BACKFILL_MAX_READS:
        click.secho(f"Refusing: {p['reads']} reads is over X_BACKFILL_MAX_READS; use --limit "
                    f"or raise the cap deliberately in config/settings.py.", fg="red")
        raise SystemExit(1)
    if p["max_cost_usd"] > settings.X_RUN_BUDGET_USD:
        click.secho(f"Refusing: ${p['max_cost_usd']:.2f} exceeds X_RUN_BUDGET_USD "
                    f"${settings.X_RUN_BUDGET_USD:.2f}.", fg="red")
        raise SystemExit(1)

    failed = None
    if p["reads"]:
        if not yes and not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return
        import requests
        import tweepy

        from src.collectors.x_collector import get_client
        try:
            s = xb.lookup(get_client(), p["todo"])
            click.echo(f"\nRead {s['returned']} posts in {s['requests']} requests "
                       f"(≈ ${s['returned'] * settings.X_READ_COST_USD:.2f}): {s['long']} long, "
                       f"{s['missing']} not returned")
        except (tweepy.errors.TweepyException, requests.exceptions.RequestException) as e:
            failed = e  # what was journalled before the error still gets applied
            click.secho(f"\nLookup stopped: {e}. Rerun to resume from the journal.", fg="red")

    a = xb.apply()
    click.echo(f"\nJournal: {a['checked']} checked, {a['long']} long, {a['missing']} not returned. "
               f"Applied {a['changed']} new full texts ({a['already_applied']} were already in), "
               f"{a['files_rewritten']} raw files rewritten, {a['sentiment_rows_cleared']} "
               f"sentiment_all rows cleared; labels moved to *_superseded: {a['labels_moved']}")
    if a["drifted"]:
        click.secho(f"{len(a['drifted'])} cached posts changed since they were checked; left alone.",
                    fg="yellow")
    n, usd = xb.relabel_cost()
    if n:
        click.echo(f"Opus relabel owed on {n} backfilled posts ≈ ${usd:.2f} by Batch at their "
                   f"full length (`relabel estimate` prices average posts). Next: analyze, "
                   f"relabel estimate|submit|status|collect|merge, topic-label "
                   f"estimate|submit|status|collect, phases, export-web, backup.")
    if failed:
        raise SystemExit(1)


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
@click.option("--score", default=settings.STANCE_SCORE_COL, show_default=True, help="Score column to summarize")
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
    # Freeze new posts' random draws, drawn from this same frame, now that
    # they are saved: a later reply collection can't move what
    # reply-population weights (it records only draws stance_sample holds).
    from src.analysis.event_study import pick_score_col
    from src.analysis.inference import reply_draws
    reply_draws(df_replies, n_per_bucket=n_per_bucket, score_col=pick_score_col(df_replies))
    stance_summary(result)


# ── stance-model experiments (GPU Jobs; see k8s/README.md) ────────

def _load_scored_frame():
    import pandas as pd
    if not settings.SENTIMENT_OUTPUT.exists():
        click.secho(f"No {settings.SENTIMENT_OUTPUT} -- run analyze --llm first", fg="red")
        sys.exit(1)
    return pd.read_parquet(settings.SENTIMENT_OUTPUT)


def _fmt_estimate(est: dict) -> str:
    cache = ""
    if est.get("cached_prefix_calls"):
        cache = (f"; ≈ ${est['usd']:.2f} with the shared prefix cached on "
                 f"{est['cached_prefix_calls']} calls (optimistic: in the v2 pilot the cache "
                 f"engaged for 1 parent post in 8)")
    return (f"{est['calls']} calls to {est.get('model', settings.TEACHER_CHECK_MODEL)}: "
            f"~{est['input_tokens']:,} input + ~{est['output_tokens']:,} output tokens "
            f"≈ ${est['usd_no_cache']:.2f} at direct-API prices{cache} "
            f"(${est['price_in_per_mtok']:g} / ${est['price_out_per_mtok']:g} per MTok; input "
            f"tokens = characters / {settings.TEACHER_EST_CHARS_PER_TOKEN:g}; no API call)")


def _confirm_paid(n_calls: int, yes: bool) -> bool:
    """True to go ahead: nothing to pay for, --yes, or the user says so."""
    if n_calls == 0 or yes:
        return True
    if click.confirm("Run?", default=False):
        return True
    click.echo("Aborted.")
    return False


def _refuse_over_cap(todo: int, cap: int, fix: str) -> None:
    """Exit before the prompt when run_labels would refuse the run anyway, so
    a 'yes' is not answered with a traceback."""
    if todo > cap:
        click.secho(f"{todo} calls is over the cap ({cap}): {fix}", fg="red")
        sys.exit(1)


@main.command("teacher-check")
@click.option("--source", type=click.Choice(["x", "trump"]), default="x", show_default=True,
              help="x: Haiku vs the teacher on X posts; trump: the distilled scorer vs the "
                   "teacher on Trump's Truth Social feed")
@click.option("--n", type=int, default=None,
              help=f"Posts to label (default {settings.TEACHER_CHECK_N} for x, spread across "
                   f"tiers; {settings.TRUMP_TEACHER_CHECK_N} for trump)")
@click.option("--model", default=settings.TEACHER_CHECK_MODEL, show_default=True)
@click.option("--estimate", is_flag=True, help="Count and price the calls; no API call")
@click.option("--yes", is_flag=True,
              help="(trump) Run without the confirmation prompt; the x check never asks")
@click.option("--batch", type=click.Choice(["submit", "status", "collect"]), default=None,
              help="(trump) Through the Message Batches API at half price: submit the posts still "
                   "to label, poll, then collect into the same labels and report")
@click.option("--results-file", type=click.Path(exists=True), default=None,
              help="(--batch collect) Read a downloaded results JSONL instead of streaming it")
@click.option("--report", is_flag=True,
              help="(trump) Rebuild the report from the cached labels; no API call")
def teacher_check_cmd(source: str, n: int | None, model: str, estimate: bool, yes: bool,
                      batch: str | None, results_file: str | None, report: bool):
    """Relabel a sample with a stronger Claude model and report disagreement:
    with the Haiku labels on X posts (per tier + worst cases), or with the
    distilled scorer on Trump's own feed (by phase, war posts). API only.
    The Trump check runs direct or, with --batch, at batch prices."""
    from src.analysis import stance_local as sl
    if source == "trump":
        _check_batch_flags(batch, results_file, report, estimate=estimate)
        _trump_check(n or settings.TRUMP_TEACHER_CHECK_N, model, estimate, yes, batch,
                     results_file, report)
        return
    if batch or results_file or report:
        raise click.UsageError("--batch, --results-file and --report go with --source trump")
    from src.analysis.sentiment import llm_prompt
    n = n or settings.TEACHER_CHECK_N
    df = _load_scored_frame()
    sample, cached, todo = sl.teacher_check_todo(df, n=n, model=model)
    est = sl.estimate_direct_cost([llm_prompt(t, u) for t, u in zip(todo["text"], todo["user"])])
    click.echo(f"{len(sample)} sampled, {len(cached)} cached; {_fmt_estimate({**est, 'model': model})}")
    # No prompt: the k8s teacher-check Job runs this with no stdin, and
    # starting that Job is the approval (AGENTS.md, paid runs).
    if estimate:
        return
    joined = sl.teacher_check(df, n=n, model=model)
    rep = sl.teacher_report(joined, model)
    click.echo(f"\nHaiku vs {model} on {len(joined)} posts (ref = Haiku score_llm):")
    click.echo(f"{'tier':22s}{'n':>6s}{'pearson':>9s}{'mae':>7s}{'sign agr':>10s}{'flips':>7s}")
    for r in rep["by_tier"]:
        click.echo(f"{r['tier']:22s}{r['n']:>6d}{r['pearson']:>9.3f}{r['mae']:>7.3f}"
                   f"{r['sign_agreement']:>10.1%}{r['sign_flip_rate']:>7.1%}")
    click.echo("\nLargest disagreements:")
    for w in rep["worst"][:10]:
        click.echo(f"  [{w['tier']}] @{w['user']} haiku={w['score_llm']:+.2f} "
                   f"{model.split('-')[1]}={w['score_teacher']:+.2f} | {w['text']}")


def _agreement_row(name: str, r: dict) -> str:
    return (f"{name:24s}{r['n']:>5d}{r['pearson']:>9.3f}{r['mae']:>7.3f}{r['sign_flip_rate']:>7.1%}"
            f"{r['mean_teacher']:>+9.3f}{r['mean_model']:>+9.3f}{r['mean_diff']:>+9.3f}")


def _check_batch_flags(batch: str | None, results_file: str | None, report: bool, *,
                       estimate: bool, limit: int | None = None) -> None:
    """Reject flag mixes that would be silently ignored."""
    if report and (batch or estimate or limit):
        raise click.UsageError("--report runs on its own: it rebuilds the report from the cache")
    if batch in ("status", "collect") and (estimate or limit):
        raise click.UsageError(f"--estimate and --limit go with --batch submit, "
                               f"not --batch {batch}")
    if results_file and batch != "collect":
        raise click.UsageError("--results-file goes with --batch collect")


def _fmt_batch_estimate(est: dict) -> str:
    """The token estimate's ceiling at batch prices. No cached-prefix figure:
    the pilot showed it low, and caching in a batch is best-effort."""
    return (f"at batch prices ({est['batch_discount']:g} x direct): "
            f"≈ ${est['usd_batch_no_cache']:.2f} with no prefix cache (the ceiling); no API call")


def _fmt_pilot_estimate(est: dict, batch: bool) -> str:
    """The run priced at what the 2026-10-04 v2 pilot was really billed."""
    p = settings.TEACHER_V2_PILOT
    at = (f"≈ ${est['usd_pilot_batch']:.2f} at batch prices (≈ ${est['usd_pilot']:.2f} direct)"
          if batch else f"≈ ${est['usd_pilot']:.2f} direct")
    return (f"pilot-calibrated: {at}, at the rates the {p['calls']}-call v2 pilot was billed on "
            f"2026-10-04 (${est['pilot_usd_per_call']:.4f} a call direct on prompts averaging "
            f"{p['prompt_chars'] / p['calls']:,.0f} characters; here input scales with "
            f"{est['prompt_chars']:,} prompt characters, output is "
            f"{p['output_tokens'] / p['calls']:.0f} tokens a call)")


def _refuse_open_batch(check: str, doing: str) -> None:
    """Exit before the prompt while the check's last batch is not collected."""
    from src.analysis import stance_local as sl
    st = sl.open_batch(check)
    if st:
        click.secho(sl.open_batch_message(check, st, doing), fg="red")
        sys.exit(1)


def _batch_poll(check: str, action: str, results_file: str | None, show) -> None:
    """--batch status / collect for a check; `show(report, state)` prints the
    report a collect writes."""
    from pathlib import Path as _P

    from src.analysis import stance_local as sl
    try:
        if action == "status":
            st, rep = sl.teacher_batch_status(check), None
        else:
            st, rep = sl.teacher_batch_collect(
                check, results_file=_P(results_file) if results_file else None)
    except FileNotFoundError as e:
        click.secho(str(e), fg="yellow")
        sys.exit(1)
    except RuntimeError as e:      # a submit that never recorded its batch id
        click.secho(str(e), fg="red")
        sys.exit(1)
    click.echo(f"batch {st['batch_id']}: {st['status']}  {st.get('collected') or st.get('counts')}")
    if st.get("spend"):
        s = st["spend"]
        click.echo(f"billed: {s['input_tokens']:,} input, "
                   f"{s['cache_creation_input_tokens']:,} cache-write, "
                   f"{s['cache_read_input_tokens']:,} cache-read, {s['output_tokens']:,} output "
                   f"tokens ≈ ${s['usd_batch']:.2f} at batch prices "
                   f"(${s['usd_direct']:.2f} direct)")
        if s.get("calls_without_usage"):
            click.secho(f"{s['calls_without_usage']} results carried no usage: the bill above "
                        "leaves them out", fg="yellow")
    if st.get("short_read") and st["status"] != "collected":
        sr = st["short_read"]
        click.secho(f"short read ({sr['source']}): {sr['missing']} of {st['n_submitted']} "
                    f"submitted ids have no result in it, {sr['ignored']['not_submitted']} "
                    f"results were for ids not in this batch; {sr['labelled']} labels added. "
                    "The batch stays open: --batch collect again from the stream or the whole "
                    "results file.", fg="red")
        sys.exit(1)
    if rep is not None:
        show(rep, st)


def _echo_submitted(st: dict | None, cmd: str) -> None:
    if st is None:
        click.echo("nothing submitted: every row is labelled")
        return
    click.echo(f"batch {st['batch_id']} {st['status']}  ({st['n_submitted']} requests); "
               f"then: {cmd} --batch status / collect")


def _print_trump_report(rep: dict, model: str) -> None:
    click.echo(f"\n{rep['col']} vs {model} on {rep['n']} Trump posts "
               f"(mean diff = model - teacher; war posts by topic={rep['topic_source']}):")
    click.echo(f"{'':24s}{'n':>5s}{'pearson':>9s}{'mae':>7s}{'flips':>7s}{'teacher':>9s}{'model':>9s}{'diff':>9s}")
    click.echo(_agreement_row("all posts", rep["all"]))
    for r in rep["by_phase"]:
        click.echo(_agreement_row(f"  {r['phase']}", r))
    click.echo(_agreement_row("war posts", rep["war_posts"]))
    for r in rep["war_by_phase"]:
        click.echo(_agreement_row(f"  {r['phase']}", r))


def _trump_check(n: int, model: str, estimate: bool, yes: bool, batch: str | None = None,
                 results_file: str | None = None, report: bool = False) -> None:
    """teacher-check --source trump: direct, or through a batch."""
    from src.analysis import stance_local as sl
    if report:
        _print_trump_report(sl.trump_teacher_report(model), model)
        return
    if batch in ("status", "collect"):
        _batch_poll("trump", batch, results_file,
                    lambda rep, st: _print_trump_report(rep, st["model"]))
        return
    est = sl.trump_check_estimate(n=n, model=model)
    click.echo(f"Trump feed: {est['sample']} sampled posts with text, {est['cached']} cached, "
               f"cap {est['cap']}; {_fmt_estimate(est)}")
    if batch:
        click.echo(_fmt_batch_estimate(est))
    click.echo(_fmt_pilot_estimate(est, bool(batch)))
    if sl.open_batch("trump"):
        if estimate:
            click.secho("a Trump batch is open: these calls include its requests", fg="yellow")
        else:
            _refuse_open_batch("trump", "a new batch" if batch else "a direct run")
    _refuse_over_cap(est["todo"], est["cap"],
                     "raise TRUMP_TEACHER_CHECK_MAX_CALLS deliberately or lower --n")
    if estimate or not _confirm_paid(est["todo"], yes):
        return
    if batch == "submit":
        params = {"n": n, "seed": settings.TEACHER_SAMPLE_SEED, "col": "score_opus_distilled"}
        st = sl.teacher_batch_submit("trump", params=params, model=model)
        _echo_submitted(st, "teacher-check --source trump")
        return
    rep = sl.trump_teacher_check(n=n, model=model)
    _print_trump_report(rep, model)


@main.command("stance-distill")
@click.option("--base-model", default=settings.DISTILL_BASE_MODEL, show_default=True)
@click.option("--epochs", default=settings.DISTILL_EPOCHS, show_default=True)
@click.option("--label-col", default="score_llm", show_default=True,
              help="Teacher column to regress (e.g. score_opus after `relabel merge`)")
@click.option("--recipe", default=None,
              help="Name of a settings.DISTILL_SWEEP recipe; overrides base-model/epochs")
@click.option("--fit-all", is_flag=True,
              help="After the holdout evaluation, refit on all labels -> stance_distilled_final[_<label>]")
@click.option("--with-replies", is_flag=True,
              help="Mix the teacher-labelled reply sample into training (tier=reply_<post>; 20%% held out)")
def stance_distill_cmd(base_model: str, epochs: int, label_col: str, recipe: str | None, fit_all: bool,
                       with_replies: bool):
    """Fine-tune an encoder to reproduce the teacher stance score (GPU)."""
    from src.analysis.stance_local import distill, reply_label_frame
    extra = reply_label_frame(label_col=label_col) if with_replies else None
    m = distill(_load_scored_frame(), base_model=base_model, epochs=epochs, label_col=label_col,
                recipe=recipe, fit_all=fit_all, extra=extra)
    for r in m.get("distilled_by_tier", []):
        if str(r["tier"]).startswith("reply_") or r["tier"] == "ALL":
            click.echo(f"  {r['tier']:26s} n={r['n']:4d} pearson={r['pearson']:.3f} "
                       f"sign_agr={r['sign_agreement']:.1%} flips={r['sign_flip_rate']:.1%}")
    click.echo(f"distilled vs teacher: {m['distilled_vs_teacher']}")
    if "roberta_valence_vs_teacher" in m:
        click.echo(f"RoBERTa valence vs teacher (same rows): {m['roberta_valence_vs_teacher']}")


@main.command("stance-sweep")
@click.option("--folds", default=settings.DISTILL_CV_FOLDS, show_default=True)
@click.option("--label-col", default="score_llm", show_default=True)
def stance_sweep_cmd(folds: int, label_col: str):
    """Run every distillation recipe in settings.DISTILL_SWEEP, cross-validate
    the best, and fit it on all labels (GPU; hours). Resumable."""
    from src.analysis.stance_local import sweep
    s = sweep(_load_scored_frame(), folds=folds, label_col=label_col)
    click.echo(f"{'recipe':16s}{'pearson':>9s}{'mae':>7s}{'sign agr':>10s}{'flips':>7s}")
    for r in s["results"]:
        h = r.get("holdout")
        if h:
            click.echo(f"{r['name']:16s}{h['pearson']:>9.3f}{h['mae']:>7.3f}{h['sign_agreement']:>10.1%}{h['sign_flip_rate']:>7.1%}")
        else:
            click.echo(f"{r['name']:16s}  FAILED: {r.get('error', '')[:60]}")
    click.echo(f"\nbest: {s.get('best')}  cv: {s.get('cv')}")


@main.command("stance-local-llm")
@click.option("--model", default=settings.LOCAL_LLM_MODEL, show_default=True)
@click.option("--n", default=settings.LOCAL_LLM_EVAL_N, show_default=True)
@click.option("--thinking/--no-thinking", default=False, show_default=True,
              help="Enable the model's reasoning mode (Qwen3 enable_thinking)")
@click.option("--max-new-tokens", default=None, type=int,
              help=f"Generation budget (default {settings.LOCAL_LLM_MAX_NEW_TOKENS}; use ~1024 with --thinking)")
@click.option("--batch-size", default=None, type=int,
              help=f"Prompts per generate() call (default {settings.LOCAL_LLM_BATCH}; 2-4 with --thinking "
                   "on a 10 GB GPU, the KV cache scales with batch x max_new_tokens)")
def stance_local_llm_cmd(model: str, n: int, thinking: bool, max_new_tokens: int | None,
                         batch_size: int | None):
    """Score a held-out sample with an open instruct model (GPU) and report
    agreement with the teacher."""
    from src.analysis.stance_local import local_llm_eval
    kw = {"thinking": thinking}
    if max_new_tokens:
        kw["max_new_tokens"] = max_new_tokens
    if batch_size:
        kw["batch_size"] = batch_size
    m = local_llm_eval(_load_scored_frame(), model_name=model, n=n, **kw)
    click.echo(f"unparseable: {m['unparseable_rate']:.1%}")
    click.echo(f"local vs teacher: {m['local_vs_teacher']}")
    if "roberta_valence_vs_teacher" in m:
        click.echo(f"RoBERTa valence vs teacher (same rows): {m['roberta_valence_vs_teacher']}")


@main.command("score-distilled")
@click.option("--model-dir", default=None, help="Model directory (default MODELS_DIR/stance_distilled)")
@click.option("--col", default="score_distilled", show_default=True, help="Output column")
@click.option("--max-len", type=int, default=None,
              help="Token cap at inference (default: the model's training length, recipe.txt)")
def score_distilled_cmd(model_dir: str | None, col: str, max_len: int | None):
    """Score every cached reply with a distilled model (population-level stance)."""
    from pathlib import Path as _P
    from src.analysis.stance_local import score_replies
    df = score_replies(_P(model_dir) if model_dir else None, col=col, max_len=max_len)
    click.echo(df.groupby("tracked_slug")[col].agg(["mean", "count"]).round(3).to_string())


# ── relabel (Opus teacher via the Batch API) ──────────────────────

@main.command("relabel")
@click.argument("action", type=click.Choice(["estimate", "submit", "status", "collect", "merge", "resubmit"]))
@click.option("--model", default=settings.TEACHER_CHECK_MODEL, show_default=True)
@click.option("--yes", is_flag=True, help="Submit without the confirmation prompt")
def relabel_cmd(action: str, model: str, yes: bool):
    """Relabel every labelled post with a stronger teacher through the Batch
    API (half price, ~1 h). Steps: estimate -> submit -> status -> collect
    -> merge; `resubmit` retries the ids that failed in the last collect."""
    from src.analysis import relabel as rl
    if action in ("estimate", "submit", "resubmit", "merge"):
        df = _load_scored_frame()
    if action == "estimate":
        n = len(rl.posts_to_label(df, model))
        click.echo(f"{n} posts to label with {model}; batch cost ≈ ${rl.estimate_cost(n):.2f} "
                   f"(direct would be ≈ ${rl.estimate_cost(n) / rl.BATCH_DISCOUNT:.2f})")
        return
    if action in ("submit", "resubmit"):
        only = None
        if action == "resubmit":
            st = json.loads(rl.state_path().read_text())
            only = set(st.get("collected", {}).get("failed_ids", []))
            if not only:
                click.echo("nothing to resubmit")
                return
        n = len(rl.posts_to_label(df, model, only))
        click.echo(f"{n} posts -> {model}, batch cost ≈ ${rl.estimate_cost(n):.2f}")
        if n and not yes and not click.confirm("Submit?", default=False):
            click.echo("Aborted.")
            return
        st = rl.submit(df, model=model, only_ids=only)
        click.echo(f"batch {st.get('batch_id')} {st.get('status')}  ({st.get('n_submitted')} requests)")
        return
    if action == "status":
        st = rl.status()
        click.echo(f"batch {st['batch_id']}: {st['status']}  {st.get('counts')}")
        return
    if action == "collect":
        st = rl.collect()
        click.echo(f"batch {st['batch_id']}: {st['status']}  {st.get('collected') or st.get('counts')}")
        return
    if action == "merge":
        out = rl.merge(df, model)
        out.to_parquet(settings.SENTIMENT_OUTPUT, index=False)
        tag = rl.tag_for(model)
        click.echo(f"score_{tag} on {int(out[f'score_{tag}'].notna().sum())}/{len(out)} posts -> "
                   f"{settings.SENTIMENT_OUTPUT}")


# ── phases / reply-population / teacher-retest ─────────────────────

def _fmt_ci(r, k: str) -> str:
    return f"{r[k]:+.3f} [{r[k + '_lo']:+.2f}, {r[k + '_hi']:+.2f}]"


def _phase_frame(source: str, score: str | None):
    """(frame, score column, group column) for `phases`: the X accounts or
    Trump's own Truth Social feed."""
    if source == "trump":
        df = pd.read_parquet(settings.TRUMP_FEED_STANCE)
        return df, score or "score_opus_distilled", "user"
    return _load_scored_frame(), score or settings.STANCE_SCORE_COL, None


@main.command()
@click.option("--source", type=click.Choice(["x", "trump"]), default="x", show_default=True,
              help="X broadcaster accounts, or Trump's own Truth Social feed")
@click.option("--by", "group", type=click.Choice(["tier", "user"]), default="tier", show_default=True)
@click.option("--score", default=None, help="Score column (default: stance of record for the source)")
@click.option("--topic", "topic_source", type=click.Choice(["llm", "keyword", "either"]),
              default=settings.TOPIC_SOURCE, show_default=True,
              help="About-the-war flag: Haiku labels (strict), keyword pattern (loose), or either")
@click.option("--n-boot", default=settings.BOOTSTRAP_N, show_default=True)
def phases(source: str, group: str, score: str | None, topic_source: str, n_boot: int):
    """Stance by phase with account-day block-bootstrap 95% CIs, split into
    war posts and the rest, and each phase's change from the first split into
    topic-share and on-war stance effects. Writes CSVs to data/processed/."""
    from src.analysis import inference as inf
    df, score, default_group = _phase_frame(source, score)
    group = default_group or group
    d = inf.prepare(df, score, group=group, topic_source=topic_source)
    boot = inf.bootstrap_cells(d, score, n_boot=n_boot)
    stats, contrasts = inf.phase_stats(boot), inf.phase_contrasts(boot)
    gaps = inf.phase_gaps(boot, settings.PHASE_GAPS) if group == "tier" else pd.DataFrame()
    tag = f"{source}_{group}_{score}_{topic_source}"
    stats.to_csv(settings.PROCESSED_DIR / f"phases_{tag}.csv", index=False)
    contrasts.to_csv(settings.PROCESSED_DIR / f"phase_contrasts_{tag}.csv", index=False)
    if not gaps.empty:
        gaps.to_csv(settings.PROCESSED_DIR / f"phase_gaps_{tag}.csv", index=False)

    chk = inf.topic_filter_check(d, score)
    click.echo(f"\n{len(d)} posts, {score}, topic={topic_source}: {chk['on_topic_share']:.0%} about the war; "
               f"mean |stance| {chk['mean_abs_on']:.2f} on vs {chk['mean_abs_off']:.2f} off; "
               f"{chk['strong_captured']:.0%} of |stance|>=0.3 posts are flagged on-war")
    click.echo(f"\n{'group':22s}{'phase':22s}{'n':>6s}{'war':>6s}  {'all posts':24s}  {'war posts only':24s}")
    for _, r in stats.iterrows():
        click.echo(f"{r['group']:22s}{str(r['phase']):22s}{r['n']:>6d}{r['share']:>6.0%}  "
                   f"{_fmt_ci(r, 'mean_all'):24s}  {_fmt_ci(r, 'mean_on'):24s}")
    click.echo(f"\nChange from the first phase (all = share + on-war + off-war effects):")
    click.echo(f"{'group':22s}{'phase':22s}{'all':24s}{'share effect':24s}{'war-post stance change':24s}")
    for _, r in contrasts.iterrows():
        click.echo(f"{r['group']:22s}{str(r['phase']):22s}{_fmt_ci(r, 'all'):24s}"
                   f"{_fmt_ci(r, 'share_effect'):24s}{_fmt_ci(r, 'on_topic_change'):24s}")
    if not gaps.empty:
        click.echo(f"\nGaps:")
        for _, r in gaps.iterrows():
            click.echo(f"{r['pair']:30s}{str(r['phase']):22s}all {_fmt_ci(r, 'mean_all'):24s}"
                       f"war {_fmt_ci(r, 'mean_on')}")


@main.command("reply-population")
@click.option("--col", default="score_opus_distilled", show_default=True)
@click.option("--n-boot", default=settings.BOOTSTRAP_N, show_default=True)
@click.option("--labels", "labels_version", type=click.Choice(["v1", "v2"]),
              default=settings.REPLY_TEACHER_LABELS_VERSION, show_default=True,
              help="Reply teacher labels to weight (settings.REPLY_TEACHER_LABELS_VERSION)")
def reply_population_cmd(col: str, n_boot: int, labels_version: str):
    """Opus stance of each tracked post's whole reply audience, estimated from
    the Opus-labelled reply sample: direct (weighted sample) and model-assisted
    (distilled census + weighted correction), with bootstrap CIs. No API.
    The CSV name carries the label version past v1."""
    from src.analysis.inference import reply_population
    rp = reply_population(col=col, n_boot=n_boot, labels_version=labels_version)
    suffix = "" if labels_version == "v1" else f"_{labels_version}"
    rp.to_csv(settings.PROCESSED_DIR / f"reply_population_{col}{suffix}.csv", index=False)
    click.echo(f"Opus labels: {labels_version}")
    click.echo(f"\n{'post':20s}{'replies':>8s}{'lab':>5s}  {'model mean':>10s}  {'Opus-corrected mean':24s}"
               f"{'model pro':>10s}  {'corrected pro':24s}{'model anti':>11s}  {'corrected anti'}")
    for _, r in rp.iterrows():
        click.echo(f"{r['post']:20s}{r['n_replies']:>8d}{r['n_labelled']:>5d}  {r['model_mean']:>+10.3f}  "
                   f"{_fmt_ci(r, 'assisted_mean'):24s}{r['model_pro']:>10.0%}  "
                   f"{r['assisted_pro']:.0%} [{r['assisted_pro_lo']:.0%}, {r['assisted_pro_hi']:.0%}]{'':8s}"
                   f"{r['model_anti']:>11.0%}  {r['assisted_anti']:.0%} [{r['assisted_anti_lo']:.0%}, {r['assisted_anti_hi']:.0%}]")


@main.command("teacher-retest")
@click.option("--model", default=settings.TEACHER_CHECK_MODEL, show_default=True)
def teacher_retest_cmd(model: str):
    """How much the Opus label moves when the same post is labelled twice
    (direct teacher check vs batch relabel). No API."""
    from src.analysis.inference import teacher_retest
    by, s = teacher_retest(model)
    click.echo(f"\n{model} test-retest on {s['n']} posts: {s['identical']:.0%} identical, "
               f"{s['within_0.1']:.0%} within 0.1, per-label noise SD {s['noise_sd_per_label']:.3f}, "
               f"mean shift {s['mean_shift']:+.3f}")
    click.echo(f"{'tier':22s}{'n':>6s}{'pearson':>9s}{'mae':>7s}{'sign agr':>10s}{'flips':>7s}")
    for tier, r in by.iterrows():
        click.echo(f"{tier:22s}{int(r['n']):>6d}{r['pearson']:>9.3f}{r['mae']:>7.3f}"
                   f"{r['sign_agreement']:>10.1%}{r['sign_flip_rate']:>7.1%}")


# ── export-web ──────────────────────────────────────────────────────

@main.command("export-web")
@click.option("--out", "out_dir", default=str(settings.WEB_EXPORT_DIR), show_default=True,
              type=click.Path(), help="Post folder in the dkweb repo")
def export_web_cmd(out_dir: str):
    """Write the chart JSON for the dkweb blog post (Observable Plot at build time)."""
    from pathlib import Path as _P
    from src.visualization.web_export import export
    for p in export(_P(out_dir)):
        click.echo(f"  {p}")


# ── backup ──────────────────────────────────────────────────────────

@main.command()
@click.option("--dry-run", is_flag=True, help="Show what would be uploaded")
def backup(dry_run: bool):
    """Sync data/raw, data/processed and data/models to the off-box S3 bucket
    (append-only, versioned). Run after any paid collect / relabel."""
    from src.backup import run
    failed = [(p, rc) for p, rc in run(dry_run=dry_run) if rc != 0]
    for p, rc in failed:
        click.secho(f"sync to {p} failed (exit {rc})", fg="red")
    if failed:
        sys.exit(1)
    click.secho("✓ backup synced" + (" (dry run)" if dry_run else ""), fg="green")


# ── topic-label ─────────────────────────────────────────────────────

@main.command("topic-label")
@click.argument("action", type=click.Choice(["estimate", "submit", "status", "collect"]))
@click.option("--yes", is_flag=True, help="Submit without the confirmation prompt")
@click.option("--results-file", type=click.Path(exists=True), default=None,
              help="collect from a downloaded results JSONL instead of streaming it")
def topic_label_cmd(action: str, yes: bool, results_file: str | None):
    """Label every broadcaster post as about the Iran war or not (Haiku,
    Batch API). `phases` uses the labels to split stance changes into topic
    share and on-war stance. Steps: estimate -> submit -> status -> collect."""
    from src.analysis import topic_label as tl
    if action in ("estimate", "submit"):
        n = len(tl.posts_to_label())
        click.echo(f"{n} posts to label with {settings.LLM_MODEL}; batch cost ≈ ${tl.estimate_cost(n):.2f}")
        if action == "estimate" or n == 0:
            return
        if not yes and not click.confirm("Submit?", default=False):
            click.echo("Aborted.")
            return
        st = tl.submit()
        click.echo(f"batch {st['batch_id']} {st['status']}  ({st['n_submitted']} requests)")
        return
    from pathlib import Path as _P
    st = tl.status() if action == "status" else tl.collect(_P(results_file) if results_file else None)
    click.echo(f"batch {st['batch_id']}: {st['status']}  {st.get('collected') or st.get('counts')}")


# ── score-posts / reply-teacher-check ─────────────────────────────

@main.command("score-posts")
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path(), help="Output parquet")
@click.option("--model-dir", default=None, help="Distilled model dir (default MODELS_DIR/stance_distilled)")
@click.option("--col", default="score_opus_distilled", show_default=True)
@click.option("--max-len", type=int, default=None,
              help="Token cap at inference (default: the model's training length, recipe.txt)")
def score_posts_cmd(inputs, out: str, model_dir: str | None, col: str, max_len: int | None):
    """Score raw JSONL post files (e.g. Trump's Truth Social feed) with a distilled model (GPU)."""
    from pathlib import Path as _P
    from src.analysis.stance_local import score_post_file
    df = score_post_file([_P(i) for i in inputs], _P(out), _P(model_dir) if model_dir else None,
                         col=col, max_len=max_len)
    click.echo(f"{len(df)} posts -> {out}; mean {col} = {df[col].mean():+.3f}")


@main.command("reply-teacher-check")
@click.option("--model", default=settings.TEACHER_CHECK_MODEL, show_default=True)
@click.option("--col", default="score_opus_distilled", show_default=True,
              help="Reply column to test against the teacher")
@click.option("--v2", "v2", is_flag=True,
              help="Label with the full reply and the Trump post it answers into the v2 file "
                   "(the v1 labels are left as they are)")
@click.option("--estimate", is_flag=True, help="(--v2) Count and price the calls; no API call")
@click.option("--limit", type=click.IntRange(min=1), default=None,
              help="(--v2) Pilot: label only the first N replies still to do, dealt across posts")
@click.option("--yes", is_flag=True, help="(--v2) Run without the confirmation prompt")
@click.option("--batch", type=click.Choice(["submit", "status", "collect"]), default=None,
              help="(--v2) Through the Message Batches API at half price: submit the replies still "
                   "to label, poll, then collect into the same v2 labels and report")
@click.option("--results-file", type=click.Path(exists=True), default=None,
              help="(--batch collect) Read a downloaded results JSONL instead of streaming it")
@click.option("--report", is_flag=True,
              help="(--v2) Rebuild the report from the cached v2 labels; no API call")
def reply_teacher_check_cmd(model: str, col: str, v2: bool, estimate: bool, limit: int | None,
                            yes: bool, batch: str | None, results_file: str | None, report: bool):
    """Relabel the reply stance sample with the teacher and report how well
    the distilled reply scores reproduce it (domain-shift check). API only.
    v1 (default) is the broadcaster prompt on the stored first 100 characters;
    --v2 sends the full reply with its parent post and compares with v1,
    direct or, with --batch, at batch prices."""
    if not v2:
        if estimate or limit or yes or batch or results_file or report:
            raise click.UsageError("--estimate, --limit, --yes, --batch, --results-file and "
                                   "--report go with --v2")
        _reply_teacher_v1(model, col)
        return
    from src.analysis import stance_local as sl
    _check_batch_flags(batch, results_file, report, estimate=estimate, limit=limit)
    if report:
        _print_reply_v2_report(sl.reply_teacher_v2_report(model, col), col, model)
        return
    if batch in ("status", "collect"):
        _batch_poll("reply_v2", batch, results_file,
                    lambda rep, st: _print_reply_v2_report(rep, st["params"]["col"], st["model"]))
        return
    est = sl.reply_teacher_v2_estimate(model=model, limit=limit)
    click.echo(f"reply sample: {est['sampled']} replies, {est['no_text']} without text (skipped), "
               f"{est['with_text']} with text, {est['cached']} cached in v2; cap {est['cap']}")
    click.echo(f"to do by post: {est['todo_by_post']}")
    click.echo(_fmt_estimate(est))
    if batch:
        click.echo(_fmt_batch_estimate(est))
    click.echo(_fmt_pilot_estimate(est, bool(batch)))
    if sl.open_batch("reply_v2"):
        if estimate:
            click.secho("a v2 batch is open: these calls include its requests", fg="yellow")
        else:
            _refuse_open_batch("reply_v2", "a new batch" if batch else "a direct run")
    _refuse_over_cap(est["todo"], est["cap"],
                     "raise REPLY_TEACHER_V2_MAX_CALLS deliberately or use --limit")
    if estimate or not _confirm_paid(est["todo"], yes):
        return
    if batch == "submit":
        params = {"seed": settings.TEACHER_SAMPLE_SEED, "col": col}
        st = sl.teacher_batch_submit("reply_v2", params=params, model=model, limit=limit)
        _echo_submitted(st, "reply-teacher-check --v2")
        return
    rep = sl.reply_teacher_check_v2(model=model, col=col, limit=limit)
    _print_reply_v2_report(rep, col, model)


def _print_reply_v2_report(rep: dict, col: str, model: str) -> None:
    if not rep["n"]:
        click.secho("no v2 labels yet (every call failed?)", fg="yellow")
        return
    d, v = rep["distilled_vs_teacher"], rep["roberta_valence_vs_teacher"]
    click.echo(f"\n{col} vs {model} v2 labels on {rep['n']} replies: pearson {d['pearson']:.3f}, "
               f"sign agr {d['sign_agreement']:.1%}, flips {d['sign_flip_rate']:.1%}")
    click.echo(f"RoBERTa valence vs v2 (same rows): pearson {v['pearson']:.3f}, "
               f"sign agr {v['sign_agreement']:.1%}, flips {v['sign_flip_rate']:.1%}")
    if "v1_vs_v2" in rep:
        a = rep["v1_vs_v2"]
        click.echo(f"v1 vs v2 labels on {a['n']} replies: pearson {a['pearson']:.3f}, "
                   f"sign agr {a['sign_agreement']:.1%}, flips {a['sign_flip_rate']:.1%}")
        for r in rep["v1_vs_v2_by_input"]:
            click.echo(f"  v1 saw the {r['v1_input']}: n {r['n']}, pearson {r['pearson']:.3f}, "
                       f"flips {r['sign_flip_rate']:.1%}")
        click.echo(f"\n{'post':22s}{'n':>5s}{'v1 mean':>9s}{'v2 mean':>9s}{'v1 pro':>8s}{'v2 pro':>8s}"
                   f"{'v1 anti':>9s}{'v2 anti':>9s}")
        for r in rep["v1_vs_v2_by_post"]:
            click.echo(f"{r['post']:22s}{r['n']:>5d}{r['v1_mean']:>+9.3f}{r['v2_mean']:>+9.3f}"
                       f"{r['v1_pro']:>8.0%}{r['v2_pro']:>8.0%}{r['v1_anti']:>9.0%}{r['v2_anti']:>9.0%}")


def _reply_teacher_v1(model: str, col: str) -> None:
    """reply-teacher-check without --v2: the 2026-09 run, unchanged."""
    from src.analysis.stance_local import reply_teacher_check
    rep = reply_teacher_check(model=model, col=col)
    d, v = rep["distilled_vs_teacher"], rep["roberta_valence_vs_teacher"]
    click.echo(f"\n{col} vs {model} on {rep['n']} replies: pearson {d['pearson']:.3f}, "
               f"sign agr {d['sign_agreement']:.1%}, flips {d['sign_flip_rate']:.1%}")
    click.echo(f"RoBERTa valence vs {model} (same rows): pearson {v['pearson']:.3f}, "
               f"sign agr {v['sign_agreement']:.1%}, flips {v['sign_flip_rate']:.1%}")
    click.echo(f"\n{'post':22s}{'n':>5s}{'pearson':>9s}{'sign agr':>10s}{'flips':>7s}")
    for r in rep["by_post"]:
        click.echo(f"{r['tier']:22s}{r['n']:>5d}{r['pearson']:>9.3f}{r['sign_agreement']:>10.1%}{r['sign_flip_rate']:>7.1%}")


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
