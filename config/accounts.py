"""
Accounts and search terms to track across platforms.

Tiers encode political stance so that plots can group them.
Handles marked with ✓ have been verified against the live X API.

Tier definitions:
  admin         - official Trump administration accounts
  maga_prowar   - pro-Trump influencers supporting the Iran war
  maga_antiwar  - pro-Trump influencers *opposing* the Iran war
                  (the MAGA civil war — the most interesting signal)
  opposition    - Democratic / progressive anti-war voices
  media         - journalists covering the conflict (neutral baseline)
  iran_official - Iranian government-linked accounts
"""

# ── X (Twitter) handles ─────────────────────────────────────────────
X_ACCOUNTS: dict[str, list[str]] = {
    "admin": [
        "POTUS",          # ✓ Trump (personal POTUS account)
        "SecRubio",       # ✓ Marco Rubio — Secretary of State
        "PeteHegseth",    # ✓ Pete Hegseth — Secretary of Defense
        # Optional expansion (uncomment and verify):
        # "VP",            # JD Vance
        # "StateDept",
        # "DOD",
        # "WhiteHouse",
    ],
    "maga_prowar": [
        "LauraLoomer",    # ✓ far-right influencer, pro-strikes
        "marklevinshow",  # ✓ Mark Levin — radio host (HIGH VOLUME, ~2500 tweets)
        # "DaveRubin",    # 0 tweets in window — may post mostly video
    ],
    "maga_antiwar": [
        "TuckerCarlson",  # ✓ (low volume, ~70 tweets)
        "RealCandaceO",   # ✓ Candace Owens
        "RealAlexJones",  # ✓ InfoWars (high volume, capped)
        "mtgreenee",      # ✓ Marjorie Taylor Greene (personal acct — not @RepMTG, she's former Rep)
    ],
    "opposition": [
        "SenSanders",     # ✓ Bernie Sanders
        # Optional expansion:
        # "ChrisMurphyCT",
        # "RashidaTlaib",
        # "AOC",
        # "SenSchumer",
    ],
    "media": [
        "BarakRavid",     # ✓ Axios — Israeli/Iran diplomacy
        # Optional expansion:
        # "FarnazFassihi",
        # "richard_engel",
        # "NatashaBertrand",
    ],
    # Not yet collected:
    # "iran_official": [...],
}


# ── Truth Social handles ────────────────────────────────────────────
# Truth Social has limited public API access. Trump and Vance are
# available without auth; most other accounts require a logged-in client.
TRUTH_SOCIAL_ACCOUNTS: dict[str, list[str]] = {
    "admin": [
        "realDonaldTrump",
        "JDVance",
        "PeteHegseth",
    ],
}


# ── Search keywords / hashtags ──────────────────────────────────────
# These hit the /search/recent endpoint, which only returns the last
# ~7 days. They provide a public-sentiment proxy, distinct from the
# tracked-account signal.
SEARCH_TERMS: list[str] = [
    "Iran war",
    # Heavier searches — uncomment to expand (each adds ~$0.75 to run):
    # "Iran strikes",
    # "Iran ceasefire",
    # "#IranWar",
    # "#NoWarWithIran",
    # "Strait of Hormuz",
    # "Power Plant Day",
]


def flatten_accounts(accounts: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Turn the tiered dict into a flat list of (tier, handle) tuples."""
    return [(tier, handle) for tier, handles in accounts.items() for handle in handles]
