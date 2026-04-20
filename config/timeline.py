"""
Key events in the 2026 Iran war for correlation with sentiment shifts.

Each event has:
  - date       : ISO date string
  - label      : short label for plots
  - description: what happened
  - category   : military | diplomatic | protest | media | political
  - impact     : expected sentiment direction (hawkish / dovish / polarizing)
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Event:
    date: date
    label: str
    description: str
    category: str
    impact: str


EVENTS: list[Event] = [
    # ── Pre-war escalation ──────────────────────────────────────────
    Event(
        date=date(2025, 6, 13),
        label="Twelve-Day War begins",
        description="Israel bombs Iranian military & nuclear facilities; US intercepts "
        "Iranian retaliatory strikes and bombs 3 nuclear sites on Jun 22.",
        category="military",
        impact="hawkish",
    ),
    Event(
        date=date(2025, 6, 24),
        label="Twelve-Day War ceasefire",
        description="Twelve-Day War between Israel and Iran ends after 12 days.",
        category="diplomatic",
        impact="dovish",
    ),
    Event(
        date=date(2025, 12, 20),
        label="Iran protests erupt",
        description="Massive nationwide anti-government protests begin in Iran, "
        "the largest since 1979. Driven by economic crisis, escalate to "
        "calls for regime overthrow.",
        category="protest",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 1, 15),
        label="Iran massacre of protesters",
        description="Iranian security forces massacre thousands of civilian "
        "protesters in crackdown.",
        category="protest",
        impact="hawkish",
    ),
    Event(
        date=date(2026, 1, 28),
        label="Trump: 'Armada heading to Iran'",
        description="Trump declares on Truth Social that a 'massive Armada is "
        "heading to Iran'. Largest US military buildup since 2003 Iraq invasion.",
        category="military",
        impact="hawkish",
    ),
    Event(
        date=date(2026, 2, 1),
        label="Experiment window opens",
        description="Start of primary data collection window.",
        category="media",
        impact="polarizing",
    ),

    # ── Negotiations collapse ───────────────────────────────────────
    Event(
        date=date(2026, 2, 25),
        label="Iran FM: deal 'within reach'",
        description="Iranian FM Araghchi says 'historic' agreement to avert war "
        "is 'within reach' ahead of Geneva talks.",
        category="diplomatic",
        impact="dovish",
    ),
    Event(
        date=date(2026, 2, 27),
        label="Oman: 'breakthrough' in talks",
        description="Omani FM says a 'breakthrough' has been reached; peace "
        "'within reach'.",
        category="diplomatic",
        impact="dovish",
    ),

    # ── War begins ──────────────────────────────────────────────────
    Event(
        date=date(2026, 2, 28),
        label="US-Israel strike Iran; Khamenei killed",
        description="Israel and US launch strikes on Iran. Khamenei assassinated "
        "in Israeli air attack on Leadership House compound.",
        category="military",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 3, 1),
        label="Iran confirms Khamenei dead",
        description="Iranian state media confirms Supreme Leader Khamenei killed.",
        category="military",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 3, 2),
        label="IRGC HQ destroyed",
        description="Video shows IRGC Malek-Ashtar building in Tehran completely "
        "destroyed by joint US-Israel missile strike.",
        category="military",
        impact="hawkish",
    ),
    Event(
        date=date(2026, 3, 3),
        label="State broadcaster HQ hit",
        description="IRIB headquarters in Tehran hit in Israeli air operation.",
        category="military",
        impact="hawkish",
    ),
    Event(
        date=date(2026, 3, 5),
        label="'Boom Boom' propaganda videos",
        description="White House X account posts military propaganda videos with "
        "movie/video-game splicing. 100M+ impressions by Apr 1.",
        category="media",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 3, 7),
        label="Admin explains war rationale",
        description="Trump administration publicly lays out rationale for war "
        "with Iran via NPR/major outlets.",
        category="political",
        impact="polarizing",
    ),

    # ── Iran retaliates ─────────────────────────────────────────────
    Event(
        date=date(2026, 3, 8),
        label="Iran retaliatory strikes",
        description="Iran launches hundreds of drones and ballistic missiles at "
        "Israel and US bases in Bahrain, Jordan, Kuwait, Qatar, "
        "Saudi Arabia, and UAE.",
        category="military",
        impact="polarizing",
    ),

    # ── Polling & political reaction ────────────────────────────────
    Event(
        date=date(2026, 3, 9),
        label="Quinnipiac: 56% oppose",
        description="Quinnipiac poll: 56% oppose military action, 74% oppose "
        "ground troops.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 3, 12),
        label="CNN: 'no point' narrative",
        description="CNN analysis: 'Americans don't see the point of this war'. "
        "Fox poll: 51-29 say war made US less safe.",
        category="media",
        impact="dovish",
    ),
    Event(
        date=date(2026, 3, 25),
        label="Pew: 61% disapprove",
        description="Pew Research: 61% disapprove of Trump's handling. "
        "90% of Dems disapprove vs 69% of GOP approve.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 3, 27),
        label="120+ heritage sites damaged",
        description="Iran reports damage to at least 120 historical/heritage "
        "sites from US-Israeli strikes.",
        category="military",
        impact="dovish",
    ),
    Event(
        date=date(2026, 3, 31),
        label="Hegseth 'top cheerleader'",
        description="CNN profiles Hegseth as 'top cheerleader' for military "
        "power in Iran war.",
        category="media",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 4, 1),
        label="Iran Lego meme propaganda",
        description="CNBC reports on AI-generated Lego-style Iranian propaganda "
        "memes going viral.",
        category="media",
        impact="polarizing",
    ),

    # ── Escalation & ceasefire ──────────────────────────────────────
    Event(
        date=date(2026, 4, 5),
        label="Trump profane rant; 'Power Plant Day'",
        description="Trump posts expletive-filled Truth Social rant threatening "
        "'hell' for Iran over Hormuz. Announces 'Power Plant Day' "
        "and 'Bridge Day'.",
        category="political",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 4, 7),
        label="'Civilisation will die tonight'",
        description="Trump posts 'A whole civilisation will die tonight'. "
        "Hours later, US-Iran announce 2-week ceasefire.",
        category="diplomatic",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 4, 8),
        label="Ceasefire takes effect",
        description="Ceasefire takes effect after 40 days of sustained combat.",
        category="diplomatic",
        impact="dovish",
    ),

    # ── Fragile ceasefire & Vance negotiations ──────────────────────
    Event(
        date=date(2026, 4, 9),
        label="Lebanon flare; Hormuz still closed",
        description="Israel resumes major Lebanon strikes; Netanyahu says "
        "ceasefire 'does not include Lebanon'. Iran blockade of Strait of "
        "Hormuz not lifted as agreed; Iran accuses US/Israel of violations.",
        category="military",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 4, 11),
        label="Vance arrives Islamabad",
        description="VP Vance arrives in Islamabad with envoy Witkoff and "
        "Kushner for direct talks with Iranian FM Araghchi and parliament "
        "speaker Ghalibaf, mediated by Pakistan.",
        category="diplomatic",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 12),
        label="Talks collapse after 21 hours",
        description="Vance leaves Pakistan after 21-hour marathon yields no "
        "deal. Sticking points: Iran uranium enrichment freeze, frozen "
        "asset release. Trump announces US naval blockade of Iran.",
        category="diplomatic",
        impact="polarizing",
    ),
    Event(
        date=date(2026, 4, 13),
        label="US blockade of Hormuz begins",
        description="US Navy begins blockading ships entering/exiting Iranian "
        "ports. Trump warns Iranian ships approaching the blockade will be "
        "'eliminated'.",
        category="military",
        impact="hawkish",
    ),
    Event(
        date=date(2026, 4, 15),
        label="Framework deal nears",
        description="US officials say negotiators are close to a framework "
        "agreement before Apr 21 ceasefire expiry. Trump: war 'very close "
        "to over', 'we've beaten them militarily, totally'.",
        category="diplomatic",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 16),
        label="Israel-Lebanon 10-day ceasefire",
        description="Trump announces 10-day ceasefire between Israel and "
        "Lebanon, easing one of the main flashpoints threatening the "
        "broader US-Iran truce.",
        category="diplomatic",
        impact="dovish",
    ),

    # ── Pope Leo XIV / Vatican axis ─────────────────────────────────
    Event(
        date=date(2026, 3, 1),
        label="Pope: 'spiral of violence'",
        description="Pope Leo XIV's first major Iran-war intervention — "
        "calls on parties to halt the 'spiral of violence' across Iran "
        "and the Middle East.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 4),
        label="Pope: 'not in God's name'",
        description="CNN analysis frames Pope Leo as actively pushing back "
        "on divine justifications of war, drawing the Vatican into open "
        "moral conflict with the Trump administration's framing.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 7),
        label="St Peter's peace vigil",
        description="Pope Leo leads peace vigil at St. Peter's Basilica "
        "and demands leaders 'cease fire'. Hours later, US-Iran "
        "announce 2-week ceasefire.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 11),
        label="Pope: 'delusion of omnipotence'",
        description="Pope Leo denounces 'delusion of omnipotence' fueling "
        "US-Israel war and calls Trump's threat to annihilate Iranian "
        "civilization 'truly unacceptable'. Same day Vance arrives in "
        "Islamabad for talks.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 13),
        label="Pope on Africa tour, defies Trump",
        description="Pope Leo brushes off Trump criticism while beginning "
        "Africa tour — vows to continue peace appeals despite escalating "
        "Vatican-Washington tensions.",
        category="political",
        impact="dovish",
    ),
    Event(
        date=date(2026, 4, 15),
        label="Trump attacks Pope Leo",
        description="Trump again publicly attacks Pope Leo over Iran war "
        "stance, days after calling him 'weak on crime'. CBS reports the "
        "Pope's stance is 'inspiring American cardinals to speak out'.",
        category="political",
        impact="polarizing",
    ),
]

# Date range for primary analysis window
ANALYSIS_START = date(2026, 2, 1)
ANALYSIS_END = date(2026, 4, 21)
