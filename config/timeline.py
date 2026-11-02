"""
Key events in the 2026 Iran war for correlation with sentiment shifts.

Each event has:
  - date       : ISO date string
  - label      : short label for plots
  - description: what happened
  - category   : military | diplomatic | protest | media | political
  - impact     : expected sentiment direction (hawkish / dovish / polarizing)
  - importance : 1 (minor) .. 5 (defining). Events from Feb-Apr predate the
                 field and default to 3. Plots label only events at or above
                 settings.EVENT_LABEL_MIN_IMPORTANCE (lines are drawn for all).
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
    importance: int = 3


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
    # ── Post-ceasefire negotiation and renewed war (May 12 – Sep 15) ──
    # Added 2026-09-16 from a web-research pass (5 period researchers, each
    # independently fact-checked; only events confirmed by a second source
    # are here). Full candidate list with source URLs:
    # docs/timeline_candidates_2026-05-12_to_2026-09-15.json
    Event(
        date=date(2026, 5, 15),
        label="Trump-Xi: no Iran nuke, Hormuz must open",
        description="White House readout of the Beijing state visit: both sides "
        "agree Iran 'can never have a nuclear weapon' and Hormuz 'must remain "
        "open'. Trump leaves without an Iran breakthrough.",
        category="diplomatic", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 5, 18),
        label="Trump holds off 'planned Military attack'",
        description="Trump posts that Qatar, Saudi Arabia and the UAE asked him to "
        "hold off an attack 'scheduled for tomorrow' because serious talks are "
        "under way; orders the military ready for 'a full, large scale assault'.",
        category="military", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 5, 19),
        label="Senate advances war powers res. 50-47",
        description="On the eighth attempt the Senate discharges S.J.Res. 185 "
        "(remove US forces from hostilities against Iran) 50-47; Collins, "
        "Murkowski, Paul, Cassidy join Democrats. Procedural, not passage.",
        category="political", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 5, 23),
        label="Trump: deal 'largely negotiated', walks back",
        description="Trump posts that an Iran agreement 'has been largely "
        "negotiated'; within 24 hours says it is not 'even fully negotiated "
        "yet'. Iran says no signing is imminent.",
        category="political", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 5, 25),
        label="US 'self-defense' strikes near Bandar Abbas",
        description="CENTCOM strikes two IRGC minelaying boats in Hormuz and a "
        "missile site near Bandar Abbas 'while using restraint during the "
        "ongoing ceasefire'. IRGC claims a downed MQ-9 and warns of retaliation.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 5, 28),
        label="Tentative US-Iran MOU; Trump undecided",
        description="Negotiators reach a tentative memorandum: 60-day ceasefire "
        "extension, Iran clears Hormuz mines within 30 days, US gradually lifts "
        "the blockade and eases oil sanctions. Vance: Trump's endorsement 'TBD'.",
        category="diplomatic", impact="dovish", importance=5,
    ),
    Event(
        date=date(2026, 5, 28),
        label="Levin: 'finish him off', no deal",
        description="Mark Levin attacks the reported MOU on air: frozen funds would "
        "flow to the IRGC, no word on ballistic missiles, 'a deal may be the "
        "worst way to attempt to end the war'.",
        category="media", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 6, 3),
        label="House passes war powers 215-208",
        description="House passes a resolution directing Trump to end US military "
        "involvement in Iran, 215-208, the first such measure to pass. Rubio "
        "tells House Foreign Affairs the initial phase of Epic Fury is over.",
        category="political", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 6, 6),
        label="Pope Leo: Iran war not a 'just war'",
        description="On the papal plane to Spain Pope Leo says 'in Iran, the "
        "criteria for a just war are not present', extending his May 25 "
        "encyclical Magnifica Humanitas, which called just-war doctrine outdated.",
        category="political", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 6, 7),
        label="Iran fires missiles at Israel",
        description="Iran launches ballistic missiles at Ramat David airbase after "
        "Israeli strikes on Beirut, its first launches at Israel since the April "
        "ceasefire. Israel strikes Tehran, Isfahan and Tabriz on Jun 8, then "
        "halts at Trump's request.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 6, 11),
        label="Trump threatens Kharg Island, cancels strikes",
        description="After a three-day exchange (Apache downed Jun 9, Tomahawk "
        "strikes Jun 10), Trump vows to hit Iran 'very hard tonight' and seize "
        "Kharg Island, then cancels hours later because 'talks are in progress'.",
        category="military", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 6, 14),
        label="Trump: Iran deal 'complete', blockade lifted",
        description="Trump posts 'The Deal with the Islamic Republic of Iran is "
        "now complete', authorizing toll-free reopening of Hormuz and "
        "immediate removal of the naval blockade. Source: Truth Social post "
        "116750587569914985 (12,932 replies); formal MOU signing Jun 17.",
        category="diplomatic", impact="dovish", importance=5,
    ),
    Event(
        date=date(2026, 6, 17),
        label="Islamabad MOU signed; blockade lifted",
        description="Trump (remotely from Versailles) and Pezeshkian sign the "
        "14-point Islamabad MOU: 60-day ceasefire extension, toll-free Hormuz, "
        "US blockade lifted (ended Jun 18), 60-day nuclear and sanctions talks.",
        category="diplomatic", impact="dovish", importance=5,
    ),
    Event(
        date=date(2026, 6, 17),
        label="Tucker: MOU a 'humiliating loss'",
        description="Tucker Carlson calls the MOU 'a pretty humiliating loss for "
        "the United States' and 'why people opposed a regime change war'; Mark "
        "Levin attacks the same deal from the pro-war side ('Release the damn MoU').",
        category="media", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 6, 22),
        label="Tucker: won't vote Republican",
        description="Carlson says there is 'no chance' he supports the GOP in the "
        "midterms, blaming the war on Netanyahu's pressure and calling support "
        "for the party 'immoral'.",
        category="media", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 6, 22),
        label="MTG: 'DONE with the America LAST Republican Party'",
        description="Marjorie Taylor Greene backs Carlson's break, posting that "
        "'a LOT of us are absolutely fed up and will not support a party that "
        "betrays its voters' and 'we are DONE with the America LAST Republican "
        "Party'; next day: 'too conservative to be a Democrat and too honest "
        "to be a Republican'.",
        category="media", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 6, 23),
        label="Senate passes war powers 50-48",
        description="Senate passes the concurrent resolution to end hostilities "
        "with Iran 50-48 (Paul, Murkowski, Collins, Cassidy with Democrats). "
        "Both chambers have now rebuked the war; no force of law.",
        category="political", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 6, 24),
        label="Quinnipiac: 60% say war not worth it",
        description="Quinnipiac: 60% say military action against Iran was not "
        "worth it, Trump's Iran approval 34/62, overall approval 38/55, record "
        "48% say the US is too supportive of Israel.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 6, 26),
        label="Pope: war 'never blessed by God'",
        description="Opening an extraordinary consistory of 178 cardinals on "
        "Magnifica Humanitas, Pope Leo says 'war is never worthy of humanity, "
        "and it is never blessed by God'.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 6, 27),
        label="US strikes resume; Iran hits Kuwait, Bahrain",
        description="After drone hits on the Ever Lovely (Jun 25) and tanker Kiku, "
        "CENTCOM strikes Iranian missile, radar, air-defense and minelayer "
        "sites over two nights; IRGC fires on US facilities in Kuwait and Bahrain.",
        category="military", impact="hawkish", importance=5,
    ),
    Event(
        date=date(2026, 7, 1),
        label="Carlson floats anti-war third party",
        description="In a CJR interview Carlson says 'there's going to be a third "
        "party, and I'm going to do everything I can to bring that about' and "
        "that he has not spoken to Trump since the war began.",
        category="media", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 7, 8),
        label="Trump: MOU 'over' at NATO summit",
        description="After Iran strikes shipping in Hormuz and the US hits 80+ "
        "targets overnight, Trump tells reporters at the Ankara NATO summit the "
        "MOU is over: 'I don't want to deal with them anymore. They're scum'.",
        category="military", impact="hawkish", importance=5,
    ),
    Event(
        date=date(2026, 7, 11),
        label="US hits ~140 targets, largest package",
        description="CENTCOM completes a third round of strikes on about 140 "
        "Iranian military targets. IRGC Navy declares Hormuz closed 'until "
        "further notice'; Iran fires on five Gulf states.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 7, 14),
        label="US naval blockade reinstated",
        description="The Navy reinstates the blockade of Iranian ports lifted "
        "under the MOU; the tanker Belma is disabled by missile fire near Kharg "
        "Island the next day. Trump drops a planned 20% Hormuz toll.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 7, 17),
        label="US troops killed in Jordan; MTG blasts Hegseth",
        description="Two US service members are killed defending Muwaffaq Salti "
        "Air Base from an Iranian missile and drone attack; a third is later "
        "confirmed dead. Marjorie Taylor Greene attacks Hegseth's response.",
        category="military", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 7, 21),
        label="Hegseth hearing: $37.5B cost, $67B ask",
        description="Hegseth and Gen. Caine tell Senate Appropriations the war has "
        "cost $37.5B and defend a $67B munitions supplemental; US deaths put at "
        "18. Sen. Peters: 'You, sir, are the failure.'",
        category="political", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 7, 22),
        label="Trump: a bridge or power plant per ship",
        description="On the 12th consecutive night of strikes Trump posts that "
        "each Iranian attack on a ship in Hormuz will cost 'ONE BRIDGE OR POWER "
        "PLANT'. Attends the dignified transfer of the Jordan dead.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 7, 23),
        label="House passes war powers 214-208",
        description="Second House rebuke of the war, 214-208 (Massie, Davidson, "
        "Fitzpatrick, Barrett with Democrats); the Senate rejects the companion "
        "discharge motion 47-49 the same day.",
        category="political", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 7, 24),
        label="Trump pauses strikes for mediation",
        description="After 13 consecutive nights Trump orders no strikes as "
        "Oman-mediated talks in Tehran progress; Iran says it will hold fire as "
        "long as the US does, while denying negotiations are under way.",
        category="diplomatic", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 7, 26),
        label="Pope: suspend attacks, reopen talks",
        description="At the Castel Gandolfo Angelus Pope Leo says intensified "
        "operations have 'again wreaked violence and destruction' and exhorts "
        "'all parties involved to suspend the attacks and urgently reopen paths "
        "of dialogue'.",
        category="diplomatic", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 7, 28),
        label="Iran missiles at Jordan base end pause",
        description="IRGC fires ballistic missiles at Muwaffaq Salti Air Base, "
        "all intercepted, Iran's first attack on a US base since the Jul 24 "
        "pause, as Trump hosts Netanyahu. US-Saudi strikes on militia sites in "
        "Iraq follow within hours.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 7, 29),
        label="Polls: 60-64% oppose war, Trump at 32%",
        description="Quinnipiac puts Trump at a record-low 32% with a majority "
        "opposing military action in Iran; AP-NORC (Jul 30) finds 64% say the "
        "war is not worth fighting.",
        category="political", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 7, 30),
        label="Senate rejects war powers 49-50",
        description="Senate rejects discharge of S.J.Res. 181, 49-50, after "
        "Fetterman again votes with Republicans; Collins, Murkowski and Paul "
        "vote with Democrats.",
        category="political", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 8, 2),
        label="Trump cancels attack: 'perimeters of a deal'",
        description="Trump posts that the US is 'locked and loaded' at levels 'not "
        "seen since World War II' but cancels the attack at Iran's and Gulf "
        "states' request because 'the perimeters of a deal has been agreed "
        "to', including full opening of Hormuz. Source: Truth Social post "
        "117023461141824050 (9,653 replies).",
        category="diplomatic", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 8, 3),
        label="Trump: Iran 'duplicitous', talks 'last chance'",
        description="After Tehran denies direct talks with Washington, Trump calls "
        "Iran's leadership 'unbelievably duplicitous' and says negotiations are "
        "Iran's 'last chance' to sign 'before decapitation'.",
        category="political", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 8, 17),
        label="MOU expires; Trump threatens Oman",
        description="The 60-day Islamabad window expires with no deal and no "
        "extension. Trump tells Fox Iran should 'put up the white flag' and "
        "'if Oman gets in the way, we'll bomb the s*** out of them'. Brent above $90.",
        category="diplomatic", impact="hawkish", importance=5,
    ),
    Event(
        date=date(2026, 8, 19),
        label="Trump declares 'economic D-Day'",
        description="Trump posts that the US will launch an 'economic D-Day' on "
        "Iran and punish any country giving it a lifeline; Bessent unveils the "
        "measures as 'Operation Economic Outcast' on Aug 24.",
        category="political", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 8, 24),
        label="Reuters/Ipsos: war support 31%",
        description="Reuters/Ipsos: support for military action against Iran at "
        "31% (37% in March), Republican support 69% (77% in March), Trump "
        "approval 33%, the lowest of either term.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 8, 26),
        label="Iran-Oman temporary Hormuz corridor",
        description="Iran and Oman agree a seven-mile commercial corridor through "
        "Hormuz and a joint mine-clearing project, without the US. Tehran says "
        "it does not consider the strait open.",
        category="diplomatic", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 8, 30),
        label="US strikes Larak Island; IRGC hits Jordan",
        description="First US strike on Iran in over a month hits IRGC rocket "
        "launchers on Larak Island said to be readying sea mines; IRGC fires "
        "missiles at King Hussein and Al-Azraq air bases in Jordan.",
        category="military", impact="hawkish", importance=5,
    ),
    Event(
        date=date(2026, 9, 1),
        label="US strikes southern Iran; Sirik wedding hit",
        description="CENTCOM strikes dozens of IRGC coastal targets after a month "
        "of relative calm. A strike on a wedding in Sirik County kills four to "
        "five including children and wounds 60+; Iran says 18 killed nationwide.",
        category="military", impact="hawkish", importance=5,
    ),
    Event(
        date=date(2026, 9, 1),
        label="Pezeshkian offers to honor MOU; Trump dismisses",
        description="At the SCO summit Pezeshkian says Tehran will 'immediately "
        "reciprocate' if the US returns to the Islamabad MOU; Trump dismisses "
        "talks.",
        category="diplomatic", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 9, 3),
        label="Vance: 'I wouldn't call it a war'",
        description="Vance tells the White House briefing 'I wouldn't call it a "
        "war... right now there is no active shooting', framing renewed strikes "
        "as not needing authorization, and is 'extremely skeptical' of the Sirik "
        "wedding reports: 'sometimes things happen'.",
        category="political", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 9, 4),
        label="Trump: war 'small potatoes'; Pickaxe threat",
        description="Trump defends Vance's framing, calls the conflict 'small "
        "potatoes' ('we lost 18 people'), and says the US 'may hit Pickaxe "
        "Mountain very soon'. White House: no talks until attacks on shipping stop.",
        category="political", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 9, 5),
        label="US destroys 3 Iranian tankers",
        description="After IRGC ballistic missiles target a US carrier and a "
        "destroyer, CENTCOM disables the crude carriers Downy and Stark 1 and "
        "destroys a third. Adm. Cooper: 'shoot at two of our ships... taking "
        "out three of yours'.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 9, 8),
        label="US destroys 5 tankers; Iran hits Jordan base",
        description="US forces destroy five IRGC-linked tankers after the IRGC "
        "targets a Navy warship twice in two days; overnight Iran fires a "
        "20-missile barrage at Muwaffaq Salti Air Base, 18 intercepted.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 9, 9),
        label="Trump: war ends 'immediately after our election'",
        description="At Joint Base Andrews and the Dallas midterm convention Trump "
        "says 'this war will end immediately after our election because they "
        "can't hold out any longer'; on talks: 'we're not looking for it'. "
        "Brent above $100.",
        category="political", impact="polarizing", importance=5,
    ),
    Event(
        date=date(2026, 9, 15),
        label="Polling avg: 34% support, 58% oppose",
        description="Silver Bulletin polling average: about 34% of Americans "
        "support the Iran war, 58% oppose.",
        category="media", impact="dovish", importance=3,
    ),
    # ── Midterm hold (Sep 16 -> Oct 31; verified 2026-10-31, sources in
    # docs/timeline_candidates_2026-09-16_to_2026-10-31.json) ─────────────
    Event(
        date=date(2026, 9, 17),
        label="Trump to Axios: 'big decision' on resuming strikes",
        description="In a Thursday Axios interview Trump said he faces a 'big decision' on "
        "whether to resume large-scale attacks on Iran and 'annihilate' its "
        "government ('Anything could happen with me'). He gave no timeline, said "
        "Tehran was still in contact and wanted a deal, and said he would first hear "
        "from the GCC states on the UNGA sidelines the next Tuesday. Axios also "
        "reported that Trump and Hegseth had told the military to hold Middle East "
        "force levels through year-end.",
        category="political", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 9, 17),
        label="IRGC claims strike on tanker Trend",
        description="The IRGC Navy said it hit the Togo-flagged tanker Trend, which caught fire "
        "and stopped, during an 'illegal passage' of Hormuz after being 'provoked and "
        "deceived' by the US military, and warned that such ships face destruction. "
        "UKMTO reported an attack on the Omani side of the strait without naming the "
        "vessel, and there is no independent confirmation. Date corrected from the "
        "candidate's Sep 18: Reuters carried the claim on Sep 17.",
        category="military", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 9, 17),
        label="Fox poll: 60% say war wrong, 71% see no plan",
        description="Fox News poll (1,211 registered voters, Sep 11-14), published Sep 17: 60% "
        "say military action against Iran was the wrong decision and 37% right. 71% "
        "say the administration lacks a plan to end the war, including 43% of "
        "Republicans. Trump's overall job approval is 39/61.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 9, 22),
        label="Trump at UN: deal or 'annihilate' Iran",
        description="Trump's UNGA address (Tue Sep 22), about 40 minutes and largely a defence of "
        "the war, framed a choice between a deal that lets Iran rebuild and quickly "
        "'annihilating' the Islamic Republic. He said a deal could come after the "
        "midterms and that the two sides would 'get it done, one way or the other'. "
        "Iran's delegation walked out. The meeting with GCC leaders on the sidelines "
        "had been set for the same day (Axios); no outcome was confirmed.",
        category="political", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 9, 22),
        label="Araghchi meets Witkoff, Kushner at the UN",
        description="Hours after Trump's speech, FM Araghchi met envoys Witkoff and Kushner on "
        "the UNGA sidelines, at the US side's request according to Iranian state TV. "
        "Trump said the session lasted about three hours, called it productive, and "
        "said another round would come 'in the very near future'. Accounts differ on "
        "whether the two sides met directly: the NYT, citing Iranian officials, said "
        "Pakistan and Qatar mediated. Iran set out its terms for reopening Hormuz: "
        "lift the naval blockade, release frozen assets, and end the war on all "
        "'resistance' fronts.",
        category="diplomatic", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 9, 23),
        label="Pezeshkian at UN: Iran 'ready for dialogue', won't bow",
        description="A day after Trump's 'annihilate' threat, President Pezeshkian gave a rare "
        "wartime UNGA speech. He said Iranians were the 'victims of terrorism' and "
        "held up images of the dead, including Ali Khamenei and the Minab "
        "schoolchildren. He told 'Mr. Trump and those who seek to bully us' that Iran "
        "is ready for negotiations 'without accepting the language of force'. The "
        "lone US delegate in the hall walked out.",
        category="diplomatic", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 9, 23),
        label="Bulk carrier Cape Dao hit; Indian sailor killed",
        description="The Antigua and Barbuda-flagged bulk carrier Cape Dao was hit about 2.5 nm "
        "off Oman's Musandam; UKMTO said by an unknown projectile, while one outlet "
        "reported torpedoes. One Indian crew member was killed and Oman's navy "
        "rescued the rest; the ship was left burning and adrift. No one claimed the "
        "attack.",
        category="military", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 9, 24),
        label="Senate rejects war powers 49-50; Tillis flips",
        description="The Senate voted 49-50 (Thu Sep 24) against the House-passed resolution to "
        "remove US forces from hostilities with Iran. Paul, Collins and Murkowski "
        "voted yes, joined for the first time by Tillis (after 12 'no' votes), who "
        "cited 60 days since the administration told Congress in July that 'military "
        "action' had restarted. Fetterman voted no and Alsobrooks did not vote.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 9, 24),
        label="Carlson on NPR: Trump said 'we'll all die anyway'",
        description="In an NPR Newsmakers interview with Steve Inskeep, Tucker Carlson said that "
        "when he warned Trump in the Oval Office days before the war, Trump answered "
        "'we'll all die anyway' (also quoted as 'we all die anyway'). Carlson called "
        "it nihilism and blamed Israel for pushing the war. The quote rests on "
        "Carlson's account alone. The White House did not address it and sent a Trump "
        "post calling Carlson and others 'stupid people'.",
        category="media", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 9, 24),
        label="Cohen files 26 impeachment articles, one on Iran",
        description="Retiring Rep. Steve Cohen (D-Tenn.) introduced 26 articles of impeachment "
        "against Trump. Article II charges abuse of power for waging war on Iran "
        "without congressional approval. Cohen acknowledged the resolution will not "
        "advance.",
        category="political", impact="dovish", importance=1,
    ),
    Event(
        date=date(2026, 9, 26),
        label="Trump rejects Iran's seven-day plan",
        description="Leaving the White House on Saturday Sep 26, Trump said Iran's seven-day plan "
        "'would not be acceptable' and that Iran wants the strait opened 'because "
        "they're losing so badly'. Under the plan the US would lift the blockade, "
        "waive oil sanctions, release about $12bn in frozen assets and observe a "
        "wider ceasefire including Lebanon and Yemen; Iran would then reopen Hormuz "
        "and begin final-deal talks. He also shared a map labelling the waterway the "
        "'Trump Strait'. A report cited by Fox said he told aides he expects strikes "
        "to resume after the midterms.",
        category="diplomatic", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 9, 29),
        label="IRGC open letter urges US voters to reject Trump",
        description="IRGC spokesman Hossein Mohebbi released an open letter 'to the American "
        "people' (25 or 26 pages; reports differ) urging voters to reject Trump's "
        "policies at the midterms. He said it was aimed at voters, not the "
        "administration. On Sep 30 a State Department spokesperson called it 'nothing "
        "more than a smokescreen'.",
        category="media", impact="polarizing", importance=2,
    ),
    Event(
        date=date(2026, 9, 30),
        label="Qatar relays messages; Iran gets US reply on 7-day plan",
        description="Qatari foreign ministry spokesman Majed Al-Ansari said on Tuesday Sep 29 "
        "that Doha is 'currently relaying messages between the American and Iranian "
        "sides'. One candidate dated this Oct 6, but Anadolu published it Sep 29. On "
        "Sep 30 Reuters reported that Araghchi had received Washington's feedback on "
        "Iran's seven-day trust-building plan from Qatari mediators in Doha. The "
        "dispute is over the order of steps, not the plan's content, despite Trump's "
        "public rejection.",
        category="diplomatic", impact="dovish", importance=2,
    ),
    Event(
        date=date(2026, 9, 30),
        label="Trump video: Iran 'crazy', 'maybe you blow them up'",
        description="In a Truth Social video post (9:36 PM EDT Sep 30) Trump calls Iran's leaders "
        "'crazy' and irrational and asks how a deal can be made with them. 'Maybe you "
        "blow them up': the choice is to 'blow them up, make a deal', and it will end "
        "'very soon one way or the other'.",
        category="political", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 10, 1),
        label="~9,000 troops, third carrier head to Mideast",
        description="AP reported on Thursday Oct 1 that about 9,000 US personnel were heading to "
        "the Middle East with the USS Theodore Roosevelt carrier group, which left "
        "San Diego, and the USS Makin Island amphibious ready group, as Trump weighs "
        "new strikes. He called a post-election bombing ramp-up 'possible'. A US "
        "official said the Roosevelt is due by the end of November, not October as "
        "one candidate had it.",
        category="military", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 1),
        label="AP-NORC: 69% say war not worth fighting",
        description="AP-NORC poll of more than 2,000 adults, released Thursday Oct 1: 69% say the "
        "Iran war has not been worth fighting, up five points since July, and 71% "
        "disapprove of Trump's handling of Iran.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 10, 4),
        label="Tanker attacks hit a weekly record",
        description="CNBC (Oct 9) counted 11 tankers attacked in Hormuz in the week ending Oct 4, "
        "the most strikes on commercial ships since the war began. Another analysis "
        "counted a dozen ships attacked Oct 5-11, also a weekly high. Reuters, citing "
        "a think tank, called October the worst month yet for ship attacks. The date "
        "marks the end of the record week, not a single incident.",
        category="military", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 10, 5),
        label="Trump: 'Let them take out Los Angeles'",
        description="At a midterm rally in Grand Island, Nebraska, Trump said of Iran: 'Let them "
        "take out Los Angeles. Let them take out San Diego. This is a very small "
        "price to pay.' Newsom called him 'deranged and dangerous', LA Mayor Bass "
        "objected, and the backlash crossed party lines. The White House said the "
        "remark was taken out of context and that he meant a nuclear Iran could "
        "strike US cities.",
        category="political", impact="polarizing", importance=4,
    ),
    Event(
        date=date(2026, 10, 6),
        label="Vance: Iran must cut enrichment to end war",
        description="In a Reuters interview given Monday Oct 5 in Fairbanks and published Oct 6, "
        "Vance said any deal needs a 'meaningful' reduction in Iran's enrichment "
        "capacity, not assurances: 'If you don't want a nuclear weapon, then why do "
        "you need 60% enriched fuel?' He said it is unclear how much authority Iran's "
        "negotiators hold. A senior Iranian official replied that recognition of "
        "Iran's right to enrich is Tehran's red line.",
        category="diplomatic", impact="dovish", importance=2,
    ),
    Event(
        date=date(2026, 10, 8),
        label="Trump: no attack on Iran before midterms",
        description="On Thursday Oct 8 Trump posted on Truth Social that 'we will not be "
        "attacking Iran at any time prior to the Midterm Elections' (Nov 3), citing "
        "'productive discussions' with Iran. The blockade and sanctions stay. It "
        "followed an Atlantic report that the White House had asked the Pentagon for "
        "strike options usable before the vote, and an oil-price surge. It frames the "
        "rest of the window as a pre-midterm hold: there have been no US strikes on "
        "Iran since early September.",
        category="political", impact="dovish", importance=5,
    ),
    Event(
        date=date(2026, 10, 8),
        label="Treasury sanctions 17 shadow-fleet tankers",
        description="Under Operation Economic Outcast, Treasury sanctioned 17 vessels in Iran's "
        "shadow fleet on Oct 8, saying this neutralizes most of what remains of the "
        "network. A same-day State Department action added entities, individuals and "
        "five vessels; Iran International put the total at 22 vessels.",
        category="diplomatic", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 10, 8),
        label="Pew: 65% disapprove; GOP approval slips to 63%",
        description="Pew survey (Sep 28-Oct 4, 5,062 adults), published Oct 8: 65% disapprove of "
        "Trump's handling of the Iran military action and 33% approve. 34% say the "
        "decision to use force was right, down 4 points from March. Republican "
        "approval is 63%, down from 69% in March. 47% expect the war to last more "
        "than another year.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 10, 8),
        label="Carlson calls the war 'treason'; WH: 'whiny turncoat'",
        description="Tucker Carlson told Arabian Business that Trump's 'voluntary, ruinous' war "
        "with Iran amounts to treason against the US. A White House spokesman did not "
        "dispute his account, restated that Iran will never get a nuclear weapon, and "
        "called Carlson 'a whiny turncoat who dramatically overestimates his "
        "influence and relevance'. The search reported the article as dated Oct 8; "
        "the page itself was not opened.",
        category="media", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 10, 10),
        label="Houthi missile hits Riyadh airport, 12-13 killed",
        description="A Houthi missile struck King Khalid International Airport in Riyadh on "
        "Saturday Oct 10. Saudi authorities first said 12 were killed and 309 "
        "injured; later reports say at least 13, the deadliest attack on Saudi soil "
        "in over a decade. An Oct 8 strike had killed three, and Riyadh was targeted "
        "again on Oct 11-12. The US later ordered non-emergency staff out of Saudi "
        "Arabia.",
        category="military", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 13),
        label="Container ship GFS Pride sunk near Hormuz",
        description="The Singapore-flagged container ship GFS Pride was hit on Oct 13, caught "
        "fire and had sunk by Oct 14. Singapore's MPA said all 21 crew were "
        "evacuated; CNN reported one death, unconfirmed elsewhere. Singapore cited an "
        "unidentified projectile; the IRGC said the ship hit a naval mine. Its sister "
        "ship GFS Giselle was damaged. The IMO counts it as the third commercial ship "
        "sunk in the war. CNN tied the surge in attacks to the midterms.",
        category="military", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 14),
        label="Trump: 'should have probably left their leaders'",
        description="At a Wednesday-night rally in Kalispell, Montana, for House candidate Aaron "
        "Flint, Trump said Iran has no navy or air force and its leaders are gone, "
        "then added: 'I should have probably left their leaders because I don't know "
        "who the hell we're negotiating with.' He said Iran's main ask in talks is "
        "'Will you please allow us to live?'. The Iran International post decodes to "
        "Oct 15 03:36 UTC, which is Oct 14 evening US time.",
        category="political", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 10, 15),
        label="Trump rules out nukes; MOAB strike plans reported",
        description="To Newsmax (Oct 15) Trump said 'I wouldn't use nuclear, we don't have to', "
        "called the 'mother of all bombs' conventional, and on ground troops for a "
        "Pickaxe Mountain strike said 'No, I don't see that.' The same day CNN "
        "reported new US strike plans, including possible MOAB use, against Iranian "
        "energy and nuclear sites. Bloomberg said he declined to rule out using the "
        "MOAB after the midterms.",
        category="political", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 10, 16),
        label="Polls: war a top midterm issue; FP approval lows",
        description="Fox News poll (Oct 9-12, 1,202 RV): 75% extremely or very concerned about "
        "the war, six in ten oppose military action, and Trump's foreign policy "
        "approval is 35%, Fox's lowest of his second term. AP-NORC (Oct 8-12): 62% of "
        "registered voters call the war one of the most important issues, and foreign "
        "policy approval is 33%, the low across eight polls this term. The date is "
        "the Responsible Statecraft roundup (Oct 16); the two polls' release days "
        "were not seen.",
        category="political", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 10, 17),
        label="Pakistan suspends trade at Iran border crossings",
        description="Pakistan's Commerce Ministry withdrew its Oct 3 order allowing trade through "
        "the Taftan and Gabd-Rimdan crossings, citing security and smuggling. "
        "Commercial traffic stopped from Oct 17 while passenger traffic continued. "
        "Iran said the border stayed open, and trade only partly resumed by Oct 26.",
        category="diplomatic", impact="hawkish", importance=1,
    ),
    Event(
        date=date(2026, 10, 17),
        label="Vance: 'in some ways we have won decisively'",
        description="At a CBS News 'Things That Matter' town hall in Michigan that aired Saturday "
        "evening (Oct 17), Tony Dokoupil asked why the US hasn't decisively won. "
        "Vance said 'in some ways we have won decisively' but 'the enemy gets a vote "
        "here'. A man whose wife died in an Iranian strike pressed him on "
        "'fundamental failures'.",
        category="political", impact="polarizing", importance=2,
    ),
    Event(
        date=date(2026, 10, 19),
        label="Hegseth meets Netanyahu in unannounced visit",
        description="Hegseth met Netanyahu and Defense Minister Katz at Ovda Air Base during an "
        "unannounced visit, first reported by Axios, eight days before Israel's "
        "election. Iran dominated the talks, which came amid US-Israeli discussion of "
        "resuming major combat. Hegseth said only that he visited troops and spoke "
        "with the prime minister. Some outlets date it Oct 20, but Hegseth confirmed "
        "Oct 19.",
        category="diplomatic", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 10, 20),
        label="Rezaei: no uranium leaves Iran; Netanyahu vows regime's end",
        description="SNSC secretary Mohsen Rezaei told visiting Pakistani army chief Asim Munir "
        "that Iran will not hand over enriched uranium under any deal and will not "
        "negotiate until the US builds trust; Speaker Ghalibaf echoed him. The same "
        "day Netanyahu said at Kiryat Shmona, 'We must complete the elimination of "
        "the Iranian regime — that is the primary objective.'",
        category="diplomatic", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 20),
        label="Rubio: Iran using 'Nazi U-boat tactics'",
        description="In an interview with India's News18 during a Mumbai visit, Rubio said Iran "
        "is targeting merchant ships and trying to kill civilian sailors with Nazi "
        "U-boat-style tactics. He said a deal is hard with 'radical Shia clerics' in "
        "charge and promised 'extraordinary economic pressure', while claiming Hormuz "
        "traffic was back to 80-100% of prewar levels.",
        category="diplomatic", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 10, 21),
        label="Reports: Trump blocked Israeli strike; post-midterm plan",
        description="Two reports on Oct 21. The FT, citing three sources, said Israel had "
        "prepared a multi-day campaign against Iranian nuclear, power and regime "
        "targets, aimed at around the weekend of Oct 10, which Trump opposed before "
        "the midterms; Netanyahu called the report 'biased and untrue'. The NYT said "
        "the US and Israel had drawn up plans for a 'short but intensive' operation "
        "after Nov 3, possibly including Pickaxe Mountain. The White House did not "
        "comment on plans 'that may or may not exist'.",
        category="military", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 10, 21),
        label="MTG: attack Iran and El-Sayed wins Michigan",
        description="Marjorie Taylor Greene posted on X (Oct 21, 21:40 UTC, from @FmrRepMTG): 'If "
        "Trump is dumb enough to attack Iran again before the midterms, @AbdulElSayed "
        "will win the senate seat in Michigan. It's a stupid senseless war and "
        "America hates it.' She said it was not an endorsement.",
        category="media", impact="dovish", importance=2,
    ),
    Event(
        date=date(2026, 10, 22),
        label="Trump: 'not negotiating'; calls ship-hit claims fake",
        description="Asked by NewsNation (reported late Oct 21 to Oct 22, US time) whether he was "
        "done negotiating, Trump said talks were happening 'a little bit', then 'I "
        "think we're not negotiating', that something will happen 'fairly soon', and "
        "that 'most people seem to think it's after the election.' On Oct 22 the IRGC "
        "claimed it had set four ship-to-ship tankers on fire in the northern Gulf "
        "and that three more had hit mines west of Hormuz. CENTCOM called the mine "
        "claims false. In a Truth Social post at 11:40 AM EDT, Trump accused Iran of "
        "making up attacks and mines to drive up oil before the midterms.",
        category="political", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 23),
        label="Trump: 'We may do another little Epic Fury'",
        description="At a Friday rally in Fort Myers, Florida, Trump said 'We may do another "
        "little Epic Fury, or they'll agree.' The same day, envoy Tom Barrack told "
        "Mario Nawfal that a US strike could come as soon as that weekend; Trump said "
        "he knew nothing about it.",
        category="political", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 23),
        label="DNI memo: Iranian plots in 'advanced stages'",
        description="An unclassified memo from DNI Jay Clayton to Rubio, dated Oct 23 and first "
        "reported by Axios (dated Oct 24), warns that Iranian lethal plots against US "
        "officials have reached 'advanced stages'. State ordered non-emergency staff "
        "out of the Riyadh embassy and its consulates; reports date that order Oct "
        "21-23, and the public announcement cited Houthi threats.",
        category="diplomatic", impact="hawkish", importance=2,
    ),
    Event(
        date=date(2026, 10, 23),
        label="European, Mideast FMs meet in Istanbul on Hormuz",
        description="Turkey's FM Fidan hosted the foreign ministers of Britain, France, Germany "
        "and Italy with Saudi Arabia, Pakistan and Egypt, the first meeting in this "
        "format. They urged restraint and diplomacy, the reopening of Hormuz and an "
        "end to Houthi attacks on Saudi Arabia. Fidan said military intervention is "
        "not a solution.",
        category="diplomatic", impact="dovish", importance=2,
    ),
    Event(
        date=date(2026, 10, 23),
        label="Alex Jones: Trump wrong on Iran, vote GOP anyway",
        description="Alex Jones on X (status decodes to Oct 23 17:24 UTC): 'Yes Trump Was Wrong "
        "To Start The Iran War, But We Must Live In Reality And We Cannot Flush Our "
        "Country Down The Toilet And Let The Democrat Socialists Win. Vote For Your "
        "Local Republicans!' He called for impeachment in July, so this marks an "
        "anti-war MAGA account closing ranks before the vote. @RealAlexJones is a "
        "tracked account.",
        category="media", impact="polarizing", importance=2,
    ),
    Event(
        date=date(2026, 10, 25),
        label="Trump: 'deal or finish them off?'; Qatari draft",
        description="On Fox's Sunday Morning Futures (Oct 25) Trump asked 'Do I make a deal with "
        "them or do I finish them off?' ('Getting to be decision time') and said "
        "there would be no strike that day. Per an official briefed on the talks "
        "(Reuters, reported Oct 28), Qatari mediators gave Iran a new draft that day: "
        "daily reciprocal steps over seven days, with guarantor states. A senior "
        "Iranian official called it 'unacceptable ideas and not an offer'.",
        category="diplomatic", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 10, 26),
        label="Mecca Pact envoys in Tehran to avert renewed war",
        description="Saudi Defence Minister Khalid bin Salman, Pakistan's Field Marshal Asim "
        "Munir and Turkish intelligence chief Ibrahim Kalin met Pezeshkian and Rezaei "
        "in Tehran on Monday Oct 26. They came as members of the Mecca (Makkah) Joint "
        "Defence Agreement, to de-escalate between Iran and the US and over the "
        "Houthi attacks on Saudi Arabia. Iran International called it a possible "
        "'last chance' before military action. No agreement was announced.",
        category="diplomatic", impact="dovish", importance=2,
    ),
    Event(
        date=date(2026, 10, 27),
        label="Israel election: neither bloc reaches 61",
        description="Final count of the Oct 27 Knesset election: Netanyahu's bloc 53 seats, the "
        "Eisenkot-led bloc 48, Arab parties 19. Neither side can form a government "
        "without the other or the Arab parties, so Netanyahu stays on as caretaker "
        "during coalition talks. Turnout was 77.9%.",
        category="political", impact="polarizing", importance=2,
    ),
    Event(
        date=date(2026, 10, 27),
        label="Pope: 'There is no such thing as a holy war'",
        description="Recovering from his Oct 13 lung-cancer surgery, Pope Leo sent a video "
        "message to the Assisi interfaith gathering marking 40 years since John Paul "
        "II's 1986 prayer for peace: 'There is no such thing as a holy war: Only "
        "peace is holy.' He condemned the 'revival of war as an instrument of "
        "international politics' and religious justifications for war, without naming "
        "Iran or the US.",
        category="media", impact="dovish", importance=3,
    ),
    Event(
        date=date(2026, 10, 27),
        label="Trump: 'You'll see Iran in a couple of weeks'",
        description="Not in either researcher's list; found while verifying. At a Lubbock, Texas "
        "rally Trump said 'Venezuela, Iran, you'll see Iran in a couple of weeks, if "
        "you don't mind', and that Iran was 'defeated militarily, financially and "
        "otherwise. And it will all end very soon.' The Washington Examiner read it "
        "as a hint that strikes resume after the midterms.",
        category="political", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 28),
        label="Vance letter to Rezaei; Trump: 'medical supplies'",
        description="On Oct 28 the WSJ and NYT reported a letter from Vance to SNSC secretary "
        "Mohsen Rezaei, passed through Qatari mediators on Sunday (Oct 25). It "
        "proposes direct talks with the nuclear programme on the table from the "
        "start, and the WSJ said Trump directed it. Trump said he had not and that "
        "the letter was about 'medical supplies and various things'. The WaPo later "
        "reported that Vance offered to meet Rezaei in person, likely in Doha or "
        "Istanbul. Tasnim said Iran had not yet replied.",
        category="diplomatic", impact="dovish", importance=4,
    ),
    Event(
        date=date(2026, 10, 28),
        label="US tells allies to prepare for post-midterm strikes",
        description="AP, citing two diplomats, reported on Oct 28 that US officials told Middle "
        "East allies to prepare for possible renewed strikes on Iran after the Nov 3 "
        "midterms, and for Iranian retaliation against energy and other "
        "infrastructure. Gulf officials told MS NOW the first round could come as "
        "early as Nov 4 and would be brief but intense. AP said Trump had not made a "
        "final decision.",
        category="military", impact="hawkish", importance=4,
    ),
    Event(
        date=date(2026, 10, 28),
        label="Hegseth fires Joint Staff planner Lt. Gen. Sylvia",
        description="Hegseth removed Lt. Gen. Brett Sylvia as the Joint Staff's director of "
        "strategy, plans and policy (J5). CBS said he was coordinating the Iran "
        "battle plan; a War Department official denied he was central to it. A "
        "Pentagon official alleged leaks of classified information on Iran and "
        "Ukraine and referred the case to the Justice Department; another source "
        "cited lack of confidence. He is the second J5 ousted under Hegseth.",
        category="political", impact="polarizing", importance=2,
    ),
    Event(
        date=date(2026, 10, 28),
        label="Murphy: post-midterm strike plan 'insane'",
        description="At a Middle East Institute event, Sen. Chris Murphy said that after a "
        "briefing on the war plan, 'the plan is insane'. He said it would not achieve "
        "its objectives, could hit civilian infrastructure, and would set off "
        "'devastating escalation'. IBTimes dates the event Wednesday Oct 28; the "
        "Daily Beast says Thursday.",
        category="political", impact="dovish", importance=2,
    ),
    Event(
        date=date(2026, 10, 30),
        label="Brent at 6-month high; first limpet-mine hit",
        description="The expiring December Brent contract settled up 2.4% at $117.64 on Friday "
        "Oct 30, its highest close since late April (January: $113.43). Brent rose "
        "13.6% in October and WTI 15.2%. UKMTO said a small boat attached explosives "
        "to a tanker in Hormuz, which maritime sources called the first known "
        "successful limpet-mine strike of the war. The IRGC claimed more tankers hit "
        "mines; the US called that claim false. The IMO counted 121 attacks and 24 "
        "seafarer deaths as of Oct 27.",
        category="military", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 30),
        label="Polling avg: 36% support, 57% oppose",
        description="Silver Bulletin's Iran war polling average, updated Oct 30: 36% support and "
        "57% oppose, up from 32/60 at the start of October. The existing Sep 15 event "
        "recorded 34/58.",
        category="political", impact="polarizing", importance=3,
    ),
    Event(
        date=date(2026, 10, 30),
        label="Trump: Iran war 'going to get wrapped up soon'",
        description="On Kansas City radio KCMO (Fri Oct 30) Trump said the war will be 'wrapped "
        "up soon', without a timeline, and 'we need a guarantee there'll never be a "
        "nuclear weapon'. At the White House he said Iran keeps seeking talks and the "
        "US is in no rush. The WSJ reported that the CENTCOM commander had presented "
        "strike options to him. Qatari mediators returned to Tehran the same day.",
        category="political", impact="polarizing", importance=2,
    ),
    Event(
        date=date(2026, 10, 31),
        label="Khamenei orders 'geographically unrestricted' response",
        description="After his first publicly disclosed meeting with the armed forces chiefs "
        "since the war began, Supreme Leader Mojtaba Khamenei said on Saturday Oct 31 "
        "that 'any level or type of threat must be met with an immediate, impactful "
        "and geographically unrestricted response'. He ordered the target list "
        "updated to cover all 'American-Zionist' assets and said Iran will decide "
        "when and how any renewed war ends.",
        category="military", impact="hawkish", importance=3,
    ),
    Event(
        date=date(2026, 10, 31),
        label="Rubio: Trump 'hasn't made a decision' on strikes",
        description="Closing the G20 foreign ministers' meeting in Atlanta on Saturday Oct 31, "
        "Rubio said Trump 'hasn't made a decision' on resuming strikes, prefers a "
        "diplomatic solution ('If there is a diplomatic solution, we would be open to "
        "it') and has set no deadline, while not ruling out military action. The "
        "meeting produced no joint communiqué.",
        category="diplomatic", impact="dovish", importance=2,
    ),
]

# Date range for primary analysis window
ANALYSIS_START = date(2026, 2, 1)
# Frozen at the Sep 18 -> Nov 1 pull (2026-11-01): the cycle ends before the Nov 3
# midterms and any post-election strikes (user's call, 2026-11-02).
ANALYSIS_END = date(2026, 11, 1)

# The five phases the README tables and `phases` command report on:
# (label, first day, last day), both inclusive.
PHASES = [
    ("strikes, ceasefire", date(2026, 2, 1), date(2026, 4, 21)),
    ("talks, MOU", date(2026, 4, 22), date(2026, 6, 17)),
    ("collapse, blockade", date(2026, 6, 18), date(2026, 8, 17)),
    ("expiry, strikes", date(2026, 8, 18), date(2026, 9, 18)),
    ("midterm hold", date(2026, 9, 19), date(2026, 11, 1)),
]
