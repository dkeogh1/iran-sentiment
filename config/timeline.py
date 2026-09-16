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
]

# Date range for primary analysis window
ANALYSIS_START = date(2026, 2, 1)
ANALYSIS_END = date(2026, 5, 12)  # last DATA refresh. Events run to Sep 15 2026;
                                  # move this forward when the data is re-pulled.
