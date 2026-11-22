"""config/timeline.py: the phases (closed 1-5, phase 6 rolling with
ANALYSIS_END since 2026-11-22) and the events' fields. No data, no API."""

from datetime import date
from itertools import pairwise

from config import timeline


def test_phases_tile_the_window_and_only_the_last_rolls():
    ps = timeline.PHASES
    assert ps[0][1] == timeline.ANALYSIS_START and ps[-1][2] == timeline.ANALYSIS_END
    assert all((b[1] - a[2]).days == 1 for a, b in pairwise(ps))  # no gap, no overlap
    assert all(start <= end for _, start, end in ps)
    # Phases 1-5 are closed and keep their published dates (decisions.md 2026-11-22).
    assert ps[:5] == [
        ("strikes, ceasefire", date(2026, 2, 1), date(2026, 4, 21)),
        ("talks, MOU", date(2026, 4, 22), date(2026, 6, 17)),
        ("collapse, blockade", date(2026, 6, 18), date(2026, 8, 17)),
        ("expiry, strikes", date(2026, 8, 18), date(2026, 9, 18)),
        ("midterm hold", date(2026, 9, 19), date(2026, 11, 1)),
    ]
    assert ps[5][:2] == ("midterms, standoff", date(2026, 11, 2))
    assert len({label for label, _, _ in ps}) == len(ps)


def test_events_have_valid_fields():
    for e in timeline.EVENTS:
        assert e.category in {"military", "diplomatic", "protest", "media", "political"}, e
        assert e.impact in {"hawkish", "dovish", "polarizing"}, e
        assert 1 <= e.importance <= 5 and len(e.label) <= 60, e
    late = [e.date for e in timeline.EVENTS if e.date >= date(2026, 11, 1)]
    assert late == sorted(late) and max(late) <= timeline.ANALYSIS_END
