"""Shared test setup."""

import pytest

from config import settings


@pytest.fixture(autouse=True)
def _reply_population_col(monkeypatch):
    """The test stubs' replies carry score_opus_distilled; the published reply
    figures' census column (settings.REPLY_POPULATION_COL, a GPU cross-fit
    since 2026-11-23) is one they don't build. A test that needs it sets it."""
    monkeypatch.setattr(settings, "REPLY_POPULATION_COL", "score_opus_distilled")
