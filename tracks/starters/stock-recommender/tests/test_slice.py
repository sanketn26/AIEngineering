"""One working slice: fixture quotes in, research card out, non-advice UX."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import DISCLAIMER, lookup_quote, research  # noqa: E402


def test_research_uses_fixture_quote_and_disclaimer():
    card = research("aapl")
    quote = lookup_quote("AAPL")
    assert card["symbol"] == "AAPL"
    assert card["quote"]["close"] == quote["close"]
    assert card["quote"]["source"] == "fixture"
    assert card["disclaimer"] == DISCLAIMER
    assert "not personalized investment advice" in card["disclaimer"].lower()
    assert card["citations"] == []


def test_unknown_symbol_fails_closed():
    try:
        research("NOTATICKER")
    except KeyError:
        return
    raise AssertionError("unknown ticker must not invent a price")
