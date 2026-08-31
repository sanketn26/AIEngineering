"""Day-1 vertical slice: research card from *fixture* quotes.

    uvicorn app:app --reload   # needs fastapi (see capstone-starter/requirements.txt)

Prices never come from the model. Unknown tickers fail closed.
This is not a 90-day solution — see PROGRESS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "quotes.json"
DISCLAIMER = (
    "Educational research-assistant slice only. Not personalized investment "
    "advice. Quotes are fixtures, not a live feed."
)


def load_quotes() -> dict:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return payload["quotes"]


def lookup_quote(symbol: str) -> dict:
    quotes = load_quotes()
    key = symbol.strip().upper()
    if key not in quotes:
        raise KeyError(f"no fixture quote for {key!r}")
    return dict(quotes[key])


def research(symbol: str) -> dict:
    """Stub research card. RAG citations and classical signal are TODOs."""
    quote = lookup_quote(symbol)
    return {
        "symbol": quote["symbol"],
        "quote": quote,
        "narrative": (
            f"{quote['symbol']} fixture close is {quote['close']} "
            f"{quote['currency']} as of the fixture date. No live price, "
            "no 10-K, no recommendation."
        ),
        "citations": [],  # Gate/track: RAG must fill this from retrieved docs
        "disclaimer": DISCLAIMER,
    }


def create_app():
    from fastapi import FastAPI, HTTPException

    api = FastAPI(title="Stock research starter", version="0.1.0")

    @api.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @api.get("/research")
    def research_endpoint(symbol: str = "AAPL"):
        try:
            return research(symbol)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return api


try:
    app = create_app()
except ImportError:  # tests import research() without FastAPI
    app = None
