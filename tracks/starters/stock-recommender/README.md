# Starter — stock research assistant

**Not the 90-day solution.** One vertical slice: `/research` (or `research()`) returns a card whose **numbers come from `fixtures/quotes.json`**, never from a model, and whose UX is **not advice**.

Full track: [docs/tracks/stock-recommender.md](../../../docs/tracks/stock-recommender.md).

## Run

```bash
cd tracks/starters/stock-recommender
python3 -c "from app import research; print(research('AAPL'))"
# Optional HTTP:
# pip install fastapi uvicorn
# uvicorn app:app --reload
# GET /healthz   GET /research?symbol=AAPL
```

```bash
python3 -m pytest tests/test_slice.py -v
```

## What this slice proves

- Prices are **tools/fixtures**, not generated text.
- Unknown tickers **fail closed**.
- The card carries a **non-advice** disclaimer.

## What it does not prove (milestones in PROGRESS.md)

Time-safe split, RAG citations, PEFT, compression, live quote tools, FastAPI+Docker+CI as a product. Do not invent a 10-K section to fill `citations: []`.
