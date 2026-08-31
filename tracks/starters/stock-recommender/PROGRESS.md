# Stock starter — milestone TODOs

Tick only when the slice in this folder grew the *capability*, not when you read the track page.

- [x] Day 1: fixture quote → research card + disclaimer (`tests/test_slice.py`)
- [ ] **Time-safe split** — lagged features; train cutoff before val; no shuffle on the series
- [ ] **Quote tool** — live (or recorded) OHLCV; model still cannot invent a close
- [ ] **RAG citations** — filings/news chunks with ids; `citations` non-empty and validated
- [ ] **Non-advice UX** — API, README, and any UI still say educational; no "buy" badge
- [ ] **Evals** — golden research cards: must-cite, must-refuse, no invented prices
- [ ] **Ship** — FastAPI `/healthz` + `/research`, Docker, CI eval subset

Track rubric: [docs/reference/assessment.md](../../../docs/reference/assessment.md).
