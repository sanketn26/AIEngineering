# Hands-on exercises (repo package)

These exercises use the **runnable** `src/` package. Complete them alongside the matching core module.

```bash
poetry install
poetry run pytest tests/ -v
```

---

## EX-01 — Templates (`src.prompts`)

1. Render `email_reply` with a real email snippet.  
2. Add a new template `bug_triage` in `src/prompts.py` + a unit test.  
3. Log three outputs at different temperatures (API optional).

**Check:** `pytest tests/test_prompts.py -v`

---

## EX-02 — Security (`src.security`)

1. Run `sanitize_user_text` on three injection strings; all should flag.  
2. Redact a paragraph containing email + phone.  
3. Wire `prepare_user_message` before any mock “LLM call” function.

**Check:** `pytest tests/test_security.py -v`

---

## EX-04 — Golden evals (`src.evals`)

1. Open `tests/fixtures/invoice_golden.jsonl`.  
2. Write a `predict(text) -> dict` heuristic (regex is fine).  
3. Use `run_suite` and print accuracy; improve until ≥ 0.66 on the fixture.

**Check:** `pytest tests/test_evals.py -v`

---

## EX-05 — Memory budget (`src.context_memory`)

1. Fill `SessionMemory` with 15 turns; set `max_recent=5`.  
2. Assert `build_messages` length stays bounded.  
3. Use `fit_budget` to drop low-priority history under a tight budget.

**Check:** `pytest tests/test_context_memory.py -v`

---

## EX-07 — Tiny RAG (`src.rag`)

1. Chunk two short notes (company handbook style).  
2. Ask an answerable and unanswerable query.  
3. Validate citations with `validate_citations`.

**Check:** `pytest tests/test_rag.py -v`

---

## EX-10 — Cost controls (`src.cost`)

1. Route `classify` vs `complex_reason` with `ModelRouter`.  
2. Cache one payload; assert second `get` hits.  
3. Enforce a $1.00 user budget with `UsageLedger`.

**Check:** `pytest tests/test_cost.py -v`

---

## EX-11 — Agent loop (`src.agents`)

1. Implement tools `add(a,b)` and `echo(text)`.  
2. Script an LLM stub that calls `add` then `final`.  
3. Confirm repeated identical tool calls abort.

**Check:** `pytest tests/test_agents.py -v`

---

## EX-14 — Audit log (`src.audit`)

1. Record three tool events with hashed inputs.  
2. Write JSONL to a temp path via `AuditLog`.  
3. Ensure raw secrets never appear in the log file.

**Check:** `pytest tests/test_audit.py -v`

---

## Track stretch

| Track | Stretch exercise |
|-------|------------------|
| Stock | Replace `TinyRAG` bag-of-words with real embeddings; keep citation tests |
| Hybrid | Export a tiny MLP from PyTorch; serve predict via FastAPI sketch |
| Agentic plugin | Run `Agent` over `read_file` / `list_files` tools on this repo (read-only) |
