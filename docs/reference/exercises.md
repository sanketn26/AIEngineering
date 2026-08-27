# Hands-on exercises (repo package)

Complete these alongside the matching core module. Numbered `src.*` exercises hit the teaching package; the others are the module labs (no extra library required).

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

## EX-03 — Structured extract (Module 03 lab)

No extra package code — use the Module 03 lab:

1. Pydantic model for an invoice (or your domain).  

2. Two few-shot edge cases.  

3. `parse_success_rate` on ≥20 raw strings (model or hand-crafted).  

4. Justify CoT yes/no in one paragraph.

**Check:** `pytest tests/test_prompts.py -v` plus your parser tests.

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

## EX-06 — Fine-tune or not (Module 06 lab)

1. One-page decision memo: why FT vs RAG/tools.  

2. 30 train + 10 held-out instruction rows, no PII.  

3. Baseline score on the 10 (API or local). GPU LoRA is optional.

---

## EX-07 — Tiny RAG (`src.rag`)

1. Chunk two short notes (company handbook style).  

2. Ask an answerable and unanswerable query.  

3. Validate citations with `validate_citations`.

**Check:** `pytest tests/test_rag.py -v`

---

## EX-08 — MCP policy (Module 08 lab)

1. Read [modelcontextprotocol.io](https://modelcontextprotocol.io/).  

2. In **dev only**, list tools from a reviewed filesystem/git server.  

3. Write `mcp-policy.md`: allowed servers per env, approval-required tools, pin/update process.
4. Run `pytest tests/test_mcp_prod.py -v` and extend one case (e.g. extra write tool blocked in CI).

---

## EX-09 — Hybrid retrieval (Module 09 lab)

1. 20 questions with `must_have` ids (include keyword/ID and multi-hop).  

2. Dense-only vs `rrf` hybrid **Hit@5** and **MRR**.  

3. Log intermediate queries for 5 multi-hop items.

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

## EX-12 — Multi-agent vs single (Module 12 lab)

1. Researcher → writer → critic with max 2 critique rounds.  

2. Structured payloads; reject invalid JSON.  

3. On 10 tasks, compare success and cost to a single `src.agents.Agent`.

---

## EX-13 — Production endpoint (Module 13 lab)

1. FastAPI `/healthz` + `/v1/generate` with `request_id`.  

2. Timeout + fallback (stubs OK).  

3. Dockerfile; 5-case PR eval subset.

---

## EX-14 — Audit log (`src.audit`)

1. Record three tool events with hashed inputs.  

2. Write JSONL to a temp path via `AuditLog`.  

3. Ensure raw secrets never appear in the log file.

**Check:** `pytest tests/test_audit.py -v`

---

## EX-15 — Vertical refuse path (Module 15 lab)

1. One-page policy: allowed / refused / escalate.  

2. 10-case eval with ≥3 must-refuse.  

3. `policy_check` + audit event on refuse.

---

## EX-16 — Jobs or hybrid route (Module 16 lab)

1. `POST /jobs` → worker (in-memory queue is fine).  

2. Server-side `data_class` routing to two stub endpoints.  

3. `request_id` in API log and worker log.

---

## EX-17 — Local SLM vs mini (Module 17 lab)

1. Run a 3B–8B-class local model on 20 golden tasks.
2. Score vs a cloud mini model.
3. Escalate on schema fail; record which tasks the SLM owns.
4. Print `recommend_local_setup(HardwareBudget(ram_gb=...))` for your machine; the model in step 1 should fit that row (no swap).

**Check:** `pytest tests/test_local_agents.py -v`

---

## EX-18 — Leaf patterns (Module 18 lab)

Apply **three** of: Subroutine (validated output), Tool Gate (split messages), Rejection Sampler (`max_trials`), Consensus (n=5 + entropy), Adaptive Retriever. Name the failure each one fixes.

---

## EX-19 — Orchestration shape (Module 19 lab)

Apply **three** of: Map-Reduce, Router, Planner, ReAct, Memory, Duet to **one** workflow. See the Module 19 lab for the combo rule.

---

## EX-20 — Failure detectors (`src.reliability`)

1. Build a two-step trajectory with identical `search` args; assert `runaway_loop`.
2. Propose a tool not in `known_tools`; assert `tool_hallucination`.
3. Trip `CircuitBreaker(fail_max=2)` and assert `allow` is false until cooldown.

**Check:** `pytest tests/test_reliability.py -v`

---

## EX-21 — Sandbox (`src.sandbox`)

1. Register a read tool and a write tool with `requires_approval=True`.
2. Deny without grant; deny without human; allow after approval.
3. `WorktreeExecutor`: edit a copy; assert the source file is unchanged.

**Check:** `pytest tests/test_sandbox.py -v`

---

## EX-22 — Trajectory evals (`src.agent_evals`)

1. Score a clean success vs a looping success; composite must drop on the loop.
2. `dashboard` on both; note `budget_violations`.
3. `regression_delta` with a worsened candidate; assert `ok is False`.

**Check:** `pytest tests/test_agent_evals.py -v`

---

## EX-23 — Prompt drift (`src.drift`)

1. Pin a `PromptConfig`; change only `tools`; assert `kind == "changed"`.
2. Drop the id from live; assert `missing`.
3. `eval_regression` with `parse_rate` 0.92 → 0.70; gate fails.

**Check:** `pytest tests/test_drift.py -v`

---

## EX-24 — Local-first (`src.local_agents`)

1. `TokenBudget(10)` refuses `allow(11)`.
2. Router: `classify` → local; `plan` + schema fail → strong.
3. `run_local_first` with a tiny budget; abort reason mentions budget.

**Check:** `pytest tests/test_local_agents.py -v`

---

## EX-25 — Durable graph (`src.durable`)

1. Child evidence raises parent score on `HypothesisTree`.
2. `DurableStore` round-trips `phase_done` from JSONL.
3. Coordinator pauses on `ask_human`; denial must not run the next phase; `MergeGate` blocks failed tests.

**Check:** `pytest tests/test_durable.py -v`

---

## EX-26 — Orchestrator pick (`src.orchestrators`)

1. Compare custom / LangGraph / CrewAI / MCP hosts; write when you’d pick each.
2. Two `CostEvent`s; writer USD > researcher.
3. `TraceRecorder` export includes `agent` on every span.

**Check:** `pytest tests/test_orchestrators.py -v`

---

## EX-27 — Harness (`src.harness`)

1. `verifier_required=True` with no `verify` callable → `stopped == "no_verifier"`.
2. Propose `tool="bash"` against a spec that only lists `write_note`; notes include `denied:`.
3. First artifact `"draft"`, second `"REFUND …"`; report is `verified` in two steps, not at `step_cap`.
4. Two context windows: persist `progress.json` after a failed draft; `load_progress` in session 2 and verify.

**Check:** `pytest tests/test_harness.py -v`

---

## Track stretch

| Track | Stretch exercise |
|-------|------------------|
| Stock | Replace `TinyRAG` bag-of-words with real embeddings; keep citation tests |
| Hybrid | Export a tiny MLP from PyTorch; serve predict via FastAPI sketch |
| Agentic plugin | Run `Agent` over `read_file` / `list_files` tools on this repo (read-only); add worktree + merge gate + trajectory eval |
