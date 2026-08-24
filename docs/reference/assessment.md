# Assessment Rubrics

Use these rubrics to judge **module exercises** and **day-90 track demos**. Score honestly; demos beat slideware.

---

## Scoring scale (all rubrics)

| Score | Meaning |
|------:|---------|
| 0 | Missing |
| 1 | Attempted but incorrect or non-runnable |
| 2 | Partial — core idea present, major gaps |
| 3 | Solid — works with minor issues |
| 4 | Strong — complete, tested, documented |

**Pass threshold:** average ≥ 3.0 on required criteria, with no required criterion at 0.

---

## Module exercise rubric (generic)

| Criterion | Weight | 4 looks like |
|-----------|-------:|--------------|
| **Runnable artifact** | 25% | Script/tests run without secret hardcoding |
| **Concept fidelity** | 25% | Matches module learning objectives |
| **Safety / honesty** | 20% | Fail closed; no fake metrics; disclaimers where needed |
| **Eval or tests** | 20% | At least one automated check |
| **Write-up** | 10% | `PROGRESS.md` notes: what worked / failed / next |

### Module-specific must-haves

| Module | Must demonstrate |
|--------|------------------|
| 01 | Structured prompt + temperature experiment log |
| 02 | Sanitization or redaction on hostile inputs |
| 03 | Schema-validated output (or Pydantic parse) |
| 04 | Golden set with pass/fail threshold |
| 05 | Token budget or rolling summary in code |
| 06 | Written fine-tune vs RAG decision |
| 07 | Tool allowlist **or** RAG with citations |
| 08 | MCP security policy (allowed servers) |
| 09 | Hybrid/rerank **or** multi-hop eval numbers |
| 10 | Cost or token log + one optimization |
| 11 | Agent with `max_steps` + tool log |
| 12 | Multi-role handoff with budget |
| 13 | Healthcheck API + structured logs sketch |
| 14 | Audit events without raw secrets |
| 15 | Must-refuse cases tested |
| 16 | Async/job **or** data-class routing |
| 17 | Local model run + task fit notes + hardware fit (RAM/KV, one resident model) |
| 18 | At least three leaf patterns (subroutine, gate, sampler, consensus, or retriever) with a named failure each one fixes |
| 19 | At least three orchestration patterns (map-reduce, router, planner, ReAct, memory, or duet) on one workflow |
| 20 | Detector hits at least two failure modes on a stub trajectory; circuit breaker unit test |
| 21 | Least-privilege registry + approval on a write + isolated worktree or process |
| 22 | Process + outcome scores and a regression_delta that fails a worsened suite |
| 23 | Prompt bundle hash; silent edit detected; eval gate on a metric drop |
| 24 | Token budget abort + local vs strong routing stub |
| 25 | Durable pause/resume + merge gate refusing failed tests |
| 26 | Written orchestrator pick with ranks + per-agent cost events |

---

## Gate assessment ladder

Five levels, one per depth of understanding — use these to check whether a student actually owns a gate's material or can only recite its vocabulary. Each level up requires the one below it; a student who can Design but not Predict is pattern-matching, not reasoning from the mechanism.

| Level | Tests | 4 looks like |
|---|---|---|
| **Explain** | Why the mechanism exists at all | A causal explanation in the student's own words, not a definition copied from the module |
| **Predict** | Forward simulation — given a change, what breaks | Names the *specific* failure and the mechanism that produces it, not "it gets worse" |
| **Diagnose** | Reading a trace/log/symptom and naming the failure | Names the failure mode **and** the evidence in the trace that points to it — a correct guess without evidence doesn't count |
| **Design** | Proposing a concrete fix | A specific, testable mechanism (code, config, or policy) — restating the principle is not a design |
| **Defend** | Justifying a decision against a real objection | Names the tradeoff that was accepted and why, not a restatement of the original choice |

### Gate 1 — Dependable Model Service

1. **Explain:** Why is an LLM API call a different kind of dependency than a deterministic library call?
2. **Predict:** A provider's median latency doubles overnight. If no request has a deadline, what happens to worker utilization under sustained load?
3. **Diagnose:** A response passes JSON parsing but fails your Pydantic schema on a field type. What's the actual failure — and is "just retry" the right fix?
4. **Design:** Add a fail-closed path for schema-invalid output that doesn't silently drop the user's request.
5. **Defend:** Why does temperature 0 not remove the need for an eval suite (Gate 2)?

### Gate 2 — Measurable Quality

1. **Explain:** Why do deterministic code paths get unit tests while model behavior gets an eval suite instead?
2. **Predict:** A one-line prompt "polish" ships without touching the golden set. What's the earliest point this course's CI setup could have caught a regression — and would it have?
3. **Diagnose:** Golden-set accuracy holds steady but user complaints rise. What's not being measured?
4. **Design:** Add a CI gate that blocks a merge when eval score regresses past a threshold, without blocking on noise from run-to-run variance.
5. **Defend:** Why is LLM-as-judge only trustworthy after it's checked against human-labeled agreement — what fails if you skip that check?

### Gate 3 — External Knowledge

1. **Explain:** Why should raw-query retrieval be the baseline you measure against, not the thing you replace on day one?
2. **Predict:** You add HyDE-based query rewriting without an eval gate. A query with an exact order-ID identifier stops retrieving correctly. Why?
3. **Diagnose:** Retrieval recall is fine but generated answers still cite the wrong policy. Is this a retrieval problem or a packing/context problem?
4. **Design:** Add a retrieval-confidence threshold that gates escalation to agentic RAG — and the evaluation step that has to happen before you trust it.
5. **Defend:** Why is "the model needs more context" often the wrong diagnosis for a grounding failure?

### Gate 4 — Actions and Agents

1. **Explain:** Why must tool authorization live outside the model instead of in a system-prompt instruction?
2. **Predict:** An agent's `max_steps` cap is set, but duplicate-tool-call detection is not. Where does the budget actually get spent when the agent loops?
3. **Diagnose:** A trace shows the same tool called four times in a row with near-identical arguments. What failure mode is this, and what's the earliest step it could have been caught?
4. **Design:** Add loop-control and tool-safety mechanisms to a single-agent loop that currently only checks a step counter.
5. **Defend:** Why is a prompt-injection classifier a *mitigation*, not a *security boundary* — what does it not protect against that an allowlist does?

### Gate 5 — Operate It

1. **Explain:** Why does "it works in the notebook" not imply "it's a production service"?
2. **Predict:** A provider starts rate-limiting during a traffic spike, and no request has a timeout. Trace the failure through workers, health checks, and the autoscaler.
3. **Diagnose:** Given an Agent Flight Recorder trace with rising `retry_count` and flat `success_rate`, what's developing, and is it visible on a dashboard that only tracks success rate?
4. **Design:** Add a dashboard alert that would have caught the Gate 5 incident story before the bill did.
5. **Defend:** Why must a rollback path exist for prompt/model/config versions specifically, separate from a code rollback?

---

## Track demo rubrics (day 90)

### Stock recommender

| Criterion | Weight | Evidence |
|-----------|-------:|----------|
| Time-safe baseline | 15% | Split description + metrics |
| SLM or LLM path | 15% | Repro script / adapter card |
| RAG citations | 15% | Live demo of cited answers |
| Compression tradeoff | 10% | Size/latency/quality table; lite fits RAM |
| Deploy/CI | 15% | Container or Actions green; prompt digest on ready |
| Evals | 15% | Golden questions Hit@k or accuracy; `eval_regression` floor |
| Ethics / non-advice | 15% | UX + README disclaimers |

### Hybrid Transformer+MLP

| Criterion | Weight | Evidence |
|-----------|-------:|----------|
| Two baselines | 15% | MLP-only + Transformer-only |
| Hybrid design | 20% | Diagram + config |
| Fair comparison | 20% | Same data/splits/metrics |
| Ablations | 15% | ≥2 architecture levers; frozen config **digest** |
| Repro | 15% | One-command train/eval |
| Deploy/export | 15% | ONNX/API/CLI |

### Agentic editor plugin

| Criterion | Weight | Evidence |
|-----------|-------:|----------|
| Extension command | 15% | Installable VSIX or debug launch |
| Agent tools | 20% | Read-only tools logged |
| Write safety | 20% | Approval before edits; worktree + merge gate; deny does not apply |
| Local model path | 15% | Ollama sized to RAM; token budget; escalate visible |
| Workflow | 15% | ≥2 steps with persisted HITL state |
| Tests/docs | 15% | Trajectory eval + security notes + prompt digest |

---

## Capstone oral (optional, 15 min)

1. Architecture sketch (2 min)  
2. Live happy path (5 min)  
3. Failure demo (injection, empty retrieval, or tool error) (3 min)  
4. What you would build next (2 min)  
5. Q&A (3 min)  

**Pass:** Happy path works; one failure handled; student can explain tradeoffs.

---

## Course package mapping

Automated checks for teaching patterns live in the repo:

| Package module | Course module | Tests |
|----------------|---------------|-------|
| `src.security` | 02 | `tests/test_security.py` |
| `src.prompts` | 01/03 | `tests/test_prompts.py` |
| `src.context_memory` | 05 | `tests/test_context_memory.py` |
| `src.rag` | 07/09 | `tests/test_rag.py` |
| `src.evals` | 04 | `tests/test_evals.py` |
| `src.cost` | 10 | `tests/test_cost.py` |
| `src.agents` | 11 | `tests/test_agents.py` |
| `src.audit` | 14 | `tests/test_audit.py` |
| `src.reliability` | 20 | `tests/test_reliability.py` |
| `src.sandbox` | 21 | `tests/test_sandbox.py` |
| `src.agent_evals` | 22 | `tests/test_agent_evals.py` |
| `src.mcp_prod` | 08 | `tests/test_mcp_prod.py` |
| `src.drift` | 23 | `tests/test_drift.py` |
| `src.local_agents` | 24 | `tests/test_local_agents.py` |
| `src.durable` | 25 | `tests/test_durable.py` |
| `src.orchestrators` | 26 | `tests/test_orchestrators.py` |

```bash
poetry install
poetry run pytest tests/ -v
```
