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
| 17 | Local model run + task fit notes |
| 18 | At least three leaf patterns (subroutine, gate, sampler, consensus, or retriever) with a named failure each one fixes |
| 19 | At least three orchestration patterns (map-reduce, router, planner, ReAct, memory, or duet) on one workflow |

---

## Track demo rubrics (day 90)

### Stock recommender

| Criterion | Weight | Evidence |
|-----------|-------:|----------|
| Time-safe baseline | 15% | Split description + metrics |
| SLM or LLM path | 15% | Repro script / adapter card |
| RAG citations | 15% | Live demo of cited answers |
| Compression tradeoff | 10% | Size/latency/quality table |
| Deploy/CI | 15% | Container or Actions green |
| Evals | 15% | Golden questions Hit@k or accuracy |
| Ethics / non-advice | 15% | UX + README disclaimers |

### Hybrid Transformer+MLP

| Criterion | Weight | Evidence |
|-----------|-------:|----------|
| Two baselines | 15% | MLP-only + Transformer-only |
| Hybrid design | 20% | Diagram + config |
| Fair comparison | 20% | Same data/splits/metrics |
| Ablations | 15% | ≥2 architecture levers |
| Repro | 15% | One-command train/eval |
| Deploy/export | 15% | ONNX/API/CLI |

### Agentic editor plugin

| Criterion | Weight | Evidence |
|-----------|-------:|----------|
| Extension command | 15% | Installable VSIX or debug launch |
| Agent tools | 20% | Read-only tools logged |
| Write safety | 20% | Approval before edits |
| Local model path | 15% | Ollama or equivalent |
| Workflow | 15% | ≥2 steps with state |
| Tests/docs | 15% | Agent tests + security notes |

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

```bash
poetry install
poetry run pytest tests/ -v
```
