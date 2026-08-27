# Capability Progression Summary

What you should be able to build after each module. Times are rough calendar estimates at ~1 focused hour/day unless noted.

| Module | Core skills | What you can build | Time |
|--------|-------------|--------------------|------|
| 01 Prompting | Clear prompts, formats | Chatbots, generators | 2–3 days |
| 02 Security | Injection awareness, PII | Safer input layers | 1–2 days |
| 03 Advanced prompting | CoT, few-shot, schemas | Reliable extractors | 3–5 days |
| 04 Testing & evals | Golden sets, metrics | Regression gates | 2–3 days |
| 05 Context engineering | Budgets, memory tiers | Long-session apps | 5–7 days |
| 06 Fine-tuning | PEFT decisioning | Domain adapters | 7–10 days |
| 07 Tools & RAG | Tool loop, citations | Knowledge Q&A | 5–7 days |
| 08 MCP | Protocol, security, host policy | Portable connectors with pins, authz, failover | 4–6 days |
| 09 Advanced RAG | Hybrid, rerank, agentic | Research assistants | 7–10 days |
| 10 Cost | Route, cache, ledger | Efficient prod paths | 2–3 days |
| 11 Single agents | Plan–act–observe | Autonomous task runners | 7–10 days |
| 12 Multi-agent | Roles, handoffs | Collaborative workflows | 10–14 days |
| 13 Production | Serve, observe, CI | Hardened APIs | 14–21 days |
| 14 Compliance | Audit, data maps | Governance-ready systems | 3–5 days |
| 15 Domains | Vertical patterns | Prototypes with refuses | 7–14 days |
| 16 Integration | Events, hybrid | Platform-style LLM I/O | 10–14 days |
| 17 Small models | Local SLMs, quant + re-eval, limited-hardware fit | Offline / cheap tiers on a laptop | 5–7 days |
| 18 Agent design patterns | Subroutine, guardrail, resampler, consensus, retriever | Composable, testable agent internals | 5–8 days |
| 19 Orchestration patterns | Map-reduce, router, planner, ReAct, memory, duet | Large-input, multi-step, persistent workflows | 6–9 days |
| 20 Agent reliability | Failure taxonomy, detectors, circuit breakers | Bounded loops that abort on named modes | 4–6 days |
| 21 Secure tool use | Least privilege, HITL, worktrees, process isolation | Sandboxed tools the model cannot escape | 5–7 days |
| 22 Agent evals | Trajectory, process vs outcome, regression, dashboard | CI gate on multi-step agent suites | 5–7 days |
| 23 Prompt drift | Versioned bundles, hashes, eval regression | Silent prompt/config change detection | 3–5 days |
| 24 Local-first agents | Token budgets, hybrid local/cloud routing | Laptop-useful agents with hard meters | 4–6 days |
| 25 Durable orchestration | Coordinators, hypothesis trees, merge gates, HITL | Restartable graphs with isolated writes | 7–10 days |
| 26 Orchestrators in prod | Custom vs LangGraph vs CrewAI vs MCP; $ attribution | Defensible stack pick + per-step receipts | 5–7 days |
| 27 Harness engineering | Prompt vs context vs harness; verify/persist/stop outside the model | Control layer that makes the same weights finish a long job | 4–6 days |

## Cumulative milestones

| After modules | Capability band |
|---------------|-----------------|
| 01–04 | Reliable, testable single-turn apps |
| 05–08 | Context-aware apps with tools/RAG/MCP |
| 09–11 | Advanced retrieval + autonomous loops |
| 12–14 | Coordinated, observable, governable systems |
| 15–19 | Vertical + platform + local/edge + composable-pattern + orchestration options |
| 20–27 | Production agents: named failures, sandboxes, harness, trajectory evals, drift, durable graphs |
| All 27 | [Capstone](../core/capstone.md)-ready: an evaluated, authorized, tool-using production AI service |

Use this table in `PROGRESS.md` to mark completion honestly (demo > notes).
