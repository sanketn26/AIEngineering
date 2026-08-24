# Specialization Tracks

Ninety-day, **project-shaped tutorials** that reuse the [core modules](../index.md). These are not phase checklists alone — each track page has mental models, explainers, code sketches, traps, and exit gates.

Complete [Setup](../getting-started/setup.md) first. Close core gaps listed on the track before you start building.

Tracks are domain-specific; the [Capstone](../core/capstone.md) is domain-agnostic and can run instead of or alongside a track — it proves the six generic parts (service, evals, knowledge, agent, ops, security) hold together, without committing to a vertical.

| Track | Outcome | Core dependencies | Vibe |
|-------|---------|-------------------|------|
| [Stock recommender](stock-recommender.md) | Research assistant / recommender prototype: data → baseline ML → SLM → RAG → compression → ship | 01–07, 09–10, 13–14, 17, 23; 22/24 if you add a tool loop | Markets + retrieval + MLOps (**not** financial advice) |
| [Hybrid models](hybrid-models.md) | Custom Transformer + MLP fusion in PyTorch, ablations, deploy | 05–06 + DL fundamentals; 23 analog (config pins), 04/22 regression helper | From-scratch architecture engineering |
| [Agentic editor plugin](agentic-plugin.md) | VS Code extension + agent backend + local models | 01–05, 07–08, 11–12, 17, 20–25 | IDE product + agent safety |

---

## How to run a track (intuition-first)

```mermaid
flowchart LR
  Setup[Setup + core gaps] --> Story[Read track story + architecture]
  Story --> Phase[One phase at a time]
  Phase --> Lab[Ship phase exit artifact]
  Lab --> Log[PROGRESS.md + honest metrics]
  Log --> Phase
  Log --> Demo[Day-90 demo + rubric]
```

1. **Skim the whole track once** — know the day-90 shape before day 1.  
2. **Read the track’s Intuition lock** out loud; if you can’t restate it, you’re not ready to code.  
3. **Map each phase → core modules**; finish those modules’ labs first.  
4. **Ship phase exits** (repo artifacts), not only notes.  
5. **Score yourself** with [Assessment rubrics](../reference/assessment.md) at mid-track and day 90.  
6. Optional: keep [Progress XP](../getting-started/progress.md) for core modules; tracks are graded by demos.

---

## Shared engineering bar

| Bar | Why |
|-----|-----|
| Git from day 1 | Tracks die in “works on my laptop” folders |
| Tests for deterministic code | Splits, tools, parsers, shapes |
| Evals for model behavior | Golden Q&A, must-refuse, Hit@k — not vibes |
| Agent hardening (if you ship a loop) | Failure detectors, sandbox/HITL, trajectory eval, prompt pins (modules 20–23) |
| Hardware honesty (local models) | Size to RAM/KV ([17 §7](../core/17-small-models.md#7-working-effectively-on-limited-hardware)); do not swap |
| README with limits + ethics | Especially finance and write-capable agents |
| No secrets in git | Keys in env / SecretStorage only |
| Time / data leakage honesty | Shuffle is cheating on markets and sequences |

---

## Choosing a track

| If you want… | Pick |
|--------------|------|
| End-to-end product with data + RAG + deploy | Stock recommender |
| Deep understanding of architectures and training | Hybrid models |
| Shipping developer tooling with agent safety | Agentic editor plugin |

You may run a track **in parallel** with later core modules if you already ship Python/TS services confidently.

### Which new core patterns belong where

Modules 20–26 (the agent-hardening tail of [Gate 4](../core/index.md#gate-4-actions-and-agents) and [Gate 5](../core/index.md#gate-5-operate-it)) and [17 §7 limited hardware](../core/17-small-models.md#7-working-effectively-on-limited-hardware) are not a tax on every track. Cargo-culting LangGraph, worktrees, or trajectory evals onto a tabular hybrid is how you get costume jewelry.

| Pattern | Stock recommender | Hybrid models | Agentic plugin |
|---------|-------------------|---------------|----------------|
| 17 §7 RAM / one resident model | **Yes** — PEFT + lite serve | **Partial** — shrink `d_model` / `max_len` / batch, not GGUF | **Yes** — Ollama default |
| 23 Prompt/config digest | **Yes** — research prompt pack | **Yes analog** — train YAML + metrics JSON | **Yes** — system + tool list |
| 22 Trajectory / regression | Optional (only if you add a tool loop) | **eval_regression** on MAE, not agent traces | **Yes** — stubbed agent CI |
| 24 Token budget / local-first | If `/research` calls an SLM in a loop | No | **Yes** |
| 20–21 Breakers, manifests, worktrees | Quote-tool timeouts; not a coding agent | No | **Yes** — this *is* the incident |
| 25 Durable HITL / merge gate | No | No | **Yes** — approve then apply |
| 26 Orchestrator comparison | Optional cost-per-path | Params/latency log, not CrewAI | Written custom vs LangGraph |

---

---

## Day-90 definition of done (all tracks)

- [ ] Demo runs from a clean clone + documented setup  
- [ ] At least one automated test suite and one model/behavior eval  
- [ ] Architecture diagram in README matches the code  
- [ ] Known failure modes written down (not hidden)  
- [ ] Ethics / safety / non-advice notes where relevant  
- [ ] Patterns from the table above that apply to *this* track are in the demo, not only in notes  

**Next:** open a track page and start with its story + system diagram.
