# Course Material Review Analysis

**Original review:** 2026-08-06  
**Status refresh:** 2026-08-07  
**Scope (historical):** Root README (stock roadmap), `first_principles.md`, `AgenticEngineering.md`, `effective_llm_usage_guide.md` (~11k lines), and supporting repo layout.

This page is the **diagnostic behind the restructure** plus a **closure record**. Critical defects from the original materials are resolved in the current MkDocs curriculum. Only intentionally deferred items remain open.

---

## Current status (2026-08)

| Area | Pre-restructure | Now |
|------|-----------------|-----|
| Pedagogical intent | Strong skeleton | **Strong** — CS-engineer tutorials + tracks |
| Structural integrity | Poor (scrambled monolith) | **Good** — ordered modules 01–17 |
| Currency | Weak (gpt-3.5-era, wrong MCP) | **Good** — provider-agnostic, correct MCP |
| Teachability | Mixed (outlines + dead code) | **Good** — explainers, labs, quizzes, mermaid |
| Repo coherence | Weak | **Good** — `docs/` spine + `src/` sandbox |
| GitHub Pages | None | **Ready** — MkDocs Material + Actions |

**Bottom line:** The restructure is **complete** for publishing and teaching core AI engineering. Optional follow-ups (notebooks, multi-version docs, server-side progress) are product enhancements, not defect cleanup.

---

## Original inventory (provenance)

Preserved under `archive/source/` at the **repository root** (not part of the MkDocs site tree). Summary of what we inherited:

| Source | Role | Problem |
|--------|------|---------|
| `effective_llm_usage_guide.md` (~11k lines) | Primary curriculum | Scrambled order; thin early levels; mega code dumps |
| Root README | Stock 90-day track | Isolated; misused “MCP” for compression |
| `first_principles.md` | Hybrid DL track | Isolated; no shared assessment |
| `AgenticEngineering.md` | IDE agent track | Citation noise; outdated SLM framing |
| `src/` (then) | Minimal examples | Not aligned with full tracks |

---

## Critical defects — closed

All **must-fix** items from the original review are closed. Evidence points at the live site tree under `docs/`.

### D1 — Structural corruption → **Closed**

| Was | Fixed by |
|-----|----------|
| Level 5.5 / Level 9 fragments after Quick Start | Modular `docs/core/01`–`17` in order |
| Broken TOC vs body | MkDocs `nav` + one topic per page |
| Incomplete code fences in scramble zone | Not ported; short runnable patterns instead |

### D2 — Incorrect MCP definition → **Closed**

| Was | Fixed by |
|-----|----------|
| “MCP” as multi-model load balancer | [Module 08](../core/08-model-context-protocol.md) = tools / resources / prompts |
| Stock track “MCP” = compression | Explicit **model compression** wording on [stock track](../tracks/stock-recommender.md) |
| Routing / load balancing | [Production](../core/13-production.md), [Integration](../core/16-integration-patterns.md), [Cost](../core/10-cost-optimization.md) |

### D3 — Stale model & API defaults → **Closed**

| Was | Fixed by |
|-----|----------|
| `gpt-3.5-turbo` defaults | Provider-agnostic examples (OpenAI-compatible, Claude, Gemini, Ollama) |
| No local / open weights | [Module 17](../core/17-small-models.md); tracks updated for Phi/Llama/Qwen-class SLMs |
| No modern eval / agent stack | Modules [04](../core/04-testing-evals.md), [11](../core/11-single-agents.md)–[12](../core/12-multi-agents.md); [Resources](../reference/resources.md) |

### D4 — Pedagogy vs. pseudo-frameworks → **Closed**

| Was | Fixed by |
|-----|----------|
| Multi-hundred-line unrunnable classes | Minimal patterns + `src/` modules with tests |
| Principles buried in scaffolding | Explainers, mental models, labs, quizzes on every core module |

### D5 — Three tracks, zero shared spine → **Closed**

| Was | Fixed by |
|-----|----------|
| Isolated 90-day plans | [Tracks overview](../tracks/index.md) with core dependencies |
| No shared setup / assessment | [Setup](../getting-started/setup.md), [paths](../getting-started/paths.md), [Assessment](../reference/assessment.md) |

### D6 — Not publishable as a docs site → **Closed**

| Was | Fixed by |
|-----|----------|
| Flat root markdown | `mkdocs.yml` + Material theme |
| No CI Pages | `.github/workflows/deploy-docs.yml` |
| 387 KB single page | Split curriculum under `docs/` |

---

## Content decisions (kept themes)

Themes retained from the original guide; rewritten for depth and currency:

Prompting · security · advanced patterns · evals · context engineering · fine-tuning · tools/RAG · MCP · advanced RAG · cost · single/multi-agent · production · compliance · domain patterns · integration · small/local models · three specialization tracks.

**Not ported (by design):** unrunnable mega-frameworks, duplicate HRAG/domain scramble blocks, misnamed MCP load-balancer design, root-README-as-only-home.

---

## Relevance map (before → after)

| Topic | Before | After |
|-------|--------|--------|
| Default chat API | gpt-3.5-centric | Provider-agnostic; cite current frontier + mini/SLM |
| Local inference | Barely present | Ollama, vLLM, llama.cpp patterns |
| Agents | Custom class dumps | Plan–act–observe + framework study (LangGraph, etc.) |
| MCP | Invented load balancer | Real Model Context Protocol |
| Evals | Ad-hoc harness only | Unit tests + golden sets + judge caveats + OSS tools |
| Context | Memory class dumps | Budgets, packing hierarchy, tiers |
| Small models | DistilBERT-era framing | Modern SLMs, quant, routing |
| Teachability | TOC + dumps | Tutorials + optional static XP |

---

## Information architecture (current)

```text
docs/
├── index.md
├── assets/                 # Gamification CSS/JS (static, localStorage)
├── review/analysis.md      # This page
├── getting-started/        # Setup, paths, progress/XP
├── core/                   # Modules 01–17 (tutorials)
├── tracks/                 # 90-day specializations
└── reference/              # Progression, exercises, assessment, resources
archive/source/             # Pre-restructure originals
src/ + tests/               # Learner sandbox (Poetry)
```

| Track | Days | Core dependencies |
|-------|------|-------------------|
| Stock recommender | 90 | 01–07, 09–10, 13, 17 |
| Hybrid Transformer+MLP | 90 | 05–06 + DL fundamentals |
| Agentic VS Code plugin | 90 | 01–05, 07–08, 11–12, 17 |

---

## Success criteria

### Restructure (original checklist)

- [x] Ordered curriculum without scrambled sections  
- [x] Correct MCP definition and updated model landscape  
- [x] Three tracks linked to shared core modules  
- [x] MkDocs Material site builds statically  
- [x] GitHub Actions workflow for Pages  
- [x] Review analysis published as a first-class page  
- [x] Original sources preserved in `archive/source/`  
- [x] Runnable `src/` teaching package + unit tests  
- [x] Assessment rubrics and exercise index  
- [x] Core deps installable without heavy native builds (track extras optional)  

### Tutorial depth & engagement (follow-on)

- [x] Core modules are tutorials (mental model, explainers, labs), not TOC stubs  
- [x] Inquisitive questions (“Think about it”) and interactive quizzes  
- [x] Mermaid diagrams where architecture matters  
- [x] **Intuition pass** — incident stories, sticky analogies, “kill this idea” locks on all 17 modules  
- [x] Curated OSS curricula on [Resources](../reference/resources.md)  
- [x] Static gamification (XP, badges, HUD) — [Progress](../getting-started/progress.md)  

---

## Residual work (only open items)

| Item | Status | Notes |
|------|--------|--------|
| Legal/domain content is educational only | **Policy** | Warnings stay in modules 14–15 and finance track |
| Multi-version docs (`mike`) | **Optional** | Add when yearly curriculum forks exist |
| Full notebooks / GPU fine-tune CI | **Future** | Local GPU optional; not required for the site |
| Server-side leaderboards / accounts | **Out of scope** | Static GitHub Pages; progress is `localStorage` |
| Track pages at same tutorial depth as core | **Done** | All three tracks expanded with stories, mermaid, explainers, code, traps, exits |

No open **must-fix** defects remain from the 2026-08-06 diagnostic.

---

*Course home: [index](../index.md) · Setup: [getting-started/setup](../getting-started/setup.md)*
