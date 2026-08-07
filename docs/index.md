# AI Engineering Course

**Build LLM applications from first prompt to production agents.**

This site is a restructured curriculum for **CS and software engineers**: prompt design, security, RAG, agents, local/small models, and production MLOps — plus three optional 90-day specialization tracks.

Modules are written as **tutorials**, not a table of contents: mental models, explainers, inquisitive questions, labs, and interactive quizzes. Optional **client-side gamification** (XP, badges, progress HUD) works on static GitHub Pages via `localStorage` — no backend required.

!!! tip "How to use this site"
    1. Complete [Setup](getting-started/setup.md)  
    2. Pick a [learning path](getting-started/paths.md)  
    3. Open the [Progress dashboard](getting-started/progress.md) (optional XP)  
    4. Work through **Core modules** — read explainers, answer “Think about it”, ship the lab  
    5. Optionally commit to a [specialization track](tracks/index.md)

---

## Who this is for

| Persona | Goal | Suggested path |
|---------|------|----------------|
| Curious builder | Ship a chatbot / RAG app in a weekend | Weekend Warrior |
| Software engineer | Production-ready services with evals | Professional |
| Systems / platform | Multi-agent, compliance, hybrid deploy | Enterprise |
| Researcher / ML | Fine-tuning, advanced RAG, SLMs | Researcher |
| Project-focused | Stock system, hybrid DL, or IDE agent | Specialization tracks |

**Primary audience:** engineers who already ship APIs, services, and tests — and need to treat models as unreliable components with budgets, security boundaries, and eval gates.

---

## Core curriculum map

| Module | Topic | You will be able to |
|--------|-------|---------------------|
| [01](core/01-prompt-engineering.md) | Prompt engineering | Write reliable, structured prompts |
| [02](core/02-security-privacy.md) | Security & privacy | Resist injection; handle PII |
| [03](core/03-advanced-prompting.md) | Advanced prompting | CoT, few-shot, structured outputs |
| [04](core/04-testing-evals.md) | Testing & evals | Regression tests and quality metrics |
| [05](core/05-context-engineering.md) | Context engineering | Budget tokens; memory & packing |
| [06](core/06-fine-tuning.md) | Fine-tuning | Decide when to PEFT/LoRA |
| [07](core/07-tools-and-rag.md) | Tools & basic RAG | Tool calling + retrieve-then-generate |
| [08](core/08-model-context-protocol.md) | Model Context Protocol | Connect tools/resources via MCP |
| [09](core/09-advanced-rag.md) | Advanced RAG | Hybrid search, rerank, agentic RAG |
| [10](core/10-cost-optimization.md) | Cost optimization | Route, cache, measure spend |
| [11](core/11-single-agents.md) | Single agents | Plan–act–observe loops |
| [12](core/12-multi-agents.md) | Multi-agent | Orchestrate specialized roles |
| [13](core/13-production.md) | Production | Serve, observe, harden |
| [14](core/14-compliance.md) | Compliance | Audit trails & governance basics |
| [15](core/15-domain-apps.md) | Domain apps | Vertical patterns (not legal advice) |
| [16](core/16-integration-patterns.md) | Integration | Events, hybrid cloud, services |
| [17](core/17-small-models.md) | Small / local models | SLMs, Ollama, quantization |

Progression tables: [Capability summary](reference/progression.md) · [Exercises](reference/exercises.md) · [Assessment rubrics](reference/assessment.md) · [Open-source resources](reference/resources.md).

---

## How these modules teach (intuition first)

Each core module is built so you **picture the system**, not only collect vocabulary:

| Element | What it does in your head |
|---------|---------------------------|
| **Incident story** | A Friday-afternoon failure you can feel before the theory |
| **Mental model** | Mermaid of control/data flow — where trust and tokens move |
| **Intuition lock** | One sticky analogy + the wrong idea to kill |
| **Explainers** | Why a design exists, not only how to type it |
| **Think about it** | Tradeoffs that force re-reading, not skim checkboxes |
| **Lab + quizzes** | Prove the picture works under a small shippable artifact |

If a page still feels like a bland outline, that is a curriculum bug — open an issue.

---

## Gamification (GitHub Pages–safe)

| Feature | How it works |
|---------|----------------|
| Progress HUD | Fixed panel on every page |
| XP & levels | Quizzes, reflections, module complete |
| Badges | Foundations, Retrieval Pro, Ship It, Full Core, … |
| Dashboard | [Progress & XP](getting-started/progress.md) |

All state is **browser `localStorage`**. No accounts, no analytics backend, works on static hosting.

---

## Specialization tracks (90 days)

Full tutorials (architecture, code, traps, phase exits) — not bare phase tables. Start at [Tracks overview](tracks/index.md).

| Track | Focus |
|-------|--------|
| [Stock recommender](tracks/stock-recommender.md) | Data → classical ML → SLM → RAG → compression → deploy |
| [Hybrid models from scratch](tracks/hybrid-models.md) | Transformer + MLP fusion in PyTorch |
| [Agentic editor plugin](tracks/agentic-plugin.md) | VS Code extension + agent backend + local models |

---

## Repository layout

```text
docs/                 # This site (source of truth for the course)
  assets/             # Gamification CSS/JS
archive/source/       # Pre-restructure originals (provenance)
src/                  # Optional Python sandbox (Poetry)
tests/                # Unit tests for sandbox code
mkdocs.yml            # Site config
.github/workflows/    # GitHub Pages deploy
```

---

## About the restructure

A full diagnostic of the prior materials — scrambled sections, outdated APIs, incorrect MCP definition, and publishing gaps — lives in the [Review analysis](review/analysis.md).

**Last curriculum refresh:** 2026-08 (tutorial depth + static gamification).
