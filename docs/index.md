# AI Engineering Course

**Build LLM applications from first prompt to production agents.**

This site is a restructured, relevance-updated curriculum for AI engineering: prompt design, security, RAG, agents, local/small models, and MLOps — plus three optional 90-day specialization tracks.

!!! tip "How to use this site"
    1. Complete [Setup](getting-started/setup.md)  
    2. Pick a [learning path](getting-started/paths.md)  
    3. Work through **Core modules** in order (or jump where the path allows)  
    4. Optionally commit to a [specialization track](tracks/index.md)

---

## Who this is for

| Persona | Goal | Suggested path |
|---------|------|----------------|
| Curious builder | Ship a chatbot / RAG app in a weekend | Weekend Warrior |
| Software engineer | Production-ready services with evals | Professional |
| Systems / platform | Multi-agent, compliance, hybrid deploy | Enterprise |
| Researcher / ML | Fine-tuning, advanced RAG, SLMs | Researcher |
| Project-focused | Stock system, hybrid DL, or IDE agent | Specialization tracks |

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

Progression tables: [Capability summary](reference/progression.md) · [Exercises](reference/exercises.md) · [Assessment rubrics](reference/assessment.md).

---

## Specialization tracks (90 days)

| Track | Focus |
|-------|--------|
| [Stock recommender](tracks/stock-recommender.md) | Data → classical ML → Phi/SLM → RAG → compression → deploy |
| [Hybrid models from scratch](tracks/hybrid-models.md) | Transformer + MLP fusion in PyTorch |
| [Agentic editor plugin](tracks/agentic-plugin.md) | VS Code extension + local agents |

---

## Repository layout

```text
docs/                 # This site (source of truth for the course)
archive/source/       # Pre-restructure originals (provenance)
src/                  # Optional Python sandbox (Poetry)
tests/                # Unit tests for sandbox code
mkdocs.yml            # Site config
.github/workflows/    # GitHub Pages deploy
```

---

## About the restructure

A full diagnostic of the prior materials — scrambled sections, outdated APIs, incorrect MCP definition, and publishing gaps — lives in the [Review analysis](review/analysis.md).

**Last curriculum refresh:** 2026-08.
