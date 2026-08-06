# Track: Agentic VS Code Plugin (90 days)

**Goal:** Ship a VS Code extension that orchestrates **agentic coding workflows** with Python services and **small/local models**, using modern tool protocols where useful.

**Updated for 2026:** Prefer tool-calling agents, optional **MCP** servers for repo tools, LangGraph-style workflows, SLMs via Ollama; study open agent extensions for patterns (do not violate licenses).

**Core modules:** [01](../core/01-prompt-engineering.md)–[05](../core/05-context-engineering.md), [07](../core/07-tools-and-rag.md)–[08](../core/08-model-context-protocol.md), [11](../core/11-single-agents.md)–[12](../core/12-multi-agents.md), [17](../core/17-small-models.md).

---

## Phase overview

| Phase | Days | Focus | Deliverable |
|-------|------|-------|-------------|
| Foundations | 1–14 | Extension scaffold + Python agent stub | Hello-world command |
| Agent basics | 15–28 | Tools: search, summarize, edit plan | CLI agent over a folder |
| Integrate | 29–42 | Extension ↔ agent ↔ LLM | In-editor invoke |
| Workflows | 43–56 | Multi-step graphs | explain → patch → test flow |
| Small models | 57–70 | Local/remote switch | Ollama + cloud fallback |
| UX & tests | 71–80 | Prompt UI, reliability | Test suite + settings |
| Advanced | 81–90 | Planning, docs, beta | Public beta README |

---

## Days 1–14 — Foundations

- Node.js + VS Code Extension API (`yo code` or equivalent template)  
- Python 3.11 agent package `core_agent/`  
- WSL if on Windows ([WSL](https://learn.microsoft.com/en-us/windows/wsl/install))  
- Command: “AIEngineering: Hello Agent” shows a message  

**Exit:** Extension + Python process communication path (stdio/HTTP).

---

## Days 15–28 — Agent essentials

- Agent class: goal, tools, max steps (Module 11)  
- Tools: `list_files`, `read_file`, `search_text` (read-only first)  
- Structured logs for every tool call  

**Exit:** CLI: `python -m core_agent "summarize src/"` works on sample repo.

---

## Days 29–42 — Plugin + LLM

- Sidebar or CodeLens actions  
- Provider abstraction: OpenAI-compatible + Anthropic + Ollama  
- Settings: endpoint, model, API key via SecretStorage  

**Exit:** Select code → “Explain” returns streamed or panel answer.

---

## Days 43–56 — Workflows

- Graph: explain → propose diff → (optional) apply → run tests  
- Human approval before writes  
- Optional: LangGraph or lightweight custom graph  

**Exit:** End-to-end demo on a toy project with approval gates.

---

## Days 57–70 — Small models

- Default local model via Ollama  
- Auto-escalate to cloud on low confidence / long context  
- Cache repeated explains  

**Exit:** Documented RAM/latency notes; settings for model tiers.

---

## Days 71–80 — UX & quality

- Prompt template picker  
- Integration tests for the Python agent  
- Extension tests for command registration  
- Telemetry **opt-in only**  

---

## Days 81–90 — Harden & publish

- Modular commands: refactor, docify, testgen  
- MCP: optional filesystem/git servers for power users (Module 08 security!)  
- README, GIFs, issue templates  
- Beta tag + feedback channel  

---

## Milestones

| Day | Checkpoint |
|-----|------------|
| 14 | Hello extension + agent stub |
| 28 | Read-only CLI agent |
| 42 | In-editor LLM explain |
| 56 | Multi-step approved workflow |
| 70 | Local/cloud routing |
| 80 | Tests + prompt UI |
| 90 | Public beta |

---

## Study references (patterns, not endorsements)

- VS Code extension docs: https://code.visualstudio.com/api  
- MCP: https://modelcontextprotocol.io/  
- LangGraph conceptual guides / docs  
- CrewAI multi-agent concepts (optional)  
- Open-source agentic editors/extensions — read licenses before copying  

---

## Success checklist

- [ ] Write actions require explicit user approval  
- [ ] Tools are allowlisted and logged  
- [ ] Local model path works offline for basic tasks  
- [ ] Tests cover agent tool failures  
- [ ] Security notes for MCP/servers in README  
