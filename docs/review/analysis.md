# Course Material Review Analysis

**Review date:** 2026-08-06  
**Scope:** Root README (stock roadmap), `first_principles.md`, `AgenticEngineering.md`, `effective_llm_usage_guide.md` (~11k lines), and supporting repo layout.

This document is the diagnostic behind the restructure. It records what was wrong, what was kept, and how the material was reorganized for GitHub Pages.

---

## Executive summary

| Area | Rating | Verdict |
|------|--------|---------|
| Pedagogical intent | Strong | Clear multi-level LLM path + three project tracks |
| Structural integrity | Poor | Scrambled section order, duplicates, broken TOC |
| Currency (2025–2026) | Weak | gpt-3.5-era APIs; wrong MCP definition; missing modern stack |
| Teachability | Mixed | Good outlines; many mega code dumps, incomplete snippets |
| Repo coherence | Weak | Course docs vs. minimal Poetry/stock code mixed |
| GitHub Pages readiness | None | Flat markdown at repo root; no site config |

**Bottom line:** High-value curriculum skeleton buried under structural corruption and outdated tooling. Restructure into a modular MkDocs site; rewrite for current models, protocols, and agent frameworks; treat the three 90-day plans as optional specialization tracks.

---

## Inventory of source materials

### 1. `effective_llm_usage_guide.md` (primary curriculum)

- **Size:** ~10,878 lines / ~387 KB — too large for a single page or PR review.
- **Intended arc:** Levels 1 → 10 (prompting → production/domain), plus small models, troubleshooting.
- **Actual state:** Section order corrupted; incomplete “Learning Path Recommendations”; Level 9 content spliced into early path personas; Level 5.5 appears *before* Prerequisites.

**Documented H2 order in source (broken):**

```
Quick Start
→ Level 5.5 Cost Optimization   ← too early
→ Prerequisites
→ Learning Path Overview
→ Level 1 … Level 10 (mostly sequential)
→ Small models / Troubleshooting
```

**Line-budget by major section (approx.):**

| Section | Lines | Assessment |
|---------|------:|------------|
| Level 2.5 Testing & QA | ~1,150 | Over-weighted; framework dump |
| Level 3.5 Fine-tuning | ~1,100 | Useful concepts; heavy PEFT scaffolding |
| Level 5 Advanced RAG | ~1,100 | Duplicated subsections (HRAG twice) |
| Level 6 Single agents | ~1,130 | Solid architecture ideas; not production code |
| Level 4.5 “MCP” | ~1,025 | **Factually misaligned** with industry MCP |
| Level 1 Prompting | ~140 | Too thin relative to later code |
| Level 9 Domains | ~120 | Thin stubs; partial duplicate earlier |

### 2. README.md (stock recommender track)

- 90-day Phi-based stock recommender with RAG, compression, MLOps.
- Coherent phase table and link list.
- **Issues:** Windows/WSL-centric; “MCP” used for model compression/pruning (collides with Model Context Protocol); Phi-3-centric; some dead/fragile links; not integrated with the LLM mastery guide.

### 3. `first_principles.md` (hybrid model track)

- 90-day Transformer + MLP hybrid from scratch.
- Clean phase structure; good PyTorch links.
- **Issues:** Isolated from main guide; some paper links may rot; no shared setup or assessment model.

### 4. `AgenticEngineering.md` (agentic IDE plugin track)

- 90-day VS Code extension + small models + LangGraph/CrewAI.
- Most aligned with 2025 agent wave.
- **Issues:** Citation noise (`[1][8]` style); DistilBERT/TinyBERT era for “small models” vs. modern SLMs; incomplete reference definitions; no shared standards with main guide.

### 5. Code package (`src/`, Poetry)

- Minimal: `example.py`, `pandas_intro.py`, basic tests.
- Not aligned with any full track deliverable.
- **Implication:** Treat `src/` as learner sandbox, not the course body of knowledge.

---

## Critical defects (must-fix)

### D1 — Structural corruption

- Level 5.5 and fragments of Level 9 inserted after “Quick Start”.
- “Weekend Warrior” path interrupted by domain-app copy.
- Incomplete / broken code fences in the scrambled Level 9 fragment.
- TOC claims a linear path that the file body does not deliver.

### D2 — Incorrect industry terminology (MCP)

The guide’s “Model Context Protocol” is described as a **model load-balancing / multi-provider abstraction**.

In industry practice (Anthropic open standard, 2024–), **MCP** is a protocol for connecting AI applications to **tools, resources, and prompts** (filesystems, DBs, APIs, IDEs). Load balancing and multi-model routing remain valid topics — they are not MCP.

**Action:** Replace the MCP module with the real protocol; move load-balancing content under production / integration patterns.

### D3 — Stale model & API defaults

- Defaults to `gpt-3.5-turbo` and occasional `gpt-4`.
- No first-class treatment of Claude, Gemini, open weights (Llama, Qwen, Phi-4, Gemma), or local runtimes (Ollama, vLLM, llama.cpp).
- No LangGraph (agents), structured outputs, or modern eval stacks.

### D4 — Pedagogy vs. pseudo-frameworks

Many sections ship multi-hundred-line class hierarchies that:

- Cannot run as-is (undefined `llm`, incomplete methods, broken control flow).
- Read as generated scaffolding rather than teachable minimal examples.
- Hide the few sentences of principle that actually matter.

**Action:** Prefer short, runnable patterns + checklists; link out for full frameworks.

### D5 — Three tracks, zero shared curriculum spine

Learners cannot see:

- Shared prerequisites
- Which core modules unlock which track
- How assessment works across tracks

### D6 — Not publishable as a docs site

- No `mkdocs.yml` / Jekyll / Docusaurus config
- No navigation, search, or version metadata
- Root README is *one track*, not the course home
- Single 387 KB page will crush GitHub Pages UX and SEO

---

## Content quality by theme

| Theme | Keep? | Notes |
|-------|-------|-------|
| Prompt anatomy & principles | Yes | Expand slightly; refresh examples |
| Security / prompt injection / PII | Yes | High value; slim implementation |
| CoT, few-shot, ToT, self-consistency | Yes | Keep patterns; cut dead code |
| Testing & A/B prompts | Yes (rewrite) | Point to eval frameworks (RAGAS, promptfoo, DeepEval) |
| Context & memory | Yes | Align with “context engineering” vocabulary |
| Fine-tuning / LoRA | Yes | When-to-finetune decision tree; PEFT sketch |
| Tools + basic RAG | Yes | Add embeddings options, chunking heuristics |
| Advanced RAG | Yes | Deduplicate HRAG; add hybrid search, rerank, agentic RAG |
| Cost optimization | Yes | Move after RAG/tools; update pricing language |
| Single / multi-agent | Yes | LangGraph / tool-calling loops; human-in-the-loop |
| Production | Yes | Observability, rate limits, fallbacks |
| Compliance | Yes (lightweight) | Principles + checklists, not legal advice |
| Domain stubs | Yes (brief) | Patterns only; avoid fake medical/finance advice |
| Stock / hybrid / agent tracks | Yes | Update models & tools; share setup |

---

## Relevance updates applied in the restructure

| Topic | Before | After (2026-oriented) |
|-------|--------|------------------------|
| Default chat API | OpenAI gpt-3.5-turbo | Provider-agnostic client; cite gpt-4o / Claude / Gemini / open models |
| Local inference | Barely present | Ollama, vLLM, llama.cpp patterns |
| Agents | Custom class dumps | Tool-calling loop + LangGraph-style graphs |
| MCP | Invented load balancer | Anthropic Model Context Protocol (tools/resources) |
| Evals | Custom test harness only | Unit tests + LLM-as-judge + RAGAS-style metrics |
| Context | Memory classes | Context budgets, packing, compaction, tool results |
| Small models | DistilBERT/TinyBERT framing | Phi-4-class SLMs, quantization, task routing |
| Compression “MCP” (stock track) | Misnamed | Explicit “model compression” vocabulary |
| IDE agents | Generic | Align with modern agentic coding practices |

---

## Restructure design

### Information architecture

```
docs/
├── index.md                 # Course home
├── review/analysis.md       # This review
├── getting-started/         # Setup + learning paths
├── core/                    # Sequential modules (1–17)
├── tracks/                  # 90-day specializations
└── reference/               # Troubleshooting, resources, progression
```

### Module map (core)

| # | Module | Source levels |
|---|--------|---------------|
| 01 | Prompt engineering | L1 |
| 02 | Security & privacy | L1.5 |
| 03 | Advanced prompting | L2 |
| 04 | Testing & evals | L2.5 |
| 05 | Context engineering | L3 |
| 06 | Fine-tuning | L3.5 |
| 07 | Tools & basic RAG | L4 |
| 08 | Model Context Protocol | L4.5 (rewritten) |
| 09 | Advanced RAG | L5 |
| 10 | Cost optimization | L5.5 (repositioned) |
| 11 | Single-agent workflows | L6 |
| 12 | Multi-agent systems | L7 |
| 13 | Production systems | L8 |
| 14 | Compliance & governance | L8.5 |
| 15 | Domain applications | L9 |
| 16 | Integration patterns | L10 |
| 17 | Small / local models | Small models chapter |

### Specialization tracks

| Track | Days | Depends on core |
|-------|------|-----------------|
| Stock recommender (Phi + RAG + MLOps) | 90 | 01–07, 09–10, 13, 17 |
| Hybrid Transformer+MLP from scratch | 90 | 05–06, first-principles DL |
| Agentic VS Code plugin | 90 | 01–05, 07–08, 11–12, 17 |

### Publishing constraints (GitHub Pages)

- **MkDocs Material**: search, nav tabs, code copy, dark mode, mobile.
- **One topic per page** under ~400–600 lines ideal.
- **Absolute-friendly internal links** via MkDocs nav.
- **CI build** on `main` → GitHub Pages artifact deploy.
- **`site_url`** configurable via repo name.
- Source archives retained under `archive/source/` for provenance.

---

## What was intentionally *not* ported

1. Multi-thousand-line unrunnable “frameworks” (replaced by minimal patterns).
2. Duplicate Level 9 / HRAG blocks.
3. Broken mid-fence domain fragments from the scramble zone.
4. Misnamed MCP load-balancer design (concepts redistributed).
5. Root README-as-only-track (replaced by course home + track pages).

---

## Success criteria for the restructure

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

---

## Residual risks / future work

| Item | Status |
|------|--------|
| Live code in `src/` + tests for modules 01–02, 04–05, 07, 10–11, 14 | **Done** — see [Exercises](../reference/exercises.md) |
| Assessment rubrics for modules + day-90 tracks | **Done** — [Assessment](../reference/assessment.md) |
| Prefer primary vendor docs in resource lists | **Done** — [Resources](../reference/resources.md) |
| Legal/domain content remains educational only | **Accepted** — warnings retained in modules 14–15 |
| Tutorial depth (explainers, questions, diagrams) vs TOC feel | **Done** — core modules ~300–500 lines each with mermaid, labs, OSS refs |
| Static gamification (XP, badges, quizzes, HUD) | **Done** — `docs/assets/{js,css}/gamify.*`, [Progress](../getting-started/progress.md); GitHub Pages–safe |
| Multi-version docs with `mike` | **Optional** — add when yearly forks exist |
| Full notebooks per module / real GPU fine-tune CI | **Future** — local GPU optional; not required for course site |
| Server-side leaderboards / accounts | **Out of scope** — static hosting only; localStorage progress |

---

*Next: see [Getting Started](../getting-started/setup.md) and the [core module index](../index.md).*
