# Learning Paths

Choose intensity and depth. All paths share [Setup](setup.md). Core modules are listed in the [home map](../index.md).

---

## Weekend Warrior (2–3 days)

**Goal:** Working chatbot or document Q&A.

| Day | Focus | Modules |
|-----|-------|---------|
| 1 | Prompts + safety basics | 01, 02 (skim) |
| 2 | Tools or basic RAG | 07 |
| 3 | Minimal eval + ship | 04 (unit smoke), 13 (FastAPI sketch) |

**Skip for now:** Fine-tuning, multi-agent, compliance deep-dives.

You will not have Module 05 (context packing) yet. Cap pasted documents by hand — do not dump whole PDFs into the prompt. Treat that cap as a stand-in for the packer you will build later.

---

## Professional Developer (8–12 weeks)

**Goal:** Production-minded app with tests, caching, and observability. Times below assume ~1 focused hour most weekdays, matching the module time boxes — not a two-week cram.

| Phase | Modules |
|-------|---------|
| Foundations | 01 → 04 |
| Knowledge | 05, 07, 09 |
| Connectors & cost | 08, 10 |
| Ship | 11 (optional), 13 |
| Production agents (optional) | 20, 21, 22 |

**Prerequisites:** API experience; basic cloud or container familiarity.

---

## Enterprise Architect (12–16 weeks)

**Goal:** Scalable, multi-component systems with governance. Longer than the Professional path because it covers the rest of the core (compliance, domains, integration, local models), not because the modules are harder to skim.

| Phase | Modules |
|-------|---------|
| Full core | 01 → 14 |
| Integration | 15, 16 |
| Local/hybrid | 17 |
| Patterns (optional) | 18, 19 |
| Production agents | 20–26 |

**Emphasize:** Security, evals, multi-agent orchestration, audit trails, hybrid routing, agent failure modes, sandboxes, trajectory evals.

---

## AI Researcher (4–6 weeks of core, then a 90-day track)

**Goal:** Customization and advanced systems. The week count is **core modules only**. A specialization track is extra (~90 days) and is not folded into those 4–6 weeks.

| Phase | Modules |
|-------|---------|
| Theory + practice | 03, 05, 06 |
| Retrieval frontier | 09 |
| Agents | 11, 12 |
| SLMs | 17 |
| Production agents (optional) | 20, 22, 24, 26 |
| Track | Hybrid models or stock research stack |

---

## Specialization tracks (90 days)

| Track | Best after | Link |
|-------|------------|------|
| Stock recommender | 01–07, 09, 10, 13, 14, 17, 23 | [Track](../tracks/stock-recommender.md) |
| Hybrid Transformer+MLP | DL basics + 05–06; config pins (23 analog) | [Track](../tracks/hybrid-models.md) |
| Agentic VS Code plugin | 01–05, 07–08, 11–12, 17, 20–25 | [Track](../tracks/agentic-plugin.md) |

Tracks can run **in parallel** with later core modules if you already code comfortably.

---

## Daily cadence (any path)

1. **Review** (10–15 min) — previous notes / `PROGRESS.md` + [Progress dashboard](progress.md)
2. **Story → picture** (5 min) — read the opening incident and the **Intuition lock** first; say the sticky picture out loud
3. **Concept** (20–30 min) — explainers and mental model; answer every **Think about it** *before* revealing
4. **Hands-on** (20–30 min) — lab artifact (smallest thing that can fail in CI)
5. **Check** (5–10 min) — **predict → run → compare → explain**, then quizzes + checkpoint; mark module complete only if you can teach the kill-this-idea line ([Assessment](../reference/assessment.md))
6. **Log** (5 min) — what failed, what you’d redesign, next question

!!! tip "Anti-skim rule"
    If you only remember the **sticky picture** and the **kill this idea** line from a module, you still own the core intuition. Vocabulary without those is bland memorization.

---

## Capability checkpoints

See [Progression summary](../reference/progression.md) for “what you can build” after each module.  
Optional XP is a motivator, not a grade — production judgment still comes from evals and reviews.
