---
description: Pick a learning path, from a 2-3 day Weekend Warrior sprint to a 12-16 week Enterprise Architect track, all sharing the same setup and modules.
---

# Learning Paths

Choose intensity and depth. All paths share [Setup](setup.md). Core modules are listed in the [home map](../index.md).

The opening cases are self-contained. A shorter path does not require story context from skipped modules; use each module's **Depends on** line for prerequisites and its closing limitation to choose a useful next step.

---

## Weekend Warrior (2–3 days)

**Goal:** Working chatbot or document Q&A.

| Day | Focus | Modules |
|-----|-------|---------|
| 1 | Prompts + safety basics | 01, 02 (skim) |
| 2 | Tools or basic RAG | 07 (weekend slice) |
| 3 | Minimal eval + one trace | 04 (unit smoke), 13 (weekend slice) |

**Skip for now:** Fine-tuning, multi-agent, compliance deep-dives.

**Path overrides.** These slices are smaller than the module time boxes and dependency lines.

- **Module 07.** The full module depends on 01–05. This weekend slice depends on Module 01 and a skim of 02. Cap pasted documents by hand. Finish 03–05 before Advanced RAG.
- **Module 13.** The full module is 2–3 weeks alongside a project. This weekend slice is `GET /healthz` plus one trace that carries a `request_id`.

---

## Professional Developer (8–12 weeks)

**Goal:** Production-minded app with tests, caching, and observability. Times below assume ~1 focused hour most weekdays, matching the module time boxes. Module 13’s 2–3 weeks sits inside Ship and inside this 8–12 week total, worked alongside the app. Modules 17 (5–7 days) and 28 (2–4 days) are additional Ship time. The Production agents row sits outside the 8–12 weeks. Module 11 is optional inside Ship. The weekend slice of Module 13 is `GET /healthz` plus one trace that carries a `request_id`.

| Phase | Modules |
|-------|---------|
| Foundations | 01 → 04 |
| Knowledge | 05, 07, 09 |
| Connectors & cost | 08, 10 |
| Ship | 11 (optional), 13, 17, 28 |
| Production agents (optional) | 20, 21, 27, 22 |

**Prerequisites:** API experience; basic cloud or container familiarity.

---

## Enterprise Architect (12–16 weeks)

**Goal:** Scalable, multi-component systems with governance. Longer than the Professional path because it covers the rest of the core (compliance, domains, integration, local models), not because the modules are harder to skim.

| Phase | Modules |
|-------|---------|
| Full core | 01 → 14 |
| Integration | 15, 16 |
| Local/hybrid | 17, 28 |
| Patterns (optional) | 18, 19 |
| Production agents | 20–27 |

**Emphasize:** Security, evals, multi-agent orchestration, audit trails, hybrid routing, agent failure modes, sandboxes, trajectory evals.

---

## AI Researcher (4–6 weeks of core, then a 90-day track)

**Goal:** Customization and advanced systems. The week count is **core modules only**. A specialization track is extra (~90 days) and is not folded into those 4–6 weeks.

| Phase | Modules |
|-------|---------|
| Theory + practice | 03, 05, 06 |
| Retrieval frontier | 09 |
| Agents | 11, 12 |
| SLMs | 17, 28 |
| Production agents (optional) | 20, 22, 24, 26, 27 |
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

Module 28 follows 10, 13, and 17. The worked exercises need no GPU or API key. The hands-on GPU runs stay optional. The hybrid track's encoder does not use autoregressive KV caching, continuous decode batching, or speculative decoding.

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
