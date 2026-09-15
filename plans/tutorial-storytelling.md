# Plan: Make the tutorials hold attention through storytelling

**Status:** audit complete, implementation not started
**Branch:** `plan/tutorial-storytelling`
**Scope:** rewrite *how* the curriculum is told, not *what* it teaches
**Non-goal:** turning modules into fiction, lengthening every page 2×, or changing labs / quizzes / Python APIs

---

## 1. Verdict

Reader feedback is **validated**. The content is appreciated because it is specific, production-minded, and technically dense. Attention drops because the pages are **reference manuals with a one-paragraph anecdote taped to the front**, not stories that a reader is pulled through.

This is not “there are no stories.” There already are. They are the wrong *kind*, in the wrong *place*, and they **stop** after the intro.

| Claim | Evidence from this scan |
|---|---|
| People like the content | 27 ordered modules, diagrams, labs, quizzes, kill-this-idea boxes, a five-gate running app |
| Content does not hold attention | Homogeneous 50–90 word italic asides, then 300–500 lines of tables and numbered sections |
| “They lack storytelling” | Accurate as a *reader experience* claim; inaccurate as a “zero narrative” claim |

Treat the work as **upgrading incident notes into a serial case**, not adding more callout boxes.

---

## 2. What was scanned

| Surface | Files | Role |
|---|---|---|
| Core modules | `docs/core/01`–`27` | Primary reader path |
| Core framing | `docs/core/index.md`, `capstone.md`, `capstone-gates.md` | Serial spine and exit criteria |
| Home / getting started | `docs/index.md`, `docs/getting-started/*` | First 10 minutes |
| Tracks | `docs/tracks/*.md` | 90-day deep dives (~750–940 lines each) |
| Reference | `docs/reference/*` | Rubrics, exercises — keep as reference |
| Presentation | `docs/assets/css/gamify.css` (`.aieng-story`) | How story *looks* |

Method: every core module opening, every `aieng-story` block, counts of story words / “running app” links / think boxes / quizzes, plus mid-page samples of “Core tutorial” sections.

---

## 3. What already exists (do not throw away)

The course already invested in narrative *ingredients*:

1. **Incident openers** — 27/27 core modules have exactly one `<div class="aieng-story">`. Stock track has one. Hybrid and agentic tracks have “The incident” prose without the class.
2. **A serial idea** — `docs/core/index.md` § “The running app”: a support-ticket triage service whose *previous gate’s failure* forces the next gate.
3. **Pedagogy blocks** — intuition locks, kill-this-idea, think-about-it, labs, quizzes.
4. **Gate plot** — dependable model → measured quality → grounded knowledge → safe action → operate it. That *is* a story structure.

Those ingredients are why the content is appreciated. They are not enough to hold a reader for a 400–550 line module.

---

## 4. Why attention still drops

### 4.1 The hook is buried

Every core module opens:

1. Title + time + depends-on
2. **Learning objectives** (syllabus list)
3. Heading **“Why this matters (CS engineer view)”**
4. Then the story, in italics

A reader who is already tired meets a bullet list of competencies before any human is on the page. That is a textbook, not a chapter.

### 4.2 The “stories” are postmortems, not scenes

Median story length is **~50–90 words** (modules 18 and 19 are the outliers at ~135–154). Pattern:

> Friday 4:47pm. Anonymous team. Bot does a bad thing. Bolded root cause. End.

Missing from almost every opener:

- A **named person** the reader can track
- A **want** (ship the demo, close the ticket, survive the questionnaire)
- A **clock** that continues *inside* the lesson
- **Dialogue or a concrete artifact** (Slack, pager, PR, dashboard)
- A **resolution** later on the same page

They read as engineering slogans in narrative clothing. After three modules the Friday-timestamp formula itself becomes wallpaper.

### 4.3 The story dies at the intro

After the italic paragraph, pages switch to:

- CS-engineer lecture (“an LLM call is a distributed dependency…”)
- Mermaid mental model
- `## Core tutorial` with numbered subsections, tables, and code

There is almost never a return to the incident. Teaching does not *use* the scene; it *cites* it and moves on.

### 4.4 The serial is a table, not a plot

“The running app” is the right idea. It is implemented as a **five-row failure table** on the core index, then mentioned in **only six modules**: 01, 04, 05, 08, 13, 27. The other 21 modules invent a *new* anonymous incident (research crew, wellness assistant, CrewAI vs LangGraph, MCP server, weekend repo agent…). The reader cannot accumulate attachment.

### 4.5 Visual design demotes the story

```css
.aieng-story {
  font-style: italic;
  opacity: 0.95;
  margin: 0.35rem 0 0.75rem;
  padding-left: 0.75rem;
  border-left: 3px solid rgba(245, 158, 11, 0.7);
}
```

Story is styled as a *sidebar quote*. Quizzes, labs, and intuition locks get full colored panels. The thing that should hook the eye looks like a caption.

### 4.6 Density without beats

Typical core module: **240–570 lines**. Tracks: **750–940 lines** of day-by-day syllabus after one incident. Attention needs **beats** (scene → concept → decision → proof → next scene). Right now the beat is: scene (80 words) → concept dump (the rest).

### 4.7 Uneven secondary narrative

| Device | Coverage |
|---|---|
| `aieng-story` | 27/27 core, 1/3 tracks |
| Think boxes | 0 in modules 18 and 19; 1–4 elsewhere |
| Running-app callback | 6/27 modules |
| Capstone as story | 0 — it is a build spec |
| Getting started as story | 0 — it is a checklist |

Modules 18–19 have the *longest* openers and then the *least* mid-page human texture.

---

## 5. What “storytelling” should mean here

Not a novel. Not “once upon a time.” The target is the density of a good incident write-up plus a recurring company:

**Case-driven tutorial.** Each module is one chapter in the life of a support-triage product. The reader is inside the on-call rotation. Concepts are the tools the team uses to get out of the hole they just fell into.

Closest existing models: SRE postmortems, *The Phoenix Project* (without the padding), Stripe-style API guides that open on a failing request.

### Hard constraints

- Keep every learning objective, diagram, lab, quiz, and code sample unless a rewrite makes it clearer.
- Do not invent capabilities the code does not teach.
- Do not add more than ~400 narrative words per module (cold open + one mid-return + close). The page stays a tutorial.
- Do not start every chapter at “Friday 4:47pm.”
- Fictional company and people are **teaching devices**, labeled once as composite, not case studies of real outages.

---

## 6. The serial: Helix at Northstar Support

Promote the existing “running app” from a table into a **named product and a small crew**. One canon, reused everywhere.

### Company and product

| Item | Canon |
|---|---|
| Company | **Northstar** — mid-size commerce / billing company (fictional) |
| Product | **Helix** — the support-ticket triage service the course already describes |
| Stakes | Wrong category/priority, invented refunds, silent quality drops, runaway tool spend, hung workers, failed enterprise renewal |
| Disclaimer | One line on the core index: composite incidents for teaching; not a real vendor postmortem |

This matches the capstone starter (`capstone-starter/` is already a triage service). Story and code finally share a protagonist.

### Recurring people (keep the cast small)

| Name | Role | Why they exist |
|---|---|---|
| **Maya Chen** | Backend / on-call for Helix | The reader’s stand-in. Ships, pages, writes the fix. |
| **Priya Shah** | Product | Deadline, “make it friendlier,” demo pressure. Turns vague requests into contracts. |
| **Jules Okonkwo** | Security | Injection, MCP, sandbox, confused deputy. |
| **Noah Berg** | Finance / ops | Bills, `cost_per_success`, questionnaires, SLOs. |
| **Helix** | The system | Not a person. Treat it like a flaky dependency with a name. |

Tracks that are *not* Helix (stock, hybrid, plugin) get their **own** short cast, not a forced Helix cameo. See §9.

### Serial spine (gates as seasons)

This is the plot the core index already has, named so modules can continue it instead of resetting:

| Gate | Season title | Helix state at the start | Failure that ends the season |
|---|---|---|---|
| 1 | Soft contracts | Helix is a prompt in a PR | Invented refunds; hostile input; unparseable JSON |
| 2 | Unmeasured quality | Output is schema-valid | A “tiny wording tweak” drops accuracy 12 points with no CI red |
| 3 | Confident ignorance | Evals are green | Helix invents `POLICY-404` / drowns policy under tool JSON |
| 4 | Unsafe action | Retrieval works | Refund JSON treated as authority; 400 tool calls overnight |
| 5 | Laptop ≠ production | The loop works locally | Hung workers, no `request_id`, prompt edited in a dashboard, renewal questionnaire |

Each module is **one episode** of that season, not a new TV show.

---

## 7. Module template (the actual change)

Replace the current skeleton with this. Same technical body; different *arc*.

```text
# Module NN — Title
Time / depends / next          (keep)

## Cold open                    (NEW, before objectives)
Named people. Concrete artifact. Clock.
150–250 words. Not italic-only.
Ends on a question the module will answer.

## What this chapter is for     (RENAME learning objectives)
Same bullets. Framed as “by the end Maya can …”
not “the learner will be able to …”

## Mental model                 (keep)
Diagram first. One sticky picture.

## Walkthrough                  (RENAME Core tutorial)
Numbered sections stay.
Each H3 opens with 1–3 sentences that advance the incident
(“Maya pins the template; Priya’s ‘friendlier’ now has a rubric”).
Then the current tables/code/explainers.

## Return to the scene          (NEW, once, mid-page)
80–120 words. Apply the last two techniques to Helix.
Not a recap of the opener.

## Lab / quizzes                (keep)

## Close and next failure       (NEW)
Resolve *this* episode.
Name the residual failure that is the next module’s cold open.
Link it.
```

### Opening-order change (do this even if nothing else ships)

**Before:** objectives → “Why this matters (CS engineer)” → 60-word italic story → lecture.

**After:** cold open → one-line “this is the Helix failure for this gate” → objectives → mental model → walkthrough.

Drop the heading **“Why this matters (CS engineer view)”**. The scene *is* why it matters. The CS framing moves into the mental-model paragraph, where it already lives.

### Voice rules

- Specific artifacts: ticket `#88421`, prompt digest, `cost_per_success`, p95, golden-set 71% vs 92%.
- People speak in Slack-length lines, not speeches.
- Root cause is earned in the walkthrough, not bolded in sentence three of the opener.
- Humor is dry and rare. No sitcom, no “dear reader.”
- Kill-this-idea boxes stay. They are the moral of the episode.

### CSS

Promote `.aieng-story` from caption to **chapter cold-open**:

- Roman (not italic) body text
- Full-width panel, same visual weight as `.aieng-lab`
- Optional eyebrow: `Northstar · Helix · Gate N`
- Mid-page returns use a sibling class (e.g. `.aieng-scene`) so they are not confused with the opener

Do not make story blocks look like ads. Keep them readable in light and slate.

---

## 8. Gold-standard snippet (Module 01)

Current opener (~52 words, italic, after objectives):

> Friday 4:47pm: a support bot ships after a “quick prompt polish.” By Monday, finance is chasing three refunds the bot invented…

Target cold open (illustrative, not final copy):

> Thursday, 4:47 p.m. Priya slacks Maya a screenshot: Helix told a customer they were “approved for a courtesy refund.” There is no such policy. Maya greps the repo. The system prompt is an f-string in `triage.py` that says *be helpful*. No role. No “do not invent money rules.” No output schema. Finance will see the tickets Monday.
>
> Maya has until standup to put a contract on the model — something two engineers could grade the same way — without rewriting Helix in three services.

Then the existing anatomy table, temperature section, and `src.prompts` lab **are the fix**, narrated as Maya’s PR, not as “§1 Anatomy of a good prompt” in a vacuum.

Close of 01 hands the residual to 02: the prompt is versioned, but a PDF in the knowledge base is about to become a work order (Jules’s incident).

---

## 9. File-by-file change list

Priority: **P0** = attention-critical path; **P1** = complete the serial; **P2** = polish / tracks / chrome.

### 9.1 Canon and chrome — P0

| File | Change |
|---|---|
| `docs/core/index.md` | Rewrite “The running app” as Helix at Northstar: named crew, season table, “read this as a serial.” Keep gate exit criteria. |
| `docs/index.md` | One narrative beat in the hero or method section: you follow Helix from a soft prompt to a production service. Do not turn the landing page into a short story. |
| `docs/assets/css/gamify.css` | Restyle `.aieng-story`; add `.aieng-scene` for mid-page returns. |
| `plans/helix-story-bible.md` *(new, with implementation)* | One-pager: names, product facts, what Helix can/cannot do at each gate, forbidden contradictions. Writers use this so Module 12 does not invent a different company. |

### 9.2 Gate 1 (prove the template) — P0

Do these first. They set voice for everything else. If 01–03 do not hold attention in a read-aloud, stop and revise the template before touching Gate 4.

| Module | Current story (words) | Change |
|---|---|---|
| **01 Prompt engineering** | 52. Anonymous Friday polish. Running-app note exists. | Gold-standard Helix cold open (Priya screenshot / no contract). Mid-return: “friendlier” VP request becomes versioned policy (the existing think box, lifted into the scene). Close → 02. |
| **02 Security** | 61. Tuesday standup, RAG PDF as work order. No running-app link. | Same incident, Jules + Maya. Helix forwarded a runbook because a KB PDF issued orders. Walkthrough = trust boundaries on *that* path. Close → 03 (parse failures from stacked techniques). |
| **03 Advanced prompting** | 52. 2:14am invoice `json.loads`. | Helix on-call: CoT + eight few-shots shipped “to be safe,” parse still broken, bill spiked. Decision map is how Maya chooses *one* lever. Close → 04 (still no evals). |

### 9.3 Gate 2–3 — P0 / P1

| Module | Current | Change |
|---|---|---|
| **04 Testing & evals** | 51. Green sprint review, 92%→71%. Has running-app link. | Noah notices wrong amounts; CI never went red. Golden set is the chapter’s object. Close → 05 (correct but ignorant). |
| **05 Context engineering** | 57. Day 19, policy drowned. Has running-app link. | Helix invents account IDs because the packer drowned policy. Maya owns the window. Close → 06/07 fork: weights vs retrieve. |
| **06 Fine-tuning** | 49. Catalog baked into weights. | Product wants Helix to “know the catalog.” Train loss great; retired SKUs persist. Decision tree is the episode. Stay on Helix *or* a Northstar catalog sidecar — do not switch industries. |
| **07 Tools & RAG** | 75. Refund JSON `eval`’d + POLICY-404. Strongest Gate-3 opener. | Split into two beats of the *same* week (action vs knowledge), both Helix. Intern `eval` becomes a named PR. Close → 08/09. |
| **09 Advanced RAG** | 61. `ERR_INV_88421` missed by dense search. | Helix ops bot; ticket `#88421` already used in 14 — **reuse it**. Crime scene is retrieval. Do not “fix quality” with a bigger generator. |

### 9.4 Gate 4 — P1

These modules currently each invent a new anonymous agent. Re-home them on Helix’s tool loop unless the topic *cannot* live there (MCP host / editor sandbox can be Jules’s laptop still at Northstar).

| Module | Current | Change |
|---|---|---|
| **08 MCP** | 93. Trendy MCP server + PM “load balancer” confusion. Has running-app link. | Jules enables auto-approve so Helix’s IDE “sees the monorepo.” Peripheral vs host policy. Keep the naming confusion as Priya’s deck. |
| **10 Cost** | 64. Two incidents mashed (mini routing + Alice/Bob cache). | One episode: Noah’s 40% token screenshot vs reopen rate. Cache cross-tenant is a *second beat* mid-page, not a second opener. Metric: `cost_per_success`. |
| **11 Single agents** | 60. Overnight 400 tool calls. | Helix “research” path. Maya adds `max_steps` / signature abort. Personality vs state machine. |
| **12 Multi-agent** | 79. Hackathon persona theater. | Priya asks for CEO/engineer/designer agents. Cost 10×, README worse. Topology vs theater, on Helix’s doc-draft path. |
| **16 Integration** | 85. `POST /chat` 120s timeout, refresh storms. | Helix chat behind a gateway. Jobs/queues. Maya’s `request_id` dies at hop 1. |
| **18 Patterns** | 135. Missing-person hard drive (off-canon). 0 think boxes. | **Do not keep the hard-drive plot** if we are serializing Helix. Recast as Helix scanning a large ticket attachment / order-history dump. Add 1–2 think boxes. |
| **19 Orchestration** | 154. Contract audit vs 40 clauses. 0 think boxes. | Northstar vendor-contract audit *or* Helix policy-clause audit — pick one and stick. Long opener is good; still needs mid-page scene and think boxes. |
| **20 Reliability** | 94. Friday 17:10 research crew, $186, green 200. | Helix research crew. Named failure families. Close → 21. |
| **21 Secure tools** | 71. `bash` god-tool, novel in sibling folder. | Jules’s incident: Helix’s editor agent. Policy in English vs sandbox. Can stay “laptop” as long as the user is Jules at Northstar. |
| **27 Harness** | 91. Helix already named (lookup_order / write_note). Has running-app link. | Best existing serial beat. Expand to Maya discovering the prompt was fine. Close Gate 4. |

### 9.5 Gate 5 — P1

| Module | Current | Change |
|---|---|---|
| **13 Production** | 84. Hung p95, no timeout, dashboard prompt edit. Has running-app link. | Season 5 premiere. Keep; name Maya/Noah; `request_id` hunt. |
| **14 Compliance** | 92. Renewal questionnaire, ticket `#88421`. | Noah + legal. Same ticket id as 09. Controls/provenance. |
| **15 Domain apps** | 81. Wellness assistant (off-canon). | Either a **Northstar-adjacent** vertical Helix should not pretend to be (medical/legal adjacent feature request from Priya) or a clearly marked side-quest. Do not introduce a new company without saying so. |
| **17 Small models** | 94. Swap everything to 3B Q4. | Noah’s bill-cut + Maya’s schema collapse. Router as the fix. |
| **22 Agent evals** | 73. Extract 94%, path 6× cost. | Helix agentic bot. Trajectory vs extract. |
| **23 Drift** | 69. Playground “warmer” prompt, pin lied. | Priya’s warmth tweak; digest vs `v3` name. |
| **24 Local-first** | 81. Weekend personal repo agent (off-canon). | Recast as Maya’s laptop loop on Helix, or mark as Jules’s weekend side project *at Northstar*. Keep token budget lesson. |
| **25 Durable** | 82. Codebase investigator, laptop sleep, Slack merge. | Helix billing double-charge investigation. Coordinator + worktree + HITL state. |
| **26 Orchestrators** | 70. Team A/B/C framework tourism. | Three Northstar squads, one Helix workflow, three engines. Question remains: “who spent money on step 7?” |

### 9.6 Capstone and getting started — P1 / P2

| File | Change |
|---|---|
| `docs/core/capstone.md` | Open on Helix with four planted holes (the starter already has them). The spec table stays. Reader should feel they are finishing Maya’s service, not starting a new assignment. |
| `docs/core/capstone-gates.md` | One-line episode framing per gate checkpoint. Do not novelize checklists. |
| `docs/getting-started/setup.md` | Keep commands. Add a 80-word “you are joining Northstar; this is your laptop” beat so setup is onboarding, not a package list. |
| `docs/getting-started/paths.md` | Each path is a different *pace through Helix’s seasons*, not a different product. |
| `docs/getting-started/index.md` / `progress.md` | Light touch. Progress UX can stay mechanical. |

### 9.7 Tracks — P2 (separate casts)

Tracks are 90-day syllabi. Story will not save a 900-line day list by itself. Changes:

| Track | Current | Change |
|---|---|---|
| Stock | One 75-word `aieng-story`, then pipeline. | Named researcher + PM; **do not** reuse Helix. Cold open + “week N return” at each major milestone (time-safe split, citations, evals). Keep non-advice warning. |
| Hybrid | “The incident” at 2:14 a.m., no `aieng-story` class. | Same incident, wrap in story class, named ML engineer. Ablation days get a return-to-scene (“the Transformer-only path still loses on SKU X”). |
| Agentic plugin | Strong incident (11:40 p.m. auto-apply). No story class. | Wrap in story class. Named extension author. Recurring rule: model proposes, runtime disposes. Milestone returns at approval gate, MCP trust, local SLM. |
| `docs/tracks/index.md` | Catalog. | One paragraph: tracks are *other companies*; core is Helix. |

### 9.8 Leave as reference (no storytelling pass)

`docs/reference/assessment.md`, `exercises.md`, `progression.md`, `resources.md`, `troubleshooting.md`. These should stay lookup documents. Optional: exercises can *mention* Helix ticket ids so they feel like the same world.

Do not story-wrap `src/` or tests.

---

## 10. Per-module quality bar (acceptance)

A module is done only if all of the following are true:

1. **Cold open before objectives**, 150–250 words, named Northstar people (or the track’s own cast).
2. **Same incident** is visible in at least one mid-page beat and the close.
3. Close names the **next module’s failure** with a link.
4. Technical claims, diagrams, labs, quizzes, and code paths are unchanged in meaning.
5. No second fictional company unless labeled a side-quest (15, 21 laptop, tracks).
6. Read-aloud test: first 400 words hold a listener who is not looking at the headings.
7. Skim test: a returning reader can still jump to `## Mental model` and labs without reading the fiction.

If (6) and (7) conflict, cut narrative, not the mental model.

---

## 11. What we will not do

- Add illustrations of characters or comic panels (out of scope; diagrams stay mermaid).
- Generate a real-person likeness or “based on a true outage.”
- Rewrite Python teaching modules to print story text.
- Publish this plan in the MkDocs nav (internal).
- Homogenize every opener into the same timestamp joke.
- Double page length. If a module grows more than ~15%, cut lecture repetition, not labs.

---

## 12. Implementation sequence (PRs)

Each PR independently reviewable. Do not stack all 27 modules in one diff.

| PR | Title | Files | Depends on |
|---|---|---|---|
| **PR1** | Helix canon, core index serial, story CSS | `docs/core/index.md`, `docs/index.md` (light), `docs/assets/css/gamify.css`, `plans/helix-story-bible.md` | — |
| **PR2** | Gate 1 template + modules 01–03 | `docs/core/01`–`03` | PR1 |
| **PR3** | Gates 2–3: modules 04–07, 09 | `docs/core/04`–`07`, `09` | PR2 (voice freeze) |
| **PR4** | Gate 4: 08, 10–12, 16, 18–21, 27 | those core files | PR2 |
| **PR5** | Gate 5: 13–15, 17, 22–26 | those core files | PR2 |
| **PR6** | Capstone + getting started | `docs/core/capstone*.md`, `docs/getting-started/*` | PR1 |
| **PR7** | Tracks | `docs/tracks/*` | PR1 |

**Stop after PR2** for a read-aloud with the people who gave the original feedback. If 01–03 still “don’t hold attention,” the template is wrong; do not roll it across 24 more files.

Suggested review protocol for PR2: one reviewer reads 01 on a phone, out loud, without scrolling to the TOC. Mark the first sentence they disengage. That line is the bug.

---

## 13. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Diagnosis | Attention failure is structural (order, length, no continuation), not missing callouts | 27 story boxes already exist and still fail |
| Form | Case-driven serial, not a novel | Preserves labs, evals, and CS audience |
| Protagonist system | Helix at Northstar + 4 people | Already implied by the running app and capstone starter |
| Off-canon plots | Recast 18, 24, 15, 21 onto Northstar or label side-quest | Serial attachment cannot survive a new industry every chapter |
| Opening order | Scene before objectives | Objectives are why attention dies in the first screen |
| Visual | Story panels equal to labs, not italic captions | Current CSS tells the eye the hook is optional |
| Rollout | Template on Gate 1, then fan out | Prevents 27 mediocre rewrites of the same 80-word blurb |
| Tracks | Separate casts, milestone returns | 90-day lists are a different genre; forcing Helix would confuse |

---

## 14. Open questions (resolve before PR2 copy is final)

1. **Names:** Keep Maya / Priya / Jules / Noah, or pick different ones? (Avoid names of real teammates.)
2. **Company name:** Northstar vs keep unnamed “the running app”? Named is stronger for memory; unnamed is safer if the course is rebranded.
3. **Module 15 (domain apps):** Helix-adjacent refusal (don’t ship a medical bot) vs a marked side-quest in another vertical.
4. **Module 18:** Recast the 400GB drive example onto ticket/order dumps, or keep it as a labeled side-quest because the parallel-subroutine lesson is clearer on files.
5. **Feedback loop:** Who from the original readers reviews PR2? Schedule that before PR3.

---

## 15. Success metrics

Qualitative (primary, matches the original complaint):

- Unprompted comments shift from “dry / hard to finish” to “I wanted the next module.”
- PR2 read-aloud: listeners can retell *what happened to Helix* in 01–03 without looking at headings.

Structural (objective, from this audit):

| Check | Today | After |
|---|---|---|
| Story before learning objectives | 0/27 | 27/27 |
| Named recurring cast in core | 0 | Helix + crew on ≥24/27 |
| Running-app / Helix callback | 6/27 | 27/27 |
| Mid-page scene return | ~0 | 27/27 |
| Next-module cliffhanger | ~0 | 27/27 |
| Opener length | ~50–90 words | 150–250 words |
| Think boxes in 18–19 | 0 | ≥1 each |

Do not A/B “time on page” in analytics; this site tracks progress in `localStorage` only and should stay that way.

---

## 16. Audit appendix (counts)

Story word counts are words inside `.aieng-story` only.

| Module | Lines | Story words | Running-app mention | Think | Quiz |
|---|---|---|---|---|---|
| 01 | 449 | 52 | yes | 2 | 2 |
| 02 | 403 | 61 | no | 2 | 2 |
| 03 | 420 | 52 | no | 2 | 2 |
| 04 | 440 | 51 | yes | 2 | 2 |
| 05 | 407 | 57 | yes | 2 | 3 |
| 06 | 362 | 49 | no | 2 | 3 |
| 07 | 400 | 75 | no | 3 | 3 |
| 08 | 516 | 93 | yes | 4 | 4 |
| 09 | 547 | 61 | no | 3 | 3 |
| 10 | 506 | 64 | no | 3 | 2 |
| 11 | 535 | 60 | no | 3 | 3 |
| 12 | 516 | 79 | no | 3 | 3 |
| 13 | 478 | 84 | yes | 2 | 2 |
| 14 | 410 | 92 | no | 2 | 2 |
| 15 | 383 | 81 | no | 1 | 2 |
| 16 | 411 | 85 | no | 1 | 2 |
| 17 | 515 | 94 | no | 3 | 3 |
| 18 | 519 | 135 | no | **0** | 3 |
| 19 | 570 | 154 | no | **0** | 3 |
| 20 | 336 | 94 | no | 1 | 3 |
| 21 | 325 | 71 | no | 1 | 2 |
| 22 | 320 | 73 | no | 1 | 2 |
| 23 | 242 | 69 | no | 1 | 2 |
| 24 | 286 | 81 | no | 1 | 2 |
| 25 | 293 | 82 | no | 1 | 2 |
| 26 | 266 | 70 | no | 1 | 2 |
| 27 | 337 | 91 | yes | 1 | 2 |
| Capstone | 56 | 0 | no | 0 | 0 |
| Tracks (stock / hybrid / plugin) | 943 / 747 / 935 | 75 / 0 / 0 | no | 4 / 0 / 0 | 0 |

---

## 17. PR plan (summary)

1. **Canon + CSS + core index** — name Helix, restyle story, write the story bible.
2. **Modules 01–03** — implement the template; **external read-aloud gate**.
3. **Modules 04–07, 09** — Gate 2–3 serial.
4. **Gate 4 modules** — re-home anonymous agents onto Helix.
5. **Gate 5 modules** — production season; recast off-canon plots.
6. **Capstone + getting started** — finish Maya’s service; setup as onboarding.
7. **Tracks** — own casts, milestone scene-returns, wrap existing incidents in story chrome.

No production code or eval fixtures change in this workstream.
