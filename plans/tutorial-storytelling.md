# Plan: Make the tutorials hold attention through storytelling

**Status:** Module 01 pilot implemented; content checks passed; visual and reader validation pending. See [pilot evidence record](tutorial-storytelling-pilot.md).
**Branch:** `plan/tutorial-storytelling`
**Scope:** improve how the curriculum connects problems, decisions, practice, and results while preserving what it teaches
**Non-goals:** a compulsory fictional serial, rewriting Python APIs or eval fixtures, replacing existing labs/quizzes, or substantially lengthening pages
**Validation:** source findings rechecked and corrections incorporated on 2026-09-17; baseline `e4d0f8c`. Evidence, limitations, and counts are recorded below.

---

## 1. Verdict and evidence boundary

Readers report that the content is valuable but difficult to stay engaged with. The source scan supports a plausible explanation: short incidents often introduce technically organized lessons without carrying the opening problem through to a demonstrated outcome. It does not establish that objectives, anonymous characters, or short openers cause disengagement, or that a named cast will solve it.

| Evidence level | Finding | Implication |
|---|---|---|
| Reported experience | Readers appreciate the content but report weak attention | Preserve technical value and investigate where attention drops |
| Observed structure | 27/27 core modules put incidents after objectives; story median is 75 words; six explicitly reference the running app | Test opening order and stronger continuity |
| Observed exceptions | Module 19 continues its contract case in code; tracks have milestones and reflective activities | Improve existing case progression instead of assuming it is absent |
| Hypothesis | Weak problem-to-result continuity contributes to disengagement | Compare a revised module with the original using readers |
| Unresolved alternatives | Difficulty, repetition, navigation, delayed practice, and visual density | Include these in the pilot rather than attributing all friction to storytelling |

The target is a **case-driven tutorial: problem → decision → action → demonstrated result**. A recurring company is an optional means of supporting that structure.

## 2. What was scanned and what remains to validate

| Surface | Scope | What the source review can establish |
|---|---|---|
| Core | All 27 openings and story blocks; counts; representative middle/code/lab/closing sections | Structure, inventory, and examples of continuity |
| Core framing | Index, capstone, gate checkpoints, starter | Gate transitions and story-to-code compatibility |
| Home / getting started | Landing, setup, paths, progress | Entry routes and setup expectations |
| Tracks | Stock, hybrid, plugin structure and representative sections | Existing incidents, milestones, reflection, and reference density |
| Reference | Assessment, exercises, progression and related lookup surfaces | Material whose retrieval usability must be preserved |
| Presentation | `.aieng-story` and related CSS | Styling rules, not their perceived effect |

This was not an exhaustive technical audit, rendered usability test, or observation of learners. Appendix counts describe source, not reading time. Before broad rewriting, inspect rendered pages and ask readers where they stopped, what they came to do, their prior knowledge, their device, and whether they were studying or looking up an answer.

## 3. Existing strengths to preserve

- All 27 core modules have one incident block; the stock track has one, and hybrid/plugin tracks have untagged incident prose.
- The core index already connects five gates through a support-ticket triage service and residual failures.
- Intuition locks, explainers, think prompts, labs, quizzes, and checkpoints already provide varied learning activities.
- Module 19 carries contract auditing into `Finding`, `chunk_by_section`, `map_find_predatory_clauses`, and `audit_contract`. Preserve that continuity.
- Hybrid has Markdown “Think-about-it” sections and ablation milestones; plugin has “Without vs. with” sections and staged working artifacts.
- The capstone's planted holes and failure injections already give the learner a problem → action → proof arc.

These are observed strengths. Reader feedback, rather than the number of components, is the evidence that the content is appreciated.

## 4. Problems to test

### Opening and continuity

All core incidents follow objectives; 18–19 also insert “What you can build.” Try a concise problem before objectives, but do not assume the current order causes abandonment. Stories range from 49–154 words (median 75). Length alone is not the defect: several openers reveal a root cause immediately, then the walkthrough changes to topic-by-topic exposition.

Concrete artifacts already exist: Module 09 has an exact error identifier and Hit@5; 14 has a questionnaire and ticket; 27 has tool calls. The question is whether the artifact drives subsequent decisions and proof. Explicit running-app references appear in 01, 04, 05, 08, 13, 27; this is not a count of all thematic continuity.

### Density and first action

Core pages span 241–569 source lines. The formal lab starts at line 354 of 448 in Module 01 and 478 of 569 in Module 19. Earlier code and questions may already engage learners, so measure the first meaningful action rather than treating lab position as proof of inactivity.

For pilot pages, inventory repeated definitions, new concepts, runnable versus illustrative code, first prediction/change/run, and visible milestones. Test a small early action using the existing material. Tracks already contain instructional progression; improve milestone visibility and chunking before adding prose.

### Presentation

The current `.aieng-story` uses italics, a left border, and no shared card background. That is a styling difference, not proof that readers perceive it as a caption. Compare readability and first-screen content in the browser before prescribing full-width colored panels. More panel weight can also compete with existing labs and quizzes.

### Semantic audit

Record whether each revised module has an unresolved problem, a learner choice, application to that problem, visible consequences, and a resolved outcome. Count plain Markdown activities as well as HTML blocks. Modules 18–19 have zero `aieng-think` blocks but three quizzes each; do not add boxes merely to meet a quota.

## 5. Editorial constraints

- Preserve learning objectives, diagrams, labs, quizzes, and code meaning. If a case cannot be resolved by existing material, narrow the promised outcome or explicitly describe transfer to another artifact.
- Keep objectives about what **the learner** can do. Retain `Learning objectives`, `Mental model`, `Core tutorial`, and lab headings by default for navigation and existing anchors.
- Use no minimum opener length. Budget at most **400 total narrative words per revised module**, including opening, all section transitions, callbacks, and close—not 400 net additions. Aim lower whenever the case is clear.
- Keep total source-word growth within approximately 15%, comparing before/after with the same whitespace-count method. This is a size guardrail, not a reading-time metric. Cut repetition or narrative before removing instructional substance.
- Vary openings: a failing request, test result, trace, conflicting requirement, or short scene. Names, dialogue, clocks, and cliffhangers are optional.
- Label invented incidents as fictional teaching scenarios where a standalone reader can see it. Distinguish illustrative artifacts/results from observed lab output.
- Do not invent repository files, runtime capabilities, or measured outcomes. Prompt constraints alone do not establish authorization or guarantee correctness.

## 6. Running case and reading routes

### Optional names, deferred commitment

The current docs and starter do **not** name Helix. Candidate names are **Helix** (service) at **Northstar** (company), with Maya (engineering), Priya (product), Jules (security), and Noah (finance/ops). Use only people needed by a case. Commit to a recurring cast only if pilot feedback supports it; a named company and four-person cast are not acceptance requirements.

The starter is a schema-valid triage endpoint with a deterministic keyword mock. A fictional pre-contract service is not its current implementation. Preserve this distinction whenever the story connects to starter work.

### Gate progression

| Gate | Problem motivating this gate | Capability at exit | Residual limitation / next question |
|---|---|---|---|
| 1 — Dependable model service | Soft output contracts and hostile input | Structured output checked; invalid output handled; policy/input boundaries established | Schema-valid output can still be wrong: measure quality |
| 2 — Measurable quality | Changes regress behavior unnoticed | Golden-set checks can block regressions | Passing known tests does not supply missing business knowledge |
| 3 — External knowledge | Missing or poorly retrieved policy | Budgeted retrieval with measured grounding and citation checks | Grounded answers do not authorize actions |
| 4 — Actions and agents | Unbounded or unauthorized tool execution | Runtime authorization, budgets, verification, and persistence | A local workflow still needs production hardening |
| 5 — Operate it | Traffic, drift, outages, and unclear costs | Observable, versioned, tested operations and rollback | Continued monitoring and incident rehearsal; no promise of permanent reliability |

These summarize the core gates; retain the existing detailed exit criteria and capstone checkpoint mapping, including serving discipline borrowed from Module 13. Do not relabel a gate's resolved entry failure as its exit failure.

### Navigation contract

- Numbering is catalog order, not a mandatory reading chain; `Depends on` remains authoritative.
- Every case states the current system capabilities briefly and stands alone for search arrivals and returning readers. Callbacks must not require remembering earlier fiction.
- Preserve Weekend Warrior, Professional Developer, AI Researcher, and other supported routes. A route that reaches 07 before 04 cannot assume the reader completed 04's story or lab.
- Use prerequisite-aware next-step links, with a reason to continue. No compulsory next-episode cliffhanger.
- Before writing cross-module closes, map catalog order, gate groups, `Depends on`/`Next`, and learning paths. Resolve discrepancies explicitly; for example, 27 currently links to 22 while Gate 5's table begins with 13. Do not silently change prerequisites to fit a plot.
- If a recommended serial route is useful, document it separately from catalog numbering and retain alternate entry paths.

## 7. Flexible module structure

```text
# Module NN — Title
Time / depends / next

Concise concrete problem (test before objectives in the pilot)
State current capabilities and the outcome the reader will demonstrate.
A scene, trace, failing request, or test can do this job.

## Learning objectives
Keep the learner-facing objectives and anchor.

## Mental model
Keep the diagram and useful CS framing; remove redundant motivation.

## Core tutorial
Use existing numbered sections, tables, and code.
Invite an early prediction or small action using existing material.
Apply the concepts to the opening problem where they actually fit.
Show a decision and its consequence; a code result can be the callback.

## Lab / quizzes
Keep existing tasks and acceptance criteria.
Explain whether the lab resolves the case or practices a transferable part.

Close
State what the artifact demonstrates and what it cannot yet guarantee.
Link a useful next step compatible with prerequisites and supported routes.
```

Do not prepend a fictional sentence to every H3. A mid-page scene is optional when code or a worked decision already continues the case. Preserve the CS framing that helps understanding. If removing or renaming a heading, check incoming links and preserve its anchor where needed.

For each pilot, fill in: **failure → taught decision → actual artifact → observable result → remaining limitation**. This mapping is required before drafting the opener.

## 8. Module 01 pilot: align the problem with the lab

The current 52-word incident concerns invented refunds. The actual lab asks learners to generate a Markdown reply for a real email or GitHub issue, compare five runs each at temperatures 0.2 and 0.8, and add a constraint based on a failure they observed. `src.prompts` integration is optional.

Use a case about inconsistent reply structure or invented details, which this lab can investigate. Illustrative opening, not final copy or measured output:

> You ask for a reply to the same customer email twice. One draft invents a meeting time; the other omits the risks section your reviewer needs. Before adding more examples, inspect the request: did it specify the facts the model may use and the sections it must return? In this module, you will turn one real message into a clearer contract, compare repeated outputs, and document what improved—and what still varies.

| Mapping | Pilot implementation |
|---|---|
| Failure | Inconsistent sections or unsupported details in a reply |
| Decision | Specify task, boundaries, format, and sampling choice |
| Artifact | Existing lab's prompt and ten output observations; optional `src.prompts` template |
| Early action | Predict what an underspecified request leaves open before reading the anatomy table |
| Proof | Compare section presence, length, and invented details using the learner's actual results |
| Limitation | A small sample and a constraint do not prove correctness or authorize side effects |
| Next step | Link Module 02 for untrusted-input boundaries without assuming a RAG system already exists |

Do not claim this lab patches a `triage.py` implementation or fixes refund authorization. Explain how the same contract discipline transfers to triage. Preserve the original version for comparison through a recorded commit or review artifact.

## 9. File-by-file implementation inventory

**P0:** pilot and evidence. **P1:** expansion after pilot gates. **P2:** later surfaces where evidence warrants changes. Items below describe candidate treatments, not a mandatory cast or guaranteed rewrite of every page.

### 9.1 Pilot and supporting material — P0

| File | Change |
|---|---|
| `docs/core/01-prompt-engineering.md` | Implement §8; compare opening order, early action, case-to-lab continuity, and close |
| `plans/tutorial-storytelling-pilot.md` (new during implementation) | Record baseline, artifact mapping, participant tasks, findings, route map, and go/revise decision |
| `docs/assets/css/gamify.css` | Only if rendered pilot review identifies a styling need; test scoped treatment before a global change |
| `docs/core/02-security-privacy.md`, `03-advanced-prompting.md` | Extend after Module 01 review; test continuity without requiring earlier plot recall |
| `docs/core/19-orchestration-patterns.md` or `27-harness-engineering.md` | Test standalone entry before broad rollout; record which was selected and why |

### 9.2 Gates 1–3 — P0 pilot / P1 expansion

| Module | Candidate treatment and alignment check |
|---|---|
| 01 Prompt engineering | Reply-contract case and existing lab; see §8 |
| 02 Security | Keep the malicious-document incident; state the system boundary and connect to existing security exercises; do not imply learners have built retrieval already |
| 03 Advanced prompting | Keep parse failure and cost tension; use the decision map to choose a lever and state what the existing lab proves |
| 04 Testing & evals | Carry the quality regression into the golden-set decision and test result; distinguish illustrative scores from measured output |
| 05 Context engineering | Use drowned policy to motivate packing choices and an observable context-budget result |
| 06 Fine-tuning | Keep the stale-catalog decision case; show why the FT-versus-RAG decision follows from the evidence; no forced industry change |
| 07 Tools & RAG | Distinguish knowledge failure from unsafe action; preserve the authorization boundary and show what each example actually fixes |
| 09 Advanced RAG | Carry `ERR_INV_88421` into retrieval diagnosis and metrics; it is an error/runbook id, not Module 14's ticket |

### 9.3 Gate 4 — P1

| Module | Candidate treatment and alignment check |
|---|---|
| 08 MCP | Preserve host-versus-server policy distinction; show the runtime decision that resolves auto-approval risk |
| 10 Cost | Tie routing savings to success/reopen measures; treat cache isolation as a separate failure when necessary |
| 11 Single agents | Carry repeated tool calls into step caps and repeated-argument abort evidence |
| 12 Multi-agent | Connect topology decisions to an existing task and its cost/quality evidence; do not invent a Helix doc-draft path as implemented code |
| 16 Integration | Follow a timed-out request through jobs/queues and request-id propagation |
| 18 Patterns | Keep the file-decomposition case if it teaches the primitives clearly; recast only if existing code and lab still fit. Add reflective activity only where the semantic audit finds a gap |
| 19 Orchestration | Preserve contract-audit continuity already present in code; strengthen choice and outcome rather than adding a redundant scene |
| 20 Reliability | Connect named failure families to detectors, tests, and residual risk |
| 21 Secure tools | Keep the editor/sandbox case as a self-contained example; an unrelated service cameo is unnecessary |
| 27 Harness | Preserve the existing triage/tool-loop continuity. Helix is not currently named. Show external verification and stop conditions; check 22/13 route ambiguity before changing the close |

### 9.4 Gate 5 — P1

| Module | Candidate treatment and alignment check |
|---|---|
| 13 Production | Follow hung requests and prompt drift into deadlines, traces, and versioning |
| 14 Compliance | Carry ticket `#88421` into provenance and controls; do not equate it with 09's error code. Any relationship would be explicitly fictional and newly introduced |
| 15 Domain apps | Preserve the domain-specific case where it explains the decision best; introduce its context locally |
| 17 Small models | Connect the model swap's failures to measured routing tradeoffs |
| 22 Agent evals | Compare trajectory evidence and outcome quality on the same task |
| 23 Drift | Carry the warmth tweak into digest-versus-version-name checks and actual limits |
| 24 Local-first | Keep the laptop/repo example and budget lesson; no mandatory company recast |
| 25 Durable | Match the interruption/resume incident to actual coordinator, worktree, and approval-state material |
| 26 Orchestrators | Compare engines on one workflow and observable cost/step ownership, with a self-contained entry |

### 9.5 Framing, capstone, setup, and tracks — P1 / P2

| Surface | Change after pilot evidence |
|---|---|
| `docs/core/index.md` | Clarify running-case capabilities, entry/exit/residual failures, and routes; retain gate criteria |
| `docs/index.md` | Optional brief problem-to-result promise consistent with the tested teaching approach |
| `plans/helix-story-bible.md` (optional new file) | Only if recurring names help: record fictional names, capabilities, identifiers, code mappings, and contradictions to avoid |
| `docs/core/capstone.md`, `capstone-gates.md` | Strengthen the learner's existing planted-hole → fix → proof arc; preserve schema-valid mock starting state and operational checklists |
| `docs/getting-started/setup.md` | Keep the first successful command easy to reach; add context only if it aids onboarding |
| `docs/getting-started/paths.md` | Preserve distinct goals, skipped modules, and cadence; explain optional case continuity |
| `docs/getting-started/index.md`, `progress.md` | Light touch based on entry/lookup testing |
| Stock track | Carry time-safe split, citations, and eval decisions through existing milestones; preserve the non-advice warning |
| Hybrid track | Preserve incident, Markdown reflection, ablations, and exits; strengthen visible before/after evidence |
| Plugin track | Preserve “Without vs. with” progression, approval gate, MCP trust, and local-model milestones |
| `docs/tracks/index.md` | Explain distinct projects and prerequisites; no required separate companies or casts |

Keep reference documents as lookup material. Do not story-wrap `src/`, tests, or eval fixtures.

## 10. Per-module acceptance criteria

1. The opening defines a concrete problem and a learner-relevant outcome using only necessary context.
2. The walkthrough applies a consequential decision to that problem; code, a worked example, or a scene can provide continuity.
3. The lab/result connection is truthful: identify what the existing artifact proves, transfers, or leaves unresolved.
4. The reader gets an early opportunity to predict, inspect, or act where appropriate; do not duplicate exercises solely to add a beat.
5. Technical meaning and existing learning objectives, labs, quizzes, diagrams, and code paths are preserved.
6. A standalone reader can follow the case, and next steps respect prerequisites and supported paths.
7. The narrative and total-growth budgets in §5 are met; names, opener word minima, think-box counts, and cliffhangers are not completion gates.
8. Headings, anchors, links, code readability, and rendered desktop/mobile light/dark layouts remain usable.
9. Pilot evidence meets §15 before the treatment becomes a course-wide default. Read-aloud is an editorial check, not sufficient acceptance evidence.

## 11. Boundaries

No character illustrations, comic panels, real-person likenesses, or claims of real outages. Keep this plan internal to the repository, outside MkDocs navigation. Do not change production code or evaluation fixtures in this workstream. If a proposed narrative needs new functionality to be true, revise the narrative or raise a separate scoped implementation proposal.

## 12. Implementation sequence and review gates

| PR | Scope | Entry / exit condition |
|---|---|---|
| PR1 | Module 01 pilot and pilot record; minimal scoped styling only if needed | Record baseline and comparison tasks, draft artifact mapping, render/check links, then reader comparison |
| PR2 | Modules 02–03 continuity pilot | Proceed after 01 gives directional evidence of improvement without learning/lookup regressions; test skipped-entry comprehension |
| PR3 | Standalone-entry pilot on 19 or 27; finalize route rules and reusable guidance | Proceed after PR2 review; preserve existing case strengths and test an advanced/search arrival |
| PR4 | Core framing and optional names/canon/global CSS | Only after pilot evidence supports these choices; no full cast or global panel rollout as a prerequisite to learning from PR1 |
| PR5 | Gates 2–3: 04–07, 09 | Apply tested functional guidance; each module gets an artifact mapping and route check |
| PR6 | Remaining Gate 4 modules | Exclude whichever advanced pilot is already done; use small independently reviewable batches |
| PR7 | Gate 5 modules | Apply the same checks; keep domain/repo cases where clearer |
| PR8 | Capstone and getting started | Preserve starter entry conditions and first-command/lookup usability |
| PR9 | Tracks | Build on milestones and existing reflective activities; no automatic scene-wrapper pass |

Pause expansion after each pilot for actual reader evidence. If readers are unavailable, complete a reviewable pilot and its technical/rendered checks, record reader validation as pending, and do not treat elapsed time or self-review as a passed gate. No outreach is part of this plan unless separately authorized.

## 13. Decisions and hypotheses

| Item | Position |
|---|---|
| Diagnosis | Structural continuity is a supported hypothesis; reader behavior is still to be observed |
| Teaching form | Self-contained cases with optional recurring context |
| Opening order | Test concise problem before objectives in Module 01 |
| Protagonist | Learner's engineering decision and artifact; Helix/cast optional |
| Existing varied domains | Preserve where they teach the concept clearly |
| Visual treatment | Decide from rendered pilot evidence, not CSS alone |
| Rollout | 01 → 02–03 → advanced standalone entry → broader adoption |
| Track treatment | Strengthen existing milestone progression before adding fiction |

## 14. Questions to resolve during the pilot

1. Where did original readers disengage, and were they studying, skimming, or solving a specific problem?
2. Which intended readers can compare versions, including newcomers, experienced engineers, and search/short-path arrivals?
3. Does early action or reduced repetition help more than adding narrative? Record which changes were bundled so their effects are not falsely separated.
4. Does a recurring name help comprehension or just story recall? Keep the unnamed case as a valid outcome.
5. Which advanced module best tests standalone use, and how should route links handle existing catalog/gate differences?

Names and company branding do not block the first pilot. Reader evidence does block broad standardization.

## 15. Success measures and reader protocol

Before sessions, record the original version, participant context, tasks, intended outcomes, and go/revise criteria in the pilot record. Use a small mix of intended readers. Preserve the original for comparison; where someone sees both versions, vary order and account for familiarity. Small samples provide directional evidence, not statistical proof.

Ask readers to study normally before asking about the fiction. Record confusing/skippable passages and where they stop. Then ask them to explain the technical choice, attempt the existing lab, apply the idea to a fresh related scenario, and find a particular answer as a returning reader. Include a standalone entry and a shortened reading route during the later pilots. Read-aloud may supplement these tasks but must not replace silent reading and code use.

| Measure | Desired evidence |
|---|---|
| Willingness to continue | Clearer reported interest, fewer identified passages readers want to skip, with reasons |
| Technical understanding | Correct explanation of the decision and its limitations |
| Transfer | Appropriate application to a fresh related problem |
| Lab progress | Independent progress on the existing task, without extra confusion about what the code implements |
| Lookup usability | Can find the relevant concept, code, and lab without reading fictional setup |
| Standalone comprehension | Can understand the case without earlier episodes |
| Narrative recall (secondary) | Remembers the problem and consequence; character names are not required |

**Go/revise rule:** expand when reader feedback indicates better willingness to continue without observed worsening of technical understanding, lab progress, transfer, or lookup usability. Record contrary findings; resolve material regressions and repeat only the affected checks before expansion. Do not pass a pilot solely on story recall or format compliance.

Track semantic continuity and size budgets for editorial QA. Do not target 27 named casts, 27 cliffhangers, or longer openers as success measures. Reading time can reflect interest or confusion; do not optimize for it. Keep site progress in `localStorage`; use consented reader sessions without adding analytics.

## 16. Audit appendix (counts)


Baseline: source reviewed at `e4d0f8c`. Lines use Python `splitlines()`; story words use whitespace splitting inside `.aieng-story` after removing HTML tags. Think and Quiz count CSS classes only. Running-app mentions are explicit wording, not a semantic continuity score. Zero tagged story/think blocks does not mean zero narrative/reflection; hybrid and plugin tracks use Markdown equivalents. These are inventory measures, not engagement outcomes.

| Module | Lines | Story words | Running-app mention | Think | Quiz |
|---|---|---|---|---|---|
| 01 | 448 | 52 | yes | 2 | 2 |
| 02 | 402 | 61 | no | 2 | 2 |
| 03 | 419 | 52 | no | 2 | 2 |
| 04 | 439 | 51 | yes | 2 | 2 |
| 05 | 406 | 57 | yes | 2 | 3 |
| 06 | 361 | 49 | no | 2 | 3 |
| 07 | 399 | 75 | no | 3 | 3 |
| 08 | 515 | 93 | yes | 4 | 4 |
| 09 | 546 | 61 | no | 3 | 3 |
| 10 | 505 | 64 | no | 3 | 2 |
| 11 | 534 | 60 | no | 3 | 3 |
| 12 | 515 | 79 | no | 3 | 3 |
| 13 | 477 | 84 | yes | 2 | 2 |
| 14 | 409 | 92 | no | 2 | 2 |
| 15 | 382 | 81 | no | 1 | 2 |
| 16 | 410 | 85 | no | 1 | 2 |
| 17 | 514 | 94 | no | 3 | 3 |
| 18 | 518 | 135 | no | **0** | 3 |
| 19 | 569 | 154 | no | **0** | 3 |
| 20 | 335 | 94 | no | 1 | 3 |
| 21 | 324 | 71 | no | 1 | 2 |
| 22 | 319 | 73 | no | 1 | 2 |
| 23 | 241 | 69 | no | 1 | 2 |
| 24 | 285 | 81 | no | 1 | 2 |
| 25 | 292 | 82 | no | 1 | 2 |
| 26 | 265 | 70 | no | 1 | 2 |
| 27 | 336 | 91 | yes | 1 | 2 |
| Capstone | 55 | 0 | no | 0 | 0 |
| Tracks (stock / hybrid / plugin) | 942 / 746 / 934 | 75 / 0 / 0 | no | 4 / 0 / 0 | 0 |


---

## 17. Next action

Complete the pending visual and reader checks in the [pilot evidence record](tutorial-storytelling-pilot.md), then compare with the original before extending the approach to Modules 02–03. The content build and generated-link checks have passed; they do not substitute for reader evidence.
