# Module 01 storytelling pilot

**Date:** 2026-09-17
**Branch:** `plan/tutorial-storytelling` (existing branch)
**Status:** implemented; source and generated-HTML checks passed; browser visual review and reader comparison pending
**Plan:** [Tutorial storytelling](tutorial-storytelling.md)
**Pilot:** [Module 01](../docs/core/01-prompt-engineering.md)
**Original:** `git show e4d0f8c:docs/core/01-prompt-engineering.md` preserves the comparison text without duplicating a published page.

## What changed and why

The opener now follows an email reply that invents an appointment and omits a required section. It appears before the unchanged learner objectives and explicitly labels the scenario as fictional. An early writing prompt asks the learner to propose a constraint and a format instruction before reading the anatomy table.

The walkthrough applies those choices to the same reply, then distinguishes correct headings from supported claims. A note after the runnable sketch explains what the learner still needs to add for the existing lab. The close describes observed variation and remaining uncertainty, then links to untrusted-input handling in Module 02. No recurring cast or global styling change is introduced.

Opening order, early action, and continuity are bundled in this version. A reader preference for it would not isolate the effect of any one change.

## Artifact mapping

| Stage | Implementation |
|---|---|
| Failure | Illustrative draft invents a Tuesday appointment; another omits a risks section |
| Decision | Specify permitted facts, behavior when availability is unknown, and required headings |
| Early action | Write one anti-invention instruction and one output-format instruction; no API call required |
| Application | Anatomy callback supplies example constraints; output-shape callback separates formatting from factual support |
| Actual artifact | Existing real-email/issue lab: five outputs at each temperature, observations, and a constraint motivated by an observed failure |
| Evidence | Learner records section presence, length, and invented facts; no successful model output is claimed by this editorial change |
| Limit | Small samples do not establish universal correctness; this lab does not modify the triage starter or authorize money movement |

The code blocks, lab steps and acceptance criteria, learning-objective bullets, and quizzes are unchanged. The old `why-this-matters-cs-engineer-view` heading anchor is preserved explicitly at the opener; other section headings retain their anchors.

## Reading-route findings

| Route | Pilot implication |
|---|---|
| Catalog / Gate 1 | 01 → 02 is unchanged; setup is the only prerequisite for 01 |
| Weekend Warrior | 01 and skimmed 02 precede 07, then 04/13; no assumption of earlier retrieval or eval implementation |
| Professional Developer | Foundations 01–04 remain compatible; no dependence on optional agent work |
| Enterprise Architect | Numeric early-core route remains compatible; later gate/catalog differences still require review before expansion |
| AI Researcher | Starts with selected later modules; future callbacks must not depend on reading this scene |
| Search / returning reader | Scenario states its own input and failure; objectives, mental model, code, lab, and quiz remain directly accessible |

For later rollout, distinguish Module 27's existing next link to 22 from Gate 5's first table entry, 13. Treat the former as an agent-evaluation continuation, not a claim that it is the universal next gate entry. No links or prerequisites outside Module 01 changed. A complete dependency/route review remains part of the advanced pilot.

## Verification evidence

- Source words: **3,103 → 3,265 (+5.2%)**, using whitespace splitting; below the approximate 15% growth guardrail.
- Narrative budget: **379 words**, conservatively including the opener, early action, added case callbacks, lab framing, close, and next link; below 400 total words.
- Compared all fenced code blocks with the original: unchanged. Also compared objective bullets, lab steps/acceptance criteria, and the entire quiz section: unchanged.
- Strict content build passed using the existing MkDocs configuration with only `plugins=['search', 'tags']` overridden in memory. Repository configuration was not changed.
- Full `mkdocs build --strict` is **not passed**: the existing environment lacks `PIL` and `cairosvg`, required by the social-card plugin. Social-image generation is outside the passing content build.
- Parsed generated Module 01 HTML: required section anchors exist; the problem precedes objectives; local page links and fragments resolve. This is structural validation, not visual validation.
- Browser connection failed twice before any page interaction because the tool reported missing `sandboxPolicy` metadata. Desktop/mobile layout, light/dark readability, first-screen density, and interactive lookup checks are **pending**. No screenshots or visual pass are claimed.
- No new tests were added for this prose change. No runtime code or eval fixtures changed.

## Reader comparison protocol

Recruit a small mix of intended readers, including the original feedback providers where available: a newcomer meeting setup prerequisites, an experienced engineer, and a search/short-path reader. Record experience, goal, device, version seen, order, and prior familiarity. No reader outreach has been sent.

Let participants read silently and normally; do not ask them to memorize the fiction. Preserve both versions using the baseline commit. Vary presentation order if a participant sees both, and record familiarity effects rather than treating the second reading as independent evidence.

1. Ask where they would stop or skip and why; ask whether they want to continue.
2. Ask them to explain why a reply with all required headings can still fail, and what the proposed constraint cannot guarantee.
3. Have them attempt the existing lab. Record progress and help needed, especially whether they understand the sketch still needs the lab's format requirements.
4. Use a fresh transfer case: a model proposes a price absent from the supplied product information. Ask what they would change and how they would check the output.
5. Ask a returning-reader lookup task: find the required lab headings and the section on versioning prompts. Record detours and confusion.
6. Ask about recall of the opening problem only after technical tasks. Character names and exact wording are not success criteria.

Read-aloud may supplement these tasks to detect awkward prose. Record conflicting observations, not just positive comments. Do not add analytics; use consented sessions and minimize identifying notes.

| Session | Reader context / order | Engagement | Technical explanation / transfer | Lab / lookup | Decision |
|---|---|---|---|---|---|
| Pending | No participants observed | Not measured | Not measured | Not measured | No rollout decision |

**Go/revise:** proceed to 02–03 only when feedback gives directional evidence of improved willingness to continue without material worsening of technical understanding, transfer, lab progress, or lookup usability. Resolve regressions and repeat affected checks. A small sample is not statistical proof. Record visual findings before declaring the pilot fully verified.

## Remaining work

- Restore browser access and inspect desktop/mobile, light/dark, opening density, anchors, and quiz/lookup interaction.
- Complete the full docs build when social-card dependencies are available.
- Run the reader comparison and record a go/revise decision. Module 02–03 and broader rollout remain pending under the plan's evidence gate.
