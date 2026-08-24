# Capstone — An Evaluated, Authorized, Tool-Using Production AI Service

The five gates are learned in sequence; the capstone proves you can hold all five at once, in a single service, under the same request path. It is **domain-agnostic** — unlike the [specialization tracks](../tracks/index.md), which go deep on one vertical (stock research, hybrid architectures, an editor plugin) over 90 days, the capstone is the generic "does the whole chain actually work together" checkpoint. Run it instead of a track, or alongside one — a track's day-90 demo can double as your capstone if it satisfies every row below.

For the oral defense format (architecture sketch, live happy path, failure demo, Q&A), see [Assessment rubrics → Capstone oral](../reference/assessment.md#capstone-oral-optional-15-min). This page is the **build spec**; that section is the **defense format**.

## The six parts

| Part | You must ship | Gate(s) it proves | Modules to have completed |
|---|---|---|---|
| **Core service** | Provider abstraction, structured/schema-valid output, deadlines on every model call, retries, fail-closed validation | 1 | 01, 02, 03, 13 |
| **Evaluation** | A golden set, deterministic checks, quality metrics, and a CI gate that can fail a build on regression | 2 | 04, 22 |
| **Knowledge** | Retrieval with citations, a measured decision on whether retrieval is even needed, freshness handling, retrieval metrics (recall/precision/groundedness) | 3 | 05, 07, 09 |
| **Agent** | Tool use with an enforced permission boundary, loop/step/cost caps, persisted state, human approval on at least one write action | 4 | 08, 10, 11, 12, 20, 21 |
| **Operations** | Traces tied to a `request_id`, a cost dashboard, p50/p95/p99 latency, a documented fallback path, one rehearsed incident scenario | 5 | 13, 17, 22, 23 |
| **Security** | Authorization enforced outside the model (not a prompt instruction), parameter validation on every tool call, resilience to a prompt-injection test case, least privilege on anything that writes | 1, 4 | 02, 08, 21 |

## Definition of done

- [ ] Demo runs from a clean clone with documented setup
- [ ] At least one automated test suite **and** one model/behavior eval, both runnable in CI
- [ ] An architecture diagram in the README that matches the code, not an aspirational one
- [ ] Known failure modes are written down, with which ones are handled and which ones aren't
- [ ] A live failure demo works: kill a dependency, feed it a prompt-injection payload, or force an empty retrieval, and show the system degrade the way it's supposed to
- [ ] Every claim in the "Operations" and "Security" rows above is demonstrated live, not asserted in a slide

## What "done" is not

A capstone that only has a happy-path demo has not closed Gate 4 or Gate 5 — those gates exist specifically because happy-path demos are where most production AI systems stop and where most production AI incidents start. Budget real time for the failure demo; it is graded, not optional.
