# Command-runtime progress

Tick a box only when the exit criteria in [docs/core/capstone-command-gates.md](../docs/core/capstone-command-gates.md) are true for this runtime. The build spec is [docs/core/capstone-command.md](../docs/core/capstone-command.md).

Starter holes (leave the `GATE` comments in place until the gate is closed, then update the test that records the hole):

- [ ] Gate 1 — free-form intent still returns a `code.patch` candidate; `unknowns` still validate; `spec.draft` still patches; `DRAFT` can enter `GENERATING`; `target`, `forbidden_paths`, and `output_contract` are not on the model yet
- [ ] Gate 2 — a `must` requirement with no acceptance criterion still validates
- [ ] Gate 3 — context is whatever the mock ignores; repository comments are not fenced as untrusted data
- [ ] Gate 4 — patch policy accepts `setup.py`; verification is a shell string; repair cap is 99; a repeated patch is not detected; `risk_level: low` waives the envelope
- [ ] Gate 5 — `freeze` stores `revision: null`, so a later spec edit cannot invalidate an approval; attempts are not traced

---

## Gate 1 — A frozen specification

- [ ] `unknowns` non-empty returns `SPEC_INCOMPLETE` and does not call the model
- [ ] A natural-language sentence can draft a spec and cannot return a diff
- [ ] `spec.draft` stops at a contract the user must confirm
- [ ] Each field is `provided`, `inferred`, `unknown`, or `not_applicable`
- [ ] `target`, `forbidden_paths`, structured verification, and `output_contract` validate
- [ ] No transition from `draft` to `generating`
- [ ] Failure injection: `python cmdai.py run --intent "add retries"` prints `SPEC_INCOMPLETE`

## Gate 2 — Evidence for every must

- [ ] Every `must` requirement has an acceptance criterion, and every criterion cites a real requirement
- [ ] Vague criteria ("robust", "clean", "fast") are rejected or labeled `human_review`
- [ ] Verification is `{runner, target, timeout_seconds}` from an allowlist
- [ ] The same 4B model has been run through the five stages (free-form, schema, context, validators, two repairs) on at least 20 tasks
- [ ] Numeric thresholds were chosen on a calibration slice, before the final table
- [ ] You used the Module 04 paired-release rule before claiming the contract helped

## Gate 3 — Bounded context

- [ ] The model receives the frozen spec, target excerpts with path and line range, related tests, and (on repair) the latest failure only
- [ ] Missing evidence returns `ERROR:INSUFFICIENT_CONTEXT`
- [ ] A source comment that says "ignore the spec" does not change the diff
- [ ] The trace records the context manifest (paths and ranges), not a dumped repository

## Gate 4 — The runtime decides

- [ ] Paths, file count, added lines, renames, binaries, symlinks, lockfiles, secrets, and dependency files are checked before apply
- [ ] Tests run with no credentials, no home-directory mount, no network, and a timeout
- [ ] `verification_plan` builds an argv list; `shell` is false
- [ ] At most two repair attempts; an identical patch hash escalates
- [ ] A policy violation stops; it is not fed back as a repair hint
- [ ] `eligible_for_4b` reads file count, line budget, dependencies, and unknowns — not `risk_level`
- [ ] The candidate is shown for human approval and is not merged

## Gate 5 — A revision you can roll back

- [ ] Freeze stores a hash of the canonical spec
- [ ] Every attempt logs that hash, the context manifest, model and decoding settings, the raw candidate, policy and test outcomes, latency, and a failure code from the taxonomy
- [ ] Editing the spec after approval requires a new revision
- [ ] You can quote first-pass success, final success, hidden-test success, scope violations, invalid outputs, retry yield, escalation precision, and human correction time
