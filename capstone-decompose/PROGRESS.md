# Divide, solve, and join — progress

Tick a box only when the exit line in [docs/core/capstone-decompose-gates.md](../docs/core/capstone-decompose-gates.md) is true. Hints for the judgment calls are in the [build spec](../docs/core/capstone-decompose.md).

Starter holes:

- [ ] Gate 1 — `run` is still `one_shot` / `done`; a dropped requirement, an invented check, a missing seam, and a cycle still plan cleanly
- [ ] Gate 2 — join check V4 can be ignored; nothing stops a divider that writes its own checks
- [ ] Gate 3 — no part is solved on its own context; `route_part` never returns `32b`
- [ ] Gate 4 — `join_parts` returns `done` with `join_checks_ran: []`
- [ ] Gate 5 — no spec hash, no frontier ruler, no paired comparison

## Gate 1 — The one-shot prompt stays a baseline

Hint: keep the failing one-shot diff. Name the check it missed and the file it actually edited.

- [ ] `run` reports the one-shot as a failed baseline, not as the solution
- [ ] A plan that drops a `must`, invents a check id, cycles, or imports a seam nobody exports is rejected before any solve

## Gate 2 — Your checks, cited by id

Hint: if you have only per-module tests, add one `scope: join` check through the public entry point before you trust a green part.

- [ ] Every `must` has one owner, and every cited check id is one you wrote
- [ ] V4 is not assigned to a part as a way to skip implementing R3

## Gate 3 — Solve one part at a time

Hint: if the part prompt contains the original mega-prompt, delete that paste. That paste is the call that fails.

- [ ] Each solve call sees only that part's goal, files, seams, and checks
- [ ] Two failures because the part is still two subsystems route that part to 32B, and the other parts stay on 20B

## Gate 4 — Join the parts into one solution

Hint: when the joiner wants to edit `ledger.py`, re-run V1. If V1 breaks, revert the part and fix the call site.

- [ ] An imported seam that was not exported is `SEAM_MISMATCH`
- [ ] Every `scope: join` check runs on the assembled tree
- [ ] A join that edits an owned file re-runs that part's checks
- [ ] The model's "done" is stored and is not the status

## Gate 5 — A comparison you can re-run

Hint: the frontier run uses your checks. It does not replace the join.

- [ ] One-shot, parts, and join are logged under a hash of the spec and the checks
- [ ] A frontier run on that same hash is in the notes
- [ ] The claim that the divided path beat the one-shot, or matched the frontier ruler, uses the Module 04 paired rule
