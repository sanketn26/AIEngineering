---
description: Design durable workflow orchestration with a persistent coordinator, hypothesis trees, isolated worktrees, and human-in-the-loop merge gates.
---

# Module 25 — Durable Orchestration & Real Agent Patterns

**Time:** 7–10 days · **Depends on:** [12](12-multi-agents.md), [19](19-orchestration-patterns.md), [21](21-secure-tool-use.md), [22](22-agent-evaluation.md) · **Next:** [Orchestrators in production](26-orchestrator-comparison.md)

<span data-module-id="25" hidden></span>

---

<span id="why-this-matters-cs-engineer-view"></span>

<div class="aieng-story" markdown>

*Fictional teaching scenario.*

A “codebase investigator” is supposed to find why billing double-charges. It chats for forty minutes, holds the hypothesis in free-form CoT, writes the user’s tree directly, dies on a laptop sleep, and comes back with no memory. A junior re-runs it; it files a PR that fails tests; merge is a Slack thumbs-up. Durable orchestration is the opposite design: **coordinator with a log**, **tree of claims with scores**, **photocopy worktree**, **tests + approval before merge**, **HITL as a state**, not a hope.

</div>

**Case question:** What must be persisted, isolated, tested, and approved so the investigation can resume after interruption without corrupting the user tree?

## Learning objectives

- Run a **long-running coordinator** that persists phases and can pause/resume
- Grow a **hypothesis tree** and back-propagate evidence to parent claims
- Execute writes in an **isolated worktree**, then pass a **merge gate**
- Keep **durable state** (append-only log) and **human-in-the-loop** as first-class states
- Eval the whole graph under **cost and latency** caps (Module 22), not only the last message

---


Modules 18–19 gave leaf patterns and workflow *shape*. This module is the **end-to-end machine**: something that can survive a restart and refuse to merge junk.

<div class="aieng-intuition" markdown>
<p class="label">Intuition lock</p>

**Sticky picture:** The coordinator is a **job queue with named phases**. The hypothesis tree is a **bug tracker**: children prove or kill the parent. A worktree is a **scratch branch**. The merge gate is **CI + CODEOWNERS**. The journal is a **receipt book**: a finished line records one decision; an unfinished last line is not a receipt. HITL is a **paused coroutine**, not a print statement.

<div class="kill" markdown>
**Kill this idea:** “Long-running means a bigger context window and a while loop.” → **Replace with:** Persist events, isolate side effects, gate merges, pause for humans, resume from the log.
</div>
</div>

---

## Mental model

```mermaid
flowchart TB
  Goal --> Coord[Coordinator + DurableStore]
  Coord --> H[HypothesisTree]
  H -->|frontier| W[WorktreeExecutor]
  W --> Tests[run_tests in copy]
  Tests --> Gate[MergeGate]
  Gate -->|allow| Apply[User tree / PR]
  Gate -->|ask_human| HITL[paused event]
  HITL -->|human payload| Coord
  Coord --> Eval[Module 22 dashboard]
```

**Invariant:** replay preserves each complete, committed decision. An unfinished final record is discarded; corruption in a complete record fails closed. A worker interrupted before completion may run again, so its side effects need stable idempotency keys. The journal has one writer; it is not a multi-worker database.

---

## Core tutorial

### 1. The crash between “yes” and “done”

A human clicks **Deny**. You write “approval resolved” to disk. Before you write
“workflow aborted,” the process dies. On restart, the program sees no pending
approval and no abort. Which fact should it believe?

The problem is that one decision was split into two writes. `Coordinator.resume`
now records a single `hitl_resolved` event containing the boolean decision, phase,
and structured result. Replay derives the next state from that event. There is no
second write it must hope survived.

```python
from pathlib import Path
from src.durable import DurableStore

store = DurableStore(Path("events.jsonl"))
store.append("hitl", {"phase": "review", "result": {"facts": ["duplicate charge"]}})
store.append("hitl_resolved", {"phase": "review", "approved": False})
# Restart: the denial itself is sufficient to recover the aborted state.
store = DurableStore(Path("events.jsonl"))
```

`append` writes a complete newline-terminated record, flushes it, and calls `fsync`
before acknowledging success. In-memory state advances only after that succeeds.
On restart, an unfinished final line is removed before another append; a malformed
complete line is an error, not something the loader quietly skips. The loader reports
`recovered_tail_bytes` so a torn write is visible.

A failure during `fsync` is an **uncertain commit**: the caller cannot assume the
record is absent. Stop and reopen the journal to recover its state. Do not keep
appending from stale memory. The directory should already be on persistent storage;
this example does not establish guarantees for network filesystems or failing disks.
Only one process may own a journal at a time. Use a transactional store and leases
when workers can race.

**Predict:** what if the worker already sent the refund, but crashed before recording
`phase_done`? The journal cannot infer whether the recipient received it. We will
make that awkward moment happen in the lab below.

### 2. Coordinator: run until a gate

```python
def research(ctx):
    return {"facts": [...], "ask_human": "approve write?"}

def write(ctx):
    return {"ok": True}

c = Coordinator(store, ["research", "write"], {"research": research, "write": write})
paused = c.run_until_gate({})
assert paused["status"] == "paused"
resumed = c.resume({}, {"approved": True})   # next phase runs
# c.resume({}, {"approved": False}) → status "denied"; write never runs
```

`ask_human` does **not** complete the phase. One `hitl_resolved` event records the decision: approval completes that phase; denial aborts. Only a real boolean is accepted—`"false"` is a nonempty string, not permission. Replayed results are available in `context["phase_results"]`. An application must schedule approval expiry and submit a denial; this coordinator has no background timeout scheduler.

```mermaid
stateDiagram-v2
  [*] --> running
  running --> paused: phase returns ask_human (HITL event persisted)
  paused --> running: one approved decision event, next phase starts
  paused --> denied: resume(approved=False)
  paused --> denied: application records expiry as denial
  denied --> [*]
  running --> done: last phase completes
  done --> [*]
```

Once the pause event has committed, a restart recovers the pending question. Once the decision event has committed, it recovers that decision. Killing the process between those states leaves it paused; killing it after the denial leaves it denied. The tests terminate a real child process to check this boundary.

---

### 3. Hypothesis trees and insight backprop

```python
from src.durable import Hypothesis, HypothesisTree

tree = HypothesisTree()
tree.add(Hypothesis(id="root", claim="double charge is a webhook retry"))
tree.add(Hypothesis(id="c1", claim="idempotency key missing", parent_id="root"))
tree.record_evidence("c1", "logs: two POSTs same invoice", delta=0.4)
# child score high → parent score rises (backpropagate)
```

Use this for research/investigation agents:

1. Manager proposes 2–5 **competing** hypotheses (cap the fan-out).
2. Workers gather evidence **in isolation** (Module 18 subroutine).
3. Evidence updates the child; a fraction **back-propagates** to the parent.
4. `frontier()` is what you work next — do not DFS the whole tree.

This is not gradient descent. It is a **scored AND/OR tree** so the coordinator does not forget why it is reading file 40.

<div class="aieng-explainer" markdown>
<p class="label">Explainer</p>

**Backpropagation of insight** means: a leaf finding should change the parent’s priority, or you will keep exploring a refuted story. If the child is refuted (`score <= 0.2`), the parent should drop too (negative delta). Cap depth; cap total nodes; each node has a token/step budget (Module 20/24). Unbounded trees are runaway loops with prettier names.
</div>

```mermaid
flowchart TB
  Root["root: 'double charge is a webhook retry'<br/>score: rising"]
  C1["c1: 'idempotency key missing'<br/>evidence: two POSTs same invoice<br/>delta +0.4"]
  C2["c2: 'race in payment worker'<br/>evidence: refuted by logs<br/>delta -0.3"]
  Root --> C1
  Root --> C2
  C1 -.->|score rise backpropagates| Root
  C2 -.->|score drop backpropagates| Root
```

`record_evidence` moves the child's own score by `delta`, then backpropagates **half** that delta to the parent — one level up, not recursively to the grandparent. So `c1`'s `+0.4` raises `root` by `+0.2`, and `c2`'s `-0.3` lowers `root` by `-0.15`; the parent's score is a damped blend of what its children are finding. `frontier(min_score=0.4)` only returns **leaf** nodes still `open` and above the floor — so once `c2` drops to `refuted`, work shifts to `c1` and any new children of it, not back to the root.

---

### 4. Isolated executor + merge gate

```python
from src.sandbox import WorktreeExecutor
from src.durable import MergeGate

with WorktreeExecutor(source) as wt:
    wt.write_file("src/foo.py", new_src)
    tests_ok = run_pytest(wt.path)
    decision = MergeGate().review(
        tests_passed=tests_ok,
        diff_files=["src/foo.py"],
        approved=human_said_yes,
    )
# original tree unchanged unless you apply after decision["allow"]
```

| Gate check | Why |
|------------|-----|
| Tests passed | Don’t merge red |
| Human approved | Dual control on writes |
| Diff size cap | Stop “rewrite the monorepo” |

Production: this is a PR, not `shutil.copy` back. The teaching gate is the **policy**, GitHub/GitLab is the **transport**.

---

### 5. Eval the graph, not the soloist

Long coordinators fail on **cost and latency** even when the answer is right:

- Per-phase `CostEvent` (Module 26)
- Trajectory per worker + composite (Module 22)
- Wall-clock SLO: pause and HITL rather than spin

A coordinator that takes 25 minutes and $12 to find a one-line fix is a failed design unless you documented that trade.

<div class="aieng-think" markdown>
<p class="label">Think about it</p>

**Question:** The coordinator persists phases, but each worker still dumps its full scratchpad into the next phase’s context. What went wrong?

<details data-think-id="25-t1"><summary>Reveal a strong answer</summary>

You persisted **the wrong artifact**. Durable events should carry **compressed results** (`facts[]`, hypothesis scores, diff paths) — Module 19 planner rule. Scratchpads stay inside the worker. Otherwise restart is durable *and* you still drown the window (Module 05). Pair store payloads with schemas; reject oversized events.

</details>
</div>

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Restart replays side effects | Non-idempotent workers | Idempotency keys; worktree until merge |
| Tree explodes | No cap on children | Max nodes; min_score frontier |
| Merge thumbs-up in Slack | No `MergeGate` | Tests + approval object |
| HITL deadlock | No timeout | Deny by default (Module 21) |
| Cheap demo, $ prod | No per-phase budget | Modules 22 + 26 |

---

<div class="aieng-case-checkpoint" markdown>
<p class="label">Case checkpoint</p>

**Opening failure:** A long investigation lost its hypothesis on restart and wrote an untested patch directly into the user's tree.

**What this lab demonstrates:** The durable log, hypothesis frontier, pause/resume path, denial assertion, and merge gate prove state and side effects survive interruption safely.

**What it does not prove:** Persistence can faithfully preserve a wrong hypothesis, so evidence scoring and human review still matter.

</div>

---

## Lab

1. `HypothesisTree`: child evidence raises parent score; `frontier()` returns leaves.
2. `DurableStore` on a temp JSONL; new instance sees the committed event. Append half a final record and restart: recovery reports discarded bytes. Corrupt a complete record: recovery must refuse it.
3. Coordinator pauses on `ask_human`; **denial does not run the next phase**; approval then resumes.
4. `MergeGate`: fail tests → `allow` false; tests + approval → true.
5. Stretch: run a worktree write + gate in one script (no live model).

```bash
poetry run pytest tests/test_durable.py tests/test_sandbox.py -v
```

---

## Crash experiment — the receipt arrived, the notebook did not

Before running, choose a prediction: zero refunds, one refund, or two?

```bash
python -m examples.durability.crash
```

The first process commits a simulated effect at the receiver, then exits before
writing `phase_done`. The next process replays the workflow and runs the worker
again. There are **two attempts but one effect**: the receiver stores the stable
operation ID `ticket-7:refund` under a unique constraint in the same transaction
as its effect. A fresh UUID on every retry would defeat that protection.

```mermaid
sequenceDiagram
  participant W as Worker
  participant R as Receiver
  participant J as Journal
  W->>R: refund(ticket-7:refund)
  R-->>W: committed once
  Note over W,J: Worker dies before phase_done
  W->>J: replay after restart
  J-->>W: refund phase incomplete
  W->>R: refund(ticket-7:refund) again
  R-->>W: existing receipt; no second effect
  W->>J: phase_done
```

Change the receiver's idempotency handling in a scratch copy and repeat. Explain
why a durable coordinator alone does not buy exactly-once effects. For a remote
payment API, the receiver must honor the key; if it cannot, you need reconciliation
or a human decision for uncertain outcomes.

**Prove:** terminate immediately after approval or denial commits; restart and verify
the correct phase. Also interrupt between the simulated effect and completion.
`tests/test_durable.py` and the experiment cover different crash windows.

---

## Quizzes

<div class="aieng-quiz" data-quiz-id="25-q1" data-xp="25" data-success="HITL is a persisted pause, not a blocking print." data-fail="Re-read coordinator states." markdown>
<p class="label">Quiz · +25 XP</p>
<p class="quiz-prompt">What should happen when a phase returns ask_human?</p>
<div class="quiz-options">
<button type="button" class="quiz-opt" data-correct="false">The worker busy-loops until stdin has a line</button>
<button type="button" class="quiz-opt" data-correct="true">The coordinator persists a HITL event and returns paused so a UI can resume later</button>
<button type="button" class="quiz-opt" data-correct="false">The merge gate auto-approves to save time</button>
<button type="button" class="quiz-opt" data-correct="false">The hypothesis tree is deleted</button>
</div>
<p class="quiz-feedback"></p>
</div>

<div class="aieng-quiz" data-quiz-id="25-q2" data-xp="25" data-success="Merge gates require tests and approval; worktrees isolate writes." data-fail="Think CI + CODEOWNERS." markdown>
<p class="label">Quiz · +25 XP</p>
<p class="quiz-prompt">A worktree patch is ready. What does MergeGate require before allow=True?</p>
<div class="quiz-options">
<button type="button" class="quiz-opt" data-correct="false">A longer system prompt</button>
<button type="button" class="quiz-opt" data-correct="true">Tests passed, human approval, and a bounded diff</button>
<button type="button" class="quiz-opt" data-correct="false">At least five hypotheses</button>
<button type="button" class="quiz-opt" data-correct="false">The original files already overwritten</button>
</div>
<p class="quiz-feedback"></p>
</div>

---

## Open source materials

| Resource | Use it for |
|----------|------------|
| `src/durable.py`, `src/sandbox.py` | Coordinator, tree, gate, worktree |
| LangGraph checkpoints / HITL | Industry analog after you can name the states |
| 12-factor agents | Owned control flow |
| Module 22 / 26 | Eval and $ attribution on the graph |

---

## Checkpoint

- [ ] Restart does not lose phase  
- [ ] Hypotheses are data with scores, not a paragraph  
- [ ] Writes happen off the user’s original tree until a gate  
- [ ] HITL is a state in the log  
- [ ] You can say what the run cost and how long it took  

<div class="aieng-complete" data-module-id="25" data-xp="130" markdown>
<p>Mark complete when you can pause a coordinator, resume it from JSONL, and refuse a red merge.</p>
<button type="button">Complete module · +130 XP</button>
</div>

## Exercise

- **Catalog:** [EX-25 — Durable graph](../reference/exercises.md#ex-25)
- **Prove:** Pause/resume from JSONL; a denied HITL does not run the next phase; merge gate blocks failed tests.
- **Test:** `pytest tests/test_durable.py -v`

**Return to the case:** An append-only log, isolated worktree, tests, and a human merge state let the investigation survive interruption without writing directly to the user tree. Durability preserves state; it does not validate the hypothesis by itself.

**Next:** [Orchestrators in production](26-orchestrator-comparison.md)
