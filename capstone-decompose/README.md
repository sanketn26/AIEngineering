# Capstone — divide, solve, and join

A **deliberately incomplete** loop for a task a 20B model fails as one prompt. The same model can finish it by dividing the work, solving each part, and joining the parts. Your checks, not the model's "done", decide the join.

Build spec: [docs/core/capstone-decompose.md](../docs/core/capstone-decompose.md). Gates: [docs/core/capstone-decompose-gates.md](../docs/core/capstone-decompose-gates.md). Checklist: [PROGRESS.md](PROGRESS.md).

No API key. The one-shot "model" is a stub that claims success.

## Why it is incomplete on purpose

| Hole | Where | Gate |
|---|---|---|
| The whole task is one 20B prompt, and `done` is the result | `runtime/execute.py` `run_task` | 1 |
| The mock plan drops R3, cites `V-looks-right`, imports `record_refund` (never exported), and is accepted | `MOCK_PLAN`, `accept_plan` | 1 and 2 |
| A dependency cycle is accepted | `accept_plan` | 1 |
| A part that is still too wide stays on 20B | `route_part` | 3 |
| Join returns `done` without seam checks or V4 | `join_parts` | 4 |

## Run

```bash
cd capstone-decompose
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python divide.py check fixtures/refund_task.yaml
python divide.py run fixtures/refund_task.yaml
python divide.py plan fixtures/refund_task.yaml
pytest tests/ -v
```

`check` prints `SPEC_READY`. `run` prints `one_shot` and `done`. `plan` accepts the mock division and joins it with no checks. Those three lines are the holes.

Prompts for the real calls, once the holes are closed: [prompts/divide.txt](prompts/divide.txt), [prompts/solve-part.txt](prompts/solve-part.txt), [prompts/join.txt](prompts/join.txt).

## What already holds

Unsafe paths are rejected. A part file outside the task allowlist is rejected. The user spec's check ids and the `part` / `join` scope are real fields. The gates add the decisions the starter skips: coverage, seams, one part at a time, and a join that runs V4.
