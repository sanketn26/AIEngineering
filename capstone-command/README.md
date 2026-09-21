# Capstone — specification-gated coding runtime

A **deliberately incomplete** runtime for a small code model. It runs. It is not done.

A 4B model may propose a patch. This process decides whether that patch is legal. The build spec is [docs/core/capstone-command.md](../docs/core/capstone-command.md). Close the holes in order: [docs/core/capstone-command-gates.md](../docs/core/capstone-command-gates.md). Your checklist is [PROGRESS.md](PROGRESS.md).

No API key. The model is a mock diff. The prompts you will send a real 4B model are in [`prompts/`](prompts/). The target contract, the five-stage measurement, and the executor are in the build spec.

## Why it is incomplete on purpose

| Hole | Where | Gate that closes it |
|---|---|---|
| A sentence is executed as `code.patch` | `runtime/execute.py` `execute` | 1 Frozen specification |
| `unknowns` still validate; `spec.draft` still returns a diff | `runtime/models.py`, `runtime/execute.py` | 1 |
| `DRAFT` may transition to `GENERATING` | `runtime/state.py` | 1 |
| A `must` requirement with no criterion still validates | `runtime/models.py` `graph_is_well_formed` | 2 Evidence |
| Verification is a shell string | `runtime/policy.py` `verification_plan` | 4 Runtime authority |
| The mock diff touches `setup.py` and policy returns PASS | `runtime/policy.py` | 4 |
| Repair cap is 99; repeated patches are invisible | `runtime/policy.py` | 4 |
| `risk_level: low` makes any spec eligible | `runtime/capability.py` | 4 |
| Freeze stores no revision hash | `runtime/policy.py` `freeze` | 5 Revision |

Do not close a hole by telling the mock to be careful. The mock is the untrusted writer.

## Run

```bash
cd capstone-command
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cmdai.py spec check fixtures/retry_api_client.yaml
python cmdai.py run --intent "Add retry support to the API client."
pytest tests/ -v
```

`spec check` on the sample prints `SPEC_READY`. `run --intent` prints `candidate`. Gate 1 is done when that sentence cannot produce a patch.

## What already holds

Path traversal, absolute paths, unknown fields, duplicate ids, and acceptance criteria that cite a missing requirement are rejected. Those checks stay. The gates add the decisions the starter still skips.
