# Capstone starter — support-ticket triage

A **deliberately incomplete** production-AI service. It runs. It is not done.

Domain is **support-ticket triage** so it stays track-agnostic: no markets, no editor, no training loop. The model is a **mock keyword heuristic** — no API keys, no paid providers.

Start here. Close the five gates in [docs/core/capstone-gates.md](../docs/core/capstone-gates.md). The build spec (what "done" means) is [docs/core/capstone.md](../docs/core/capstone.md). Your checklist lives in [PROGRESS.md](PROGRESS.md).

## Why it is incomplete on purpose

| Hole | Where | Gate that closes it |
|------|--------|---------------------|
| No deadline / retry on the model call | `model.py` `_call_provider` | 1 Dependable service |
| One golden row the heuristic mis-labels | `evals/golden.jsonl` `planted-mixed-ticket` | 2 Measurable quality |
| Retrieval returns `[]` | `model.retrieve` | 3 External knowledge |
| `refund_customer` is not role/scope gated | `authorization.py` vs `tools.py` | 4 Authorized actions |
| Every classify uses the "large" mock model | `model.classify` | 5 Operate and optimize |

Do not paper over a hole with a prompt instruction. The model is the untrusted dependency.

## Run

```bash
cd capstone-starter
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

- `GET /healthz` → `{"status":"ok"}`
- `POST /v1/triage` `{"text":"I forgot my password"}` → schema-valid category/priority

```bash
curl -s localhost:8000/healthz
curl -s localhost:8000/v1/triage -H 'content-type: application/json' \
  -d '{"text":"Please refund last month invoice"}'
```

Docker: `docker build -t triage-starter . && docker run -p 8000:8000 triage-starter`

## Tests

```bash
pytest tests/test_api.py -v     # must pass
pytest tests/test_eval.py -v    # suite runs; planted miss is *reported*
```

`test_eval` does **not** assert 100% accuracy. That would fail this repo's CI while the starter is still honest. Gate 2 is to make `planted-mixed-ticket` pass, then tighten the assertion.

## Architecture (as the code is today)

```text
POST /v1/triage
  → classify()          # mock-large, no timeout
  → retrieve()          # always empty
  → propose_actions()   # refund_customer proposed, never executed
  → TriageResponse      # schema-valid JSON
```

Authorization is a function the write tool *calls*, and that function currently returns True. Retrieval is a function the request path *calls*, and it currently returns nothing. Those are the right seams — fill them in rather than adding a second stack.

## Five gates (operational checkpoints)

Full entry/build/eval/failure/exit/artifact for each gate: [capstone-gates.md](../docs/core/capstone-gates.md). Short version:

1. **Dependable service** — deadline, fail-closed schema, mapped errors.
2. **Measurable quality** — golden set in CI; planted case passes without wrecking the others.
3. **External knowledge** — policy KB + citations; empty retrieval degrades, it does not invent policy.
4. **Authorized actions** — `refund_customer` denied for viewer/missing actor; writes still not executed here unless you add a dry-run ledger.
5. **Operate and optimize** — `request_id` traces, route trivial classify to `mock-small`, cost/latency you can quote.

Work the gates in order. A beautiful agent loop on top of a hanging model call is costume jewelry.
