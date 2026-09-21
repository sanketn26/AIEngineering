# The refund that survived a restart

A ticket says: “The package arrived, but I was billed twice.” The classifier sees
shipping nouns. The customer wants their money back. Even when the classifier gets
that right, who gave it permission to issue a refund?

This is the instructor reference for the course's five gates. Follow the request
through a small completed service, then interrupt it at the least convenient moment.
The original `capstone-starter/` keeps its planted failures for learners to solve.

## Run from the repository root

```bash
python3.11 -m venv .venv-reference
source .venv-reference/bin/activate
pip install -r examples/production_triage/requirements.txt
python -m pytest examples/production_triage/tests -q
python -m examples.production_triage.evaluate
python -m examples.production_triage.rehearse  # temporary local HTTP + restart + load
```

The default provider is a deterministic mock; no model credentials are required.
The release eval has 20 synthetic regression cases, separate from the four development
cases. This is a contract check, not evidence of real-world classification accuracy.
Once inspected or used to tune a change, these rows are a regression set; collect a
fresh sealed set for the next real release.

Generate one local admin credential in your shell; do not paste it into source:

```bash
export TRIAGE_BEARER="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export TRIAGE_TOKENS_JSON="$(python -c 'import json,os; print(json.dumps({os.environ["TRIAGE_BEARER"]:{"id":"local-admin","role":"admin","scopes":["refund:write"]}}))')"
export TRIAGE_DB=/tmp/triage-reference.sqlite
uvicorn examples.production_triage.service:create_app --factory
```

In another terminal with those environment variables:

```bash
curl http://127.0.0.1:8000/v1/triage \
  -H "Authorization: Bearer $TRIAGE_BEARER" -H 'Content-Type: application/json' \
  -d '{"ticket_id":"ticket-7","text":"The package arrived but I was billed twice"}'
```

You receive billing/high, `refund-v1` evidence, and a **pending** proposal. There is
no ledger entry yet. Submit the returned proposal ID to the approval route:

```bash
curl http://127.0.0.1:8000/v1/proposals/REPLACE_WITH_PROPOSAL_ID/decision \
  -H "Authorization: Bearer $TRIAGE_BEARER" -H 'Content-Type: application/json' \
  -d '{"approve":true}'
```

Repeat that request. The ledger still contains one entry. Stop the server, restart
with the same database, and repeat it again. One entry. Try to reverse an already
final decision: `409`. These outcomes follow from transaction and uniqueness rules,
not the model remembering what it did.

## One request, five gates

```text
Bearer token → trusted principal → per-principal rate limit
  → prepare/redact ticket → choose small/large model
  → reserve cost → deadline + bounded transient retries → validate schema
  → category policy lookup, excluding expired records
  → copy evidence and citation IDs → permission-gated proposal
  → request-ID spans + version/digest + cost/latency event

Separate human decision → scope + owner checks
  → SQLite transaction: final decision AND unique simulated ledger entry
```

The model cannot call tools, supply an actor, select a ledger amount, or approve its
own proposal. A request body containing extra identity fields fails validation.
The policy lookup supports the explanation; it does not feed the classifier. That
makes the “do we need retrieval for classification?” baseline explicit.

| Gate | Evidence to inspect |
|---|---|
| Contract | Timeout → 504; unavailable provider → 503; invalid model output → 502 |
| Quality | `evaluate` exits nonzero on a changed expected category, priority, or citation set |
| Knowledge | Missing/expired policy → `not_in_policy`; no invented policy or pending refund |
| Actions | Viewer and wrong-owner decisions fail; repeated approval creates one ledger entry |
| Operation | `/ops/metrics`, redacted request spans, release digest, bounded load script |

## Put a real model behind the same boundary

Copy `data/release-v1.json` into an untracked runtime directory. Set `provider` to
`http`, set `small_model` and `large_model` to actual installed/provider model IDs,
and enter the input/output rates for those models. This compact reference uses one
rate pair: use the higher rates of the two models for conservative accounting, or
extend it to a per-model price table. The rates are configuration, not live pricing.

```bash
export TRIAGE_RELEASE=/absolute/path/to/your/release.json
export TRIAGE_PROVIDER_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions
# Hosted endpoint: HTTPS and TRIAGE_PROVIDER_KEY supplied through your secret manager.
python -m examples.production_triage.evaluate --live
```

The adapter requires a chat-completion endpoint returning `choices[0].message.content`
and usage counts. The content must be JSON with exactly `category` and `priority`.
A local runtime must have its model pulled separately. The same eval gate applies;
a provider that cannot satisfy the contract fails visibly. Run the uncertainty lab
with repeated, freshly labeled cases before making a quality claim.

Every attempt reserves a conservative token budget. A failed request may still be
billed, so it retains its reserved cost estimate. Successful calls reconcile against
usage. The UTF-8-byte estimate plus wrapper allowance is a teaching approximation,
not a universal tokenizer bound: use provider tokenization and billing reconciliation
before treating this as a hard dollar guarantee. Events mark estimated cost explicitly. Mock calls cost zero; they do not
prove savings in dollars. The event's model ID shows whether routing changed.

## Rehearse a release and rollback

1. Start v1. Save the authenticated `/readyz` response's digest alongside the code SHA.
2. Copy the release file to v2; change the model or routing configuration. Run the
   release eval and the bounded load probe before promotion.
3. Set `TRIAGE_EXPECTED_DIGEST` to the reviewed digest when starting that release.
   A mismatched bundle fails startup. The digest includes prompt, model settings,
   and the policy fixture; pair it with the image/code SHA because code is separate.
4. Revert to the earlier image/SHA, release file, and expected digest, restart, then
   check `/readyz` and the eval. Do not roll back the database blindly.

```bash
python -m examples.production_triage.load --requests 30 --concurrency 4
curl http://127.0.0.1:8000/ops/metrics -H "Authorization: Bearer $TRIAGE_BEARER"
```

The automated `rehearse` command also starts a deliberately over-tight rate limit in a temporary deployment, observes 429, then restores v1 and its digest and checks for 200. It preserves the ledger across those restarts. This is a configuration rollback rehearsal; a real image rollback also needs the recorded code SHA.

The probe reports failures as well as latency. Thirty requests are a rehearsal,
not a tail-latency capacity study. Raise the workload only after deciding the SLO,
budget, and stop condition. Model spans share the request ID with retrieval and
approval spans; this is a local JSON flight recorder, not an OpenTelemetry collector.

## Container

```bash
docker build -f examples/production_triage/Dockerfile -t triage-reference .
docker run --rm -p 127.0.0.1:8000:8000 --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp \
  -v triage-data:/data -e TRIAGE_TOKENS_JSON triage-reference
```

Use an image digest for a real deployment. The SQLite volume survives the container;
without persistent storage, a restart has nothing to remember.

## What was actually run

See [execution evidence](verification/README.md) for test results, the measured local HTTP report, and optional paths that were not executed.

## What this small deployment does not establish

- Refunds are simulated ledger entries, never payment transfers. Real payments need
  a receiver-enforced idempotency key and reconciliation/outbox design.
- Tokens map to trusted principals in server configuration. Replace this local
  identity setup with your identity provider, rotation, and tenant authorization.
- Proposal ownership uses the authenticated principal; it is not a customer-account
  entitlement system. A real order lookup must check ownership before returning data.
- SQLite supports this single-host reference. The rate counter is shared by workers
  using the same file, but this is not a distributed rate limiter. Logs and counters
  need retention, access controls, and bounded storage before internet exposure.
- The five-gate route is bounded to two model calls and one proposal; there is no
  open-ended agent loop. This completes the five-gate teaching service, not every
  optional agent feature in the broader capstone specification.
- Async timeouts cancel this async HTTP adapter. A blocking library or subprocess
  needs its own cancellation/isolation design.
