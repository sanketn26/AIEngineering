# Execution evidence — 2026-09-21

These are observed local results from the working tree, not expected output.
Environment: macOS ARM64, Python 3.11.16; reference dependencies are pinned in
`../requirements.txt`. No external model, payment API, or real customer data was used.

| Check | Observed result |
|---|---|
| Core package | 106 passed, 4 optional-dependency tests skipped |
| Completed reference service | 17 passed |
| Original capstone starter | 24 passed |
| Three original track starters | 6 passed |
| Strict MkDocs build | Passed |
| Synthetic release eval | 20/20; a deliberately regressed provider fails the gate |
| Crash after receiver effect, before journal completion | Worker retried; one receiver effect |
| Local HTTP restart + repeated approval | One simulated ledger entry |
| Configuration rollback | Bad limit returned 429; restored v1 returned 200 |
| HTTP load rehearsal | 30 requests, concurrency 4, 30 successes, zero errors |

The raw HTTP report is [local-mock.json](local-mock.json). Its client p95 was about
20 ms on this run. Thirty calls to a mock do not establish production capacity,
provider latency, model accuracy, or dependable p99 behavior. Re-run on your own
workload; keep failed requests in the accounting.

The evaluation fixtures returned inconclusive for 80% → 85% (paired interval
−10 to +20 percentage points), inconclusive for two cases repeated 100 times,
and regression for the deliberately large quality drop. These are constructed
fixtures and deterministic bootstrap calculations, not empirical model results.

The controlled retrieval comparison ran all four methods over 12 queries.
Its annotated fixture reranker reached source support 1.0 but question relevance
10/12: copying true evidence still cannot answer a question absent from the corpus.
Those fixture numbers are not a learned-model benchmark.

## Not executed locally

- Docker isolation probe and reference image build: no running Docker daemon.
  CI jobs are added, but their remote results are not claimed here.
- Optional real embedding/cross-encoder downloads and inference.
- Live model-provider classification. The HTTP adapter's contract, redirect refusal,
  and body bound were tested with a controlled transport instead.

## Reproduce

From the repository root in the appropriate environments:

```bash
python -m pytest tests -q
python -m pytest examples/production_triage/tests -q
python -m examples.production_triage.evaluate
python -m examples.production_triage.rehearse
python -m examples.durability.crash
python -m examples.evaluation.compare
python -m examples.retrieval.compare
python -m mkdocs build --strict
```

Run the starter and track tests in their documented environments as separate
invocations. The reference tests require its requirements file; the core tests
do not import its FastAPI dependencies. One dependency deprecation warning appeared
in the FastAPI TestClient suites; it did not fail the tests.
