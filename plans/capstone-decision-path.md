# Plan: Capstone Gate 6 (stretch) — fixed-answer scoring

**Status:** implemented on branch `additions`, 2026-09-21. Mock path tested in CI. Real-model calibration and benchmark run on Apple M-series (Qwen2.5 0.5B and 1.5B, Transformers MPS and MLX); results in `capstone-starter/decision/README.md`.
**Source idea:** Avi Chawla's post on reproducing "Jev-style" decisions (x.com/_avichawla/status/2101563610644496464). We teach the inference pattern, not the product.

## Why

The capstone is a support-ticket triage service whose category is a bounded decision. Scoring the declared choices in one forward pass yields a choice plus a distribution, which feeds the Gate 5 "stop paying for the large model on everything" hole with a principled router.

## Decisions

- **No SGLang.** The article serves through SGLang `/v1/score`; the course has not covered SGLang. We use Hugging Face Transformers in process (already used by `examples/inference/` and `examples/fine-tuning/`) and read `logits[0, -1]` directly, which shows the mechanism instead of hiding it behind an endpoint. Serving engines are mentioned as a batching choice only.
- **Separate package** `capstone-starter/decision/`. The five planted holes in `model.py` stay untouched.
- **Mock by default**, real model opt-in via `requirements-decision.txt`. Starter CI stays stdlib + FastAPI.
- **Escape choice `other`** is mandatory; tested against injection and security rows.
- **Label check in context**: encode prompt vs prompt+label, require exactly one extra token with an unchanged prefix. Stricter than tokenizing the bare label.
- **Calibration is the teaching payload** (ECE, reliability table, threshold pick). The article explicitly leaves it out.
- **Benchmark is sequential in one process.** No server concurrency claims; learners quote their own numbers.

## Open

- Run on CPU-only and a T4 to see whether the speed-up ratio holds off Apple silicon.
- Headline finding to teach: at 1.5B, single-pass scoring loses to generation on accuracy (0.72 vs 0.84) because of letter bias; rotations recover it (0.88) at most of the latency cost.
- Decide whether `app.py` should expose the scorer behind a flag or leave integration to the learner (currently left to the learner).
- A 25-row set is enough to teach ECE, too small to set production thresholds; say so or grow it.
