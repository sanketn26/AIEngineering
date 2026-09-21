# Inference experiments

Run from the repository root. The full walkthrough, interpretation, GPU commands,
and limitations are in [the Module 28 hands-on](../../docs/core/inference/hands-on.md).

```bash
python3.11 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install -r examples/inference/requirements.txt
python -m src.inference_bench --smoke --output results/inference/smoke.json
python -m src.inference_bench --experiment cache --output results/inference/cache.json
python -m src.inference_bench --experiment attention --output results/inference/attention.json
```

- `src/inference_bench.py`: paired local generation benchmarks; CPU default.
- `vllm_prefix.py`: optional GPU prefix-caching experiment, one fresh process per setting.
- Requirements are optional; core course imports remain dependency-free.
- Real weights download only without `--smoke`. Smoke models have random weights.
- Reports contain generated text: use the supplied fictional tickets or non-sensitive fixtures.
- No speedup is assumed. Compare measured timing, token lengths, parity, and task quality.

Actual small CPU runs and their limitations are in [verification/](verification/README.md).
