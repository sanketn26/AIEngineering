# Execution evidence — 2026-09-21

These are actual CPU runs, not expected results or performance promises.

- Target: `HuggingFaceTB/SmolLM2-135M`, revision recorded in each JSON.
- Environment: macOS ARM64, PyTorch 2.14.0, Transformers 4.57.1, float32, two threads.
- Three prompts, one measured repeat, output cap eight tokens, prompt repetition two.
- Both experiments matched baseline/candidate greedy token IDs on all three prompts.
- The cache run shows the prefill/decode split the lesson predicts: local TTFT is flat
  (43.8 → 44.6 ms, same prompt processing) while median local TPOT halves
  (45.6 → 22.9 ms). Three single runs on a laptop; directionally illustrative only.
- This sample is too small for production tail-latency or quality conclusions.
- The model is a small base model; these outputs are not a support-quality evaluation.
- FlashAttention/CUDA and vLLM were not executed on this machine.

Reproduction from the repository root after installing the optional requirements:

```bash
python -m src.inference_bench --experiment cache \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --repeats 1 --new-tokens 8 --prompt-repeats 2 \
  --output results/inference/cpu-cache.json
python -m src.inference_bench --experiment attention \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --repeats 1 --new-tokens 8 --prompt-repeats 2 \
  --output results/inference/cpu-attention.json
```

Offline tiny-model tests additionally exercised speculative generation and paired
metric aggregation: `7 passed`. The core suite (without optional ML dependencies)
reported `83 passed, 4 skipped`; three skips were the optional generation tests.
