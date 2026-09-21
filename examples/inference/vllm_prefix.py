"""GPU lab: run in separate fresh processes with --cache on and off.

This measures generation wall time, not TTFT. vLLM may write logs to stdout;
--output is the machine-readable report. No vLLM dependency in core course.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--revision", required=True, help="Model commit SHA for repeatability"
    )
    parser.add_argument("--cache", choices=["on", "off"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    # Paged KV allocation is owned by this engine, not implemented in the client.
    engine = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        trust_remote_code=False,
        enable_prefix_caching=args.cache == "on",
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
        seed=7,
    )
    params = SamplingParams(temperature=0, max_tokens=16)
    # Warm execution without populating the measured policy prefix.
    engine.generate(
        ["A completely unrelated warmup about mountains."], params, use_tqdm=False
    )
    policy = "Support policy: verify the order and ask for missing details.\n" * 48
    requests = [policy + f"Ticket {i}: My order is late. Reply:" for i in range(6)]
    rows = []
    for i, prompt in enumerate(requests):
        start = time.perf_counter()
        result = engine.generate([prompt], params, use_tqdm=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        completion = result.outputs[0]
        rows.append(
            {
                "request": i,
                "phase": "first_policy_prefix" if i == 0 else "repeated_policy_prefix",
                "elapsed_ms": elapsed_ms,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_tokens": len(result.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "output_ids": list(completion.token_ids),
                "text": completion.text,
                "finish_reason": completion.finish_reason,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": args.revision,
                "vllm": importlib.metadata.version("vllm"),
                "prefix_cache": args.cache,
                "rows": rows,
                "limits": {
                    "max_model_len": 2048,
                    "max_num_seqs": 4,
                    "gpu_memory_utilization": 0.7,
                    "max_tokens": 16,
                },
                "note": "First prefix is cold, not the runtime. Later requests may "
                "reuse blocks. Inspect cache-hit counters; timing cannot prove a hit.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
