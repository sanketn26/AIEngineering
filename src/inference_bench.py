"""Optional real-model inference experiments. Run with python -m src.inference_bench.

Heavy dependencies are imported only when a model experiment starts.
--smoke uses randomly initialized tiny Llama models, with no downloads.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

PROMPTS = [
    "Ticket: I cannot sign in after changing my password. Support reply:",
    "Ticket: My parcel arrived with a broken handle. Support reply:",
    "Ticket: I was charged twice for the same order. Support reply:",
]


def percentile(values, fraction):
    """Linear interpolation; never invent a value for an empty population."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(rows):
    """Generation-only timings, including prompt processing but not tokenization."""
    elapsed = [r["elapsed_ms"] for r in rows]
    ttft = [r["local_ttft_ms"] for r in rows if r.get("local_ttft_ms") is not None]
    tpot = [r["local_tpot_ms"] for r in rows if r.get("local_tpot_ms") is not None]
    seconds = sum(elapsed) / 1000
    # null distinguishes "not measured" from a measured zero; never collapse the two.
    measured_peaks = [
        r["cuda_peak_allocated_bytes"]
        for r in rows
        if r["cuda_peak_allocated_bytes"] is not None
    ]
    return {
        "requests": len(rows),
        "median_ms": statistics.median(elapsed) if elapsed else None,
        "p95_ms": percentile(elapsed, 0.95),
        "median_local_ttft_ms": statistics.median(ttft) if ttft else None,
        "p95_local_ttft_ms": percentile(ttft, 0.95),
        "median_local_tpot_ms": statistics.median(tpot) if tpot else None,
        "tpot_sample_count": len(tpot),
        "output_tokens_per_second": (
            sum(r["output_tokens"] for r in rows) / seconds if seconds else None
        ),
        "cuda_peak_allocated_bytes": max(measured_peaks) if measured_peaks else None,
    }


def paired_comparison(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row["repeat"], row["prompt_index"]), {})[row["variant"]] = (
            row["output_ids"]
        )
    complete = [p for p in pairs.values() if set(p) == {"baseline", "candidate"}]
    return {
        "paired_requests": len(complete),
        "identical_greedy_outputs": sum(
            p["baseline"] == p["candidate"] for p in complete
        ),
        "meaning": "Token parity is a regression signal, not semantic quality.",
    }


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def run(args):
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        LogitsProcessor,
    )

    torch.manual_seed(7)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = None
    assistant = None
    if args.smoke:
        if args.new_tokens + args.prompt_repeats * 4 > 256:
            raise ValueError("Smoke input plus output must fit its 256-token window")
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
        inputs = [
            {
                "input_ids": torch.tensor(
                    [[1, 4 + i, 9, 12] * args.prompt_repeats], device=device
                )
            }
            for i in range(2)
        ]
        if args.experiment == "speculative":
            draft_config = LlamaConfig(**config.to_dict())
            draft_config.num_hidden_layers = 1
            assistant = (
                LlamaForCausalLM(draft_config).to(device=device, dtype=dtype).eval()
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=False
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )
        prompts = PROMPTS
        if args.prompts:
            prompts = json.loads(args.prompts.read_text())
            if (
                not isinstance(prompts, list)
                or not prompts
                or not all(isinstance(p, str) and p.strip() for p in prompts)
            ):
                raise ValueError("--prompts must contain a nonempty JSON list of text")
        prefix = "Support policy: ask for missing details; do not invent facts.\n"
        inputs = [
            tokenizer(prefix * args.prompt_repeats + prompt, return_tensors="pt").to(
                device
            )
            for prompt in prompts
        ]
        if args.experiment == "speculative":
            if not args.assistant:
                raise ValueError("speculative requires --assistant (or --smoke)")
            draft_tokenizer = AutoTokenizer.from_pretrained(
                args.assistant,
                revision=args.assistant_revision,
                trust_remote_code=False,
            )
            if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
                raise ValueError(
                    "This lab requires identical target/draft vocabularies"
                )
            if tokenizer.all_special_ids != draft_tokenizer.all_special_ids:
                raise ValueError("Target/draft special token IDs must match")
            assistant = (
                AutoModelForCausalLM.from_pretrained(
                    args.assistant,
                    revision=args.assistant_revision,
                    dtype=dtype,
                    attn_implementation="eager",
                    trust_remote_code=False,
                )
                .to(device)
                .eval()
            )

    def context_limit(config, role):
        """Absent on some architectures; report that rather than crashing on lookup."""
        limit = getattr(config, "max_position_embeddings", None)
        if limit is None:
            print(f"Warning: {role} declares no max_position_embeddings; not checked.")
        return limit

    target_limit = context_limit(model.config, "target")
    draft_limit = context_limit(assistant.config, "draft") if assistant else None
    for item in inputs:
        item["attention_mask"] = torch.ones_like(item["input_ids"])
        length = item["input_ids"].shape[-1] + args.new_tokens
        if target_limit is not None and length > target_limit:
            raise ValueError("Input plus requested output exceeds model context")
        if draft_limit is not None and length > draft_limit:
            raise ValueError("Input plus requested output exceeds draft context")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.config.eos_token_id

    class FirstTokenClock(LogitsProcessor):
        """Timestamp the first logits call: prefill is done, token one is available.

        This is a local prefill/decode split, not a served TTFT: it excludes
        queueing, network, and tokenization, which dominate a real deployment.
        """

        def __init__(self):
            self.first_call_s = None

        def __call__(self, input_ids, scores):
            if self.first_call_s is None:
                # Without this the CUDA timestamp records kernel launch, not result.
                sync()
                self.first_call_s = time.perf_counter()
            return scores

    def generate(item, candidate):
        backend = (
            args.backend if candidate and args.experiment == "attention" else "eager"
        )
        model.set_attn_implementation(backend)
        use_cache = candidate if args.experiment == "cache" else True
        draft = assistant if candidate and args.experiment == "speculative" else None
        if draft:
            # Hold the proposal policy fixed instead of adapting it between trials.
            draft.generation_config.num_assistant_tokens = 3
            draft.generation_config.num_assistant_tokens_schedule = "constant"
            draft.generation_config.assistant_confidence_threshold = 0.0
        clock = FirstTokenClock()
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **item,
                do_sample=False,
                max_new_tokens=args.new_tokens,
                use_cache=use_cache,
                assistant_model=draft,
                pad_token_id=pad_token_id,
                logits_processor=[clock],
            )
        sync()
        elapsed = (time.perf_counter() - start) * 1000
        ids = output[0, item["input_ids"].shape[-1] :].tolist()
        # Assisted generation calls logits processors during drafting and
        # verification, so the first call need not correspond to a committed
        # output token. Both metrics derive from that timestamp, so both are
        # left unset here rather than reported as a misleading speedup.
        ttft_ms = None
        if args.experiment != "speculative" and clock.first_call_s is not None:
            ttft_ms = (clock.first_call_s - start) * 1000
        # Undefined below two tokens: one token has no inter-token interval.
        tpot_ms = (
            (elapsed - ttft_ms) / (len(ids) - 1)
            if ttft_ms is not None and len(ids) >= 2
            else None
        )
        return {
            "elapsed_ms": elapsed,
            "local_ttft_ms": ttft_ms,
            "local_tpot_ms": tpot_ms,
            "output_tokens": len(ids),
            "input_tokens": item["input_ids"].shape[-1],
            "input_sha256": hashlib.sha256(
                json.dumps(item["input_ids"].tolist()).encode()
            ).hexdigest(),
            "output_ids": ids,
            "output_text": tokenizer.decode(ids) if tokenizer else None,
            "attention_implementation": model.config._attn_implementation,
            "use_cache": use_cache,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        }

    # Warm every workload/variant; model loading and these calls are excluded.
    for item in inputs:
        generate(item, False)
        generate(item, True)
    rows = []
    for repeat in range(args.repeats):
        for i, item in enumerate(inputs):
            for candidate in [False, True] if repeat % 2 == 0 else [True, False]:
                row = generate(item, candidate)
                row.update(
                    repeat=repeat,
                    prompt_index=i,
                    variant="candidate" if candidate else "baseline",
                )
                rows.append(row)
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    metadata = {
        "experiment": args.experiment,
        "smoke_random_weights": args.smoke,
        "model": "random-tiny-llama" if args.smoke else args.model,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "requested_revision": args.revision,
        "assistant": args.assistant if assistant and not args.smoke else None,
        "assistant_resolved_revision": (
            getattr(assistant.config, "_commit_hash", None) if assistant else None
        ),
        "assistant_resident_in_both_variants": assistant is not None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "dtype": args.dtype,
        "platform": platform.platform(),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.machine()
        ),
        "threads": args.threads,
        "repeats": args.repeats,
        "new_token_cap": args.new_tokens,
        "prompt_repeats": args.prompt_repeats,
        "attention_candidate": args.backend if args.experiment == "attention" else None,
        "query_heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
        "logical_kv_bytes_per_token": 2
        * model.config.num_hidden_layers
        * model.config.num_key_value_heads
        * head_dim
        * next(model.parameters()).element_size(),
    }
    return {
        "metadata": metadata,
        "rows": rows,
        "summary": {
            name: summarize([r for r in rows if r["variant"] == name])
            for name in ("baseline", "candidate")
        },
        "parity": paired_comparison(rows),
        "limitations": [
            "Local TTFT splits prefill from decode in-process; it excludes queueing, "
            "network and tokenization, so it is not a served TTFT.",
            "TPOT is derived from wall time after first logits, not per-token stamps.",
            "Speculative runs report no local TTFT/TPOT: the first logits call can "
            "precede any committed token, so both would misstate user-visible timing.",
            "SDPA is dispatch, not proof that a FlashAttention kernel ran.",
            "CPU/MPS peak memory is not measured; null is not zero.",
            "Review output quality on your held-out tasks before adopting a change.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=["cache", "attention", "speculative"], default="cache"
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--assistant")
    parser.add_argument("--assistant-revision", default="main")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--backend", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    parser.add_argument("--new-tokens", type=positive_int, default=32)
    parser.add_argument("--prompt-repeats", type=positive_int, default=8)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke and args.prompts:
        parser.error("--prompts needs a real tokenizer; drop --smoke to use it")
    if args.backend == "flash_attention_2" and args.experiment == "attention":
        if args.device != "cuda" or args.dtype == "float32":
            parser.error(
                "FlashAttention 2 requires compatible CUDA hardware and fp16/bf16"
            )
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"summary": report["summary"], "parity": report["parity"]}, indent=2)
    )


if __name__ == "__main__":
    main()
