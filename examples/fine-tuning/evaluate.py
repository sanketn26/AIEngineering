"""Generate on held-out rows with the base model or an adapter, then score.

    python examples/fine-tuning/evaluate.py --split data/processed/test.jsonl \
        --output results/fine-tuning/base.json
    python examples/fine-tuning/evaluate.py --split data/processed/test.jsonl \
        --adapter artifacts/adapters/v1 --output results/fine-tuning/adapter.json

Automatic checks only (plain-text compliance, unsupported numbers, latency).
Usefulness still needs a human rubric on the same rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.finetune_data import scorecard, unsupported_numbers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    rows = [json.loads(line) for line in args.split.read_text().splitlines() if line]
    results = []
    for row in rows:
        prompt_messages = row["messages"][:2]
        source = "\n".join(m["content"] for m in prompt_messages)
        inputs = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        latency = time.perf_counter() - start
        text = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        results.append(
            {
                "source": source,
                "generated": text,
                "reference": row["messages"][2]["content"],
                "unsupported_numbers": sorted(unsupported_numbers(source, text)),
                "latency_s": round(latency, 3),
                "new_tokens": int(out.shape[1] - inputs.shape[1]),
            }
        )

    card = scorecard((r["source"], r["generated"]) for r in results)
    card["avg_latency_s"] = sum(r["latency_s"] for r in results) / max(len(results), 1)
    report = {
        "base_model": args.base_model,
        "adapter": str(args.adapter) if args.adapter else None,
        "split": str(args.split),
        "scorecard": card,
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
