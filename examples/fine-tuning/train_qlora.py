"""QLoRA SFT: 4-bit frozen base + LoRA adapter. Needs a CUDA GPU.

    accelerate launch examples/fine-tuning/train_qlora.py \
        --data data/processed --output artifacts/adapters/v1
    # Colab: point --data/--output at /content/drive/MyDrive/... and pass --resume
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import peft
import torch
import transformers
import trl
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapters/v1"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("4-bit QLoRA needs CUDA; use Colab or a GPU host.")
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(json.dumps(environment(), indent=2))

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data / "train.jsonl"),
            "validation": str(args.data / "validation.jsonl"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_length=2048,
        packing=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
