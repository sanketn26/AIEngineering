# Fine-tuning lab (QLoRA)

Run from the repository root. The full walkthrough — data rights, cleaning,
company-level splits, Colab, evaluation, serving, and retraining — is in
[the Module 06 lab](../../docs/core/06-qlora.md).

```bash
# 1. Data hygiene: stdlib only, runs anywhere
PYTHONPATH=. python examples/fine-tuning/prepare_data.py \
  --input examples/fine-tuning/data/sample.jsonl --out data/processed

# 2. Training + eval: CUDA GPU (Colab T4 is enough)
python -m pip install -r examples/fine-tuning/requirements.txt
PYTHONPATH=. python examples/fine-tuning/evaluate.py --split data/processed/test.jsonl \
  --output results/fine-tuning/base.json
PYTHONPATH=. accelerate launch examples/fine-tuning/train_qlora.py \
  --data data/processed --output artifacts/adapters/v1
PYTHONPATH=. python examples/fine-tuning/evaluate.py --split data/processed/test.jsonl \
  --adapter artifacts/adapters/v1 --output results/fine-tuning/adapter.json
```

- `prepare_data.py`: validates every row (`src/finetune_data.py`), refuses to write if any row fails, then splits **by company** and writes a split manifest.
- `train_qlora.py`: 4-bit NF4 base + LoRA adapter via TRL `SFTTrainer`; checkpoints every 50 steps; `--resume` after a Colab disconnect.
- `evaluate.py`: same held-out rows for base and adapter; plain-text rate, unsupported-number rate, latency. Pair it with a human usefulness rubric.
- `data/sample.jsonl`: 8 **fictional** startup briefs. It proves the pipeline runs, not that an adapter helps — real results need 200–500 reviewed, rights-cleared examples.
- Keep `data/`, `artifacts/`, and `results/` out of git; commit code, schemas, and manifests.
