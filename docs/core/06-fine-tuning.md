# Module 06 — Fine-Tuning & Model Customization

**Time:** 7–10 days · **Depends on:** 01–05 · **Next:** [Tools & RAG](07-tools-and-rag.md)

!!! tip "Default bias"
    Prefer **prompting + RAG + tools** first. Fine-tune when you have stable, high-volume behavioral gaps those cannot fix economically.

---

## Learning objectives

- Decide *when* fine-tuning beats alternatives
- Prepare and validate instruction datasets
- Understand LoRA/QLoRA at a practical level

## What you can build

- Domain-style adapters on open models
- Evaluation harness comparing base vs. adapter
- Dataset QA checklist

---

## Decision tree

```text
Need private knowledge that changes often?  → RAG / tools
Need reliable tool formats / JSON?          → Structured outputs + prompts + evals
Need a new capability model lacks?          → Maybe train/finetune or bigger model
Need tone/format locked at high volume?     → Fine-tune (or strong system + few-shot)
Need on-device / offline specialized skill? → SLM fine-tune + quantize
```

| Approach | Pros | Cons |
|----------|------|------|
| Prompt / few-shot | Fast, reversible | Token cost; weaker style lock |
| RAG | Fresh facts | Retrieval quality dependent |
| Fine-tune | Sticky behavior, shorter prompts | Data cost, drift, eval burden |
| Distill | Cheap student model | Teacher quality ceiling |

---

## Data requirements (instruction tuning)

Typical row:

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise support agent for Acme."},
    {"role": "user", "content": "How do I reset my API key?"},
    {"role": "assistant", "content": "1. Open Settings → API...\n2. ..."}
  ]
}
```

**Quality beats quantity.** Hundreds of clean, diverse examples often beat tens of thousands of noisy ones for style tasks.

### Validation checklist

- [ ] No PII / secrets  
- [ ] Consistent schema  
- [ ] Balanced intents / edge cases  
- [ ] Gold answers reviewed by a human  
- [ ] Held-out eval set never used in training  

---

## LoRA / QLoRA (conceptual)

- **LoRA:** train small rank adapters on attention/MLP projections; freeze base weights.  
- **QLoRA:** quantize base (e.g. 4-bit) + train LoRA for consumer GPUs.  

```text
# Illustrative training stack (check current HF docs for flags)
# transformers + peft + trl + bitsandbytes / torchao
```

Track: learning rate, rank `r`, target modules, epochs, eval loss **and** task metrics (not only loss).

---

## Minimal PEFT-shaped sketch

```python
# Pseudocode — pin package versions from current PEFT docs before running
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-4-mini-instruct"  # example SLM family; verify card

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # load_in_4bit=True,  # when using bitsandbytes/QLoRA toolchain
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # model-specific
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
# → SFTTrainer / custom loop on tokenized messages
```

---

## Evaluate the adapter

Compare **base vs. adapter** on the same golden set (Module 04):

- Task accuracy / rubric  
- Regression on general capabilities (don’t destroy math/coding if you need them)  
- Latency and VRAM  

Serve adapters via merged weights or runtime adapter load (vLLM / TGI patterns).

---

## Exercise

1. Write a one-page “fine-tune or not” decision for your project.  
2. Create 30 high-quality instruction rows + 10 held-out.  
3. If hardware allows, run one epoch LoRA and report metric delta vs. base + RAG.

---

## Checkpoint

- [ ] You can argue for/against fine-tuning on your use case  
- [ ] Dataset schema is validated  
- [ ] Eval is task-based, not only train loss  

**Next:** [Module 07 — Tools & basic RAG](07-tools-and-rag.md)
