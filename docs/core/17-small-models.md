# Module 17 — Small & Local LLM Models

**Time:** 5–7 days · **Depends on:** 01, 05, 10 · **Pairs with:** tracks using Phi / Ollama

---

## Learning objectives

- Match SLMs to tasks they can actually do
- Run local inference (Ollama / llama.cpp / vLLM)
- Combine SLMs with routers, RAG, and distillation

## What you can build

- Offline assistants for private data
- Cheap classifiers / routers in front of large models
- Quantized deployments on laptops or small GPUs

---

## Advantages & limits

| Advantages | Limits |
|------------|--------|
| Latency, cost, privacy | Weaker multi-step reasoning |
| Air-gapped options | Smaller context (model-dependent) |
| Fine-tune friendly | Brittle to sloppy prompts |
| Edge / on-prem | Multilingual & world knowledge gaps |

**Modern SLM families to evaluate (verify latest cards):** Phi-4 class, Llama 3.x compact, Gemma, Qwen2.5 smaller sizes, Mistral small models.

---

## Local runtimes

| Runtime | Fit |
|---------|-----|
| [Ollama](https://ollama.com) | Dev laptop ergonomics |
| llama.cpp | CPU/metal-friendly GGUF |
| vLLM / TGI | Throughput serving |
| LM Studio | GUI local exploration |

```bash
ollama pull llama3.2
ollama run llama3.2 "Summarize RAG in 3 bullets."
```

---

## Prompting SLMs

- Shorter instructions; explicit formats  
- More few-shot for structure  
- Prefer decompose → many small calls over one giant reasoning soup  
- Validate outputs hard (schemas)  

```python
def classify_short(text: str, labels: list[str], llm) -> str:
    label_line = ", ".join(labels)
    prompt = (
        f"Classify into exactly one label: {label_line}.\n"
        f"Reply with the label only.\n\nText: {text}"
    )
    return llm(prompt).strip()
```

---

## Quantization

| Approach | Notes |
|----------|-------|
| 8-bit / 4-bit weights | Big VRAM wins; measure quality |
| GGUF Q4/Q5 | Common for llama.cpp/Ollama |
| Speculative decoding | Speed with draft model (advanced) |
| Distillation | Teacher large → student small |

Always re-run your golden eval after quantizing.

---

## Routing architecture

```text
User → cheap SLM router
         ├─ easy → SLM answer
         └─ hard / low confidence → large model
```

This is usually the highest ROI “small model” production pattern.

---

## Exercise

1. Run a 3B–8B class model locally on 20 golden tasks.  
2. Measure accuracy vs. a cloud mini model.  
3. Implement confidence routing (e.g. schema fail → escalate).

---

## Checkpoint

- [ ] You ran at least one local model end-to-end  
- [ ] You know which of *your* tasks SLMs can own  
- [ ] Quantization decisions are eval-backed  

**Next:** Pick a [specialization track](../tracks/index.md) or review [Troubleshooting](../reference/troubleshooting.md)
