# Module 17 — Small & Local LLM Models

<span data-module-id="17" hidden></span>

**Time:** 5–7 days · **Depends on:** 01, 05, 10 · **Pairs with:** tracks using Phi / Ollama · **Next:** [Agent design patterns](18-agent-design-patterns.md) · **Agents on SLMs:** [24 Local-first](24-local-first-agents.md)

---

## Learning objectives

- Match small language models (SLMs) to tasks they can actually own
- Run local inference with Ollama, llama.cpp, and/or vLLM
- Apply quantization deliberately and **re-eval** quality after every compress step
- Size a model to **limited hardware** (RAM/VRAM, KV cache, one resident model) so the laptop stays out of swap
- Build a router that sends easy work to SLMs and hard work to larger models

## What you can build

- Offline / private assistant over local files
- Cheap classifier or router in front of a large model
- Quantized deployment on a laptop or small GPU with measured quality

---

## Why this matters (CS engineer)

<div class="aieng-story" markdown>

Finance wants the bill cut in half. The team swaps every call to a 3B local model “because demos looked fine,” then quantizes to Q4 so it fits on a laptop GPU. Schema pass rate on extraction collapses; the agent loops on tools the small model cannot plan. There was no **router**, no **re-eval after quant**, and no list of tasks the SLM actually owns. Cost went down; product quality and on-call load went up. The fix was not “bigger GPU” — it was **specialist first-line + escalate**, with golden metrics as the gate.

</div>

Not every token deserves a frontier model. Most production traffic is **classification, routing, extraction, short rewrite, and retrieval-augmented lookup** — tasks where a 1B–8B-class model (or a “mini” cloud tier) wins on **latency, cost, and privacy**. CS engineers who only know one cloud chat API overspend and cannot ship air-gapped or VPC-only features.

SLMs are not “GPT but free.” They need **tighter prompts, harder validation, and honest evals**. Treated as specialized workers in a system (Module 10 routing, Module 16 hybrid), they are one of the highest-ROI tools in the stack.

---

## Mental model

```mermaid
flowchart TB
  U[User task] --> R[Router SLM or rules]
  R -->|easy / high confidence| S[Local or mini SLM]
  R -->|hard / low confidence / schema fail| L[Large model]
  S --> V[Validate schema / policy]
  L --> V
  V -->|fail| L
  V -->|pass| Out[Response]
  Q[Quantize weights] --> S
  E[Golden eval] --> Q
  E --> R
```

**Invariant:** choose SLMs by **task fit + measured quality**, not by parameter count marketing. Always re-run golden evals after quantization or prompt changes.

<div class="aieng-intuition" markdown>

<p class="label">Intuition lock</p>

**Sticky picture:** an SLM is a **specialist intern** — fast, cheap, great on narrow labeled work. A frontier model is the **senior consultant** you escalate to when confidence is low or the schema fails. **Quantize, then re-eval** (never blog-trust a quant level). Privacy and latency economics often justify local run even when raw accuracy is a few points lower *on the right tasks*.

<p class="kill"><strong>Kill this idea:</strong> “Small model = free GPT for everything.” Parameter count marketing is not a task assignment. Without routing and validation you only move failure modes around.</p>

</div>

---

## 1. Strengths and limits

| Advantages | Limits |
|------------|--------|
| Lower latency and $ per call | Weaker multi-step / long-horizon reasoning |
| Privacy and air-gap options | Smaller context (model-dependent) |
| Fine-tune / LoRA friendly on modest hardware | Brittle to sloppy or huge prompts |
| Edge and on-prem control | Gaps in world knowledge and some multilingual settings |
| Great as routers and extractors | Tool-heavy agents may thrash without strong validation |

**Modern families to evaluate (verify latest model cards):** Phi-class, Llama 3.x compact sizes, Gemma, Qwen2.5 smaller sizes, Mistral small models, and cloud “mini/haiku/flash” tiers when local is not required.

<div class="aieng-explainer" markdown>

<p class="label">Explainer · small ≠ weak at everything</p>

A 3B model that **only** outputs one of five labels with a strict schema can beat a frontier model on **cost-adjusted reliability** for that task. A 70B model asked to “handle the ticket” with no structure can still fail product SLOs. Task design and validation often matter more than raw size.

</div>

---

## 2. Local runtimes

| Runtime | Fit | Notes |
|---------|-----|-------|
| [Ollama](https://ollama.com) | Dev laptop ergonomics | Pull/run UX; good default for learning |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU / Apple Metal, GGUF | Fine control of quant and threads |
| vLLM / TGI | Throughput serving on GPU | Batching, continuous batching for multi-user |
| LM Studio | GUI exploration | Fast qualitative checks |

```bash
# Ollama quickstart (install from ollama.com first)
ollama pull llama3.2
ollama run llama3.2 "Summarize RAG in 3 bullets."

# OpenAI-compatible local endpoint (typical pattern)
# curl http://localhost:11434/v1/chat/completions ...
```

```python
# Conceptual OpenAI-compatible client pointed at local server
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def local_chat(prompt: str, model: str = "llama3.2") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
```

**Serving note:** for multi-user prod on GPU, prefer vLLM/TGI-style servers; Ollama is excellent for dev and small deployments but measure concurrency limits before betting the company on it.

---

## 3. Prompting SLMs well

- **Shorter instructions**; explicit output formats  
- **More few-shot** when structure is fragile  
- **Decompose** → many small calls beat one giant reasoning soup  
- **Validate hard** (JSON schema, enum labels, regex)  
- Prefer temperature **0–0.2** for extract/classify  

```python
def classify_short(text: str, labels: list[str], llm) -> str:
    label_line = ", ".join(labels)
    prompt = (
        f"Classify into exactly one label: {label_line}.\n"
        f"Reply with the label only.\n\nText: {text}"
    )
    out = llm(prompt).strip()
    if out not in labels:
        raise ValueError(f"invalid_label:{out!r}")
    return out
```

On schema failure: **retry once with a repair prompt**, then **escalate** to a larger model (router pattern below).

<div class="aieng-think" markdown>

<p class="label">Think · when not to use an SLM</p>

<details data-think-id="17-t1">
<summary>Reveal: red flags that you need a larger model (or a non-LLM system)</summary>

- Multi-hop reasoning over long, conflicting documents without a strong RAG scaffold  
- Open-ended strategy / novel coding on large repos  
- Safety-critical nuance where small models fail your golden **risk** cases  
- Tasks already solved better by rules, SQL, or classical ML classifiers  

If a deterministic extractor works, do not pay for tokens — small or large.

</details>

</div>

---

## 4. Quantization

Quantization reduces weight precision so models fit in RAM/VRAM and run faster — **at a quality cost you must measure**.

A weight that was stored as 16-bit floating point (many possible values, smooth math) is stored as 8- or 4-bit integers (a short menu of values plus a scale). Matmul still works; the numbers are just coarser. That coarseness shows up first on **brittle tasks**: multi-digit arithmetic, strict JSON, long-horizon tool plans. Sentiment and short classify often look fine until a cliff. That is why “Q4 is fine” is not a product statement until *your* golden set says so.

| Approach | Notes |
|----------|-------|
| 8-bit / 4-bit weights | Big VRAM wins; measure task metrics |
| GGUF Q4 / Q5 / Q6 | Common for llama.cpp / Ollama |
| Speculative decoding | Draft small model + verify large (advanced speed) |
| Distillation | Teacher large → student small (training pipeline) |

```text
FP16 model → quantize Q4 → golden eval
                │
                ├─ pass SLO → ship Q4
                └─ fail → try Q5/Q8, different model, or keep FP16 for hard path only
```

**Rules**

1. Never ship a quant level because a blog said “Q4 is fine.”  
2. Re-run **your** golden set (Module 04) after every quant change.  
3. Watch **refuse / schema failure rate**, not only average “vibe.”  
4. You may run **Q4 for router** and **higher precision for final answer** on the same host.

<div class="aieng-explainer" markdown>

<p class="label">Explainer · quality cliffs</p>

Quantization error is uneven: some tasks (sentiment, short classify) stay flat until aggressive quants; others (multi-digit reasoning, brittle JSON) fall off a cliff. Plot metric vs quant level. The right product answer is often **mixed precision routing**, not one global quant for every call.

</div>

---

## 5. Router pattern (highest ROI)

```text
User → cheap SLM router / rules
         ├─ easy → SLM answer (+ validate)
         └─ hard / low confidence / validation fail → large model
```

```python
from dataclasses import dataclass


@dataclass
class RouteDecision:
    model: str
    reason: str


class ModelRouter:
    def __init__(self, cheap: str, strong: str):
        self.cheap = cheap
        self.strong = strong

    def pick(self, task: str, prompt: str, *, confidence: float | None = None) -> RouteDecision:
        if task in {"classify", "route", "extract_fields"}:
            return RouteDecision(self.cheap, "narrow_task")
        if task == "complex_reason" or len(prompt) > 8000:
            return RouteDecision(self.strong, "hard_or_long")
        if confidence is not None and confidence < 0.6:
            return RouteDecision(self.strong, "low_confidence")
        return RouteDecision(self.cheap, "default_cheap")


def generate_with_escalation(task: str, prompt: str, llms: dict, router: ModelRouter) -> str:
    decision = router.pick(task, prompt)
    text = llms[decision.model](prompt)
    # example: escalate if JSON parse fails
    if task == "extract_fields":
        try:
            import json

            json.loads(text)
        except Exception:
            text = llms[router.strong](prompt)
    return text
```

Combine with Module 10 cost ledgers: track **% escalated**, **$ per successful task**, and **quality** on a fixed eval set.

<div class="aieng-think" markdown>

<p class="label">Think · economics of escalate</p>

<details data-think-id="17-t2">
<summary>Reveal: when is a high escalate rate still a win?</summary>

If 80% of traffic is cheap classify/extract that the SLM nails, and 20% escalates to a large model, blended **$ per success** and p95 can still beat “always large” — *if* escalate catches the hard tail and quality SLOs hold. A 90% escalate rate means your router is noise: fix task labels, prompts, or stop pretending the SLM owns the hard path. Track escalate rate next to quality; optimize the blend, not “never call large.”

</details>

</div>

---

## 6. SLMs + RAG + privacy

Local models shine when documents must not leave the device or VPC:

1. Embed and retrieve on-prem (or on-laptop).  
2. Generate with local SLM grounded on retrieved chunks.  
3. Keep audit logs local (Module 14).  

Still apply **injection hygiene** (Module 02): retrieved text is data, not instructions. Small models can be *more* suggestible — validation and allowlisted tools matter more, not less.

---

## 7. Working effectively on limited hardware

This course assumes a **laptop, often no discrete GPU**. That is a product constraint, not an apology. A 3B model that stays in RAM and answers in 200 ms will beat an 8B that thrashes swap and fans for 40 seconds — on quality *and* on whether you actually use it.

<div class="aieng-intuition" markdown>
<p class="label">Intuition lock</p>

**Sticky picture:** RAM is a **loading dock**, not a warehouse. Weights + KV cache + OS have to stand on it at once. Swap is **shipping the dock to another city between tokens**. Context length is **rent on the dock** (KV cache grows with every token). One resident model is **one truck**; two 7Bs is a traffic jam.

<div class="kill" markdown>
**Kill this idea:** “Pull the biggest GGUF that Ollama will download; RAM will figure it out.” → **Replace with:** Fit weights + KV + ~5 GB OS headroom. Cap `num_ctx`. Keep **one** model loaded. Prefer a smaller model that is **hot in RAM** over a larger one that is **cold on disk**.
</div>
</div>

### RAM is the limiter

```text
working set ≈ weights + KV cache + runtime + OS
weights     ≈ params_B × (bits / 8)   GB     (Q4 ≈ 0.5 byte/param)
KV cache    grows with context × layers × batch  (often the surprise)
```

```python
from src.local_agents import HardwareBudget, recommend_local_setup, weight_gb

assert weight_gb(8.0, bits=4) == 4.0   # 8B Q4 ≈ 4 GB of weights
fit = recommend_local_setup(HardwareBudget(ram_gb=16))
# LocalFit(params_b=8.0, quant="Q4", max_ctx=4096, ...)
```

`recommend_local_setup` is a **teaching table**, not a profiler. It reserves ~5 GB for macOS/Windows, the browser, and Python. Re-eval after you pick a real GGUF — cards lie about “fits in 8 GB” because they forgot KV cache and Chrome.

| Machine | Honest local default | Do not |
|---------|----------------------|--------|
| **8 GB RAM, CPU/Metal** | 1–3B Q4, `num_ctx` 2k, one model | 7B Q4 (will swap) + a second model |
| **16 GB** | 7–8B Q4, `num_ctx` 4k | 13B Q4 + 32k context “for RAG” |
| **32 GB** | 8B Q8 **or** 13–14B Q4, ctx 8k if evals hold | Two 13Bs resident |
| **6–8 GB VRAM dGPU** | Offload 7B Q4 layers to GPU; short ctx | Full 13B FP16 |
| **No GPU, old CPU** | 1B–3B Q4; fewer threads than you think | Benchmarking while compiling Chromium |

Apple Silicon: **Metal** is the reason 7–8B Q4 is pleasant. x86 laptop CPU: expect **single-digit to low tens of tok/s** — still enough for classify/extract, painful for long chat. Measure `tok/s` after the **second** prompt (first prompt pays load + compile).

<div class="aieng-explainer" markdown>
<p class="label">Explainer · why context eats RAM</p>

Weights are mostly **fixed**. The KV cache is **per token of context** (keys and values for every layer). Doubling `num_ctx` can add more RAM than dropping one quant level saves. A “32k context” 7B on 16 GB often loses to a 4k context 7B that actually stays resident. Module 05 packing is a **hardware** feature here: retrieve 3 chunks, not 30.
</div>

### Knobs that matter on a laptop

| Knob | What to do | Why |
|------|------------|-----|
| **One resident model** | `ollama stop` extras; don’t keep 8B + 3B + embedder if RAM is tight | Each model’s weights sit in RAM/VRAM |
| **`num_ctx` / `n_ctx`** | Set to what you **use** (2k–4k for SLM tasks) | KV cache |
| **`num_predict` / max tokens** | Cap completions (64–256 for extract/classify) | Latency and RAM |
| **Threads** | Physical cores, not “all logical” | Oversubscription thrashes |
| **mmap** | Keep on (llama.cpp default) | OS pages weights; don’t force a full copy |
| **Keep-alive** | Keep the **one** model loaded while you work; unload overnight | Avoid reload tax vs RAM hog |
| **Batch = 1** | Interactive laptop | Throughput servers (vLLM) are a different machine |
| **Embedder** | Tiny (e.g. <100M) or hash/keyword until RAM allows | A 7B *plus* a large embedder is two models |

Ollama-shaped example (names vary; check `ollama help`):

```bash
# Prefer a tag that matches your RAM (see table), then pin context
ollama run llama3.2  # 3B-class; good 8–16 GB default
# In a Modelfile / API options:
# num_ctx: 2048
# num_predict: 128
# num_thread: 4      # set to physical cores
```

**Swap is a stop-the-line signal.** If Activity Monitor / `htop` shows swap climbing while you generate, the model is too big. Shrink params, quant, or context — do not “give it a few more minutes.” Swap-backed inference is slower than a cloud mini and wrecks the SSD.

**Thermals:** laptop CPU/GPU will **throttle**. Do not publish tok/s from the first 10 seconds on a cold chassis. Steady state after a minute is the number that matters.

### Prompt and system design that small hardware can survive

Hardware limits and prompt limits are the same list:

1. **Short system prompt.** A 2k sermon leaves no room for the user task in a 2k window.
2. **JSON / enums**, temperature 0–0.2, few-shot of **one** compact example — not five essays (already §3).
3. **Decompose.** Five 200-token SLM calls beat one 4k “think hard” call that OOMs the KV cache.
4. **RAG:** top-k 2–4, chunk small, citations required (Modules 07/09). Unbounded retrieve is a RAM attack.
5. **Agents:** `max_steps` 4–8, truncate tool dumps (Module 11/24). A small model with a god-tool will loop until the fan is the loudest component.
6. **Escalate** (this module’s router + Module 24) when JSON fails — that is cheaper than stuffing a 70B into 16 GB.

<div class="aieng-think" markdown>
<p class="label">Think about it</p>

**Question:** 16 GB Mac, Ollama, you want “repo Q&A.” You can load 8B Q4 at `num_ctx=4096` at ~15 tok/s, or 3B Q4 at `num_ctx=2048` at ~40 tok/s. Retrieval already returns the right 3 chunks. Which do you ship for classify + short answer, and what do you measure?

<details data-think-id="17-t3"><summary>Reveal a strong answer</summary>

Ship the **smallest model that clears the golden set** at the context you actually pack. If 3B + 3 chunks matches 8B on schema-pass and citation hit, take 3B: more headroom, less swap risk, snappier UI. Measure schema-pass, Hit@k of cites, p95 latency **after warmup**, and whether swap is zero. If 3B fails JSON, try 8B Q4 at 4k *or* escalate that 10% of calls (Module 10/24) — don’t jump to 32k context as a quality fix. Context is RAM.

</details>
</div>

### What not to do on this hardware

| Temptation | What happens |
|------------|----------------|
| 32k / 128k context “because the card says so” | KV cache evicts the OS; fans; silence |
| LoRA-train 8B overnight on 16 GB | OOM or multi-hour swap; use PEFT on a smaller base or a rented GPU (Module 06) |
| vLLM on a 16 GB laptop | Wrong runtime; that’s a throughput GPU server |
| Three Ollama models idle | 12 GB of weights before the first token |
| Fine-tune instead of retrieve | RAM + time; try RAG first (Module 07) |

Fine-tunes and 13B+ still belong in this course — on **comfortable** hardware (setup table) or a rented box. The laptop path is **specialize, cap context, validate, escalate**.

```bash
poetry run pytest tests/test_local_agents.py -v
```

---

## Failure modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| One giant prompt to 3B | Gibberish / truncate | Decompose + retrieve less junk |
| Quant without eval | Silent quality drop | Golden suite gate |
| SLM as sole agent brain | Looping tools, bad plans | Router + max steps + large-model escalate |
| Trusting “label only” without check | Free-form prose labels | Enum validate / repair / escalate |
| Undersized context | Lost instructions | Shorter system prompts; external memory |
| Oversized context on 16 GB | Swap, thermal throttle, “model hung” | Cap `num_ctx`; pack (Module 05); smaller top-k |
| Two models resident | Mystery OOM / 2 tok/s | One hot model; unload the rest |
| Local server open to LAN | Data exposure | Bind localhost / auth / firewall |

---

## Lab

<div class="aieng-lab" markdown>

<p class="label">Lab · measure before you commit</p>

1. Run a **3B–8B-class** model locally (Ollama is fine) on **20 golden tasks** from your project.  
2. Score accuracy / schema-pass vs a cloud mini model on the same set.  
3. Implement confidence or validation routing (schema fail → escalate).  
4. If you quantize (e.g. compare two GGUF levels), **re-run the same 20** and record the delta.  
5. Write a short decision: which tasks SLM owns, which escalate, and why.  
6. Run `recommend_local_setup(HardwareBudget(ram_gb=<yours>))`. Confirm the model you used in step 1 **fits** that row (weights + headroom). If Activity Monitor showed swap, drop a quant level or a billion parameters and re-run the 20.

</div>

---

## Quizzes

<div class="aieng-quiz" data-quiz-id="17-q1" data-xp="25" data-success="Correct — quantization is a product decision only after evals say quality is still good enough." data-fail="Re-read §4: always re-run golden evals after quantizing." markdown>

<p class="label">Quiz · 25 XP</p>
<p class="quiz-prompt">You quantize a local model from Q8 to Q4 to fit on a laptop. What must you do before shipping?</p>
<div class="quiz-options">
<button type="button" class="quiz-opt" data-correct="false">Nothing — Q4 is always within 1% of FP16</button>
<button type="button" class="quiz-opt" data-correct="true">Re-run your golden evals and compare task metrics / failure modes</button>
<button type="button" class="quiz-opt" data-correct="false">Only check that `ollama run` still prints text</button>
<button type="button" class="quiz-opt" data-correct="false">Switch temperature to 1.0 to compensate</button>
</div>
<div class="quiz-feedback"></div>
</div>

<div class="aieng-quiz" data-quiz-id="17-q2" data-xp="25" data-success="Yes — routers + validation capture most of the economic win." data-fail="Think about Module 10/17: SLMs as first-line workers with escalate paths." markdown>

<p class="label">Quiz · 25 XP</p>
<p class="quiz-prompt">What is usually the highest-ROI production pattern involving SLMs?</p>
<div class="quiz-options">
<button type="button" class="quiz-opt" data-correct="false">Replace every frontier call with the smallest model that fits in RAM, no validation</button>
<button type="button" class="quiz-opt" data-correct="true">Use an SLM (or rules) to handle easy tasks and route hard/low-confidence work to a larger model</button>
<button type="button" class="quiz-opt" data-correct="false">Always ensemble five SLMs and majority vote</button>
<button type="button" class="quiz-opt" data-correct="false">Fine-tune a 70B model on a laptop overnight</button>
</div>
<div class="quiz-feedback"></div>
</div>

<div class="aieng-quiz" data-quiz-id="17-q3" data-xp="25" data-success="Swap means the working set does not fit; shrink weights or context." data-fail="Re-read §7: swap is a stop-the-line signal, not a waiting room." markdown>

<p class="label">Quiz · 25 XP</p>
<p class="quiz-prompt">Your 16 GB laptop starts paging (swap climbing) while a local 13B Q4 generates. What is the right first move?</p>
<div class="quiz-options">
<button type="button" class="quiz-opt" data-correct="false">Raise `num_ctx` to 32k so the model can “see more” and finish faster</button>
<button type="button" class="quiz-opt" data-correct="true">Unload extra models, cap context, or drop to an 8B Q4 / 7B that stays in RAM — then re-eval</button>
<button type="button" class="quiz-opt" data-correct="false">Set temperature to 1.0 so it uses fewer tokens</button>
<button type="button" class="quiz-opt" data-correct="false">Start vLLM to “use memory more efficiently” on the same laptop</button>
</div>
<div class="quiz-feedback"></div>
</div>

---

## OSS & further materials

| Resource | Why |
|----------|-----|
| [Ollama](https://ollama.com) | Fast local dev loop |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF + CPU/Metal control; `n_ctx`, threads, mmap |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput GPU serving (not a 16 GB laptop default) |
| Course `src.local_agents.recommend_local_setup` | Teaching RAM fit table |
| Hugging Face model cards | License, context length, intended use |
| Module 10 Cost optimization | Routing and unit economics |
| Module 04 Testing & evals | Golden sets for quant gates |

---

## Checkpoint

- [ ] You ran at least one local model end-to-end  
- [ ] You know which of *your* tasks SLMs can own  
- [ ] Quantization (if any) is eval-backed  
- [ ] You can size a model to **your** RAM (weights + KV + OS) and name the knobs (`num_ctx`, one resident model)  
- [ ] A router or validation-escalation path exists on paper or in code  

<div class="aieng-complete" data-module-id="17" data-xp="100" markdown>
<p>Mark Module 17 complete when local run + eval comparison are done honestly.</p>
<button type="button">Complete module · +100 XP</button>
</div>

**Next:** [Module 18 — Agent design patterns](18-agent-design-patterns.md) · or jump to a [specialization track](../tracks/index.md)
