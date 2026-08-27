# Troubleshooting Guide

## Prompting

**Symptom:** Inconsistent answers  

```text
# Weak
Analyze this

# Stronger
Role: data analyst
Task: return 3 findings, 1 risk, 2 questions
Format: Markdown H2 sections
Constraints: no speculation beyond data
```

**Symptom:** Token limit exceeded  

- Summarize history (Module 05)  
- Retrieve less, rerank better (Module 09)  
- Cap tool output size before re-feeding  

---

## RAG

**Symptom:** Irrelevant chunks  

- Check chunk boundaries (tables/code)  
- Hybrid search + rerank  
- Evaluate Hit@k separately from answer quality  

**Symptom:** Hallucinated cites  

- Require ids from the provided set only  
- Post-validate cite ids ∈ retrieved set  

---

## Agents

**Symptom:** Infinite tool loops  

- `max_steps`, detect repeated (tool,args) pairs  
- Fail closed to `ask_user`  

**Symptom:** Destructive actions  

- Approval gates; dry-run tools; least privilege  

---

## Performance & cost

**Symptom:** High latency  

- Stream tokens; cache; smaller model for easy tasks  
- Parallelize independent retrievals  

**Symptom:** Bill shock  

- Per-user budgets; routing; prompt caching; log `$/success`  

---

## Local models

**Symptom:** Garbled or off-format outputs  

- Lower temperature; tighter schemas; more few-shot  
- Escalate hard cases to a larger model  

**Symptom:** OOM / swap climbing / fans at 100% for a “small” local model

- Working set = weights + **KV cache** + OS. Cap `num_ctx`; keep **one** resident model
- Drop params (8B → 3B) before you drop to Q2 folklore
- See [Module 17 §7](../core/17-small-models.md#7-working-effectively-on-limited-hardware)  

---

## Agents (modules 11–12, 20–27)

**Symptom:** Same tool call forever

- Abort on canonical `name+args` (Module 11 / 20 `FailureDetector`)
- Empty hits: reformulate once, then final / I-don’t-know — don’t open the search circuit

**Symptom:** Model invents `run_sql` / `bash`

- Allowlist + `PrivilegeError`; never `eval` model text (Module 21)

**Symptom:** Writes landed on the user’s tree

- `WorktreeExecutor` + `MergeGate`; approval default deny (Modules 21, 25)

**Symptom:** Quality dropped, HTTP still 200

- Trajectory composite + `eval_regression` vs pinned prompt digest (Modules 22–23)

**Symptom:** Bill unexplained

- `CostEvent` per agent/step (Module 26); token budget before the next call (24)

**Symptom:** MCP server hung or returned “ignore your policy”

- Circuit + failover; wrap as untrusted resource; pin version/digest (Module 08 §8)

---

## API / ops

**Checklist**

1. Keys present, not expired, correct project  
2. Network / proxy / regional endpoint  
3. Model id still valid  
4. Timeouts vs provider SLA  
5. Rate limits → backoff  
6. Schema validation errors on tool args  

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 8))
def robust_call(fn, *args, **kwargs):
    return fn(*args, **kwargs)
```

---

## Docs site build

```bash
pip install -r requirements-docs.txt
mkdocs build --strict
```

If `--strict` fails: fix broken internal links listed in the error output.
