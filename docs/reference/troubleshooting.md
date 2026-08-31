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

---

## Progressive debugging (production AI)

Work the layer that is actually failing. Each section: what it looks like, where it usually is, what to open, and **what not to change yet** (so you do not "fix" a schema bug by adding another agent).

### Structured output

**Symptom:** HTTP 200 with a blob you cannot parse, or 422/5xx after a schema validator you added.

**Likely cause:** The model (or heuristic) is emitting prose, extra keys, or a type the contract forbids. Temperature too high. Schema not shown to the model. A "helpful" regex rescue that almost works.

**What to inspect:** The raw provider payload *before* Pydantic. The schema (`capstone-starter/schemas.py` or your model). A 20-row parse-success sample, not one demo. Logs keyed by `request_id`.

**What not to change yet:** Prompts for tone, RAG chunk size, adding a second model, or loosening the schema to `dict[str, Any]` so the test goes green.

### Tool calling

**Symptom:** The model asks for `run_sql` / a tool you never registered, or calls the right tool with the wrong types, or never stops calling.

**Likely cause:** No allowlist in the *runtime*. Args not schema-validated. Repeated `(name, args)` not detected. The system prompt "lists" tools but the dispatcher `eval`s free text.

**What to inspect:** The decision object the model returned. The allowlist vs the call. `FailureDetector` / `max_steps` (Module 20). One golden trajectory that must abort.

**What not to change yet:** The tool implementation internals, the KB, or swapping agent frameworks. If dispose() is not in your process, a better prompt will not save you.

### RAG

**Symptom:** Fluent answer, wrong or invented citations; or "I don't know" on questions the corpus answers.

**Likely cause:** Chunk boundaries split tables/ids. Dense-only miss on tokens/IDs. Cite ids not intersected with the retrieved set. Empty index still answering from weights.

**What to inspect:** Hit@k on a labeled slice *before* you look at the prose. The retrieved ids vs `citations`. The unanswerable query. `validate_citations` (Module 07).

**What not to change yet:** The generator prompt, the agent loop, or fine-tuning. If Hit@k is bad, the generator is decorating a miss.

### Orchestration

**Symptom:** Steps run in the wrong order, state is lost between nodes, or two agents overwrite the same field.

**Likely cause:** Hidden shared mutable state. No typed payload between nodes. A graph library used as a substitute for a schema. Planner emitted a step the runtime cannot run.

**What to inspect:** The message/event between hop N and N+1. The persisted run record (`request_id`, phase). One failing fixture that is *not* the happy path.

**What not to change yet:** Model choice, temperature, or adding a fourth agent to "coordinate the coordinators."

### Retries

**Symptom:** Latency cliffs, duplicate side effects, or a worker that never returns.

**Likely cause:** No deadline on the provider call. Retrying non-transient 4xx / schema failures. Retrying a write tool. Unbounded backoff.

**What to inspect:** `capstone-starter/model.py` `_call_provider` (Gate 1 hole). Status codes you retry vs fail closed. Whether the tool is idempotent. Trace timestamps vs the timeout you *think* you set.

**What not to change yet:** Batch size, a larger model, or "just raise max_retries." A hang is a deadline bug.

### Agent loops

**Symptom:** Same search forever; cost alert; "success" after 40 steps of junk.

**Likely cause:** Empty hits re-issued with the same query. No `max_steps`. Success defined as "the model said final" rather than a verifier. Repeated-args not canonicalized (key order).

**What to inspect:** The trajectory log. Canonical `tool+args` signatures. The stop condition in code (Module 11/20/27), not in the prompt. Composite eval on a looping-success fixture (Module 22).

**What not to change yet:** The system prompt's "please stop when done." If stop lives only there, it does not live.

### Sandboxing

**Symptom:** The agent edited the user's tree, or a path like `../.env` was read, or a write landed without a click.

**Likely cause:** Tools using the process CWD. Approval as a prompt instruction. Worktree skipped. `requires_approval` not checked in the executor.

**What to inspect:** Absolute paths vs sandbox root. Whether `apply_diff` ran on the source tree. Grant + human flags (Module 21). `tracks/starters/agentic-plugin` `read_file` prefix check as the minimum.

**What not to change yet:** Model vendor, MCP server list, or "we'll add git undo later."

### Evaluation drift

**Symptom:** CI still green, users say quality dropped. Or every change looks like a win because you edited the golden labels.

**Likely cause:** Prompt/tool-list/model id changed without a pin (Module 23). Golden set too small or contaminated by tuning. LLM-as-judge without agreement. Threshold not actually failing the build.

**What to inspect:** Prompt digest vs live config. `evals/golden.jsonl` vs the last merge. `eval_regression` / `regression_delta`. The planted miss in the capstone starter — if you deleted it to go green, that is the bug.

**What not to change yet:** The production prompt "just a little," the judge model, or the threshold. Freeze the pin, re-run the suite, then change one thing.
