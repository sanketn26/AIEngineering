# Module 11 — Single-Agent Workflows

**Time:** 7–10 days · **Depends on:** 03–05, 07 · **Next:** [Multi-agent](12-multi-agents.md)

---

## Learning objectives

- Implement a plan–act–observe loop with hard stops
- Add reflection and tool-error recovery
- Keep agents observable and testable

## What you can build

- Research assistant with tools
- Code refactor agent over a repo snapshot
- Ticket triage agent with human approval gates

---

## Core loop

```text
goal
  → plan (optional)
  → select action (tool | respond | ask_user)
  → observe result
  → update state
  → until done | max_steps | abort
```

Implemented as `src.agents.Agent` with max-steps, allowlisted tools, and repeated-call abort (`tests/test_agents.py`):

```python
import json
from src.agents import Agent

def llm(prompt: str) -> str:
    # stub: real systems call a model with structured outputs
    if "Scratchpad:" in prompt and "Tool" not in prompt.split("Scratchpad:")[-1]:
        return json.dumps({"type": "tool", "name": "echo", "args": {"text": "hi"}})
    return json.dumps({"type": "final", "content": "hi"})

agent = Agent(llm=llm, tools={"echo": lambda text: text}, max_steps=5)
state = agent.run("echo hi then finish")
assert state.result == "hi"
```

Prefer battle-tested graphs (LangGraph, etc.) once you understand the loop.

---

## Patterns

### Reflection

After a draft answer: “Critique against the goal; list gaps; revise once.” Cap to 1–2 reflections to control cost.

### Planning

For multi-step goals, produce a short checklist first; execute item by item; mark complete in state (don’t keep the plan only inside free text).

### Tool-use discipline

- Small tool surface  
- Typed args  
- Idempotent tools when possible  
- Explicit errors in observations  

---

## Stop conditions

| Condition | Why |
|-----------|-----|
| `max_steps` | Prevent infinite tool loops |
| Token / $ budget | Economic safety |
| Repeated tool signature | Detect thrashing |
| User gate | Irreversible actions |
| Validation pass | Structured success criteria |

---

## Observability

Log for each step: timestamp, tool name, latency, arg hashes, success/fail, tokens. You cannot improve what you cannot replay.

---

## Exercise

1. Build an agent with tools: `search_notes`, `calculator`, `final_answer`.  
2. Force a tool failure; ensure the agent recovers or aborts cleanly.  
3. Add `max_steps=5` and a unit test that a noop goal terminates.

---

## Checkpoint

- [ ] Loop has hard stop  
- [ ] Tools are allowlisted  
- [ ] Steps are logged for replay  

**Next:** [Module 12 — Multi-agent systems](12-multi-agents.md)
