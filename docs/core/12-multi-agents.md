# Module 12 — Multi-Agent Coordination

**Time:** 10–14 days · **Depends on:** 11 · **Next:** [Production](13-production.md)

---

## Learning objectives

- Split work across specialized roles without chaos
- Choose sequential, hierarchical, or graph topologies
- Handle disagreement, handoffs, and shared memory

## What you can build

- Research → writer → critic pipelines
- Support triage with specialist escalations
- Multi-role coding workflows (plan / implement / test)

---

## When multi-agent helps

| Helps | Hurts |
|-------|-------|
| Clear role boundaries | Tiny tasks (overhead > value) |
| Parallelizable research | Shared mutable state without locks |
| Independent critique | No success metric |

Default to **one agent + tools** until a second role has a crisp interface.

---

## Topologies

```text
Sequential:   A → B → C → answer
Hierarchical: Manager → delegates → workers → merge
Peer graph:   Agents message on channels / shared store
```

Frameworks to study: **LangGraph**, **CrewAI**, **AutoGen/AG2**, provider agent SDKs. Learn concepts first; adopt a framework second.

---

## Message contract

```python
from dataclasses import dataclass
from typing import Any
import time

@dataclass
class Message:
    sender: str
    recipient: str  # or "broadcast"
    type: str       # task | result | critique | question
    payload: dict[str, Any]
    ts: float = time.time()
```

Keep payloads **structured**. Natural language only at the edges.

---

## Manager–worker sketch

```python
class Manager:
    def __init__(self, workers: dict[str, callable], llm):
        self.workers = workers
        self.llm = llm

    def run(self, goal: str) -> str:
        plan = self.llm(
            f"Split into tasks as JSON list of "
            f"{{role, instruction}}. Goal: {goal}"
        )
        # parse plan → for each task call workers[role]
        results = []
        # ... execute ...
        return self.llm(
            f"Synthesize final answer for: {goal}\nResults:\n{results}"
        )
```

---

## Consensus & conflict

- **Critic role** with authority to request revision once  
- **Judge** rubric for pairwise ranking  
- **Human** for high-impact disagreements  
- Prefer merge strategies (union of facts + flag conflicts) over silent overwrite  

---

## Shared memory

| Store | Use |
|-------|-----|
| Scratchpad per agent | Local reasoning |
| Shared task board | Status of subtasks |
| Vector store | Long research notes |
| Single writer for final artifact | Avoid clobbering |

---

## Exercise

1. Implement researcher + writer + critic with a max of 2 critique rounds.  
2. Log every handoff message.  
3. Compare quality/cost vs. a single-agent baseline on 10 tasks.

---

## Checkpoint

- [ ] Each agent has a written charter and I/O schema  
- [ ] There is a max round / budget  
- [ ] You measured whether multi-agent beat single-agent  

**Next:** [Module 13 — Production systems](13-production.md)
