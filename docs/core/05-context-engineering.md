# Module 05 — Context Engineering & Memory

**Time:** 5–7 days · **Depends on:** 01–04 · **Next:** [Fine-tuning](06-fine-tuning.md)

---

## Learning objectives

- Treat the context window as a **scarce, ordered resource**
- Design packing, summarization, and memory tiers
- Avoid “stuff the transcript” as a product strategy

## What you can build

- Chat apps with sliding-window + summary memory
- Context packers that prioritize system > tools > user > history
- Document pipelines that fit token budgets

---

## Why “context engineering” (not only prompt engineering)

Modern failures are often:

- Wrong *facts* in the window  
- Stale tool results  
- History drowning the instruction  
- Retrieval junk crowding the answer space  

**Prompt engineering** shapes instructions. **Context engineering** decides *what enters the window, in what order, at what fidelity*.

---

## Information hierarchy (recommended order)

```text
1. System policy & non-negotiables
2. Task instructions for this turn
3. High-signal retrieved facts / tool results
4. Compact conversation memory
5. Low-signal history / raw dumps (first to drop)
```

When over budget: drop from the bottom; never drop safety policy.

---

## Token budgeting

Course package (`src.context_memory`) uses a dependency-free estimate; production can swap in `tiktoken`:

```python
from src.context_memory import SessionMemory, estimate_tokens, fit_budget

parts = [
    ("system", "You are a careful assistant."),
    ("summary", "User prefers concise answers."),
    ("history", "..." * 200),
]
kept = fit_budget(parts, budget=50)
assert kept[0][0] == "system"
print(estimate_tokens("hello world"))
```

Leave headroom for the **completion** (e.g. 10–20% of total window).

---

## Memory tiers

| Tier | Contents | Lifetime |
|------|----------|----------|
| Working | Current turn + tool results | Turn |
| Session | Rolling summary + last k turns | Session |
| User profile | Preferences, stable facts | Long-lived (explicit write) |
| World / RAG | Docs, tickets, code | External store |

### Rolling summary pattern

```python
def should_summarize(history: list[dict], max_messages: int = 20) -> bool:
    return len(history) > max_messages

SUMMARY_PROMPT = """Summarize the conversation for future turns.
Keep: user goals, decisions, constraints, open questions, names/IDs.
Drop: chit-chat, duplicate clarifications.
Max 200 words.

Transcript:
{transcript}
"""
```

Store summaries as first-class messages or side-channel state — do not rely on the model “remembering” forever.

---

## Conversational state (minimal)

```python
from dataclasses import dataclass, field

@dataclass
class SessionMemory:
    summary: str = ""
    recent: list[dict] = field(default_factory=list)  # {role, content}
    max_recent: int = 10

    def add(self, role: str, content: str) -> None:
        self.recent.append({"role": role, "content": content})
        self.recent = self.recent[-self.max_recent :]

    def build_messages(self, system: str, user: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        if self.summary:
            msgs.append({
                "role": "system",
                "content": f"Conversation summary:\n{self.summary}",
            })
        msgs.extend(self.recent)
        msgs.append({"role": "user", "content": user})
        return msgs
```

---

## Context packing for RAG

- Chunk for retrieval (semantic + structure-aware)  
- Rerank top-k  
- Deduplicate near-identical chunks  
- Cite sources with IDs the UI can open  
- Cap total retrieved tokens (Module 07 / 09)  

---

## Failure modes

| Symptom | Likely cause |
|---------|--------------|
| Ignores instructions mid-chat | History / tools outrank system |
| Contradicts earlier decision | Summary lost a constraint |
| High cost, mediocre quality | Dumping full PDFs every turn |
| Hallucinated IDs | No structured memory write |

---

## Exercise

1. Implement `SessionMemory` with forced summary every N turns.  
2. Measure tokens/turn before and after.  
3. Write a test: a constraint stated in turn 1 still appears in summary after turn 15 (simulate).

---

## Checkpoint

- [ ] You can draw your app’s memory tiers  
- [ ] You enforce a token budget in code  
- [ ] History is not your only “database”  

**Next:** [Module 06 — Fine-tuning](06-fine-tuning.md) · or skip to [Tools & RAG](07-tools-and-rag.md) if you are on the Weekend path
