# Module 10 — Cost Optimization & Economics

**Time:** 2–3 days · **Depends on:** 01, 05, 07 · **Next:** [Single agents](11-single-agents.md)

---

## Learning objectives

- Measure unit economics (cost per successful task)
- Apply routing, caching, and prompt compression
- Set budgets and alerts before scale

## What you can build

- Model router (SLM / mini / full)
- Response cache with safe keys
- Token usage dashboards

---

## Cost drivers

| Driver | Levers |
|--------|--------|
| Input tokens | Shorter prompts, summaries, better retrieval (less junk) |
| Output tokens | Strict formats, max_tokens, “be concise” |
| Model tier | Mini/SLM for easy tasks |
| Retries / tools | Cap steps; cache tool results |
| Embeddings | Batch; don’t re-embed unchanged docs |

**Unit metric:** `cost_per_success = total_$ / successful_tasks` — optimize this, not raw token thrift that tanks quality.

---

## Token helpers

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))
```

Track **prompt + completion** separately in your logger.

---

## Model routing

```python
class ModelRouter:
    def __init__(self, cheap: str, strong: str):
        self.cheap = cheap
        self.strong = strong

    def pick(self, task: str, prompt: str) -> str:
        if task in {"classify", "route", "extract_fields"}:
            return self.cheap
        if task == "complex_reason" or len(prompt) > 8000:
            return self.strong
        return self.cheap

# Example mapping (verify current names/pricing):
# cheap = "gpt-4o-mini" | "claude-haiku" | local "llama3.2"
# strong = "gpt-4o" | "claude-sonnet-4-..." | "gemini-2.5-pro"
```

Escalate to strong models only on low-confidence or failed validation.

---

## Caching

```python
import hashlib
import json
import time
from typing import Any

class MemoryCache:
    def __init__(self, ttl_s: int = 3600):
        self.ttl_s = ttl_s
        self.store: dict[str, tuple[float, Any]] = {}

    def _key(self, namespace: str, payload: str) -> str:
        h = hashlib.sha256(payload.encode()).hexdigest()
        return f"{namespace}:{h}"

    def get(self, namespace: str, payload: str) -> Any | None:
        k = self._key(namespace, payload)
        item = self.store.get(k)
        if not item:
            return None
        exp, val = item
        if time.time() > exp:
            del self.store[k]
            return None
        return val

    def set(self, namespace: str, payload: str, value: Any) -> None:
        k = self._key(namespace, payload)
        self.store[k] = (time.time() + self.ttl_s, value)
```

Cache keys must include **model id + prompt template version + critical params**. Do not cache personalized/sensitive outputs in shared layers without isolation.

For production: Redis, provider prompt caching (where available), HTTP caches for idempotent tools.

---

## Spend guardrails

```python
from collections import defaultdict

class UsageLedger:
    def __init__(self):
        self.by_user: dict[str, float] = defaultdict(float)

    def add(self, user_id: str, cost_usd: float) -> None:
        self.by_user[user_id] += cost_usd

    def allowed(self, user_id: str, limit_usd: float) -> bool:
        return self.by_user[user_id] < limit_usd
```

Pair with provider hard limits and anomaly alerts (sudden 10× traffic).

---

## Prompt economy checklist

- [ ] System prompt stable → enable provider prefix caching if offered  
- [ ] Retrieve fewer, better chunks  
- [ ] Summarize long tool outputs before re-feeding  
- [ ] Batch classification jobs  
- [ ] Prefer extractive answers when possible  

---

## Exercise

1. Log tokens and estimated $ for 50 real requests.  
2. Route 50% of “easy” traffic to a mini/local model; measure quality delta.  
3. Add cache hit-rate to your log line.

---

## Checkpoint

- [ ] You know cost per successful task for one flow  
- [ ] At least one of: routing, caching, budget cap  
- [ ] Quality gate still enforced after savings  

**Next:** [Module 11 — Single-agent workflows](11-single-agents.md)
