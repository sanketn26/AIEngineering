# Module 07 — Tool Integration & Basic RAG

**Time:** 5–7 days · **Depends on:** 01–05 · **Next:** [MCP](08-model-context-protocol.md)

---

## Learning objectives

- Implement tool/function calling safely
- Build a minimal retrieve → generate pipeline
- Choose tools vs. parametric knowledge vs. RAG

## What you can build

- Assistants that call calculators, HTTP APIs, or DB queries
- Doc Q&A over a small corpus with citations
- Hybrid “router” that picks knowledge source

---

## When to use what

| Need | Mechanism |
|------|-----------|
| Live data (price, weather, ticket) | **Tool / API** |
| Private/static corpus | **RAG** |
| General reasoning / style | **Model weights** |
| Strict enterprise actions | Tool + **human approval** |

---

## Tool calling loop

```text
User → Model (may request tool)
         ↓
      Execute tool in your runtime (allowlisted)
         ↓
      Tool result → Model → Final answer
```

```python
import json
from typing import Callable

ToolFn = Callable[..., str]

TOOLS: dict[str, ToolFn] = {
    "get_weather": lambda city: f"Weather in {city}: 22C clear (demo)",
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

def run_tool(name: str, arguments_json: str) -> str:
    if name not in TOOLS:
        return json.dumps({"error": "tool not allowed"})
    args = json.loads(arguments_json or "{}")
    # Validate keys rigorously in production (JSON Schema)
    return TOOLS[name](**args)
```

Wire `TOOL_SPECS` into your provider’s tools/functions API; loop until the model stops requesting tools or hits a step limit.

**Safety:** allowlist tools, sanitize args, timeouts, and never expose shell as an open tool without sandboxing.

---

## Basic RAG pipeline

```text
Ingest → Chunk → Embed → Index
User query → Embed → Retrieve top-k → Prompt with sources → Answer + citations
```

### Chunking heuristics

| Content | Starting point |
|---------|----------------|
| Prose docs | 400–800 tokens, 10–20% overlap |
| Code | By symbol / file section |
| Tables | Keep row groups together |

### Minimal in-memory sketch

Shipped as `src.rag` (see `tests/test_rag.py`):

```python
from src.rag import TinyRAG, simple_chunks

chunks = simple_chunks("Cats sleep in sunbeams. Markets move on news.", "notes", size=6)
rag = TinyRAG(chunks)
print(rag.retrieve("cat sleep", k=1)[0].text)
print(rag.build_prompt("Where do cats sleep?", k=1))
assert rag.validate_citations("They sleep in sunbeams (cite: notes:0).")
```

Replace bag-of-words with **sentence-transformers**, **OpenAI embeddings**, or provider embeddings + **FAISS / Chroma / Qdrant / Pinecone**.

---

## Citation pattern

```text
Answer in Markdown.
After each claim that uses a source, add (cite: chunk_id).
End with a Sources section listing ids → titles.
```

---

## Exercise

1. Index 5–10 of your own notes or READMEs.  
2. Ask questions that are answerable and unanswerable; confirm “I don’t know” works.  
3. Add one tool (e.g. `get_time` or HTTP GET to a public API) with an allowlist.

---

## Checkpoint

- [ ] Tool execution happens in *your* code, not free-form model text alone  
- [ ] RAG answers cite sources  
- [ ] Unanswerable queries fail closed  

**Next:** [Module 08 — Model Context Protocol](08-model-context-protocol.md)
