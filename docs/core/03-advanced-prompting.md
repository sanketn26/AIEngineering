# Module 03 — Advanced Prompting Techniques

**Time:** 3–5 days · **Depends on:** 01–02 · **Next:** [Testing & evals](04-testing-evals.md)

---

## Learning objectives

- Apply CoT, few-shot, role, and self-consistency patterns deliberately
- Use structured outputs for reliable parsing
- Build reusable prompt templates without a mega-framework

## What you can build

- Multi-step reasoners with checkable intermediate form
- Extractors that return validated JSON
- Template libraries for product surfaces

---

## Chain-of-Thought (CoT)

Ask for **reasoning then answer** — or hide reasoning and require a final tagged answer.

```text
Solve the problem carefully.
Put the final answer after a line that says FINAL:
If uncertain, say UNCERTAIN and list what is missing.
```

Use CoT for math, multi-hop questions, policy decisions — not for pure extraction (adds cost and sometimes noise).

---

## Few-shot learning

Show 1–5 **exemplars** closest to the live task. Prefer **diverse** edge cases over many near-duplicates.

```text
Extract company and amount as JSON.

Example 1:
Input: Acme Corp invoiced us $12,400 on March 3.
Output: {"company":"Acme Corp","amount":12400,"currency":"USD"}

Example 2:
Input: Payment of EUR 900 to Globex pending.
Output: {"company":"Globex","amount":900,"currency":"EUR"}

Input: {user_text}
Output:
```

**Dynamic few-shot:** embed exemplars; retrieve top-k by similarity to the query (bridge to RAG).

---

## Role-based prompting

Roles work when they constrain **style, priorities, and refusal boundaries** — not as magic credentials.

```text
You are a staff SRE. Prefer blast-radius reduction over cleverness.
Never invent metrics you do not have. Ask for missing graphs.
```

---

## Self-consistency (high-stakes answers)

1. Sample N solutions at moderate temperature  
2. Parse final answers  
3. Majority vote / cluster  

Costly — reserve for hard questions after a cheap first pass.

---

## Structured outputs

Prefer provider **JSON schema / structured output** modes when available. Fallback: strict instructions + validator.

```python
from pydantic import BaseModel, ValidationError
import json

class Invoice(BaseModel):
    company: str
    amount: float
    currency: str

def parse_invoice(raw: str) -> Invoice:
    # Prefer provider structured output; this is the fallback path
    data = json.loads(raw)
    return Invoice.model_validate(data)

def safe_parse(raw: str) -> Invoice | None:
    try:
        return parse_invoice(raw)
    except (json.JSONDecodeError, ValidationError):
        return None
```

---

## Lightweight template system

```python
from string import Template

TEMPLATES = {
    "summarize": Template(
        "Summarize for a busy $audience in $bullets bullets.\n\nText:\n$content"
    ),
    "classify": Template(
        "Classify the text into one of: $labels.\n"
        "Return JSON: {\"label\": string, \"confidence\": number}\n\nText:\n$content"
    ),
}

def render(name: str, **kwargs) -> str:
    return TEMPLATES[name].safe_substitute(**kwargs)
```

Promote templates to versioned files (`prompts/v1/summarize.md`) and test them (Module 04).

---

## Tree-of-Thought / multi-path (when useful)

For planning problems: generate multiple candidate plans → score → expand best. Implement as **explicit steps in code**, not a single magical prompt, so you can log and test each stage.

---

## Technique selection

| Problem | Technique |
|---------|-----------|
| Format reliability | Schema / structured out |
| Domain tone | Role + 2–3 shots |
| Hard reasoning | CoT or multi-step tools |
| Ambiguous label | Self-consistency or human review |
| Long doc | Context engineering / RAG (05, 07) |

---

## Exercise

1. Convert a free-text extraction prompt to **Pydantic-validated JSON**.  
2. Add two few-shot edge cases (missing currency; multiple companies).  
3. Measure parse-success rate over 20 samples (manual is fine).

---

## Checkpoint

- [ ] You can justify CoT vs. direct answer for a task  
- [ ] At least one output path is schema-validated  
- [ ] Templates live outside ad-hoc string soup  

**Next:** [Module 04 — Testing & evals](04-testing-evals.md)
