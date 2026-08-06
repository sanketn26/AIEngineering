# Module 01 — Prompt Engineering Fundamentals

**Time:** 2–3 days · **Builds:** reliable single-turn apps · **Next:** [Security](02-security-privacy.md)

---

## Learning objectives

- Structure prompts with role, task, constraints, and output format
- Prefer specificity over cleverness
- Spot beginner failure modes

## What you can build

- Content generators with stable format
- Simple classifiers / extractors
- Email or ticket reply assistants

---

## Anatomy of a good prompt

| Part | Purpose | Example |
|------|---------|---------|
| **Role** | Stance and expertise | “You are a senior support engineer.” |
| **Task** | Single clear verb | “Summarize the ticket and propose next steps.” |
| **Context** | Only what is needed | Ticket text, product tier, SLA |
| **Constraints** | Safety and style | “No legal claims. Max 120 words.” |
| **Format** | Machine- or human-parseable | Bullet list / JSON schema |
| **Examples** (optional) | Few-shot anchors | 1–3 input→output pairs |

### Minimal pattern

```text
System: You are {role}. Follow policies: {policies}.

User:
Task: {task}
Context:
{context}

Constraints:
- {c1}
- {c2}

Output format:
{format}
```

### Runnable sketch (OpenAI-style)

```python
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY in env

def smart_email_responder(email_content: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap default for drills
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional email assistant. "
                    "Be concise, polite, and action-oriented."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Write a reply to this email.\n\n"
                    f"Email:\n{email_content}\n\n"
                    f"Requirements:\n"
                    f"- Acknowledge the request\n"
                    f"- Answer questions if possible\n"
                    f"- Propose a clear next step\n"
                    f"- Under 150 words"
                ),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    print(
        smart_email_responder(
            "Hi, interested in a product demo next week. Any slots Tuesday?"
        )
    )
```

Use Anthropic / Gemini / Ollama with the same **message roles** idea; only the SDK differs.

---

## Core principles

### 1. Specific beats generic

```text
# Weak
Analyze this.

# Strong
You are a data analyst. Given the CSV summary below, return:
1) three quantitative findings
2) one data-quality risk
3) two follow-up questions
Use plain language. No speculation beyond the data.
```

### 2. Context is a scarce budget

Include facts the model cannot know (internal IDs, policy, date). Exclude noise (entire ticket history when one paragraph suffices).

### 3. Control the output shape

Ask for JSON, Markdown sections, or CSV when a downstream system will parse the result. Prefer **structured outputs** / JSON schema APIs when available (Module 03).

### 4. Temperature matches the task

| Task | Temperature |
|------|-------------|
| Extraction, classification, routing | 0–0.2 |
| Support replies, summaries | 0.2–0.5 |
| Brainstorming, marketing variants | 0.7–1.0 |

---

## Common beginner mistakes

| Mistake | Fix |
|---------|-----|
| Vague verbs (“handle this”) | Name the artifact and success criteria |
| No format | Provide a template or schema |
| Overlong context | Summarize or retrieve (Module 07) |
| One prompt for all tasks | Split classify → extract → write |
| Trusting raw model text in prod | Validate + tests (Module 04) |
| Stuffing secrets into prompts | Use tools/server-side fetches |

---

## Exercise

1. Pick a real email or GitHub issue you wrote.
2. Write a system + user prompt that produces a **stable** Markdown reply with sections: Summary, Response, Risks.
3. Run it 5 times at `temperature=0.2` and `0.8`. Note variance.
4. Add one negative constraint that removes a failure mode you saw.

Log results in `PROGRESS.md`.

---

## Checkpoint

- [ ] You can name the six prompt parts from memory  
- [ ] You have one runnable script that returns structured text  
- [ ] You know when to lower temperature  

**Next:** [Module 02 — Security & privacy](02-security-privacy.md)
