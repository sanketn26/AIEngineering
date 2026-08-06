# Module 15 — Domain-Specific Applications

**Time:** 1–2 weeks (patterning, not full vertical certification) · **Depends on:** 02, 07, 14

!!! warning
    Domain examples are **illustrative**. Do not deploy medical, legal, or financial advice systems without licensed professionals, validation, and compliance review.

---

## Learning objectives

- Map domain constraints onto the core stack (security, RAG, audit)
- Build vertical prototypes with fail-closed behavior
- Separate “product UX” from “regulated decisioning”

## Pattern: every vertical needs

1. **Policy layer** — what the model may not do  
2. **Knowledge layer** — approved sources only  
3. **Action layer** — tools with approvals  
4. **Evidence layer** — citations + audit  
5. **Human loop** — escalation paths  

---

## Healthcare-shaped assistant (pattern only)

| Concern | Engineering response |
|---------|----------------------|
| PHI | Redact/minimize; BAA with vendors; access logs |
| Safety | No diagnosis claims; emergency escalation copy |
| Knowledge | Curated guidelines; cite sources; date-stamp |
| Audit | Who asked what (hashed), what was retrieved |

```python
def medical_style_answer(question: str, redacted: str, llm) -> str:
    prompt = f"""You are an informational assistant, not a clinician.
Do not diagnose or prescribe.
Encourage professional care for personal medical decisions.
Question: {redacted}
"""
    return llm(prompt)
```

---

## Finance-shaped assistant (pattern only)

| Concern | Engineering response |
|---------|----------------------|
| Market data | Tools with timestamps; never invent prices |
| Advice boundaries | Disclaimers; suitability is human process |
| Records | Retain prompts/outputs per policy |
| Risk | Separate education vs. trade execution tools |

```python
def finance_style_answer(question: str, quote: dict | None, llm) -> str:
    ctx = f"Quote as of {quote['ts']}: {quote}" if quote else "No live quote available."
    return llm(
        f"Educational only, not investment advice.\n{ctx}\nQuestion: {question}"
    )
```

---

## Legal document helper (pattern only)

- Privilege & confidentiality → private deployments / strict vendors  
- Jurisdiction awareness → user-supplied jurisdiction field  
- Outputs are **drafts** for attorney review, not filings  

---

## Research / literature assistant

- Prefer API access to papers (arXiv, publisher APIs) over scraped PDFs when possible  
- Store bibliographic metadata with chunks  
- Separate “summarize this PDF” from “what does the field conclude?” (latter needs broader retrieval + caution)

---

## Build-your-own vertical

1. Stakeholders & prohibited outputs  
2. Source-of-truth systems  
3. Eval set with domain expert labels  
4. Escalation UX  
5. Monitoring for policy violations  

---

## Exercise

Pick one domain. Write a **one-page policy** + a **10-case eval set** (including 3 “must refuse” cases). Implement refuse-and-escalate for those three.

---

## Checkpoint

- [ ] Domain policy is written down  
- [ ] Must-refuse cases are tested  
- [ ] Citations/audit exist for knowledge answers  

**Next:** [Module 16 — Integration patterns](16-integration-patterns.md)
