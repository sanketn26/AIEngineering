# Module 14 — Legal, Compliance & Governance

**Time:** 3–5 days · **Depends on:** 02, 13 · **Next:** [Domain apps](15-domain-apps.md)

!!! warning "Not legal advice"
    This module is an engineering orientation. Consult qualified counsel for real deployments in regulated industries.

---

## Learning objectives

- Map product data flows for privacy reviews
- Implement audit trails for high-impact model actions
- Establish lightweight AI governance in a team

## What you can build

- Audit log schema for prompts/actions (with redaction)
- Data classification → model routing policy
- Model/prompt change approval checklist

---

## Frameworks you will hear about

| Area | Examples (jurisdiction-dependent) |
|------|-------------------------------------|
| Privacy | GDPR, CCPA/CPRA, sector rules |
| Healthcare | HIPAA (US) |
| Finance | SEC/FINRA recordkeeping, model risk guidance |
| Safety / AI acts | EU AI Act risk tiers, internal policies |
| IP / training data | Vendor DPAs, license constraints, customer data terms |

Engineering takeaway: **know where data goes** (logs, vendors, vector DBs, fine-tune sets) and **who can approve** changes.

---

## Audit trail pattern

```python
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json

@dataclass
class AuditEvent:
    ts: str
    actor_id: str
    action: str
    resource: str
    request_id: str
    input_hash: str
    policy_version: str
    model_id: str | None = None
    metadata: dict | None = None

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def make_event(actor: str, action: str, resource: str, raw_input: str, **kwargs) -> dict:
    ev = AuditEvent(
        ts=datetime.now(timezone.utc).isoformat(),
        actor_id=actor,
        action=action,
        resource=resource,
        request_id=kwargs.get("request_id", ""),
        input_hash=sha256_text(raw_input),
        policy_version=kwargs.get("policy_version", "v1"),
        model_id=kwargs.get("model_id"),
        metadata=kwargs.get("metadata"),
    )
    return asdict(ev)
```

Prefer **hashes + redacted snippets** over storing full sensitive prompts when regulations allow summary logs only.

---

## Data governance checklist

- [ ] Data inventory: training, RAG corpora, logs, eval sets  
- [ ] Classification labels (public / internal / confidential / restricted)  
- [ ] Retention & deletion workflows  
- [ ] Vendor subprocessors list  
- [ ] Access control (RBAC) on indexes and traces  
- [ ] Customer data never used for training without contract  

---

## Change management

Treat prompts, tools, and models like code:

1. PR with eval results  
2. Reviewer approval for high-risk surfaces  
3. Version pin in prod config  
4. Rollback path  

---

## Exercise

1. Draw a data-flow diagram for your app (user → API → model vendor → vector DB → logs).  
2. Implement append-only audit events for tool calls.  
3. Write a one-page “allowed data by model destination” table.

---

## Checkpoint

- [ ] You can list every system that stores user content  
- [ ] High-impact actions are auditable  
- [ ] Prompt/model changes are versioned  

**Next:** [Module 15 — Domain applications](15-domain-apps.md)
