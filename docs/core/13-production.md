# Module 13 — Production-Grade Systems

**Time:** 2–3 weeks (alongside a real project) · **Depends on:** 04, 07, 10 · **Next:** [Compliance](14-compliance.md)

---

## Learning objectives

- Serve models behind stable APIs with timeouts and fallbacks
- Add observability (traces, metrics, logs)
- Automate CI checks for evals and deploys

## What you can build

- FastAPI (or similar) inference service
- Dashboarded latency/error/token metrics
- Blue/green or canary prompt versions

---

## Serving skeleton

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

app = FastAPI(title="AI Engineering Demo API")

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    user_id: str = "anonymous"

class GenerateResponse(BaseModel):
    text: str
    model: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "No model credentials configured")
    # call provider with timeout; catch and map errors
    text = f"echo:{req.prompt[:200]}"  # replace with real client
    return GenerateResponse(text=text, model="demo")
```

### Production checklist

- [ ] Timeouts on all egress  
- [ ] Retries with jitter (idempotent only)  
- [ ] Fallback model or degraded mode  
- [ ] Authn/z on public routes  
- [ ] Rate limits per user  
- [ ] Structured logging (JSON)  
- [ ] No secrets in logs  
- [ ] Horizontal scale of *stateless* app tier  

---

## Error handling

```python
import logging
import time
from typing import Callable

log = logging.getLogger("llm")


class RateLimitError(Exception):
    pass


class ProviderOutage(Exception):
    pass


def call_model_with_fallback(
    prompt: str,
    primary: Callable[[str], str],
    backup: Callable[[str], str],
    *,
    retries: int = 3,
) -> str:
    """Retry primary on rate limits; fall back on outage."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return primary(prompt)
        except RateLimitError as e:
            last_err = e
            log.warning("rate_limited attempt=%s", attempt + 1)
            time.sleep(2**attempt)
        except ProviderOutage:
            log.error("primary_outage; using backup")
            return backup(prompt)
    raise RuntimeError(f"primary failed after retries: {last_err}")
```

Map provider errors to **your** API error model (`429`, `503`, `422`).

---

## Observability

| Signal | Examples |
|--------|----------|
| Metrics | QPS, p95 latency, error rate, tokens, $ , cache hit % |
| Traces | Request → retrieve → rerank → generate |
| Logs | request_id, user_hash, template_version, model |

Tools: OpenTelemetry, Langfuse, Phoenix, Helicone, Prometheus/Grafana, provider dashboards.

---

## CI/CD

```text
PR → lint/typecheck/unit tests → golden eval subset → build image → deploy staging → smoke → prod
```

Pin **prompt versions** and **model ids** in config, not buried in code paths you cannot roll back.

---

## Docker sketch

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
ENV PORT=8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Exercise

1. Wrap one module’s feature in FastAPI with `/healthz`.  
2. Add request_id middleware and structured logs.  
3. Run a 20-case eval in CI (can be nightly if costly).

---

## Checkpoint

- [ ] Timeouts + fallback path exist  
- [ ] You can diagnose a slow request from logs/traces  
- [ ] Deploy is reproducible from git SHA  

**Next:** [Module 14 — Compliance & governance](14-compliance.md)
