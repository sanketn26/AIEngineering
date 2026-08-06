# Module 16 — Advanced Integration Patterns

**Time:** 1–2 weeks · **Depends on:** 13 · **Next:** [Small models](17-small-models.md)

---

## Learning objectives

- Embed LLMs into event-driven and microservice architectures
- Design hybrid cloud / on-prem routing by data class
- Choose sync vs async vs batch generation paths

## What you can build

- Queue-backed generation workers
- Hybrid router (sensitive → on-prem; bulk → cloud)
- LLM microservice with gateway policies

---

## Event-driven pipeline

```text
Producer → Kafka/SQS/PubSub topic → worker pool → results topic → consumers
```

```python
# Conceptual worker
def handle_message(msg: dict, llm) -> dict:
    prompt = msg["prompt"]
    # enforce schema, authz, budget
    text = llm(prompt)
    return {"id": msg["id"], "text": text, "model": msg.get("model")}
```

**Why:** absorb spikes, retry poison messages, scale workers independently of the API tier.

---

## Sync vs async vs batch

| Mode | Use |
|------|-----|
| Sync HTTP | Chat UX, low latency, small work |
| Async job | Long agents, multi-doc analysis |
| Batch | Overnight classification, embedding rebuilds |

Expose `POST /jobs` → `GET /jobs/{id}` for long work; never block a public request for 10-minute agent runs without streaming/job UX.

---

## Hybrid cloud / on-prem

```python
def route_endpoint(is_sensitive: bool, need_gpu: bool) -> str:
    if is_sensitive:
        return "onprem-vllm"
    if need_gpu:
        return "cloud-gpu"
    return "cloud-mini"
```

Pair with private networking, CMEK/customer keys when required, and explicit data-classification tags on every request.

---

## Microservice boundaries

```text
API gateway → orchestration service → {retriever, tool service, generator}
```

- Keep **generator** stateless  
- Own **retrieval** as a separate SLI (hit rate, latency)  
- Version **tool contracts** like public APIs  

FastAPI / gRPC sketches from Module 13 apply; add service mesh or API gateway for auth and quotas.

---

## Streaming

For chat UX, stream tokens (SSE/WebSocket). Propagate `request_id` through the stream; handle client disconnect (cancel upstream when possible).

---

## Exercise

1. Move a sync generate endpoint to a queue worker; return job ids.  
2. Tag requests with `data_class=public|confidential` and route differently.  
3. Load-test p95 latency and error rate; document scaling knobs.

---

## Checkpoint

- [ ] Long work is async  
- [ ] Data class influences routing  
- [ ] Contracts between services are versioned  

**Next:** [Module 17 — Small & local models](17-small-models.md)
