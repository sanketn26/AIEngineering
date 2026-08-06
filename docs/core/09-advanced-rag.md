# Module 09 — Advanced RAG & Knowledge Systems

**Time:** 7–10 days · **Depends on:** 07–08 · **Next:** [Cost optimization](10-cost-optimization.md)

---

## Learning objectives

- Improve retrieval beyond naive top-k embedding search
- Use query rewriting, hybrid search, and reranking
- Design agentic and hierarchical retrieval patterns

## What you can build

- Production-shaped knowledge assistants
- Multi-hop research flows with intermediate queries
- Feedback loops that improve chunking/retrieval

---

## Failure modes of basic RAG

| Symptom | Fix direction |
|---------|----------------|
| Misses keyword SKUs / IDs | Hybrid BM25 + dense |
| Right doc, wrong span | Smaller chunks, late chunking, rerank |
| Multi-hop questions fail | Query decomposition / agentic RAG |
| Contradictory sources | Attribution, conflict prompts, graph/time filters |
| Stale answers | Ingestion freshness, TTL, re-embed policy |

---

## Query understanding

```python
import json

DECOMPOSE = """
Break the user question into independent search queries.
Return JSON: {{"queries": ["...", "..."], "needs_multi_hop": bool}}

Question: {q}
"""

def parse_queries(model_json: str) -> list[str]:
    data = json.loads(model_json)
    return list(data.get("queries") or [])
```

Other rewrites: HyDE (hypothetical answer → embed), step-back prompting (generalize then retrieve).

---

## Hybrid search + rerank

```text
query → dense top-N + sparse top-N → fuse (RRF) → cross-encoder rerank top-k → LLM
```

**Reciprocal Rank Fusion (RRF)** sketch:

```python
def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
```

---

## Hierarchical RAG

Index at multiple granularities:

1. **Doc-level** summaries for routing  
2. **Section-level** chunks for context  
3. **Sentence-level** for precise cites  

Route: retrieve summaries → open top documents → retrieve fine chunks only inside those docs (reduces noise).

---

## Graph-oriented RAG (when relationships matter)

Use a knowledge graph or entity index when questions are about *relationships* (“who reports to whom”, “which services depend on X”). Combine vector search for text with graph traversal for structure. Start simple (entity linking table) before full GraphRAG stacks.

---

## Agentic RAG

```text
while not done and steps < limit:
  plan next info need
  retrieve / tool call
  critique sufficiency
  answer or continue
```

Keep a hard step budget and log every retrieval for eval (Module 04).

---

## Multimodal notes

- Embed images with vision-capable models or caption-then-embed  
- Store modality metadata; don’t mix incompatible spaces without a plan  
- For PDFs: preserve layout (tables!) via proper parsers, not naive text dump  

---

## Evaluation

| Metric | Meaning |
|--------|---------|
| Hit@k / Recall@k | Gold doc in top-k? |
| MRR | Rank of first relevant |
| Faithfulness | Answer supported by context? |
| Answer relevance | On-topic? |
| Context precision | Junk in the window? |

Build a labeled set of (question, must-have doc ids, optional gold answer).

---

## Exercise

1. Add BM25 or keyword filter alongside embeddings on your corpus.  
2. Introduce a reranker (API or local cross-encoder).  
3. Create 20 multi-hop questions; measure Hit@5 before/after decomposition.

---

## Checkpoint

- [ ] You can explain hybrid search in one paragraph  
- [ ] You rerank or fuse, not only single-vector top-k  
- [ ] You measure retrieval separately from generation  

**Next:** [Module 10 — Cost optimization](10-cost-optimization.md)
