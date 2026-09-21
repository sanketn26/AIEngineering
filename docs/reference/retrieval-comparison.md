---
description: Compare four retrieval paths, then catch an answer whose citation exists but whose claim does not.
---

# The citation was real. The answer was wrong.

*Fictional teaching scenario.*

A customer asks about order **A17**. The assistant returns a neatly cited paragraph
about **A71**. Every citation validator passes. The document exists. The quotation
is faithful. The customer still has the wrong answer.

Three questions have become tangled: did we retrieve useful evidence, does the
answer follow from that evidence, and does it answer the question? This experiment
gives each question its own number.

**Use after:** [Module 09](../core/09-advanced-rag.md). Allow 60–90 minutes,
plus optional model downloads.

## Make four predictions

The fixture contains six policy/order documents and twelve labeled queries. Some
use synonyms, two contain easily confused order IDs, and two have no answer.

1. Which path should preserve **A17** better: a keyword score or a semantic vector?
2. What happens when “parcel” must find a document that says “shipping”?
3. Can a reranker recover a document that never entered its candidate pool?
4. Will a faithful quotation necessarily answer an unanswerable question?

Run the controlled experiment first:

```bash
python -m examples.retrieval.compare --output /tmp/retrieval-fixture.json
```

| Path | What runs |
|---|---|
| Sparse | BM25 term matching |
| Dense | Cosine over three hand-authored semantic features |
| Hybrid | Reciprocal rank fusion of the two rankings |
| Reranked | Top four hybrid results ordered by annotated support |

The hybrid step combines **rank positions**, because a BM25 score and a cosine
similarity are different units:

```python
def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for position, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (k + position)
    return sorted(scores, key=scores.get, reverse=True)

hybrid = rrf([sparse_ranking, dense_ranking])
rerank_candidates = hybrid[:4]  # the reranker cannot recover rank 5
```

The default vectors and reranker are **transparent test fixtures**, not learned
models. The reranker deliberately consults annotations; it tests candidate-pool and
metric wiring. Its score is not evidence of a real model's quality. That distinction
is why the report names its mode prominently.

## Let real models surprise you

In a separate environment, install `examples/retrieval/requirements.txt`. The real
path uses `SentenceTransformer` embeddings and a `CrossEncoder` reranker. The same
corpus, query labels, candidate budget, and metrics stay fixed.

```bash
python -m examples.retrieval.compare --real \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-revision REPLACE_WITH_MODEL_COMMIT_SHA \
  --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --reranker-revision REPLACE_WITH_MODEL_COMMIT_SHA \
  --output /tmp/retrieval-real.json
```

Find each commit SHA in the revision history of the [embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) before running.
The first real run downloads weights; CPU execution is supported. Save the package
versions and hardware alongside the report. Query timing excludes model loading;
the fixture is too small for capacity or production quality claims.

Now compare your predictions with the rankings. If fusion buries an exact identifier,
inspect both lists before adding more machinery. If reranking fails, first ask
whether the right document survived the top-four cut. A brilliant reviewer cannot
choose a page that never reached their desk.

## One answer, three checks

| Metric | What it answers | What it cannot prove |
|---|---|---|
| Hit@1, Recall@2, MRR | Did relevant labeled sources rise to the top? | That the generated statement is supported |
| `source_support` | Does each annotated claim occur in its cited, retrieved document? | That it answers this question |
| `answers_question` | Does the first extractive answer contain the requested labeled fact? | General natural-language correctness |

The answer in this experiment copies the first document. Source support can therefore
be perfect while question relevance is poor. That is the point: a high grounding
score can coexist with an unhelpful assistant.

The support checker uses a small, closed vocabulary of human-labeled atomic claims.
Try citing `refund-window` while claiming `unused_window_365_days`; an existing ID
must not rescue the invented claim. This is executable evidence beyond ID validity,
not a universal entailment checker. For free-form answers, label claim/evidence pairs
and evaluate a human-calibrated verifier separately.

```python
def claims_supported(claims, documents, retrieved_ids):
    allowed = {
        doc["id"]: set(doc["claims"])
        for doc in documents
        if doc["id"] in retrieved_ids
    }
    return bool(claims) and all(
        claim["claim"] in allowed.get(claim["cite"], set())
        for claim in claims
    )

assert not claims_supported(
    [{"cite": "refund-window", "claim": "unused_window_365_days"}],
    documents,
    retrieved_ids=["refund-window"],
)
```

That assertion says something narrow and valuable: the cited document does not
contain the annotated claim. It does not turn string membership into a general
faithfulness metric.

Unanswerable queries have no positive document label, so Recall@k is undefined for
them and is reported as `null`. They remain in the relevance results. The first run
deliberately forces retrieval to expose the error; your job is to add abstention.
Tune its threshold on development cases, then test on fresh answerable and
unanswerable cases. Raw cosine is not a probability.

## Close the case

**Artifact:** one table comparing all four paths on a frozen query set, with
candidate budgets, retrieval metrics, answer support, relevance, latency, and model
revisions. Include an A17/A71 failure trace and an unanswerable query that abstains.
Use new documents and at least 30 independent questions for your assessed version;
the shipped twelve are a microscope, not a benchmark.

**Check:** `pytest tests/test_retrieval_experiment.py -q`.

Implementation references: [SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) and [CrossEncoder](https://sbert.net/docs/package_reference/cross_encoder/model.html). The latter scores query-document pairs jointly; its scores rank candidates rather than serving as calibrated confidence.
