"""Sparse, dense, hybrid, and reranked retrieval on the same labeled queries.

Default: transparent hand-authored semantic vectors and an oracle-like fixture
reranker, solely to test the pipeline. --real loads actual embedding/reranking
models. Neither mode claims a general retrieval leaderboard result.
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from src.rag import bag_of_words, rrf

ROOT = Path(__file__).parent


def bm25(query, documents):
    query = set(bag_of_words(query))
    bags = [bag_of_words(d["text"]) for d in documents]
    lengths = [sum(b.values()) for b in bags]
    average = sum(lengths) / len(lengths)
    scores = []
    for doc, bag, length in zip(documents, bags, lengths):
        score = 0
        for word in query:
            df = sum(word in b for b in bags)
            idf = math.log(1 + (len(bags) - df + 0.5) / (df + 0.5))
            tf = bag.get(word, 0)
            score += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * length / average))
        scores.append((doc["id"], score))
    return [key for key, score in sorted(scores, key=lambda pair: (-pair[1], pair[0]))]


def fixture_vector(text):
    # Intentionally visible ontology: this is NOT a learned embedding model.
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    groups = [
        {"refund", "reimburse", "money", "duplicate", "charged", "billing"},
        {"delivery", "shipping", "parcel", "package", "tracking"},
        {"password", "login", "locked", "account", "access"},
    ]
    return [float(bool(words & group)) for group in groups]


def vector_cosine(a, b):
    denom = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0


def claims_supported(
    claims: list[dict], documents: list[dict], retrieved: list[str]
) -> bool:
    """Closed-world support check, stronger than recognizing a citation ID.

    Claim keys are human-authored annotations. Do not apply this exact-match
    checker to arbitrary natural-language entailment and call it solved.
    """
    allowed = {d["id"]: set(d["claims"]) for d in documents if d["id"] in retrieved}
    return bool(claims) and all(
        c["claim"] in allowed.get(c["cite"], set()) for c in claims
    )


def run(
    real=False,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2",
    embedding_revision=None,
    reranker_revision=None,
):
    data = json.loads((ROOT / "cases.json").read_text())
    docs, queries = data["documents"], data["queries"]
    by_id = {d["id"]: d for d in docs}
    encoder = reranker = None
    if real:
        from sentence_transformers import SentenceTransformer, CrossEncoder

        encoder = SentenceTransformer(embedding_model, revision=embedding_revision)
        reranker = CrossEncoder(reranker_model, revision=reranker_revision)
        doc_vectors = encoder.encode(
            [d["text"] for d in docs], normalize_embeddings=True
        ).tolist()
    else:
        doc_vectors = [fixture_vector(d["text"]) for d in docs]
    rows = []
    for query in queries:
        start = time.perf_counter()
        sparse = bm25(query["text"], docs)
        vector = (
            encoder.encode([query["text"]], normalize_embeddings=True).tolist()[0]
            if real
            else fixture_vector(query["text"])
        )
        dense = [
            d["id"]
            for d, _ in sorted(
                zip(docs, doc_vectors),
                key=lambda pair: (-vector_cosine(vector, pair[1]), pair[0]["id"]),
            )
        ]
        hybrid = rrf([sparse, dense])
        candidates = hybrid[:4]
        if real:
            scores = reranker.predict(
                [(query["text"], by_id[key]["text"]) for key in candidates]
            )
        else:
            # Controlled stand-in: annotated support, never presented as a model.
            scores = [
                float(query["claim"] in by_id[key]["claims"]) for key in candidates
            ]
        ranked = [
            key for key, _ in sorted(zip(candidates, scores), key=lambda pair: -pair[1])
        ]
        for name, ranking in {
            "sparse": sparse,
            "dense": dense,
            "hybrid": hybrid,
            "reranked": ranked,
        }.items():
            gold = set(query["must_have"])
            top = ranking[:2]
            rank = next((i + 1 for i, key in enumerate(ranking) if key in gold), None)
            # An extractive answer copies the first result; evaluate both source
            # support and whether it contains the requested annotated fact.
            first = by_id[ranking[0]]
            claims = [
                {"claim": claim, "cite": first["id"]} for claim in first["claims"]
            ]
            rows.append(
                {
                    "query": query["id"],
                    "method": name,
                    "hit_at_1": int(ranking[0] in gold) if gold else None,
                    "recall_at_2": len(gold & set(top)) / len(gold) if gold else None,
                    "mrr": 1 / rank if rank else (0 if gold else None),
                    "source_support": claims_supported(claims, docs, top),
                    "answers_question": query["claim"] in first["claims"],
                    "ranking": ranking,
                    "answer": first["text"],
                }
            )
        rows[-1]["whole_query_ms"] = (time.perf_counter() - start) * 1000
    summary = {}
    for method in ("sparse", "dense", "hybrid", "reranked"):
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {}
        for metric in (
            "hit_at_1",
            "recall_at_2",
            "mrr",
            "source_support",
            "answers_question",
        ):
            values = [r[metric] for r in selected if r[metric] is not None]
            summary[method][metric] = sum(values) / len(values)
    return {
        "mode": "real_models" if real else "controlled_fixture_not_a_model_benchmark",
        "embedding_model": embedding_model if real else "hand_authored_3d_vectors",
        "reranker_model": reranker_model if real else "annotated_support_fixture",
        "embedding_revision": embedding_revision,
        "reranker_revision": reranker_revision,
        "n_queries": len(queries),
        "summary": summary,
        "rows": rows,
        "limitation": "Unanswerable queries deliberately expose forced retrieval; support does not imply relevance. Model loading excluded from query timing.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    parser.add_argument("--embedding-revision")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.real and (not args.embedding_revision or not args.reranker_revision):
        parser.error(
            "pin both model revisions (commit SHAs) for a reproducible comparison"
        )
    result = run(
        args.real,
        args.embedding_model,
        args.reranker_model,
        args.embedding_revision,
        args.reranker_revision,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
