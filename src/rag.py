"""Module 07 — minimal retrieve-then-generate helpers (no embedding deps)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str


def simple_chunks(text: str, source: str, size: int = 50) -> list[Chunk]:
    """Chunk by word count (size = words per chunk)."""
    words = text.split()
    if not words:
        return [Chunk(id=f"{source}:0", text="", source=source)]
    out: list[Chunk] = []
    for i in range(0, len(words), size):
        piece = " ".join(words[i : i + size])
        out.append(Chunk(id=f"{source}:{i}", text=piece, source=source))
    return out


def bag_of_words(text: str) -> dict[str, float]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    tf: dict[str, float] = {}
    for t in toks:
        tf[t] = tf.get(t, 0.0) + 1.0
    return tf


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over ranked id lists."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class TinyRAG:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        self.vecs = [bag_of_words(c.text) for c in self.chunks]
        self._by_id = {c.id: c for c in self.chunks}

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        if not self.chunks:
            return []
        qv = bag_of_words(query)
        scored = sorted(
            zip(self.chunks, self.vecs),
            key=lambda cv: cosine(qv, cv[1]),
            reverse=True,
        )
        return [c for c, _ in scored[:k]]

    def retrieve_ids(self, query: str, k: int = 3) -> list[str]:
        return [c.id for c in self.retrieve(query, k=k)]

    def build_prompt(self, query: str, k: int = 3) -> str:
        docs = self.retrieve(query, k=k)
        blocks = "\n\n".join(f"[{c.id}] (source={c.source})\n{c.text}" for c in docs)
        return (
            "Answer using only the sources. Cite chunk ids.\n"
            "If sources are insufficient, say you do not know.\n\n"
            f"Sources:\n{blocks}\n\nQuestion: {query}"
        )

    def validate_citations(
        self, answer: str, allowed_ids: set[str] | None = None
    ) -> bool:
        """True if every (cite: id) in answer is in allowed set (default: corpus)."""
        allowed = allowed_ids if allowed_ids is not None else set(self._by_id)
        cites = re.findall(r"\(cite:\s*([^)]+)\)", answer)
        if not cites:
            return True
        return all(c.strip() in allowed for c in cites)
