from src.rag import (
    Chunk,
    TinyRAG,
    bag_of_words,
    cosine,
    filter_by_metadata,
    rrf,
    simple_chunks,
)


def test_simple_chunks():
    chunks = simple_chunks("one two three four five six", "doc", size=2)
    assert len(chunks) == 3
    assert chunks[0].id == "doc:0"
    assert chunks[0].text == "one two"


def test_cosine_identical():
    v = bag_of_words("alpha beta")
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_tiny_rag_retrieve():
    chunks = [
        Chunk("a", "cats meow and purr", "s1"),
        Chunk("b", "stock prices and markets", "s2"),
        Chunk("c", "cats sleep in sunbeams", "s1"),
    ]
    rag = TinyRAG(chunks)
    hits = rag.retrieve("cat behavior", k=2)
    assert len(hits) == 2
    assert hits[0].id in {"a", "c"}


def test_build_prompt_and_citations():
    rag = TinyRAG([Chunk("x:0", "blue sky", "note")])
    prompt = rag.build_prompt("what color is the sky?", k=1)
    assert "blue sky" in prompt
    assert rag.validate_citations("It is blue (cite: x:0).")
    assert not rag.validate_citations("It is blue (cite: evil).")


def test_rrf():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[0] in {"a", "b"}
    assert "d" in fused


def test_metadata_filter_runs_before_top_k():
    chunks = [
        Chunk(
            "refund-window",
            "Standard refund window is 30 days.",
            "policy",
            {"tier": "standard", "department": "support"},
        ),
        Chunk(
            "refund-window-enterprise",
            "Enterprise SKUs: 14 days.",
            "policy",
            {"tier": "enterprise", "department": "support"},
        ),
        Chunk(
            "payroll",
            "Payroll runs on Friday.",
            "hr",
            {"tier": "standard", "department": "hr"},
        ),
    ]
    rag = TinyRAG(chunks)
    assert rag.retrieve("refund window", k=1)[0].id == "refund-window"
    hits = rag.retrieve(
        "refund window",
        k=1,
        where={"tier": "enterprise", "department": "support"},
    )
    assert [c.id for c in hits] == ["refund-window-enterprise"]
    assert rag.retrieve("refund window", k=3, where={"department": "hr"})[0].id == "payroll"
    assert rag.retrieve("refund window", k=3, where={"department": "legal"}) == []
    assert filter_by_metadata(chunks, {}) == chunks
    hidden = Chunk("secret", "executive compensation", "hr", None)
    assert filter_by_metadata([hidden], {"department": "hr"}) == []
