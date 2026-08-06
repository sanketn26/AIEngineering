from src.rag import Chunk, TinyRAG, bag_of_words, cosine, rrf, simple_chunks


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
