import pytest

from src.orchestrators import (
    CostAttribution,
    CostEvent,
    TraceRecorder,
    compare_orchestrators,
    tradeoff_score,
)


def test_compare_and_weights():
    rows = compare_orchestrators("custom loop", "LangGraph", "CrewAI", "MCP hosts")
    assert len(rows) == 4
    custom = next(r for r in rows if r["name"] == "custom loop")
    graph = next(r for r in rows if r["name"] == "LangGraph")
    # custom wins control; LangGraph wins durable
    assert custom["control"] < graph["control"]
    assert graph["durable"] < custom["durable"]
    score = tradeoff_score(graph, {"hitl": 2.0, "durable": 1.0})
    assert score == pytest.approx(1.0)


def test_cost_attribution_and_traces():
    ledger = CostAttribution()
    ledger.record(
        CostEvent(
            agent="researcher",
            step=0,
            model="mini",
            tokens_in=100,
            tokens_out=40,
            usd=0.01,
            latency_ms=80,
            tool="search",
        )
    )
    ledger.record(
        CostEvent(
            agent="writer",
            step=0,
            model="strong",
            tokens_in=400,
            tokens_out=200,
            usd=0.09,
            latency_ms=400,
        )
    )
    by = ledger.by_agent()
    assert by["researcher"]["usd"] == 0.01
    assert ledger.total_usd() == 0.1
    rec = TraceRecorder()
    rec.span("decide", "researcher", tool="search")
    rec.span("final", "writer")
    exported = rec.export()
    assert exported[0]["agent"] == "researcher"
    assert len(exported) == 2
    rec.spans[0].attrs["name"] = "clobber"
    rec.spans[0].attrs["agent"] = "other"
    out = rec.export()[0]
    assert out["name"] == "decide"
    assert out["agent"] == "researcher"
