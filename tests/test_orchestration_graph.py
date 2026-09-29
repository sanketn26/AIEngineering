import pytest

from src.orchestration_graph import (
    PlanError,
    check_edge,
    fanout_pays,
    independence_fraction,
    layered_batches,
    loop_until_dry,
    reduce_findings,
    scale_gate,
    speedup,
    verifier_input,
)


def test_speedup_matches_the_ceiling():
    assert independence_fraction([True, False, False]) == pytest.approx(2 / 3)
    assert speedup(0.95, 16) == pytest.approx(9.142857, rel=1e-5)
    assert speedup(0.70, 16) == pytest.approx(2.909090, rel=1e-5)
    assert fanout_pays(0.95, 16) is True
    assert fanout_pays(0.70, 16) is False


def test_reduce_is_code_and_the_skeptic_sees_only_the_finding():
    reduced = reduce_findings(
        [
            [{"source": "a", "claim": "30 days"}, {"source": "b", "claim": "ssl"}],
            [{"source": "a", "claim": "duplicate"}],
        ]
    )
    assert [item["source"] for item in reduced] == ["a", "b"]
    assert reduced[0]["claim"] == "30 days"
    payload = verifier_input({"source": "a"})
    assert payload == {"finding": {"source": "a"}}
    assert "transcript" not in payload
    with pytest.raises(PlanError, match="transcript"):
        verifier_input({"source": "a"}, transcript="I already reasoned that this is true")


def test_layered_fan_in_never_hands_the_raw_pile_across():
    batches = layered_batches(list(range(100)), 40)
    assert [len(batch) for batch in batches] == [40, 40, 20]
    assert max(len(batch) for batch in batches) <= 40


def test_seen_set_is_written_on_discovery():
    rounds = [["a", "a"], ["a"], []]

    def find_round(i: int) -> list[str]:
        return rounds[i] if i < len(rounds) else []

    report = loop_until_dry(find_round, max_dry=2, token_budget=100, cost_per_round=1)
    assert report["confirmed"] == ["a"]
    assert report["seen"] == {"a"}
    assert report["dry"] == 2
    assert report["iterations"] == 3


def test_token_budget_stops_a_loop_that_never_goes_dry():
    report = loop_until_dry(
        lambda i: [f"bug-{i}"],
        max_dry=5,
        max_iter=20,
        token_budget=100,
        cost_per_round=40,
    )
    assert report["iterations"] == 2
    assert report["spent"] == 80
    assert report["confirmed"] == ["bug-0", "bug-1"]


def test_plan_check_refuses_illegal_edges_and_allows_a_legal_one():
    with pytest.raises(PlanError, match="transcript"):
        check_edge({"kind": "verify", "includes_transcript": True})
    with pytest.raises(PlanError, match="raw pile"):
        check_edge({"kind": "synthesize", "item_count": 1000, "window_limit": 40})
    with pytest.raises(PlanError, match="two writers"):
        check_edge({"kind": "write", "path": "api.py", "owned_paths": ["api.py"]})
    with pytest.raises(PlanError, match="seen-set"):
        check_edge({"kind": "cycle", "seen_on_discovery": False})
    with pytest.raises(PlanError, match="does not pay"):
        check_edge({"kind": "fanout", "p": 0.70, "n": 16})
    with pytest.raises(PlanError, match="schema"):
        check_edge({"kind": "map", "schema": None})
    check_edge({"kind": "fanout", "p": 0.95, "n": 16})
    check_edge({"kind": "verify", "includes_transcript": False})


def test_scale_gate_needs_all_three_yeses():
    assert scale_gate(found_new=True, verifier_caught=True, cost_justified=True) is True
    assert scale_gate(found_new=True, verifier_caught=True, cost_justified=False) is False
