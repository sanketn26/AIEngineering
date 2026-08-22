from pathlib import Path

from src.durable import Coordinator, DurableStore, Hypothesis, HypothesisTree, MergeGate


def test_hypothesis_backprop():
    tree = HypothesisTree()
    tree.add(Hypothesis(id="root", claim="vendor is late"))
    tree.add(Hypothesis(id="c1", claim="invoice date mismatch", parent_id="root"))
    tree.record_evidence("c1", "line 12 vs PO", 0.4)
    assert tree.nodes["c1"].status == "supported"
    assert tree.nodes["root"].score > 0.5


def test_durable_store_roundtrip(tmp_path: Path):
    p = tmp_path / "state.jsonl"
    s = DurableStore(p)
    s.append("phase_done", {"phase": "research", "result": {"ok": True}})
    s2 = DurableStore(p)
    assert s2.last("phase_done").payload["phase"] == "research"


def test_coordinator_pauses_for_human():
    store = DurableStore()

    def research(ctx):
        return {"facts": ["a"], "ask_human": "approve write?"}

    def write(ctx):
        return {"ok": True}

    c = Coordinator(
        store, ["research", "write"], {"research": research, "write": write}
    )
    paused = c.run_until_gate({})
    assert paused["status"] == "paused"
    resumed = c.resume({}, {"approved": True})
    assert resumed["phase"] == "write"


def test_coordinator_denial_does_not_run_next_phase():
    store = DurableStore()
    writes = {"n": 0}

    def research(ctx):
        return {"facts": ["a"], "ask_human": "approve write?"}

    def write(ctx):
        writes["n"] += 1
        return {"ok": True}

    c = Coordinator(
        store, ["research", "write"], {"research": research, "write": write}
    )
    assert c.run_until_gate({})["status"] == "paused"
    denied = c.resume({}, {"approved": False})
    assert denied["status"] == "denied"
    assert denied["phase"] == "research"
    assert writes["n"] == 0
    assert c.current_phase() == "aborted"
    assert c.run_until_gate({})["status"] == "denied"
    assert writes["n"] == 0


def test_merge_gate():
    g = MergeGate()
    blocked = g.review(tests_passed=False, diff_files=["a.py"], approved=True)
    assert blocked["allow"] is False
    ok = g.review(tests_passed=True, diff_files=["a.py"], approved=True)
    assert ok["allow"] is True
