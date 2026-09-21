from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

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
    assert c.run_until_gate({}) == {"status": "denied", "phase": "research"}
    assert writes["n"] == 0


def test_merge_gate():
    g = MergeGate()
    blocked = g.review(tests_passed=False, diff_files=["a.py"], approved=True)
    assert blocked["allow"] is False
    ok = g.review(tests_passed=True, diff_files=["a.py"], approved=True)
    assert ok["allow"] is True


def test_torn_tail_is_removed_before_next_append(tmp_path):
    path = tmp_path / "events.jsonl"
    DurableStore(path).append("hitl", {"phase": "review"})
    with path.open("ab") as stream:
        stream.write(b'{"seq":2,"kind":"hitl_resolved"')
    recovered = DurableStore(path)
    assert recovered.recovered_tail_bytes > 0
    recovered.append("hitl_resolved", {"phase": "review", "approved": False})
    assert len(DurableStore(path).events) == 2


def test_complete_corrupt_event_fails_closed(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"{broken}\n")
    with pytest.raises(json.JSONDecodeError):
        DurableStore(path)
    assert path.read_bytes() == b"{broken}\n"


@pytest.mark.parametrize("approved", [False, True])
def test_process_dies_immediately_after_approval_commit(tmp_path, approved):
    path = tmp_path / "events.jsonl"
    DurableStore(path).append("hitl", {"phase": "review", "result": {"facts": ["a"]}})
    script = """
import os, sys
from pathlib import Path
from src.durable import DurableStore
s = DurableStore(Path(sys.argv[1]))
s.append("hitl_resolved", {"phase":"review", "approved":sys.argv[2] == "True", "result":{"facts":["a"]}})
os._exit(73)
"""
    proc = subprocess.run([sys.executable, "-c", script, str(path), str(approved)])
    assert proc.returncode == 73
    writes = []

    def forbidden_review(ctx):
        pytest.fail("completed/rejected review ran again")

    def write(ctx):
        writes.append(ctx["phase_results"]["review"]["facts"])
        return {"ok": True}

    coordinator = Coordinator(
        DurableStore(path),
        ["review", "write"],
        {"review": forbidden_review, "write": write},
    )
    outcome = coordinator.run_until_gate({})
    assert outcome["status"] == ("continue" if approved else "denied")
    assert writes == ([["a"]] if approved else [])


def test_failed_fsync_does_not_advance_memory(tmp_path, monkeypatch):
    store = DurableStore(tmp_path / "events.jsonl")

    def fail(_fd):
        raise OSError("disk failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        store.append("phase_done", {"phase": "review"})
    assert store.events == []
    with pytest.raises(RuntimeError):
        store.append("phase_done", {"phase": "review"})


def test_string_false_is_not_approval():
    store = DurableStore()
    store.append("hitl", {"phase": "review"})
    coordinator = Coordinator(store, ["review"], {})
    with pytest.raises(ValueError):
        coordinator.resume({}, {"approved": "false"})


def test_journal_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    store = DurableStore(path)
    store.append("phase_done", {"phase": "a"})
    with path.open("a") as f:
        f.write("\n")
    store.append("phase_done", {"phase": "b"})
    assert [e.seq for e in DurableStore(path).events] == [1, 2]


def test_old_journal_truthy_approval_replays(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    rows = [
        {"seq": 1, "kind": "hitl", "payload": {"phase": "research"}},
        {"seq": 2, "kind": "hitl_resolved", "payload": {"approved": "yes", "phase": "research"}},
        {"seq": 3, "kind": "phase_done", "payload": {"phase": "research"}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = Coordinator(DurableStore(path), ["research", "write"], {})
    assert c.current_phase() == "write"
