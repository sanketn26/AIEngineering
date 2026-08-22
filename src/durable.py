"""Module 25 — durable coordinators, hypothesis trees, merge gates, HITL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Hypothesis:
    id: str
    claim: str
    parent_id: str | None = None
    score: float = 0.5
    status: str = "open"  # open | supported | refuted
    evidence: list[str] = field(default_factory=list)


class HypothesisTree:
    """Branching research: child evidence back-propagates to parent scores."""

    def __init__(self) -> None:
        self.nodes: dict[str, Hypothesis] = {}

    def add(self, node: Hypothesis) -> None:
        if node.parent_id and node.parent_id not in self.nodes:
            raise KeyError(f"missing parent {node.parent_id}")
        self.nodes[node.id] = node

    def record_evidence(self, node_id: str, note: str, delta: float) -> None:
        node = self.nodes[node_id]
        node.evidence.append(note)
        node.score = min(1.0, max(0.0, node.score + delta))
        if node.score >= 0.8:
            node.status = "supported"
        elif node.score <= 0.2:
            node.status = "refuted"
        self.backpropagate(node_id, delta * 0.5)

    def backpropagate(self, node_id: str, delta: float) -> None:
        node = self.nodes[node_id]
        if not node.parent_id or abs(delta) < 0.01:
            return
        parent = self.nodes[node.parent_id]
        parent.score = min(1.0, max(0.0, parent.score + delta))
        self.backpropagate(parent.id, delta * 0.5)

    def frontier(self, *, min_score: float = 0.4) -> list[Hypothesis]:
        kids = {n.parent_id for n in self.nodes.values() if n.parent_id}
        return [
            n
            for n in self.nodes.values()
            if n.id not in kids and n.status == "open" and n.score >= min_score
        ]


@dataclass
class DurableEvent:
    seq: int
    kind: str
    payload: dict[str, Any]


class DurableStore:
    """Append-only JSONL state. Restart = replay."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[DurableEvent] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw = json.loads(line)
                    self.events.append(DurableEvent(**raw))

    def append(self, kind: str, payload: dict[str, Any]) -> DurableEvent:
        ev = DurableEvent(seq=len(self.events) + 1, kind=kind, payload=payload)
        self.events.append(ev)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev)) + "\n")
        return ev

    def last(self, kind: str | None = None) -> DurableEvent | None:
        for ev in reversed(self.events):
            if kind is None or ev.kind == kind:
                return ev
        return None


class MergeGate:
    """Refuse to merge isolated work unless tests pass and a reviewer approves."""

    def review(
        self,
        *,
        tests_passed: bool,
        diff_files: list[str],
        approved: bool,
        max_files: int = 20,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if not tests_passed:
            reasons.append("tests_failed")
        if not approved:
            reasons.append("needs_approval")
        if len(diff_files) > max_files:
            reasons.append("diff_too_large")
        return {"allow": not reasons, "reasons": reasons, "files": list(diff_files)}


class Coordinator:
    """Long-running manager: persist phases, pause for humans, resume later."""

    def __init__(
        self,
        store: DurableStore,
        phases: list[str],
        workers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        if not phases:
            raise ValueError("phases must be non-empty")
        self.store = store
        self.phases = phases
        self.workers = workers

    def current_phase(self) -> str:
        last = self.store.last()
        if last is not None and last.kind == "aborted":
            return "aborted"
        ev = self.store.last("phase_done")
        if ev is None:
            return self.phases[0]
        idx = self.phases.index(ev.payload["phase"])
        if idx + 1 >= len(self.phases):
            return "done"
        return self.phases[idx + 1]

    def pending_hitl(self) -> DurableEvent | None:
        hitl = self.store.last("hitl")
        if hitl is None:
            return None
        resolved = self.store.last("hitl_resolved")
        if resolved is not None and resolved.seq > hitl.seq:
            return None
        return hitl

    def run_until_gate(self, context: dict[str, Any]) -> dict[str, Any]:
        last = self.store.last()
        if last is not None and last.kind == "aborted":
            return {"status": "denied", "phase": last.payload.get("phase")}
        pending = self.pending_hitl()
        if pending is not None:
            return {
                "status": "paused",
                "phase": pending.payload["phase"],
                "result": pending.payload.get("result"),
            }
        phase = self.current_phase()
        if phase == "done":
            return {"status": "done", "context": context}
        worker = self.workers.get(phase)
        if worker is None:
            raise KeyError(f"no worker for phase {phase}")
        result = worker(context)
        if result.get("ask_human"):
            # Do not record phase_done until a human approves.
            self.store.append(
                "hitl",
                {
                    "phase": phase,
                    "prompt": result["ask_human"],
                    "result": result,
                },
            )
            return {"status": "paused", "phase": phase, "result": result}
        self.store.append("phase_done", {"phase": phase, "result": result})
        return {"status": "continue", "phase": phase, "result": result}

    def resume(self, context: dict[str, Any], human: dict[str, Any]) -> dict[str, Any]:
        pending = self.pending_hitl()
        if pending is None:
            raise ValueError("no pending human approval")
        phase = pending.payload["phase"]
        self.store.append("hitl_resolved", {**human, "phase": phase})
        approved = bool(human.get("approved"))
        if not approved:
            self.store.append(
                "aborted",
                {"phase": phase, "reason": "denied_by_human"},
            )
            return {"status": "denied", "phase": phase}
        self.store.append(
            "phase_done",
            {"phase": phase, "result": pending.payload.get("result") or {}},
        )
        context = {**context, "human": human}
        return self.run_until_gate(context)
