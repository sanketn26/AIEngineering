"""Module 25 — durable coordinators, hypothesis trees, merge gates, HITL."""

from __future__ import annotations

import json
import os
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
    """Single-writer JSONL journal; one newline-terminated event is a transition.

    fsync before acknowledging an append. On restart, discard only an unfinished
    final record; corruption in a committed record fails closed. This is not a
    multi-worker database or an exactly-once executor for external side effects.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.events: list[DurableEvent] = []
        self._failed = False
        self.recovered_tail_bytes = 0
        if self.path is not None and self.path.exists():
            data = self.path.read_bytes()
            end = data.rfind(b"\n") + 1
            # Parse first: never silently repair corruption in a complete record.
            for line in data[:end].splitlines():
                if not line.strip():
                    continue
                ev = DurableEvent(**json.loads(line))
                if ev.seq != len(self.events) + 1:
                    raise ValueError("journal sequence is corrupt")
                self.events.append(ev)
            self.recovered_tail_bytes = len(data) - end
            if self.recovered_tail_bytes:
                with self.path.open("r+b") as f:
                    f.truncate(end)
                    f.flush()
                    os.fsync(f.fileno())

    def append(self, kind: str, payload: dict[str, Any]) -> DurableEvent:
        if self._failed:
            raise RuntimeError("append failed; reopen the journal before continuing")
        # Snapshot mutable caller data and validate serialization before writing.
        ev = DurableEvent(len(self.events) + 1, kind, json.loads(json.dumps(payload)))
        record = (json.dumps(asdict(ev)) + "\n").encode("utf-8")
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            created = not self.path.exists()
            try:
                with self.path.open("ab") as f:
                    f.write(record)
                    f.flush()
                    os.fsync(f.fileno())
                # Persist a newly created directory entry on POSIX too.
                if created and os.name == "posix":
                    fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            except OSError:
                self._failed = True
                raise
        self.events.append(ev)
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

    def _denial(self) -> DurableEvent | None:
        """First event that ended the run, if any."""
        for ev in self.store.events:
            if ev.kind == "aborted" or (
                ev.kind == "hitl_resolved" and not ev.payload.get("approved")
            ):
                return ev
        return None

    def current_phase(self) -> str:
        if self._denial() is not None:
            return "aborted"
        completed: set[str] = set()
        for ev in self.store.events:
            if ev.kind == "hitl_resolved":
                # Old journals may hold any truthy 'approved'; new ones hold bools.
                completed.add(ev.payload["phase"])
            elif ev.kind == "phase_done":
                completed.add(ev.payload["phase"])
        if all(phase in completed for phase in self.phases):
            return "done"
        return next(phase for phase in self.phases if phase not in completed)

    def restored_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Recover completed results without letting caller data override history."""
        results: dict[str, Any] = {}
        restored = dict(context)
        for ev in self.store.events:
            if ev.kind == "phase_done" or (
                ev.kind == "hitl_resolved" and ev.payload.get("approved")
            ):
                results[ev.payload["phase"]] = json.loads(
                    json.dumps(ev.payload.get("result", {}))
                )
            if ev.kind == "hitl_resolved":
                restored["human"] = {"approved": ev.payload.get("approved")}
        restored["phase_results"] = results
        return restored

    def pending_hitl(self) -> DurableEvent | None:
        hitl = self.store.last("hitl")
        if hitl is None:
            return None
        resolved = self.store.last("hitl_resolved")
        if resolved is not None and resolved.seq > hitl.seq:
            return None
        return hitl

    def run_until_gate(self, context: dict[str, Any]) -> dict[str, Any]:
        denial = self._denial()
        if denial is not None:
            return {"status": "denied", "phase": denial.payload.get("phase")}
        context = self.restored_context(context)
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
        approved = human.get("approved")
        if type(approved) is not bool:
            raise ValueError("approved must be a boolean")
        # One authoritative event: a crash cannot split 'resolved' from its
        # consequence. Old two-record journals replay through current_phase too.
        self.store.append(
            "hitl_resolved",
            {
                "phase": phase,
                "approved": approved,
                "result": pending.payload.get("result") or {},
            },
        )
        if not approved:
            return {"status": "denied", "phase": phase}
        return self.run_until_gate(context)
