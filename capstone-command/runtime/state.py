"""Legal transitions. Gate 1 removes the edge from draft straight to generating."""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    DRAFT = "draft"
    SPEC_INVALID = "spec_invalid"
    READY = "ready"
    GENERATING = "generating"
    CANDIDATE_INVALID = "candidate_invalid"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    SUCCEEDED = "succeeded"
    ESCALATED = "escalated"
    FAILED = "failed"


# GATE 1 — planted: DRAFT may enter GENERATING. A frozen READY state has to come first.
_EDGES: dict[State, set[State]] = {
    State.DRAFT: {State.SPEC_INVALID, State.READY, State.GENERATING},
    State.READY: {State.GENERATING},
    State.GENERATING: {State.CANDIDATE_INVALID, State.VERIFYING, State.FAILED},
    State.VERIFYING: {State.SUCCEEDED, State.REPAIRING, State.ESCALATED, State.FAILED},
    State.REPAIRING: {State.GENERATING, State.ESCALATED},
}


def can_transition(source: State, destination: State) -> bool:
    return destination in _EDGES.get(source, set())
