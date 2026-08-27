"""Module 27 — harness engineering: the control layer around the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HarnessSpec:
    """The software around one model loop: instructions, tools, caps, verifier."""

    name: str
    instructions: str
    tools: tuple[str, ...]
    step_cap: int = 8
    cost_cap_usd: float = 1.0
    verifier_required: bool = True


@dataclass
class ExternalState:
    """Progress the harness persists. The model does not own this."""

    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str


@dataclass(frozen=True)
class HarnessReport:
    stopped: str  # verified | step_cap | cost_cap | denied_all | no_verifier
    steps: int
    cost_usd: float
    verified: bool
    notes: tuple[str, ...]


def verify_artifact(
    artifact: str, *, must_contain: tuple[str, ...] = ()
) -> VerifyResult:
    missing = [s for s in must_contain if s not in artifact]
    if missing:
        return VerifyResult(False, "missing " + ",".join(missing))
    if not artifact.strip():
        return VerifyResult(False, "empty")
    return VerifyResult(True, "ok")


def run_harness(
    spec: HarnessSpec,
    *,
    propose: Callable[[ExternalState], dict[str, Any]],
    verify: Callable[[str], VerifyResult] | None = None,
    state: ExternalState | None = None,
) -> HarnessReport:
    """Model proposes; harness authorizes, persists, verifies, and stops."""
    if spec.verifier_required and verify is None:
        return HarnessReport("no_verifier", 0, 0.0, False, ())
    state = state or ExternalState()
    cost = 0.0
    taken = 0
    for _ in range(spec.step_cap):
        proposal = propose(state)
        tool = str(proposal.get("tool") or "")
        if tool not in spec.tools:
            state.notes.append(f"denied:{tool or 'missing'}")
            taken += 1
            continue
        cost += float(proposal.get("cost_usd") or 0.0)
        taken += 1
        if cost > spec.cost_cap_usd:
            return HarnessReport(
                "cost_cap", taken, cost, False, tuple(state.notes)
            )
        artifact = str(proposal.get("artifact") or "")
        if artifact:
            state.artifacts["last"] = artifact
            state.notes.append("wrote")
        if verify is not None:
            result = verify(state.artifacts.get("last", ""))
            if result.ok:
                return HarnessReport(
                    "verified", taken, cost, True, tuple(state.notes)
                )
            state.notes.append(f"verify_fail:{result.reason}")
    stopped = "step_cap" if taken else "denied_all"
    return HarnessReport(stopped, taken, cost, False, tuple(state.notes))
