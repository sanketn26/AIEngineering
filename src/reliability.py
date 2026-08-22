"""Module 20 — agent failure taxonomy, detectors, and circuit breakers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class FailureMode(str, Enum):
    RUNAWAY_LOOP = "runaway_loop"
    TOOL_HALLUCINATION = "tool_hallucination"
    STATE_CORRUPTION = "state_corruption"
    PARTIAL_EXECUTION = "partial_execution"
    COST_EXPLOSION = "cost_explosion"
    SILENT_DEGRADATION = "silent_degradation"


@dataclass
class StepRecord:
    """One decide/act/observe tick, independent of any framework."""

    index: int
    decision_type: str
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    observation: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None


@dataclass
class Detection:
    mode: FailureMode
    reason: str
    step_index: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def tool_signature(name: str, args: dict[str, Any] | None) -> str:
    payload = json.dumps(args or {}, sort_keys=True, default=str)
    return f"{name}:{payload}"


class FailureDetector:
    """Scan a finished (or in-flight) trajectory for the six teaching modes."""

    def __init__(
        self,
        *,
        max_repeat: int = 2,
        cost_budget_usd: float = 1.0,
        quality_floor: float = 0.7,
        known_tools: Iterable[str] | None = None,
    ) -> None:
        self.max_repeat = max_repeat
        self.cost_budget_usd = cost_budget_usd
        self.quality_floor = quality_floor
        self.known_tools = set(known_tools or [])

    def scan(
        self,
        steps: list[StepRecord],
        *,
        expected_commits: int = 0,
        committed: int = 0,
        quality_score: float | None = None,
        state_ok: bool = True,
    ) -> list[Detection]:
        found: list[Detection] = []
        found.extend(self._runaway(steps))
        found.extend(self._hallucinated_tools(steps))
        if not state_ok:
            found.append(
                Detection(
                    FailureMode.STATE_CORRUPTION,
                    "state checksum or schema failed",
                    evidence={"steps": len(steps)},
                )
            )
        if expected_commits > committed:
            found.append(
                Detection(
                    FailureMode.PARTIAL_EXECUTION,
                    "committed fewer side effects than the plan required",
                    evidence={
                        "expected": expected_commits,
                        "committed": committed,
                    },
                )
            )
        spend = sum(s.cost_usd for s in steps)
        if spend > self.cost_budget_usd:
            found.append(
                Detection(
                    FailureMode.COST_EXPLOSION,
                    "trajectory exceeded USD budget",
                    evidence={
                        "spend_usd": round(spend, 6),
                        "budget": self.cost_budget_usd,
                    },
                )
            )
        if quality_score is not None and quality_score < self.quality_floor:
            found.append(
                Detection(
                    FailureMode.SILENT_DEGRADATION,
                    "quality below floor with no hard abort",
                    evidence={"quality": quality_score, "floor": self.quality_floor},
                )
            )
        return found

    def _runaway(self, steps: list[StepRecord]) -> list[Detection]:
        sigs = [
            tool_signature(s.tool_name, s.args)
            for s in steps
            if s.decision_type == "tool" and s.tool_name
        ]
        counts = Counter(sigs)
        out: list[Detection] = []
        for sig, n in counts.items():
            if n >= self.max_repeat:
                out.append(
                    Detection(
                        FailureMode.RUNAWAY_LOOP,
                        "repeated tool signature",
                        evidence={"signature": sig, "count": n},
                    )
                )
        return out

    def _hallucinated_tools(self, steps: list[StepRecord]) -> list[Detection]:
        if not self.known_tools:
            return []
        out: list[Detection] = []
        for s in steps:
            if s.decision_type == "tool" and s.tool_name not in self.known_tools:
                out.append(
                    Detection(
                        FailureMode.TOOL_HALLUCINATION,
                        "model proposed a tool that is not allowlisted",
                        step_index=s.index,
                        evidence={"name": s.tool_name},
                    )
                )
        return out


class CircuitBreaker:
    """Closed → open after `fail_max` consecutive failures; one half-open probe."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_max: int = 3, cooldown_s: float = 30.0) -> None:
        if fail_max < 1:
            raise ValueError("fail_max must be >= 1")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        self.fail_max = fail_max
        self.cooldown_s = cooldown_s
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self._probe_outstanding = False

    def allow(self, now: float) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if now - self.opened_at >= self.cooldown_s:
                self.state = self.HALF_OPEN
                self._probe_outstanding = True
                return True
            return False
        # half-open: at most one in-flight probe until success/failure is recorded
        if self._probe_outstanding:
            return False
        self._probe_outstanding = True
        return True

    def record_success(self) -> None:
        self.failures = 0
        self._probe_outstanding = False
        self.state = self.CLOSED

    def record_failure(self, now: float) -> None:
        self.failures += 1
        self._probe_outstanding = False
        if self.state == self.HALF_OPEN or self.failures >= self.fail_max:
            self.state = self.OPEN
            self.opened_at = now


def state_checksum(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


class SpendGuard:
    """Hard stop when estimated USD spend would exceed a remaining budget."""

    def __init__(self, budget_usd: float) -> None:
        if budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        self.budget_usd = budget_usd
        self.spent = 0.0

    def remaining(self) -> float:
        return max(0.0, self.budget_usd - self.spent)

    def allow(self, estimated_usd: float) -> bool:
        if estimated_usd < 0:
            raise ValueError("estimated_usd must be >= 0")
        return self.spent + estimated_usd <= self.budget_usd

    def charge(self, actual_usd: float) -> None:
        if actual_usd < 0:
            raise ValueError("actual_usd must be >= 0")
        self.spent += actual_usd
        if self.spent > self.budget_usd:
            raise RuntimeError("spend exceeded budget")
