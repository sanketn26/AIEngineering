"""Module 22 — trajectory evaluation for multi-step agent runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.reliability import FailureDetector, FailureMode, StepRecord


@dataclass
class Trajectory:
    run_id: str
    goal: str
    steps: list[StepRecord]
    outcome: str | None = None
    success: bool = False
    expected_commits: int = 0
    committed: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class ProcessScore:
    """How the agent worked, independent of whether the answer is right."""

    loop_violations: int
    hallucinated_tools: int
    budget_ok: bool
    avg_latency_ms: float
    steps: int
    spend_usd: float


@dataclass
class OutcomeScore:
    success: bool
    exact: bool
    notes: str = ""


@dataclass
class TrajectoryReport:
    run_id: str
    process: ProcessScore
    outcome: OutcomeScore
    detections: list[FailureMode]
    composite: float


def score_process(
    traj: Trajectory,
    *,
    detector: FailureDetector | None = None,
    budget_usd: float = 1.0,
) -> tuple[ProcessScore, list[FailureMode]]:
    det = detector or FailureDetector(cost_budget_usd=budget_usd)
    hits = det.scan(
        traj.steps,
        expected_commits=traj.expected_commits,
        committed=traj.committed,
    )
    modes = [h.mode for h in hits]
    spend = sum(s.cost_usd for s in traj.steps)
    lat = [s.latency_ms for s in traj.steps]
    return (
        ProcessScore(
            loop_violations=modes.count(FailureMode.RUNAWAY_LOOP),
            hallucinated_tools=modes.count(FailureMode.TOOL_HALLUCINATION),
            budget_ok=spend <= budget_usd,
            avg_latency_ms=(sum(lat) / len(lat)) if lat else 0.0,
            steps=len(traj.steps),
            spend_usd=round(spend, 6),
        ),
        modes,
    )


def score_outcome(
    traj: Trajectory,
    *,
    expect: str | None = None,
    grader: Callable[[str, str], bool] | None = None,
) -> OutcomeScore:
    if expect is None:
        return OutcomeScore(success=traj.success, exact=False, notes="label-only")
    actual = traj.outcome or ""
    if grader is not None:
        ok = grader(actual, expect)
        return OutcomeScore(success=ok, exact=ok)
    exact = actual.strip() == expect.strip()
    return OutcomeScore(success=exact, exact=exact)


def composite_score(process: ProcessScore, outcome: OutcomeScore) -> float:
    """Outcome-heavy, but process violations cap the grade.

    `exact` is a process-independent label from `score_outcome`; it already
    implies `success`, so the grade is success minus penalties.
    """
    base = 1.0 if outcome.success else 0.0
    penalty = 0.0
    penalty += 0.25 * process.loop_violations
    penalty += 0.35 * process.hallucinated_tools
    if not process.budget_ok:
        penalty += 0.4
    return max(0.0, min(1.0, base - penalty))


def evaluate_trajectory(
    traj: Trajectory,
    *,
    expect: str | None = None,
    detector: FailureDetector | None = None,
    budget_usd: float = 1.0,
) -> TrajectoryReport:
    process, modes = score_process(traj, detector=detector, budget_usd=budget_usd)
    outcome = score_outcome(traj, expect=expect)
    return TrajectoryReport(
        run_id=traj.run_id,
        process=process,
        outcome=outcome,
        detections=modes,
        composite=composite_score(process, outcome),
    )


def regression_delta(
    baseline: Iterable[TrajectoryReport],
    candidate: Iterable[TrajectoryReport],
    *,
    floor: float = -0.05,
) -> dict[str, Any]:
    """Compare two suites by run_id. Negative delta below floor is a regression."""
    base = {r.run_id: r for r in baseline}
    cand = {r.run_id: r for r in candidate}
    shared = sorted(set(base) & set(cand))
    if not shared:
        return {"n": 0, "mean_delta": 0.0, "regressions": [], "ok": True}
    deltas = []
    regressions = []
    for rid in shared:
        d = cand[rid].composite - base[rid].composite
        deltas.append(d)
        if d < floor:
            regressions.append(
                {
                    "run_id": rid,
                    "delta": round(d, 4),
                    "baseline": round(base[rid].composite, 4),
                    "candidate": round(cand[rid].composite, 4),
                }
            )
    mean_delta = sum(deltas) / len(deltas)
    return {
        "n": len(shared),
        "mean_delta": round(mean_delta, 4),
        "regressions": regressions,
        "ok": not regressions,
    }


def nearest_rank_percentile(sorted_vals: list[float], p: float) -> float:
    """NIST nearest-rank: rank = ceil(p * n), 1-indexed. Conservative for latency."""
    if not sorted_vals:
        return 0.0
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    n = len(sorted_vals)
    rank = max(1, min(n, math.ceil(p * n)))
    return float(sorted_vals[rank - 1])


def dashboard(reports: list[TrajectoryReport]) -> dict[str, Any]:
    """Numbers you'd put on a cost/latency/quality wall, not a vibe chart."""
    if not reports:
        return {
            "n": 0,
            "success_rate": 0.0,
            "mean_composite": 0.0,
            "mean_steps": 0.0,
            "mean_spend_usd": 0.0,
            "p95_latency_ms": 0.0,
            "budget_violations": 0,
        }
    spends = [r.process.spend_usd for r in reports]
    lats = [r.process.avg_latency_ms for r in reports]
    p95 = nearest_rank_percentile(sorted(lats), 0.95)
    return {
        "n": len(reports),
        "success_rate": round(
            sum(1 for r in reports if r.outcome.success) / len(reports), 4
        ),
        "mean_composite": round(sum(r.composite for r in reports) / len(reports), 4),
        "mean_steps": round(sum(r.process.steps for r in reports) / len(reports), 2),
        "mean_spend_usd": round(sum(spends) / len(spends), 6),
        "p95_latency_ms": round(p95, 1),
        "budget_violations": sum(1 for r in reports if not r.process.budget_ok),
    }
