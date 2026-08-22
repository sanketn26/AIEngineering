from src.agent_evals import (
    OutcomeScore,
    ProcessScore,
    Trajectory,
    TrajectoryReport,
    dashboard,
    evaluate_trajectory,
    nearest_rank_percentile,
    regression_delta,
    score_outcome,
)
from src.reliability import FailureDetector, FailureMode, StepRecord


def _traj(
    run_id: str, *, success: bool, spend: float, tool: str = "search"
) -> Trajectory:
    return Trajectory(
        run_id=run_id,
        goal="find refund window",
        steps=[
            StepRecord(
                index=0,
                decision_type="tool",
                tool_name=tool,
                args={"q": "refund"},
                cost_usd=spend,
                latency_ms=40,
            )
        ],
        outcome="30 days" if success else "unsure",
        success=success,
    )


def test_process_vs_outcome_and_dashboard():
    det = FailureDetector(known_tools={"search"}, cost_budget_usd=1.0)
    good = evaluate_trajectory(
        _traj("a", success=True, spend=0.1),
        expect="30 days",
        detector=det,
        budget_usd=1.0,
    )
    assert good.outcome.exact
    assert good.composite == 1.0
    bad = evaluate_trajectory(
        _traj("b", success=True, spend=5.0, tool="shell"),
        expect="30 days",
        detector=det,
        budget_usd=1.0,
    )
    assert FailureMode.TOOL_HALLUCINATION in bad.detections
    assert FailureMode.COST_EXPLOSION in bad.detections
    assert bad.composite < 0.5
    dash = dashboard([good, bad])
    assert dash["n"] == 2
    assert dash["budget_violations"] == 1
    assert dash["success_rate"] == 1.0  # both labeled success; process still fails


def test_regression_detection():
    det = FailureDetector(known_tools={"search"}, cost_budget_usd=1.0)
    base = [
        evaluate_trajectory(
            _traj("a", success=True, spend=0.1), expect="30 days", detector=det
        )
    ]
    worse = [
        evaluate_trajectory(
            _traj("a", success=False, spend=2.0),
            expect="30 days",
            detector=det,
            budget_usd=1.0,
        )
    ]
    delta = regression_delta(base, worse, floor=-0.05)
    assert delta["ok"] is False
    assert delta["regressions"]


def test_score_outcome_does_not_trust_self_reported_success():
    traj = _traj("c", success=True, spend=0.1)
    traj.outcome = "wrong answer"
    result = score_outcome(traj, expect="30 days")
    assert result.exact is False
    assert result.success is False


def test_p95_nearest_rank_not_low_index():
    assert nearest_rank_percentile([10.0, 1000.0], 0.95) == 1000.0
    slow = ProcessScore(0, 0, True, 1000.0, 1, 0.0)
    fast = ProcessScore(0, 0, True, 10.0, 1, 0.0)
    ok = OutcomeScore(True, True)

    def report(rid: str, proc: ProcessScore) -> TrajectoryReport:
        return TrajectoryReport(rid, proc, ok, [], 1.0)

    dash = dashboard([report("a", fast), report("b", slow)])
    assert dash["p95_latency_ms"] == 1000.0
