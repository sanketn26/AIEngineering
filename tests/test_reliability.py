from src.reliability import (
    CircuitBreaker,
    FailureDetector,
    FailureMode,
    SpendGuard,
    StepRecord,
    state_checksum,
)


def _tool(i: int, name: str, **args) -> StepRecord:
    return StepRecord(
        index=i,
        decision_type="tool",
        tool_name=name,
        args=args,
        cost_usd=0.2,
        tokens_in=10,
        tokens_out=5,
    )


def test_detects_runaway_and_cost_and_hallucination():
    steps = [_tool(0, "search", q="x"), _tool(1, "search", q="x"), _tool(2, "nope")]
    det = FailureDetector(
        max_repeat=2, cost_budget_usd=0.3, known_tools={"search", "final"}
    )
    modes = {d.mode for d in det.scan(steps)}
    assert FailureMode.RUNAWAY_LOOP in modes
    assert FailureMode.COST_EXPLOSION in modes
    assert FailureMode.TOOL_HALLUCINATION in modes


def test_partial_and_silent_and_state():
    det = FailureDetector(quality_floor=0.8, cost_budget_usd=10)
    hits = det.scan(
        [],
        expected_commits=2,
        committed=0,
        quality_score=0.4,
        state_ok=False,
    )
    modes = {h.mode for h in hits}
    assert FailureMode.PARTIAL_EXECUTION in modes
    assert FailureMode.SILENT_DEGRADATION in modes
    assert FailureMode.STATE_CORRUPTION in modes


def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(fail_max=2, cooldown_s=10)
    assert cb.allow(0)
    cb.record_failure(1)
    assert cb.allow(2)
    cb.record_failure(3)
    assert cb.state == CircuitBreaker.OPEN
    assert cb.allow(4) is False
    assert cb.allow(14) is True
    assert cb.state == CircuitBreaker.HALF_OPEN
    assert cb.allow(15) is False
    assert cb.allow(16) is False
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED
    assert cb.allow(17) is True


def test_spend_guard_and_checksum():
    g = SpendGuard(1.0)
    assert g.allow(0.4)
    g.charge(0.4)
    assert g.remaining() == 0.6
    assert g.allow(0.7) is False
    a = state_checksum({"a": 1, "b": [2]})
    b = state_checksum({"b": [2], "a": 1})
    assert a == b
