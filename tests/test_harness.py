from src.harness import (
    ExternalState,
    HarnessSpec,
    run_harness,
    verify_artifact,
)


def _spec(**kwargs) -> HarnessSpec:
    base = dict(
        name="triage",
        instructions="Cite policy. Stop when the note has REFUND.",
        tools=("write_note", "lookup_policy"),
        step_cap=4,
        cost_cap_usd=0.5,
    )
    base.update(kwargs)
    return HarnessSpec(**base)


def test_missing_verifier_fails_closed():
    report = run_harness(_spec(), propose=lambda s: {"tool": "write_note"})
    assert report.stopped == "no_verifier"
    assert report.verified is False


def test_unknown_tool_is_denied_and_hits_step_cap():
    def propose(_state: ExternalState) -> dict:
        return {"tool": "bash", "artifact": "oops", "cost_usd": 0.01}

    report = run_harness(
        _spec(),
        propose=propose,
        verify=lambda a: verify_artifact(a, must_contain=("REFUND",)),
    )
    assert report.verified is False
    assert report.stopped == "step_cap"
    assert any(n.startswith("denied:") for n in report.notes)


def test_verify_success_stops_before_cap():
    def propose(state: ExternalState) -> dict:
        n = len(state.notes)
        text = "draft" if n == 0 else "REFUND window is 30 days"
        return {"tool": "write_note", "artifact": text, "cost_usd": 0.01}

    report = run_harness(
        _spec(),
        propose=propose,
        verify=lambda a: verify_artifact(a, must_contain=("REFUND",)),
    )
    assert report.stopped == "verified"
    assert report.verified is True
    assert report.steps == 2
    assert report.steps < 4


def test_cost_cap_stops_even_if_unverified():
    def propose(_state: ExternalState) -> dict:
        return {
            "tool": "write_note",
            "artifact": "draft",
            "cost_usd": 0.4,
        }

    report = run_harness(
        _spec(cost_cap_usd=0.5),
        propose=propose,
        verify=lambda a: verify_artifact(a, must_contain=("REFUND",)),
    )
    assert report.stopped == "cost_cap"
    assert report.verified is False
