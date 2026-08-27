from src.harness import (
    ExternalState,
    HarnessSpec,
    load_progress,
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


def test_second_window_resumes_from_progress_file(tmp_path):
    progress = tmp_path / "progress.json"
    spec = _spec(step_cap=1)
    check = lambda a: verify_artifact(a, must_contain=("REFUND",))

    def session_one(_state: ExternalState) -> dict:
        return {"tool": "write_note", "artifact": "draft", "cost_usd": 0.01}

    first = run_harness(
        spec, propose=session_one, verify=check, progress_path=progress
    )
    assert first.verified is False
    assert progress.exists()

    saved = load_progress(progress)
    assert saved.artifacts["last"] == "draft"

    def session_two(state: ExternalState) -> dict:
        assert state.artifacts["last"] == "draft"
        return {
            "tool": "write_note",
            "artifact": "REFUND window is 30 days",
            "cost_usd": 0.01,
        }

    second = run_harness(
        spec,
        propose=session_two,
        verify=check,
        state=saved,
        progress_path=progress,
    )
    assert second.stopped == "verified"
    assert load_progress(progress).artifacts["last"].startswith("REFUND")


def test_denied_tool_does_not_write_artifact():
    def propose(_state: ExternalState) -> dict:
        return {"tool": "bash", "artifact": "rm -rf", "cost_usd": 0.01}

    state = ExternalState()
    run_harness(
        _spec(step_cap=2),
        propose=propose,
        verify=lambda a: verify_artifact(a, must_contain=("REFUND",)),
        state=state,
    )
    assert "last" not in state.artifacts
