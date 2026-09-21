"""Today's behavior. Comments name the gate that has to change the assertion."""

from pathlib import Path

from runtime.execute import MOCK_PLAN, join_parts, route_part, run_task
from runtime.models import Part, Seam
from runtime.validate import load_yaml, validate_document, accept_plan

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _spec():
    return validate_document(load_yaml(FIXTURES / "refund_task.yaml"))["spec"]


def test_user_spec_is_ready_and_paths_are_rejected():
    assert validate_document(load_yaml(FIXTURES / "refund_task.yaml"))["status"] == "SPEC_READY"
    document = load_yaml(FIXTURES / "refund_task.yaml")
    document["scope"]["allowed_files"] = ["../secrets/env"]
    result = validate_document(document)
    assert result["status"] == "SPEC_INVALID"
    assert result["errors"][0]["code"] == "UNSAFE_PATH"


def test_file_outside_the_allowlist_is_rejected():
    plan = {
        "parts": [
            {
                "id": "P1",
                "owns": ["R1"],
                "checks": ["V1"],
                "files": ["unrelated.py"],
                "depends_on": [],
                "exports": [],
                "imports": [],
            }
        ]
    }
    verdict = accept_plan(_spec(), plan)
    assert verdict["status"] == "PLAN_INVALID"
    assert verdict["errors"][0]["code"] == "PATH_NOT_ALLOWED"


def test_mock_plan_drops_a_requirement_and_is_still_accepted():
    # Gate 2: R3 has no owner and P3 cites V-looks-right. Both must be plan errors.
    verdict = accept_plan(_spec(), MOCK_PLAN)
    assert verdict["status"] == "PLAN_READY"
    owned = {req for part in verdict["plan"].parts for req in part.owns}
    assert "R3" not in owned
    assert any("V-looks-right" in part.checks for part in verdict["plan"].parts)


def test_missing_seam_and_a_cycle_are_still_accepted():
    # Gate 1: record_refund is not exported, and a cycle must be CYCLE.
    verdict = accept_plan(_spec(), MOCK_PLAN)
    assert verdict["status"] == "PLAN_READY"
    assert "record_refund" in verdict["plan"].parts[2].imports

    cycled = {
        "parts": [
            {**MOCK_PLAN["parts"][0], "depends_on": ["P2"]},
            {**MOCK_PLAN["parts"][1], "id": "P2", "depends_on": ["P1"]},
        ]
    }
    assert accept_plan(_spec(), cycled)["status"] == "PLAN_READY"


def test_run_is_still_one_shot_done():
    # Gate 1: the one-shot result is a baseline failure, not the solution.
    outcome = run_task(_spec())
    assert outcome["mode"] == "one_shot"
    assert outcome["model"] == "20b"
    assert outcome["status"] == "done"
    assert outcome["user_checks_ran"] == []


def test_join_skips_user_join_checks():
    # Gate 4: V4 must run, and a missing seam must be SEAM_MISMATCH rather than done.
    joined = join_parts(_spec(), MOCK_PLAN)
    assert joined["status"] == "done"
    assert joined["join_checks_ran"] == []
    assert joined["seam_conflicts"] == []


def test_a_too_wide_part_stays_on_20b():
    # Gate 3: two failures and too_wide select 32b for this part only.
    part = Part(
        id="P1",
        owns=["R1"],
        checks=["V1"],
        files=["ledger.py"],
        exports=[Seam(name="apply_once", kind="function", signature="(proposal_id: str) -> LedgerEntry")],
    )
    assert route_part(part, too_wide=True, failures=2) == "20b"
