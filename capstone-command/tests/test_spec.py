"""The starter's honest behavior. Comments name the gate that changes each assertion."""

from pathlib import Path

from runtime.capability import eligible_for_4b
from runtime.execute import execute
from runtime.models import TaskSpec
from runtime.policy import MAX_REPAIRS, MOCK_PATCH, repeated_candidate, verification_plan
from runtime.state import State, can_transition
from runtime.validate import load_yaml, validate_document

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_sample_spec_is_structurally_ready():
    result = validate_document(load_yaml(FIXTURES / "retry_api_client.yaml"))
    assert result["status"] == "SPEC_READY"
    assert result["errors"] == []


def test_absolute_and_parent_paths_are_rejected():
    document = load_yaml(FIXTURES / "retry_api_client.yaml")
    document["scope"]["allowed_files"] = ["../secrets/env"]
    result = validate_document(document)
    assert result["status"] == "SPEC_INVALID"
    assert result["errors"][0]["code"] == "UNSAFE_PATH"


def test_unknown_fields_and_dangling_requirement_refs_are_rejected():
    document = load_yaml(FIXTURES / "retry_api_client.yaml")
    document["surprise"] = True
    assert validate_document(document)["status"] == "SPEC_INVALID"

    document = load_yaml(FIXTURES / "retry_api_client.yaml")
    document["acceptance_criteria"][0]["maps_to"] = ["R99"]
    dangling = validate_document(document)
    assert dangling["status"] == "SPEC_INVALID"
    assert dangling["errors"][0]["code"] == "UNKNOWN_REQUIREMENT_REF"


def test_open_unknowns_still_count_as_ready():
    # Gate 1: a non-empty unknowns list must become SPEC_INCOMPLETE, and code.patch must not run.
    result = validate_document(load_yaml(FIXTURES / "open_unknowns.yaml"))
    assert result["status"] == "SPEC_READY"


def test_uncovered_must_requirement_still_validates():
    # Gate 2: R3 has no acceptance criterion. That must become MISSING_ACCEPTANCE_COVERAGE.
    document = load_yaml(FIXTURES / "retry_api_client.yaml")
    document["acceptance_criteria"] = [
        item for item in document["acceptance_criteria"] if item["id"] != "AC3"
    ]
    assert validate_document(document)["status"] == "SPEC_READY"


def test_free_form_intent_still_returns_a_candidate():
    # Gate 1: a sentence may draft a spec. It must not return a patch.
    outcome = execute("Add retry support to the API client.")
    assert outcome["status"] == "candidate"
    assert outcome["command"] == "code.patch"


def test_draft_command_still_patches_and_policy_accepts_the_extra_file():
    # Gate 1: spec.draft stops after a contract. Gate 4: setup.py is outside the allowlist.
    document = load_yaml(FIXTURES / "retry_api_client.yaml")
    document["command"] = "spec.draft"
    outcome = execute(document)
    assert outcome["status"] == "candidate"
    assert outcome["policy"] == "PASS"
    assert "setup.py" in MOCK_PATCH
    assert outcome["revision"] is None


def test_verification_is_a_shell_string_and_repairs_are_unbounded():
    # Gate 4: runner is an allowlist, the repair cap is 2, and a repeated hash escalates.
    spec = validate_document(load_yaml(FIXTURES / "retry_api_client.yaml"))["spec"]
    plan = verification_plan(spec.acceptance_criteria[0])
    assert plan["shell"] is True
    assert plan["command"].startswith("pytest")
    assert MAX_REPAIRS > 2
    assert repeated_candidate({"same"}, "same") is False


def test_risk_label_overrides_the_envelope():
    # Gate 4: allow_new_dependencies and a high line budget cannot be waived by risk_level: low.
    document = load_yaml(FIXTURES / "retry_api_client.yaml")
    document["constraints"]["allow_new_dependencies"] = True
    document["scope"]["max_added_lines"] = 500
    document["risk_level"] = "low"
    spec = TaskSpec.model_validate(document)
    assert eligible_for_4b(spec) is True


def test_draft_can_start_generating():
    # Gate 1: there is no edge from draft to generating.
    assert can_transition(State.DRAFT, State.GENERATING) is True
