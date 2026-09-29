import pytest

from src.answer_contract import decide, score_contract, validate_decision


def _rows():
    return [
        {
            "id": "ok-standard",
            "policy_window": 30,
            "tool_window": 30,
            "citations": ["refund-window"],
            "retrieved": ["refund-window"],
            "missing_slot": None,
            "expect": {"decision": "answer"},
        },
        {
            "id": "conflict-enterprise",
            "policy_window": 30,
            "tool_window": 14,
            "citations": ["refund-window"],
            "retrieved": ["refund-window"],
            "missing_slot": None,
            "expect": {"decision": ["abstain", "escalate"], "must_not_cite": "refund-window"},
        },
        {
            "id": "missing-order",
            "policy_window": None,
            "tool_window": None,
            "citations": [],
            "retrieved": [],
            "missing_slot": "Which order?",
            "expect": {"decision": "clarify"},
        },
        {
            "id": "empty",
            "policy_window": None,
            "tool_window": None,
            "citations": [],
            "retrieved": [],
            "missing_slot": None,
            "expect": {"decision": "abstain"},
        },
        {
            "id": "false-escalation",
            "policy_window": 30,
            "tool_window": 30,
            "citations": ["refund-window"],
            "retrieved": ["refund-window"],
            "missing_slot": None,
            "expect": {"decision": "answer"},
        },
    ]


def _predict(row):
    return decide(
        policy_window=row["policy_window"],
        tool_window=row["tool_window"],
        citations=row["citations"],
        retrieved=row["retrieved"],
        missing_slot=row["missing_slot"],
    )


def test_contract_passes_the_five_slices():
    report = score_contract(_rows(), _predict)
    assert report["failures"] == []
    assert report["passed"] == 5


def test_always_citing_the_policy_fails_the_conflict():
    def cite_policy(row):
        return {
            "decision": "answer",
            "question": None,
            "citations": ["refund-window"],
            "packet": {"retrieved": row["retrieved"]},
        }

    report = score_contract(_rows(), cite_policy)
    failed = {item["id"] for item in report["failures"]}
    assert "conflict-enterprise" in failed
    assert "ok-standard" not in failed


def test_always_escalating_fails_a_settled_refund():
    def escalate(_row):
        return {
            "decision": "escalate",
            "question": None,
            "citations": [],
            "packet": {"conflict": "reflex"},
        }

    report = score_contract(_rows(), escalate)
    failed = {item["id"] for item in report["failures"]}
    assert "false-escalation" in failed
    assert "ok-standard" in failed
    assert "conflict-enterprise" not in failed


def test_answer_without_citations_is_illegal():
    with pytest.raises(ValueError, match="citations"):
        validate_decision(
            {"decision": "answer", "citations": [], "question": None, "packet": {}}
        )


def test_unresolved_citation_is_illegal():
    with pytest.raises(ValueError, match="unresolved citation"):
        validate_decision(
            {
                "decision": "answer",
                "question": None,
                "citations": ["invented"],
                "packet": {"retrieved": ["refund-window"]},
            }
        )
    with pytest.raises(ValueError, match="citations must be ids"):
        validate_decision(
            {
                "decision": "answer",
                "question": None,
                "citations": [12],
                "packet": {"retrieved": ["refund-window"]},
            }
        )


def test_clarify_must_be_one_question():
    with pytest.raises(ValueError, match="one question"):
        validate_decision(
            {
                "decision": "clarify",
                "question": "Which order? Which SKU?",
                "citations": [],
                "packet": {},
            }
        )
