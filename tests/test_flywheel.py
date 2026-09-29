import pytest

from src.flywheel import freeze_row, gradeable, name_lever, redact_trace


def test_redact_drops_pii_and_keeps_the_bug():
    clean = redact_trace(
        {
            "request_id": "req_8f3",
            "email": "a@ex.com",
            "name": "Ada",
            "policy_window": 30,
            "tool_window": 14,
            "customer": {"email": "a@ex.com", "sku": "enterprise"},
            "notes": [{"phone": "555", "text": "day 20"}],
        }
    )
    assert "email" not in clean and "name" not in clean
    assert clean["customer"] == {"sku": "enterprise"}
    assert clean["notes"] == [{"text": "day 20"}]
    assert clean["policy_window"] == 30
    assert clean["request_id"] == "req_8f3"


def test_freeze_quarantines_a_complete_snapshot():
    row = freeze_row(
        {"request_id": "req_8f3", "email": "a@ex.com", "input": "refund on day 20?"},
        prompt_digest="sha256:prompt",
        corpus_hash="sha256:corpus",
        tool_fixtures={"get_subscription": {"window_days": 14, "email": "a@ex.com"}},
        found_in_release="2026-04",
        lever="answer-contract",
    )
    assert row["status"] == "quarantine"
    assert "email" not in row
    assert row["tool_fixtures"]["get_subscription"] == {"window_days": 14}


def test_freeze_rejects_a_row_that_cannot_be_replayed():
    with pytest.raises(ValueError):
        freeze_row(
            {"input": "no id"},
            prompt_digest="sha256:prompt",
            corpus_hash="sha256:corpus",
            tool_fixtures={"get_subscription": "x"},
            found_in_release="2026-04",
            lever="retrieval",
        )


def test_this_weeks_miss_cannot_grade_this_weeks_release():
    rows = [
        {"id": "old", "status": "held-out", "found_in_release": "2026-03"},
        {"id": "new", "status": "quarantine", "found_in_release": "2026-04"},
        {"id": "relabeled", "status": "held-out", "found_in_release": "2026-04"},
        {"id": "still_quarantine", "status": "quarantine", "found_in_release": "2026-03"},
    ]
    ids = [row["id"] for row in gradeable(rows, "2026-04")]
    assert ids == ["old"]


def test_name_lever_prefers_a_policy_tool_conflict():
    assert name_lever({"policy_window": 30, "tool_window": 14, "gold_in_shortlist": True}) == "answer-contract"
    assert name_lever({"gold_in_shortlist": False}) == "retrieval"
    assert name_lever({"gold_in_shortlist": True, "answer_used_gold": False}) == "prompt"
    assert name_lever({"duplicate_write": True}) == "idempotency"
    assert name_lever({"looped": True, "final_ok": True}) == "trajectory"
    assert name_lever({"bundle_unchanged": True, "behavior_moved": True}) == "pin-upstream"
    with pytest.raises(ValueError):
        name_lever({"input": "looks fine"})
