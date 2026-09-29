from src.corpus import (
    chunk_rule,
    content_hash,
    drop_superseded,
    needs_reembed,
    pack_with_parents,
)


def test_hash_changes_only_when_the_text_changes():
    text = "Standard refund window is 30 days."
    digest = content_hash(text)
    assert needs_reembed(digest, text) is False
    assert needs_reembed(digest, text + " ") is True
    assert needs_reembed(None, text) is True


def test_exception_keeps_its_parent_and_packing_includes_both():
    chunks = chunk_rule(
        "refund-window",
        "Standard refund window is 30 days.",
        [("refund-window-enterprise", "Enterprise SKUs: 14 days.")],
    )
    by_id = {c.id: c for c in chunks}
    child = by_id["refund-window-enterprise"]
    assert child.role == "child" and child.parent_id == "refund-window"
    packed = pack_with_parents(["refund-window-enterprise"], by_id)
    assert packed == ["refund-window", "refund-window-enterprise"]


def test_only_a_reviewed_supersession_retires_a_clause():
    ids = ["refund-window-v3", "refund-window-v4"]
    reviewed = [{"from": "refund-window-v3", "to": "refund-window-v4", "reviewed_by": "legal"}]
    draft = [{"from": "refund-window-v3", "to": "refund-window-v4"}]
    assert drop_superseded(ids, reviewed) == ["refund-window-v4"]
    assert drop_superseded(ids, draft) == ids
    assert drop_superseded(["refund-window-v3"], reviewed) == ["refund-window-v3"]
