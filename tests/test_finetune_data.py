import json
from pathlib import Path

from src.finetune_data import (
    check_source_rights,
    clean_text,
    is_plain_text,
    numeric_strings,
    scorecard,
    split_by_company,
    unsupported_numbers,
    validate_example,
)

SAMPLE = Path(__file__).resolve().parents[1] / "examples/fine-tuning/data/sample.jsonl"


def _rows():
    return [json.loads(line) for line in SAMPLE.read_text().splitlines() if line]


def test_rights_gate_blocks_assumed_permission():
    record = {
        "source_id": "decks",
        "source_url": "https://example.com",
        "rights_status": "assumed",
        "permission_scope": "training",
        "rights_reviewed_at": "2026-09-17",
    }
    assert check_source_rights(record)
    record["rights_status"] = "approved"
    assert check_source_rights(record) == []


def test_clean_text_keeps_numbers_untouched():
    raw = "Revenue\x00  grew\u00a0to $12.SM\n\n\n\nnext"
    assert clean_text(raw) == "Revenue grew to $12.SM\n\nnext"


def test_numeric_strings_and_unsupported():
    assert numeric_strings("We grew 80% to $12.5M in 2024.") == {
        "80%",
        "$12.5M",
        "2024",
    }
    assert unsupported_numbers("Seed round.", "We have 40 clinics.") == {"40"}
    assert unsupported_numbers("Serving 40 clinics.", "Now 40 clinics.") == set()


def test_plain_text_compliance():
    assert is_plain_text("Clinics lose revenue every week.")
    assert not is_plain_text('{"title": "Pitch"}')
    assert not is_plain_text("Slide 1: Problem\nClinics lose money")
    assert not is_plain_text("   ")


def test_sample_dataset_is_valid():
    rows = _rows()
    assert len(rows) >= 6
    for row in rows:
        assert validate_example(row) == [], row["company_id"]


def test_validate_flags_invented_metric():
    row = _rows()[0]
    row["messages"][2]["content"] += " We already serve 300 clinics."
    assert any("300" in p for p in validate_example(row))


def test_split_never_leaks_a_company():
    rows = _rows() + _rows()  # same companies twice, like multiple pages
    splits, manifest = split_by_company(rows)
    seen = [set(manifest[f"{k}_company_ids"]) for k in splits]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert sum(len(v) for v in splits.values()) == len(rows)
    assert split_by_company(rows)[1] == manifest  # deterministic


def test_scorecard():
    card = scorecard([("brief", "Clean copy."), ("brief", "We have 9 customers.")])
    assert card["plain_text_rate"] == 1.0
    assert card["unsupported_number_rate"] == 0.5
