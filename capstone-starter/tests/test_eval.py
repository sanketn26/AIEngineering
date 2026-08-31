"""Golden-set runner.

This test is allowed to see a failing case. Gate 2 is to make
``planted-mixed-ticket`` pass *without* claiming 100% accuracy here —
main-repo CI must stay green while the starter still has a known miss.
"""

from __future__ import annotations

import json
from pathlib import Path

from model import classify

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl"
# Gate 2 work: make this id pass (heuristic currently labels it shipping).
PLANTED_ID = "planted-mixed-ticket"
FIELDS = ("category", "priority")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_suite(rows: list[dict]) -> dict:
    failures = []
    passed = 0
    for row in rows:
        pred = classify(row["input"])
        expect = row["expect"]
        ok = all(pred.get(f) == expect.get(f) for f in FIELDS)
        if ok:
            passed += 1
        else:
            failures.append(
                {
                    "id": row.get("id"),
                    "input": row["input"],
                    "expect": expect,
                    "pred": {k: pred.get(k) for k in FIELDS},
                }
            )
    n = len(rows)
    return {
        "n": n,
        "passed": passed,
        "accuracy": (passed / n) if n else 0.0,
        "failures": failures,
    }


def test_golden_suite_runs_and_reports_known_failure():
    rows = load_jsonl(GOLDEN)
    result = run_suite(rows)
    assert result["n"] >= 4, "golden set should have more than the planted miss"
    failed_ids = {f["id"] for f in result["failures"]}
    assert PLANTED_ID in failed_ids, (
        "expected the planted mixed-ticket miss to be reported; "
        "if you already fixed it in Gate 2, update this assertion to "
        "require PLANTED_ID not in failed_ids and accuracy == 1.0"
    )
    assert result["accuracy"] < 1.0
    assert result["passed"] >= 1
