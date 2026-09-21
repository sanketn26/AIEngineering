"""The starter still one-shots the 20B model and calls the answer done."""

from __future__ import annotations

from runtime.models import Part, TaskSpec

# A division the mock divider likes. R3 has no owner. P3 cites a check the user
# never wrote and imports a seam P1 does not export. V4, the join check, is unused.
MOCK_PLAN: dict = {
    "parts": [
        {
            "id": "P1",
            "owns": ["R1"],
            "checks": ["V1"],
            "files": ["ledger.py", "tests/test_ledger.py"],
            "depends_on": [],
            "exports": [{"name": "apply_once", "kind": "function", "signature": "(proposal_id: str) -> LedgerEntry"}],
            "imports": [],
        },
        {
            "id": "P2",
            "owns": ["R2"],
            "checks": ["V2"],
            "files": ["auth.py", "tests/test_auth.py"],
            "depends_on": [],
            "exports": [{"name": "require_scope", "kind": "function", "signature": "(actor, scope: str) -> None"}],
            "imports": [],
        },
        {
            "id": "P3",
            "owns": ["R1"],
            "checks": ["V-looks-right"],
            "files": ["api.py"],
            "depends_on": ["P1"],
            "exports": [],
            "imports": ["record_refund"],
        },
    ]
}


def route_part(part: Part, *, too_wide: bool, failures: int) -> str:
    del part, too_wide, failures
    # GATE 3 — planted: a part that is still too wide stays on 20B.
    return "20b"


def run_task(spec: TaskSpec) -> dict:
    del spec
    # GATE 1 — planted: one prompt, and the model's "done" is the result.
    # The divided path, the part checks, and the join never run.
    return {
        "mode": "one_shot",
        "model": "20b",
        "status": "done",
        "user_checks_ran": [],
        "final_checks_ran": False,
    }


def join_parts(spec: TaskSpec, plan_document: dict) -> dict:
    del spec, plan_document
    # GATE 4 — planted: parts are concatenated. Seams are not checked.
    # Join-scoped user checks do not run. A failed part does not stop the join.
    return {
        "status": "done",
        "joined": True,
        "join_checks_ran": [],
        "seam_conflicts": [],
        "stale_parts": [],
    }
