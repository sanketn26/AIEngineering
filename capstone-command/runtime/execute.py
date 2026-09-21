"""One execution entry. Free-form text still reaches a patch. That is the first hole."""

from __future__ import annotations

from typing import Any

from runtime.models import Command
from runtime.policy import MOCK_PATCH, enforce_patch_policy, freeze
from runtime.validate import validate_document


def draft_from_intent(text: str) -> dict[str, Any]:
    """A spec assistant may draft. It may not confirm its own guesses."""

    return {
        "schema_version": "1.0",
        "task_id": "draft-from-intent",
        "command": Command.DRAFT.value,
        "goal": text.strip() or "unspecified change",
        "scope": {
            "allowed_files": ["client.py"],
            "max_files_changed": 1,
            "max_added_lines": 80,
        },
        "requirements": [
            {
                "id": "R1",
                "statement": "Behavior implied by the intent sentence.",
                "priority": "must",
            }
        ],
        "acceptance_criteria": [
            {
                "id": "AC1",
                "given": "the current code",
                "when": "the change is applied",
                "then": "the intent is satisfied",
                "verification": "pytest",
                "maps_to": ["R1"],
            }
        ],
        "non_goals": ["unspecified"],
        "unknowns": ["which files may change", "what proves the change", "what is out of scope"],
        "risk_level": "low",
    }


def execute(document: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(document, str):
        # GATE 1 — planted: a sentence is routed to code.patch.
        document = draft_from_intent(document)
        document["command"] = Command.PATCH.value

    result = validate_document(document)
    if result["status"] != "SPEC_READY":
        return {"status": result["status"], "errors": result["errors"]}

    spec = result["spec"]
    frozen = freeze(spec)
    policy = enforce_patch_policy(MOCK_PATCH, spec)
    return {
        "status": "candidate",
        "command": spec.command.value,
        "revision": frozen["revision"],
        "policy": policy["status"],
        "patch": MOCK_PATCH,
    }
