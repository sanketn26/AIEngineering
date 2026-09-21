"""Turn a document into SPEC_READY, SPEC_INVALID, or (once you close Gate 1) SPEC_INCOMPLETE."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import ValidationError

from runtime.models import TaskSpec


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("specification must be a mapping")
    return document


def _errors_from_validation(exc: ValidationError) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for item in exc.errors():
        loc = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        message = item.get("msg", "invalid")
        code = "UNSAFE_PATH" if "unsafe path" in message else "SCHEMA"
        if "duplicate" in message:
            code = "DUPLICATE_ID"
        if "unknown requirement" in message:
            code = "UNKNOWN_REQUIREMENT_REF"
        errors.append({"code": code, "path": loc, "message": message})
    return errors


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = TaskSpec.model_validate(document)
    except ValidationError as exc:
        return {"status": "SPEC_INVALID", "errors": _errors_from_validation(exc), "spec": None}
    return {"status": "SPEC_READY", "errors": [], "spec": spec}


def render_contract(spec: TaskSpec) -> str:
    musts = [item.id for item in spec.requirements if item.priority.value == "must"]
    lines = [
        f"Task {spec.task_id} ({spec.command.value})",
        f"Goal: {spec.goal}",
        f"Editable: {', '.join(spec.scope.allowed_files)}",
        f"Budget: {spec.scope.max_files_changed} files / {spec.scope.max_added_lines} added lines",
        f"Must requirements: {', '.join(musts) if musts else '(none)'}",
        f"Acceptance criteria: {len(spec.acceptance_criteria)}",
        f"Unknowns: {len(spec.unknowns)}",
        "Frozen: no — close Gate 5 before an approval can name a revision.",
    ]
    return "\n".join(lines)
