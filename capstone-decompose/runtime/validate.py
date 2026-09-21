"""Load a user spec. Plan acceptance still has the divide and join holes in it."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import ValidationError

from runtime.models import Plan, TaskSpec


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("document must be a mapping")
    return document


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = TaskSpec.model_validate(document)
    except ValidationError as exc:
        errors = []
        for item in exc.errors():
            message = item.get("msg", "invalid")
            code = "UNSAFE_PATH" if "unsafe path" in message else "SCHEMA"
            loc = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
            errors.append({"code": code, "path": loc, "message": message})
        return {"status": "SPEC_INVALID", "errors": errors, "spec": None}
    return {"status": "SPEC_READY", "errors": [], "spec": spec}


def accept_plan(spec: TaskSpec, plan_document: dict[str, Any]) -> dict[str, Any]:
    """Accept a division. Several agentic defects are still allowed on purpose."""

    try:
        plan = Plan.model_validate(plan_document)
    except ValidationError as exc:
        return {"status": "PLAN_INVALID", "errors": [{"code": "SCHEMA", "message": str(exc)}], "plan": None}

    allowed = set(spec.scope.allowed_files)
    errors: list[dict[str, str]] = []
    for part in plan.parts:
        for path in part.files:
            if path not in allowed:
                errors.append({
                    "code": "PATH_NOT_ALLOWED",
                    "path": part.id,
                    "message": f"{path} is outside the task allowlist",
                })
    # GATE 1 / 2 — planted holes:
    # a missing must-owner, an invented check id, a cycle, and an imported seam
    # that nobody exports are all still PLAN_READY.
    if errors:
        return {"status": "PLAN_INVALID", "errors": errors, "plan": None}
    return {"status": "PLAN_READY", "errors": [], "plan": plan}
