"""Who may attempt the task. Gate 4 derives this from the spec instead of trusting a label."""

from __future__ import annotations

from runtime.models import TaskSpec


def eligible_for_4b(spec: TaskSpec) -> bool:
    # GATE 4 — planted: risk_level is whatever the document says.
    return spec.risk_level in {"low", "medium"}
