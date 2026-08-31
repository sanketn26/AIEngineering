"""Tool surface. Writes are *proposed*, never executed in this starter.

``refund_customer`` exists so Gate 4 has something to authorize. The
HTTP path must not move money — it returns a proposal the learner later
binds to ``authorization.authorize``.
"""

from __future__ import annotations

from typing import Any

from authorization import SENSITIVE_ACTIONS, authorize


def refund_customer(
    *,
    ticket_id: str | None,
    amount_cents: int,
    actor: Any,
) -> dict[str, Any]:
    """Propose a refund. Does not call a payment API.

    GATE 4: ``authorize`` is currently a no-op. After you implement it,
    this function should raise PermissionError (or return status=denied)
    for viewers / missing actors. Still do not execute a real write.
    """
    allowed = authorize(actor, "refund_customer")
    if not allowed:
        return {
            "tool": "refund_customer",
            "status": "denied",
            "args": {"ticket_id": ticket_id, "amount_cents": amount_cents},
            "reason": "authorization failed",
        }
    return {
        "tool": "refund_customer",
        "status": "proposed",  # not executed
        "args": {"ticket_id": ticket_id, "amount_cents": amount_cents},
        "reason": "write tools propose only; no payment side effect",
    }


def lookup_order(*, ticket_id: str | None, actor: Any) -> dict[str, Any]:
    """Read-shaped stub. Empty until Gate 3 grows a real store."""
    return {
        "tool": "lookup_order",
        "status": "proposed",
        "args": {"ticket_id": ticket_id},
        "reason": "no order store wired (Gate 3)",
    }


def propose_actions(triage: dict[str, Any], *, ticket_id: str | None, actor: Any) -> list[dict[str, Any]]:
    """Map a classification to tool proposals. Never executes writes."""
    category = triage.get("category")
    actions: list[dict[str, Any]] = []
    if category == "billing":
        actions.append(
            refund_customer(ticket_id=ticket_id, amount_cents=0, actor=actor)
        )
    elif category == "shipping":
        actions.append(lookup_order(ticket_id=ticket_id, actor=actor))
    return actions


def is_write_tool(name: str) -> bool:
    return name in SENSITIVE_ACTIONS
