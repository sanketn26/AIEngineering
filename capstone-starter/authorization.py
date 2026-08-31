"""Authorization lives *outside* the model.

GATE 4 remaining work
---------------------
``tools.refund_customer`` is a write. This module currently does **not**
enforce role or scope: ``authorize`` returns True for any actor (including
None). A prompt that says "you may only refund if admin" is not a control.

Exit criteria for Gate 4: refunds fail closed unless ``actor.role`` is in
{support, admin} *and* ``refund:write`` is in ``actor.scopes``. Viewers,
missing actors, and the model itself must never execute the write. Tests
should cover deny-by-default, not just the happy path.
"""

from __future__ import annotations

from typing import Any

# Write tools that Gate 4 must bind to a role/scope check.
SENSITIVE_ACTIONS = frozenset({"refund_customer"})


def authorize(actor: Any, action: str) -> bool:
    """Return whether ``action`` is allowed for ``actor``.

    TODO(gate-4): fail closed. Today this is a stub so the happy-path
    triage endpoint still runs. Do not "fix" it by adding a system-prompt
    instruction — the model is the untrusted party.
    """
    # Intentionally permissive. Gate 4 work starts here.
    return True
