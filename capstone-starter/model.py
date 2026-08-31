"""Mock classifier. No API keys, no network.

Deliberate incompleteness
-------------------------
GATE 1 remaining: ``classify`` has no deadline, timeout, or retry. A hung
"provider" (see ``_call_provider``) would block the worker forever. Add a
timeout + mapped failure + bounded retry before you call this dependable.

GATE 5 remaining: every call uses ``MODEL_LARGE``, including trivial
"forgot my password" tickets. Route easy classify to ``MODEL_SMALL`` and
keep a cost ledger; do not start with a bigger prompt.
"""

from __future__ import annotations

MODEL_SMALL = "mock-small"
MODEL_LARGE = "mock-large"

# Keyword lists are intentionally naive. The planted golden row in
# evals/golden.jsonl mixes shipping nouns with a billing intent so this
# heuristic labels ``shipping`` when the expected label is ``billing``.
_SHIPPING = ("package", "shipping", "delivery", "arrived", "tracking")
_BILLING = ("refund", "charge", "billed", "invoice", "payment", "duplicate")
_ACCOUNT = ("password", "login", "log in", "account", "locked")
_PRODUCT = ("crash", "bug", "feature", "settings", "app")
_HIGH = ("twice", "duplicate", "two weeks", "cannot log in", "locked", "urgent")


def classify(text: str) -> dict[str, str]:
    """Return category, priority, rationale, model_id.

    Always billed as the large mock model (Gate 5 hole).
    """
    raw = _call_provider(text, model_id=MODEL_LARGE)
    return raw


def _call_provider(text: str, *, model_id: str) -> dict[str, str]:
    """Stand-in for an HTTP model call.

    TODO(gate-1): wrap this in a deadline (e.g. ``signal`` / ``TimeoutError``
    or a client ``timeout=``). Retry only on transient failures, with a cap,
    and fail closed to a 504/503 mapping in the API — not an infinite hang.
    """
    return _heuristic(text, model_id=model_id)


def _heuristic(text: str, *, model_id: str) -> dict[str, str]:
    t = text.lower()
    # Shipping is checked *before* billing. That is the planted miss:
    # "package arrived … billed twice … refund" becomes shipping.
    if any(w in t for w in _SHIPPING):
        category = "shipping"
    elif any(w in t for w in _BILLING):
        category = "billing"
    elif any(w in t for w in _ACCOUNT):
        category = "account"
    elif any(w in t for w in _PRODUCT):
        category = "product"
    else:
        category = "other"

    if any(w in t for w in _HIGH):
        priority = "high"
    elif category in {"billing", "shipping", "account"}:
        priority = "medium"
    else:
        priority = "low"

    rationale = (
        f"keyword heuristic via {model_id}; "
        f"shipping tokens win over billing when both appear"
    )
    return {
        "category": category,
        "priority": priority,
        "rationale": rationale,
        "model_id": model_id,
    }


def retrieve(query: str) -> list[dict[str, str]]:
    """GATE 3 stub: empty corpus. Do not invent policy text to fill this."""
    return []
