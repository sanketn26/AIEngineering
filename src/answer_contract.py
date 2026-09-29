"""Module 07 answer contract — one decision, checked in code."""

from __future__ import annotations

from typing import Any

DECISIONS = ("answer", "clarify", "abstain", "escalate")


def decide(
    *,
    policy_window: int | None,
    tool_window: int | None,
    citations: list[str],
    retrieved: list[str],
    missing_slot: str | None = None,
) -> dict[str, Any]:
    """Propose a decision. validate_decision is what makes it legal."""
    packet = {
        "retrieved": list(retrieved),
        "tools": [{"window_days": tool_window}] if tool_window is not None else [],
    }
    if missing_slot:
        return {
            "decision": "clarify",
            "question": missing_slot,
            "citations": [],
            "packet": packet,
        }
    if policy_window is None and tool_window is None:
        return {
            "decision": "abstain",
            "question": None,
            "citations": [],
            "packet": {**packet, "conflict": "no-evidence"},
        }
    if (
        policy_window is not None
        and tool_window is not None
        and policy_window != tool_window
    ):
        return {
            "decision": "abstain",
            "question": None,
            "citations": [],
            "packet": {**packet, "conflict": f"policy-{policy_window}d vs tool-{tool_window}d"},
        }
    if not citations:
        return {
            "decision": "abstain",
            "question": None,
            "citations": [],
            "packet": {**packet, "conflict": "no-citation"},
        }
    return {
        "decision": "answer",
        "question": None,
        "citations": list(citations),
        "packet": packet,
    }


def validate_decision(decision: dict[str, Any]) -> None:
    kind = decision.get("decision")
    if kind not in DECISIONS:
        raise ValueError(f"unknown decision: {kind}")
    citations = decision.get("citations") or []
    question = decision.get("question")
    packet = decision.get("packet") or {}
    retrieved = packet.get("retrieved") or []
    if not isinstance(citations, list) or not isinstance(retrieved, list):
        raise ValueError("citations and retrieved must be lists of ids")
    if kind == "answer" and not citations:
        raise ValueError("answer requires citations")
    for cite in citations:
        if not isinstance(cite, str) or not cite:
            raise ValueError("citations must be ids")
        if cite not in retrieved:
            raise ValueError(f"unresolved citation: {cite}")
    if kind == "clarify":
        if not question or question.count("?") != 1:
            raise ValueError("clarify asks one question")
    if kind in {"abstain", "escalate"} and not packet:
        raise ValueError(f"{kind} requires a packet")
    if kind == "answer" and packet.get("conflict"):
        raise ValueError("answer is illegal when sources conflict")


def score_contract(rows: list[dict[str, Any]], predict) -> dict[str, Any]:
    """Score decision rows. expect.decision may be a string or a set of strings."""
    failures = []
    passed = 0
    for row in rows:
        pred = predict(row)
        expect = row["expect"]
        try:
            validate_decision(pred)
        except ValueError as exc:
            failures.append({"id": row["id"], "expect": expect, "pred": pred, "error": str(exc)})
            continue
        allowed = expect["decision"]
        if isinstance(allowed, str):
            allowed = {allowed}
        ok = pred["decision"] in allowed
        banned = expect.get("must_not_cite")
        if banned and banned in pred.get("citations", []):
            ok = False
        if ok:
            passed += 1
        else:
            failures.append({"id": row["id"], "expect": expect, "pred": pred})
    n = len(rows)
    return {"n": n, "passed": passed, "accuracy": (passed / n if n else 0.0), "failures": failures}
