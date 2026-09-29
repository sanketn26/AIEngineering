"""Gate 5 eval flywheel — sample, redact, freeze, quarantine, name one lever."""

from __future__ import annotations

from typing import Any

PII_FIELDS = ("email", "name", "phone", "account_id")


def _redact(value: Any) -> Any:
    """Drop PII keys at every level. Strings and numbers pass through."""
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items() if k not in PII_FIELDS}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def redact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Drop PII fields a labeler does not need, including nested ones."""
    redacted = _redact(trace)
    if not isinstance(redacted, dict):
        raise ValueError("trace must be an object")
    return redacted


def freeze_row(
    trace: dict[str, Any],
    *,
    prompt_digest: str,
    corpus_hash: str,
    tool_fixtures: dict[str, Any],
    found_in_release: str,
    lever: str,
) -> dict[str, Any]:
    """Bind a redacted trace to the snapshot that produced it."""
    if not trace.get("request_id"):
        raise ValueError("request_id is required")
    if not prompt_digest or not corpus_hash or not tool_fixtures:
        raise ValueError("prompt digest, corpus hash, and tool fixtures are required")
    if not found_in_release or not lever:
        raise ValueError("release and lever are required")
    row = redact_trace(trace)
    row.update(
        {
            "status": "quarantine",
            "prompt_digest": prompt_digest,
            "corpus_hash": corpus_hash,
            "tool_fixtures": redact_trace(tool_fixtures),
            "found_in_release": found_in_release,
            "lever": lever,
        }
    )
    return row


def gradeable(rows: list[dict[str, Any]], current_release: str) -> list[dict[str, Any]]:
    """Grade a row only after it leaves quarantine, and never in the release that found it."""
    return [
        row
        for row in rows
        if row.get("status") != "quarantine"
        and row.get("found_in_release") != current_release
    ]


def name_lever(trace: dict[str, Any]) -> str:
    """Pick one backlog lever from fields on the frozen trace.

    Disagreement between the policy window and the tool window wins over
    retrieval, because both calls can succeed and the answer is still wrong.
    """
    policy = trace.get("policy_window")
    tool = trace.get("tool_window")
    if policy is not None and tool is not None and policy != tool:
        return "answer-contract"
    if trace.get("gold_in_shortlist") is False:
        return "retrieval"
    if trace.get("gold_in_shortlist") and trace.get("answer_used_gold") is False:
        return "prompt"
    if trace.get("duplicate_write"):
        return "idempotency"
    if trace.get("looped") and trace.get("final_ok"):
        return "trajectory"
    if trace.get("bundle_unchanged") and trace.get("behavior_moved"):
        return "pin-upstream"
    raise ValueError("trace does not name a single lever")
