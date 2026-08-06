"""Module 02 — input sanitization and lightweight PII redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(your\s+)?(system|developer)\s+prompt", re.I),
    re.compile(r"reveal\s+(your\s+)?(system|hidden)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+DAN", re.I),
    re.compile(r"jailbreak", re.I),
]

PII_REGEX: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone_us": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn_us": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

_FAKE_TAG = re.compile(r"</?(system|assistant|tool)>", re.I)


@dataclass
class SanitizeResult:
    text: str
    flagged: bool
    reasons: list[str] = field(default_factory=list)


def sanitize_user_text(text: str, max_chars: int = 8000) -> SanitizeResult:
    """Bound length, strip fake role tags, flag common injection patterns."""
    if text is None:
        raise TypeError("text must be a string")
    reasons: list[str] = []
    cleaned = text.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
        reasons.append("truncated")
    cleaned, n_tags = _FAKE_TAG.subn("", cleaned)
    if n_tags:
        reasons.append("stripped_fake_tags")
    flagged = False
    for pat in INJECTION_PATTERNS:
        if pat.search(cleaned):
            flagged = True
            reasons.append(f"pattern:{pat.pattern}")
    return SanitizeResult(text=cleaned, flagged=flagged, reasons=reasons)


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace simple PII patterns with placeholders. Educational, not exhaustive."""
    counts: dict[str, int] = {}
    out = text
    for name, rx in PII_REGEX.items():
        out, n = rx.subn(f"[REDACTED_{name.upper()}]", out)
        if n:
            counts[name] = n
    return out, counts


def prepare_user_message(
    text: str,
    *,
    max_chars: int = 8000,
    redact: bool = True,
) -> tuple[str, SanitizeResult, dict[str, int]]:
    """Sanitize then optionally redact PII.

    Returns (safe_text, sanitize_result, pii_counts).
    """
    san = sanitize_user_text(text, max_chars=max_chars)
    counts: dict[str, int] = {}
    safe = san.text
    if redact:
        safe, counts = redact_pii(safe)
    return safe, san, counts
