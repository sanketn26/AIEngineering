"""Module 01/03 — lightweight prompt templates."""

from __future__ import annotations

from string import Template
from typing import Mapping

TEMPLATES: dict[str, Template] = {
    "summarize": Template(
        "Summarize for a busy $audience in $bullets bullets.\n\nText:\n$content"
    ),
    "classify": Template(
        "Classify the text into one of: $labels.\n"
        'Return JSON: {"label": string, "confidence": number}\n\n'
        "Text:\n$content"
    ),
    "email_reply": Template(
        "You are a professional email assistant.\n"
        "Write a polite reply under $max_words words.\n\n"
        "Email:\n$content\n\n"
        "Requirements:\n"
        "- Acknowledge the request\n"
        "- Answer questions if possible\n"
        "- Propose a clear next step"
    ),
    "rag_answer": Template(
        "Answer using only the sources. Cite chunk ids.\n"
        "If sources are insufficient, say you do not know.\n\n"
        "Sources:\n$sources\n\nQuestion: $question"
    ),
}


def render(name: str, **kwargs: object) -> str:
    """Render a named template. Unknown keys are left as placeholders."""
    if name not in TEMPLATES:
        known = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"Unknown template {name!r}. Known: {known}")
    # Template.safe_substitute expects str-ish mapping values
    mapping: Mapping[str, object] = {k: str(v) for k, v in kwargs.items()}
    return TEMPLATES[name].safe_substitute(mapping)


def list_templates() -> list[str]:
    return sorted(TEMPLATES)
