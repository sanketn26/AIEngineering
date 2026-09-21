"""Module 06 — fine-tuning data hygiene: rights, cleaning, numbers, splits, scoring.

Stdlib only. Training itself lives in ``examples/fine-tuning/`` (optional GPU deps).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable

REQUIRED_RIGHTS_FIELDS = (
    "source_id",
    "source_url",
    "rights_status",
    "permission_scope",
    "rights_reviewed_at",
)

NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:[$€£₹]\s*)?\d[\d,.]*(?:\s*(?:%|[xXMBKmk]\b))?"
)

REQUIRED_ROLES = ("system", "user", "assistant")


def check_source_rights(record: dict[str, Any]) -> list[str]:
    """Problems that block a source from training. Empty list means cleared."""
    problems = [f"missing {f}" for f in REQUIRED_RIGHTS_FIELDS if not record.get(f)]
    if record.get("rights_status") not in (None, "", "approved"):
        problems.append(f"rights_status is {record['rights_status']!r}, not 'approved'")
    if record.get("permission_scope") not in (None, "", "training"):
        problems.append("permission_scope does not cover training")
    return problems


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    """Remove extraction noise only. Never rewrites words or numbers."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def numeric_strings(text: str) -> set[str]:
    found = set()
    for match in NUMBER_PATTERN.findall(text):
        normalized = match.replace(" ", "").rstrip(".,")
        if normalized:
            found.add(normalized)
    return found


def unsupported_numbers(source: str, generated: str) -> set[str]:
    """Numbers in the output that never appear in the source evidence."""
    return numeric_strings(generated) - numeric_strings(source)


def is_plain_text(output: str) -> bool:
    """Target format is prose: not JSON, no slide scaffolding."""
    stripped = output.strip()
    if not stripped:
        return False
    try:
        json.loads(stripped)
        return False
    except ValueError:
        pass
    return not re.search(r"(?im)^\s*(slide\s*\d+|#{1,6}\s|title:)", stripped)


def validate_example(row: dict[str, Any]) -> list[str]:
    """Schema + grounding checks for one chat-format training row."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ["messages must be a list"]
    roles = tuple(m.get("role") for m in messages)
    if roles != REQUIRED_ROLES:
        return [f"roles must be {REQUIRED_ROLES}, got {roles}"]
    if any(not str(m.get("content", "")).strip() for m in messages):
        return ["empty message content"]
    problems = []
    if not row.get("company_id"):
        problems.append("missing company_id (needed for company-level split)")
    source = messages[0]["content"] + "\n" + messages[1]["content"]
    target = messages[2]["content"]
    if not is_plain_text(target):
        problems.append("assistant target is not plain text")
    invented = unsupported_numbers(source, target)
    if invented:
        problems.append(f"target has numbers not in brief: {sorted(invented)}")
    return problems


def _bucket(company_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{company_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def split_by_company(
    rows: Iterable[dict[str, Any]],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: str = "v1",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deterministic train/validation/test split where no company crosses splits."""
    rows = list(rows)
    train_cut = ratios[0]
    val_cut = ratios[0] + ratios[1]
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    companies: dict[str, set[str]] = {k: set() for k in splits}
    for row in rows:
        cid = row["company_id"]
        b = _bucket(cid, seed)
        name = "train" if b < train_cut else "validation" if b < val_cut else "test"
        splits[name].append(row)
        companies[name].add(cid)
    dataset_hash = sha256_bytes(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()
    )
    manifest = {
        "split_version": seed,
        "dataset_hash": dataset_hash,
        **{f"{k}_company_ids": sorted(v) for k, v in companies.items()},
    }
    return splits, manifest


def scorecard(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Score (source, generated) pairs on the automatic release checks."""
    pairs = list(pairs)
    if not pairs:
        return {"n": 0, "plain_text_rate": 0.0, "unsupported_number_rate": 0.0}
    plain = sum(is_plain_text(g) for _, g in pairs)
    invented = sum(bool(unsupported_numbers(s, g)) for s, g in pairs)
    return {
        "n": len(pairs),
        "plain_text_rate": plain / len(pairs),
        "unsupported_number_rate": invented / len(pairs),
    }
