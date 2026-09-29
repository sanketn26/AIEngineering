"""Module 09 corpus lesson — parent/child chunks, content hashes, supersession."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def needs_reembed(stored_hash: str | None, text: str) -> bool:
    """Unchanged canonical text keeps its vectors."""
    return stored_hash != content_hash(text)


@dataclass(frozen=True)
class PolicyChunk:
    id: str
    role: str
    text: str
    parent_id: str | None = None


def chunk_rule(
    rule_id: str,
    rule_text: str,
    exceptions: list[tuple[str, str]],
) -> list[PolicyChunk]:
    """One parent for the rule. Each exception is a child that points at it."""
    parent = PolicyChunk(id=rule_id, role="parent", text=rule_text)
    children = [
        PolicyChunk(id=exc_id, role="child", text=exc_text, parent_id=rule_id)
        for exc_id, exc_text in exceptions
    ]
    return [parent, *children]


def pack_with_parents(
    retrieved: list[str], chunks: dict[str, PolicyChunk]
) -> list[str]:
    """A retrieved child is packed only together with its parent."""
    packed: list[str] = []
    for chunk_id in retrieved:
        chunk = chunks[chunk_id]
        if chunk.parent_id and chunk.parent_id not in packed:
            packed.append(chunk.parent_id)
        if chunk_id not in packed:
            packed.append(chunk_id)
    return packed


def drop_superseded(
    ids: list[str], edges: list[dict[str, str]]
) -> list[str]:
    """Drop a clause only when its reviewed replacement is also in the shortlist.

    An edge with no reviewed_by is a draft and does not retire anything.
    """
    present = set(ids)
    retired = {
        edge["from"]
        for edge in edges
        if edge.get("reviewed_by")
        and edge["from"] in present
        and edge["to"] in present
    }
    return [chunk_id for chunk_id in ids if chunk_id not in retired]
