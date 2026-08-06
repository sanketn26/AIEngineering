"""Module 14 — append-oriented audit events (educational)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AuditEvent:
    ts: str
    actor_id: str
    action: str
    resource: str
    request_id: str
    input_hash: str
    policy_version: str
    model_id: str | None = None
    metadata: dict[str, Any] | None = None


def make_event(
    actor: str,
    action: str,
    resource: str,
    raw_input: str,
    *,
    request_id: str = "",
    policy_version: str = "v1",
    model_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ev = AuditEvent(
        ts=datetime.now(timezone.utc).isoformat(),
        actor_id=actor,
        action=action,
        resource=resource,
        request_id=request_id,
        input_hash=sha256_text(raw_input),
        policy_version=policy_version,
        model_id=model_id,
        metadata=metadata,
    )
    return asdict(ev)


@dataclass
class AuditLog:
    """In-memory + optional JSONL append log."""

    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def for_actor(self, actor_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("actor_id") == actor_id]
