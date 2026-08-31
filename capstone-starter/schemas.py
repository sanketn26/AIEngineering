"""Request/response contracts for the support-ticket triage service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "shipping", "account", "product", "other"]
Priority = Literal["low", "medium", "high", "urgent"]
Role = Literal["viewer", "agent", "support", "admin"]


class Actor(BaseModel):
    """Who is asking. Gate 4 must treat role/scope as the authz source of truth."""

    id: str
    role: Role = "viewer"
    scopes: list[str] = Field(default_factory=list)


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    ticket_id: str | None = None
    actor: Actor | None = None


class ProposedAction(BaseModel):
    tool: str
    status: Literal["proposed", "denied", "executed"] = "proposed"
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class TriageResponse(BaseModel):
    category: Category
    priority: Priority
    rationale: str
    citations: list[str] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    model_id: str
    request_id: str
