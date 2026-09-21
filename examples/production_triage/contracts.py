"""Contracts at the HTTP, model, and human-approval boundaries."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Ticket(StrictModel):
    text: str = Field(min_length=1, max_length=8000)
    ticket_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


class Classification(StrictModel):
    category: Literal["billing", "shipping", "account", "product", "other"]
    priority: Literal["low", "medium", "high"]


class Decision(StrictModel):
    approve: bool


class Principal(BaseModel):
    id: str
    role: Literal["viewer", "support", "admin"]
    scopes: frozenset[str] = frozenset()

    def can_refund(self) -> bool:
        return self.role in {"support", "admin"} and "refund:write" in self.scopes
