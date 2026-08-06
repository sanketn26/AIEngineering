"""Module 10 — routing, cache, and spend ledger."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any


class ModelRouter:
    def __init__(self, cheap: str, strong: str):
        self.cheap = cheap
        self.strong = strong

    def pick(
        self, task: str, prompt: str, *, hard_tasks: set[str] | None = None
    ) -> str:
        hard = hard_tasks or {"complex_reason", "deep_reason", "plan"}
        if task in {"classify", "route", "extract_fields"}:
            return self.cheap
        if task in hard or len(prompt) > 8000:
            return self.strong
        return self.cheap


class MemoryCache:
    def __init__(self, ttl_s: int = 3600):
        self.ttl_s = ttl_s
        self.store: dict[str, tuple[float, Any]] = {}

    def _key(self, namespace: str, payload: str) -> str:
        h = hashlib.sha256(payload.encode()).hexdigest()
        return f"{namespace}:{h}"

    def get(self, namespace: str, payload: str) -> Any | None:
        k = self._key(namespace, payload)
        item = self.store.get(k)
        if not item:
            return None
        exp, val = item
        if time.time() > exp:
            del self.store[k]
            return None
        return val

    def set(self, namespace: str, payload: str, value: Any) -> None:
        k = self._key(namespace, payload)
        self.store[k] = (time.time() + self.ttl_s, value)

    def clear(self) -> None:
        self.store.clear()


class UsageLedger:
    def __init__(self) -> None:
        self.by_user: dict[str, float] = defaultdict(float)
        self.tokens_by_user: dict[str, int] = defaultdict(int)

    def add(self, user_id: str, cost_usd: float, tokens: int = 0) -> None:
        if cost_usd < 0 or tokens < 0:
            raise ValueError("cost and tokens must be non-negative")
        self.by_user[user_id] += cost_usd
        self.tokens_by_user[user_id] += tokens

    def allowed(self, user_id: str, limit_usd: float) -> bool:
        return self.by_user[user_id] < limit_usd

    def usage(self, user_id: str) -> dict[str, float | int]:
        return {
            "cost_usd": self.by_user[user_id],
            "tokens": self.tokens_by_user[user_id],
        }
