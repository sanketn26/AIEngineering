"""Module 05 — session memory and simple token budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Rough token estimate without external deps (~4 chars/token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def fit_budget(
    parts: list[tuple[str, str]],
    budget: int,
) -> list[tuple[str, str]]:
    """Keep highest-priority parts (list order) that fit in token budget."""
    kept: list[tuple[str, str]] = []
    used = 0
    for label, text in parts:
        n = estimate_tokens(text)
        if used + n <= budget:
            kept.append((label, text))
            used += n
        else:
            break
    return kept


@dataclass
class SessionMemory:
    summary: str = ""
    recent: list[dict[str, str]] = field(default_factory=list)
    max_recent: int = 10

    def add(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid role: {role}")
        self.recent.append({"role": role, "content": content})
        self.recent = self.recent[-self.max_recent :]

    def should_summarize(self, max_messages: int = 20) -> bool:
        return len(self.recent) > max_messages

    def build_messages(self, system: str, user: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
        if self.summary:
            msgs.append(
                {
                    "role": "system",
                    "content": f"Conversation summary:\n{self.summary}",
                }
            )
        msgs.extend(self.recent)
        msgs.append({"role": "user", "content": user})
        return msgs

    def transcript(self) -> str:
        lines = [f"{m['role']}: {m['content']}" for m in self.recent]
        if self.summary:
            lines.insert(0, f"summary: {self.summary}")
        return "\n".join(lines)
