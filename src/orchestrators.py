"""Module 26 — orchestrator comparison + per-step cost attribution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Teaching comparison, not a benchmark. Numbers are ordinal ranks (1=best).
ORCHESTRATOR_MATRIX: list[dict[str, Any]] = [
    {
        "name": "custom loop",
        "control": 1,
        "ops_cost": 1,
        "hitl": 3,
        "durable": 4,
        "ecosystem": 4,
        "lock_in": 1,
        "when": "You can name the state machine and want tests first.",
        "fail": "You reimplement graphs, persistence, and replay poorly.",
    },
    {
        "name": "LangGraph",
        "control": 2,
        "ops_cost": 3,
        "hitl": 1,
        "durable": 1,
        "ecosystem": 2,
        "lock_in": 3,
        "when": "Branching graphs, checkpoints, and HITL are load-bearing.",
        "fail": "Graph soup without budgets; debug the framework not the policy.",
    },
    {
        "name": "CrewAI",
        "control": 4,
        "ops_cost": 2,
        "hitl": 4,
        "durable": 4,
        "ecosystem": 3,
        "lock_in": 3,
        "when": "Role-play crews with a thin manager and short tasks.",
        "fail": "Persona theater; unbounded debate; weak isolation.",
    },
    {
        "name": "MCP hosts",
        "control": 3,
        "ops_cost": 2,
        "hitl": 2,
        "durable": 3,
        "ecosystem": 1,
        "lock_in": 2,
        "when": "Portable tools/resources across IDE and product hosts.",
        "fail": "Protocol without host policy; untrusted servers as root.",
    },
]


def compare_orchestrators(*names: str) -> list[dict[str, Any]]:
    wanted = set(names) if names else {row["name"] for row in ORCHESTRATOR_MATRIX}
    rows = [r for r in ORCHESTRATOR_MATRIX if r["name"] in wanted]
    if len(rows) != len(wanted):
        known = {r["name"] for r in ORCHESTRATOR_MATRIX}
        raise KeyError(f"unknown orchestrator(s): {wanted - known}")
    return rows


def tradeoff_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    """Lower is better (ranks). Weighted sum of selected rank columns."""
    total = 0.0
    wsum = 0.0
    for key, weight in weights.items():
        if key not in row:
            raise KeyError(key)
        total += float(row[key]) * weight
        wsum += weight
    if wsum <= 0:
        raise ValueError("weights must sum to > 0")
    return total / wsum


@dataclass
class CostEvent:
    agent: str
    step: int
    model: str
    tokens_in: int
    tokens_out: int
    usd: float
    latency_ms: int
    tool: str | None = None


class CostAttribution:
    """Per-agent, per-step ledger for multi-agent bills and traces."""

    def __init__(self) -> None:
        self.events: list[CostEvent] = []

    def record(self, event: CostEvent) -> None:
        if event.usd < 0 or event.tokens_in < 0 or event.tokens_out < 0:
            raise ValueError("cost and tokens must be non-negative")
        self.events.append(event)

    def by_agent(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(
            lambda: {"usd": 0.0, "tokens": 0.0, "steps": 0.0, "latency_ms": 0.0}
        )
        for e in self.events:
            row = out[e.agent]
            row["usd"] += e.usd
            row["tokens"] += e.tokens_in + e.tokens_out
            row["steps"] += 1
            row["latency_ms"] += e.latency_ms
        return {k: dict(v) for k, v in out.items()}

    def total_usd(self) -> float:
        return round(sum(e.usd for e in self.events), 6)


@dataclass
class TraceSpan:
    name: str
    agent: str
    attrs: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    """Reasoning-trace stand-in: structured spans, not hidden chain-of-thought."""

    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []

    def span(self, name: str, agent: str, **attrs: Any) -> TraceSpan:
        item = TraceSpan(name=name, agent=agent, attrs=attrs)
        self.spans.append(item)
        return item

    def export(self) -> list[dict[str, Any]]:
        # Identity last: attrs must not clobber name/agent if those keys appear.
        return [{**s.attrs, "name": s.name, "agent": s.agent} for s in self.spans]
