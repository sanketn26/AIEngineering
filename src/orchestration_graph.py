"""Module 19 graph lesson — diamond, ceiling, fan-in, dry loop, context graph."""

from __future__ import annotations

from typing import Any, Callable


class PlanError(ValueError):
    """The runtime refused an edge before any model call."""


def independence_fraction(uses_previous: list[bool]) -> float:
    """p is the share of edges that do not need the previous step's output."""
    if not uses_previous:
        raise ValueError("an empty workflow has no p")
    independent = sum(1 for needed in uses_previous if not needed)
    return independent / len(uses_previous)


def speedup(p: float, n: int) -> float:
    """Amdahl ceiling. Agent steps are not uniform CPU work; this is a ceiling."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    return 1.0 / ((1.0 - p) + (p / n))


def fanout_pays(p: float, n: int) -> bool:
    """Buy N only when the ceiling is at least half of N.

    p=0.95, N=16 → about ×9, which clears half of 16.
    p=0.70, N=16 → about ×2.9, which does not.
    """
    return speedup(p, n) >= 0.5 * n


def reduce_findings(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Dedupe by source in code. The first copy of a source wins."""
    seen: dict[str, dict[str, Any]] = {}
    for group in groups:
        for finding in group:
            source = finding["source"]
            if source not in seen:
                seen[source] = finding
    return list(seen.values())


def verifier_input(finding: dict[str, Any], *, transcript: str | None = None) -> dict[str, Any]:
    """The skeptic receives the finding. A transcript is an illegal edge."""
    if transcript:
        raise PlanError("verifier sees worker transcript")
    return {"finding": finding}


def layered_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def loop_until_dry(
    find_round: Callable[[int], list[str]],
    *,
    max_dry: int = 2,
    max_iter: int = 10,
    token_budget: int = 100,
    cost_per_round: int = 1,
) -> dict[str, Any]:
    """Mark a finding seen when it is found, then stop on three stacked conditions."""
    if max_dry < 1 or max_iter < 1 or token_budget < 1 or cost_per_round < 1:
        raise ValueError("stops must be positive")
    seen: set[str] = set()
    confirmed: list[str] = []
    dry = 0
    iterations = 0
    spent = 0
    while dry < max_dry and iterations < max_iter and spent + cost_per_round <= token_budget:
        found = find_round(iterations)
        iterations += 1
        spent += cost_per_round
        fresh: list[str] = []
        for bug in found:
            if bug in seen:
                continue
            seen.add(bug)
            fresh.append(bug)
        if not fresh:
            dry += 1
            continue
        dry = 0
        confirmed.extend(fresh)
    return {
        "confirmed": confirmed,
        "seen": seen,
        "iterations": iterations,
        "dry": dry,
        "spent": spent,
    }


def check_edge(edge: dict[str, Any]) -> None:
    """Refuse an illegal edge at plan time. Return means the edge may run."""
    kind = edge["kind"]
    if kind == "verify" and edge.get("includes_transcript"):
        raise PlanError("verifier sees worker transcript")
    if kind == "synthesize" and edge.get("item_count", 0) > edge.get("window_limit", 0):
        raise PlanError("raw pile exceeds the window")
    if kind == "write" and edge.get("path") in set(edge.get("owned_paths") or []):
        raise PlanError("two writers one file")
    if kind == "cycle" and not edge.get("seen_on_discovery"):
        raise PlanError("cycle without seen-set")
    if kind == "fanout" and not fanout_pays(edge["p"], edge["n"]):
        raise PlanError("fan-out does not pay")
    if kind == "map" and not edge.get("schema"):
        raise PlanError("no output schema")


def scale_gate(*, found_new: bool, verifier_caught: bool, cost_justified: bool) -> bool:
    """Double the cap only when all three answers are yes."""
    return bool(found_new and verifier_caught and cost_justified)
