"""Routing policy: the model supplies scores, ordinary code owns the decision.

Thresholds live here, where they can be tested and changed — not in a prompt.
Pick them from labelled data (``python -m decision.calibration``), not by feel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from decision.scoring import TRIAGE_CHOICES, Choice, Scorer, build_prompt

Route = Literal["automate", "escalate", "human_review"]


@dataclass(frozen=True)
class Policy:
    min_top: float = 0.80  # top choice must hold at least this share
    min_margin: float = 0.20  # ...and lead the runner-up by at least this much
    escalate_floor: float = 0.50  # below this, a larger model will not help; ask a human
    human_choices: frozenset[str] = frozenset({"other"})


@dataclass(frozen=True)
class Decision:
    choice: str
    probabilities: dict[str, float]
    route: Route
    scorer: str


def route(probabilities: dict[str, float], policy: Policy) -> Route:
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top_choice, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_choice in policy.human_choices:
        return "human_review"
    if top >= policy.min_top and top - second >= policy.min_margin:
        return "automate"
    if top >= policy.escalate_floor:
        return "escalate"
    return "human_review"


def decide(
    ticket: str,
    scorer: Scorer,
    *,
    choices: tuple[Choice, ...] = TRIAGE_CHOICES,
    policy: Policy = Policy(),
    rotations: int = 1,
) -> Decision:
    """Score ``choices``; with ``rotations > 1``, average over cyclic reorderings.

    Small models prefer some label letters regardless of content. Rotating the
    choice order moves each choice through different letters, so averaging per
    choice cancels a letter bias. Costs ``rotations`` forward passes, no decode.
    """
    rotations = max(1, min(rotations, len(choices)))
    totals = {c.value: 0.0 for c in choices}
    for r in range(rotations):
        order = choices[r:] + choices[:r]
        prompt, labels = build_prompt(ticket, order)
        for c, s in zip(order, scorer.score(prompt, labels), strict=True):
            totals[c.value] += float(s) / rotations
    probabilities = {c.value: totals[c.value] for c in choices}
    choice = max(probabilities, key=probabilities.get)
    return Decision(choice, probabilities, route(probabilities, policy), scorer.name)
