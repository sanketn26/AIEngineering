"""Gate 6 (stretch): fixed-answer scoring as the triage decision path.

See README.md in this directory. Nothing here changes the five planted holes.
"""

from decision.policy import Decision, Policy, decide, route
from decision.scoring import (
    TRIAGE_CHOICES,
    Choice,
    LabelTokenError,
    MLXScorer,
    MockScorer,
    TransformersScorer,
    build_prompt,
    resolve_label_ids,
    restricted_softmax,
)

__all__ = [
    "TRIAGE_CHOICES",
    "Choice",
    "Decision",
    "LabelTokenError",
    "MLXScorer",
    "MockScorer",
    "Policy",
    "TransformersScorer",
    "build_prompt",
    "decide",
    "resolve_label_ids",
    "restricted_softmax",
    "route",
]
