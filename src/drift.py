"""Module 23 — prompts and agent config as versioned artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.reliability import state_checksum as canonical_hash


@dataclass(frozen=True)
class PromptConfig:
    """A prompt is deployable config: id, version, body, decoding, tools."""

    prompt_id: str
    version: str
    template: str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 512
    tools: tuple[str, ...] = ()
    policy_version: str = "v1"

    def bundle(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "template": self.template,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": list(self.tools),
            "policy_version": self.policy_version,
        }

    def digest(self) -> str:
        return canonical_hash(self.bundle())


@dataclass
class ConfigSnapshot:
    """Pinned production (or golden) hashes keyed by prompt_id."""

    env: str
    pins: dict[str, str] = field(default_factory=dict)

    def pin(self, cfg: PromptConfig) -> None:
        self.pins[cfg.prompt_id] = cfg.digest()


@dataclass
class DriftFinding:
    prompt_id: str
    expected_hash: str
    actual_hash: str
    kind: str  # missing | changed | extra


def detect_drift(
    snapshot: ConfigSnapshot, live: dict[str, PromptConfig]
) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    for pid, expected in snapshot.pins.items():
        cfg = live.get(pid)
        if cfg is None:
            findings.append(DriftFinding(pid, expected, "", "missing"))
            continue
        actual = cfg.digest()
        if actual != expected:
            findings.append(DriftFinding(pid, expected, actual, "changed"))
    for pid, cfg in live.items():
        if pid not in snapshot.pins:
            findings.append(DriftFinding(pid, "", cfg.digest(), "extra"))
    return findings


def eval_regression(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    floor: float = -0.03,
) -> dict[str, Any]:
    """Compare named golden metrics. Silent quality drop is a failed gate."""
    regressions = []
    for name, base in baseline.items():
        if name not in candidate:
            regressions.append({"metric": name, "reason": "missing"})
            continue
        delta = candidate[name] - base
        if delta < floor:
            regressions.append(
                {
                    "metric": name,
                    "baseline": base,
                    "candidate": candidate[name],
                    "delta": round(delta, 4),
                }
            )
    return {
        "ok": not regressions,
        "regressions": regressions,
        "n_metrics": len(baseline),
    }
