"""Module 04 — golden-set evaluation helpers."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def exact_fields(
    pred: dict[str, Any], expect: dict[str, Any], fields: Iterable[str]
) -> bool:
    return all(pred.get(f) == expect.get(f) for f in fields)


def run_suite(
    rows: list[dict[str, Any]],
    predict_fn: Callable[[str], dict[str, Any]],
    fields: list[str],
    input_key: str = "input",
    expect_key: str = "expect",
) -> dict[str, Any]:
    """Run predict_fn over golden rows; return accuracy and failures."""
    if not rows:
        return {"accuracy": 0.0, "n": 0, "failures": []}
    ok = 0
    failures: list[dict[str, Any]] = []
    for row in rows:
        pred = predict_fn(row[input_key])
        expect = row[expect_key]
        if exact_fields(pred, expect, fields):
            ok += 1
        else:
            failures.append(
                {
                    "id": row.get("id"),
                    "input": row[input_key],
                    "expect": expect,
                    "pred": pred,
                }
            )
    n = len(rows)
    return {"accuracy": ok / n, "n": n, "passed": ok, "failures": failures}


def parse_success_rate(raw_outputs: list[str], parser: Callable[[str], Any]) -> float:
    if not raw_outputs:
        return 0.0
    ok = 0
    for raw in raw_outputs:
        try:
            parser(raw)
            ok += 1
        except Exception:  # noqa: BLE001
            pass
    return ok / len(raw_outputs)


def paired_release_gate(
    baseline: list[list[float]],
    candidate: list[list[float]],
    *,
    margin: float = 0.02,
    confidence: float = 0.95,
    resamples: int = 4000,
    seed: int = 7,
    min_cases: int = 20,
) -> dict[str, Any]:
    """Paired case-bootstrap interval on candidate minus baseline mean score.

    Each outer row is the SAME held-out case in A and B. Inner values are
    repeated model runs; they travel together when a case is resampled. Cases
    have equal weight. Use independent cases (group by customer/document first).
    This estimates case-sampling uncertainty, not future distribution drift.
    """
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("nonempty aligned cases required")
    if (
        not 0 <= margin < 1
        or not 0 < confidence < 1
        or resamples < 100
        or min_cases < 2
    ):
        raise ValueError("invalid interval or gate settings")
    for a, b in zip(baseline, candidate):
        if not a or len(a) != len(b):
            raise ValueError("each case needs equal, nonempty repeat counts")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*a, *b]):
            raise ValueError("scores must be finite and in [0, 1]")
    deltas = [
        statistics.mean(b) - statistics.mean(a) for a, b in zip(baseline, candidate)
    ]
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(
        statistics.mean(rng.choices(deltas, k=n)) for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    low = samples[int(tail * (resamples - 1))]
    high = samples[math.ceil((1 - tail) * (resamples - 1))]
    enough = n >= min_cases and all(len(row) >= 2 for row in baseline)
    degenerate = len(set(deltas)) == 1
    if not enough or degenerate:
        verdict = "inconclusive"
    elif low >= -margin:
        verdict = "pass"
    elif high < -margin:
        verdict = "regression"
    else:
        verdict = "inconclusive"
    return {
        "n_cases": n,
        "runs_per_case": [len(row) for row in baseline],
        "baseline": statistics.mean(map(statistics.mean, baseline)),
        "candidate": statistics.mean(map(statistics.mean, candidate)),
        "delta": statistics.mean(deltas),
        "interval": [low, high],
        "confidence": confidence,
        "margin": margin,
        "verdict": verdict,
        "ok": verdict == "pass",
        "seed": seed,
        "resamples": resamples,
        "warning": (
            "too few cases/repeats or no observed case variation"
            if not enough or degenerate
            else None
        ),
    }
