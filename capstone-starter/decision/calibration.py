"""Is 0.91 right 91% of the time? Measure it.

    python -m decision.calibration                      # mock scorer
    python -m decision.calibration --backend transformers   # real model
    python -m decision.calibration --backend mlx            # Apple silicon

Reports accuracy, expected calibration error (ECE), a reliability table, and
the smallest ``min_top`` whose automated slice meets a target accuracy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "evals" / "decision.jsonl"


@dataclass(frozen=True)
class Scored:
    id: str
    expect: str
    choice: str
    confidence: float

    @property
    def correct(self) -> bool:
        return self.choice == self.expect


def load_rows(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reliability(items: list[Scored], bins: int = 5) -> list[dict]:
    """Per confidence bin: count, mean confidence, observed accuracy."""
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [s for s in items if lo <= s.confidence < hi or (b == bins - 1 and s.confidence == 1.0)]
        if not members:
            continue
        table.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(members),
                "confidence": sum(s.confidence for s in members) / len(members),
                "accuracy": sum(s.correct for s in members) / len(members),
            }
        )
    return table


def expected_calibration_error(items: list[Scored], bins: int = 5) -> float:
    """Weighted mean |confidence - accuracy| across bins. 0 is perfect."""
    n = len(items)
    return sum(row["n"] / n * abs(row["confidence"] - row["accuracy"]) for row in reliability(items, bins))


def sweep(items: list[Scored], thresholds: list[float]) -> list[dict]:
    """Coverage and accuracy of the slice you would automate at each threshold."""
    out = []
    for t in thresholds:
        kept = [s for s in items if s.confidence >= t]
        out.append(
            {
                "min_top": t,
                "coverage": len(kept) / len(items),
                "accuracy": (sum(s.correct for s in kept) / len(kept)) if kept else None,
                "n": len(kept),
            }
        )
    return out


def pick_threshold(items: list[Scored], target: float, min_n: int = 5) -> float | None:
    """Lowest threshold (most automation) whose automated slice meets ``target``."""
    grid = [round(0.05 * i, 2) for i in range(4, 20)]
    for row in sweep(items, grid):
        if row["n"] >= min_n and row["accuracy"] is not None and row["accuracy"] >= target:
            return row["min_top"]
    return None


def score_dataset(scorer, rows: list[dict], rotations: int = 1) -> list[Scored]:
    from decision.policy import decide

    items = []
    for row in rows:
        d = decide(row["input"], scorer, rotations=rotations)
        items.append(Scored(row["id"], row["expect"], d.choice, d.probabilities[d.choice]))
    return items


def main(argv: list[str] | None = None) -> None:
    from decision.scoring import MLXScorer, MockScorer, TransformersScorer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("mock", "transformers", "mlx"), default="mock")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--rotations", type=int, default=1, help="average over N cyclic choice orders (cancels letter bias)")
    p.add_argument("--target", type=float, default=0.95)
    args = p.parse_args(argv)

    backends = {"mock": lambda: MockScorer(), "transformers": lambda: TransformersScorer(args.model), "mlx": lambda: MLXScorer(args.model)}
    scorer = backends[args.backend]()
    items = score_dataset(scorer, load_rows(), args.rotations)
    acc = sum(s.correct for s in items) / len(items)
    print(f"scorer={scorer.name} rotations={args.rotations} n={len(items)} accuracy={acc:.3f} ece={expected_calibration_error(items):.3f}")
    print("\nreliability (confidence vs observed accuracy):")
    for row in reliability(items):
        print(f"  {row['bin']}  n={row['n']:>3}  conf={row['confidence']:.2f}  acc={row['accuracy']:.2f}")
    print(f"\nlowest min_top reaching {args.target:.0%} on the automated slice: {pick_threshold(items, args.target)}")
    for s in items:
        if not s.correct:
            print(f"  miss {s.id}: expected {s.expect}, got {s.choice} @ {s.confidence:.2f}")


if __name__ == "__main__":
    main()
