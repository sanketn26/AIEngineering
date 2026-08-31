"""Tiny stdlib MLP + scaled-dot attention. No GPU, no training loop.

This is a *shape* demo: forward passes you can test. The 90-day track
adds leakage-safe data, dual-path fusion, and honest ablations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tiny_data.json"


def load_tiny_data() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def matvec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [dot(row, vec) for row in matrix]


def relu(xs: list[float]) -> list[float]:
    return [max(0.0, x) for x in xs]


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def mlp_forward(
    x: list[float],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> list[float]:
    """One hidden layer: relu(W1 x + b1) then W2 h + b2."""
    h = relu([u + v for u, v in zip(matvec(w1, x), b1, strict=True)])
    return [u + v for u, v in zip(matvec(w2, h), b2, strict=True)]


def attention(
    query: list[float],
    keys: list[list[float]],
    values: list[list[float]],
) -> list[float]:
    """Single-head scaled dot-product over a tiny sequence."""
    d = len(query)
    scale = math.sqrt(d) or 1.0
    scores = [dot(query, k) / scale for k in keys]
    weights = softmax(scores)
    out = [0.0] * len(values[0])
    for w, val in zip(weights, values, strict=True):
        for i, v in enumerate(val):
            out[i] += w * v
    return out


def default_xor_weights() -> dict:
    """Fixed weights that separate the four XOR corners well enough for a test.

    Hand-set, not trained. The track's job is to *train* and to beat
    MLP-only / attention-only with a measured fusion — later.
    """
    # Hidden units approximate AND / OR / NAND-style planes.
    w1 = [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ]
    b1 = [-1.5, -0.5, -0.5, -0.5]
    w2 = [[-4.0, 1.0, 1.0, 1.0]]
    b2 = [-0.2]
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def predict_label(x: list[float], weights: dict | None = None) -> int:
    weights = weights or default_xor_weights()
    logit = mlp_forward(x, weights["w1"], weights["b1"], weights["w2"], weights["b2"])[0]
    return 1 if logit > 0 else 0
