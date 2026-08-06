"""Foundations helpers for the stock / data track (pandas drills).

Requires optional extras: ``poetry install -E track-data``
"""

from __future__ import annotations

try:
    import numpy as np
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pandas/numpy required. Install with: poetry install -E track-data"
    ) from e


def create_series() -> pd.Series:
    return pd.Series([1, 3, 5, np.nan, 6, 8])


def create_dataframe(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("20250101", periods=6)
    return pd.DataFrame(
        rng.standard_normal((6, 4)), index=dates, columns=list("ABCD")
    )


def create_dataframe_from_dict(data: dict | None = None) -> pd.DataFrame:
    if data is None:
        data = {
            "A": 1.0,
            "B": pd.Timestamp("20130102"),
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    return pd.DataFrame(data)
