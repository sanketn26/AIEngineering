"""Tests for the pandas_intro module (requires poetry install -E track-data)."""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import pandas as pd  # noqa: E402

from src.pandas_intro import (  # noqa: E402
    create_dataframe,
    create_dataframe_from_dict,
    create_series,
)

pytestmark = pytest.mark.track_data


def test_create_series():
    result = create_series()
    assert isinstance(result, pd.Series)
    assert len(result) == 6
    assert result.iloc[0] == 1
    assert result.iloc[1] == 3
    assert result.iloc[2] == 5
    assert result.iloc[4] == 6
    assert result.iloc[5] == 8
    assert pd.isna(result.iloc[3])


def test_create_dataframe():
    result = create_dataframe()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (6, 4)
    assert list(result.columns) == ["A", "B", "C", "D"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result.index) == 6
    assert result.index[0] == pd.Timestamp("2025-01-01")
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])


def test_create_dataframe_from_dict_default():
    result = create_dataframe_from_dict()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["A", "B", "C", "D", "E", "F"]
    assert result.shape == (4, 6)
    assert result["A"].dtype == "float64"
    assert all(result["A"] == 1.0)
    assert pd.api.types.is_datetime64_any_dtype(result["B"])
    assert all(result["B"] == pd.Timestamp("20130102"))
    assert result["C"].dtype == "float32"
    assert all(result["C"] == 1.0)
    assert result["D"].dtype == "int32"
    assert all(result["D"] == 3)
    assert isinstance(result["E"].dtype, pd.CategoricalDtype)
    assert list(result["E"]) == ["test", "train", "test", "train"]
    assert result["F"].dtype == "object"
    assert all(result["F"] == "foo")


def test_create_dataframe_from_dict_custom():
    custom_data = {
        "X": [1, 2, 3],
        "Y": ["a", "b", "c"],
        "Z": [10.5, 20.5, 30.5],
    }
    result = create_dataframe_from_dict(custom_data)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["X", "Y", "Z"]
    assert result.shape == (3, 3)
    assert list(result["X"]) == [1, 2, 3]
    assert list(result["Y"]) == ["a", "b", "c"]
    assert list(result["Z"]) == [10.5, 20.5, 30.5]


def test_create_series_data_consistency():
    result1 = create_series()
    result2 = create_series()
    assert isinstance(result1, pd.Series)
    assert isinstance(result2, pd.Series)
    assert len(result1) == len(result2) == 6
    for i in [0, 1, 2, 4, 5]:
        assert result1.iloc[i] == result2.iloc[i]
    assert pd.isna(result1.iloc[3])
    assert pd.isna(result2.iloc[3])


def test_create_dataframe_structure_consistency():
    result1 = create_dataframe()
    result2 = create_dataframe()
    assert result1.shape == result2.shape
    assert list(result1.columns) == list(result2.columns)
    assert result1.index.equals(result2.index)
    for col in result1.columns:
        assert result1[col].dtype == result2[col].dtype
