"""
Tests for the pandas_intro module.
"""
import pytest
import pandas as pd
import numpy as np
from src.pandas_intro import create_series, create_dataframe, create_dataframe_from_dict


def test_create_series():
    """Test create_series function."""
    result = create_series()
    
    # Verify it returns a pandas Series
    assert isinstance(result, pd.Series)
    
    # Verify the length
    assert len(result) == 6
    
    # Verify specific values (excluding NaN)
    assert result.iloc[0] == 1
    assert result.iloc[1] == 3
    assert result.iloc[2] == 5
    assert result.iloc[4] == 6
    assert result.iloc[5] == 8
    
    # Verify NaN at index 3
    assert pd.isna(result.iloc[3])


def test_create_dataframe():
    """Test create_dataframe function."""
    result = create_dataframe()
    
    # Verify it returns a pandas DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verify shape (6 rows, 4 columns)
    assert result.shape == (6, 4)
    
    # Verify column names
    expected_columns = ['A', 'B', 'C', 'D']
    assert list(result.columns) == expected_columns
    
    # Verify index is DatetimeIndex starting from 2025-01-01
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result.index) == 6
    assert result.index[0] == pd.Timestamp('2025-01-01')
    
    # Verify data types (should be float64 for random data)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])


def test_create_dataframe_from_dict_default():
    """Test create_dataframe_from_dict with default data."""
    result = create_dataframe_from_dict()
    
    # Verify it returns a pandas DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verify columns
    expected_columns = ['A', 'B', 'C', 'D', 'E', 'F']
    assert list(result.columns) == expected_columns
    
    # Verify shape (4 rows for default data)
    assert result.shape == (4, 6)
    
    # Verify specific column data types and values
    assert result['A'].dtype == 'float64'
    assert all(result['A'] == 1.0)
    
    assert pd.api.types.is_datetime64_any_dtype(result['B'])
    assert all(result['B'] == pd.Timestamp("20130102"))
    
    assert result['C'].dtype == 'float32'
    assert all(result['C'] == 1.0)
    
    assert result['D'].dtype == 'int32'
    assert all(result['D'] == 3)
    
    assert isinstance(result['E'].dtype, pd.CategoricalDtype)
    expected_categories = ["test", "train", "test", "train"]
    assert list(result['E']) == expected_categories
    
    assert result['F'].dtype == 'object'
    assert all(result['F'] == "foo")


def test_create_dataframe_from_dict_custom():
    """Test create_dataframe_from_dict with custom data."""
    custom_data = {
        'X': [1, 2, 3],
        'Y': ['a', 'b', 'c'],
        'Z': [10.5, 20.5, 30.5]
    }
    
    result = create_dataframe_from_dict(custom_data)
    
    # Verify it returns a pandas DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verify columns
    expected_columns = ['X', 'Y', 'Z']
    assert list(result.columns) == expected_columns
    
    # Verify shape
    assert result.shape == (3, 3)
    
    # Verify data
    assert list(result['X']) == [1, 2, 3]
    assert list(result['Y']) == ['a', 'b', 'c']
    assert list(result['Z']) == [10.5, 20.5, 30.5]


def test_create_series_data_consistency():
    """Test that create_series returns consistent data across calls."""
    result1 = create_series()
    result2 = create_series()
    
    # Both should be Series with same length
    assert isinstance(result1, pd.Series)
    assert isinstance(result2, pd.Series)
    assert len(result1) == len(result2) == 6
    
    # Non-NaN values should be identical
    for i in [0, 1, 2, 4, 5]:
        assert result1.iloc[i] == result2.iloc[i]
    
    # Both should have NaN at index 3
    assert pd.isna(result1.iloc[3])
    assert pd.isna(result2.iloc[3])


def test_create_dataframe_structure_consistency():
    """Test that create_dataframe returns consistent structure across calls."""
    result1 = create_dataframe()
    result2 = create_dataframe()
    
    # Structure should be identical
    assert result1.shape == result2.shape
    assert list(result1.columns) == list(result2.columns)
    assert result1.index.equals(result2.index)
    
    # Data types should be consistent
    for col in result1.columns:
        assert result1[col].dtype == result2[col].dtype