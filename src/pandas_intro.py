import numpy as np
import pandas as pd

def create_series():
    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    return s

def create_dataframe():
    dates = pd.date_range("20250101", periods=6)
    return pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

def create_dataframe_from_dict(data = None):
    if data is None:
        data =  {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
    return pd.DataFrame(data)
