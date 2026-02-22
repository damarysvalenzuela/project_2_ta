"""
data.py — Load and preprocess BTC 5-minute OHLC data.
"""

import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_train = pd.read_csv("data/btc_project_train.csv")
    data_test  = pd.read_csv("data/btc_project_test.csv")
    return data_train, data_test


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Parse datetimes, set index, sort, deduplicate, and drop rows with
    missing OHLC values.
    """
    data = raw_data.copy()

    data.index = pd.to_datetime(
        data["Datetime"].values,
        format="mixed",
        dayfirst=True,
    )
    data.index.name = "Datetime"

    # Keep only the columns the backtest/indicators need
    cols_keep = ["Open", "High", "Low", "Close"]
    if "Timestamp" in data.columns:
        cols_keep = ["Timestamp"] + cols_keep
    data = data[[c for c in cols_keep if c in data.columns]].copy()

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="first")]
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    return data