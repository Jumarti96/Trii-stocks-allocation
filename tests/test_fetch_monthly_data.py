import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import fetch_monthly_data as fmd


def test_clean_drops_missing_and_builds_returns():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    raw = pd.DataFrame({
        "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "B": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
        "C": [5.0, np.nan, 7.0, 8.0, 9.0, 10.0],   # 1 NaN of 6 (>15%) -> dropped
    }, index=idx)

    prices, rets = fmd.clean_to_prices_returns(raw, period_freq="M")

    assert list(prices.columns) == ["A", "B"]        # C dropped
    assert list(rets.columns) == ["A", "B"]
    assert len(rets) == 5                             # one row lost to pct_change
    assert rets.index[0] == "2020-02"                # string YYYY-MM index
    assert rets["A"].iloc[0] == np.float64(11.0 / 10.0 - 1.0)
