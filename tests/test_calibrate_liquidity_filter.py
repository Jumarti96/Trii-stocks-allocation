import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import calibrate_liquidity_filter as clf


def test_sweep_thresholds_counts_survivors():
    idx = ["p1", "p2", "p3", "p4"]
    close = pd.DataFrame({
        "US0000000001": [10, 10, 10, 10], "US0000000002": [10, 10, 10, 10],
        "US0000000003": [10, 10, 10, 10], "US0000000004": [10, 10, 10, 10],
        "US0000000005": [10, 10, 10, 10],
    }, index=idx)
    volume = pd.DataFrame({
        "US0000000001": [100, 100, 100, 100], "US0000000002": [100, 100, 100, 100],
        "US0000000003": [100, 100, 100, 100], "US0000000004": [100, 100, 100, 100],
        "US0000000005": [1, 1, 1, 1],          # tiny -> dropped at high thresholds
    }, index=idx)
    table = clf.sweep_thresholds(close, volume, window=4, grid=[0.0, 0.5], min_group_size=5)
    assert table.loc[0.0, "kept_total"] == 5
    assert table.loc[0.5, "kept_total"] == 4      # the tiny name drops at 50% of median
