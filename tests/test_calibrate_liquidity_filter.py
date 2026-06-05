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
    idx = ["p1", "p2", "p3", "p4", "p5"]
    close = pd.DataFrame({"A": [10] * 5, "B": [10] * 5}, index=idx)
    volume = pd.DataFrame({"A": [1, 1, 1, 1, 1],    # active 100%
                           "B": [1, 1, 1, 0, 0]},   # active 60%
                          index=idx)
    table = clf.sweep_thresholds(close, volume, window=5, grid=[0.5, 0.9])
    assert table.loc[0.5, "kept_total"] == 2     # both >= 50%
    assert table.loc[0.9, "kept_total"] == 1     # only A >= 90%
