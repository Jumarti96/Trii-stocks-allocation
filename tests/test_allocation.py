import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from allocation import msr_eliminate

CFG = {
    "rf_period": 0.0, "rf_rate": 0.0,
    "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12,
    "allocation_method": "parametric_michaud",
    "michaud_spread": 1.0, "michaud_mc_draws": 200, "michaud_seed": 0,
}


def _cov(names, var=0.04):
    return pd.DataFrame(np.diag([var] * len(names)), index=names, columns=names)


class TestMsrEliminate:
    def test_sums_to_one(self):
        names = ["A", "B", "C"]
        mu = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_returns_series_over_full_index(self):
        names = ["A", "B", "C"]
        mu = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert list(w.index) == names

    def test_respects_max_weight(self):
        names = ["A", "B", "C", "D", "E"]
        mu = pd.Series(dict(zip(names, [0.30, 0.05, 0.05, 0.05, 0.05])))
        w = msr_eliminate(mu, _cov(names), CFG)
        assert (w <= CFG["max_weight"] + 1e-6).all()

    def test_two_assets_both_held(self):
        names = ["A", "B"]
        mu = pd.Series({"A": 0.20, "B": 0.10})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert (w.abs() > 0).sum() == 2
        assert abs(w.sum() - 1.0) < 1e-6
