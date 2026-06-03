import math
import os

import numpy as np
import pandas as pd
import pytest

from experiments.backtest_allocation import (
    realized_block_return, pairwise_turnover, max_drawdown,
)

CFG = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6,
       "min_weight": 0.05, "periods_per_year": 12}


class TestMetricHelpers:
    def test_block_return_single_asset(self):
        w = pd.Series({"A": 1.0})
        block = pd.DataFrame({"A": [0.1, 0.1]})
        assert abs(realized_block_return(w, block) - (1.1 * 1.1 - 1)) < 1e-9

    def test_block_return_two_assets(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        block = pd.DataFrame({"A": [0.1, 0.0], "B": [0.0, 0.0]})
        # A compounds to 0.10, B to 0.0 -> 0.5*0.1 + 0.5*0 = 0.05
        assert abs(realized_block_return(w, block) - 0.05) < 1e-9

    def test_block_return_ignores_unheld_columns(self):
        w = pd.Series({"A": 1.0})
        block = pd.DataFrame({"A": [0.0], "B": [0.5]})
        assert abs(realized_block_return(w, block) - 0.0) < 1e-9

    def test_turnover_identical_zero(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        assert pairwise_turnover(w, w) == 0.0

    def test_turnover_disjoint_one(self):
        assert abs(pairwise_turnover(pd.Series({"A": 1.0}), pd.Series({"B": 1.0})) - 1.0) < 1e-9

    def test_turnover_partial(self):
        assert abs(pairwise_turnover(pd.Series({"A": 1.0}),
                                     pd.Series({"A": 0.5, "B": 0.5})) - 0.5) < 1e-9

    def test_max_drawdown_monotonic_zero(self):
        assert max_drawdown([0.1, 0.1, 0.1]) == 0.0

    def test_max_drawdown_decline(self):
        # equity [1.5, 0.75]; peak 1.5; trough drawdown = 0.5
        assert abs(max_drawdown([0.5, -0.5]) - 0.5) < 1e-9

    def test_max_drawdown_empty(self):
        assert max_drawdown([]) == 0.0
