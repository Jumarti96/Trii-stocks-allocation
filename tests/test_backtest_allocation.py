import math
import os

import numpy as np
import pandas as pd
import pytest

from experiments.backtest_allocation import (
    realized_block_return, pairwise_turnover, max_drawdown,
    annualized_stats, summarize_arm,
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

    def test_max_drawdown_opening_decline(self):
        # A series starting with a loss must register it (equity anchored at 1.0).
        # r = [-0.1, 0.2]: equity [1.0, 0.9, 1.08]; max drawdown = 0.1 at the first block.
        assert abs(max_drawdown([-0.1, 0.2]) - 0.1) < 1e-9


class TestAnnualizedStats:
    def test_zero_vol_constant_returns(self):
        out = annualized_stats([0.02] * 12, blocks_per_year=12, rf_period=0.0)
        assert abs(out["cum_return"] - (1.02 ** 12 - 1)) < 1e-9
        assert abs(out["ann_return"] - (1.02 ** 12 - 1)) < 1e-9   # n == blocks_per_year
        assert abs(out["ann_vol"]) < 1e-15  # numerically zero
        assert math.isnan(out["sharpe"])

    def test_vol_and_sharpe(self):
        out = annualized_stats([0.1, -0.1], blocks_per_year=4, rf_period=0.0)
        assert abs(out["ann_vol"] - 0.2) < 1e-9       # std 0.1 * sqrt(4)
        assert abs(out["sharpe"] - 0.0) < 1e-9        # mean 0
        assert abs(out["cum_return"] - (1.1 * 0.9 - 1)) < 1e-9

    def test_empty(self):
        out = annualized_stats([], blocks_per_year=12, rf_period=0.0)
        assert math.isnan(out["ann_return"])


class TestSummarizeArm:
    def test_full_row(self):
        out = summarize_arm([0.1, -0.1, 0.1], [None, 0.2, 0.4], [3, 4, 3],
                            blocks_per_year=12, rf_period=0.0)
        assert set(out) >= {"cum_return", "ann_return", "ann_vol", "sharpe",
                            "max_dd", "mean_turnover", "avg_names", "hit_rate"}
        assert abs(out["mean_turnover"] - 0.3) < 1e-9      # mean of 0.2, 0.4 (None ignored)
        assert abs(out["avg_names"] - (10 / 3)) < 1e-9
        assert abs(out["hit_rate"] - (2 / 3)) < 1e-9       # 2 of 3 blocks positive


from experiments.backtest_allocation import equal_weight, label_of, compute_arm_weights


@pytest.fixture
def cov5():
    names = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.diag([0.04] * 5), index=names, columns=names)


def _mu5(vals):
    return pd.Series(dict(zip(["A", "B", "C", "D", "E"], vals)))


class TestLabelOf:
    def test_labels(self):
        assert label_of("current") == "current"
        assert label_of("empirical") == "empirical"
        assert label_of(("parametric", 2.0)) == "parametric_s2"
        assert label_of(("parametric", 1.0)) == "parametric_s1"


class TestComputeArmWeights:
    def test_current_sums_to_one(self, cov5):
        w = compute_arm_weights("current", _mu5([0.2, 0.02, 0.02, 0.02, 0.02]), None,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
        assert abs(w.sum() - 1.0) < 1e-6

    def test_equal_weight(self, cov5):
        w = compute_arm_weights("equal_weight", _mu5([0.1] * 5), None,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
        assert (abs(w - 0.2) < 1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-9

    def test_empirical_sums_to_one(self, cov5):
        per_run = [_mu5([0.2, 0.02, 0.02, 0.02, 0.02]), _mu5([0.02, 0.2, 0.02, 0.02, 0.02])]
        w = compute_arm_weights("empirical", _mu5([0.1] * 5), per_run,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
        assert abs(w.sum() - 1.0) < 1e-6

    def test_parametric_sums_to_one(self, cov5):
        w = compute_arm_weights(("parametric", 2.0), _mu5([0.2, 0.02, 0.02, 0.02, 0.02]),
                                None, cov5, CFG, 100, 500, np.random.default_rng(0))
        assert abs(w.sum() - 1.0) < 1e-6

    def test_unknown_arm_raises(self, cov5):
        with pytest.raises(ValueError):
            compute_arm_weights("bogus", _mu5([0.1] * 5), None,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
