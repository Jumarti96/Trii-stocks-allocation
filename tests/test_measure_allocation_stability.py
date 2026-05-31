"""
Tests for experiments/measure_allocation_stability.py

Run with:  python -m pytest tests/test_measure_allocation_stability.py -v

Pure helpers are tested with synthetic data only — no files, network, or GPU.
The run_experiment loop is tested with injected stubs so torch is never imported.
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.measure_allocation_stability import allocate_msr


CFG = {"rf_rate": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}


@pytest.fixture
def three_assets():
    """Annualised returns + a diagonal covariance for three assets."""
    returns = pd.Series({"A": 0.20, "B": 0.12, "C": 0.02})
    covmat = pd.DataFrame(
        np.diag([0.04, 0.05, 0.06]), index=returns.index, columns=returns.index
    )
    return returns, covmat


@pytest.fixture
def five_assets_one_dominated():
    """Four equivalent good assets + one dominated asset (E).

    Equal returns/variance among A-D give ~0.25 each; E's strongly negative
    return forces it to ~0, so the elimination loop drops it (cumulative weight
    < min_weight) and re-optimises the remaining four (len > 2) back to weights
    that still sum to 1. With only three assets, dropping one would trip the
    `len(optimal) <= 2` guard before re-optimising, so >=5 assets are needed to
    exercise the elimination path cleanly.
    """
    returns = pd.Series({"A": 0.15, "B": 0.15, "C": 0.15, "D": 0.15, "E": -0.50})
    covmat = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04, 0.04, 0.04]),
        index=returns.index, columns=returns.index,
    )
    return returns, covmat


class TestAllocateMsr:
    def test_weights_sum_to_one(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_index_preserved(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert list(w.index) == list(returns.index)

    def test_respects_max_weight(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert (w <= CFG["max_weight"] + 1e-6).all()

    def test_no_negative_weights(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert (w >= -1e-8).all()

    def test_dominated_asset_eliminated(self, five_assets_one_dominated):
        # E has a strongly negative return; the optimizer zeroes it, the
        # elimination loop drops it (weight exactly 0), and the remaining four
        # re-optimise to weights that still sum to 1.
        returns, covmat = five_assets_one_dominated
        w = allocate_msr(returns, covmat, CFG)
        assert w["E"] == 0.0
        assert abs(w.sum() - 1.0) < 1e-6


from experiments.measure_allocation_stability import portfolio_metrics


class TestPortfolioMetrics:
    def test_known_values(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.20})
        covmat = pd.DataFrame(
            np.diag([0.04, 0.09]), index=["A", "B"], columns=["A", "B"]
        )
        m = portfolio_metrics(weights, returns, covmat, rf=0.02)
        assert abs(m["ret"] - 0.15) < 1e-10
        # vol = sqrt(0.25*0.04 + 0.25*0.09) = sqrt(0.0325)
        assert abs(m["vol"] - math.sqrt(0.0325)) < 1e-10
        assert abs(m["sharpe"] - (0.15 - 0.02) / math.sqrt(0.0325)) < 1e-10

    def test_uses_only_weight_index_names(self):
        # returns/covmat carry an extra name C that weights omit; it must be ignored.
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.20, "C": 9.0})
        covmat = pd.DataFrame(
            np.diag([0.04, 0.09, 1.0]), index=["A", "B", "C"], columns=["A", "B", "C"]
        )
        m = portfolio_metrics(weights, returns, covmat, rf=0.0)
        assert abs(m["ret"] - 0.15) < 1e-10


from experiments.measure_allocation_stability import (
    selection_frequency,
    weight_dispersion,
)


@pytest.fixture
def weights_df():
    """3 iterations x 3 names. C is held in 1 of 3 runs."""
    return pd.DataFrame(
        [
            {"A": 0.6, "B": 0.4, "C": 0.0},
            {"A": 0.5, "B": 0.5, "C": 0.0},
            {"A": 0.4, "B": 0.3, "C": 0.3},
        ]
    )


class TestSelectionFrequency:
    def test_fraction_held(self, weights_df):
        freq = selection_frequency(weights_df)
        assert freq["A"] == 1.0
        assert freq["B"] == 1.0
        assert abs(freq["C"] - 1 / 3) < 1e-12

    def test_returns_series_over_all_names(self, weights_df):
        freq = selection_frequency(weights_df)
        assert set(freq.index) == {"A", "B", "C"}


class TestWeightDispersion:
    def test_zero_for_constant_column(self):
        df = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.1, 0.2, 0.3]})
        disp = weight_dispersion(df)
        assert disp["A"] == 0.0
        assert disp["B"] > 0.0


from experiments.measure_allocation_stability import mean_turnover, mean_jaccard


class TestMeanTurnover:
    def test_two_disjoint_rows(self):
        # rows [1,0] and [0,1]: turnover = 0.5*(|1|+|1|) = 1.0
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        assert abs(mean_turnover(df) - 1.0) < 1e-12

    def test_identical_rows_zero(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}])
        assert mean_turnover(df) == 0.0

    def test_single_row_returns_none(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}])
        assert mean_turnover(df) is None


class TestMeanJaccard:
    def test_identical_sets_is_one(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"A": 0.4, "B": 0.6}])
        assert abs(mean_jaccard(df) - 1.0) < 1e-12

    def test_disjoint_sets_is_zero(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        assert mean_jaccard(df) == 0.0

    def test_single_row_returns_none(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}])
        assert mean_jaccard(df) is None


from experiments.measure_allocation_stability import (
    metric_dispersion,
    amplification_factor,
)


class TestMetricDispersion:
    def test_mean_std_cov(self):
        df = pd.DataFrame({"ret": [0.10, 0.20], "vol": [0.05, 0.05], "sharpe": [1.0, 3.0]})
        disp = metric_dispersion(df)
        assert abs(disp["ret"]["mean"] - 0.15) < 1e-12
        assert disp["vol"]["std"] == 0.0
        # CoV = std / |mean| ; sharpe mean=2, std(ddof=0)=1 -> 0.5
        assert abs(disp["sharpe"]["cov"] - 0.5) < 1e-12


class TestAmplificationFactor:
    def test_ratio_of_mean_stds(self):
        # mu std per name: A->0, B->0.5 ; mean = 0.25
        mu_df = pd.DataFrame({"A": [0.10, 0.10], "B": [0.10, 1.10]})
        # weight std per name: A->0.05, B->0.05 ; mean = 0.05
        w_df = pd.DataFrame({"A": [0.45, 0.55], "B": [0.55, 0.45]})
        amp = amplification_factor(mu_df, w_df)
        assert abs(amp - (0.05 / 0.25)) < 1e-9

    def test_zero_input_std_is_nan(self):
        mu_df = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, 0.2]})
        w_df = pd.DataFrame({"A": [0.4, 0.6], "B": [0.6, 0.4]})
        assert math.isnan(amplification_factor(mu_df, w_df))


from experiments.measure_allocation_stability import select_stocks


@pytest.fixture
def synthetic_prices_returns():
    """60 weekly points for 4 tickers: 2 uptrending, 2 downtrending."""
    np.random.seed(7)
    n = 60
    idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
    up = lambda drift: 100 * np.cumprod(1 + np.random.normal(drift, 0.01, n))
    prices = pd.DataFrame(
        {
            "UP1": up(0.01), "UP2": up(0.008),
            "DN1": up(-0.01), "DN2": up(-0.008),
        },
        index=idx,
    )
    returns = prices.pct_change().dropna()
    return prices, returns


class TestSelectStocks:
    def test_subset_of_columns(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        selected = select_stocks(prices, returns, cfg)
        assert set(selected).issubset(set(prices.columns))

    def test_deterministic(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        assert select_stocks(prices, returns, cfg) == select_stocks(prices, returns, cfg)

    def test_returns_only_names_present_in_returns(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        selected = select_stocks(prices, returns, cfg)
        assert all(name in returns.columns for name in selected)
