"""Tests for experiments/sweep_transformer_runs.py

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py -v

The sweep loop is tested with injected stubs so torch is never imported.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.sweep_transformer_runs import run_sweep


def _inputs():
    rng = np.random.RandomState(2)
    cols = ["A", "B", "C", "D"]
    rets = pd.DataFrame(rng.normal(0.001, 0.02, (40, 4)), index=range(40), columns=cols)
    prices = (1 + rets).cumprod() * 100
    cfg = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05,
           "periods_per_year": 12}
    return prices, rets, cfg


def _stubs():
    # train_runs_fn returns runs where run k is the constant k across (periods, stocks),
    # so the mean of the first n runs is (n-1)/2 everywhere -> prefix-averaging is checkable.
    def train_runs_fn(rets, cfg, n_runs=None):
        periods, stocks = 2, rets.shape[1]
        return np.stack([np.full((periods, stocks), float(k)) for k in range(n_runs)])

    def winsorize_fn(preds_df, rets):
        return preds_df

    def period_mu_fn(preds_df):
        return preds_df.mean(axis=0)

    def select_fn(prices, rets, cfg):
        return ["A", "B", "C", "D"]

    def seed_fn(seed):
        pass

    return train_runs_fn, winsorize_fn, period_mu_fn, select_fn, seed_fn


class TestRunSweep:
    def test_structure_and_shapes(self):
        prices, rets, cfg = _inputs()
        tr, wz, pm, sel, sd = _stubs()
        res = run_sweep(prices, rets, cfg, iterations=3, grid=[10, 20, 50], seed=0,
                        train_runs_fn=tr, winsorize_fn=wz, period_mu_fn=pm,
                        select_fn=sel, seed_fn=sd, verbose=False)
        assert sorted(res["by_n"].keys()) == [10, 20, 50]
        assert res["selected"] == ["A", "B", "C", "D"]
        for n in (10, 20, 50):
            assert res["by_n"][n]["mu"].shape == (3, 4)
            assert res["by_n"][n]["weights"].shape == (3, 4)

    def test_prefix_averaging_uses_first_n_runs(self):
        # mu(n) = mean over the first n runs (= (n-1)/2 for the stub) -> identical per stock.
        prices, rets, cfg = _inputs()
        tr, wz, pm, sel, sd = _stubs()
        res = run_sweep(prices, rets, cfg, iterations=2, grid=[10, 20], seed=0,
                        train_runs_fn=tr, winsorize_fn=wz, period_mu_fn=pm,
                        select_fn=sel, seed_fn=sd, verbose=False)
        assert np.allclose(res["by_n"][10]["mu"].values, (10 - 1) / 2)
        assert np.allclose(res["by_n"][20]["mu"].values, (20 - 1) / 2)


from experiments.sweep_transformer_runs import summarize_sweep


def _mk(weights_rows, mu_rows):
    return {"weights": pd.DataFrame(weights_rows), "mu": pd.DataFrame(mu_rows)}


def _decreasing_result():
    # mu across-iteration std strictly shrinks with n; metrics keep changing >5% each
    # step, so no point "converges" (converged_n is None).
    by_n = {
        10: _mk([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 0.0}],
                [{"A": 0.20, "B": 0.05}, {"A": 0.05, "B": 0.20}, {"A": 0.20, "B": 0.05}]),
        20: _mk([{"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, {"A": 0.55, "B": 0.45}],
                [{"A": 0.120, "B": 0.100}, {"A": 0.100, "B": 0.120}, {"A": 0.120, "B": 0.100}]),
        30: _mk([{"A": 0.52, "B": 0.48}, {"A": 0.50, "B": 0.50}, {"A": 0.51, "B": 0.49}],
                [{"A": 0.112, "B": 0.108}, {"A": 0.108, "B": 0.112}, {"A": 0.112, "B": 0.108}]),
    }
    return {"selected": ["A", "B"], "by_n": by_n}


def _converged_result():
    # n=20 and n=30 have IDENTICAL weights/mu -> every metric matches at the 20->30 step
    # (relative delta 0), so converged_n is 30. n=10 differs enough that 20 is not converged.
    n10 = _mk([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 0.0}],
              [{"A": 0.20, "B": 0.05}, {"A": 0.05, "B": 0.20}, {"A": 0.20, "B": 0.05}])
    common = _mk([{"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, {"A": 0.55, "B": 0.45}],
                 [{"A": 0.12, "B": 0.10}, {"A": 0.10, "B": 0.12}, {"A": 0.11, "B": 0.11}])
    common2 = _mk(common["weights"].copy(), common["mu"].copy())
    return {"selected": ["A", "B"], "by_n": {10: n10, 20: common, 30: common2}}


class TestSummarizeSweep:
    def test_table_columns_and_index(self):
        t = summarize_sweep(_decreasing_result())["table"]
        assert list(t.index) == [10, 20, 30]
        for c in ("turnover", "jaccard", "mean_mu_std", "d_turnover", "d_jaccard", "d_mu_std"):
            assert c in t.columns

    def test_mu_std_decreases_with_n(self):
        t = summarize_sweep(_decreasing_result())["table"]
        assert t.loc[10, "mean_mu_std"] > t.loc[20, "mean_mu_std"] > t.loc[30, "mean_mu_std"]

    def test_converged_n_detected(self):
        # 20 and 30 are identical -> all three metrics unchanged at the 20->30 step.
        assert summarize_sweep(_converged_result())["converged_n"] == 30

    def test_steady_descent_does_not_converge(self):
        # Every step still changes >5% -> no converged point.
        assert summarize_sweep(_decreasing_result())["converged_n"] is None

    def test_single_grid_point_returns_none(self):
        one = {"selected": ["A", "B"], "by_n": {
            10: _mk([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}],
                    [{"A": 0.2, "B": 0.0}, {"A": 0.0, "B": 0.2}])}}
        assert summarize_sweep(one)["converged_n"] is None
