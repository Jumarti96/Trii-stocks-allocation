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
