import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import monthly_env_backtest as meb


def _weekly_cfg():
    return {
        "rf_rate": 0.11, "max_weight": 0.5, "min_weight": 0.05,
        "periods_per_year": 54, "interval": "1wk",
        "time_window": 54, "periods_to_forecast": 4,
    }


def test_build_monthly_cfg_sets_monthly_fields():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=6)
    assert cfg["interval"] == "1mo"
    assert cfg["periods_per_year"] == 12
    assert cfg["time_window"] == 12
    assert cfg["periods_to_forecast"] == 6
    assert cfg["period_freq"] == "M"
    assert cfg["future_freq"] == "MS"
    assert cfg["rf_period"] == pytest.approx((1 + 0.11) ** (1 / 12) - 1)
    # original weekly cfg is not mutated
    assert _weekly_cfg()["periods_per_year"] == 54


def test_select_all_returns_every_name():
    rets = pd.DataFrame(np.zeros((3, 4)), columns=["A", "B", "C", "D"])
    assert meb.select_all(None, rets, {}) == ["A", "B", "C", "D"]


def _stub_runs_fn(n_stocks):
    """Deterministic per-run forecasts: list of (periods_to_forecast x n_stocks) frames."""
    rng = np.random.default_rng(0)

    def _runs_fn(rets, cfg, n_runs=None, verbose=False):
        h = cfg["periods_to_forecast"]
        return [pd.DataFrame(rng.normal(0.01, 0.005, size=(h, n_stocks)),
                             columns=list(rets.columns)) for _ in range(n_runs)]
    return _runs_fn


def _mean_period_mu(preds_df):
    return preds_df.mean(axis=0)


def test_run_monthly_env_structure_and_arms():
    cols = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.04, size=(30, 6)), columns=cols)
    prices = pd.DataFrame(100 * (1 + rets).cumprod().values, columns=cols)

    out = meb.run_monthly_env(
        prices, rets, _weekly_cfg(),
        horizons=[1], seeds=[0], oos_periods=6, n_runs=2, mc_draws=8, spreads=[4.0],
        runs_fn=_stub_runs_fn(6), period_mu_fn=_mean_period_mu, seed_fn=lambda s: None,
    )

    assert set(out.keys()) == {1}
    assert out[1]["cfg"]["periods_to_forecast"] == 1
    res = out[1]["per_seed"][0]
    labels = [k for k in res if k != "rebalance_index"]
    assert labels == ["current", "parametric_s4", "equal_weight"]
    # filter off -> equal_weight holds the full 6-name universe
    assert res["equal_weight"]["n_held"][0] == 6
