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


def _fake_arm(block_returns):
    n = len(block_returns)
    return {"block_returns": block_returns,
            "turnover": [None] + [0.1] * (n - 1),
            "n_held": [3] * n,
            "weights": [pd.Series([1.0], index=["A"])] * n,
            "dates": [f"2020-{i+1:02d}" for i in range(n)]}


def test_aggregate_across_seeds_mean_and_std():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=1)
    seed0 = {"current": _fake_arm([0.02, 0.02, 0.02]),
             "equal_weight": _fake_arm([0.01, 0.01, 0.01]),
             "rebalance_index": [10, 11, 12]}
    seed1 = {"current": _fake_arm([0.04, 0.04, 0.04]),
             "equal_weight": _fake_arm([0.01, 0.01, 0.01]),
             "rebalance_index": [10, 11, 12]}

    agg = meb.aggregate_across_seeds({0: seed0, 1: seed1}, cfg, rebalance_every=1)

    assert agg["n_blocks"] == 3
    table = agg["table"]
    # constant-return arms -> zero vol -> NaN sharpe; check cum_return mean/std instead
    cur_mean = table.loc["current", "cum_return_mean"]
    cur_std = table.loc["current", "cum_return_std"]
    # seed0 cum = 1.02^3-1 ~ 0.0612; seed1 cum = 1.04^3-1 ~ 0.1249
    assert cur_mean == pytest.approx((0.061208 + 0.124864) / 2, abs=1e-4)
    assert cur_std == pytest.approx(abs(0.061208 - 0.124864) / 2, abs=1e-4)


def test_format_monthly_summary_content():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=6)
    seed0 = {"current": _fake_arm([0.02] * 9),
             "equal_weight": _fake_arm([0.01] * 9),
             "rebalance_index": list(range(9))}
    agg = meb.aggregate_across_seeds({0: seed0}, cfg, rebalance_every=6)
    text = meb.format_monthly_summary({6: agg})

    assert "horizon 6-month" in text
    assert "blocks: 9" in text
    assert "DIRECTIONAL" in text          # <15 blocks caveat fires
    assert "current" in text
    assert "equal_weight" in text
    assert "filter DISABLED" in text
