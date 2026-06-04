import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import nstudy_transformer_runs as ns


def _identity_winsorize(preds_df, rets):
    return preds_df


def _mean_period_mu(preds_df):
    return preds_df.mean(axis=0)


def test_prefix_forecast_averages_first_n_runs():
    # 3 runs, 2 forecast periods, 2 stocks
    runs = np.array([
        [[1.0, 2.0], [1.0, 2.0]],   # run 0
        [[3.0, 4.0], [3.0, 4.0]],   # run 1
        [[9.0, 9.0], [9.0, 9.0]],   # run 2 (excluded when n=2)
    ])
    rets = pd.DataFrame(np.zeros((5, 2)), columns=["A", "B"])
    mu = ns.prefix_forecast(runs, 2, rets, _identity_winsorize, _mean_period_mu)
    # first 2 runs averaged: A=(1+3)/2=2, B=(2+4)/2=3; period-mean leaves them unchanged
    assert mu["A"] == pytest.approx(2.0)
    assert mu["B"] == pytest.approx(3.0)


def test_parametric_arm_draws_are_paired_across_spreads():
    names = ["A", "B", "C"]
    mu = pd.Series([0.01, 0.02, 0.03], index=names)
    cov = pd.DataFrame(np.eye(3) * 0.04, index=names, columns=names)
    draws = ns.parametric_arm_draws(
        mu, cov, n_periods=100, n_draws=5, spreads=[4.0, 8.0], rng_seed=(7, 25)
    )
    # same z reused -> s=8 perturbation is exactly 2x the s=4 perturbation
    for k in range(5):
        d4 = draws[4.0][k] - mu
        d8 = draws[8.0][k] - mu
        assert np.allclose(d8.values, 2.0 * d4.values)


@pytest.fixture
def tiny_cfg():
    return {
        "rf_period": 0.0,
        "max_weight": 0.5,
        "min_weight": 0.05,
        "periods_per_year": 54,
    }


def _make_stub_runs_fn(n_stocks, periods=4):
    """Deterministic per-iteration runs: shape (n_runs, periods, n_stocks)."""
    rng = np.random.default_rng(0)

    def _runs_fn(rets, cfg, n_runs=None, verbose=False):
        return rng.normal(0.01, 0.005, size=(n_runs, periods, n_stocks))

    return _runs_fn


def test_run_nstudy_seed_shapes_and_arms(tiny_cfg):
    cols = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.02, size=(40, 5)), columns=cols)
    prices = pd.DataFrame(np.ones((40, 5)), columns=cols)  # unused (select_fn injected)

    result = ns.run_nstudy_seed(
        prices, rets, tiny_cfg,
        grid=[2, 3], iterations=2, seed=0, spreads=[4.0, 8.0], n_draws=8,
        train_runs_fn=_make_stub_runs_fn(5),
        winsorize_fn=_identity_winsorize,
        period_mu_fn=_mean_period_mu,
        select_fn=lambda p, r, c: cols,
        seed_fn=lambda s: None,
    )

    assert result["arms"] == ["current", "s4", "s8"]
    assert result["grid"] == [2, 3]
    # weights: iterations x selected names; metrics: iterations x [ret, vol, sharpe]
    w = result["data"]["s4"][3]["weights"]
    m = result["data"]["s4"][3]["metrics"]
    assert w.shape == (2, 5)
    assert list(m.columns) == ["ret", "vol", "sharpe"]
    assert m.shape == (2, 3)
    assert result["data"]["current"][2]["weights"].shape == (2, 5)


def _fake_seed_result(arms, grid, weights_by_arm_n, metrics_by_arm_n):
    data = {
        arm: {
            n: {
                "weights": weights_by_arm_n[arm][n],
                "metrics": metrics_by_arm_n[arm][n],
            }
            for n in grid
        }
        for arm in arms
    }
    return {"selected": list(weights_by_arm_n[arms[0]][grid[0]].columns),
            "arms": arms, "grid": grid, "data": data}


def test_summarize_nstudy_mean_and_std_across_seeds():
    arms, grid = ["current"], [10]
    names = ["A", "B"]
    metrics = pd.DataFrame({"ret": [0.02, 0.02], "vol": [0.1, 0.1], "sharpe": [0.2, 0.2]})

    # seed 0: weights churn between iterations -> turnover 1.0
    w_churn = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=names)
    # seed 1: identical weights -> turnover 0.0
    w_same = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], columns=names)

    r0 = _fake_seed_result(arms, grid, {"current": {10: w_churn}}, {"current": {10: metrics}})
    r1 = _fake_seed_result(arms, grid, {"current": {10: w_same}}, {"current": {10: metrics}})

    summary = ns.summarize_nstudy({0: r0, 1: r1})
    mean = summary["mean"]
    std = summary["std"]
    # turnover mean over seeds = (1.0 + 0.0)/2 = 0.5; population std = 0.5
    assert mean.loc[("current", 10), "turnover"] == pytest.approx(0.5)
    assert std.loc[("current", 10), "turnover"] == pytest.approx(0.5)


def test_first_flattening_n_finds_first_small_delta():
    grid = [10, 25, 50, 75, 100]
    # big drops early, then flat from 75 on
    values = {10: 0.84, 25: 0.74, 50: 0.66, 75: 0.65, 100: 0.648}
    assert ns.first_flattening_n(values, grid, tol=0.05) == 75
    # never flattens
    steep = {10: 0.84, 25: 0.6, 50: 0.4, 75: 0.25, 100: 0.1}
    assert ns.first_flattening_n(steep, grid, tol=0.05) is None


def test_format_nstudy_summary_has_arms_and_advisory():
    arms, grid = ["current", "s4"], [10, 25]
    names = ["A", "B"]
    metrics = pd.DataFrame({"ret": [0.02, 0.02], "vol": [0.1, 0.1], "sharpe": [0.2, 0.2]})
    # churning weights -> turnover is non-zero (1.0) but identical across n -> flat
    w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=names)
    weights = {a: {n: w for n in grid} for a in arms}
    mets = {a: {n: metrics for n in grid} for a in arms}
    r = _fake_seed_result(arms, grid, weights, mets)
    summary = ns.summarize_nstudy({0: r})

    text = ns.format_nstudy_summary(summary, primary_arm="s4", tol=0.05)
    assert "current" in text
    assert "s4" in text
    assert "turnover" in text
    # s4 turnover is flat across n (both = 1.0) -> advisory fires at n=25
    assert "turnover flattens at n=25" in text


def test_write_nstudy_outputs_creates_files(tmp_path):
    arms, grid = ["current", "s4"], [10, 25]
    names = ["A", "B"]
    metrics = pd.DataFrame({"ret": [0.02, 0.02], "vol": [0.1, 0.1], "sharpe": [0.2, 0.2]})
    w = pd.DataFrame([[1.0, 0.0], [0.6, 0.4]], columns=names)
    weights = {a: {n: w for n in grid} for a in arms}
    mets = {a: {n: metrics for n in grid} for a in arms}
    r = _fake_seed_result(arms, grid, weights, mets)
    summary = ns.summarize_nstudy({0: r})

    outdir = str(tmp_path / "nstudy")
    paths = ns.write_nstudy_outputs({0: r}, summary, outdir)

    assert os.path.exists(paths["summary"])
    assert os.path.exists(os.path.join(outdir, "nstudy_current_seed0_weights.csv"))
    assert os.path.exists(os.path.join(outdir, "nstudy_table_s4.csv"))
    # raw weights reload: 2 iterations x 2 grid points = 4 rows, with n + iteration cols
    raw = pd.read_csv(os.path.join(outdir, "nstudy_s4_seed0_weights.csv"))
    assert len(raw) == 4
    assert {"n", "iteration"}.issubset(raw.columns)


def test_arg_parser_parses_lists():
    parser = ns.build_arg_parser()
    args = parser.parse_args(
        ["--seeds", "0,100", "--grid", "10,25,50", "--spreads", "4,8",
         "--iterations", "10", "--mc-draws", "1000"]
    )
    assert [int(x) for x in args.seeds.split(",")] == [0, 100]
    assert [int(x) for x in args.grid.split(",")] == [10, 25, 50]
    assert [float(x) for x in args.spreads.split(",")] == [4.0, 8.0]
    assert args.iterations == 10
    assert args.mc_draws == 1000
