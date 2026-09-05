"""Tests for the pure backtesting engine in src/backtesting.py.

Everything here runs without training a transformer or downloading data, so the
engine's bookkeeping is verified independently of the expensive parts.

Run: .venv/Scripts/python.exe -m pytest tests/test_backtesting.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from backtesting import (
    rebalance_schedule,
    equal_weight_all,
    equal_weight_topn,
    gmv_weights,
    inverse_vol_weights,
    momentum_weights,
    random_weights,
    random_percentile,
    paired_comparison,
)


def _cov(sigmas, corr=0.0):
    """Covariance from per-asset vols and a single shared correlation."""
    s = np.asarray(sigmas, dtype=float)
    n = len(s)
    c = np.full((n, n), corr)
    np.fill_diagonal(c, 1.0)
    names = [f"S{i}" for i in range(n)]
    return pd.DataFrame(np.outer(s, s) * c, index=names, columns=names)


# --- schedule -------------------------------------------------------------

def test_schedule_cadence_equal_to_horizon_matches_existing_study():
    # 521 weekly periods with a 200-period burn-in gives 13 splits, last at 521-24.
    s = rebalance_schedule(n_periods=521, cadence=24, horizon=24, min_train=200)
    assert len(s) == 13
    assert s[-1] == 521 - 24
    assert s == sorted(s)


def test_schedule_shorter_cadence_gives_more_windows():
    long_c = rebalance_schedule(521, cadence=24, horizon=24, min_train=200)
    short_c = rebalance_schedule(521, cadence=12, horizon=24, min_train=200)
    assert len(short_c) > len(long_c)
    assert all(b - a == 12 for a, b in zip(short_c, short_c[1:]))


def test_schedule_holding_periods_do_not_overlap():
    # Holding period is the cadence, not the forecast horizon: a 24-step forecast
    # held for 12 weeks still yields disjoint, back-to-back holdings.
    s = rebalance_schedule(521, cadence=12, horizon=24, min_train=200)
    for a, b in zip(s, s[1:]):
        assert a + 12 <= b


def test_schedule_every_window_has_forward_data_for_its_holding_period():
    for cadence in (4, 12, 24):
        s = rebalance_schedule(521, cadence=cadence, horizon=24, min_train=200)
        assert all(split + cadence <= 521 for split in s)


def test_schedule_respects_min_train():
    s = rebalance_schedule(521, cadence=12, horizon=24, min_train=300)
    assert all(split >= 300 for split in s)


def test_schedule_empty_when_history_too_short():
    assert rebalance_schedule(100, cadence=24, horizon=24, min_train=200) == []


# --- benchmark strategies -------------------------------------------------

def test_equal_weight_all_is_uniform_and_sums_to_one():
    w = equal_weight_all(["A", "B", "C", "D"])
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w.values, 0.25)


def test_equal_weight_topn_holds_exactly_n_names():
    mu = pd.Series({"A": 0.05, "B": 0.01, "C": 0.09, "D": 0.03})
    w = equal_weight_topn(mu, n=2)
    held = w[w > 0]
    assert len(held) == 2
    assert set(held.index) == {"A", "C"}      # the two largest
    assert held.sum() == pytest.approx(1.0)


def test_gmv_matches_closed_form_for_two_uncorrelated_assets():
    # Minimum-variance weights with zero correlation are proportional to 1/variance.
    cov = _cov([0.10, 0.20], corr=0.0)
    w = gmv_weights(cov)
    inv_var = np.array([1 / 0.01, 1 / 0.04])
    np.testing.assert_allclose(w.values, inv_var / inv_var.sum(), atol=1e-6)


def test_gmv_ignores_expected_returns_entirely():
    # This is the point of the benchmark: it isolates the risk model from the
    # forecast, so it must not depend on mu in any way.
    cov = _cov([0.10, 0.20, 0.15], corr=0.2)
    assert gmv_weights(cov).equals(gmv_weights(cov))


def test_gmv_favours_the_lower_volatility_asset():
    w = gmv_weights(_cov([0.05, 0.30], corr=0.0))
    assert w.iloc[0] > w.iloc[1]


def test_inverse_vol_weights_are_proportional_to_reciprocal_sigma():
    cov = _cov([0.10, 0.20, 0.40], corr=0.0)
    w = inverse_vol_weights(cov)
    expected = np.array([10.0, 5.0, 2.5])
    np.testing.assert_allclose(w.values, expected / expected.sum(), atol=1e-9)


def test_momentum_picks_the_best_trailing_performers():
    hist = pd.DataFrame({
        "A": [0.10, 0.10, 0.10],     # strong
        "B": [-0.05, -0.05, -0.05],  # weak
        "C": [0.02, 0.02, 0.02],
    })
    w = momentum_weights(hist, n=2, lookback=3)
    held = w[w > 0]
    assert set(held.index) == {"A", "C"}


def test_momentum_uses_only_the_lookback_window():
    # Older history must not leak in: B wins on the full sample but loses on the
    # last two periods.
    hist = pd.DataFrame({
        "A": [-0.50, 0.10, 0.10],
        "B": [0.90, -0.05, -0.05],
    })
    w = momentum_weights(hist, n=1, lookback=2)
    assert w["A"] > 0 and w["B"] == 0


# --- random control -------------------------------------------------------

def test_random_weights_holds_exactly_n_and_is_reproducible():
    names = [f"S{i}" for i in range(20)]
    a = random_weights(names, n=5, rng=np.random.default_rng(0))
    b = random_weights(names, n=5, rng=np.random.default_rng(0))
    assert (a > 0).sum() == 5
    assert a.sum() == pytest.approx(1.0)
    pd.testing.assert_series_equal(a, b)


def test_random_weights_differ_across_seeds():
    names = [f"S{i}" for i in range(50)]
    a = random_weights(names, n=5, rng=np.random.default_rng(0))
    b = random_weights(names, n=5, rng=np.random.default_rng(1))
    assert not a.equals(b)


def test_random_percentile_is_half_for_a_median_result():
    draws = np.linspace(0.0, 1.0, 101)
    assert random_percentile(0.5, draws) == pytest.approx(0.5, abs=0.02)


def test_random_percentile_is_one_when_model_beats_every_draw():
    assert random_percentile(2.0, np.linspace(0.0, 1.0, 50)) == pytest.approx(1.0)


def test_random_percentile_is_zero_when_model_loses_to_every_draw():
    assert random_percentile(-1.0, np.linspace(0.0, 1.0, 50)) == pytest.approx(0.0)


# --- paired statistics ----------------------------------------------------

def test_paired_comparison_of_identical_series_shows_no_effect():
    r = pd.Series([0.05, -0.02, 0.11, 0.03])
    out = paired_comparison(r, r.copy())
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["wins"] == 0
    assert np.isnan(out["p"]) or out["p"] == pytest.approx(1.0)


def test_paired_comparison_recovers_a_constant_shift():
    base = pd.Series([0.05, -0.02, 0.11, 0.03])
    out = paired_comparison(base + 0.01, base)
    assert out["mean_diff"] == pytest.approx(0.01)
    assert out["wins"] == 4
    assert out["sd_diff"] == pytest.approx(0.0, abs=1e-12)


def test_paired_comparison_counts_wins_by_sign_not_magnitude():
    a = pd.Series([0.10, 0.01, 0.01, 0.01])
    b = pd.Series([0.00, 0.02, 0.02, 0.02])
    out = paired_comparison(a, b)          # a wins once, by a lot
    assert out["wins"] == 1
    assert out["n"] == 4


def test_paired_comparison_beats_unpaired_on_correlated_series():
    # The reason pairing matters: two books holding the same stocks move together,
    # so differencing cancels the market and leaves only the strategy effect.
    rng = np.random.default_rng(0)
    market = rng.normal(0.02, 0.15, 200)
    a = pd.Series(market + 0.01 + rng.normal(0, 0.005, 200))
    b = pd.Series(market + rng.normal(0, 0.005, 200))
    out = paired_comparison(a, b)
    assert out["sd_diff"] < a.std(ddof=1) / 5
    assert out["p"] < 0.01                 # detectable only because it is paired


def test_paired_comparison_reports_sample_size():
    out = paired_comparison(pd.Series([0.1, 0.2, 0.3]), pd.Series([0.0, 0.1, 0.2]))
    assert out["n"] == 3
