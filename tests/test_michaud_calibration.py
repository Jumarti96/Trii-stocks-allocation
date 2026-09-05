"""Tests for the pure helpers in experiments/michaud_calibration.py.

All of these run without training a transformer or calling the optimiser, so the
harness's bookkeeping is verified independently of the expensive parts.

Run: .venv/Scripts/python.exe -m pytest tests/test_michaud_calibration.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from michaud_calibration import (
    rebalance_splits,
    turnover,
    drift_weights,
    realised_returns,
    effective_n,
    regime_tertiles,
    annual_cost_drag,
    net_of_cost_sharpe,
    downside_deviation,
    max_drawdown,
    bootstrap_best_spread,
)


# --- rebalance schedule ---------------------------------------------------

def test_rebalance_splits_windows_are_non_overlapping_and_in_range():
    splits = rebalance_splits(n_periods=521, horizon=24, min_train=200)
    assert splits == sorted(splits)
    assert all(s >= 200 for s in splits)
    assert all(s + 24 <= 521 for s in splits)          # full forward data
    assert all(b - a == 24 for a, b in zip(splits, splits[1:]))


def test_rebalance_splits_anchors_final_window_at_the_end():
    # Anchoring from the end keeps the most recent regime in the sample.
    splits = rebalance_splits(n_periods=521, horizon=24, min_train=200)
    assert splits[-1] == 521 - 24


def test_rebalance_splits_empty_when_history_too_short():
    assert rebalance_splits(n_periods=100, horizon=24, min_train=200) == []


# --- turnover -------------------------------------------------------------

def test_turnover_zero_for_identical_weights():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    assert turnover(w, w) == pytest.approx(0.0)


def test_turnover_one_for_disjoint_holdings():
    a = pd.Series({"A": 0.5, "B": 0.5})
    b = pd.Series({"C": 0.5, "D": 0.5})
    assert turnover(a, b) == pytest.approx(1.0)


def test_turnover_handles_partial_overlap():
    a = pd.Series({"A": 0.6, "B": 0.4})
    b = pd.Series({"A": 0.4, "C": 0.6})
    # |0.6-0.4| + |0.4-0| + |0-0.6| = 0.2 + 0.4 + 0.6 = 1.2 -> half = 0.6
    assert turnover(a, b) == pytest.approx(0.6)


# --- drift ----------------------------------------------------------------

def test_drift_weights_shift_toward_the_winner():
    w = pd.Series({"A": 0.5, "B": 0.5})
    fwd = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.0]})
    drifted = drift_weights(w, fwd)
    # A grows 10%, B flat -> 0.55 / 1.05 vs 0.50 / 1.05
    assert drifted["A"] == pytest.approx(0.55 / 1.05)
    assert drifted["B"] == pytest.approx(0.50 / 1.05)
    assert drifted.sum() == pytest.approx(1.0)


def test_drift_adjusted_turnover_differs_from_naive():
    # If drift is ignored, turnover is overstated: part of the gap between last
    # period's target and this period's target closes on its own.
    w_prev = pd.Series({"A": 0.5, "B": 0.5})
    w_new  = pd.Series({"A": 0.55, "B": 0.45})
    fwd = pd.DataFrame({"A": [0.20], "B": [0.0]})
    naive = turnover(w_prev, w_new)
    drift = turnover(drift_weights(w_prev, fwd), w_new)
    assert drift != pytest.approx(naive)
    assert drift < naive


def test_drift_weights_flat_returns_are_identity():
    w = pd.Series({"A": 0.4, "B": 0.6})
    fwd = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    pd.testing.assert_series_equal(drift_weights(w, fwd), w, check_names=False)


# --- realised returns -----------------------------------------------------

def test_realised_returns_first_period_is_weighted_average():
    w = pd.Series({"A": 0.5, "B": 0.5})
    fwd = pd.DataFrame({"A": [0.10], "B": [-0.02]})
    out = realised_returns(w, fwd)
    assert out.iloc[0] == pytest.approx(0.5 * 0.10 + 0.5 * -0.02)


def test_realised_returns_are_buy_and_hold_not_rebalanced():
    # Buy-and-hold lets the winner compound at its grown weight; a constantly
    # rebalanced portfolio would give 0.5*0.1 + 0.5*0.1 = 0.10 in period 2.
    w = pd.Series({"A": 0.5, "B": 0.5})
    fwd = pd.DataFrame({"A": [1.00, 0.10], "B": [0.00, 0.10]})
    out = realised_returns(w, fwd)
    # after p1: A = 1.0, B = 0.5, total 1.5. p2: both +10% -> 1.65
    assert out.iloc[0] == pytest.approx(0.5)
    assert out.iloc[1] == pytest.approx(1.65 / 1.5 - 1)


def test_realised_returns_ignore_unheld_names():
    w = pd.Series({"A": 1.0})
    fwd = pd.DataFrame({"A": [0.05], "Z": [-0.90]})
    assert realised_returns(w, fwd).iloc[0] == pytest.approx(0.05)


# --- concentration --------------------------------------------------------

def test_effective_n_equals_holding_count_when_equally_weighted():
    for k in (2, 5, 13):
        w = pd.Series(np.repeat(1 / k, k), index=[f"S{i}" for i in range(k)])
        assert effective_n(w) == pytest.approx(k)


def test_effective_n_falls_when_concentrated():
    spread = pd.Series([0.25, 0.25, 0.25, 0.25])
    conc   = pd.Series([0.70, 0.10, 0.10, 0.10])
    assert effective_n(conc) < effective_n(spread)


# --- regimes --------------------------------------------------------------

def test_regime_tertiles_partition_every_date_exactly_once():
    dates = list(range(13))
    groups = regime_tertiles(dates)
    assert len(groups) == 3
    flat = [d for g in groups for d in g]
    assert sorted(flat) == dates
    assert len(flat) == len(set(flat))


def test_regime_tertiles_preserve_time_order():
    groups = regime_tertiles(list(range(12)))
    assert max(groups[0]) < min(groups[1])
    assert max(groups[1]) < min(groups[2])


# --- net-of-cost selection ------------------------------------------------

def test_annual_cost_drag_scales_with_rebalance_frequency():
    # 24-period cadence on a 54-period year = 2.25 rebalances/yr.
    drag = annual_cost_drag(mean_turnover=0.70, cost=0.005,
                            periods_per_year=54, horizon=24)
    assert drag == pytest.approx(0.005 * 0.70 * (54 / 24))


def test_annual_cost_drag_is_zero_without_trading_or_cost():
    assert annual_cost_drag(0.0, 0.005, 54, 24) == pytest.approx(0.0)
    assert annual_cost_drag(0.70, 0.0, 54, 24) == pytest.approx(0.0)


def test_net_sharpe_equals_gross_when_costless():
    gross = net_of_cost_sharpe(ann_return=0.20, ann_vol=0.15, mean_turnover=0.7,
                               rf=0.11, cost=0.0, periods_per_year=54, horizon=24)
    assert gross == pytest.approx((0.20 - 0.11) / 0.15)


def test_net_sharpe_penalises_higher_turnover():
    kw = dict(ann_return=0.20, ann_vol=0.15, rf=0.11, cost=0.005,
              periods_per_year=54, horizon=24)
    assert (net_of_cost_sharpe(mean_turnover=0.9, **kw)
            < net_of_cost_sharpe(mean_turnover=0.2, **kw))


def test_net_sharpe_can_invert_a_gross_ranking():
    # The whole point of the rule: a higher-gross, higher-churn portfolio should
    # be able to lose to a lower-gross, low-churn one once costs are charged.
    kw = dict(ann_vol=0.15, rf=0.11, cost=0.02, periods_per_year=54, horizon=24)
    churny = net_of_cost_sharpe(ann_return=0.24, mean_turnover=0.95, **kw)
    calm   = net_of_cost_sharpe(ann_return=0.22, mean_turnover=0.10, **kw)
    assert churny < calm


def test_downside_deviation_ignores_upside():
    # Resampling is meant to protect the bad periods; a metric that punishes big
    # gains would credit it for the wrong thing.
    calm = pd.Series([0.01, 0.01, 0.01, -0.05])
    spiky = pd.Series([0.40, 0.30, 0.50, -0.05])
    assert downside_deviation(calm, 0.0) == pytest.approx(
        downside_deviation(spiky, 0.0))


def test_downside_deviation_zero_when_never_below_target():
    assert downside_deviation(pd.Series([0.02, 0.03, 0.01]), 0.0) == pytest.approx(0.0)


def test_downside_deviation_grows_with_worse_losses():
    mild = pd.Series([0.02, -0.01])
    harsh = pd.Series([0.02, -0.20])
    assert downside_deviation(harsh, 0.0) > downside_deviation(mild, 0.0)


def test_max_drawdown_of_monotonic_gains_is_zero():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_max_drawdown_matches_hand_computation():
    # +100% then -50% -> wealth 1, 2, 1 -> trough is 50% below the peak.
    assert max_drawdown(pd.Series([1.0, -0.5])) == pytest.approx(-0.5)


def _boot_frames(cols, per_period):
    """Realised-return and turnover frames for a set of spreads."""
    rets = pd.DataFrame({c: per_period[c] for c in cols})
    turn = pd.DataFrame({c: [0.4] * len(rets) for c in cols})
    return rets, turn


_BOOT_KW = dict(rf=0.11, cost=0.005, periods_per_year=54, horizon=24,
                n_boot=400, seed=0)


def test_bootstrap_probabilities_sum_to_one():
    rets, turn = _boot_frames([1.0, 4.0], {
        1.0: [0.10, 0.05, -0.02, 0.08, 0.03],
        4.0: [0.06, 0.04, 0.00, 0.05, 0.02]})
    p, se = bootstrap_best_spread(rets, turn, **_BOOT_KW)
    assert p.sum() == pytest.approx(1.0)
    assert set(p.index) == {1.0, 4.0}
    assert (se >= 0).all()


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    rets, turn = _boot_frames([1.0, 4.0], {
        1.0: [0.10, 0.05, -0.02, 0.08, 0.03],
        4.0: [0.06, 0.04, 0.00, 0.05, 0.02]})
    a, _ = bootstrap_best_spread(rets, turn, **_BOOT_KW)
    b, _ = bootstrap_best_spread(rets, turn, **_BOOT_KW)
    pd.testing.assert_series_equal(a, b)


def test_bootstrap_splits_evenly_between_identical_spreads():
    # The case that matters: if two settings are indistinguishable, neither
    # should be reported as the winner.
    same = [0.10, 0.05, -0.02, 0.08, 0.03]
    rets, turn = _boot_frames([1.0, 4.0], {1.0: same, 4.0: list(same)})
    p, _ = bootstrap_best_spread(rets, turn, **_BOOT_KW)
    assert p[1.0] == pytest.approx(0.5, abs=0.05)


def test_bootstrap_concentrates_on_a_dominant_spread():
    rets, turn = _boot_frames([1.0, 4.0], {
        1.0: [0.20, 0.19, 0.21, 0.20, 0.19],     # higher and steadier
        4.0: [0.01, -0.05, 0.02, -0.03, 0.00]})
    p, _ = bootstrap_best_spread(rets, turn, **_BOOT_KW)
    assert p[1.0] > 0.95


def test_bootstrap_standard_error_grows_with_dispersion():
    # Near-constant rather than exactly constant: zero dispersion gives zero vol,
    # which makes Sharpe genuinely undefined rather than merely precise.
    steady = [0.050, 0.052, 0.049, 0.051, 0.050]
    wild   = [0.40, -0.30, 0.35, -0.25, 0.20]
    _, se_a = bootstrap_best_spread(*_boot_frames([1.0], {1.0: steady}), **_BOOT_KW)
    _, se_b = bootstrap_best_spread(*_boot_frames([1.0], {1.0: wild}), **_BOOT_KW)
    assert se_b[1.0] > se_a[1.0]


def test_max_drawdown_uses_running_peak_not_first_value():
    # Rises to 1.5, falls to 1.2 -> 20% drawdown from the peak, not a gain vs start.
    dd = max_drawdown(pd.Series([0.5, -0.2]))
    assert dd == pytest.approx(-0.2)


def test_net_sharpe_nan_on_zero_vol():
    out = net_of_cost_sharpe(ann_return=0.2, ann_vol=0.0, mean_turnover=0.5,
                             rf=0.11, cost=0.005, periods_per_year=54, horizon=24)
    assert np.isnan(out)
