"""
Backtesting engine: walk-forward evaluation of allocation strategies.

Pure functions, no I/O -- the calling harness does the file reading, training
and reporting, the same way pipeline/03_allocate.py sits in front of allocation.py.

Two things here are deliberately separated that the pipeline conflates:

  * **Holding period vs forecast horizon.** The pipeline implicitly holds a book
    for periods_to_forecast. Here `cadence` (how often you rebalance) is distinct
    from `horizon` (how far the model forecasts), so a 24-step forecast can be
    held for 12 weeks -- a different strategy, with roughly double the turnover.

  * **Forecast value vs risk-model value.** gmv_weights uses only the covariance
    and ignores mu entirely. If it matches the model, the transformer's forecast
    is contributing nothing and the value sits in the Ledoit-Wolf estimate.

Statistical note: comparisons are PAIRED. Two long-only books drawn from the same
universe are ~98% correlated, so comparing return levels means reading a small
effect through the market's much larger swings. Differencing period-by-period
cancels the common move; measured on this universe that cut the standard
deviation of the comparison from 0.159 to 0.049, a ~10x variance reduction.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

# Re-exported so this module's public surface is unchanged: callers and tests keep
# importing the strategies from `backtesting`. They live in strategies.py because
# src/allocation.py now exposes the same rules as live allocation methods, and a
# strategy defined in two places drifts.
from strategies import (                                          # noqa: F401
    cap_weights, equal_weight_all, equal_weight_topn, gmv_weights,
    inverse_vol_weights, momentum_weights, random_weights,
)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def rebalance_schedule(n_periods, cadence, horizon, min_train):
    """Split indices for a walk-forward backtest, anchored at the end.

    cadence: periods between rebalances -- also the holding period, so windows are
             back-to-back and non-overlapping.
    horizon: how far the model forecasts. Only used to reserve room; it does not
             have to equal cadence.
    min_train: smallest training span considered trustworthy.

    Anchoring from the end keeps the most recent regime in the sample. Returns an
    ascending list; empty when history is too short.
    """
    splits = []
    s = n_periods - cadence
    while s >= min_train:
        splits.append(s)
        s -= cadence
    return splits[::-1]


# ---------------------------------------------------------------------------
# Luck control
#
# The strategies themselves live in strategies.py and are re-exported above; what
# stays here is the scoring of a result against a distribution of random books.
# ---------------------------------------------------------------------------

def random_percentile(value, draws):
    """Fraction of `draws` the model's value beats. 0.5 means indistinguishable."""
    draws = np.asarray(draws, dtype=float)
    return float((draws < value).mean())


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def paired_comparison(strategy, baseline):
    """Paired per-period comparison of two realised return series.

    Both series must be indexed by the same rebalance windows. Returns mean
    difference, its sd and standard error, t, p, win count, n, and the sample size
    that would give 80% power at the observed effect size.
    """
    a = pd.Series(strategy).reset_index(drop=True)
    b = pd.Series(baseline).reset_index(drop=True)
    d = (a - b).dropna()
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else 0.0

    if n > 1 and sd > 0:
        t, p = stats.ttest_1samp(d, 0.0)
    else:                                  # identical or single-observation series
        t, p = (0.0, 1.0) if abs(d.mean()) < 1e-15 else (np.inf, 0.0)

    mean = float(d.mean())
    n80 = (((1.96 + 0.84) * sd / abs(mean)) ** 2
           if sd > 0 and abs(mean) > 1e-15 else float('nan'))

    return {
        'n': n,
        'mean_diff': mean,
        'sd_diff': sd,
        'se_diff': se,
        't': float(t),
        'p': float(p),
        'wins': int((d > 0).sum()),
        'n_for_80_power': n80,
    }


# ---------------------------------------------------------------------------
# Portfolio mechanics
# ---------------------------------------------------------------------------

def turnover(w_prev, w_new):
    """One-way turnover: 0.5 * sum |w_new - w_prev| over the union of holdings."""
    idx = w_prev.index.union(w_new.index)
    a = w_prev.reindex(idx).fillna(0.0)
    b = w_new.reindex(idx).fillna(0.0)
    return float(0.5 * (b - a).abs().sum())


def drift_weights(weights, fwd_rets):
    """Weights after holding through fwd_rets without rebalancing."""
    growth = (1 + fwd_rets[weights.index]).prod()
    grown = weights * growth
    return grown / grown.sum()


def realised_returns(weights, fwd_rets):
    """Buy-and-hold portfolio return series over the holding period."""
    wealth = (1 + fwd_rets[weights.index]).cumprod()
    value = (wealth * weights).sum(axis=1)
    prev = value.shift(1).fillna(weights.sum())
    return value / prev - 1


def effective_n(weights):
    """Inverse Herfindahl: the number of equally weighted names this book resembles."""
    w = weights[weights > 0]
    return float(1.0 / (w ** 2).sum())


def downside_deviation(returns, target=0.0):
    """RMS of shortfalls below `target`; upside contributes nothing."""
    short = np.minimum(np.asarray(returns, dtype=float) - target, 0.0)
    return float(np.sqrt((short ** 2).mean()))


def max_drawdown(returns):
    """Deepest peak-to-trough decline of the compounded series (<= 0)."""
    wealth = (1 + pd.Series(list(returns))).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def regime_tertiles(dates):
    """Split an ordered date list into 3 contiguous, time-ordered groups."""
    n = len(dates)
    c1, c2 = n // 3, 2 * n // 3
    return [dates[:c1], dates[c1:c2], dates[c2:]]


def annual_cost_drag(mean_turnover, cost, periods_per_year, horizon):
    """Annual return give-up from trading: turnover x cost x rebalances per year."""
    return mean_turnover * cost * (periods_per_year / horizon)


def net_of_cost_sharpe(ann_return, ann_vol, mean_turnover, rf, cost,
                       periods_per_year, horizon):
    """Sharpe after charging turnover at `cost` per unit."""
    if ann_vol <= 0 or np.isnan(ann_vol):
        return float('nan')
    net = ann_return - annual_cost_drag(mean_turnover, cost, periods_per_year, horizon)
    return (net - rf) / ann_vol


def bootstrap_best_spread(period_returns, turnovers, rf, cost, periods_per_year,
                          horizon, n_boot=2000, seed=0):
    """How often each column wins on net Sharpe, resampling the rebalance periods.

    Resamples the SAME period indices across every column, so the comparison stays
    paired. A diffuse result means the data cannot separate the settings.
    """
    rng = np.random.default_rng(seed)
    cols = list(period_returns.columns)
    n = len(period_returns)
    rets = period_returns[cols].to_numpy(dtype=float)
    turn = turnovers[cols].to_numpy(dtype=float)

    wins = np.zeros(len(cols))
    sharpes = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, t = rets[idx], turn[idx]
        ann = np.prod(1 + r, axis=0) ** (periods_per_year / (n * horizon)) - 1
        vol = r.std(axis=0, ddof=0) * np.sqrt(periods_per_year / horizon)
        net = ann - annual_cost_drag(t.mean(axis=0), cost, periods_per_year, horizon)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(vol > 0, (net - rf) / vol, np.nan)
        sharpes[b] = s
        if not np.all(np.isnan(s)):
            # Break ties at random: np.nanargmax would always return the first
            # index, reporting indistinguishable settings as a clean winner.
            tied = np.flatnonzero(np.isclose(s, np.nanmax(s), rtol=1e-12, atol=1e-15))
            wins[tied[0] if len(tied) == 1 else rng.choice(tied)] += 1

    return (pd.Series(wins / n_boot, index=cols),
            pd.Series(np.nanstd(sharpes, axis=0, ddof=1), index=cols))
