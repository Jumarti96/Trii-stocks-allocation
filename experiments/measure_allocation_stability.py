"""
Measure portfolio allocation stability across repeated predict->allocate runs.

Only mu (the Transformer expected returns) changes between runs; the Ledoit-Wolf
covariance and the technical-filter selection are deterministic and computed once.
This instrument runs N iterations and reports how much the *composition* moves
(selection frequency, turnover, Jaccard, weight std) relative to how much the
*value* moves (return/vol/Sharpe dispersion), plus an input->output amplification.

Self-contained (Approach B): the selection and MSR elimination logic are
re-implemented here to mirror pipeline/03_filter.py and pipeline/04_allocate.py.
Keep them in sync if those steps change.

Run:  python experiments/measure_allocation_stability.py --iterations 30 --transformer-runs 10
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

Phase 1 (measurement) only. Phase 2 assesses compound-annualisation of mu — see
docs/superpowers/specs/2026-05-31-allocation-stability-measurement-design.md.
"""

import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import risk_kit as rk
from config import load_config, PATHS


def allocate_msr(returns, covmat, cfg):
    """Sharpe-maximising weights with the step-4 batch-elimination loop.

    Mirrors pipeline/04_allocate.py. Returns a Series over `returns.index`;
    names eliminated for failing the min-weight floor get weight 0.0.
    """
    rf = cfg["rf_rate"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    names = list(returns.index)

    w0 = rk.msr_tuned(
        riskfree_rate=rf, returns=returns, covmat=covmat,
        max_weight=max_w, periods_per_year=ppy, debug=False,
    )
    optimal = (
        pd.DataFrame(w0, index=returns.index, columns=["Weights"])
        .sort_values("Weights")
    )

    while optimal["Weights"].sum() >= 0.9999:
        cum_weights = optimal["Weights"].cumsum()
        failing_mask = cum_weights < min_w
        if not failing_mask.any():
            break
        optimal = optimal[~failing_mask]
        if len(optimal) <= 2:
            break
        w = rk.msr_tuned(
            riskfree_rate=rf, returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
        )
        optimal = (
            pd.DataFrame(w, index=optimal.index, columns=["Weights"])
            .sort_values("Weights")
        )

    weights = pd.Series(0.0, index=names)
    weights[optimal.index] = optimal["Weights"]
    return weights


def portfolio_metrics(weights, returns, covmat, rf):
    """Portfolio return/vol/Sharpe over the names in `weights.index`.

    Computed exactly as msr_tuned's objective does (annualised mu against the
    per-period covariance) so the numbers match the optimiser's convention.
    """
    names = list(weights.index)
    w = weights.values
    r = returns.loc[names].values
    C = covmat.loc[names, names].values
    ret = float(rk.portfolio_return(w, r))
    vol = float(rk.portfolio_vol(w, C))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return {"ret": ret, "vol": vol, "sharpe": sharpe}


def selection_frequency(weights_df, eps=1e-9):
    """Fraction of iterations in which each name is held (weight > eps)."""
    return (weights_df.abs() > eps).mean(axis=0)


def weight_dispersion(weights_df):
    """Population std of each name's weight across iterations."""
    return weights_df.std(axis=0, ddof=0)


def mean_turnover(weights_df):
    """Mean over all iteration pairs of 0.5 * L1 weight distance.

    None if fewer than 2 iterations.
    """
    if len(weights_df) < 2:
        return None
    W = weights_df.values
    n = len(W)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0.5 * np.abs(W[i] - W[j]).sum()
            count += 1
    return total / count


def mean_jaccard(weights_df, eps=1e-9):
    """Mean pairwise Jaccard similarity of the held-name sets.

    None if fewer than 2 iterations.
    """
    if len(weights_df) < 2:
        return None
    held = weights_df.abs() > eps
    rows = [set(held.columns[held.iloc[i].values]) for i in range(len(held))]
    sims = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(sims))


def metric_dispersion(metrics_df):
    """mean / population-std / coefficient-of-variation for each metric column."""
    out = {}
    for col in metrics_df.columns:
        s = metrics_df[col]
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        cov = std / abs(mean) if mean != 0 else float("nan")
        out[col] = {"mean": mean, "std": std, "cov": cov}
    return out


def amplification_factor(mu_df, weights_df):
    """Mean per-name weight std divided by mean per-name mu std.

    A rough indicator of how much the optimiser magnifies forecast jitter.
    NaN if the input (mu) dispersion is zero.
    """
    mu_disp = float(mu_df.std(axis=0, ddof=0).mean())
    w_disp = float(weights_df.std(axis=0, ddof=0).mean())
    return w_disp / mu_disp if mu_disp > 0 else float("nan")


def select_stocks(prices, rets, cfg):
    """Technical-filter selection — mirrors pipeline/03_filter.py.

    Keeps tickers with at least cfg['signal_min_count'] positive signals out of
    SMA/EMA/MACD/PRC. Returns the kept names that also exist in `rets`.
    """
    ma_terms = cfg["ma_terms"]
    rows = []
    for ticker in prices.columns:
        sig = rk.technical_indicators(
            prices[ticker],
            indicators=["SMA", "EMA", "MACD", "PRC"],
            ma_terms=ma_terms,
            macd_params=[12, 26, 9],
            return_df=True,
            plot=False,
            signal_tolerance=0.975,
        ).iloc[-1]
        sig_df = pd.DataFrame(sig).T
        sig_df.index = [ticker]
        sig_df.rename(columns={ticker: "Price"}, inplace=True)
        rows.append(sig_df)

    signals = pd.concat(rows, axis=0)
    keep = signals[
        np.int64(signals["MACD Signal"])
        + np.int64(signals[f"SMA{ma_terms} Signal"])
        + np.int64(signals[f"EMA{ma_terms} Signal"])
        + np.int64(signals["PRC Signal"])
        >= cfg["signal_min_count"]
    ]
    return [t for t in keep.index if t in rets.columns]


def seed_everything(seed):
    """Seed numpy and torch (torch imported lazily to keep helpers light)."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
