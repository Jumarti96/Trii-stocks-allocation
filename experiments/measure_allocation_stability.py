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
