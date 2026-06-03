"""
Walk-forward backtest comparing allocation methods on realized out-of-sample returns.

At each monthly rebalance the transformer is retrained on data up to that date (expanding
window); every arm reuses that one forecast. Arms: current, parametric Michaud (s sweep),
empirical Michaud, equal-weight. Frictionless headline; per-rebalance turnover and weights
are saved so net-of-cost analysis is a free follow-up.

Run:  python experiments/backtest_allocation.py --oos-periods 162 --rebalance-every 4 --n-runs 50
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).
"""

import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS
from measure_allocation_stability import (
    allocate_msr, resampled_allocate, sample_mu_draws,
    select_stocks, train_runs_as_preds, seed_everything,
)


def realized_block_return(weights, block_rets):
    """Buy-and-hold realized return of `weights` over a block of actual per-period returns.

    weights: Series of target weights over held names. block_rets: DataFrame (periods x names)
    of ACTUAL per-period simple returns for the block. Each held name compounds over the block;
    the portfolio return is the weighted sum of per-name compounded returns. Only names present
    in both weights and block_rets are used. Returns a float.
    """
    names = [n for n in weights.index if n in block_rets.columns]
    compounded = (1.0 + block_rets[names]).prod(axis=0) - 1.0
    return float((weights[names] * compounded).sum())


def pairwise_turnover(w_prev, w_new):
    """Half the L1 distance between two target-weight Series.

    Aligns on the union of names (missing = 0). Returns a float in [0, 1].
    """
    names = w_prev.index.union(w_new.index)
    a = w_prev.reindex(names, fill_value=0.0)
    b = w_new.reindex(names, fill_value=0.0)
    return float(0.5 * (a - b).abs().sum())


def max_drawdown(block_returns):
    """Maximum drawdown of the equity curve built by compounding `block_returns`.

    Returns the max peak-to-trough decline as a non-negative fraction (0.2 = 20% drop).
    0.0 if the curve never declines or the input is empty.
    """
    r = np.asarray(block_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / running_max
    return float(drawdowns.max())
