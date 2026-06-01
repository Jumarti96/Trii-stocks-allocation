"""
Sweep n_transformer_runs and measure how composition stability and per-stock forecast
variance converge as more runs are averaged.

Approach: train max(grid) runs once per iteration, then derive mu(n) for each grid point
by averaging the first n runs (controlled comparison). Reuses the stability instrument's
allocation + metric helpers.

Run:  python experiments/sweep_transformer_runs.py --iterations 30 --grid 10,20,30,40,50,100
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

See docs/superpowers/specs/2026-05-31-transformer-runs-convergence-sweep-design.md.
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
sys.path.insert(0, HERE)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS
from measure_allocation_stability import (
    select_stocks, allocate_msr, mean_turnover, mean_jaccard,
    weight_dispersion, seed_everything,
)


def run_sweep(prices, rets, cfg, iterations, grid, seed,
              train_runs_fn=None, winsorize_fn=None, period_mu_fn=None,
              select_fn=None, seed_fn=None, verbose=True):
    """Run `iterations` passes; per pass, train max(grid) runs once and derive mu(n)
    from the first-n prefix for each n in grid. Returns
    {"selected": [...], "by_n": {n: {"mu": DataFrame, "weights": DataFrame}}}.

    The five *_fn arguments are dependency-injection seams (lazy real defaults) so the
    loop is unit-testable without torch. With verbose=True, prints per-iteration elapsed
    time and ETA (streams to the log when run in the background).
    """
    if train_runs_fn is None or winsorize_fn is None or period_mu_fn is None:
        from transformer_model import train_runs, winsorize_to_history, weighted_mean_return
        train_runs_fn = train_runs_fn or train_runs
        winsorize_fn = winsorize_fn or winsorize_to_history
        period_mu_fn = period_mu_fn or weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    grid = sorted(grid)
    max_n = max(grid)

    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    mu_recs = {n: [] for n in grid}
    w_recs = {n: [] for n in grid}
    start = time.time()
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        runs = train_runs_fn(rets, cfg, n_runs=max_n)
        for n in grid:
            prefix = runs[:n].mean(axis=0)
            preds_df = winsorize_fn(pd.DataFrame(prefix, columns=rets.columns), rets)
            mu = period_mu_fn(preds_df)
            mu_sel = mu.loc[selected]
            weights = allocate_msr(mu_sel, cov_sel, cfg)
            mu_recs[n].append(mu_sel)
            w_recs[n].append(weights)
        if verbose:
            elapsed = time.time() - start
            eta = elapsed / i * (iterations - i)
            print(f"[iter {i}/{iterations}] elapsed {elapsed:6.0f}s | ETA {eta:6.0f}s", flush=True)

    by_n = {
        n: {
            "mu": pd.DataFrame(mu_recs[n]).reset_index(drop=True),
            "weights": pd.DataFrame(w_recs[n]).reset_index(drop=True),
        }
        for n in grid
    }
    return {"selected": selected, "by_n": by_n}
