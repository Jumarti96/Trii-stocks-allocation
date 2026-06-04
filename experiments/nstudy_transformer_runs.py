"""
n_transformer_runs study: sweep n for the current (msr) arm and parametric-Michaud
s in {4, 8} (historical Sigma), reusing the train-once / first-n prefix-average trick.

Reports composition stability (turnover, Jaccard, overlap) and value dispersion
(ret/vol/sharpe CoV) vs n, aggregated across seeds as mean +/- cross-seed std.

This is in-sample: value is scored against the same averaged mu the current arm is the
Sharpe-argmax of, so read the composition metrics and the value CoVs, NOT the value
levels. The realized verdict is a separate (backtest) study.

Run:  python experiments/nstudy_transformer_runs.py --seeds 0,100 --iterations 10 \
        --grid 10,25,50,75,100 --spreads 4,8 --mc-draws 1000
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

See docs/superpowers/specs/2026-06-03-nstudy-transformer-runs-design.md.
"""

import argparse
import math
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
    select_stocks, allocate_msr, sample_mu_draws, resampled_allocate,
    portfolio_metrics, mean_turnover, mean_jaccard, overlap_stats,
    metric_dispersion, seed_everything,
)


def prefix_forecast(runs, n, rets, winsorize_fn, period_mu_fn):
    """Average the first `n` transformer runs into one per-stock mu vector.

    runs: ndarray (n_runs, periods_to_forecast, n_stocks). Mirrors the prefix trick in
    sweep_transformer_runs.py: average the first-n runs, winsorise to history, then reduce
    the forecast path to one per-stock number via period_mu_fn. Returns a Series over
    rets.columns.
    """
    prefix = runs[:n].mean(axis=0)
    preds_df = winsorize_fn(pd.DataFrame(prefix, columns=rets.columns), rets)
    return period_mu_fn(preds_df)


def parametric_arm_draws(mu_sel, cov_sel, n_periods, n_draws, spreads, rng_seed):
    """K Monte-Carlo mu draws per spread, all sharing the same z for a paired s-comparison.

    A fresh rng is re-seeded from rng_seed for every spread, so the standard-normal draws
    are identical across spreads and the only difference is the s scaling. Returns
    {spread: [Series, ...]} preserving mu_sel's index.
    """
    out = {}
    for s in spreads:
        rng = np.random.default_rng(rng_seed)
        out[s] = sample_mu_draws(mu_sel, cov_sel, n_periods, n_draws, s, rng)
    return out


def run_nstudy_seed(prices, rets, cfg, grid, iterations, seed, spreads, n_draws,
                    train_runs_fn=None, winsorize_fn=None, period_mu_fn=None,
                    select_fn=None, seed_fn=None, verbose=False):
    """Run `iterations` passes for one seed and collect per-(arm, n) weights + metrics.

    Sigma (Ledoit-Wolf) and the selected names are deterministic and computed once. Per
    iteration, train max(grid) runs once (train_runs_fn) and derive mu(n) from the first-n
    prefix for each n. The current arm allocates mu(n) via the msr elimination loop; each s
    arm draws K parametric mu vectors (paired across spreads) and forms the resampled
    consensus. All arms are scored against the same mu(n). The *_fn args are
    dependency-injection seams (lazy real defaults) so the loop runs without torch.

    Returns {"selected": [...], "arms": [...], "grid": [...],
             "data": {arm: {n: {"weights": DataFrame, "metrics": DataFrame}}}}.
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

    rf = cfg["rf_period"]
    grid = sorted(grid)
    max_n = max(grid)
    T = len(rets)

    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    arms = ["current"] + [f"s{int(s)}" for s in spreads]
    rec = {arm: {n: {"w": [], "m": []} for n in grid} for arm in arms}

    start = time.time()
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        runs = train_runs_fn(rets, cfg, n_runs=max_n)
        for n in grid:
            mu_sel = prefix_forecast(runs, n, rets, winsorize_fn, period_mu_fn).loc[selected]

            w_cur = allocate_msr(mu_sel, cov_sel, cfg)
            rec["current"][n]["w"].append(w_cur)
            rec["current"][n]["m"].append(
                portfolio_metrics(w_cur[w_cur.abs() > 1e-9], mu_sel, cov_sel, rf)
            )

            draws_by_s = parametric_arm_draws(
                mu_sel, cov_sel, T, n_draws, spreads, (seed + i, n)
            )
            for s in spreads:
                w_s, _ = resampled_allocate(draws_by_s[s], cov_sel, cfg)
                arm = f"s{int(s)}"
                rec[arm][n]["w"].append(w_s)
                rec[arm][n]["m"].append(
                    portfolio_metrics(w_s[w_s.abs() > 1e-9], mu_sel, cov_sel, rf)
                )
        if verbose:
            elapsed = time.time() - start
            eta = elapsed / i * (iterations - i)
            print(f"[seed {seed}] iter {i}/{iterations} | elapsed {elapsed:6.0f}s "
                  f"| ETA {eta:6.0f}s", flush=True)

    data = {
        arm: {
            n: {
                "weights": pd.DataFrame(rec[arm][n]["w"]).reset_index(drop=True),
                "metrics": pd.DataFrame(rec[arm][n]["m"]),
            }
            for n in grid
        }
        for arm in arms
    }
    return {"selected": selected, "arms": arms, "grid": grid, "data": data}
