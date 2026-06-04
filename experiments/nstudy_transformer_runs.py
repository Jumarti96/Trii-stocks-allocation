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
