"""
Transformer architecture comparison study.

Walk-forward forecast quality evaluation of five transformer architectures across
4/24/54-week horizons. Metrics: Spearman rho, ICIR, Top-150 precision, hit rate.
No portfolio allocation.

Pre-condition: data/01_returns.csv must exist (run pipeline/01_download.py first).

Usage:
    "C:/Python projects/Finance/Scripts/python.exe" experiments/arch_comparison.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

from transformer_model import train_runs, build_arch, ARCH_DECODE_STEPS, _MULTISTEP_ARCHS
from config import load_config, PATHS

# --- Experiment constants ---
ARCHITECTURES = ['current', 'A_surgical', 'B_4', 'B_24', 'C_crosssectional']
N_RUNS        = 10
N_BLOCKS      = 30
N_SEEDS       = [0, 100, 200]
HORIZONS      = [4, 24, 54]
TOP_K         = 150
_OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "arch_comparison")

# Max steps each arch can predict; None = autoregressive (unlimited)
_ARCH_MAX_HORIZON = {
    'current':          None,
    'A_surgical':       None,
    'B_4':              4,
    'B_24':             24,
    'C_crosssectional': None,
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_spearman_rho(predicted, realized):
    """Spearman rank correlation between cross-sectional predicted and realized returns.

    Returns 0.0 if predicted or realized is constant (no rank signal).
    """
    if np.std(predicted) < 1e-10 or np.std(realized) < 1e-10:
        return 0.0
    rho, _ = spearmanr(predicted, realized)
    return float(rho)


def compute_icir(rho_series):
    """ICIR = mean(rho) / std(rho, ddof=1) across OOS blocks.

    Returns 0.0 if std is zero (perfectly stable or single observation).
    """
    arr = np.asarray(rho_series, dtype=float)
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    if std < 1e-10:
        return 0.0
    return float(arr.mean() / std)


def compute_topk_precision(predicted, realized, k=150):
    """Fraction of predicted top-k stocks that appear in realized top-k."""
    n = len(predicted)
    k = min(k, n)
    pred_topk = set(np.argsort(predicted)[-k:])
    real_topk = set(np.argsort(realized)[-k:])
    return len(pred_topk & real_topk) / k


def compute_hit_rate(predicted, realized):
    """Fraction of stocks where predicted and realized return directions match.

    Denominator is always N (total stocks). Stocks with zero predicted or
    realized return contribute 0 to the numerator (counted as misses).
    """
    pred_sign = np.sign(predicted)
    real_sign = np.sign(realized)
    matches   = (pred_sign == real_sign) & (pred_sign != 0) & (real_sign != 0)
    return float(matches.mean())


# ---------------------------------------------------------------------------
# Benchmark predictors (no model training)
# ---------------------------------------------------------------------------

def predict_zero(n_stocks):
    """B1: null forecast — all stocks at zero return."""
    return np.zeros(n_stocks)


def predict_momentum(returns_window, lambda_=0.2):
    """B2: exp-decay weighted mean of past returns per stock.

    returns_window: ndarray (time_window, n_stocks)
    Same decay convention as weighted_mean_return in transformer_model.py.
    """
    n_periods = returns_window.shape[0]
    idx = np.arange(n_periods, 0, -1)
    w   = np.exp(-lambda_ * idx)
    w  /= w.sum()
    return (returns_window * w[:, np.newaxis]).sum(axis=0)


def predict_persistence(returns_window):
    """B3: last observed weekly return repeated as forecast."""
    return returns_window[-1].copy()


def predict_mean_reversion(returns_window):
    """B4: negative of last period's cross-sectional deviation."""
    last = returns_window[-1]
    return -(last - last.mean())


# ---------------------------------------------------------------------------
# Core evaluation block
# ---------------------------------------------------------------------------

_BENCHMARKS = [
    ('B1_zero',        lambda w: predict_zero(w.shape[1])),
    ('B2_momentum',    lambda w: predict_momentum(w)),
    ('B3_persistence', lambda w: predict_persistence(w)),
    ('B4_mean_rev',    lambda w: predict_mean_reversion(w)),
]


def run_one_block(arch_name, returns_train, returns_test, cfg, n_runs, horizons):
    """Train arch_name on returns_train; evaluate at each horizon vs returns_test.

    returns_train: DataFrame (n_train_periods, n_stocks)
    returns_test:  DataFrame (at least max(horizons) rows, n_stocks)
    cfg:           config dict; periods_to_forecast is overridden internally
    n_runs:        number of transformer runs to average
    horizons:      list of ints e.g. [4, 24, 54]

    Returns dict {(horizon, metric_name): float}
    Unsupported (arch, horizon) pairs -> NaN for all metrics.
    metric_names: 'rank_ic', 'topk_precision', 'hit_rate',
                  'rank_ic_B1_zero', 'rank_ic_B2_momentum',
                  'rank_ic_B3_persistence', 'rank_ic_B4_mean_rev'
    """
    arch_max    = _ARCH_MAX_HORIZON[arch_name]
    max_h       = max(horizons)
    time_window = cfg['time_window']

    # Set periods_to_forecast to max horizon for autoregressive archs
    cfg_run = {**cfg, 'periods_to_forecast': max_h}

    # Train
    preds_3d   = train_runs(returns_train, cfg_run, n_runs=n_runs,
                             verbose=False, arch=arch_name)
    preds_mean = preds_3d.mean(axis=0)   # (steps, n_stocks)

    # Last training window for benchmarks
    returns_window = returns_train.values[-time_window:]   # (time_window, n_stocks)

    results = {}
    for h in horizons:
        # Check arch support
        if arch_max is not None and h > arch_max:
            for key in ('rank_ic', 'topk_precision', 'hit_rate',
                        'rank_ic_B1_zero', 'rank_ic_B2_momentum',
                        'rank_ic_B3_persistence', 'rank_ic_B4_mean_rev'):
                results[(h, key)] = float('nan')
            continue

        if preds_mean.shape[0] < h:
            for key in ('rank_ic', 'topk_precision', 'hit_rate',
                        'rank_ic_B1_zero', 'rank_ic_B2_momentum',
                        'rank_ic_B3_persistence', 'rank_ic_B4_mean_rev'):
                results[(h, key)] = float('nan')
            continue

        pred_cum = preds_mean[:h].sum(axis=0)           # (n_stocks,)
        real_cum = returns_test.values[:h].sum(axis=0)  # (n_stocks,)

        results[(h, 'rank_ic')]        = compute_spearman_rho(pred_cum, real_cum)
        results[(h, 'topk_precision')] = compute_topk_precision(pred_cum, real_cum, k=TOP_K)
        results[(h, 'hit_rate')]       = compute_hit_rate(pred_cum, real_cum)

        for bench_name, bench_fn in _BENCHMARKS:
            bench_pred = bench_fn(returns_window)
            results[(h, f'rank_ic_{bench_name}')] = compute_spearman_rho(bench_pred, real_cum)

    return results
