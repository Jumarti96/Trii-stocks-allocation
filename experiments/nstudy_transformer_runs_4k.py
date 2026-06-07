"""
n_transformer_runs convergence study — 4k-stock universe.

Pre-condition: data/01_returns.csv must exist (run pipeline/01_download.py first).

Usage:
    "C:/Python projects/Finance/Scripts/python.exe" experiments/nstudy_transformer_runs_4k.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

from transformer_model import train_runs
from config import load_config, PATHS

CHECKPOINTS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
THRESHOLDS  = [150, 300, 500]
N_ITERS     = 30
N_RUNS      = 300
_EPS        = 1e-10


def compute_mu_per_run(preds_3d, lambda_=0.2):
    """Apply exponential-decay weights across forecast periods for each run.

    preds_3d: ndarray (n_runs, periods_to_forecast, n_stocks)
    Returns:  ndarray (n_runs, n_stocks) — per-run weighted-mean return per stock
    """
    _, periods, _ = preds_3d.shape
    idx = np.arange(1, periods + 1)
    w   = np.exp(-lambda_ * idx)
    w  /= w.sum()
    return (preds_3d * w[np.newaxis, :, np.newaxis]).sum(axis=1)


def compute_topn_overlap(scores_n, scores_ref, k):
    """Fraction of top-k stocks shared between scores_n and scores_ref.

    scores_n, scores_ref: 1-D arrays of length n_stocks (higher = better)
    k: number of top stocks to compare; must be <= len(scores_n)
    Returns float in [0, 1].
    """
    if k > len(scores_n):
        raise ValueError(f"k={k} exceeds n_stocks={len(scores_n)}")
    top_n   = set(np.argsort(scores_n)[-k:])
    top_ref = set(np.argsort(scores_ref)[-k:])
    return len(top_n & top_ref) / k


def compute_cov_across_iters(mu_snapshots, checkpoints):
    """Cross-iteration CoV of mu per stock, aggregated as median + p25/p75.

    mu_snapshots: list of dicts — one dict per iteration, each mapping n → (n_stocks,) array
    Returns: dict mapping n → (median_cov, p25_cov, p75_cov)
    """
    result = {}
    for n in checkpoints:
        stacked = np.stack([s[n] for s in mu_snapshots], axis=0)  # (n_iters, n_stocks)
        stds    = stacked.std(axis=0, ddof=1)
        means   = stacked.mean(axis=0)
        cov     = stds / (np.abs(means) + _EPS)
        result[n] = (float(np.median(cov)),
                     float(np.percentile(cov, 25)),
                     float(np.percentile(cov, 75)))
    return result


def compute_cov_sf_across_iters(sf_snapshots, checkpoints):
    """Cross-iteration CoV of sigma_forecast per stock, aggregated as median + p25/p75.

    sf_snapshots: list of dicts — one dict per iteration, each mapping n → (n_stocks,) array
    Returns: dict mapping n → (median_cov, p25_cov, p75_cov)
    """
    result = {}
    for n in checkpoints:
        stacked = np.stack([s[n] for s in sf_snapshots], axis=0)
        stds    = stacked.std(axis=0, ddof=1)
        means   = stacked.mean(axis=0)
        cov     = stds / (np.abs(means) + _EPS)
        result[n] = (float(np.median(cov)),
                     float(np.percentile(cov, 25)),
                     float(np.percentile(cov, 75)))
    return result


def aggregate_topn_overlaps(topn_by_iter, checkpoints, thresholds):
    """Aggregate per-iteration top-N overlap fractions to mean ± std.

    topn_by_iter: list of dicts — one per iteration, each mapping n → {k: overlap_fraction}
    Returns: dict mapping (n, k) → (mean, std)
    """
    result = {}
    for n in checkpoints:
        for k in thresholds:
            vals = [d[n][k] for d in topn_by_iter]
            result[(n, k)] = (float(np.mean(vals)),
                              float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
    return result


def aggregate_sigma_decay(mean_sf_by_iter, checkpoints):
    """Mean sigma_forecast across iterations for each checkpoint.

    mean_sf_by_iter: list of dicts — one per iteration, each mapping n → float
    Returns: dict mapping n → float
    """
    return {n: float(np.mean([d[n] for d in mean_sf_by_iter])) for n in checkpoints}
