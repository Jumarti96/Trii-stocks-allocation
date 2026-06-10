"""
Cross-seed top-N stability study.

Pre-condition: data/01_returns.csv must exist (run pipeline/01_download.py first).

Usage:
    "C:/Python projects/Finance/Scripts/python.exe" experiments/seed_stability.py
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

N_SEEDS     = 15
N_ARMS      = [100, 150, 200]
THRESHOLDS  = [150, 300, 500]
_CORE_PCTS  = [0.6, 0.8, 1.0]
_EPS        = 1e-10


def compute_mu_per_run(preds_3d, lambda_=0.2):
    """Exp-decay weighted mean return per run and stock.

    preds_3d: ndarray (n_runs, periods_to_forecast, n_stocks)
    Returns:  ndarray (n_runs, n_stocks)
    """
    _, periods, _ = preds_3d.shape
    idx = np.arange(1, periods + 1)
    w   = np.exp(-lambda_ * idx)
    w  /= w.sum()
    return (preds_3d * w[np.newaxis, :, np.newaxis]).sum(axis=1)


def compute_topk_sets(mu_per_run, sigma_lw, n_arms, thresholds):
    """Rank stocks by LW-Sharpe proxy at each n arm and return top-k index sets.

    mu_per_run: ndarray (n_runs, n_stocks)
    sigma_lw:   ndarray (n_stocks,)
    Returns dict[(n, k)] -> frozenset of stock indices
    """
    n_stocks = mu_per_run.shape[1]
    result = {}
    for n in n_arms:
        mu_n   = mu_per_run[:n].mean(axis=0)
        scores = mu_n / (sigma_lw + _EPS)
        for k in thresholds:
            if k > n_stocks:
                raise ValueError(f"k={k} > n_stocks={n_stocks}")
            top_idx = frozenset(np.argpartition(scores, -k)[-k:].tolist())
            result[(n, k)] = top_idx
    return result


def compute_pairwise_overlaps(topk_sets_per_seed):
    """Mean/std/min/max pairwise overlap across all seed pairs per (n, k).

    topk_sets_per_seed: list of dicts, each dict[(n, k)] -> frozenset
    Returns dict[(n, k)] -> (mean, std, min, max)
    """
    keys = list(topk_sets_per_seed[0].keys())
    result = {}
    for key in keys:
        n, k = key
        sets = [s[key] for s in topk_sets_per_seed]
        overlaps = [
            len(sets[i] & sets[j]) / k
            for i in range(len(sets))
            for j in range(i + 1, len(sets))
        ]
        arr  = np.array(overlaps)
        ddof = 1 if len(arr) > 1 else 0
        result[key] = (float(arr.mean()), float(arr.std(ddof=ddof)),
                       float(arr.min()), float(arr.max()))
    return result
