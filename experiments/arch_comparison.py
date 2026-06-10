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
