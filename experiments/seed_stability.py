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


def compute_stock_frequencies(topk_sets_per_seed, n_stocks, n_arms, thresholds):
    """Count how many seeds place each stock in top-k.

    Returns dict[(n, k)] -> ndarray of shape (n_stocks,) with integer counts
    """
    result = {}
    for n in n_arms:
        for k in thresholds:
            counts = np.zeros(n_stocks, dtype=int)
            for seed_sets in topk_sets_per_seed:
                for idx in seed_sets[(n, k)]:
                    counts[idx] += 1
            result[(n, k)] = counts
    return result


def compute_core_sets(freq_dict, n_seeds):
    """Core set size at 60%, 80%, 100% frequency thresholds.

    freq_dict: dict[(n, k)] -> ndarray of stock counts
    Returns dict[(n, k, pct)] -> int
    """
    result = {}
    for (n, k), counts in freq_dict.items():
        for pct in _CORE_PCTS:
            min_count = int(np.ceil(pct * n_seeds))
            result[(n, k, pct)] = int((counts >= min_count).sum())
    return result


def write_outputs(results, out_dir):
    """Write all 4 output files to out_dir (created if absent)."""
    os.makedirs(out_dir, exist_ok=True)

    # pairwise_overlap.csv
    rows = [
        {'n': n, 'k': k,
         'overlap_mean': mean, 'overlap_std': std,
         'overlap_min': mn,   'overlap_max': mx}
        for (n, k), (mean, std, mn, mx) in sorted(results['pairwise'].items())
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'pairwise_overlap.csv'), index=False)

    # core_set_size.csv
    rows = [
        {'n': n, 'k': k, 'threshold_pct': pct, 'core_size': size}
        for (n, k, pct), size in sorted(results['core'].items())
    ]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'core_set_size.csv'), index=False)

    # stock_frequency.csv
    freq = results['freq']
    col_names = {key: f"n{key[0]}_k{key[1]}" for key in sorted(freq)}
    df_freq = pd.DataFrame({col_names[key]: freq[key] for key in sorted(freq)})
    df_freq.index.name = 'stock_idx'
    df_freq.to_csv(os.path.join(out_dir, 'stock_frequency.csv'))

    # seed_stability_summary.txt
    lines = [
        "Cross-seed top-N stability study",
        f"n_seeds={results['n_seeds']}, n_arms={results['n_arms']}, "
        f"thresholds={results['thresholds']}, n_stocks={results['n_stocks']}",
        "",
        "== Pairwise overlap (mean +/- std [min, max]) ==",
    ]
    for (n, k) in sorted(results['pairwise']):
        mean, std, mn, mx = results['pairwise'][(n, k)]
        lines.append(f"  n={n:3d}, k={k:3d}: {mean:.4f} +/- {std:.4f}  [{mn:.4f}, {mx:.4f}]")
    lines += ["", "== Core set sizes =="]
    for (n, k, pct), size in sorted(results['core'].items()):
        lines.append(f"  n={n:3d}, k={k:3d}, {pct:.0%}: {size}")
    with open(os.path.join(out_dir, 'seed_stability_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "seed_stability")


def _load_data():
    cfg     = load_config()
    rets_df = pd.read_csv(PATHS['01_returns'], index_col=0)
    return rets_df, cfg


def run_timing_calibration(rets_df, cfg, n_cal=10):
    """Train n_cal models and return (per_run_seconds, estimated_total_seconds)."""
    t0 = time.time()
    train_runs(rets_df, cfg, n_runs=n_cal, verbose=False)
    per_run = (time.time() - t0) / n_cal
    return per_run, per_run * N_SEEDS * max(N_ARMS)


def run_one_seed(rets_df, sigma_lw, cfg, n_arms, thresholds, seed):
    """Train max(n_arms) models with a fixed seed, return top-k sets per (n, k)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    preds_3d   = train_runs(rets_df, cfg, n_runs=max(n_arms), verbose=False)
    mu_per_run = compute_mu_per_run(preds_3d)
    return compute_topk_sets(mu_per_run, sigma_lw, n_arms, thresholds)


def main():
    rets_df, cfg = _load_data()
    n_stocks = rets_df.shape[1]

    print(f"\n=== seed_stability ===")
    print(f"Universe: {n_stocks} stocks, {len(rets_df)} periods")
    print(f"Plan: {N_SEEDS} seeds, n_arms={N_ARMS}, thresholds={THRESHOLDS}")

    print("\nTiming calibration (10 runs)...")
    per_run, est_total = run_timing_calibration(rets_df, cfg, n_cal=10)
    print(f"  {per_run:.1f}s/run  ->  estimated total: {est_total/3600:.1f}h ({est_total:.0f}s)")
    print("Proceed? [y/N] ", end="", flush=True)
    if input().strip().lower() != "y":
        print("Aborted.")
        return

    sigma_lw = np.sqrt(np.diag(LedoitWolf().fit(rets_df.values).covariance_))

    topk_sets_per_seed = []
    for seed in range(N_SEEDS):
        print(f"\nSeed {seed + 1}/{N_SEEDS}  (seed={seed})")
        topk_sets_per_seed.append(
            run_one_seed(rets_df, sigma_lw, cfg, N_ARMS, THRESHOLDS, seed=seed)
        )

    print("\nAggregating...")
    freq = compute_stock_frequencies(topk_sets_per_seed, n_stocks, N_ARMS, THRESHOLDS)
    results = {
        'pairwise':   compute_pairwise_overlaps(topk_sets_per_seed),
        'core':       compute_core_sets(freq, N_SEEDS),
        'freq':       freq,
        'n_seeds':    N_SEEDS,
        'n_arms':     N_ARMS,
        'thresholds': THRESHOLDS,
        'n_stocks':   n_stocks,
    }

    print(f"Writing results to {_OUT_DIR}")
    write_outputs(results, _OUT_DIR)
    print("Done.")


if __name__ == '__main__':
    main()
