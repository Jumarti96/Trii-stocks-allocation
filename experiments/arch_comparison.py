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


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def aggregate_results(block_results, horizons):
    """Compute mean, std, ICIR, and mean benchmark metrics across OOS blocks.

    block_results: list of dicts, each from run_one_block
    horizons: list of ints
    Returns dict {(horizon, stat_name): float}
    """
    agg = {}
    for h in horizons:
        rho_vals  = np.array([r[(h, 'rank_ic')]        for r in block_results])
        topk_vals = np.array([r[(h, 'topk_precision')] for r in block_results])
        hit_vals  = np.array([r[(h, 'hit_rate')]        for r in block_results])

        agg[(h, 'mean_rank_ic')]  = float(np.nanmean(rho_vals))
        agg[(h, 'std_rank_ic')]   = (float(np.nanstd(rho_vals, ddof=1))
                                     if np.sum(~np.isnan(rho_vals)) > 1 else 0.0)
        agg[(h, 'icir')]          = compute_icir(rho_vals[~np.isnan(rho_vals)])
        agg[(h, 'mean_topk_precision')] = float(np.nanmean(topk_vals))
        agg[(h, 'std_topk_precision')]  = (float(np.nanstd(topk_vals, ddof=1))
                                           if np.sum(~np.isnan(topk_vals)) > 1 else 0.0)
        agg[(h, 'mean_hit_rate')] = float(np.nanmean(hit_vals))
        agg[(h, 'std_hit_rate')]  = (float(np.nanstd(hit_vals, ddof=1))
                                     if np.sum(~np.isnan(hit_vals)) > 1 else 0.0)

        for bench in ('B1_zero', 'B2_momentum', 'B3_persistence', 'B4_mean_rev'):
            bench_rhos = np.array([r[(h, f'rank_ic_{bench}')] for r in block_results])
            agg[(h, f'mean_rank_ic_{bench}')] = float(np.nanmean(bench_rhos))

    return agg


def write_outputs(results_by_arch_seed, horizons, out_dir):
    """Write 4 output files summarising the architecture comparison.

    results_by_arch_seed: dict {(arch_name, seed): aggregate_results(...) dict}
    horizons: list of ints
    out_dir: directory (created if absent)
    """
    os.makedirs(out_dir, exist_ok=True)

    rank_ic_rows, topk_rows, hit_rows = [], [], []
    for (arch, seed), agg in sorted(results_by_arch_seed.items()):
        for h in horizons:
            rank_ic_rows.append({
                'arch': arch, 'seed': seed, 'horizon': h,
                'mean_rank_ic': agg[(h, 'mean_rank_ic')],
                'std_rank_ic':  agg[(h, 'std_rank_ic')],
                'icir':         agg[(h, 'icir')],
                'mean_rank_ic_B1_zero':        agg[(h, 'mean_rank_ic_B1_zero')],
                'mean_rank_ic_B2_momentum':    agg[(h, 'mean_rank_ic_B2_momentum')],
                'mean_rank_ic_B3_persistence': agg[(h, 'mean_rank_ic_B3_persistence')],
                'mean_rank_ic_B4_mean_rev':    agg[(h, 'mean_rank_ic_B4_mean_rev')],
            })
            topk_rows.append({
                'arch': arch, 'seed': seed, 'horizon': h,
                'mean_topk_precision': agg[(h, 'mean_topk_precision')],
                'std_topk_precision':  agg[(h, 'std_topk_precision')],
            })
            hit_rows.append({
                'arch': arch, 'seed': seed, 'horizon': h,
                'mean_hit_rate': agg[(h, 'mean_hit_rate')],
                'std_hit_rate':  agg[(h, 'std_hit_rate')],
            })

    pd.DataFrame(rank_ic_rows).to_csv(os.path.join(out_dir, 'rank_ic.csv'), index=False)
    pd.DataFrame(topk_rows).to_csv(os.path.join(out_dir, 'topk_precision.csv'), index=False)
    pd.DataFrame(hit_rows).to_csv(os.path.join(out_dir, 'hit_rate.csv'), index=False)

    # Summary text
    df_ic = pd.DataFrame(rank_ic_rows)
    lines = [
        "Transformer Architecture Comparison — Forecast Quality Study",
        f"Horizons: {horizons} weeks | Top-k: {TOP_K} | "
        f"n_runs: {N_RUNS} | n_blocks: {N_BLOCKS} | seeds: {N_SEEDS}",
        "",
        "=== Mean Rank IC (Spearman rho) | mean +/- std [ICIR] ===",
    ]
    for h in horizons:
        lines.append(f"\n  Horizon {h:2d} weeks:")
        sub = df_ic[df_ic['horizon'] == h].sort_values('arch')
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['arch']:<20s} seed={row['seed']:3d}: "
                f"{row['mean_rank_ic']:+.4f} +/- {row['std_rank_ic']:.4f}  "
                f"ICIR={row['icir']:.2f}"
            )
        lines.append(f"  Benchmarks at h={h}:")
        bench_row = sub.iloc[0]
        for bench in ('B1_zero', 'B2_momentum', 'B3_persistence', 'B4_mean_rev'):
            lines.append(f"    {bench:<22s}: {bench_row[f'mean_rank_ic_{bench}']:+.4f}")

    with open(os.path.join(out_dir, 'arch_comparison_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _load_data():
    cfg     = load_config()
    rets_df = pd.read_csv(PATHS['01_returns'], index_col=0)
    return rets_df, cfg


def run_timing_calibration(rets_df, cfg, n_cal=5):
    """Train n_cal models with arch='current' and estimate total runtime."""
    t0 = time.time()
    train_runs(rets_df, cfg, n_runs=n_cal, verbose=False, arch='current')
    per_run = (time.time() - t0) / n_cal
    # Estimate: n_archs * n_seeds * n_blocks * n_runs * per_run
    n_train_archs = len(ARCHITECTURES)
    est_total = per_run * n_train_archs * len(N_SEEDS) * N_BLOCKS * N_RUNS
    return per_run, est_total


def main():
    rets_df, cfg = _load_data()
    n_stocks     = rets_df.shape[1]
    n_periods    = len(rets_df)
    time_window  = cfg['time_window']

    print(f"\n=== arch_comparison ===")
    print(f"Universe: {n_stocks} stocks, {n_periods} periods")
    print(f"Architectures: {ARCHITECTURES}")
    print(f"Plan: {N_SEEDS} seeds x {N_BLOCKS} blocks x n_runs={N_RUNS}, "
          f"horizons={HORIZONS}")

    print("\nTiming calibration (5 runs, arch=current)...")
    per_run, est_total = run_timing_calibration(rets_df, cfg, n_cal=5)
    print(f"  {per_run:.1f}s/run  ->  estimated total: {est_total/3600:.1f}h")
    print("Proceed? [y/N] ", end="", flush=True)
    if input().strip().lower() != "y":
        print("Aborted.")
        return

    results_by_arch_seed = {}

    for seed in N_SEEDS:
        print(f"\n--- Seed {seed} ---")
        for arch in ARCHITECTURES:
            print(f"  Architecture: {arch}")
            block_results = []
            for block_idx in range(N_BLOCKS):
                # Walk-forward split spaced 4 weeks apart.
                # Last block ends exactly max(HORIZONS) before end of data.
                block_start = n_periods - max(HORIZONS) - (N_BLOCKS - 1 - block_idx) * 4
                train = rets_df.iloc[:block_start]
                test  = rets_df.iloc[block_start:block_start + max(HORIZONS)]

                if len(train) < time_window + 2 or len(test) < max(HORIZONS):
                    continue

                np.random.seed(seed)
                torch.manual_seed(seed)

                block_results.append(
                    run_one_block(arch, train, test, cfg, N_RUNS, HORIZONS)
                )
                if (block_idx + 1) % 5 == 0:
                    print(f"    block {block_idx + 1}/{N_BLOCKS} done")

            results_by_arch_seed[(arch, seed)] = aggregate_results(block_results, HORIZONS)

    print(f"\nWriting results to {_OUT_DIR}")
    write_outputs(results_by_arch_seed, HORIZONS, _OUT_DIR)
    print("Done.")


if __name__ == '__main__':
    main()
