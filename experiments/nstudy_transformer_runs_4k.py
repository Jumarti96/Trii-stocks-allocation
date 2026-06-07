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


def write_outputs(metrics, out_dir):
    """Write all five metric CSVs and a summary text file to out_dir.

    metrics keys: cov_mu, topn_lw, sigma_decay, cov_sf, topn_fv,
                  checkpoints, thresholds, n_iters, n_runs, n_stocks
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpts = metrics['checkpoints']
    thres = metrics['thresholds']

    # convergence_metrics.csv
    pd.DataFrame([
        {'n': n, 'cov_mu_median': v[0], 'cov_mu_p25': v[1], 'cov_mu_p75': v[2]}
        for n, v in metrics['cov_mu'].items()
    ]).sort_values('n').to_csv(os.path.join(out_dir, 'convergence_metrics.csv'), index=False)

    # topn_overlap_lw.csv
    pd.DataFrame([
        {'n': n, 'k': k, 'overlap_mean': v[0], 'overlap_std': v[1]}
        for (n, k), v in metrics['topn_lw'].items()
    ]).sort_values(['n', 'k']).to_csv(os.path.join(out_dir, 'topn_overlap_lw.csv'), index=False)

    # sigma_forecast_decay.csv
    pd.DataFrame([
        {'n': n, 'mean_sigma_forecast': v}
        for n, v in metrics['sigma_decay'].items()
    ]).sort_values('n').to_csv(os.path.join(out_dir, 'sigma_forecast_decay.csv'), index=False)

    # cov_sigma_forecast.csv
    pd.DataFrame([
        {'n': n, 'cov_sf_median': v[0], 'cov_sf_p25': v[1], 'cov_sf_p75': v[2]}
        for n, v in metrics['cov_sf'].items()
    ]).sort_values('n').to_csv(os.path.join(out_dir, 'cov_sigma_forecast.csv'), index=False)

    # topn_overlap_fv.csv
    pd.DataFrame([
        {'n': n, 'k': k, 'overlap_mean': v[0], 'overlap_std': v[1]}
        for (n, k), v in metrics['topn_fv'].items()
    ]).sort_values(['n', 'k']).to_csv(os.path.join(out_dir, 'topn_overlap_fv.csv'), index=False)

    # nstudy_4k_summary.txt
    lines = [
        "n_transformer_runs convergence study — 4k-stock universe",
        f"n_iterations={metrics['n_iters']}, n_runs_per_iter={metrics['n_runs']}, "
        f"n_stocks={metrics['n_stocks']}",
        "",
        "== CoV of mu across iterations ==",
        f"{'n':>6}  {'cov_mu_median':>14}  {'cov_mu_p25':>10}  {'cov_mu_p75':>10}",
    ]
    for n in ckpts:
        med, p25, p75 = metrics['cov_mu'][n]
        lines.append(f"{n:>6}  {med:>14.4f}  {p25:>10.4f}  {p75:>10.4f}")

    lines += ["", "== Top-N overlap (LW Sharpe) ==",
              f"{'n':>6}  {'k':>6}  {'overlap_mean':>13}  {'overlap_std':>11}"]
    for n in ckpts:
        for k in thres:
            m, s = metrics['topn_lw'][(n, k)]
            lines.append(f"{n:>6}  {k:>6}  {m:>13.4f}  {s:>11.4f}")

    lines += ["", "== sigma_forecast decay ==",
              f"{'n':>6}  {'mean_sigma_forecast':>19}"]
    for n in ckpts:
        lines.append(f"{n:>6}  {metrics['sigma_decay'][n]:>19.6f}")

    lines += ["", "== CoV of sigma_forecast across iterations ==",
              f"{'n':>6}  {'cov_sf_median':>13}  {'cov_sf_p25':>10}  {'cov_sf_p75':>10}"]
    for n in ckpts:
        med, p25, p75 = metrics['cov_sf'][n]
        lines.append(f"{n:>6}  {med:>13.4f}  {p25:>10.4f}  {p75:>10.4f}")

    lines += ["", "== Top-N overlap (forecast-variance Sharpe) ==",
              f"{'n':>6}  {'k':>6}  {'overlap_mean':>13}  {'overlap_std':>11}"]
    for n in ckpts:
        for k in thres:
            m, s = metrics['topn_fv'][(n, k)]
            lines.append(f"{n:>6}  {k:>6}  {m:>13.4f}  {s:>11.4f}")

    with open(os.path.join(out_dir, 'nstudy_4k_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "nstudy_4k")


def _load_data():
    """Load returns DataFrame and pipeline config from default PATHS."""
    cfg    = load_config()
    rets_df = pd.read_csv(PATHS['01_returns'], index_col=0)
    return rets_df, cfg


def run_timing_calibration(rets_df, cfg, n_cal=10):
    """Train n_cal models and return (per_run_seconds, estimated_total_seconds).

    estimated_total_seconds covers N_ITERS × N_RUNS model trainings.
    """
    t0 = time.time()
    train_runs(rets_df, cfg, n_runs=n_cal, verbose=False)
    per_run = (time.time() - t0) / n_cal
    return per_run, per_run * N_ITERS * N_RUNS


def main():
    rets_df, cfg = _load_data()
    n_stocks = rets_df.shape[1]

    print(f"\n=== nstudy_transformer_runs_4k ===")
    print(f"Universe: {n_stocks} stocks, {len(rets_df)} periods")
    print(f"Plan: {N_ITERS} iterations × {N_RUNS} runs, "
          f"checkpoints={CHECKPOINTS}, thresholds={THRESHOLDS}")

    print("\nTiming calibration (10 runs)...")
    per_run, est_total = run_timing_calibration(rets_df, cfg, n_cal=10)
    print(f"  {per_run:.1f}s/run  →  estimated total: {est_total/3600:.1f}h "
          f"({est_total:.0f}s)")
    print("Proceed? [y/N] ", end="", flush=True)
    if input().strip().lower() != "y":
        print("Aborted.")
        return

    sigma_lw = np.sqrt(np.diag(LedoitWolf().fit(rets_df.values).covariance_))
    seeds    = list(range(N_ITERS))

    mu_snaps_all  = []
    sf_snaps_all  = []
    topn_lw_all   = []
    topn_fv_all   = []
    mean_sf_all   = []

    for i, seed in enumerate(seeds):
        print(f"\nIteration {i + 1}/{N_ITERS}  (seed={seed})")
        res = run_one_iteration(rets_df, sigma_lw, cfg, CHECKPOINTS, THRESHOLDS, seed=seed)
        mu_snaps_all.append(res['mu_snapshots'])
        sf_snaps_all.append(res['sf_snapshots'])
        topn_lw_all.append(res['topn_lw'])
        topn_fv_all.append(res['topn_fv'])
        mean_sf_all.append(res['mean_sf'])

    print("\nAggregating...")
    metrics = {
        'cov_mu':      compute_cov_across_iters(mu_snaps_all,  CHECKPOINTS),
        'topn_lw':     aggregate_topn_overlaps(topn_lw_all,    CHECKPOINTS, THRESHOLDS),
        'sigma_decay': aggregate_sigma_decay(mean_sf_all,      CHECKPOINTS),
        'cov_sf':      compute_cov_sf_across_iters(sf_snaps_all, CHECKPOINTS),
        'topn_fv':     aggregate_topn_overlaps(topn_fv_all,    CHECKPOINTS, THRESHOLDS),
        'checkpoints': CHECKPOINTS,
        'thresholds':  THRESHOLDS,
        'n_iters':     N_ITERS,
        'n_runs':      N_RUNS,
        'n_stocks':    n_stocks,
    }

    print(f"Writing results to {_OUT_DIR}")
    write_outputs(metrics, _OUT_DIR)
    print("Done.")


def run_one_iteration(rets_df, sigma_lw, cfg, checkpoints, thresholds, seed):
    """Train N_RUNS transformer models for one iteration and record per-checkpoint snapshots.

    rets_df:    DataFrame (T, n_stocks) — historical returns (from data/01_returns.csv)
    sigma_lw:   ndarray (n_stocks,)     — sqrt of LW covariance diagonal (fixed, pre-computed)
    cfg:        dict                    — pipeline config (from load_config())
    checkpoints: list[int]              — n values at which to snapshot metrics
    thresholds:  list[int]              — top-N thresholds for overlap metrics
    seed:        int                    — sets torch + numpy random state before training

    Returns dict with keys:
        mu_snapshots  — dict[n → ndarray (n_stocks,)] running mean of per-run mu
        sf_snapshots  — dict[n → ndarray (n_stocks,)] sigma_forecast (cross-run std of mu)
        topn_lw       — dict[n → dict[k → float]] LW-Sharpe top-N overlap vs n=max(checkpoints)
        topn_fv       — dict[n → dict[k → float]] forecast-variance Sharpe top-N overlap
        mean_sf       — dict[n → float] mean sigma_forecast across stocks
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_runs = max(checkpoints)
    preds_3d  = train_runs(rets_df, cfg, n_runs=n_runs, verbose=False)
    # preds_3d: (n_runs, periods_to_forecast, n_stocks)

    mu_per_run = compute_mu_per_run(preds_3d)  # (n_runs, n_stocks)

    # Reference: full n_runs average within this iteration
    mu_ref = mu_per_run.mean(axis=0)
    sf_ref = mu_per_run.std(axis=0, ddof=1) if n_runs > 1 else np.zeros(mu_per_run.shape[1])
    sharpe_lw_ref = mu_ref / (sigma_lw + _EPS)
    sharpe_fv_ref = mu_ref / (sf_ref   + _EPS)

    out = {'mu_snapshots': {}, 'sf_snapshots': {}, 'topn_lw': {}, 'topn_fv': {}, 'mean_sf': {}}

    for n in checkpoints:
        mu_n = mu_per_run[:n].mean(axis=0)
        sf_n = mu_per_run[:n].std(axis=0, ddof=1) if n > 1 else np.zeros(mu_per_run.shape[1])

        sharpe_lw_n = mu_n / (sigma_lw + _EPS)
        sharpe_fv_n = mu_n / (sf_n     + _EPS)

        out['mu_snapshots'][n] = mu_n
        out['sf_snapshots'][n] = sf_n
        out['mean_sf'][n]      = float(sf_n.mean())
        out['topn_lw'][n]      = {k: compute_topn_overlap(sharpe_lw_n, sharpe_lw_ref, k)
                                  for k in thresholds}
        out['topn_fv'][n]      = {k: compute_topn_overlap(sharpe_fv_n, sharpe_fv_ref, k)
                                  for k in thresholds}

    return out


if __name__ == '__main__':
    main()
