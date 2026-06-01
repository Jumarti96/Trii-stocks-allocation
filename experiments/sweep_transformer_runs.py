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


def summarize_sweep(result, rel_tol=0.05):
    """Per-n turnover / Jaccard / mean per-stock mu std, with consecutive-n deltas and the
    first n where all three change < rel_tol relative to the previous grid point."""
    grid = sorted(result["by_n"].keys())
    cols = ["turnover", "jaccard", "mean_mu_std"]

    data = {}
    for n in grid:
        wdf = result["by_n"][n]["weights"]
        mdf = result["by_n"][n]["mu"]
        data[n] = {
            "turnover": float(mean_turnover(wdf)),
            "jaccard": float(mean_jaccard(wdf)),
            "mean_mu_std": float(weight_dispersion(mdf).mean()),
        }
    table = pd.DataFrame(data).T[cols]
    table.index.name = "n"
    delta_names = {"turnover": "d_turnover", "jaccard": "d_jaccard", "mean_mu_std": "d_mu_std"}
    for c in cols:
        table[delta_names[c]] = table[c].diff()

    converged_n = None
    for prev, n in zip(grid[:-1], grid[1:]):
        ok = True
        for c in cols:
            base = table.loc[prev, c]
            if base == 0:
                continue
            if abs(table.loc[n, c] - base) / abs(base) >= rel_tol:
                ok = False
                break
        if ok:
            converged_n = n
            break

    return {"table": table, "converged_n": converged_n, "rel_tol": rel_tol}


def format_sweep_summary(summary):
    """Human-readable convergence report from a summarize_sweep result."""
    table = summary["table"]
    conv = summary["converged_n"]
    tol = summary["rel_tol"]

    lines = ["Transformer-runs convergence sweep", ""]
    lines.append(table.to_string(float_format=lambda v: f"{v:.5f}"))
    lines.append("")
    if conv is None:
        lines.append(f"No grid point met the <{tol:.0%} relative-delta convergence criterion.")
    else:
        lines.append(
            f"Converged at n={conv} (turnover, jaccard, and mean_mu_std each changed "
            f"<{tol:.0%} vs the previous grid point)."
        )
    return "\n".join(lines)


def write_sweep_outputs(summary, outdir):
    """Write sweep_metrics.csv and sweep_summary.txt; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "metrics": os.path.join(outdir, "sweep_metrics.csv"),
        "summary": os.path.join(outdir, "sweep_summary.txt"),
    }
    summary["table"][["turnover", "jaccard", "mean_mu_std"]].to_csv(paths["metrics"])
    with open(paths["summary"], "w") as f:
        f.write(format_sweep_summary(summary))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--grid", type=str, default="10,20,30,40,50,100",
                        help="Comma-separated n_transformer_runs values")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "sweep"))
    args = parser.parse_args()

    grid = [int(x) for x in args.grid.split(",")]
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    print(f"Sweep grid={grid} | iterations={args.iterations} | seed={args.seed}")
    result = run_sweep(prices, rets, cfg, iterations=args.iterations, grid=grid, seed=args.seed)
    summary = summarize_sweep(result)
    paths = write_sweep_outputs(summary, args.outdir)

    print()
    print(format_sweep_summary(summary))
    print(f"\nSaved: {paths['metrics']}\n       {paths['summary']}")


if __name__ == "__main__":
    main()
