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
        if train_runs_fn is None:
            train_runs_fn = train_runs
        if winsorize_fn is None:
            winsorize_fn = winsorize_to_history
        if period_mu_fn is None:
            period_mu_fn = weighted_mean_return
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


def seed_arm_metrics(seed_result):
    """Per-(arm, n) scalar metrics for one seed (one row per arm x n).

    Composition is reduced to scalars (turnover/jaccard/overlap/held); per-name
    selection_frequency is intentionally NOT a column here -- it is a per-name vector, not
    a scalar, and the raw per-seed weights CSVs already carry that signal. turnover/jaccard
    are None when iterations < 2; that flows to NaN in the aggregation and is skipped there.
    """
    rows = []
    for arm in seed_result["arms"]:
        for n in seed_result["grid"]:
            w = seed_result["data"][arm][n]["weights"]
            m = seed_result["data"][arm][n]["metrics"]
            ov = overlap_stats(w)
            disp = metric_dispersion(m)
            rows.append({
                "arm": arm, "n": n,
                "turnover": mean_turnover(w),
                "jaccard": mean_jaccard(w),
                "overlap_fraction": ov["fraction"],
                "held": ov["held"],
                "ret_mean": disp["ret"]["mean"], "ret_cov": disp["ret"]["cov"],
                "vol_mean": disp["vol"]["mean"], "vol_cov": disp["vol"]["cov"],
                "sharpe_mean": disp["sharpe"]["mean"], "sharpe_cov": disp["sharpe"]["cov"],
            })
    return pd.DataFrame(rows)


METRIC_COLS = [
    "turnover", "jaccard", "overlap_fraction", "held",
    "ret_mean", "ret_cov", "vol_mean", "vol_cov", "sharpe_mean", "sharpe_cov",
]


def summarize_nstudy(results_by_seed):
    """Aggregate per-(arm, n) scalar metrics across seeds as mean and population std.

    Returns {"mean": DataFrame, "std": DataFrame, "metric_cols": [...], "arms": [...],
    "grid": [...]} where mean/std are indexed by (arm, n).
    """
    per_seed = [seed_arm_metrics(r) for r in results_by_seed.values()]
    allm = pd.concat(per_seed, ignore_index=True)
    grouped = allm.groupby(["arm", "n"])[METRIC_COLS]
    any_result = next(iter(results_by_seed.values()))
    return {
        "mean": grouped.mean(),
        "std": grouped.std(ddof=0),
        "metric_cols": METRIC_COLS,
        "arms": any_result["arms"],
        "grid": any_result["grid"],
    }


def first_flattening_n(values_by_n, grid, tol=0.05):
    """First n in grid whose value changed < tol (relative) vs the previous grid point.

    Advisory only -- a hint at where a metric stops moving, not a verdict. Returns None
    if no consecutive pair falls under tol.
    """
    grid = sorted(grid)
    for prev, n in zip(grid[:-1], grid[1:]):
        base = values_by_n.get(prev)
        if base is None or (isinstance(base, float) and math.isnan(base)) or base == 0:
            continue
        cur = values_by_n.get(n)
        if cur is None or (isinstance(cur, float) and math.isnan(cur)):
            continue
        if abs(cur - base) / abs(base) < tol:
            return n
    return None


def _fmt_mean_std(mean, std):
    def _one(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.4f}"
    return f"{_one(mean)}+/-{_one(std)}"


def format_nstudy_summary(summary, primary_arm="s4", tol=0.05):
    """Per-arm metric-vs-n tables (mean +/- cross-seed std) plus the advisory note."""
    mean, std = summary["mean"], summary["std"]
    cols = summary["metric_cols"]
    grid = summary["grid"]

    lines = ["n_transformer_runs study (mean +/- cross-seed std)", ""]
    for arm in summary["arms"]:
        lines.append(f"== arm: {arm} ==")
        header = f"  {'n':>5}  " + "  ".join(f"{c:>20}" for c in cols)
        lines.append(header)
        for n in grid:
            cells = [f"{_fmt_mean_std(mean.loc[(arm, n), c], std.loc[(arm, n), c]):>20}"
                     for c in cols]
            lines.append(f"  {n:>5}  " + "  ".join(cells))
        lines.append("")

    if primary_arm in summary["arms"]:
        turnover_by_n = {n: mean.loc[(primary_arm, n), "turnover"] for n in grid}
        jacc_by_n = {n: mean.loc[(primary_arm, n), "jaccard"] for n in grid}
        t_flat = first_flattening_n(turnover_by_n, grid, tol)
        j_flat = first_flattening_n(jacc_by_n, grid, tol)
        lines.append(
            f"Advisory ({primary_arm}, tol={tol:.0%}): turnover flattens at "
            f"n={t_flat}; Jaccard flattens at n={j_flat}. "
            "A hint at where added runs stop buying stability -- the final pick is yours."
        )
    lines.append("")
    lines.append(
        "Note: in-sample. Read composition (turnover/Jaccard/overlap) and value CoVs, "
        "NOT value levels (current is the in-sample argmax of mu(n)). Realized verdict = "
        "the backtest study."
    )
    return "\n".join(lines)


def _stack_over_grid(arm_data, grid, value="weights"):
    """Concatenate a single arm's per-n frames into one frame with n + iteration columns."""
    frames = []
    for n in grid:
        df = arm_data[n][value].copy()
        df.index.name = "iteration"
        df = df.reset_index()
        df.insert(0, "n", n)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_nstudy_outputs(results_by_seed, summary, outdir):
    """Write per-seed raw CSVs, per-arm aggregated tables, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    grid = summary["grid"]
    arms = summary["arms"]
    paths = {"summary": os.path.join(outdir, "nstudy_summary.txt")}

    for seed, result in results_by_seed.items():
        for arm in arms:
            wpath = os.path.join(outdir, f"nstudy_{arm}_seed{seed}_weights.csv")
            mpath = os.path.join(outdir, f"nstudy_{arm}_seed{seed}_metrics.csv")
            _stack_over_grid(result["data"][arm], grid, "weights").to_csv(wpath, index=False)
            _stack_over_grid(result["data"][arm], grid, "metrics").to_csv(mpath, index=False)
            paths[f"{arm}_seed{seed}_weights"] = wpath
            paths[f"{arm}_seed{seed}_metrics"] = mpath

    for arm in arms:
        table = pd.DataFrame(index=pd.Index(grid, name="n"))
        for c in summary["metric_cols"]:
            table[f"{c}_mean"] = [summary["mean"].loc[(arm, n), c] for n in grid]
            table[f"{c}_std"] = [summary["std"].loc[(arm, n), c] for n in grid]
        tpath = os.path.join(outdir, f"nstudy_table_{arm}.csv")
        table.to_csv(tpath)
        paths[f"table_{arm}"] = tpath

    with open(paths["summary"], "w") as f:
        f.write(format_nstudy_summary(summary))
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=str, default="0,100",
                        help="Comma-separated base seeds (default 0,100; spaced >= iterations)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="predict->allocate passes per seed (default 10)")
    parser.add_argument("--grid", type=str, default="10,25,50,75,100",
                        help="Comma-separated n_transformer_runs values")
    parser.add_argument("--spreads", type=str, default="4,8",
                        help="Comma-separated parametric-Michaud spreads s (default 4,8)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="K Monte-Carlo mu draws per consensus (default 1000)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "nstudy"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    seeds = [int(x) for x in args.seeds.split(",")]
    grid = [int(x) for x in args.grid.split(",")]
    spreads = [float(x) for x in args.spreads.split(",")]

    print(f"Universe: {rets.shape[1]} stocks | seeds={seeds} | iterations={args.iterations} "
          f"| grid={grid} | spreads={spreads} | K={args.mc_draws}", flush=True)

    results_by_seed = {}
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        results_by_seed[seed] = run_nstudy_seed(
            prices, rets, cfg,
            grid=grid, iterations=args.iterations, seed=seed,
            spreads=spreads, n_draws=args.mc_draws, verbose=True,
        )

    summary = summarize_nstudy(results_by_seed)
    paths = write_nstudy_outputs(results_by_seed, summary, args.outdir)

    print()
    print(format_nstudy_summary(summary))
    print("\nSaved:")
    for p in sorted(set(paths.values())):
        print(f"       {p}")


if __name__ == "__main__":
    main()
