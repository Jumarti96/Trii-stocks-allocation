"""
Walk-forward backtest comparing allocation methods on realized out-of-sample returns.

At each monthly rebalance the transformer is retrained on data up to that date (expanding
window); every arm reuses that one forecast. Arms: current, parametric Michaud (s sweep),
empirical Michaud, equal-weight. Frictionless headline; per-rebalance turnover and weights
are saved so net-of-cost analysis is a free follow-up.

Run:  python experiments/backtest_allocation.py --oos-periods 162 --rebalance-every 4 --n-runs 50
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).
"""

import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS
from measure_allocation_stability import (
    allocate_msr, resampled_allocate, sample_mu_draws,
    select_stocks, train_runs_as_preds, seed_everything,
)


def realized_block_return(weights, block_rets):
    """Buy-and-hold realized return of `weights` over a block of actual per-period returns.

    weights: Series of target weights over held names. block_rets: DataFrame (periods x names)
    of ACTUAL per-period simple returns for the block. Each held name compounds over the block;
    the portfolio return is the weighted sum of per-name compounded returns. Only names present
    in both weights and block_rets are used (names in weights but absent from block_rets are dropped, i.e. treated as 0% cash). Returns a float.
    """
    names = [n for n in weights.index if n in block_rets.columns]
    compounded = (1.0 + block_rets[names]).prod(axis=0) - 1.0
    return float((weights[names] * compounded).sum())


def pairwise_turnover(w_prev, w_new):
    """Half the L1 distance between two target-weight Series.

    Aligns on the union of names (missing = 0). Returns a float in [0, 1] when both weight vectors sum to 1.
    """
    names = w_prev.index.union(w_new.index)
    a = w_prev.reindex(names, fill_value=0.0)
    b = w_new.reindex(names, fill_value=0.0)
    return float(0.5 * (a - b).abs().sum())


def max_drawdown(block_returns):
    """Maximum drawdown of the equity curve built by compounding `block_returns`.

    Returns the max peak-to-trough decline as a non-negative fraction (0.2 = 20% drop).
    0.0 if the curve never declines or the input is empty.
    """
    r = np.asarray(block_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    running_max = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / running_max
    return float(drawdowns.max())


def annualized_stats(block_returns, blocks_per_year, rf_period):
    """Annualized return/vol/Sharpe from a series of per-block simple returns.

    block_returns: per-block returns (each block spans one rebalance). blocks_per_year:
    blocks per year (periods_per_year / rebalance_every). rf_period: the per-BLOCK
    risk-free return for the Sharpe excess. Returns dict {cum_return, ann_return (geometric),
    ann_vol, sharpe}. NaN entries when the series is empty or has zero volatility.
    """
    r = np.asarray(block_returns, dtype=float)
    n = len(r)
    if n == 0:
        return {"cum_return": float("nan"), "ann_return": float("nan"),
                "ann_vol": float("nan"), "sharpe": float("nan")}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann_return = float((1.0 + cum) ** (blocks_per_year / n) - 1.0)
    vol_block = float(r.std(ddof=0))
    ann_vol = vol_block * math.sqrt(blocks_per_year)
    if vol_block > 1e-14:  # tolerance for numerical zero
        sharpe = ((float(r.mean()) - rf_period) / vol_block) * math.sqrt(blocks_per_year)
    else:
        sharpe = float("nan")
    return {"cum_return": cum, "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe}


def summarize_arm(block_returns, turnovers, n_held, blocks_per_year, rf_period):
    """Full per-arm metric row: annualized stats + max DD + turnover + composition.

    turnovers may contain None (e.g. the first rebalance) -- None entries are ignored.
    n_held: per-block count of held names. Returns dict with cum_return, ann_return,
    ann_vol, sharpe, max_dd, mean_turnover, avg_names, hit_rate.
    """
    out = dict(annualized_stats(block_returns, blocks_per_year, rf_period))
    r = np.asarray(block_returns, dtype=float)
    valid_turn = [t for t in turnovers if t is not None]
    out["max_dd"] = max_drawdown(block_returns)
    out["mean_turnover"] = float(np.mean(valid_turn)) if valid_turn else float("nan")
    out["avg_names"] = float(np.mean(n_held)) if len(n_held) else float("nan")
    out["hit_rate"] = float((r > 0).mean()) if len(r) else float("nan")
    return out


def equal_weight(names):
    """Equal (1/N) weight Series over `names`."""
    names = list(names)
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series([1.0 / n] * n, index=names)


def label_of(arm):
    """Stable string label for an arm identifier."""
    if isinstance(arm, tuple) and arm[0] == "parametric":
        return f"parametric_s{arm[1]:g}"
    return arm


def compute_arm_weights(arm, mu_bar, per_run_mu, covmat, cfg, n_periods, mc_draws, rng):
    """Map an arm identifier to a target-weight Series over the eligible names.

    arm: "current" | "empirical" | "equal_weight" | ("parametric", s). mu_bar: averaged
    per-period mu over the eligible names. per_run_mu: list of per-run mu Series (eligible
    names) for the empirical arm. covmat: Ledoit-Wolf cov over the eligible names. n_periods:
    T backing covmat (parametric Sigma/T scale). mc_draws: K parametric draws. rng: numpy
    Generator. Reuses allocate_msr / resampled_allocate / sample_mu_draws. Returns a Series
    over mu_bar.index (dropped names = 0.0).
    """
    if arm == "current":
        return allocate_msr(mu_bar, covmat, cfg)
    if arm == "equal_weight":
        return equal_weight(list(mu_bar.index))
    if arm == "empirical":
        consensus, _ = resampled_allocate(per_run_mu, covmat, cfg)
        return consensus
    if isinstance(arm, tuple) and arm[0] == "parametric":
        draws = sample_mu_draws(mu_bar, covmat, n_periods, mc_draws, arm[1], rng)
        consensus, _ = resampled_allocate(draws, covmat, cfg)
        return consensus
    raise ValueError(f"unknown arm: {arm!r}")


def run_backtest(prices, rets, cfg, oos_periods, rebalance_every, n_runs, mc_draws,
                 spreads, seed, arms=None, runs_fn=None, period_mu_fn=None,
                 select_fn=None, seed_fn=None):
    """Walk-forward backtest: retrain once per rebalance and score all arms on realized returns.

    Steps t from len(rets)-oos_periods by rebalance_every while a full block (t+rebalance_every)
    fits. At each t: train on rets[:t] (expanding), filter on prices[:t], Sigma=Ledoit-Wolf(rets[:t]),
    build each arm's weights over the eligible names, then realize buy-and-hold over
    rets[t:t+rebalance_every]. arms defaults to the 6-arm set built from `spreads`. The four *_fn
    are dependency-injection seams (default to the real torch-backed implementations).

    Returns dict label -> {block_returns, weights (list of Series), turnover (list, first None),
    n_held (list), dates (list)}, plus "rebalance_index" -> list of t indices.
    """
    if arms is None:
        arms = (["current"]
                + [("parametric", s) for s in spreads]
                + ["empirical", "equal_weight"])
    if runs_fn is None:
        runs_fn = train_runs_as_preds
    if period_mu_fn is None:
        from transformer_model import weighted_mean_return
        period_mu_fn = weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    T = len(rets)
    start = T - oos_periods
    rebalance_index = list(range(start, T - rebalance_every + 1, rebalance_every))

    labels = [label_of(a) for a in arms]
    results = {lab: {"block_returns": [], "weights": [], "turnover": [],
                     "n_held": [], "dates": []} for lab in labels}
    prev_weights = {lab: None for lab in labels}

    for k, t in enumerate(rebalance_index):
        seed_fn(seed + k)
        rets_hist = rets.iloc[:t]
        prices_hist = prices.iloc[:t]
        runs = runs_fn(rets_hist, cfg, n_runs=n_runs, verbose=False)
        selected = select_fn(prices_hist, rets_hist, cfg)
        covmat = pd.DataFrame(
            LedoitWolf().fit(rets_hist).covariance_,
            index=rets_hist.columns, columns=rets_hist.columns,
        )
        cov_sel = covmat.loc[selected, selected]
        preds_avg = sum(r.values for r in runs) / len(runs)
        preds_avg = pd.DataFrame(preds_avg, columns=runs[0].columns)
        mu_bar = period_mu_fn(preds_avg).loc[selected]
        per_run_mu = [period_mu_fn(r).loc[selected] for r in runs]

        block = rets.iloc[t:t + rebalance_every]
        rng = np.random.default_rng(seed + k)
        for arm, lab in zip(arms, labels):
            w = compute_arm_weights(arm, mu_bar, per_run_mu, cov_sel, cfg,
                                    len(rets_hist), mc_draws, rng)
            held = w[w.abs() > 1e-9]
            tn = pairwise_turnover(prev_weights[lab], w) if prev_weights[lab] is not None else None
            results[lab]["block_returns"].append(realized_block_return(held, block))
            results[lab]["weights"].append(w)
            results[lab]["turnover"].append(tn)
            results[lab]["n_held"].append(int((w.abs() > 1e-9).sum()))
            results[lab]["dates"].append(rets.index[t])
            prev_weights[lab] = w

    results["rebalance_index"] = rebalance_index
    return results


def format_backtest_summary(results, cfg, rebalance_every):
    """Headline metric table (one row per arm) from a run_backtest result."""
    blocks_per_year = cfg["periods_per_year"] / rebalance_every
    rf_block = (1.0 + cfg["rf_period"]) ** rebalance_every - 1.0
    labels = [k for k in results if k != "rebalance_index"]
    rows = []
    for lab in labels:
        d = results[lab]
        m = summarize_arm(d["block_returns"], d["turnover"], d["n_held"],
                          blocks_per_year, rf_block)
        m["arm"] = lab
        rows.append(m)
    cols = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_dd",
            "mean_turnover", "avg_names", "hit_rate"]
    df = pd.DataFrame(rows).set_index("arm")[cols]
    n_blocks = len(results[labels[0]]["block_returns"])
    header = (f"Walk-forward backtest | blocks: {n_blocks} | "
              f"rebalance_every: {rebalance_every} | blocks/yr: {blocks_per_year:g}\n"
              f"(frictionless / gross realized returns)\n")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    return header + "\n" + df.to_string()


def write_backtest_outputs(results, cfg, rebalance_every, outdir):
    """Write returns/turnover/per-arm-weights CSVs and the summary text; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    labels = [k for k in results if k != "rebalance_index"]
    dates = results[labels[0]]["dates"]
    paths = {}

    ret_df = pd.DataFrame({lab: results[lab]["block_returns"] for lab in labels}, index=dates)
    paths["returns"] = os.path.join(outdir, "backtest_returns.csv")
    ret_df.to_csv(paths["returns"], index_label="date")

    turn_df = pd.DataFrame({lab: results[lab]["turnover"] for lab in labels}, index=dates)
    paths["turnover"] = os.path.join(outdir, "backtest_turnover.csv")
    turn_df.to_csv(paths["turnover"], index_label="date")

    for lab in labels:
        w_df = pd.DataFrame(results[lab]["weights"], index=dates).fillna(0.0)
        p = os.path.join(outdir, f"backtest_weights_{lab}.csv")
        w_df.to_csv(p, index_label="date")
        paths[f"weights_{lab}"] = p

    paths["summary"] = os.path.join(outdir, "backtest_summary.txt")
    with open(paths["summary"], "w") as f:
        f.write(format_backtest_summary(results, cfg, rebalance_every))
    return paths


def parse_spreads(text):
    """Parse a comma-separated spreads string into a list of floats."""
    return [float(x) for x in text.split(",") if x.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--oos-periods", type=int, default=162,
                        help="Out-of-sample window length in periods (default 162 ~ 3yr weekly)")
    parser.add_argument("--rebalance-every", type=int, default=4,
                        help="Rebalance/holding period in periods (default 4 = forecast horizon)")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Transformer runs per rebalance (default 50)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="Parametric Monte-Carlo mu draws K (default 1000)")
    parser.add_argument("--spreads", type=str, default="1,2,4",
                        help="Comma-separated parametric spread values (default '1,2,4')")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; rebalance k uses seed+k (default 0)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "backtest"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)
    spreads = parse_spreads(args.spreads)

    print(f"Universe: {rets.shape[1]} stocks | oos_periods={args.oos_periods} | "
          f"rebalance_every={args.rebalance_every} | n_runs={args.n_runs} | "
          f"mc_draws={args.mc_draws} | spreads={spreads} | seed={args.seed}")

    results = run_backtest(
        prices, rets, cfg, oos_periods=args.oos_periods,
        rebalance_every=args.rebalance_every, n_runs=args.n_runs,
        mc_draws=args.mc_draws, spreads=spreads, seed=args.seed,
    )
    paths = write_backtest_outputs(results, cfg, args.rebalance_every, args.outdir)
    print()
    print(format_backtest_summary(results, cfg, args.rebalance_every))
    print("\nSaved:")
    for p in paths.values():
        print(f"       {p}")


if __name__ == "__main__":
    main()
