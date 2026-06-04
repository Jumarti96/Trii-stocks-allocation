"""
Monthly environment-robustness backtest: run the realized walk-forward backtest in a MONTHLY
regime at short (1-month) and long (6-month) horizons, comparing the allocation methods
current (msr) / parametric s4 / parametric s8 against equal_weight, with the technical filter
DISABLED. Reports realized metrics as mean +/- cross-seed std per horizon.

Reuses experiments/backtest_allocation.run_backtest unchanged; reads MONTHLY data fetched by
experiments/fetch_monthly_data.py. This is a realized out-of-sample test (the verdict is realized
Sharpe etc.), complementary to the weekly multi-seed backtest. In-sample caveats do not apply;
the few-block 6-month horizon is directional, not a verdict.

Run:  python experiments/monthly_env_backtest.py --horizons 1,6 --seeds 0,100 \
        --oos-periods 60 --n-runs 75 --mc-draws 1000 --spreads 4,8
Requires experiments/data_monthly/01_{prices,returns}.csv (run fetch_monthly_data.py first).

See docs/superpowers/specs/2026-06-04-monthly-env-backtest-design.md.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
sys.path.insert(0, HERE)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import load_config
from backtest_allocation import run_backtest, summarize_arm, write_backtest_outputs
from fetch_monthly_data import MONTHLY_PRICES, MONTHLY_RETURNS

METRIC_COLS = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_dd",
               "mean_turnover", "avg_names", "hit_rate"]


def build_monthly_cfg(weekly_cfg, horizon):
    """Copy the weekly production cfg and override the monthly + horizon fields.

    Sets interval/periods_per_year/time_window/period_freq/future_freq/date_offset to monthly,
    periods_to_forecast to the horizon, and recomputes the per-period risk-free rate for ppy=12.
    Filter/MA/weight params are left untouched (the filter is bypassed via select_all, not config).
    """
    cfg = dict(weekly_cfg)
    cfg["interval"] = "1mo"
    cfg["periods_per_year"] = 12
    cfg["time_window"] = 12
    cfg["periods_to_forecast"] = horizon
    cfg["period_freq"] = "M"
    cfg["future_freq"] = "MS"
    cfg["date_offset"] = pd.DateOffset(months=1)
    cfg["rf_period"] = (1 + cfg["rf_rate"]) ** (1 / 12) - 1
    return cfg


def select_all(prices, rets, cfg):
    """Filter-disabling selection seam: every name is eligible (no technical gate)."""
    return list(rets.columns)


def run_monthly_env(prices, rets, weekly_cfg, horizons, seeds, oos_periods,
                    n_runs, mc_draws, spreads, runs_fn=None, period_mu_fn=None,
                    seed_fn=None):
    """Run the monthly backtest over horizons x seeds; filter disabled via select_all.

    Arms = current + one parametric arm per spread + equal_weight. For each horizon a monthly cfg
    is built (periods_to_forecast = rebalance_every = horizon) and run_backtest is called once per
    seed. The *_fn args are dependency-injection seams passed straight through to run_backtest so
    the loop runs without torch. Returns {horizon: {"cfg": cfg_h, "per_seed": {seed: results}}}.
    """
    arms = ["current"] + [("parametric", s) for s in spreads] + ["equal_weight"]
    out = {}
    for horizon in horizons:
        cfg_h = build_monthly_cfg(weekly_cfg, horizon)
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = run_backtest(
                prices, rets, cfg_h, oos_periods=oos_periods, rebalance_every=horizon,
                # spreads is only used by run_backtest when arms is None; arms is pre-built
                # above (parametric arms carry their own s), so spreads is inert here.
                n_runs=n_runs, mc_draws=mc_draws, spreads=spreads, seed=seed, arms=arms,
                select_fn=select_all, runs_fn=runs_fn, period_mu_fn=period_mu_fn,
                seed_fn=seed_fn,
            )
        out[horizon] = {"cfg": cfg_h, "per_seed": per_seed}
    return out


def aggregate_across_seeds(per_seed, cfg, rebalance_every):
    """Per-arm realized metrics aggregated across seeds as mean and population std.

    per_seed: {seed: run_backtest result}. Uses backtest_allocation.summarize_arm with
    blocks_per_year = periods_per_year / rebalance_every and the per-block rf. Returns
    {"table": DataFrame indexed by arm with <metric>_mean/<metric>_std columns,
    "n_blocks": int, "metric_cols": METRIC_COLS}.
    """
    blocks_per_year = cfg["periods_per_year"] / rebalance_every
    rf_block = (1.0 + cfg["rf_period"]) ** rebalance_every - 1.0
    seeds = list(per_seed.keys())
    labels = [k for k in per_seed[seeds[0]] if k != "rebalance_index"]

    per_seed_metrics = {
        s: {lab: summarize_arm(per_seed[s][lab]["block_returns"],
                               per_seed[s][lab]["turnover"],
                               per_seed[s][lab]["n_held"],
                               blocks_per_year, rf_block)
            for lab in labels}
        for s in seeds
    }

    rows = []
    for lab in labels:
        row = {"arm": lab}
        for c in METRIC_COLS:
            vals = np.array([per_seed_metrics[s][lab][c] for s in seeds], dtype=float)
            row[f"{c}_mean"] = float(np.nanmean(vals))
            row[f"{c}_std"] = float(np.nanstd(vals))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("arm")
    n_blocks = len(per_seed[seeds[0]][labels[0]]["block_returns"])
    return {"table": table, "n_blocks": n_blocks, "metric_cols": METRIC_COLS}


def format_monthly_summary(agg_by_horizon, few_blocks=15):
    """Per-horizon realized metric tables (mean +/- cross-seed std) with block counts + caveats."""
    lines = ["Monthly environment-robustness backtest (mean +/- cross-seed std)", ""]
    for horizon in sorted(agg_by_horizon):
        a = agg_by_horizon[horizon]
        lines.append(f"== horizon {horizon}-month | blocks: {a['n_blocks']} ==")
        if a["n_blocks"] < few_blocks:
            lines.append("  (few blocks -> realized stats are DIRECTIONAL, not a verdict)")
        for arm, r in a["table"].iterrows():
            cells = "  ".join(
                f"{c}={r[f'{c}_mean']:.4f}+/-{r[f'{c}_std']:.4f}" for c in a["metric_cols"]
            )
            lines.append(f"  {arm:<16} {cells}")
        lines.append("")
    lines.append(
        "Note: technical filter DISABLED (full universe). equal_weight = 1/N anchor; an optimizer "
        "arm that fails to beat it adds no value in this regime. This filter-off MONTHLY result is "
        "confounded with the weekly->monthly change -- NOT clean evidence for removing the filter "
        "from production (that needs its own weekly with-vs-without A/B)."
    )
    return "\n".join(lines)


def write_monthly_outputs(run_out, agg_by_horizon, outdir):
    """Write per-(horizon, seed) raw CSVs, per-horizon aggregated tables, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    paths = {"summary": os.path.join(outdir, "monthly_env_summary.txt")}

    for horizon, hd in run_out.items():
        cfg_h = hd["cfg"]
        for seed, res in hd["per_seed"].items():
            sub = os.path.join(outdir, f"h{horizon}_seed{seed}")
            write_backtest_outputs(res, cfg_h, horizon, sub)
        tpath = os.path.join(outdir, f"monthly_table_h{horizon}.csv")
        agg_by_horizon[horizon]["table"].to_csv(tpath)
        paths[f"table_h{horizon}"] = tpath

    with open(paths["summary"], "w") as f:
        f.write(format_monthly_summary(agg_by_horizon))
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--horizons", type=str, default="1,6",
                        help="Comma-separated forecast horizons in months (default 1,6)")
    parser.add_argument("--seeds", type=str, default="0,100",
                        help="Comma-separated base seeds (default 0,100)")
    parser.add_argument("--oos-periods", type=int, default=60,
                        help="Out-of-sample months held out (default 60)")
    parser.add_argument("--n-runs", type=int, default=75,
                        help="Transformer runs per rebalance (default 75)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="K parametric Monte-Carlo draws (default 1000)")
    parser.add_argument("--spreads", type=str, default="4,8",
                        help="Comma-separated parametric spreads s (default 4,8)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "monthly_env"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    weekly_cfg = load_config()
    prices = pd.read_csv(MONTHLY_PRICES, index_col=0)
    rets = pd.read_csv(MONTHLY_RETURNS, index_col=0)

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    spreads = [float(x) for x in args.spreads.split(",")]

    print(f"Monthly universe: {rets.shape[1]} names | months: {rets.shape[0]} | "
          f"horizons={horizons} | seeds={seeds} | oos={args.oos_periods} | "
          f"n_runs={args.n_runs} | K={args.mc_draws} | spreads={spreads} | filter=OFF",
          flush=True)

    run_out = run_monthly_env(
        prices, rets, weekly_cfg, horizons=horizons, seeds=seeds,
        oos_periods=args.oos_periods, n_runs=args.n_runs, mc_draws=args.mc_draws,
        spreads=spreads,
    )

    agg_by_horizon = {
        h: aggregate_across_seeds(run_out[h]["per_seed"], run_out[h]["cfg"], h)
        for h in horizons
    }
    paths = write_monthly_outputs(run_out, agg_by_horizon, args.outdir)

    print()
    print(format_monthly_summary(agg_by_horizon))
    print("\nSaved:")
    for p in sorted(set(paths.values())):
        print(f"       {p}")


if __name__ == "__main__":
    main()
