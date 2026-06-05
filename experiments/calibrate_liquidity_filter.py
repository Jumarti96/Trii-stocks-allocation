"""
Calibrate the liquidity filter: download a ticker source once, then sweep the pct_of_median
threshold to see how many stocks survive (total + per market group), so the production
liquidity_pct_of_median default can be chosen empirically. Also reports grouping health.

Run:  python experiments/calibrate_liquidity_filter.py --sources stock_tickers/isins_list.csv
      python experiments/calibrate_liquidity_filter.py \
        --sources stock_tickers/colombia_stocks_trii.csv,stock_tickers/global_stocks_trii.csv \
        --list-names

Imports the REAL production functions from src/data_intake. Local/experiment-only.
See docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import load_config
from data_intake import load_tickers, download_all, liquidity_filter, grouping_health


def sweep_thresholds(close, volume, window, grid, min_group_size):
    """For each pct_of_median in grid, count survivors (total and per market group).

    Returns a DataFrame indexed by threshold with a 'kept_total' column plus one column per market
    group ('kept_<group>').
    """
    rows = {}
    for pct in grid:
        detail = liquidity_filter(close, volume, window, pct, min_group_size)
        kept = detail[detail["kept"]]
        row = {"kept_total": int(len(kept))}
        for g, sub in kept.groupby("market_group"):
            row[f"kept_{g}"] = int(len(sub))
        rows[pct] = row
    table = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    table.index.name = "pct_of_median"
    return table


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=str, required=True,
                        help="Comma-separated ticker CSV paths to analyse together")
    parser.add_argument("--grid", type=str, default="0,0.01,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--list-names", action="store_true",
                        help="Also dump kept/excluded ticker names (for small sources)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "liquidity_calibration"))
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    grid = [float(x) for x in args.grid.split(",")]
    sources = args.sources.split(",")
    os.makedirs(args.outdir, exist_ok=True)

    tickers = sorted({t for s in sources for t in load_tickers(s)})
    print(f"Sources: {sources} | {len(tickers)} tickers")
    close, volume = download_all(tickers, cfg)
    print(f"Downloaded {close.shape[1]} valid tickers.")

    table = sweep_thresholds(close, volume, cfg["liquidity_window"], grid,
                             cfg["liquidity_min_group_size"])
    tag = "_".join(os.path.splitext(os.path.basename(s))[0] for s in sources)[:60]
    table.to_csv(os.path.join(args.outdir, f"survival_{tag}.csv"))

    detail = liquidity_filter(close, volume, cfg["liquidity_window"],
                              cfg["liquidity_pct_of_median"], cfg["liquidity_min_group_size"])
    health = grouping_health(detail)
    health["groups"].to_csv(os.path.join(args.outdir, f"grouping_{tag}.csv"))

    print("\nSurvival vs threshold:")
    print(table.to_string())
    print(f"\nGrouping: {health['n_groups']} groups | OTHER {health['other_fraction']:.0%} | "
          f"flagged: {health['flagged_groups']}")

    if args.list_names:
        names_path = os.path.join(args.outdir, f"names_{tag}.csv")
        detail.sort_values("avg_dollar_volume", ascending=False).to_csv(names_path)
        print(f"Wrote kept/excluded names: {names_path}")


if __name__ == "__main__":
    main()
