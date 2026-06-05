"""
Step 1 - Download and Preprocess Stock Data (parallel batches + liquidity filter)

Downloads Close+Volume for every ticker/ISIN in stock_tickers/*.csv in parallel batches, prunes the
universe early by a currency-robust relative-liquidity filter (avg dollar-volume vs market-group
median, with conservative small/degenerate-group handling), and writes the PRUNED prices/returns.

Outputs (data/):
    01_prices.csv     - adjusted close prices for the kept (liquid) universe
    01_returns.csv    - period returns for the kept universe
    01_liquidity.csv  - per kept ticker: avg dollar-volume, market group, flag (audit)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from config import load_config, PATHS, BASE_DIR
from data_intake import load_tickers, download_all, liquidity_filter, grouping_health


def main():
    cfg = load_config()
    t0 = time.time()
    print("\n=== Step 1: Download (parallel batches) + liquidity filter ===")

    tickers = load_tickers(os.path.join(BASE_DIR, "stock_tickers", "*.csv"))
    print(f"Loaded {len(tickers)} unique tickers.")

    close, volume = download_all(tickers, cfg)
    print(f"Downloaded {close.shape[1]} valid tickers.")

    detail = liquidity_filter(
        close, volume,
        window=cfg["liquidity_window"], pct_of_median=cfg["liquidity_pct_of_median"],
        min_group_size=cfg["liquidity_min_group_size"],
    )
    health = grouping_health(detail)
    print(f"Grouping: {health['n_groups']} groups | OTHER bucket "
          f"{health['other_fraction']:.0%} | flagged groups: {health['flagged_groups']}")
    if health["other_fraction"] > 0.25:
        print("  WARNING: large OTHER bucket -> the ticker list may not fit the grouping patterns.")

    kept = detail.index[detail["kept"]]
    print(f"Kept after liquidity filter: {len(kept)} / {close.shape[1]}")

    close_kept = close[kept]
    rets = close_kept.pct_change().iloc[1:]

    os.makedirs(os.path.dirname(PATHS["01_prices"]), exist_ok=True)
    close_kept.to_csv(PATHS["01_prices"])
    rets.to_csv(PATHS["01_returns"])
    detail.loc[kept].to_csv(os.path.join(os.path.dirname(PATHS["01_prices"]), "01_liquidity.csv"))

    print(f"Prices  shape: {close_kept.shape}")
    print(f"Returns shape: {rets.shape}")
    print(f"  Step 1 completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
