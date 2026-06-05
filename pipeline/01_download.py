"""
Step 1 - Download and Preprocess Stock Data (parallel batches + activity filter)

Downloads Close+Volume for every ticker/ISIN in stock_tickers/*.csv in parallel batches, prunes the
universe early by an activity filter (keep stocks that trade in >= liquidity_min_active_fraction of
recent periods) plus the bad-data drop, and writes the PRUNED prices/returns.

Outputs (data/):
    01_prices.csv     - adjusted close prices for the kept (active) universe
    01_returns.csv    - period returns for the kept universe
    01_liquidity.csv  - per kept ticker: avg_dollar_volume (info), active_fraction, kept (audit)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from config import load_config, PATHS, BASE_DIR
from data_intake import load_tickers, download_all, activity_filter, activity_health


def main():
    cfg = load_config()
    t0 = time.time()
    print("\n=== Step 1: Download (parallel batches) + activity filter ===")

    tickers = load_tickers(os.path.join(BASE_DIR, "stock_tickers", "*.csv"))
    print(f"Loaded {len(tickers)} unique tickers.")

    close, volume = download_all(tickers, cfg)
    print(f"Downloaded {close.shape[1]} valid tickers.")
    if close.shape[1] < 0.80 * len(tickers):
        print(f"  WARNING: only {close.shape[1]}/{len(tickers)} tickers downloaded "
              f"({len(tickers) - close.shape[1]} lost to batch failures / missing data).")

    detail = activity_filter(
        close, volume,
        window=cfg["liquidity_window"],
        min_active_fraction=cfg["liquidity_min_active_fraction"],
    )
    health = activity_health(detail)
    print(f"Activity filter: kept {health['n_kept']}/{health['n_total']} "
          f"(excluded {health['n_excluded']}; zero-volume {health['zero_volume_fraction']:.0%})")
    if health["zero_volume_fraction"] > 0.25:
        print("  WARNING: many stocks have no Volume at all -> likely a Volume data-source problem.")

    kept = detail.index[detail["kept"]]
    print(f"Kept after activity filter: {len(kept)} / {close.shape[1]}")

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
