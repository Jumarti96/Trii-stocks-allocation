"""
Step 1 - Download and Preprocess Stock Data (parallel batches + activity filter)

Downloads Close+Volume for every ticker/ISIN in stock_tickers/*.csv in parallel batches, prunes the
universe early by an activity filter (keep stocks that trade in >= liquidity_min_active_fraction of
recent periods) plus the bad-data drop, and writes the PRUNED prices/returns.

Outputs (data/):
    01_prices.csv     - adjusted close prices for the kept (active) universe
    01_returns.csv    - period returns for the kept universe
    01_volume.csv     - period volume, for the step-2 universe screen
    01_currency.csv   - per identifier: quote currency, minor-unit factor, trading
                        symbol, name, resolution source
    01_fx.csv         - per-period conversion rate into USD, one column per currency
    01_liquidity.csv  - per kept ticker: avg_dollar_volume (info), active_fraction, kept (audit)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import load_config, PATHS, BASE_DIR
from data_intake import (load_tickers, download_all, activity_filter, activity_health,
                         resolve_listings, fetch_fx_rates)


def main():
    cfg = load_config()
    t0 = time.time()
    print("\n=== Step 1: Download (parallel batches) + activity filter ===")

    tickers = load_tickers(os.path.join(BASE_DIR, "stock_tickers", "*.csv"))
    print(f"Loaded {len(tickers)} unique tickers.")

    close, volume = download_all(tickers, cfg)
    print(f"Downloaded {close.shape[1]} valid tickers.")
    if close.shape[1] < cfg["download_warn_fraction"] * len(tickers):
        print(f"  WARNING: only {close.shape[1]}/{len(tickers)} tickers downloaded "
              f"({len(tickers) - close.shape[1]} lost to batch failures / missing data).")

    detail = activity_filter(close, volume)
    health = activity_health(detail)
    print(f"Activity filter: kept {health['n_kept']}/{health['n_total']} "
          f"(excluded {health['n_excluded']}; zero-volume {health['zero_volume_fraction']:.0%})")
    if health["zero_volume_fraction"] > cfg["zero_volume_warn_threshold"]:
        print("  WARNING: many stocks have no Volume at all -> likely a Volume data-source problem.")

    kept = detail.index[detail["kept"]]
    print(f"Kept after activity filter: {len(kept)} / {close.shape[1]}")

    close_kept = close[kept]
    volume_kept = volume[kept]
    rets = close_kept.pct_change().iloc[1:]

    os.makedirs(os.path.dirname(PATHS["01_prices"]), exist_ok=True)
    close_kept.to_csv(PATHS["01_prices"])
    rets.to_csv(PATHS["01_returns"])
    volume_kept.to_csv(PATHS["01_volume"])
    detail.loc[kept].to_csv(os.path.join(os.path.dirname(PATHS["01_prices"]), "01_liquidity.csv"))

    # Quote currency + FX to USD. Resolved here (the download step) so step 2 can
    # re-screen the universe repeatedly without touching the network -- experiments
    # sweep universe size off a single download.
    #
    # One network call per stock, because the suffix/ISIN heuristics measured only
    # 87.5% accurate and fail in the direction that matters: cross-listed ETFs
    # (CSPX.L is USD despite .L), KY/CN issuers listed in Hong Kong, and cents-quoted
    # Johannesburg lines. Errors run from 1.35x to 100x on exactly the magnitude the
    # universe screen ranks by.
    workers = cfg["download_workers"]
    print(f"Resolving listings for {len(kept)} stocks across {workers} workers "
          f"(~{len(kept) * 0.9 / 60 / workers:.0f} min, cached to 01_currency.csv)...")
    cur_df = resolve_listings(list(close_kept.columns), verbose=True, workers=workers)
    cur_df.to_csv(PATHS["01_currency"])

    unknown = sorted(cur_df.index[cur_df["currency"].isna()])
    if unknown:
        print(f"  WARNING: unresolved quote currency, excluded from the universe "
              f"screen: {len(unknown)} -> {unknown[:15]}")
    n_inferred = int((cur_df["source"] == "inferred").sum())
    n_minor = int((cur_df["unit_factor"] != 1.0).sum())
    n_renamed = int((cur_df["symbol"] != cur_df.index).sum())
    print(f"  {len(cur_df) - n_inferred} by lookup, {n_inferred} by inference "
          f"fallback, {n_minor} quoted in minor units (pence/cents)")
    print(f"  {n_renamed} identifiers resolved to a different trading symbol "
          f"(ISIN -> ticker)")

    currencies = sorted(cur_df["currency"].dropna().unique())
    fx = fetch_fx_rates(currencies, close_kept.index, hub="USD")
    fx.to_csv(PATHS["01_fx"])
    print(f"Currencies: {len(currencies)} -> {currencies}")

    print(f"Prices  shape: {close_kept.shape}")
    print(f"Returns shape: {rets.shape}")
    print(f"  Step 1 completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
