"""
Step 1 - Download and Preprocess Stock Data (parallel batches + activity filter)

Downloads Close+Volume for every ticker/ISIN in stock_tickers/*.csv in parallel batches, prunes the
universe early by an activity filter (keep stocks that trade in >= liquidity_min_active_fraction of
recent periods) plus the bad-data drop, and writes the PRUNED prices/returns.

Listings and FX are resolved BEFORE returns are computed, because returns are taken on
USD-converted prices -- see the comment above convert_panel below for why native-price
pct_change() is not summable across a multi-currency universe.

Outputs (data/):
    01_prices.csv     - adjusted close prices, in each stock's NATIVE currency
    01_prices_usd.csv - the same prices converted to USD (the basis of 01_returns)
    01_returns.csv    - period returns in USD for the kept universe
    01_volume.csv     - period volume, for the step-2 universe screen
    01_currency.csv   - per identifier: quote currency, minor-unit factor, trading
                        symbol, name, resolution source, sector, industry, market cap,
                        exchange, quote type
    01_fx.csv         - per-period conversion rate into USD, one column per currency
    01_liquidity.csv  - per kept ticker: avg_dollar_volume (info), active_fraction, kept (audit)

Every price/volume/returns artifact is written on the same name set: stocks whose
currency cannot be resolved are dropped from all of them together.
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
                         resolve_listings, fetch_fx_rates, convert_panel)


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

    # Checkpoint the expensive part before anything that can raise. The download is
    # ~45 minutes on a 3.9k catalogue; fetch_fx_rates deliberately raises when a
    # currency has no available pair, and resolve_listings is another ~15 minutes of
    # network. Losing all of that to a failure in the last 10% would be maddening,
    # so the raw panels land now and are rewritten below on the final name set.
    os.makedirs(os.path.dirname(PATHS["01_prices"]), exist_ok=True)
    close_kept.to_csv(PATHS["01_prices"])
    volume_kept.to_csv(PATHS["01_volume"])

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
    print(f"Currencies: {len(currencies)} -> {currencies}")

    # Returns are computed on USD-converted prices, NOT on native ones. A flat
    # COP-quoted stock held through a 20% peso depreciation lost 20% in the hands of
    # anyone whose wealth is not measured in pesos, and pct_change() on the native
    # price scores that as zero. Summing native returns across 40-odd currencies adds
    # up quantities that are not the same unit, and it makes any comparison against a
    # USD benchmark such as the S&P 500 meaningless.
    #
    # 01_prices.csv stays NATIVE on purpose: select_universe (step 2) and the report
    # (step 4) each apply their own conversion from it, and pre-converting would make
    # them convert twice.
    prices_usd = convert_panel(close_kept, cur_df["currency"].to_dict(), fx,
                               unit_factors=cur_df["unit_factor"].to_dict(),
                               unknown=cfg["unknown_currency"])
    unconverted = [t for t in close_kept.columns if t not in prices_usd.columns]
    if unconverted:
        print(f"  Dropped {len(unconverted)} stocks with no usable FX rate "
              f"(unknown_currency={cfg['unknown_currency']}): {unconverted[:15]}")

    # Every artifact is written on the SAME name set, so downstream steps that zip
    # prices against returns against volume cannot silently misalign.
    final = list(prices_usd.columns)
    close_kept = close_kept[final]
    volume_kept = volume_kept[final]
    rets = prices_usd.pct_change().iloc[1:]

    close_kept.to_csv(PATHS["01_prices"])          # rewritten on the final name set
    volume_kept.to_csv(PATHS["01_volume"])
    rets.to_csv(PATHS["01_returns"])
    fx.to_csv(PATHS["01_fx"])
    data_dir = os.path.dirname(PATHS["01_prices"])
    prices_usd.to_csv(os.path.join(data_dir, "01_prices_usd.csv"))
    detail.loc[final].to_csv(os.path.join(data_dir, "01_liquidity.csv"))

    print(f"Prices  shape: {close_kept.shape} (native)")
    print(f"Returns shape: {rets.shape} (USD)")
    print(f"  Step 1 completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
