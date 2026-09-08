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

import argparse
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
                         resolve_listings, fetch_fx_rates, convert_panel,
                         sanitise_fx, drop_bad_prices, drop_implausible_names)


def _load_checkpoint():
    """The price/volume panels a previous run already fetched, or None."""
    if not (os.path.exists(PATHS["01_prices"]) and os.path.exists(PATHS["01_volume"])):
        return None
    close = pd.read_csv(PATHS["01_prices"], index_col=0)
    volume = pd.read_csv(PATHS["01_volume"], index_col=0)
    return close, volume


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true",
                    help="reuse the price panels and the listings already resolved "
                         "on disk; re-fetch only what is missing or failed. Prices "
                         "cost ~20 min and listings ~1.5s each, so a run interrupted "
                         "by a rate limit is repaired rather than repeated.")
    ap.add_argument("--listing-workers", type=int, default=None,
                    help="override download_workers for the .info pass only. Use 1 "
                         "when recovering from a rate limit -- concurrency is what "
                         "triggers it.")
    ap.add_argument("--listing-pause", type=float, default=0.0,
                    help="seconds to wait between .info calls (single-worker only)")
    args = ap.parse_args(argv)

    cfg = load_config()
    t0 = time.time()
    print("\n=== Step 1: Download (parallel batches) + activity filter ===")

    tickers = load_tickers(os.path.join(BASE_DIR, "stock_tickers", "*.csv"))
    print(f"Loaded {len(tickers)} unique tickers.")

    cached = _load_checkpoint() if args.resume else None
    if cached is not None:
        close, volume = cached
        print(f"--resume: reusing {close.shape[1]} downloaded tickers from disk "
              f"(no price download).")
    else:
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
    workers = args.listing_workers or cfg["download_workers"]
    existing = None
    if args.resume and os.path.exists(PATHS["01_currency"]):
        existing = pd.read_csv(PATHS["01_currency"], index_col=0)
    per_call = 0.9 / workers + args.listing_pause
    print(f"Resolving listings for {len(kept)} stocks across {workers} workers "
          f"(~{len(kept) * per_call / 60:.0f} min, cached to 01_currency.csv)...")
    cur_df = resolve_listings(list(close_kept.columns), verbose=True, workers=workers,
                              existing=existing, pause=args.listing_pause)
    cur_df.to_csv(PATHS["01_currency"])

    unknown = sorted(cur_df.index[cur_df["currency"].isna()])
    if unknown:
        print(f"  WARNING: unresolved quote currency, excluded from the universe "
              f"screen: {len(unknown)} -> {unknown[:15]}")
    counts = cur_df["source"].value_counts().to_dict()
    n_minor = int((cur_df["unit_factor"] != 1.0).sum())
    n_renamed = int((cur_df["symbol"] != cur_df.index).sum())
    print(f"  {counts.get('lookup', 0)} by lookup, {counts.get('inferred', 0)} by "
          f"inference, {counts.get('failed', 0)} FAILED (heuristic currency), "
          f"{n_minor} quoted in minor units (pence/cents)")
    print(f"  {n_renamed} identifiers resolved to a different trading symbol "
          f"(ISIN -> ticker)")
    if counts.get("failed"):
        # Heuristic currencies are wrong ~12.5% of the time, and the errors run 1.35x
        # to 100x on the very magnitude the universe screen ranks by. A screen built
        # on this many of them is not trustworthy.
        print(f"  WARNING: {counts['failed']} lookups failed (rate limit?). Their "
              f"currencies are guessed from the ISIN prefix, which is ~87.5% accurate "
              f"and mis-scales minor-unit quotes 100x. Re-run with "
              f"'--resume --listing-workers 1 --listing-pause 1.5' to repair them.")

    currencies = sorted(cur_df["currency"].dropna().unique())
    fx = fetch_fx_rates(currencies, close_kept.index, hub="USD")
    print(f"Currencies: {len(currencies)} -> {currencies}")

    # Bad ticks, repaired before they reach the returns. Both defects below were
    # measured on this catalogue and both are fatal rather than cosmetic: one bad FX
    # print corrupts every stock in that currency, and one zero price yields an
    # infinity that stops LedoitWolf hours into a backtest.
    fx, n_fx_fixed = sanitise_fx(fx)
    if n_fx_fixed:
        print(f"  Repaired {n_fx_fixed} implausible single-period FX tick(s) "
              f"(spike that immediately reverses; e.g. USDIDR=X printed 0.75 for one "
              f"week against a true ~7.5e-5).")
    close_kept, n_px_fixed = drop_bad_prices(close_kept)
    if n_px_fixed:
        print(f"  Repaired {n_px_fixed} non-positive price(s) by carrying the last "
              f"good value forward.")

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

    rets_all = prices_usd.pct_change().iloc[1:]
    rets_all, implausible = drop_implausible_names(rets_all)
    if implausible:
        syms = [str(cur_df.loc[t, "symbol"]) for t in implausible[:10]]
        print(f"  Dropped {len(implausible)} stock(s) whose returns contain an "
              f"impossible move (>400% in one period) -- a quote-unit switch inside "
              f"the source series, which no single unit_factor can correct: {syms}")

    # Every artifact is written on the SAME name set, so downstream steps that zip
    # prices against returns against volume cannot silently misalign.
    final = [c for c in prices_usd.columns if c in set(rets_all.columns)]
    close_kept = close_kept[final]
    volume_kept = volume_kept[final]
    prices_usd = prices_usd[final]
    rets = rets_all[final]

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
