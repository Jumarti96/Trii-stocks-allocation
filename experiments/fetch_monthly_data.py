"""
Fetch MONTHLY price/return data into an experiment-local directory, fully isolated from the
weekly production pipeline (params.yaml and data/ are never touched).

Mirrors pipeline/01_download.py's download + clean logic but forces interval=1mo and writes to
experiments/data_monthly/. Run once; reused by the monthly environment-robustness backtest.

Run:  python experiments/fetch_monthly_data.py
Outputs:
    experiments/data_monthly/01_prices.csv
    experiments/data_monthly/01_returns.csv

See docs/superpowers/specs/2026-06-04-monthly-env-backtest-design.md.
"""

import datetime
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import load_config, BASE_DIR as CFG_BASE_DIR

MONTHLY_DIR = os.path.join(BASE_DIR, "experiments", "data_monthly")
MONTHLY_PRICES = os.path.join(MONTHLY_DIR, "01_prices.csv")
MONTHLY_RETURNS = os.path.join(MONTHLY_DIR, "01_returns.csv")


def combine_duplicate_rows(df):
    """Keep first non-null value when the source returns duplicate rows for a period."""
    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else np.nan
    return df.groupby(df.index).agg(first_non_null)


def clean_to_prices_returns(stocks_raw, period_freq, missing_frac=0.15):
    """Clean a raw Close frame into (prices, returns) with string period-end indices.

    Mirrors 01_download.py: convert the DatetimeIndex to periods, collapse duplicate rows,
    drop tickers with >= missing_frac missing, forward/back-fill, compute per-period returns,
    and stringify the index (weekly 'a/b' -> 'b'; monthly stays 'YYYY-MM'). Pure (no network,
    no today()-based trim), so it is unit-testable.
    """
    raw = stocks_raw.copy()
    raw.index = raw.index.to_period(freq=period_freq)
    stocks = combine_duplicate_rows(raw).sort_index()
    keep = stocks.columns[stocks.isna().sum() < stocks.shape[0] * missing_frac]
    stocks = stocks[keep].ffill().bfill()
    rets = stocks.pct_change().iloc[1:]
    stocks.index = stocks.index.astype("str").str.split("/").str[-1]
    rets.index = rets.index.astype("str").str.split("/").str[-1]
    return stocks, rets


def download_monthly_prices(ticker_list, days_of_data):
    """Download monthly adjusted-close prices from yfinance (network)."""
    import yfinance as yf
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_of_data)
    raw = yf.download(
        ticker_list, interval="1mo", start=start_date, end=end_date, auto_adjust=True
    )["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    return raw


def main():
    cfg = load_config()
    csv_files = glob.glob(os.path.join(CFG_BASE_DIR, "stock_tickers", "*.csv"))
    ticker_list = list({
        ticker
        for csv_file in csv_files
        for ticker in pd.read_csv(csv_file, header=None)[0].tolist()
    })
    print(f"Loaded {len(ticker_list)} unique tickers from {len(csv_files)} CSV file(s).")

    raw = download_monthly_prices(ticker_list, cfg["days_of_data"])
    prices, rets = clean_to_prices_returns(raw, period_freq="M")

    os.makedirs(MONTHLY_DIR, exist_ok=True)
    prices.to_csv(MONTHLY_PRICES)
    rets.to_csv(MONTHLY_RETURNS)

    # Sanity: how does the monthly universe compare to the weekly production one?
    try:
        weekly_cols = pd.read_csv(
            os.path.join(CFG_BASE_DIR, "data", "01_returns.csv"), index_col=0, nrows=0
        ).columns
        overlap = len([c for c in rets.columns if c in weekly_cols])
        print(f"Monthly universe: {rets.shape[1]} names | weekly: {len(weekly_cols)} | "
              f"overlap: {overlap}")
    except FileNotFoundError:
        print(f"Monthly universe: {rets.shape[1]} names (weekly file not found for overlap)")

    print(f"Monthly points: {rets.shape[0]}")
    print(f"Saved: {MONTHLY_PRICES}\n       {MONTHLY_RETURNS}")


if __name__ == "__main__":
    main()
