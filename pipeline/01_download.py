"""
Step 1 - Download and Preprocess Stock Data

Downloads historical prices from yfinance, removes tickers with >15% missing
data, forward-fills gaps, computes returns, and trims to the analysis window.

Outputs (data/):
    01_prices.csv   - adjusted close prices, string date index
    01_returns.csv  - period returns, string date index
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from config import load_config, PATHS, BASE_DIR


def combine_duplicate_rows(df):
    """Keep first non-null value when yfinance returns duplicate rows for the latest period."""
    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else np.nan
    return df.groupby(df.index).agg(first_non_null)


def main():
    cfg = load_config()

    print("\n=== Step 1: Downloading stock data ===")

    csv_files   = glob.glob(os.path.join(BASE_DIR, 'stock_tickers', '*.csv'))
    ticker_list = list({
        ticker
        for csv_file in csv_files
        for ticker in pd.read_csv(csv_file, header=None)[0].tolist()
    })
    print(f"Loaded {len(ticker_list)} unique tickers from {len(csv_files)} CSV file(s).")

    end_date   = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=cfg['days_of_data'])

    stocks_raw = yf.download(
        ticker_list,
        interval=cfg['interval'],
        start=start_date,
        end=end_date,
        auto_adjust=True
    )['Close']
    if isinstance(stocks_raw, pd.Series):
        stocks_raw = stocks_raw.to_frame()
    stocks_raw.index = stocks_raw.index.to_period(freq=cfg['period_freq'])

    stocks = combine_duplicate_rows(stocks_raw).sort_index()

    # Drop tickers with more than 15% missing data, then forward-fill remaining gaps
    stocks_not_missing = stocks.columns[stocks.isna().sum() < stocks.shape[0] * 0.15]
    stocks = stocks[stocks_not_missing].ffill().bfill()

    rets = stocks.pct_change().iloc[1:]

    # Trim to the configured analysis window
    analysis_end   = str(datetime.date.today())
    analysis_start = str(datetime.date.today() - datetime.timedelta(days=cfg['days_of_data']))
    rets   = rets.loc[analysis_start:analysis_end]
    stocks = stocks.loc[analysis_start:analysis_end]

    # Convert PeriodIndex to end-of-period date strings before writing.
    # Weekly periods come as "yyyy-mm-dd/yyyy-mm-dd"; monthly as "yyyy-mm".
    # str.split('/').str[-1] handles both forms.
    stocks.index = stocks.index.astype('str').str.split('/').str[-1]
    rets.index   = rets.index.astype('str').str.split('/').str[-1]

    os.makedirs(os.path.dirname(PATHS['01_prices']), exist_ok=True)
    stocks.to_csv(PATHS['01_prices'])
    rets.to_csv(PATHS['01_returns'])

    print(f"Prices  shape: {stocks.shape}")
    print(f"Returns shape: {rets.shape}")
    print(f"Saved: {PATHS['01_prices']}")
    print(f"       {PATHS['01_returns']}")


if __name__ == '__main__':
    main()
