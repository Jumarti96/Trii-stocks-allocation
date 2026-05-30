"""
Step 3 - Technical Signal Filtering (allocation gate)

Computes SMA, EMA, MACD, and PRC signals for every ticker and keeps only those
where at least signal_min_count signals are positive (default: 3 out of 4).

This is purely the allocation gate: it names which stocks are eligible to hold. It no
longer drives what the Transformer (step 2) trains on - the model trains on the full
universe, and step 4 restricts the optimisation to the names selected here.

Reads  (data/): 01_prices.csv, 01_returns.csv
Outputs (data/):
    03_selected_returns.csv - returns for the stocks that passed the filter
    03_selected_prices.csv  - prices for the stocks that passed the filter
    03_signals.csv          - full signal table (all tickers, for inspection)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import risk_kit as rk

from config import load_config, PATHS


def main():
    cfg = load_config()

    print("\n=== Step 3: Computing technical signals (allocation gate) ===")

    stocks = pd.read_csv(PATHS['01_prices'],  index_col=0)
    rets   = pd.read_csv(PATHS['01_returns'], index_col=0)

    ma_terms = cfg['ma_terms']

    signals = []
    for ticker in stocks.columns:
        sig = rk.technical_indicators(
            stocks[ticker],
            indicators=['SMA', 'EMA', 'MACD', 'PRC'],
            ma_terms=ma_terms,
            macd_params=[12, 26, 9],
            return_df=True,
            plot=False,
            signal_tolerance=0.975
        ).iloc[-1]
        sig_df = pd.DataFrame(sig).T
        sig_df.index = [ticker]
        sig_df.rename(columns={ticker: 'Price'}, inplace=True)
        signals.append(sig_df)

    signals = pd.concat(signals, axis=0)

    signals_filtered = signals[
        np.int64(signals['MACD Signal']) +
        np.int64(signals[f'SMA{ma_terms} Signal']) +
        np.int64(signals[f'EMA{ma_terms} Signal']) +
        np.int64(signals[f'PRC Signal']) >= cfg['signal_min_count']
    ]

    selected_returns = rets[signals_filtered.index]
    selected_prices  = stocks[signals_filtered.index]

    print(f"Passed filter: {len(signals_filtered)} / {len(signals)} stocks")
    print(f"Selected: {list(signals_filtered.index)}")

    selected_returns.to_csv(PATHS['03_selected_returns'])
    selected_prices.to_csv(PATHS['03_selected_prices'])
    signals.to_csv(PATHS['03_signals'])

    print(f"Saved: {PATHS['03_selected_returns']}")
    print(f"       {PATHS['03_selected_prices']}")
    print(f"       {PATHS['03_signals']}")


if __name__ == '__main__':
    main()
