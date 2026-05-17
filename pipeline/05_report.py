"""
Step 5 - Report Generation

Assembles the final allocation CSV from the outputs of the previous steps.
Computes portfolio index statistics from historical returns and appends a
PORTFOLIO INDEX summary row.

Reads  (data/): 01_returns.csv, 03_expected_returns.csv, 03_metadata.json,
                04_weights.csv
Outputs:
    results/allocation_output.csv - final human-readable allocation table
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import risk_kit as rk

from config import load_config, PATHS, BASE_DIR


def main():
    cfg = load_config()

    print("\n=== Step 5: Generating Report ===")

    weights_df       = pd.read_csv(PATHS['04_weights'],           index_col=0)
    expected_returns = pd.read_csv(PATHS['03_expected_returns'],  index_col=0).iloc[:, 0]
    rets             = pd.read_csv(PATHS['01_returns'],           index_col=0)

    with open(PATHS['03_metadata']) as f:
        metadata = json.load(f)

    weights_series      = weights_df['Weights']
    investment_cop      = cfg['investment_cop']
    periods_per_year    = cfg['periods_per_year']
    rf_rate             = cfg['rf_rate']
    periods_to_forecast = cfg['periods_to_forecast']

    current_prices    = pd.Series(metadata['current_prices'])
    forecasted_prices = pd.Series(metadata['forecasted_prices'])
    last_future_date  = metadata['future_dates'][-1]

    cop_per_stock = (weights_series * investment_cop / 1_000).round(2)

    # Portfolio index: track each stock's growth weighted by its allocation,
    # then sum across stocks to get a single index series.
    allocated_index = (rets[weights_series.index] + 1).cumprod() * weights_series
    allocated_index['PORTFOLIO'] = allocated_index.sum(axis=1)
    allocated_index_rets = (allocated_index / allocated_index.shift(1) - 1).dropna()

    portfolio_stats = rk.summary_stats(
        allocated_index_rets[['PORTFOLIO']],
        periods_per_year=periods_per_year,
        riskfree_rate=rf_rate
    ).loc['PORTFOLIO']

    output = pd.DataFrame({
        'Portfolio Weight':                       weights_series.round(4),
        'Expected Annual Return':                 expected_returns[weights_series.index].round(4),
        'Current Price':                          current_prices[weights_series.index].round(4),
        f'Forecasted Price ({last_future_date})': forecasted_prices[weights_series.index].round(4),
        'Investment (COP k)':                     cop_per_stock,
    }).sort_values('Portfolio Weight', ascending=False)

    portfolio_forecasted = (
        cop_per_stock
        * (1 + expected_returns[weights_series.index]) ** (periods_to_forecast / periods_per_year)
    ).sum().round(2)

    portfolio_row = pd.DataFrame({
        'Portfolio Weight':                       [1],
        'Expected Annual Return':                 [round(portfolio_stats['Annualized Return'], 4)],
        'Current Price':                          [cop_per_stock.sum()],
        f'Forecasted Price ({last_future_date})': [portfolio_forecasted],
        'Investment (COP k)':                     [cop_per_stock.sum()],
    }, index=['PORTFOLIO INDEX'])

    output = pd.concat([output, portfolio_row])

    output_path = PATHS['05_report']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path)

    print(f"\nAllocation saved to: {output_path}")
    print(f"\n{'-' * 70}")
    print(output.to_string())
    print(f"{'-' * 70}")
    print(f"\nTotal invested: COP {cop_per_stock.sum() / 1_000:.2f}M across {len(weights_series)} stocks")


if __name__ == '__main__':
    main()
