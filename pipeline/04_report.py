"""
Step 4 - Report Generation

Assembles the final allocation CSV from the outputs of the previous steps.
Computes portfolio index statistics from historical returns and appends a
PORTFOLIO INDEX summary row.

Prices arrive in each exchange's own quote currency, so they are converted into
cfg['report_currency'] before display -- otherwise the price column mixes units and no
share count can be computed from it. Weights are dimensionless, so the money split was
always correct regardless.

Current and forecasted prices MUST be converted at the same FX rate (both default to
when=-1 below). Expected returns are USD returns, so the shared rate is what makes the
forecast column mean "today's price grown by the forecast USD return, priced in
report_currency" -- converting the two legs at different rates would silently fold an
FX forecast we do not have into the price. See the comment above forecasted_prices in
pipeline/02_predict.py.

The 'Expected Annual Return' column is therefore a USD return, not a return in
report_currency.

Reads  (data/): 01_returns.csv, 01_currency.csv, 01_fx.csv, 02_expected_returns.csv,
                02_metadata.json, 03_weights.csv
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
from data_intake import convert_currency
from transformer_model import annualize_period_return

from config import load_config, PATHS, BASE_DIR


def main():
    cfg = load_config()

    print("\n=== Step 4: Generating Report ===")

    weights_df       = pd.read_csv(PATHS['03_weights'],           index_col=0)
    expected_returns = pd.read_csv(PATHS['02_expected_returns'],  index_col=0).iloc[:, 0]
    rets             = pd.read_csv(PATHS['01_returns'],           index_col=0)

    with open(PATHS['02_metadata']) as f:
        metadata = json.load(f)

    weights_series      = weights_df['Weights']
    investment          = cfg['investment']
    ccy                 = cfg['report_currency']
    periods_per_year    = cfg['periods_per_year']
    rf_rate             = cfg['rf_rate']
    periods_to_forecast = cfg['periods_to_forecast']

    current_prices    = pd.Series(metadata['current_prices'])
    forecasted_prices = pd.Series(metadata['forecasted_prices'])
    last_future_date  = metadata['future_dates'][-1]

    # Prices arrive in each exchange's own quote currency (USD for NVDA, JPY for
    # Toyota, pence for most of the LSE). Weights are dimensionless so the money split
    # was always correct, but a price column mixing currencies cannot be compared and
    # cannot be turned into a share count. Convert everything into report_currency.
    names = list(weights_series.index)
    prices_local = current_prices[names]
    fcast_local  = forecasted_prices[names]
    unconverted  = []

    if os.path.exists(PATHS['01_currency']) and os.path.exists(PATHS['01_fx']):
        cur_df  = pd.read_csv(PATHS['01_currency'], index_col=0)
        fx      = pd.read_csv(PATHS['01_fx'], index_col=0)
        cur_map = cur_df['currency'].to_dict()
        factors = (cur_df['unit_factor'].to_dict()
                   if 'unit_factor' in cur_df.columns else None)
        kw = dict(cur_map=cur_map, fx=fx, target=ccy, unit_factors=factors,
                  unknown=cfg['unknown_currency'])
        current_prices_r = convert_currency(prices_local, **kw)
        forecasted_prices_r = convert_currency(fcast_local, **kw)
        unconverted = [n for n in names if n not in current_prices_r.index]
        symbols = (cur_df['symbol'].to_dict() if 'symbol' in cur_df.columns else {})
    else:
        # Pre-currency data/: report local prices unchanged rather than fail. The
        # column is then mixed-currency, which is what it always was.
        print("  NOTE: no 01_currency.csv/01_fx.csv -- prices left in their listing "
              "currency. Re-run step 1 to enable conversion.")
        current_prices_r, forecasted_prices_r, symbols = prices_local, fcast_local, {}

    if unconverted:
        print(f"  WARNING: {len(unconverted)} holding(s) had no resolvable currency "
              f"and are shown without conversion ({cfg['unknown_currency']}): "
              f"{unconverted}")
        current_prices_r = current_prices_r.reindex(names).fillna(prices_local)
        forecasted_prices_r = forecasted_prices_r.reindex(names).fillna(fcast_local)

    money_per_stock = (weights_series * investment / 1_000).round(2)
    # Share count is only meaningful once price and money share a currency.
    shares = (money_per_stock * 1_000 / current_prices_r[names]).round(0)

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

    # expected_returns from step 2 are PER-PERIOD; compound-annualise for display only.
    expected_annual = annualize_period_return(expected_returns, periods_per_year)

    output = pd.DataFrame({
        # Index-aligned, not a bare list: the catalogue may be ISINs, and a
        # positional list would silently mislabel every row if any reindex crept in.
        'Symbol':                                       pd.Series({n: symbols.get(n, n) for n in names}),
        'Portfolio Weight':                             weights_series.round(4),
        'Expected Annual Return':                       expected_annual[names].round(4),
        f'Current Price ({ccy})':                       current_prices_r[names].round(4),
        f'Forecasted Price ({ccy}, {last_future_date})': forecasted_prices_r[names].round(4),
        f'Investment ({ccy} k)':                        money_per_stock,
        'Shares':                                       shares,
    }).sort_values('Portfolio Weight', ascending=False)

    # Per-period mu compounded over the forecast horizon (equals the old
    # (1+annual)^(periods_to_forecast/ppy) expression, now in per-period units).
    portfolio_forecasted = (
        money_per_stock
        * (1 + expected_returns[names]) ** periods_to_forecast
    ).sum().round(2)

    portfolio_period_return = (weights_series * expected_returns[names]).sum()
    portfolio_row = pd.DataFrame({
        'Symbol':                                       ['-'],
        'Portfolio Weight':                             [1],
        'Expected Annual Return':                       [round(annualize_period_return(portfolio_period_return, periods_per_year), 4)],
        f'Current Price ({ccy})':                       [money_per_stock.sum()],
        f'Forecasted Price ({ccy}, {last_future_date})': [portfolio_forecasted],
        f'Investment ({ccy} k)':                        [money_per_stock.sum()],
        # NaN, not '-': a sentinel string would make the whole column object dtype
        # and every share count parse back from CSV as text.
        'Shares':                                       [float('nan')],
    }, index=['PORTFOLIO INDEX'])

    output = pd.concat([output, portfolio_row])

    output_path = PATHS['04_report']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path)

    print(f"\nAllocation saved to: {output_path}")
    print(f"\n{'-' * 70}")
    print(output.to_string())
    print(f"{'-' * 70}")
    print(f"\nTotal invested: {ccy} {money_per_stock.sum() / 1_000:.2f}M "
          f"across {len(weights_series)} stocks")


if __name__ == '__main__':
    main()
