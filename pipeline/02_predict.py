"""
Step 2 - Transformer Prediction and Covariance Estimation

Trains the Transformer on the full cross-section - winners and losers, trends and
reversals - which avoids the optimistic bias that arises when the model only ever sees
stocks hand-picked to be in an uptrend. The technical filter (step 3) is applied later,
purely as an allocation gate in step 4.

When universe_topn is set, the cross-section is first narrowed by the universe screen
(src/data_intake.select_universe). This does NOT reintroduce that bias: the screen is
return-neutral by construction - it ranks on liquidity, price and optionally size, never
on past performance - so it cannot preferentially retain stocks that went up. It is a
*style* tilt toward large caps, which is a different thing and worth stating in results.
The screen exists because stocks are the model's feature axis: widening the universe adds
parameters and zero training samples, so a 3.9k catalogue is bounded by
parameters-per-sample long before it is bounded by memory (see transformer_model
.capacity_report). universe_topn: null disables it entirely.

Predictions are averaged across N runs (to damp random-initialisation noise), winsorised at
the 1st-99th percentile of historical returns, annualised with exponential-decay weighting,
and paired with a Ledoit-Wolf covariance matrix estimated over the full universe.

Reads  (data/): 01_prices.csv, 01_returns.csv
Outputs (data/):
    02_expected_returns.csv - annualised expected return per stock (full universe)
    02_covmat.csv           - Ledoit-Wolf covariance matrix (full universe)
    02_predictions.csv      - raw period-by-period predicted returns (full universe)
    02_metadata.json        - current prices, forecasted prices, future dates,
                              winsorisation bounds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS
from data_intake import select_universe
from transformer_model import (train_and_predict, weighted_mean_return, describe_device,
                               capacity_report)


def main():
    cfg = load_config()

    print("\n=== Step 2: Training Transformer Neural Network (full universe) ===")
    print(describe_device())

    prices = pd.read_csv(PATHS['01_prices'],  index_col=0)
    rets   = pd.read_csv(PATHS['01_returns'], index_col=0)

    periods_to_forecast = cfg['periods_to_forecast']

    # Universe screen. A null universe_topn is a complete no-op, so the default
    # path is byte-identical to the pre-screen pipeline.
    if cfg.get('universe_topn'):
        volume = pd.read_csv(PATHS['01_volume'], index_col=0)
        fx      = pd.read_csv(PATHS['01_fx'], index_col=0)
        cur_map = pd.read_csv(PATHS['01_currency'], index_col=0)['currency'].to_dict()
        universe = select_universe(
            prices, volume, cfg['universe_topn'],
            strata=cfg.get('universe_strata'),
            price_floor=cfg.get('universe_price_floor', 0.0),
            fx=fx, cur_map=cur_map,
        )
        print(f"Universe screen: {len(universe)} of {prices.shape[1]} stocks "
              f"(topn={cfg['universe_topn']}, strata={cfg.get('universe_strata')})")
        prices = prices[universe]
        rets   = rets[universe]

    # Capacity check. Widening the universe adds parameters and zero training
    # samples, so this ratio - not VRAM - is what bounds universe size.
    cap = capacity_report(rets.shape[1], rets.shape[0], cfg)
    print(f"Capacity: {cap['message']}")
    if cap['verdict'] == 'error':
        raise ValueError(
            f"universe too large to train: {cap['params_per_sample']:,.0f} "
            f"parameters per training sample. {cap['message']}")

    # Train on the (screened) universe and forecast every stock in it
    arch = cfg.get('transformer_arch', 'current')
    preds_df = train_and_predict(rets, cfg, arch=arch)
    preds_df = preds_df.iloc[:periods_to_forecast]   # no-op for current; e.g. 24->4 for B

    # Build future date range and attach it to the predictions
    last_date    = pd.to_datetime(rets.index).max()
    future_dates = pd.date_range(
        start=last_date + cfg['date_offset'],
        periods=periods_to_forecast,
        freq=cfg['future_freq']
    ).to_period(cfg['period_freq'])
    preds_df.index = future_dates

    # Winsorisation bounds (recomputed for metadata; train_and_predict already clipped)
    lower_w = float(np.percentile(rets.values, cfg['winsorization_lower_pct']))
    upper_w = float(np.percentile(rets.values, cfg['winsorization_upper_pct']))

    # Per-period expected returns (exp-decay weighted). Annualisation is a DISPLAY
    # concern handled in step 5; the optimiser consumes these per-period values directly.
    expected_returns = weighted_mean_return(preds_df)

    # Ledoit-Wolf covariance over the full universe
    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_,
        index=rets.columns, columns=rets.columns
    )

    # Current and forecasted prices (full universe)
    current_prices    = prices.iloc[-1]
    forecasted_prices = current_prices * (preds_df + 1).prod()

    # Write outputs
    expected_returns.to_csv(PATHS['02_expected_returns'], header=['Expected Period Return'])
    covmat.to_csv(PATHS['02_covmat'])

    preds_out = preds_df.copy()
    preds_out.index = preds_out.index.astype('str')
    preds_out.to_csv(PATHS['02_predictions'])

    metadata = {
        'future_dates':        [str(d) for d in future_dates],
        'last_date':           str(last_date.date()),
        'winsorization_lower': lower_w,
        'winsorization_upper': upper_w,
        'current_prices':      current_prices.to_dict(),
        'forecasted_prices':   forecasted_prices.to_dict(),
    }
    with open(PATHS['02_metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Trained on {rets.shape[1]} stocks (full universe).")
    print(f"Saved: {PATHS['02_expected_returns']}")
    print(f"       {PATHS['02_covmat']}")
    print(f"       {PATHS['02_predictions']}")
    print(f"       {PATHS['02_metadata']}")


if __name__ == '__main__':
    main()
