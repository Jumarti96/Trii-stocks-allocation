"""
Step 2 - Transformer Prediction and Covariance Estimation

Trains the Transformer on the FULL stock universe (every ticker from step 1), not a
pre-filtered subset. Training on the full cross-section - winners and losers, trends and
reversals - avoids the optimistic bias that arises when the model only ever sees stocks
hand-picked to be in an uptrend. The technical filter (step 3) is applied later, purely as
an allocation gate in step 4.

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
from transformer_model import train_and_predict, weighted_mean_return, describe_device


def main():
    cfg = load_config()

    print("\n=== Step 2: Training Transformer Neural Network (full universe) ===")
    print(describe_device())

    prices = pd.read_csv(PATHS['01_prices'],  index_col=0)
    rets   = pd.read_csv(PATHS['01_returns'], index_col=0)

    periods_to_forecast = cfg['periods_to_forecast']

    # Train on the full universe and forecast every stock
    preds_df = train_and_predict(rets, cfg)

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
