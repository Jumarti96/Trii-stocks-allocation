"""
Step 4 - Sharpe-Ratio Maximising Allocation

Finds the portfolio weights that maximise the Sharpe ratio subject to per-stock
min/max weight constraints. Iteratively drops stocks whose cumulative weight
(sorted ascending) falls below MIN_WEIGHT until all remaining stocks satisfy
the constraint.

Reads  (data/): 03_expected_returns.csv, 03_covmat.csv
Outputs (data/):
    04_weights.csv - optimal weight per selected stock
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import risk_kit as rk

from config import load_config, PATHS


def main():
    cfg = load_config()

    print("\n=== Step 4: Sharpe Ratio Maximising Allocation ===")

    expected_returns = pd.read_csv(PATHS['03_expected_returns'], index_col=0).iloc[:, 0]
    covmat = pd.read_csv(PATHS['03_covmat'], index_col=0)

    rf_rate          = cfg['rf_rate']
    max_weight       = cfg['max_weight']
    min_weight       = cfg['min_weight']
    periods_per_year = cfg['periods_per_year']

    # Align returns to the covariance matrix index (both come from the same selected set)
    returns = expected_returns[covmat.index]

    initial_weights = rk.msr_tuned(
        riskfree_rate=rf_rate, returns=returns, covmat=covmat,
        max_weight=max_weight, periods_per_year=periods_per_year, debug=False
    )
    optimal = (
        pd.DataFrame(initial_weights, index=returns.index, columns=['Weights'])
        .sort_values('Weights')
    )

    # Batch elimination: drop all stocks whose cumulative weight (ascending) < min_weight
    iteration = 0
    while optimal['Weights'].sum() >= 0.9999:
        iteration   += 1
        cum_weights  = optimal['Weights'].cumsum()
        failing_mask = cum_weights < min_weight

        if not failing_mask.any():
            break

        n_dropped = failing_mask.sum()
        print(f"  [pass {iteration}] Dropping {n_dropped} stock(s) "
              f"(cumulative weight < {min_weight:.0%}): "
              f"{list(optimal[failing_mask].index)}")

        optimal = optimal[~failing_mask]
        if len(optimal) <= 2:
            break

        w = rk.msr_tuned(
            riskfree_rate=rf_rate,
            returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_weight,
            periods_per_year=periods_per_year,
            debug=False
        )
        optimal = (
            pd.DataFrame(w, index=optimal.index, columns=['Weights'])
            .sort_values('Weights')
        )

    print(f"\nFinal portfolio: {len(optimal)} stocks")
    print(optimal.sort_values('Weights', ascending=False).to_string())

    optimal.to_csv(PATHS['04_weights'])
    print(f"\nSaved: {PATHS['04_weights']}")


if __name__ == '__main__':
    main()
