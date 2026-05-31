"""
Step 4 - Sharpe-Ratio Maximising Allocation

Finds the portfolio weights that maximise the Sharpe ratio subject to per-stock
min/max weight constraints. Iteratively drops stocks whose cumulative weight
(sorted ascending) falls below MIN_WEIGHT until all remaining stocks satisfy
the constraint.

The technical filter is applied here as the allocation gate: the full-universe
predictions from step 2 are restricted to the names selected by step 3 before optimising.

Reads  (data/): 02_expected_returns.csv, 02_covmat.csv, 03_selected_returns.csv
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

    expected_returns = pd.read_csv(PATHS['02_expected_returns'], index_col=0).iloc[:, 0]
    covmat = pd.read_csv(PATHS['02_covmat'], index_col=0)

    rf_period        = cfg['rf_period']
    max_weight       = cfg['max_weight']
    min_weight       = cfg['min_weight']
    periods_per_year = cfg['periods_per_year']

    # Allocation gate: restrict the full-universe predictions to the names selected by
    # step 3 (intersect defensively in case a selected name is missing upstream).
    selected = pd.read_csv(PATHS['03_selected_returns'], index_col=0, nrows=0).columns
    selected = [s for s in selected if s in expected_returns.index and s in covmat.index]
    print(f"Allocation universe: {len(selected)} selected stock(s).")

    returns = expected_returns[selected]
    covmat  = covmat.loc[selected, selected]

    initial_weights = rk.msr_tuned(
        riskfree_rate=rf_period, returns=returns, covmat=covmat,
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
            riskfree_rate=rf_period,
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
