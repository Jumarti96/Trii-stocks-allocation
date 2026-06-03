"""
Step 4 - Portfolio Allocation

Dispatches on cfg['allocation_method']:
  - "parametric_michaud" (default): resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
    raw msr per draw, average the weights, one min_weight floor (src/allocation.resampled_michaud).
  - "msr": the legacy Sharpe-max + batch-elimination loop (src/allocation.msr_eliminate).

The technical filter (step 3) is the allocation gate: full-universe predictions from step 2 are
restricted to the selected names before allocating.

Reads  (data/): 01_returns.csv (for T), 02_expected_returns.csv, 02_covmat.csv, 03_selected_returns.csv
Outputs (data/):
    04_weights.csv - optimal weight per held stock
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from config import load_config, PATHS
from allocation import allocate


def main():
    cfg = load_config()

    print("\n=== Step 4: Portfolio Allocation ===")

    expected_returns = pd.read_csv(PATHS['02_expected_returns'], index_col=0).iloc[:, 0]
    covmat = pd.read_csv(PATHS['02_covmat'], index_col=0)
    n_periods = len(pd.read_csv(PATHS['01_returns'], index_col=0))

    # Allocation gate: restrict the full-universe predictions to the step-3 selection.
    selected = pd.read_csv(PATHS['03_selected_returns'], index_col=0, nrows=0).columns
    selected = [s for s in selected if s in expected_returns.index and s in covmat.index]

    method = cfg.get('allocation_method', 'parametric_michaud')
    print(f"Method: {method} | Allocation universe: {len(selected)} selected stock(s).")

    returns = expected_returns[selected]
    covmat = covmat.loc[selected, selected]

    weights = allocate(returns, covmat, cfg, n_periods)

    held = weights[weights.abs() > 1e-9]
    optimal = held.sort_values().to_frame('Weights')

    print(f"\nFinal portfolio: {len(optimal)} stocks")
    print(optimal.sort_values('Weights', ascending=False).to_string())

    optimal.to_csv(PATHS['04_weights'])
    print(f"\nSaved: {PATHS['04_weights']}")


if __name__ == '__main__':
    main()
