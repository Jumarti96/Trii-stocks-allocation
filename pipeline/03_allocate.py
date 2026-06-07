"""
Step 3 - Portfolio Allocation

Dispatches on cfg['allocation_method']:
  - "parametric_michaud" (default): resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
    raw msr per draw, average the weights, one min_weight floor (src/allocation.resampled_michaud).
  - "msr": the legacy Sharpe-max + batch-elimination loop (src/allocation.msr_eliminate).

Before optimising, the full-universe mu and covmat from step 2 are pre-filtered to the
top allocation_top_n stocks (ranked by allocation_ranking) to keep optimizer compute tractable
for large universes. Set allocation_top_n: null to disable the cap.

Reads  (data/): 01_returns.csv (for T), 02_expected_returns.csv, 02_covmat.csv
Outputs (data/):
    03_weights.csv - optimal weight per held stock
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from config import load_config, PATHS
from allocation import allocate, select_top_n


def main():
    cfg = load_config()

    print("\n=== Step 3: Portfolio Allocation ===")

    returns  = pd.read_csv(PATHS['02_expected_returns'], index_col=0).iloc[:, 0]
    covmat   = pd.read_csv(PATHS['02_covmat'], index_col=0)
    n_periods = len(pd.read_csv(PATHS['01_returns'], index_col=0))

    top_n  = cfg.get('allocation_top_n')
    metric = cfg.get('allocation_ranking', 'sharpe')
    returns, covmat = select_top_n(returns, covmat, top_n, metric)

    method = cfg.get('allocation_method', 'parametric_michaud')
    print(f"Method: {method} | Universe: {len(returns)} stock(s) "
          f"(top_n={top_n}, ranking={metric})")

    weights = allocate(returns, covmat, cfg, n_periods)

    held    = weights[weights.abs() > 1e-9]
    optimal = held.sort_values().to_frame('Weights')

    print(f"\nFinal portfolio: {len(optimal)} stocks")
    print(optimal.sort_values('Weights', ascending=False).to_string())

    optimal.to_csv(PATHS['03_weights'])
    print(f"\nSaved: {PATHS['03_weights']}")


if __name__ == '__main__':
    main()
