"""
Step 3 - Portfolio Allocation

Dispatches on cfg['allocation_method']. Eight methods in two families:

  FORECAST-BASED -- consume the transformer's mu from step 2:
    "parametric_michaud" (default), "msr", "equal_weight_topn"
  MODEL-FREE -- ignore mu entirely; they were backtest benchmarks first:
    "equal_weight_all", "gmv", "inverse_vol", "momentum", "random"

See docs/PARAMETERS.md for what each does, what it reads, and how the production
form differs from the backtested one.

The allocation_top_n pre-filter is applied inside allocate(), not here, and only to
the forecast-based methods: it ranks on the model's forecast (mu/sigma by default, or
raw mu when allocation_ranking is 'return'), so applying it to a model-free method
would make that method quietly model-dependent.

Reads  (data/): 01_returns.csv (T, and the panel itself for momentum),
                02_expected_returns.csv, 02_covmat.csv
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
from allocation import allocate, MODEL_FREE_METHODS


def main():
    cfg = load_config()

    print("\n=== Step 3: Portfolio Allocation ===")

    returns  = pd.read_csv(PATHS['02_expected_returns'], index_col=0).iloc[:, 0]
    covmat   = pd.read_csv(PATHS['02_covmat'], index_col=0)
    # The panel itself, not just its length: momentum ranks on trailing returns rather
    # than on the forecast. Every other method ignores it.
    hist_rets = pd.read_csv(PATHS['01_returns'], index_col=0)
    n_periods = len(hist_rets)

    top_n  = cfg.get('allocation_top_n')
    metric = cfg.get('allocation_ranking', 'sharpe')

    # select_top_n is NOT applied here. It ranks on the model's forecast -- mu/sigma
    # under allocation_ranking 'sharpe', raw mu under 'return' -- and allocate() skips
    # it for the model-free methods: a momentum book picked from the transformer's 150
    # favourites is not momentum.
    method = cfg.get('allocation_method', 'parametric_michaud')
    model_free = method in MODEL_FREE_METHODS
    shown = len(returns) if model_free else min(top_n or len(returns), len(returns))
    print(f"Method: {method} | Universe: {shown} stock(s) "
          + ("(model-free: allocation_top_n not applied)" if model_free
             else f"(top_n={top_n}, ranking={metric})"))

    if method == 'random':
        print("\n  " + "!" * 68)
        print("  WARNING: allocation_method 'random' picks names UNIFORMLY AT RANDOM.")
        print("  It is a control for answering 'is this book better than luck?', not")
        print("  an investment strategy. Do not trade this book.")
        print("  " + "!" * 68 + "\n")

    weights = allocate(returns, covmat, cfg, n_periods, hist_rets=hist_rets)

    held    = weights[weights.abs() > 1e-9]
    optimal = held.sort_values().to_frame('Weights')

    print(f"\nFinal portfolio: {len(optimal)} stocks")
    print(optimal.sort_values('Weights', ascending=False).to_string())

    optimal.to_csv(PATHS['03_weights'])
    print(f"\nSaved: {PATHS['03_weights']}")


if __name__ == '__main__':
    main()
