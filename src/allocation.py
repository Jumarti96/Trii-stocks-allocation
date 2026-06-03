"""
Portfolio allocation policies for step 4.

Two methods behind cfg['allocation_method']:
  - "msr"                : Sharpe-max + batch-elimination (the legacy method).
  - "parametric_michaud" : resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
                           raw msr per draw, average the weight vectors, one min_weight floor.

Reuses risk_kit.msr_tuned for the per-portfolio optimisation. Pure functions (no I/O);
pipeline/04_allocate.py does the file reading/writing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import risk_kit as rk


def msr_eliminate(returns, covmat, cfg):
    """Sharpe-maximising weights with the batch-elimination loop (the legacy step-4 method).

    Iteratively drops names whose cumulative weight (sorted ascending) is below min_weight and
    re-optimises, until all survivors pass or <=2 remain. Returns a Series over returns.index;
    eliminated names get 0.0.
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    names = list(returns.index)

    w0 = rk.msr_tuned(
        riskfree_rate=rf, returns=returns, covmat=covmat,
        max_weight=max_w, periods_per_year=ppy, debug=False,
    )
    optimal = pd.DataFrame(w0, index=returns.index, columns=["Weights"]).sort_values("Weights")

    while optimal["Weights"].sum() >= 0.9999:
        cum_weights = optimal["Weights"].cumsum()
        failing_mask = cum_weights < min_w
        if not failing_mask.any():
            break
        optimal = optimal[~failing_mask]
        if len(optimal) <= 2:
            break
        w = rk.msr_tuned(
            riskfree_rate=rf, returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
        )
        optimal = pd.DataFrame(w, index=optimal.index, columns=["Weights"]).sort_values("Weights")

    weights = pd.Series(0.0, index=names)
    weights[optimal.index] = optimal["Weights"]
    return weights
