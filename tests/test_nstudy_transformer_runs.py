import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import nstudy_transformer_runs as ns


def _identity_winsorize(preds_df, rets):
    return preds_df


def _mean_period_mu(preds_df):
    return preds_df.mean(axis=0)


def test_prefix_forecast_averages_first_n_runs():
    # 3 runs, 2 forecast periods, 2 stocks
    runs = np.array([
        [[1.0, 2.0], [1.0, 2.0]],   # run 0
        [[3.0, 4.0], [3.0, 4.0]],   # run 1
        [[9.0, 9.0], [9.0, 9.0]],   # run 2 (excluded when n=2)
    ])
    rets = pd.DataFrame(np.zeros((5, 2)), columns=["A", "B"])
    mu = ns.prefix_forecast(runs, 2, rets, _identity_winsorize, _mean_period_mu)
    # first 2 runs averaged: A=(1+3)/2=2, B=(2+4)/2=3; period-mean leaves them unchanged
    assert mu["A"] == pytest.approx(2.0)
    assert mu["B"] == pytest.approx(3.0)
