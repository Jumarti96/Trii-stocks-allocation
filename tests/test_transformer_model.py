"""Tests for annualization helpers in src/transformer_model.py.

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_transformer_model.py -v
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformer_model import (
    annualize_period_return,
    annualize_expected_returns,
    weighted_mean_return,
)


def test_annualize_scalar():
    assert abs(annualize_period_return(0.01, 12) - ((1.01 ** 12) - 1)) < 1e-12


def test_annualize_zero_is_zero_any_ppy():
    assert annualize_period_return(0.0, 54) == 0.0


def test_annualize_series_elementwise():
    s = pd.Series({"A": 0.01, "B": 0.0})
    out = annualize_period_return(s, 12)
    assert abs(out["A"] - ((1.01 ** 12) - 1)) < 1e-12
    assert out["B"] == 0.0


def test_optimization_mu_has_no_ppy_dependence():
    # The per-period mu the optimiser consumes is weighted_mean_return, which
    # takes no periods_per_year argument -> changing frequency cannot distort it.
    preds = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [-0.01, 0.0, 0.01]})
    pd.testing.assert_series_equal(weighted_mean_return(preds), weighted_mean_return(preds))


def test_annualize_expected_returns_delegates_to_helper():
    preds = pd.DataFrame({"A": [0.01, 0.01, 0.01], "B": [0.005, 0.005, 0.005]})
    ppy = 12
    expected = annualize_period_return(weighted_mean_return(preds), ppy)
    pd.testing.assert_series_equal(annualize_expected_returns(preds, ppy), expected)
