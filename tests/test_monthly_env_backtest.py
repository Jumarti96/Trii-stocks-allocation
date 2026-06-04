import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import monthly_env_backtest as meb


def _weekly_cfg():
    return {
        "rf_rate": 0.11, "max_weight": 0.5, "min_weight": 0.05,
        "periods_per_year": 54, "interval": "1wk",
        "time_window": 54, "periods_to_forecast": 4,
    }


def test_build_monthly_cfg_sets_monthly_fields():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=6)
    assert cfg["interval"] == "1mo"
    assert cfg["periods_per_year"] == 12
    assert cfg["time_window"] == 12
    assert cfg["periods_to_forecast"] == 6
    assert cfg["period_freq"] == "M"
    assert cfg["future_freq"] == "MS"
    assert cfg["rf_period"] == pytest.approx((1 + 0.11) ** (1 / 12) - 1)
    # original weekly cfg is not mutated
    assert _weekly_cfg()["periods_per_year"] == 54


def test_select_all_returns_every_name():
    rets = pd.DataFrame(np.zeros((3, 4)), columns=["A", "B", "C", "D"])
    assert meb.select_all(None, rets, {}) == ["A", "B", "C", "D"]
