"""Tests for pipeline/config.py derived values.

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_pipeline_config.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from config import load_config


def test_rf_period_is_compound_per_period_equivalent():
    cfg = load_config()
    expected = (1 + cfg["rf_rate"]) ** (1 / cfg["periods_per_year"]) - 1
    assert abs(cfg["rf_period"] - expected) < 1e-15


def test_rf_period_below_annual_for_multi_period_year():
    cfg = load_config()
    # ppy > 1, positive rf -> per-period rate is strictly smaller than annual
    assert 0 < cfg["rf_period"] < cfg["rf_rate"]
