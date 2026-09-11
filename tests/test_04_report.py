"""Tests for pipeline/04_report.py currency handling.

Prices arrive in each exchange's own quote currency. The money split was always
correct (weights are dimensionless), but the price column mixed units, so it could
not be compared across rows or turned into a share count. These pin the conversion,
the share arithmetic, and the behaviour when a currency cannot be resolved.

Run: .venv/Scripts/python.exe -m pytest tests/test_04_report.py -v
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))


def _load_script():
    path = os.path.join(ROOT, "pipeline", "04_report.py")
    spec = importlib.util.spec_from_file_location("step04_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, *, with_currency=True, cur_for_eco="COP"):
    """Two holdings quoted in different currencies: NVDA in USD, ECO.CL in COP."""
    idx = pd.date_range("2024-01-07", periods=60, freq="W-SUN").astype(str)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.02, (60, 2)), columns=["NVDA", "ECO.CL"], index=idx)

    paths = {k: str(tmp_path / f"{k}.csv") for k in
             ("01_returns", "01_currency", "01_fx", "02_expected_returns", "03_weights")}
    paths["02_metadata"] = str(tmp_path / "02_meta.json")
    paths["04_report"] = str(tmp_path / "out.csv")

    rets.to_csv(paths["01_returns"])
    pd.DataFrame({"Weights": [0.5, 0.5]}, index=["NVDA", "ECO.CL"]).to_csv(paths["03_weights"])
    pd.DataFrame({"Expected Period Return": [0.001, 0.001]},
                 index=["NVDA", "ECO.CL"]).to_csv(paths["02_expected_returns"])
    with open(paths["02_metadata"], "w") as f:
        json.dump({"future_dates": ["2024-06"], "last_date": "2024-02-25",
                   "winsorization_lower": -0.1, "winsorization_upper": 0.1,
                   "current_prices": {"NVDA": 100.0, "ECO.CL": 4000.0},
                   "forecasted_prices": {"NVDA": 110.0, "ECO.CL": 4400.0}}, f)

    if with_currency:
        pd.DataFrame({"currency": ["USD", cur_for_eco], "unit_factor": [1.0, 1.0],
                      "symbol": ["NVDA", "ECOPETROL.CL"]},
                     index=["NVDA", "ECO.CL"]).to_csv(paths["01_currency"])
        # 1 USD = 4000 COP  ->  USD is 1.0 hub units, COP is 0.00025
        pd.DataFrame({"USD": [1.0] * 60, "COP": [0.00025] * 60},
                     index=idx).to_csv(paths["01_fx"])
    return paths


def _run(mod, monkeypatch, paths, **over):
    cfg = mod.load_config()
    cfg.update(investment=1_000_000, report_currency="COP",
               unknown_currency="exclude", periods_to_forecast=4)
    cfg.update(over)
    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "PATHS", paths)
    mod.main()
    return pd.read_csv(paths["04_report"], index_col=0)


def test_prices_are_converted_into_the_report_currency(tmp_path, monkeypatch):
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path))
    # NVDA quoted at 100 USD -> 400,000 COP; ECO.CL already COP, unchanged.
    assert out.loc["NVDA", "Current Price (COP)"] == pytest.approx(400_000.0)
    assert out.loc["ECO.CL", "Current Price (COP)"] == pytest.approx(4000.0)


def test_report_currency_is_configurable(tmp_path, monkeypatch):
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path), report_currency="USD")
    assert "Current Price (USD)" in out.columns
    assert out.loc["NVDA", "Current Price (USD)"] == pytest.approx(100.0)
    assert out.loc["ECO.CL", "Current Price (USD)"] == pytest.approx(1.0)   # 4000 COP


def test_share_counts_use_the_converted_price(tmp_path, monkeypatch):
    # The reason conversion matters: 500,000 COP of NVDA at 400,000 COP/share is 1
    # share. Dividing by the raw USD price of 100 would have said 5,000.
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path))
    assert out.loc["NVDA", "Shares"] == pytest.approx(1.0)
    assert out.loc["ECO.CL", "Shares"] == pytest.approx(125.0)


def test_money_split_is_unchanged_by_currency(tmp_path, monkeypatch):
    # Weights are dimensionless, so this was always correct and must stay so.
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path))
    assert out.loc["NVDA", "Investment (COP k)"] == pytest.approx(500.0)
    assert out.loc["ECO.CL", "Investment (COP k)"] == pytest.approx(500.0)


def test_symbol_column_maps_identifiers_to_tradeable_tickers(tmp_path, monkeypatch):
    # An ISIN catalogue stays ISIN-labelled without this, and nobody can trade
    # 'US67066G1040'.
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path))
    assert out.loc["ECO.CL", "Symbol"] == "ECOPETROL.CL"


def test_unresolvable_currency_does_not_break_the_run(tmp_path, monkeypatch):
    # Robustness: a holding whose currency cannot be resolved must still appear,
    # shown at its unconverted local price, rather than aborting the report.
    mod = _load_script()
    paths = _setup(tmp_path, cur_for_eco="")
    out = _run(mod, monkeypatch, paths)
    assert len(out) == 3                       # 2 holdings + PORTFOLIO INDEX
    assert out.loc["ECO.CL", "Current Price (COP)"] == pytest.approx(4000.0)
    assert out.loc["NVDA", "Current Price (COP)"] == pytest.approx(400_000.0)


def test_forecast_and_current_price_share_one_fx_rate(tmp_path, monkeypatch):
    # Expected returns are USD returns applied to a native price, so the forecast is
    # only meaningful if BOTH legs are converted at the same rate -- the rate then
    # cancels and the column reads "today's price grown by the forecast USD return,
    # priced in report_currency". Converting them at different rates would fold in an
    # FX forecast the model never made, and the error would be invisible in the
    # output. Pin the ratio rather than the level so this survives fixture changes.
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path))
    fcast_col = [c for c in out.columns if c.startswith("Forecasted Price")][0]
    for name in ("NVDA", "ECO.CL"):
        growth = out.loc[name, fcast_col] / out.loc[name, "Current Price (COP)"]
        assert growth == pytest.approx(1.10)     # 110/100 and 4400/4000 alike


def test_runs_without_currency_files(tmp_path, monkeypatch):
    # A data/ directory predating the currency work must still produce a report.
    mod = _load_script()
    out = _run(mod, monkeypatch, _setup(tmp_path, with_currency=False))
    assert len(out) == 3
    assert out.loc["NVDA", "Current Price (COP)"] == pytest.approx(100.0)   # local
