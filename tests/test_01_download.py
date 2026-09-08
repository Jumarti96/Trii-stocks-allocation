"""Smoke test for pipeline/01_download.py, with every network call stubbed.

Worth having despite the step being mostly I/O: it is the ONLY step that touches the
network, it costs ~45 minutes and ~3,000 .info calls on the full catalogue, and a
mistake in the currency wiring is not visible until the whole run is finished. The
assertions here are the two properties the rest of the pipeline depends on -- returns
are denominated in USD, and every artifact carries the same name set.
"""
import os
import sys
import importlib.util

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

IDX = [f"p{i}" for i in range(12)]


def _load_script():
    path = os.path.join(ROOT, "pipeline", "01_download.py")
    spec = importlib.util.spec_from_file_location("step01_download", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_step1(mod, tmp_path, monkeypatch, cfg_overrides=None):
    """Wire the script to fixtures: 3 stocks, one per currency situation."""
    # NVDA (USD), ECO.CL (COP, flat native price while the peso halves), MYSTERY
    # (currency unresolvable).
    close = pd.DataFrame({"NVDA": [100.0] * 12,
                          "ECO.CL": [4000.0] * 12,
                          "MYSTERY": [50.0] * 12}, index=IDX)
    volume = pd.DataFrame({c: [1_000.0] * 12 for c in close.columns}, index=IDX)

    listings = pd.DataFrame(
        {"currency": ["USD", "COP", None],
         "unit_factor": [1.0, 1.0, 1.0],
         "symbol": ["NVDA", "ECO.CL", "MYSTERY"],
         "name": ["NVIDIA", "Ecopetrol", ""],
         "source": ["lookup", "lookup", "inferred"],
         "sector": ["Technology", "Energy", None],
         "industry": ["Semiconductors", "Oil & Gas", None],
         "market_cap": [4.2e12, 5.0e10, None],
         "exchange": ["NMS", "BVC", None],
         "quote_type": ["EQUITY", "EQUITY", None]},
        index=["NVDA", "ECO.CL", "MYSTERY"])

    # The peso halves against the dollar over the window.
    cop = [0.00025] * 6 + [0.000125] * 6
    fx = pd.DataFrame({"USD": [1.0] * 12, "COP": cop}, index=IDX)

    cfg = mod.load_config()
    cfg["unknown_currency"] = "exclude"
    cfg.update(cfg_overrides or {})

    paths = {"01_prices": str(tmp_path / "01_prices.csv"),
             "01_returns": str(tmp_path / "01_returns.csv"),
             "01_volume": str(tmp_path / "01_volume.csv"),
             "01_currency": str(tmp_path / "01_currency.csv"),
             "01_fx": str(tmp_path / "01_fx.csv")}

    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "PATHS", paths)
    monkeypatch.setattr(mod, "load_tickers", lambda glob: list(close.columns))
    monkeypatch.setattr(mod, "download_all", lambda t, c: (close, volume))
    monkeypatch.setattr(mod, "resolve_listings", lambda ids, **kw: listings.reindex(ids))
    monkeypatch.setattr(mod, "fetch_fx_rates", lambda curs, index, hub="USD": fx)
    return paths


def _read(paths, key):
    return pd.read_csv(paths[key], index_col=0)


def test_returns_are_denominated_in_usd_not_native_prices(tmp_path, monkeypatch):
    # ECO.CL's native price never moves, so a native pct_change() would call it a
    # flat holding. A dollar-based investor holding it through a 50% peso
    # depreciation lost half their money, and that is what must land in 01_returns.
    mod = _load_script()
    paths = _stub_step1(mod, tmp_path, monkeypatch)
    mod.main()

    rets = _read(paths, "01_returns")
    assert rets["NVDA"].abs().max() == pytest.approx(0.0)     # USD name, flat, stays flat
    assert rets["ECO.CL"].loc["p6"] == pytest.approx(-0.5)    # the peso halving
    assert rets["ECO.CL"].drop("p6").abs().max() == pytest.approx(0.0)


def test_prices_stay_native_while_a_usd_panel_is_written_alongside(tmp_path, monkeypatch):
    # 01_prices.csv must stay native: select_universe and the step-4 report each
    # apply their own conversion from it, so pre-converting would convert twice.
    mod = _load_script()
    paths = _stub_step1(mod, tmp_path, monkeypatch)
    mod.main()

    native = _read(paths, "01_prices")
    usd = pd.read_csv(tmp_path / "01_prices_usd.csv", index_col=0)
    assert native["ECO.CL"].iloc[0] == pytest.approx(4000.0)
    assert usd["ECO.CL"].iloc[0] == pytest.approx(1.0)        # 4000 COP at 0.00025
    assert usd["ECO.CL"].iloc[-1] == pytest.approx(0.5)


def test_an_unconvertible_stock_is_dropped_from_every_artifact(tmp_path, monkeypatch):
    # A name present in prices but absent from returns would silently misalign any
    # caller that zips the two, so the drop has to be applied everywhere at once.
    mod = _load_script()
    paths = _stub_step1(mod, tmp_path, monkeypatch)
    mod.main()

    names = set(_read(paths, "01_prices").columns)
    assert "MYSTERY" not in names
    assert names == set(_read(paths, "01_returns").columns)
    assert names == set(_read(paths, "01_volume").columns)
    assert names == set(pd.read_csv(tmp_path / "01_liquidity.csv", index_col=0).index)


def test_assume_target_keeps_the_unresolved_name(tmp_path, monkeypatch):
    mod = _load_script()
    paths = _stub_step1(mod, tmp_path, monkeypatch,
                        cfg_overrides={"unknown_currency": "assume_target"})
    mod.main()
    assert "MYSTERY" in _read(paths, "01_returns").columns


def test_currency_file_carries_the_classification_columns(tmp_path, monkeypatch):
    # The universe profile reads these straight off 01_currency.csv; if step 1 stops
    # persisting them the only way to get them back is another 3,000-call pass.
    mod = _load_script()
    paths = _stub_step1(mod, tmp_path, monkeypatch)
    mod.main()

    cur = _read(paths, "01_currency")
    for col in ("sector", "industry", "market_cap", "exchange", "quote_type"):
        assert col in cur.columns
    assert cur.loc["NVDA", "sector"] == "Technology"
