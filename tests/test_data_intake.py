import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import data_intake as di


def test_load_tickers_hygiene(tmp_path):
    # one file with a BOM, whitespace, a nan, a blank line, and a duplicate
    f1 = tmp_path / "a.csv"
    f1.write_bytes("﻿AAPL\n MSFT \nnan\n\nAAPL\n".encode("utf-8"))
    f2 = tmp_path / "b.csv"
    f2.write_text("US1912161007\n")
    tickers = di.load_tickers(str(tmp_path / "*.csv"))
    assert set(tickers) == {"AAPL", "MSFT", "US1912161007"}   # BOM/space stripped, nan/blank/dup gone


def test_make_batches():
    assert di.make_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert di.make_batches([], 2) == []


def test_market_key():
    assert di.market_key("US1912161007") == "US"     # ISIN: 12 chars, country prefix
    assert di.market_key("KYG4672G1064") == "KY"
    assert di.market_key("DE0008404005") == "DE"
    assert di.market_key("ECOPETROL.CL") == "CL"      # ticker exchange suffix
    assert di.market_key("AAPL") == "OTHER"           # plain ticker -> catch-all bucket
    assert di.market_key(" us1912161007 ") == "US"    # trimmed + upper


def test_clean_batch_drops_missing_and_aligns_volume():
    idx = pd.date_range("2020-01-05", periods=6, freq="W")
    close = pd.DataFrame({
        "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "B": [20.0, np.nan, np.nan, 26.0, 28.0, 30.0],   # 2/6 ~33% missing -> dropped
    }, index=idx)
    volume = pd.DataFrame({
        "A": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "B": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    }, index=idx)
    c, v = di.clean_batch(close, volume, period_freq="W", missing_frac=0.15)
    assert list(c.columns) == ["A"]          # B dropped for missing
    assert list(v.columns) == ["A"]          # volume aligned to kept names
    assert isinstance(c.index[0], str)        # period-end string index


def test_download_all_concats_batches_with_stub():
    idx = ["2020-01", "2020-02"]
    def stub_download_fn(batch):
        # each batch returns its own 1-col close+volume
        name = batch[0]
        c = pd.DataFrame({name: [1.0, 2.0]}, index=idx)
        v = pd.DataFrame({name: [10.0, 20.0]}, index=idx)
        return c, v
    cfg = {"batch_size": 1, "download_workers": 2}
    close, volume = di.download_all(["A", "B"], cfg, download_fn=stub_download_fn)
    assert sorted(close.columns) == ["A", "B"]
    assert sorted(volume.columns) == ["A", "B"]
    assert len(close) == 2


def test_download_all_raises_if_all_fail():
    cfg = {"batch_size": 1, "download_workers": 2}
    with pytest.raises(RuntimeError):
        di.download_all(["A"], cfg, download_fn=lambda batch: None)


def _frames(cols_values):
    """Build aligned close/volume frames from {name: (close_list, vol_list)}."""
    idx = ["p1", "p2", "p3", "p4"]
    close = pd.DataFrame({k: v[0] for k, v in cols_values.items()}, index=idx)
    volume = pd.DataFrame({k: v[1] for k, v in cols_values.items()}, index=idx)
    return close, volume


def test_avg_dollar_volume():
    close, volume = _frames({"A": ([10, 10, 10, 10], [5, 5, 5, 5])})
    adv = di.avg_dollar_volume(close, volume, window=2)
    assert adv["A"] == pytest.approx(50.0)          # 10*5 averaged over last 2 periods


def test_liquidity_filter_relative_within_group():
    # group US: big median; one tiny name below 10% of median -> dropped. group CO: separate.
    close, volume = _frames({
        "US0000000001": ([10, 10, 10, 10], [100, 100, 100, 100]),   # adv 1000
        "US0000000002": ([10, 10, 10, 10], [100, 100, 100, 100]),   # adv 1000
        "US0000000003": ([10, 10, 10, 10], [100, 100, 100, 100]),   # adv 1000
        "US0000000004": ([10, 10, 10, 10], [100, 100, 100, 100]),   # adv 1000
        "US0000000005": ([10, 10, 10, 10], [1, 1, 1, 1]),           # adv 10 -> 1% of median -> drop
        "CO0000000001": ([5, 5, 5, 5], [40, 40, 40, 40]),           # adv 200
        "CO0000000002": ([5, 5, 5, 5], [40, 40, 40, 40]),
        "CO0000000003": ([5, 5, 5, 5], [40, 40, 40, 40]),
        "CO0000000004": ([5, 5, 5, 5], [40, 40, 40, 40]),
        "CO0000000005": ([5, 5, 5, 5], [40, 40, 40, 40]),
    })
    detail = di.liquidity_filter(close, volume, window=4, pct_of_median=0.10, min_group_size=5)
    assert detail.loc["US0000000005", "kept"] == False
    assert detail.loc["US0000000001", "kept"] == True
    assert detail.loc["CO0000000001", "kept"] == True       # CO group all equal -> all kept
    assert detail.loc["US0000000005", "market_group"] == "US"


def test_liquidity_filter_small_group_kept_and_flagged():
    close, volume = _frames({"DE0000000001": ([10, 10, 10, 10], [1, 1, 1, 1])})  # group size 1 < 5
    detail = di.liquidity_filter(close, volume, window=4, pct_of_median=0.10, min_group_size=5)
    assert detail.loc["DE0000000001", "kept"] == True
    assert detail.loc["DE0000000001", "flag"] == "small_group"


def test_grouping_health_reports_default_fraction_and_flags():
    detail = pd.DataFrame({
        "avg_dollar_volume": [1000, 1000, 1000, 1000, 1000, 10, 5],
        "market_group": ["US", "US", "US", "US", "US", "OTHER", "OTHER"],
        "kept": [True, True, True, True, True, True, True],
        "flag": ["", "", "", "", "", "small_group", "small_group"],
    }, index=[f"t{i}" for i in range(7)])
    health = di.grouping_health(detail)
    assert health["n_groups"] == 2
    assert health["other_fraction"] == pytest.approx(2 / 7)
    assert "OTHER" in health["flagged_groups"]      # OTHER has size 2 -> flagged
    assert health["groups"].loc["US", "count"] == 5
