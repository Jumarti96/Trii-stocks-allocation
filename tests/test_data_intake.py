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



def test_download_all_dedups_duplicate_output_columns():
    idx = ["2020-01", "2020-02"]
    def stub_download_fn(batch):
        # both batches resolve to the same output symbol 'DUP' (ISIN alias / dual listing)
        c = pd.DataFrame({"DUP": [1.0, 2.0]}, index=idx)
        v = pd.DataFrame({"DUP": [10.0, 20.0]}, index=idx)
        return c, v
    cfg = {"batch_size": 1, "download_workers": 2}
    close, volume = di.download_all(["A", "B"], cfg, download_fn=stub_download_fn)
    assert list(close.columns) == ["DUP"]          # duplicate collapsed, no crash
    assert list(volume.columns) == ["DUP"]


def test_active_fraction_counts_traded_periods():
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),     # traded every period -> 1.0
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),     # 2 of 4 -> 0.5
        "DEAD": ([10, 10, 10, 10], [0, 0, 0, 0]),     # never -> 0.0
    })
    af = di.active_fraction(volume, window=4)
    assert af["FULL"] == pytest.approx(1.0)
    assert af["HALF"] == pytest.approx(0.5)
    assert af["DEAD"] == pytest.approx(0.0)


def test_active_fraction_window_longer_than_data():
    # window > number of rows -> use all available rows, still a valid fraction (newly listed name)
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),
    })
    af = di.active_fraction(volume, window=100)
    assert af["FULL"] == pytest.approx(1.0)
    assert af["HALF"] == pytest.approx(0.5)


def test_active_fraction_all_nan_volume_is_inactive():
    # yfinance can return all-NaN Volume for some instruments -> active_fraction 0 -> excluded
    close, volume = _frames({"NOVOL": ([10, 10, 10, 10], [np.nan, np.nan, np.nan, np.nan])})
    af = di.active_fraction(volume, window=4)
    assert af["NOVOL"] == pytest.approx(0.0)
    detail = di.activity_filter(close, volume, window=4, min_active_fraction=0.90)
    assert detail.loc["NOVOL", "kept"] == False
    assert di.activity_health(detail)["zero_volume_fraction"] == pytest.approx(1.0)


def test_activity_filter_keeps_active_drops_inactive():
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),     # 1.0 -> kept
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),     # 0.5 -> dropped at 0.9
        "DEAD": ([10, 10, 10, 10], [0, 0, 0, 0]),     # 0.0 -> dropped
    })
    detail = di.activity_filter(close, volume, window=4, min_active_fraction=0.90)
    assert detail.loc["FULL", "kept"] == True
    assert detail.loc["HALF", "kept"] == False
    assert detail.loc["DEAD", "kept"] == False
    assert list(detail.columns) == ["avg_dollar_volume", "active_fraction", "kept"]


def test_activity_health_counts_and_zero_volume_fraction():
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),     # kept
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),     # dropped (not zero-volume though)
        "DEAD": ([10, 10, 10, 10], [0, 0, 0, 0]),     # dropped, zero-volume
    })
    detail = di.activity_filter(close, volume, window=4, min_active_fraction=0.90)
    health = di.activity_health(detail)
    assert health["n_total"] == 3
    assert health["n_kept"] == 1
    assert health["n_excluded"] == 2
    assert health["zero_volume_fraction"] == pytest.approx(1 / 3)   # only DEAD has af==0
