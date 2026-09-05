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


def test_avg_dollar_volume_as_of_uses_window_ending_at_that_date():
    # p1..p4 with a volume spike at the end. Selecting "as of p2" must not see it --
    # this is the look-ahead guard for backtests (2019 universe, 2026 liquidity).
    close, volume = _frames({"A": ([10, 10, 10, 10], [1, 1, 100, 100])})
    assert di.avg_dollar_volume(close, volume, window=2)["A"] == pytest.approx(1000.0)
    assert di.avg_dollar_volume(close, volume, window=2, as_of="p2")["A"] == pytest.approx(10.0)


def test_avg_dollar_volume_as_of_none_matches_tail_behaviour():
    # Production path must be unchanged: as_of=None is the pre-existing semantics.
    close, volume = _frames({"A": ([10, 10, 10, 10], [1, 2, 3, 4])})
    assert (di.avg_dollar_volume(close, volume, window=3)["A"]
            == pytest.approx(di.avg_dollar_volume(close, volume, window=3, as_of="p4")["A"]))


def test_avg_dollar_volume_as_of_early_date_uses_available_rows():
    # A newly listed name has fewer than `window` periods -- not an error.
    close, volume = _frames({"A": ([10, 10, 10, 10], [5, 5, 5, 5])})
    assert di.avg_dollar_volume(close, volume, window=10, as_of="p1")["A"] == pytest.approx(50.0)


def test_avg_dollar_volume_converts_when_fx_supplied():
    close, volume = _frames({
        "NVDA":         ([10, 10, 10, 10], [1, 1, 1, 1]),      # 10 USD
        "ECOPETROL.CL": ([10, 10, 10, 10], [1, 1, 1, 1]),      # 10 COP
    })
    fx = pd.DataFrame({"USD": [1.0] * 4, "COP": [0.00025] * 4}, index=close.index)
    cur_map = {"NVDA": "USD", "ECOPETROL.CL": "COP"}
    adv = di.avg_dollar_volume(close, volume, window=2, fx=fx, cur_map=cur_map)
    assert adv["NVDA"] == pytest.approx(10.0)
    assert adv["ECOPETROL.CL"] == pytest.approx(0.0025)
    assert adv["NVDA"] > adv["ECOPETROL.CL"]      # equal raw ADV, correctly separated


def test_avg_dollar_volume_drops_unknown_currency_rather_than_ranking_it():
    close, volume = _frames({
        "NVDA":    ([10, 10, 10, 10], [1, 1, 1, 1]),
        "FOO.ZZZ": ([10, 10, 10, 10], [1, 1, 1, 1]),
    })
    fx = pd.DataFrame({"USD": [1.0] * 4}, index=close.index)
    adv = di.avg_dollar_volume(close, volume, window=2, fx=fx,
                               cur_map={"NVDA": "USD", "FOO.ZZZ": None})
    assert "FOO.ZZZ" not in adv.index      # excluded, not silently ranked at face value
    assert "NVDA" in adv.index


def test_infer_currency_from_exchange_suffix():
    assert di.infer_currency("NVDA") == "USD"          # bare ticker -> US listing
    assert di.infer_currency("ECOPETROL.CL") == "COP"
    assert di.infer_currency("SQMBCO.SN") == "CLP"
    assert di.infer_currency("NESN.SW") == "CHF"
    assert di.infer_currency("SAP.DE") == "EUR"
    assert di.infer_currency("SHOP.TO") == "CAD"
    assert di.infer_currency("7203.T") == "JPY"
    assert di.infer_currency("BHP.AX") == "AUD"


def test_infer_currency_from_isin_country_prefix():
    # Used before download resolves ISINs to tickers.
    assert di.infer_currency("US67066G1040") == "USD"   # NVDA
    assert di.infer_currency("DE0007164600") == "EUR"   # SAP
    assert di.infer_currency("GB0002374006") == "GBP"


def test_infer_currency_returns_none_for_unknown_rather_than_defaulting():
    # THE bug being fixed: silently defaulting an unknown suffix to USD would rank it
    # by raw local-currency magnitude, which is how a Chilean mid-cap out-ranked a
    # US mega-cap. Unknown must be reported, so the caller can exclude it.
    assert di.infer_currency("FOO.ZZZ") is None
    assert di.infer_currency("") is None


def test_infer_currency_overrides_win():
    # .L is genuinely ambiguous: LSE lists USD-denominated lines (CSPX.L is an ETF
    # reporting currency USD), so the suffix table must be overridable.
    assert di.infer_currency("CSPX.L") == "GBP"                             # table default
    assert di.infer_currency("CSPX.L", overrides={"CSPX.L": "USD"}) == "USD"


def _fx_stub(pairs):
    """DI seam mirroring download_all's download_fn: {pair: value} -> fetcher."""
    def fetch(pair, index):
        if pair not in pairs:
            return None
        return pd.Series(float(pairs[pair]), index=index)
    return fetch


def test_fetch_fx_rates_hub_is_unity_and_direct_pair_used():
    idx = ["p1", "p2"]
    # GBPUSD=X quotes USD per GBP -> used directly
    fx = di.fetch_fx_rates(["USD", "GBP"], idx, hub="USD",
                           fetch_fn=_fx_stub({"GBPUSD=X": 1.25}))
    assert fx["USD"].tolist() == [1.0, 1.0]
    assert fx["GBP"].tolist() == pytest.approx([1.25, 1.25])


def test_fetch_fx_rates_inverts_when_only_reverse_pair_exists():
    idx = ["p1"]
    # COPUSD=X does not exist; USDCOP=X quotes COP per USD -> must be inverted
    fx = di.fetch_fx_rates(["COP"], idx, hub="USD",
                           fetch_fn=_fx_stub({"USDCOP=X": 4000.0}))
    assert fx["COP"].iloc[0] == pytest.approx(1 / 4000.0)


def test_fetch_fx_rates_raises_when_a_pair_is_unavailable():
    # The silent-degradation trap: CLPCOP=X 404s and CHFCOP=X is delisted. Filling
    # the gap with 1.0 (or NaN->0) leaves those names unconverted while the code
    # reports success -- reintroducing the exact ranking bug. Fail loudly instead.
    with pytest.raises(ValueError, match="CLP"):
        di.fetch_fx_rates(["CLP"], ["p1"], hub="USD", fetch_fn=_fx_stub({}))


def test_to_hub_currency_scales_amounts_by_rate():
    idx = ["p1", "p2"]
    fx = pd.DataFrame({"USD": [1.0, 1.0], "COP": [0.00025, 0.00025]}, index=idx)
    amounts = pd.Series({"NVDA": 1e9, "ECOPETROL.CL": 1e9})
    cur_map = {"NVDA": "USD", "ECOPETROL.CL": "COP"}
    out = di.to_hub_currency(amounts, cur_map, fx)
    assert out["NVDA"] == pytest.approx(1e9)          # already USD
    assert out["ECOPETROL.CL"] == pytest.approx(250_000.0)   # 1e9 COP -> 250k USD


def _universe(n, periods=8):
    """n synthetic tickers with ADV descending in ticker order (T00 most liquid)."""
    idx = [f"p{i}" for i in range(periods)]
    close = pd.DataFrame({f"T{i:02d}": [100.0] * periods for i in range(n)}, index=idx)
    volume = pd.DataFrame({f"T{i:02d}": [float(n - i)] * periods for i in range(n)}, index=idx)
    return close, volume


def test_select_universe_is_a_noop_when_topn_is_none():
    # Gate 3: with the screen disabled the pipeline must reproduce today's universe
    # exactly, so a null topn bypasses every criterion including the eligibility ones.
    close, volume = _universe(6)
    assert di.select_universe(close, volume, None) == list(close.columns)


def test_select_universe_returns_all_when_topn_exceeds_universe():
    close, volume = _universe(6)
    assert di.select_universe(close, volume, 50) == list(close.columns)


def test_select_universe_pure_topn_takes_most_liquid():
    close, volume = _universe(10)
    assert di.select_universe(close, volume, 3, window=4) == ["T00", "T01", "T02"]


def test_select_universe_stratified_reaches_beyond_the_top_band():
    # The point of strata: a pure top-N gives only mega-caps. [2,1,1] over three
    # equal ADV bands must draw from the middle and bottom bands too.
    close, volume = _universe(9)
    picked = di.select_universe(close, volume, 4, strata=[2, 1, 1], window=4)
    assert picked[:2] == ["T00", "T01"]        # top band
    assert "T03" in picked                     # middle band (T03..T05)
    assert "T06" in picked                     # bottom band (T06..T08)
    assert len(picked) == 4


def test_select_universe_price_floor_excludes_penny_stocks():
    close, volume = _universe(4)
    close["T00"] = 0.4                          # most liquid, but sub-floor priced
    picked = di.select_universe(close, volume, 2, price_floor=1.0, window=4)
    assert "T00" not in picked


def test_select_universe_respects_min_active_fraction():
    close, volume = _universe(4)
    volume["T00"] = 0.0                         # never trades
    picked = di.select_universe(close, volume, 2, min_active_fraction=0.5, window=4)
    assert "T00" not in picked


def test_select_universe_excludes_unknown_currency():
    close, volume = _universe(3)
    fx = pd.DataFrame({"USD": [1.0] * 8}, index=close.index)
    picked = di.select_universe(close, volume, 3, window=4, fx=fx,
                                cur_map={"T00": None, "T01": "USD", "T02": "USD"})
    assert picked == ["T01", "T02"]


def test_select_universe_is_return_neutral():
    # THE property that makes the screen safe: a stock that rose 10x must not be
    # selected for that reason. Screening on past returns hands the model a universe
    # of pre-selected winners and destroys any read on whether it has skill.
    close, volume = _universe(6)
    close["T05"] = [10.0 * (1.6 ** i) for i in range(8)]   # +2500%, least liquid
    picked = di.select_universe(close, volume, 3, window=4)
    assert "T05" not in picked
    assert picked == ["T00", "T01", "T02"]


def test_select_universe_applies_market_cap_floor_when_supplied():
    close, volume = _universe(4)
    mc = pd.Series({"T00": 1e6, "T01": 1e12, "T02": 1e12, "T03": 1e12})
    picked = di.select_universe(close, volume, 2, window=4,
                                market_cap=mc, min_market_cap=1e9)
    assert "T00" not in picked                  # liquid but tiny


def test_select_universe_keeps_names_with_unknown_market_cap():
    # yfinance returns marketCap=None for ETFs (CSPX.L). Treating missing as zero
    # would silently delete every ETF from the universe.
    close, volume = _universe(3)
    mc = pd.Series({"T00": np.nan, "T01": 1e12, "T02": 1e12})
    picked = di.select_universe(close, volume, 3, window=4,
                                market_cap=mc, min_market_cap=1e9)
    assert "T00" in picked


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
