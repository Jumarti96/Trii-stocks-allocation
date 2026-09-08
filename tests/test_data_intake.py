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


def test_normalise_currency_code_handles_minor_units():
    # Yahoo quotes LSE stocks in pence ('GBp') and Johannesburg in cents ('ZAc').
    # Treating those as GBP/ZAR overstates the amount 100x -- catastrophic for a
    # ranking whose whole job is comparing magnitudes across markets.
    assert di.normalise_currency_code("GBp") == ("GBP", 0.01)
    assert di.normalise_currency_code("ZAc") == ("ZAR", 0.01)
    assert di.normalise_currency_code("ILA") == ("ILS", 0.01)
    assert di.normalise_currency_code("GBP") == ("GBP", 1.0)   # case is the discriminator
    assert di.normalise_currency_code("usd") == ("USD", 1.0)
    assert di.normalise_currency_code(None) == (None, 1.0)


def test_resolve_listings_prefers_the_authoritative_lookup():
    # Exchange suffix is only a heuristic, and it is wrong for cross-listed ETFs:
    # CSPX.L is USD-denominated despite the .L suffix, and KY-domiciled ISINs are
    # frequently HKD-listed. Measured 87.5% suffix/ISIN accuracy over 56 names.
    got = di.resolve_listings(["CSPX.L", "NVDA"],
                              fetch_fn=lambda s: {"CSPX.L": {"currency": "USD"}}.get(s))
    assert got.loc["CSPX.L", "currency"] == "USD"       # lookup beat the .L -> GBP table
    assert got.loc["NVDA", "currency"] == "USD"


def test_resolve_listings_falls_back_to_inference_when_lookup_is_empty():
    got = di.resolve_listings(["ECOPETROL.CL"], fetch_fn=lambda s: None)
    assert got.loc["ECOPETROL.CL", "currency"] == "COP"
    assert got.loc["ECOPETROL.CL", "source"] == "inferred"


def test_resolve_listings_records_the_minor_unit_factor():
    got = di.resolve_listings(["VOD.L"], fetch_fn=lambda s: {"currency": "GBp"})
    assert got.loc["VOD.L", "currency"] == "GBP"
    assert got.loc["VOD.L", "unit_factor"] == 0.01


def test_resolve_listings_parallel_preserves_input_order():
    # One HTTP round-trip per stock, so a 3k catalogue is ~46 min sequentially. Threads
    # complete out of order, and the row order must still match the input -- callers
    # zip this against price columns.
    ids = [f"T{i:03d}" for i in range(40)]
    got = di.resolve_listings(ids, fetch_fn=lambda s: {"currency": "USD", "symbol": s},
                              workers=8)
    assert list(got.index) == ids


def test_resolve_listings_parallel_matches_sequential():
    ids = ["A.L", "B", "C.SN"]
    fetch = lambda s: {"A.L": {"currency": "GBp"}, "B": {"currency": "USD"}}.get(s)
    seq = di.resolve_listings(ids, fetch_fn=fetch, workers=1)
    par = di.resolve_listings(ids, fetch_fn=fetch, workers=4)
    pd.testing.assert_frame_equal(seq, par)
    assert par.loc["A.L", "unit_factor"] == 0.01        # pence survives threading
    assert par.loc["C.SN", "source"] == "inferred"      # fallback survives threading


def test_resolve_listings_survives_a_failing_lookup():
    # One bad identifier must not abort a 3k-stock run.
    def flaky(s):
        if s == "BAD":
            raise RuntimeError("boom")
        return {"currency": "USD"}
    got = di.resolve_listings(["GOOD", "BAD"], fetch_fn=flaky, workers=2,
                              retries=1, retry_wait=0.0)
    assert got.loc["GOOD", "currency"] == "USD"
    # Fell back and did not crash -- but recorded as 'failed', not 'inferred', so a
    # resume knows to retry it rather than trusting the heuristic forever.
    assert got.loc["BAD", "source"] == "failed"
    assert got.loc["BAD", "currency"] is not None       # still usable meanwhile


def test_resolve_listings_captures_symbol_and_name():
    # The catalogue is ISINs, and yfinance labels its output columns with the input
    # identifier -- so without this the final allocation report would name
    # 'US67066G1040' rather than 'NVDA', which nobody can trade against. The symbol
    # arrives in the same .info call as the currency, at no extra network cost.
    got = di.resolve_listings(
        ["US67066G1040"],
        fetch_fn=lambda s: {"currency": "USD", "symbol": "NVDA",
                            "shortName": "NVIDIA Corporation"})
    assert got.loc["US67066G1040", "symbol"] == "NVDA"
    assert got.loc["US67066G1040", "name"] == "NVIDIA Corporation"


def test_resolve_listings_falls_back_to_the_identifier_as_symbol():
    got = di.resolve_listings(["ECOPETROL.CL"], fetch_fn=lambda s: None)
    assert got.loc["ECOPETROL.CL", "symbol"] == "ECOPETROL.CL"


def test_resolve_listings_captures_classification_fields():
    # Sector, size and instrument type ride along in the .info call already being
    # made, so capturing them costs nothing -- and without them there is no way to
    # audit whether a top-N liquidity screen keeps a sensible slice of the catalogue.
    got = di.resolve_listings(
        ["US67066G1040"],
        fetch_fn=lambda s: {"currency": "USD", "symbol": "NVDA",
                            "sector": "Technology", "industry": "Semiconductors",
                            "marketCap": 4.2e12, "exchange": "NMS",
                            "quoteType": "EQUITY"})
    row = got.loc["US67066G1040"]
    assert row["sector"] == "Technology"
    assert row["industry"] == "Semiconductors"
    assert row["market_cap"] == pytest.approx(4.2e12)
    assert row["exchange"] == "NMS"
    assert row["quote_type"] == "EQUITY"


def test_resolve_listings_leaves_a_missing_market_cap_as_nan():
    # yfinance reports no marketCap for ETFs. Zero-filling would make every ETF fail
    # select_universe's min_market_cap gate, which deliberately keeps NaN so they
    # survive -- so the absence has to stay distinguishable from a genuine zero.
    got = di.resolve_listings(
        ["IE00B5BMR087"],
        fetch_fn=lambda s: {"currency": "USD", "symbol": "CSPX.L",
                            "quoteType": "ETF"})
    assert pd.isna(got.loc["IE00B5BMR087", "market_cap"])
    assert got.loc["IE00B5BMR087", "quote_type"] == "ETF"
    assert pd.isna(got.loc["IE00B5BMR087", "sector"])


def test_convert_currency_applies_unit_factors():
    idx = ["p1"]
    fx = pd.DataFrame({"GBP": [1.35], "USD": [1.0]}, index=idx)
    amounts = pd.Series({"VOD.L": 1e8, "NVDA": 1e8})
    out = di.convert_currency(amounts, {"VOD.L": "GBP", "NVDA": "USD"}, fx,
                              unit_factors={"VOD.L": 0.01, "NVDA": 1.0})
    assert out["VOD.L"] == pytest.approx(1e8 * 0.01 * 1.35)   # pence, not pounds
    assert out["NVDA"] == pytest.approx(1e8)


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


def test_convert_currency_scales_amounts_by_rate():
    idx = ["p1", "p2"]
    fx = pd.DataFrame({"USD": [1.0, 1.0], "COP": [0.00025, 0.00025]}, index=idx)
    amounts = pd.Series({"NVDA": 1e9, "ECOPETROL.CL": 1e9})
    cur_map = {"NVDA": "USD", "ECOPETROL.CL": "COP"}
    out = di.convert_currency(amounts, cur_map, fx)          # target=None -> the hub
    assert out["NVDA"] == pytest.approx(1e9)          # already USD
    assert out["ECOPETROL.CL"] == pytest.approx(250_000.0)   # 1e9 COP -> 250k USD


def test_convert_currency_to_an_arbitrary_target():
    # The report is denominated in the user's own currency, not the FX hub. The hub
    # cancels: rate = fx[local] / fx[target].
    idx = ["p1"]
    fx = pd.DataFrame({"USD": [1.0], "COP": [0.00025]}, index=idx)
    out = di.convert_currency(pd.Series({"NVDA": 100.0}), {"NVDA": "USD"}, fx,
                              target="COP")
    assert out["NVDA"] == pytest.approx(400_000.0)    # $100 at 4000 COP/USD


def test_convert_currency_target_equal_to_source_is_identity():
    idx = ["p1"]
    fx = pd.DataFrame({"COP": [0.00025]}, index=idx)
    out = di.convert_currency(pd.Series({"ECO.CL": 2700.0}), {"ECO.CL": "COP"}, fx,
                              target="COP")
    assert out["ECO.CL"] == pytest.approx(2700.0)


@pytest.mark.parametrize("policy,expected", [
    ("exclude", None),          # dropped
    ("assume_target", 500.0),   # taken at face value in the target currency
])
def test_convert_currency_unknown_policy(policy, expected):
    # Robustness: an unresolvable currency must not crash the run. Excluding is the
    # default because assuming the wrong one is how a JPY name gets ranked 150x too
    # high -- the exact failure the currency work exists to prevent.
    idx = ["p1"]
    fx = pd.DataFrame({"USD": [1.0]}, index=idx)
    out = di.convert_currency(pd.Series({"MYSTERY": 500.0}), {"MYSTERY": None}, fx,
                              target="USD", unknown=policy)
    if expected is None:
        assert "MYSTERY" not in out.index
    else:
        assert out["MYSTERY"] == pytest.approx(expected)


def test_convert_currency_rejects_an_unknown_policy_name():
    fx = pd.DataFrame({"USD": [1.0]}, index=["p1"])
    with pytest.raises(ValueError, match="unknown_currency"):
        di.convert_currency(pd.Series({"A": 1.0}), {"A": None}, fx, unknown="whatever")


def test_convert_currency_excludes_a_currency_missing_from_fx():
    # Currency resolved, but no FX rate for it -- must behave like any other unknown
    # rather than raising a KeyError mid-report.
    fx = pd.DataFrame({"USD": [1.0]}, index=["p1"])
    out = di.convert_currency(pd.Series({"A": 5.0}), {"A": "XYZ"}, fx, target="USD")
    assert "A" not in out.index


def test_convert_panel_scales_each_period_by_that_period_s_rate():
    # The whole point of the panel form: a fixed-rate conversion would leave FX moves
    # out of the returns, which is the bug it exists to fix.
    idx = ["p1", "p2", "p3"]
    fx = pd.DataFrame({"USD": [1.0, 1.0, 1.0], "COP": [0.00025, 0.00020, 0.00025]},
                      index=idx)
    prices = pd.DataFrame({"NVDA": [100.0, 100.0, 100.0],
                           "ECO.CL": [4000.0, 4000.0, 4000.0]}, index=idx)
    out = di.convert_panel(prices, {"NVDA": "USD", "ECO.CL": "COP"}, fx)
    assert out["NVDA"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    assert out["ECO.CL"].tolist() == pytest.approx([1.0, 0.8, 1.0])
    # A flat native price still produces a return, because COP depreciated.
    assert out["ECO.CL"].pct_change().iloc[1] == pytest.approx(-0.2)


def test_convert_panel_is_identity_for_a_hub_only_panel():
    idx = ["p1", "p2"]
    fx = pd.DataFrame({"USD": [1.0, 1.0]}, index=idx)
    prices = pd.DataFrame({"A": [10.0, 11.0], "B": [5.0, 5.5]}, index=idx)
    out = di.convert_panel(prices, {"A": "USD", "B": "USD"}, fx)
    pd.testing.assert_frame_equal(out, prices)


def test_convert_panel_applies_unit_factors():
    idx = ["p1"]
    fx = pd.DataFrame({"GBP": [1.35], "USD": [1.0]}, index=idx)
    prices = pd.DataFrame({"VOD.L": [7000.0], "NVDA": [100.0]}, index=idx)
    out = di.convert_panel(prices, {"VOD.L": "GBP", "NVDA": "USD"}, fx,
                           unit_factors={"VOD.L": 0.01, "NVDA": 1.0})
    assert out["VOD.L"].iloc[0] == pytest.approx(70.0 * 1.35)   # pence, not pounds
    assert out["NVDA"].iloc[0] == pytest.approx(100.0)


def test_convert_panel_to_an_arbitrary_target():
    idx = ["p1"]
    fx = pd.DataFrame({"USD": [1.0], "COP": [0.00025]}, index=idx)
    out = di.convert_panel(pd.DataFrame({"NVDA": [100.0]}, index=idx),
                           {"NVDA": "USD"}, fx, target="COP")
    assert out["NVDA"].iloc[0] == pytest.approx(400_000.0)


@pytest.mark.parametrize("policy,expected", [
    ("exclude", None),          # column dropped entirely
    ("assume_target", 500.0),   # taken at face value
])
def test_convert_panel_unknown_policy(policy, expected):
    idx = ["p1"]
    fx = pd.DataFrame({"USD": [1.0]}, index=idx)
    prices = pd.DataFrame({"MYSTERY": [500.0], "NVDA": [100.0]}, index=idx)
    out = di.convert_panel(prices, {"MYSTERY": None, "NVDA": "USD"}, fx,
                           unknown=policy)
    assert "NVDA" in out.columns
    if expected is None:
        assert "MYSTERY" not in out.columns
    else:
        assert out["MYSTERY"].iloc[0] == pytest.approx(expected)


def test_convert_panel_rejects_an_unknown_policy_name():
    fx = pd.DataFrame({"USD": [1.0]}, index=["p1"])
    with pytest.raises(ValueError, match="unknown_currency"):
        di.convert_panel(pd.DataFrame({"A": [1.0]}, index=["p1"]), {"A": None}, fx,
                         unknown="whatever")


def test_convert_panel_raises_when_fx_does_not_cover_every_period():
    # Silent misalignment would convert some periods at a neighbouring week's rate,
    # producing fabricated returns. Refuse instead.
    fx = pd.DataFrame({"USD": [1.0]}, index=["p1"])
    prices = pd.DataFrame({"A": [1.0, 2.0]}, index=["p1", "p2"])
    with pytest.raises(ValueError, match="p2"):
        di.convert_panel(prices, {"A": "USD"}, fx)


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


def test_select_universe_gates_read_the_frame_tail_not_as_of():
    # SHARP EDGE for backtest callers: as_of truncates the LIQUIDITY ranking only.
    # The price_floor and active_fraction gates read the tail of whatever frame they
    # are handed, so passing a full-history frame with an early as_of leaks future
    # data through those gates. Callers must slice the frame themselves; this pins
    # the behaviour so the requirement is discoverable rather than folklore.
    idx = [f"p{i}" for i in range(6)]
    # CHEAP was above the floor early and collapsed later; RICH did the reverse.
    close = pd.DataFrame({"CHEAP": [100.0] * 3 + [1.0] * 3,
                          "RICH": [1.0] * 3 + [100.0] * 3}, index=idx)
    volume = pd.DataFrame({"CHEAP": [1e6] * 6, "RICH": [1.0] * 6}, index=idx)

    # Full frame, as_of early: the floor still judges by the LAST rows, so the name
    # that was expensive back then is excluded.
    leaky = di.select_universe(close, volume, topn=1, window=3, price_floor=50.0,
                               as_of="p2")
    assert leaky == ["RICH"]

    # Sliced frame: the floor now sees only history up to p2, and CHEAP qualifies.
    honest = di.select_universe(close.iloc[:3], volume.iloc[:3], topn=1, window=3,
                                price_floor=50.0, as_of="p2")
    assert honest == ["CHEAP"]


# ---------------------------------------------------------------------------
# Rate-limit resilience
#
# A 2,930-name catalogue tripped Yahoo's rate limiter partway through, and because a
# raised exception fell back to inference exactly like a genuine absence, the run
# reported success while 54% of the catalogue carried heuristic currencies. The
# damage landed where it hurts: NL0015000RT3 is NRP.JO, quoted in Johannesburg cents,
# but the NL prefix inferred EUR -- wrong currency AND a missing 100x minor-unit
# factor, which floated it to rank 2 of the liquidity screen.
# ---------------------------------------------------------------------------

class _Flaky:
    """Fails the first `n_failures` calls per identifier, then succeeds."""

    def __init__(self, n_failures, info=None):
        self.n_failures = n_failures
        self.calls = {}
        self.info = info or {"currency": "USD", "symbol": "OK"}

    def __call__(self, ident):
        self.calls[ident] = self.calls.get(ident, 0) + 1
        if self.calls[ident] <= self.n_failures:
            raise RuntimeError("Too Many Requests. Rate limited.")
        return self.info


def test_resolve_listings_retries_a_failing_lookup():
    fetch = _Flaky(n_failures=2)
    got = di.resolve_listings(["A"], fetch_fn=fetch, retries=3, retry_wait=0.0)
    assert got.loc["A", "source"] == "lookup"
    assert got.loc["A", "currency"] == "USD"
    assert fetch.calls["A"] == 3


def test_resolve_listings_marks_an_exhausted_lookup_as_failed_not_inferred():
    # THE bug: an exception and a genuinely currency-less response both became
    # 'inferred', so a rate-limited run was indistinguishable from a complete one.
    # 'failed' is what makes a resume know which rows to retry.
    got = di.resolve_listings(["ZAE000351946"], fetch_fn=_Flaky(n_failures=99),
                              retries=2, retry_wait=0.0)
    assert got.loc["ZAE000351946", "source"] == "failed"


def test_resolve_listings_still_infers_when_the_lookup_genuinely_lacks_a_currency():
    # A successful response with no currency field is not a failure -- inference is
    # the right answer, and it must stay distinguishable from a dropped call.
    got = di.resolve_listings(["ECOPETROL.CL"], fetch_fn=lambda s: {"symbol": "ECO"},
                              retries=2, retry_wait=0.0)
    assert got.loc["ECOPETROL.CL", "source"] == "inferred"


def test_resolve_listings_reuses_already_resolved_rows():
    # Resuming must not re-spend a network call on a name already resolved: that is
    # the whole point, at ~1.5s per call across thousands of names.
    existing = pd.DataFrame(
        {"currency": ["USD"], "unit_factor": [1.0], "symbol": ["AAPL"],
         "name": ["Apple"], "source": ["lookup"], "sector": ["Technology"],
         "industry": ["Consumer Electronics"], "market_cap": [4.6e12],
         "exchange": ["NMS"], "quote_type": ["EQUITY"]},
        index=["US0378331005"])
    fetch = _Flaky(n_failures=0, info={"currency": "JPY", "symbol": "7269.T"})
    got = di.resolve_listings(["US0378331005", "JP3397200001"], fetch_fn=fetch,
                              existing=existing, retries=1, retry_wait=0.0)
    assert "US0378331005" not in fetch.calls          # not re-fetched
    assert got.loc["US0378331005", "symbol"] == "AAPL"
    assert got.loc["JP3397200001", "symbol"] == "7269.T"


def test_resolve_listings_retries_rows_that_previously_failed():
    # A resume must re-attempt exactly the rows the rate limiter ate.
    existing = pd.DataFrame(
        {"currency": ["EUR"], "unit_factor": [1.0], "symbol": ["NL0015000RT3"],
         "name": [None], "source": ["failed"], "sector": [None], "industry": [None],
         "market_cap": [None], "exchange": [None], "quote_type": [None]},
        index=["NL0015000RT3"])
    fetch = _Flaky(n_failures=0, info={"currency": "ZAc", "symbol": "NRP.JO"})
    got = di.resolve_listings(["NL0015000RT3"], fetch_fn=fetch, existing=existing,
                              retries=1, retry_wait=0.0)
    assert fetch.calls["NL0015000RT3"] == 1
    assert got.loc["NL0015000RT3", "currency"] == "ZAR"     # normalised from ZAc
    assert got.loc["NL0015000RT3", "unit_factor"] == pytest.approx(0.01)


def test_resolve_listings_retries_previously_inferred_rows_too():
    # Inference is a fallback, not an answer. If a later pass can reach the network,
    # an authoritative lookup should replace the heuristic.
    existing = pd.DataFrame(
        {"currency": ["USD"], "unit_factor": [1.0], "symbol": ["KYG875721634"],
         "name": [None], "source": ["inferred"], "sector": [None], "industry": [None],
         "market_cap": [None], "exchange": [None], "quote_type": [None]},
        index=["KYG875721634"])
    fetch = _Flaky(n_failures=0, info={"currency": "HKD", "symbol": "0700.HK"})
    got = di.resolve_listings(["KYG875721634"], fetch_fn=fetch, existing=existing,
                              retries=1, retry_wait=0.0)
    assert got.loc["KYG875721634", "currency"] == "HKD"
    assert got.loc["KYG875721634", "source"] == "lookup"
