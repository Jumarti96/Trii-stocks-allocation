"""
Data-intake helpers for pipeline step 1: load tickers, download Close+Volume in parallel batches,
and prune the universe by a currency-robust relative-liquidity filter (with conservative
small/degenerate-group handling and a grouping-health diagnostic).

Lives in src/ (importable) so pipeline/01_download.py (digit-prefixed, not importable) stays a thin
orchestrator and the experiment/test suite can import these functions directly.

See docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md.
"""

import glob as _glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def load_tickers(csv_glob):
    """Load and clean tickers/ISINs from every CSV matching csv_glob.

    Reads utf-8-sig (strips BOM), coerces to str, strips whitespace, drops empty and 'nan',
    de-duplicates. Returns a list of any size.
    """
    out = set()
    for path in _glob.glob(csv_glob):
        col = pd.read_csv(path, header=None, encoding="utf-8-sig")[0]
        for raw in col.astype(str).tolist():
            t = raw.strip()
            if t and t.lower() != "nan":
                out.add(t)
    return sorted(out)


def make_batches(tickers, batch_size):
    """Split a ticker list into consecutive batches of at most batch_size."""
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]


def market_key(identifier):
    """Currency-group key: ISIN country prefix, else ticker exchange suffix, else 'OTHER'.

    ISIN = 12 chars (2-letter country code + 10 alphanumerics). The 'OTHER' catch-all holds plain
    tickers and anything that fits no pattern; grouping_health reports its share so a list that does
    not fit the patterns is visible. NOTE: an ISIN prefix is issuer domicile, a strong-but-imperfect
    proxy for trading currency.
    """
    s = str(identifier).strip().upper()
    if len(s) == 12 and s[:2].isalpha() and s[2:].isalnum():
        return s[:2]
    if "." in s:
        return s.rsplit(".", 1)[-1]
    return "OTHER"


def clean_batch(close_raw, volume_raw, period_freq, missing_frac=0.15):
    """Clean a batch's Close + Volume frames: period index, drop >missing_frac-missing Close
    tickers, ffill/bfill Close, align Volume to the kept names, string period-end index.

    Pure (no network). Returns (close, volume) over the same kept columns and string index.
    """
    close = close_raw.copy()
    volume = volume_raw.copy()
    close.index = close.index.to_period(freq=period_freq)
    volume.index = volume.index.to_period(freq=period_freq)
    close = close.groupby(level=0).first().sort_index()
    volume = volume.groupby(level=0).first().sort_index()

    keep = close.columns[close.isna().sum() < close.shape[0] * missing_frac]
    close = close[keep].ffill().bfill()
    volume = volume.reindex(columns=keep)

    close.index = close.index.astype("str").str.split("/").str[-1]
    volume.index = volume.index.astype("str").str.split("/").str[-1]
    return close, volume
