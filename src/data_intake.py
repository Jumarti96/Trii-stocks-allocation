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


import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_batch(batch, cfg):
    """Download one batch's Close+Volume from yfinance and clean it. Returns (close, volume) or None.

    Network. Extracts the Close and Volume sub-frames (multi-field columns for >1 ticker; single
    field set for 1 ticker), then clean_batch. One light retry on exception.
    """
    import yfinance as yf
    end = datetime.date.today()
    start = end - datetime.timedelta(days=cfg["days_of_data"])
    for attempt in (1, 2):
        try:
            raw = yf.download(batch, interval=cfg["interval"], start=start, end=end,
                              auto_adjust=True, threads=True, timeout=cfg["download_timeout"],
                              progress=False)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                close_raw, volume_raw = raw["Close"], raw["Volume"]
            else:  # single ticker -> flat columns
                close_raw = raw[["Close"]].rename(columns={"Close": batch[0]})
                volume_raw = raw[["Volume"]].rename(columns={"Volume": batch[0]})
            return clean_batch(close_raw, volume_raw, cfg["period_freq"])
        except Exception as e:  # noqa: BLE001 - batch-level resilience at scale
            if attempt == 2:
                print(f"  batch failed ({len(batch)} tickers): {e}")
                return None


def download_all(tickers, cfg, download_fn=None):
    """Download all tickers in parallel batches and concat into aligned (close, volume) frames.

    download_fn(batch) -> (close, volume) | None is a DI seam (default: download_batch with cfg).
    Raises RuntimeError if every batch fails.
    """
    if download_fn is None:
        download_fn = lambda batch: download_batch(batch, cfg)  # noqa: E731
    batches = make_batches(tickers, cfg["batch_size"])
    closes, volumes = [], []
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as ex:
        futures = {ex.submit(download_fn, b): b for b in batches if b}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                c, v = res
                closes.append(c)
                volumes.append(v)
    if not closes:
        raise RuntimeError("No data downloaded across all batches.")
    return pd.concat(closes, axis=1), pd.concat(volumes, axis=1)


def avg_dollar_volume(close, volume, window):
    """Mean of (Close * Volume) over the last `window` periods, per ticker. NaN treated as 0."""
    dv = (close * volume).fillna(0.0)
    return dv.iloc[-window:].mean(axis=0)


def liquidity_filter(close, volume, window, pct_of_median, min_group_size, market_key_fn=market_key):
    """Per-ticker keep/drop by relative dollar-volume within market groups.

    For each market group: size < min_group_size -> keep all (flag 'small_group', unreliable
    median); median <= 0 -> keep only members with adv > 0 (flag 'zero_median'); else keep members
    >= pct_of_median * group_median. Returns a DataFrame indexed by ticker with columns
    [avg_dollar_volume, market_group, kept, flag].
    """
    adv = avg_dollar_volume(close, volume, window)
    groups = pd.Series({t: market_key_fn(t) for t in adv.index})
    rows = {}
    for g in groups.unique():
        members = adv[groups[groups == g].index]
        if len(members) < min_group_size:
            for t in members.index:
                rows[t] = (members[t], g, True, "small_group")
        else:
            med = float(members.median())
            if med <= 0:
                for t in members.index:
                    rows[t] = (members[t], g, bool(members[t] > 0), "zero_median")
            else:
                thresh = pct_of_median * med
                for t in members.index:
                    rows[t] = (members[t], g, bool(members[t] >= thresh), "")
    detail = pd.DataFrame.from_dict(
        rows, orient="index", columns=["avg_dollar_volume", "market_group", "kept", "flag"]
    )
    return detail.loc[adv.index]


def grouping_health(detail):
    """Summarise market grouping from a liquidity_filter detail frame.

    Returns {'n_groups', 'other_fraction' (share of tickers in the OTHER catch-all),
    'flagged_groups' (groups containing any small_group/zero_median flag), 'groups' (per-group
    DataFrame: count, median adv, n_kept)}.
    """
    g = detail.groupby("market_group")
    groups = pd.DataFrame({
        "count": g.size(),
        "median": g["avg_dollar_volume"].median(),
        "n_kept": g["kept"].sum(),
    })
    flagged = sorted(detail.loc[detail["flag"] != "", "market_group"].unique().tolist())
    other_fraction = float((detail["market_group"] == "OTHER").mean())
    return {
        "n_groups": int(len(groups)),
        "other_fraction": other_fraction,
        "flagged_groups": flagged,
        "groups": groups,
    }
