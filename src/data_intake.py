"""
Data-intake helpers for pipeline step 1: load tickers, download Close+Volume in parallel batches,
and prune the universe by an activity filter (keep stocks that trade in >= min_active_fraction of
recent periods) plus the bad-data drop. Currency-free, no grouping.

Lives in src/ (importable) so pipeline/01_download.py (digit-prefixed, not importable) stays a thin
orchestrator and the experiment/test suite can import these functions directly.

See docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md.
"""

import datetime
import glob as _glob
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

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
            if attempt == 1:
                print(f"  batch retry ({len(batch)} tickers): {e}")
            else:
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
    close = pd.concat(closes, axis=1)
    volume = pd.concat(volumes, axis=1)
    # Guard against yfinance resolving two input identifiers to the same output symbol
    # (dual listings / ISIN aliases) -> duplicate columns crash the scalar per-ticker logic.
    duped = close.columns[close.columns.duplicated(keep=False)]
    if len(duped):
        print(f"  WARNING: dropped duplicate output columns (kept first): {sorted(set(duped))}")
        close = close.loc[:, ~close.columns.duplicated(keep="first")]
        volume = volume.loc[:, ~volume.columns.duplicated(keep="first")]
    return close, volume


def avg_dollar_volume(close, volume, window):
    """Mean of (Close * Volume) over the last `window` periods, per ticker. NaN treated as 0."""
    dv = (close * volume).fillna(0.0)
    return dv.iloc[-window:].mean(axis=0)


def active_fraction(volume, window):
    """Per ticker, the fraction of the last `window` periods with real (Volume > 0) trading.

    NaN volume counts as not-traded (NaN > 0 is False). Returns a Series in [0, 1] -- a currency-free,
    unitless activity measure (no grouping, no magnitude).
    """
    return (volume.iloc[-window:] > 0).mean(axis=0)


def activity_filter(close, volume, window=None, min_active_fraction=0.85):
    """Keep stocks that trade in at least `min_active_fraction` of the last `window` periods.

    window defaults to the last 10% of the time-series (min 10 periods). Pass an explicit
    integer to override (useful in tests with small fixtures).
    Returns a per-ticker DataFrame [avg_dollar_volume (informational), active_fraction, kept].
    """
    if window is None:
        window = max(10, len(close) // 10)
    adv = avg_dollar_volume(close, volume, window)
    af = active_fraction(volume, window)
    detail = pd.DataFrame({
        "avg_dollar_volume": adv,
        "active_fraction": af,
        "kept": af >= min_active_fraction,
    })
    return detail


def activity_health(detail):
    """Summarise the activity filter: counts plus the share of stocks that never trade.

    zero_volume_fraction (active_fraction == 0) is the data-source alarm: if it is high, Volume is
    probably missing from the feed rather than the stocks being genuinely inactive.
    """
    n_total = len(detail)
    n_kept = int(detail["kept"].sum())
    zero_volume_fraction = float((detail["active_fraction"] == 0).mean()) if n_total else 0.0
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_excluded": n_total - n_kept,
        "zero_volume_fraction": zero_volume_fraction,
    }


