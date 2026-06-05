# Step-1 Scaling + Liquidity Filter (PR-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale pipeline step 1 to 4,000+ tickers via parallel batch downloads of Close+Volume, prune the universe early with a currency-robust relative-liquidity filter (with conservative small/degenerate-group handling + a grouping-health diagnostic), and add a local calibration experiment to choose the threshold empirically.

**Architecture:** Put the reusable, testable functions in a new importable `src/data_intake.py`; `pipeline/01_download.py` becomes a thin orchestrator that imports them (the digit-prefixed pipeline module isn't importable by tests, so the logic lives in `src/`). The calibration experiment (`experiments/calibrate_liquidity_filter.py`) imports the same functions, exercising the real production code. Pure functions + dependency-injection seams keep everything testable without the network.

**Tech Stack:** Python, numpy, pandas, yfinance (network only), `concurrent.futures.ThreadPoolExecutor`, pytest.

Spec: `docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md`.

**Conventions:**
- Test runner: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`
- Part A (`src/data_intake.py`, `pipeline/01_download.py`, `params.yaml`, `tests/test_data_intake.py`) is a **functional pipeline change → pushed as a PR**; tests ship with it. No AI attribution in commits.
- Part B (`experiments/calibrate_liquidity_filter.py`, its test) and the spec/plan stay **local**.

---

## File Structure

- Create: `src/data_intake.py` — ticker loading, batched download, liquidity filter, grouping health.
- Modify: `pipeline/01_download.py` — thin orchestrator importing `data_intake`.
- Modify: `params.yaml` — new download + liquidity config keys.
- Create: `tests/test_data_intake.py` — unit tests (no network/torch).
- Create: `experiments/calibrate_liquidity_filter.py` — threshold-sweep calibration (local).
- Create: `tests/test_calibrate_liquidity_filter.py` — calibration unit tests.

---

### Task 1: `src/data_intake.py` scaffold + `load_tickers`

**Files:**
- Create: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_intake.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_intake'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/data_intake.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add data_intake.load_tickers with BOM/nan/dup hygiene"
```

---

### Task 2: `make_batches` + `market_key`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -k "make_batches or market_key" -v`
Expected: FAIL — `AttributeError: ... 'make_batches'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/data_intake.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add make_batches and market_key grouping"
```

---

### Task 3: `clean_batch`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py::test_clean_batch_drops_missing_and_aligns_volume -v`
Expected: FAIL — `AttributeError: ... 'clean_batch'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/data_intake.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add clean_batch (Close+Volume clean transform)"
```

---

### Task 4: `download_all` (parallel, DI seam) + `download_batch`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -k download_all -v`
Expected: FAIL — `AttributeError: ... 'download_all'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/data_intake.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add parallel download_all + download_batch (Close+Volume)"
```

---

### Task 5: `avg_dollar_volume` + `liquidity_filter`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -k "avg_dollar or liquidity_filter" -v`
Expected: FAIL — `AttributeError: ... 'avg_dollar_volume'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/data_intake.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add avg_dollar_volume and relative liquidity_filter with group safeguards"
```

---

### Task 6: `grouping_health`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py::test_grouping_health_reports_default_fraction_and_flags -v`
Expected: FAIL — `AttributeError: ... 'grouping_health'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/data_intake.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add grouping_health diagnostic"
```

---

### Task 7: `pipeline/01_download.py` thin orchestrator + `params.yaml`

**Files:**
- Modify: `pipeline/01_download.py`
- Modify: `params.yaml`
- Test: (smoke only — covered in Task 9)

- [ ] **Step 1: Add config keys to `params.yaml`**

In `params.yaml`, under the `# Data and timing` block (after `days_of_data: 3650`), add:

```yaml
# Step 1 download + liquidity filter
batch_size: 500                 # tickers per download batch
download_workers: 3             # parallel batch workers
download_timeout: 10            # per-batch yfinance timeout (s)
liquidity_window: 52            # periods for avg dollar-volume (weekly -> ~1yr)
liquidity_pct_of_median: 0.10   # keep >= 10% of market-group median (set from calibration)
liquidity_min_group_size: 5     # groups smaller than this are kept-and-flagged
liquidity_topn: null            # optional cap (future knob; null = off)
```

- [ ] **Step 2: Rewrite `pipeline/01_download.py` as a thin orchestrator**

Replace the body of `pipeline/01_download.py` with:

```python
"""
Step 1 - Download and Preprocess Stock Data (parallel batches + liquidity filter)

Downloads Close+Volume for every ticker/ISIN in stock_tickers/*.csv in parallel batches, prunes the
universe early by a currency-robust relative-liquidity filter (avg dollar-volume vs market-group
median, with conservative small/degenerate-group handling), and writes the PRUNED prices/returns.

Outputs (data/):
    01_prices.csv     - adjusted close prices for the kept (liquid) universe
    01_returns.csv    - period returns for the kept universe
    01_liquidity.csv  - per kept ticker: avg dollar-volume, market group, flag (audit)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from config import load_config, PATHS, BASE_DIR
from data_intake import load_tickers, download_all, liquidity_filter, grouping_health


def main():
    cfg = load_config()
    t0 = time.time()
    print("\n=== Step 1: Download (parallel batches) + liquidity filter ===")

    tickers = load_tickers(os.path.join(BASE_DIR, "stock_tickers", "*.csv"))
    print(f"Loaded {len(tickers)} unique tickers.")

    close, volume = download_all(tickers, cfg)
    print(f"Downloaded {close.shape[1]} valid tickers.")

    detail = liquidity_filter(
        close, volume,
        window=cfg["liquidity_window"], pct_of_median=cfg["liquidity_pct_of_median"],
        min_group_size=cfg["liquidity_min_group_size"],
    )
    health = grouping_health(detail)
    print(f"Grouping: {health['n_groups']} groups | OTHER bucket "
          f"{health['other_fraction']:.0%} | flagged groups: {health['flagged_groups']}")
    if health["other_fraction"] > 0.25:
        print("  WARNING: large OTHER bucket -> the ticker list may not fit the grouping patterns.")

    kept = detail.index[detail["kept"]]
    print(f"Kept after liquidity filter: {len(kept)} / {close.shape[1]}")

    close_kept = close[kept]
    rets = close_kept.pct_change().iloc[1:]

    os.makedirs(os.path.dirname(PATHS["01_prices"]), exist_ok=True)
    close_kept.to_csv(PATHS["01_prices"])
    rets.to_csv(PATHS["01_returns"])
    detail.loc[kept].to_csv(os.path.join(os.path.dirname(PATHS["01_prices"]), "01_liquidity.csv"))

    print(f"Prices  shape: {close_kept.shape}")
    print(f"Returns shape: {rets.shape}")
    print(f"  Step 1 completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -c "import yaml; print(yaml.safe_load(open('params.yaml'))['liquidity_pct_of_median'])"`
Expected: prints `0.1` (config parses).

- [ ] **Step 4: Commit**

```bash
git add pipeline/01_download.py params.yaml
git commit -m "Rewrite step 1 as thin orchestrator over data_intake; add liquidity config"
```

---

### Task 8: Calibration experiment `experiments/calibrate_liquidity_filter.py`

**Files:**
- Create: `experiments/calibrate_liquidity_filter.py`
- Test: `tests/test_calibrate_liquidity_filter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibrate_liquidity_filter.py`:

```python
import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import calibrate_liquidity_filter as clf


def test_sweep_thresholds_counts_survivors():
    idx = ["p1", "p2", "p3", "p4"]
    close = pd.DataFrame({
        "US0000000001": [10, 10, 10, 10], "US0000000002": [10, 10, 10, 10],
        "US0000000003": [10, 10, 10, 10], "US0000000004": [10, 10, 10, 10],
        "US0000000005": [10, 10, 10, 10],
    }, index=idx)
    volume = pd.DataFrame({
        "US0000000001": [100, 100, 100, 100], "US0000000002": [100, 100, 100, 100],
        "US0000000003": [100, 100, 100, 100], "US0000000004": [100, 100, 100, 100],
        "US0000000005": [1, 1, 1, 1],          # tiny -> dropped at high thresholds
    }, index=idx)
    table = clf.sweep_thresholds(close, volume, window=4, grid=[0.0, 0.5], min_group_size=5)
    assert table.loc[0.0, "kept_total"] == 5
    assert table.loc[0.5, "kept_total"] == 4      # the tiny name drops at 50% of median
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_calibrate_liquidity_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibrate_liquidity_filter'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/calibrate_liquidity_filter.py`:

```python
"""
Calibrate the liquidity filter: download a ticker source once, then sweep the pct_of_median
threshold to see how many stocks survive (total + per market group), so the production
liquidity_pct_of_median default can be chosen empirically. Also reports grouping health.

Run:  python experiments/calibrate_liquidity_filter.py --sources stock_tickers/isins_list.csv
      python experiments/calibrate_liquidity_filter.py \
        --sources stock_tickers/colombia_stocks_trii.csv,stock_tickers/global_stocks_trii.csv \
        --list-names

Imports the REAL production functions from src/data_intake. Local/experiment-only.
See docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import load_config
from data_intake import load_tickers, download_all, liquidity_filter, grouping_health


def sweep_thresholds(close, volume, window, grid, min_group_size):
    """For each pct_of_median in grid, count survivors (total and per market group).

    Returns a DataFrame indexed by threshold with a 'kept_total' column plus one column per market
    group ('kept_<group>').
    """
    rows = {}
    for pct in grid:
        detail = liquidity_filter(close, volume, window, pct, min_group_size)
        kept = detail[detail["kept"]]
        row = {"kept_total": int(len(kept))}
        for g, sub in kept.groupby("market_group"):
            row[f"kept_{g}"] = int(len(sub))
        rows[pct] = row
    table = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    table.index.name = "pct_of_median"
    return table


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=str, required=True,
                        help="Comma-separated ticker CSV paths to analyse together")
    parser.add_argument("--grid", type=str, default="0,0.01,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--list-names", action="store_true",
                        help="Also dump kept/excluded ticker names (for small sources)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "liquidity_calibration"))
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    grid = [float(x) for x in args.grid.split(",")]
    sources = args.sources.split(",")
    os.makedirs(args.outdir, exist_ok=True)

    tickers = sorted({t for s in sources for t in load_tickers(s)})
    print(f"Sources: {sources} | {len(tickers)} tickers")
    close, volume = download_all(tickers, cfg)
    print(f"Downloaded {close.shape[1]} valid tickers.")

    table = sweep_thresholds(close, volume, cfg["liquidity_window"], grid,
                             cfg["liquidity_min_group_size"])
    tag = "_".join(os.path.splitext(os.path.basename(s))[0] for s in sources)[:60]
    table.to_csv(os.path.join(args.outdir, f"survival_{tag}.csv"))

    detail = liquidity_filter(close, volume, cfg["liquidity_window"],
                              cfg["liquidity_pct_of_median"], cfg["liquidity_min_group_size"])
    health = grouping_health(detail)
    health["groups"].to_csv(os.path.join(args.outdir, f"grouping_{tag}.csv"))

    print("\nSurvival vs threshold:")
    print(table.to_string())
    print(f"\nGrouping: {health['n_groups']} groups | OTHER {health['other_fraction']:.0%} | "
          f"flagged: {health['flagged_groups']}")

    if args.list_names:
        names_path = os.path.join(args.outdir, f"names_{tag}.csv")
        detail.sort_values("avg_dollar_volume", ascending=False).to_csv(names_path)
        print(f"Wrote kept/excluded names: {names_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_calibrate_liquidity_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/calibrate_liquidity_filter.py tests/test_calibrate_liquidity_filter.py
git commit -m "Add liquidity-filter calibration sweep experiment"
```

---

### Task 9: Full-suite check + step-1 smoke + calibration runs

**Files:** none (verification only).

- [ ] **Step 1: Run the whole test suite**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest -q`
Expected: PASS — all prior tests plus the new data_intake (10) and calibration (1) tests.

- [ ] **Step 2: Step-1 smoke (network) — keep it small**

NOTE: `pipeline/01_download.py` globs **all** `stock_tickers/*.csv`, which now includes
`isins_list.csv` (3,918). To keep the pipeline smoke quick, first move the big list aside, run,
then restore it:
```bash
mv stock_tickers/isins_list.csv ./isins_list.csv.bak
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" pipeline/01_download.py
mv ./isins_list.csv.bak stock_tickers/isins_list.csv
```
Expected: prints loaded/downloaded/grouping/kept counts + a timer; writes `data/01_prices.csv`,
`01_returns.csv`, `01_liquidity.csv`. Confirm the files exist and `01_liquidity.csv` has
`market_group` + `flag` columns. **This also reveals the yfinance column-label form for ISINs**
(ISIN vs resolved ticker) — report it. (The small-source calibration in Step 3 is the primary
integration smoke; this full-pipeline run is a secondary check.)

- [ ] **Step 3: Calibration runs (network; the real deliverable)**

Run (small, with names) and (scale):
```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/calibrate_liquidity_filter.py --sources stock_tickers/colombia_stocks_trii.csv,stock_tickers/global_stocks_trii.csv --list-names
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/calibrate_liquidity_filter.py --sources stock_tickers/isins_list.csv
```
Expected: a survival-vs-threshold table per source + grouping report under `experiments/results/liquidity_calibration/`. The 3,918-ISIN run also validates the parallel download at scale (run in the background if slow).

- [ ] **Step 4: Report for threshold selection (no commit — calibration output is gitignored)**

Report to the controller/user: the survival curves (how many survive at each threshold for the 123-name set and the 3,918 set), the excluded names at a candidate threshold for the small set (junk vs good), and the grouping health (OTHER fraction, flagged groups) — so the production `liquidity_pct_of_median` default can be chosen and set in `params.yaml`.

---

## After the plan (later)

1. Pick `liquidity_pct_of_median` from the calibration survival curves; set it in `params.yaml` (its own small commit).
2. Part A is pushable: reconstruct a clean branch off origin/main with the functional files
   (`src/data_intake.py`, `pipeline/01_download.py`, `params.yaml`, `tests/test_data_intake.py`) →
   PR (mirror PR #21/#22). Spec/plan + the calibration experiment stay local.
3. PR-2 (separate cycle): remove the technical filter (delete step 3) + rewire step 4 to optimise
   over the pruned universe.

---

## Self-Review Notes

- **Spec coverage:** parallel batch download Close+Volume (Tasks 3–4); load hygiene incl. BOM (Task 1); market_key ISIN/suffix/OTHER (Task 2); avg dollar-volume + relative-within-group filter with small-group keep-and-flag + zero-median handling (Task 5); grouping-health diagnostic + OTHER-fraction warning (Tasks 6–7); early placement / pruned 01 outputs + 01_liquidity audit (Task 7); config keys incl. min_group_size + topn-off (Task 7); calibration sweep over isins_list (3,918) and colombia+global (123) with names + grouping health (Tasks 8–9); tests ship with Part A (all tasks). Covered.
- **Out-of-scope respected:** no technical-filter removal / step-4 rewire (PR-2); no ISIN→ticker; no FX; topn config present but unused (null).
- **Type consistency:** `liquidity_filter` returns a detail DataFrame `[avg_dollar_volume, market_group, kept, flag]` consumed identically by `grouping_health`, `main`, and `sweep_thresholds` (Tasks 5–8); cfg keys (`batch_size`, `download_workers`, `download_timeout`, `liquidity_window`, `liquidity_pct_of_median`, `liquidity_min_group_size`) defined in Task 7 `params.yaml` and read in Tasks 7–8; `download_all(tickers, cfg, download_fn=)` seam consistent (Tasks 4, 7, 8).
- **Note:** `download_batch` reads `cfg['days_of_data']`/`interval`/`period_freq`/`download_timeout` — all already in cfg (`period_freq` is derived by `load_config`). Network funcs are smoke-verified, not unit-tested; the pure transforms and the `download_all` concat (via DI seam) are unit-tested.
```
