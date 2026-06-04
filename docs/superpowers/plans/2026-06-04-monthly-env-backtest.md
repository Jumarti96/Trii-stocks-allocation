# Monthly Environment-Robustness Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated monthly-data fetch plus a backtest driver that runs `backtest_allocation.run_backtest` for horizons {1,6} × seeds {0,100} with arms `current / (parametric,4.0) / (parametric,8.0) / equal_weight`, the technical filter disabled, and reports realized metrics as mean ± cross-seed spread per horizon.

**Architecture:** Two new experiment files reusing `experiments/backtest_allocation.py` unchanged: `experiments/fetch_monthly_data.py` (one-time monthly download into `experiments/data_monthly/`, never touching `data/` or `params.yaml`) and `experiments/monthly_env_backtest.py` (the driver + cross-seed aggregation + reporting). Pure functions with dependency-injection seams so the loop is unit-testable without torch or the network.

**Tech Stack:** Python, numpy, pandas, scikit-learn (LedoitWolf, via run_backtest), yfinance (fetch only), pytest. Torch only at real-run time via run_backtest's lazy import (stubbed in tests).

Spec: `docs/superpowers/specs/2026-06-04-monthly-env-backtest-design.md`.

**Conventions:**
- Test runner: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`
- Experiment-only and **local** per the push-scope rule — do NOT push; commit to the feature branch.
- No AI attribution in commit messages.

---

## File Structure

- Create: `experiments/fetch_monthly_data.py` — monthly download (pure clean transform + IO main).
- Create: `experiments/monthly_env_backtest.py` — driver, aggregation, formatting, output, CLI.
- Create: `tests/test_fetch_monthly_data.py` — pure-transform test (no network).
- Create: `tests/test_monthly_env_backtest.py` — torch-free driver tests via stubs.
- Reuse (no edits): `experiments/backtest_allocation.py`, `pipeline/01_download.py`, `pipeline/config.py`.

---

### Task 1: `fetch_monthly_data.py` — pure clean transform + download/main

The pure `clean_to_prices_returns` mirrors `01_download.py`'s clean logic (drop >15%-missing tickers, ffill/bfill, returns, string index) but takes the raw frame as an argument so it is testable without the network. The redundant date re-trim from `01_download` is dropped (the download is already date-bounded). `download_monthly_prices` + `main` do the network IO and are verified in the Task 8 smoke run.

**Files:**
- Create: `experiments/fetch_monthly_data.py`
- Test: `tests/test_fetch_monthly_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetch_monthly_data.py`:

```python
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import fetch_monthly_data as fmd


def test_clean_drops_missing_and_builds_returns():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    raw = pd.DataFrame({
        "A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "B": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
        "C": [5.0, np.nan, 7.0, 8.0, 9.0, 10.0],   # 1 NaN of 6 (>15%) -> dropped
    }, index=idx)

    prices, rets = fmd.clean_to_prices_returns(raw, period_freq="M")

    assert list(prices.columns) == ["A", "B"]        # C dropped
    assert list(rets.columns) == ["A", "B"]
    assert len(rets) == 5                             # one row lost to pct_change
    assert rets.index[0] == "2020-02"                # string YYYY-MM index
    assert rets["A"].iloc[0] == np.float64(11.0 / 10.0 - 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_fetch_monthly_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch_monthly_data'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/fetch_monthly_data.py`:

```python
"""
Fetch MONTHLY price/return data into an experiment-local directory, fully isolated from the
weekly production pipeline (params.yaml and data/ are never touched).

Mirrors pipeline/01_download.py's download + clean logic but forces interval=1mo and writes to
experiments/data_monthly/. Run once; reused by the monthly environment-robustness backtest.

Run:  python experiments/fetch_monthly_data.py
Outputs:
    experiments/data_monthly/01_prices.csv
    experiments/data_monthly/01_returns.csv

See docs/superpowers/specs/2026-06-04-monthly-env-backtest-design.md.
"""

import datetime
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import load_config, BASE_DIR as CFG_BASE_DIR

MONTHLY_DIR = os.path.join(BASE_DIR, "experiments", "data_monthly")
MONTHLY_PRICES = os.path.join(MONTHLY_DIR, "01_prices.csv")
MONTHLY_RETURNS = os.path.join(MONTHLY_DIR, "01_returns.csv")


def combine_duplicate_rows(df):
    """Keep first non-null value when the source returns duplicate rows for a period."""
    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else np.nan
    return df.groupby(df.index).agg(first_non_null)


def clean_to_prices_returns(stocks_raw, period_freq, missing_frac=0.15):
    """Clean a raw Close frame into (prices, returns) with string period-end indices.

    Mirrors 01_download.py: convert the DatetimeIndex to periods, collapse duplicate rows,
    drop tickers with >= missing_frac missing, forward/back-fill, compute per-period returns,
    and stringify the index (weekly 'a/b' -> 'b'; monthly stays 'YYYY-MM'). Pure (no network,
    no today()-based trim), so it is unit-testable.
    """
    raw = stocks_raw.copy()
    raw.index = raw.index.to_period(freq=period_freq)
    stocks = combine_duplicate_rows(raw).sort_index()
    keep = stocks.columns[stocks.isna().sum() < stocks.shape[0] * missing_frac]
    stocks = stocks[keep].ffill().bfill()
    rets = stocks.pct_change().iloc[1:]
    stocks.index = stocks.index.astype("str").str.split("/").str[-1]
    rets.index = rets.index.astype("str").str.split("/").str[-1]
    return stocks, rets


def download_monthly_prices(ticker_list, days_of_data):
    """Download monthly adjusted-close prices from yfinance (network)."""
    import yfinance as yf
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_of_data)
    raw = yf.download(
        ticker_list, interval="1mo", start=start_date, end=end_date, auto_adjust=True
    )["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    return raw


def main():
    cfg = load_config()
    csv_files = glob.glob(os.path.join(CFG_BASE_DIR, "stock_tickers", "*.csv"))
    ticker_list = list({
        ticker
        for csv_file in csv_files
        for ticker in pd.read_csv(csv_file, header=None)[0].tolist()
    })
    print(f"Loaded {len(ticker_list)} unique tickers from {len(csv_files)} CSV file(s).")

    raw = download_monthly_prices(ticker_list, cfg["days_of_data"])
    prices, rets = clean_to_prices_returns(raw, period_freq="M")

    os.makedirs(MONTHLY_DIR, exist_ok=True)
    prices.to_csv(MONTHLY_PRICES)
    rets.to_csv(MONTHLY_RETURNS)

    # Sanity: how does the monthly universe compare to the weekly production one?
    try:
        weekly_cols = pd.read_csv(
            os.path.join(CFG_BASE_DIR, "data", "01_returns.csv"), index_col=0, nrows=0
        ).columns
        overlap = len([c for c in rets.columns if c in weekly_cols])
        print(f"Monthly universe: {rets.shape[1]} names | weekly: {len(weekly_cols)} | "
              f"overlap: {overlap}")
    except FileNotFoundError:
        print(f"Monthly universe: {rets.shape[1]} names (weekly file not found for overlap)")

    print(f"Monthly points: {rets.shape[0]}")
    print(f"Saved: {MONTHLY_PRICES}\n       {MONTHLY_RETURNS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_fetch_monthly_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/fetch_monthly_data.py tests/test_fetch_monthly_data.py
git commit -m "Add monthly data fetch (isolated, pure clean transform)"
```

---

### Task 2: `monthly_env_backtest.py` scaffold + `build_monthly_cfg` + `select_all`

`build_monthly_cfg` turns the weekly production cfg into a monthly cfg for one horizon; `select_all` is the filter-disabling selection seam.

**Files:**
- Create: `experiments/monthly_env_backtest.py`
- Test: `tests/test_monthly_env_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_monthly_env_backtest.py`:

```python
import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import monthly_env_backtest as meb


def _weekly_cfg():
    return {
        "rf_rate": 0.11, "max_weight": 0.5, "min_weight": 0.05,
        "periods_per_year": 54, "interval": "1wk",
        "time_window": 54, "periods_to_forecast": 4,
    }


def test_build_monthly_cfg_sets_monthly_fields():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=6)
    assert cfg["interval"] == "1mo"
    assert cfg["periods_per_year"] == 12
    assert cfg["time_window"] == 12
    assert cfg["periods_to_forecast"] == 6
    assert cfg["period_freq"] == "M"
    assert cfg["future_freq"] == "MS"
    assert cfg["rf_period"] == pytest.approx((1 + 0.11) ** (1 / 12) - 1)
    # original weekly cfg is not mutated
    assert _weekly_cfg()["periods_per_year"] == 54


def test_select_all_returns_every_name():
    rets = pd.DataFrame(np.zeros((3, 4)), columns=["A", "B", "C", "D"])
    assert meb.select_all(None, rets, {}) == ["A", "B", "C", "D"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'monthly_env_backtest'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/monthly_env_backtest.py`:

```python
"""
Monthly environment-robustness backtest: run the realized walk-forward backtest in a MONTHLY
regime at short (1-month) and long (6-month) horizons, comparing the allocation methods
current (msr) / parametric s4 / parametric s8 against equal_weight, with the technical filter
DISABLED. Reports realized metrics as mean +/- cross-seed std per horizon.

Reuses experiments/backtest_allocation.run_backtest unchanged; reads MONTHLY data fetched by
experiments/fetch_monthly_data.py. This is a realized out-of-sample test (the verdict is realized
Sharpe etc.), complementary to the weekly multi-seed backtest. In-sample caveats do not apply;
the few-block 6-month horizon is directional, not a verdict.

Run:  python experiments/monthly_env_backtest.py --horizons 1,6 --seeds 0,100 \
        --oos-periods 60 --n-runs 75 --mc-draws 1000 --spreads 4,8
Requires experiments/data_monthly/01_{prices,returns}.csv (run fetch_monthly_data.py first).

See docs/superpowers/specs/2026-06-04-monthly-env-backtest-design.md.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
sys.path.insert(0, HERE)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import load_config
from backtest_allocation import run_backtest, summarize_arm, write_backtest_outputs
from fetch_monthly_data import MONTHLY_PRICES, MONTHLY_RETURNS

METRIC_COLS = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_dd",
               "mean_turnover", "avg_names", "hit_rate"]


def build_monthly_cfg(weekly_cfg, horizon):
    """Copy the weekly production cfg and override the monthly + horizon fields.

    Sets interval/periods_per_year/time_window/period_freq/future_freq/date_offset to monthly,
    periods_to_forecast to the horizon, and recomputes the per-period risk-free rate for ppy=12.
    Filter/MA/weight params are left untouched (the filter is bypassed via select_all, not config).
    """
    cfg = dict(weekly_cfg)
    cfg["interval"] = "1mo"
    cfg["periods_per_year"] = 12
    cfg["time_window"] = 12
    cfg["periods_to_forecast"] = horizon
    cfg["period_freq"] = "M"
    cfg["future_freq"] = "MS"
    cfg["date_offset"] = pd.DateOffset(months=1)
    cfg["rf_period"] = (1 + cfg["rf_rate"]) ** (1 / 12) - 1
    return cfg


def select_all(prices, rets, cfg):
    """Filter-disabling selection seam: every name is eligible (no technical gate)."""
    return list(rets.columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/monthly_env_backtest.py tests/test_monthly_env_backtest.py
git commit -m "Add monthly-env build_monthly_cfg and filter-off select seam"
```

---

### Task 3: `run_monthly_env` core loop

Drive `run_backtest` over horizons × seeds with the 4-arm set and the filter off.

**Files:**
- Modify: `experiments/monthly_env_backtest.py`
- Test: `tests/test_monthly_env_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monthly_env_backtest.py`:

```python
def _stub_runs_fn(n_stocks):
    """Deterministic per-run forecasts: list of (periods_to_forecast x n_stocks) frames."""
    rng = np.random.default_rng(0)

    def _runs_fn(rets, cfg, n_runs=None, verbose=False):
        h = cfg["periods_to_forecast"]
        return [pd.DataFrame(rng.normal(0.01, 0.005, size=(h, n_stocks)),
                             columns=list(rets.columns)) for _ in range(n_runs)]
    return _runs_fn


def _mean_period_mu(preds_df):
    return preds_df.mean(axis=0)


def test_run_monthly_env_structure_and_arms():
    cols = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.04, size=(30, 6)), columns=cols)
    prices = pd.DataFrame(100 * (1 + rets).cumprod().values, columns=cols)

    out = meb.run_monthly_env(
        prices, rets, _weekly_cfg(),
        horizons=[1], seeds=[0], oos_periods=6, n_runs=2, mc_draws=8, spreads=[4.0],
        runs_fn=_stub_runs_fn(6), period_mu_fn=_mean_period_mu, seed_fn=lambda s: None,
    )

    assert set(out.keys()) == {1}
    assert out[1]["cfg"]["periods_to_forecast"] == 1
    res = out[1]["per_seed"][0]
    labels = [k for k in res if k != "rebalance_index"]
    assert labels == ["current", "parametric_s4", "equal_weight"]
    # filter off -> equal_weight holds the full 6-name universe
    assert res["equal_weight"]["n_held"][0] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py::test_run_monthly_env_structure_and_arms -v`
Expected: FAIL — `AttributeError: ... 'run_monthly_env'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/monthly_env_backtest.py`:

```python
def run_monthly_env(prices, rets, weekly_cfg, horizons, seeds, oos_periods,
                    n_runs, mc_draws, spreads, runs_fn=None, period_mu_fn=None,
                    seed_fn=None):
    """Run the monthly backtest over horizons x seeds; filter disabled via select_all.

    Arms = current + one parametric arm per spread + equal_weight. For each horizon a monthly cfg
    is built (periods_to_forecast = rebalance_every = horizon) and run_backtest is called once per
    seed. The *_fn args are dependency-injection seams passed straight through to run_backtest so
    the loop runs without torch. Returns {horizon: {"cfg": cfg_h, "per_seed": {seed: results}}}.
    """
    arms = ["current"] + [("parametric", s) for s in spreads] + ["equal_weight"]
    out = {}
    for horizon in horizons:
        cfg_h = build_monthly_cfg(weekly_cfg, horizon)
        per_seed = {}
        for seed in seeds:
            per_seed[seed] = run_backtest(
                prices, rets, cfg_h, oos_periods=oos_periods, rebalance_every=horizon,
                n_runs=n_runs, mc_draws=mc_draws, spreads=spreads, seed=seed, arms=arms,
                select_fn=select_all, runs_fn=runs_fn, period_mu_fn=period_mu_fn,
                seed_fn=seed_fn,
            )
        out[horizon] = {"cfg": cfg_h, "per_seed": per_seed}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/monthly_env_backtest.py tests/test_monthly_env_backtest.py
git commit -m "Add monthly-env run loop over horizons x seeds"
```

---

### Task 4: `aggregate_across_seeds`

Reduce one horizon's per-seed `run_backtest` results to a per-arm mean ± cross-seed std table.

**Files:**
- Modify: `experiments/monthly_env_backtest.py`
- Test: `tests/test_monthly_env_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monthly_env_backtest.py`:

```python
def _fake_arm(block_returns):
    n = len(block_returns)
    return {"block_returns": block_returns,
            "turnover": [None] + [0.1] * (n - 1),
            "n_held": [3] * n,
            "weights": [pd.Series([1.0], index=["A"])] * n,
            "dates": [f"2020-{i+1:02d}" for i in range(n)]}


def test_aggregate_across_seeds_mean_and_std():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=1)
    seed0 = {"current": _fake_arm([0.02, 0.02, 0.02]),
             "equal_weight": _fake_arm([0.01, 0.01, 0.01]),
             "rebalance_index": [10, 11, 12]}
    seed1 = {"current": _fake_arm([0.04, 0.04, 0.04]),
             "equal_weight": _fake_arm([0.01, 0.01, 0.01]),
             "rebalance_index": [10, 11, 12]}

    agg = meb.aggregate_across_seeds({0: seed0, 1: seed1}, cfg, rebalance_every=1)

    assert agg["n_blocks"] == 3
    table = agg["table"]
    # constant-return arms -> zero vol -> NaN sharpe; check cum_return mean/std instead
    cur_mean = table.loc["current", "cum_return_mean"]
    cur_std = table.loc["current", "cum_return_std"]
    # seed0 cum = 1.02^3-1 ~ 0.0612; seed1 cum = 1.04^3-1 ~ 0.1249
    assert cur_mean == pytest.approx((0.061208 + 0.124864) / 2, abs=1e-4)
    assert cur_std == pytest.approx(abs(0.061208 - 0.124864) / 2, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py::test_aggregate_across_seeds_mean_and_std -v`
Expected: FAIL — `AttributeError: ... 'aggregate_across_seeds'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/monthly_env_backtest.py`:

```python
def aggregate_across_seeds(per_seed, cfg, rebalance_every):
    """Per-arm realized metrics aggregated across seeds as mean and population std.

    per_seed: {seed: run_backtest result}. Uses backtest_allocation.summarize_arm with
    blocks_per_year = periods_per_year / rebalance_every and the per-block rf. Returns
    {"table": DataFrame indexed by arm with <metric>_mean/<metric>_std columns,
    "n_blocks": int, "metric_cols": METRIC_COLS}.
    """
    blocks_per_year = cfg["periods_per_year"] / rebalance_every
    rf_block = (1.0 + cfg["rf_period"]) ** rebalance_every - 1.0
    seeds = list(per_seed.keys())
    labels = [k for k in per_seed[seeds[0]] if k != "rebalance_index"]

    per_seed_metrics = {
        s: {lab: summarize_arm(per_seed[s][lab]["block_returns"],
                               per_seed[s][lab]["turnover"],
                               per_seed[s][lab]["n_held"],
                               blocks_per_year, rf_block)
            for lab in labels}
        for s in seeds
    }

    rows = []
    for lab in labels:
        row = {"arm": lab}
        for c in METRIC_COLS:
            vals = np.array([per_seed_metrics[s][lab][c] for s in seeds], dtype=float)
            row[f"{c}_mean"] = float(np.nanmean(vals))
            row[f"{c}_std"] = float(np.nanstd(vals))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("arm")
    n_blocks = len(per_seed[seeds[0]][labels[0]]["block_returns"])
    return {"table": table, "n_blocks": n_blocks, "metric_cols": METRIC_COLS}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/monthly_env_backtest.py tests/test_monthly_env_backtest.py
git commit -m "Add monthly-env cross-seed aggregation"
```

---

### Task 5: `format_monthly_summary`

Render the per-horizon mean ± std tables with prominent block counts, the few-block caveat, and the equal_weight / filter-off notes.

**Files:**
- Modify: `experiments/monthly_env_backtest.py`
- Test: `tests/test_monthly_env_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monthly_env_backtest.py`:

```python
def test_format_monthly_summary_content():
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=6)
    seed0 = {"current": _fake_arm([0.02] * 9),
             "equal_weight": _fake_arm([0.01] * 9),
             "rebalance_index": list(range(9))}
    agg = meb.aggregate_across_seeds({0: seed0}, cfg, rebalance_every=6)
    text = meb.format_monthly_summary({6: agg})

    assert "horizon 6-month" in text
    assert "blocks: 9" in text
    assert "DIRECTIONAL" in text          # <15 blocks caveat fires
    assert "current" in text
    assert "equal_weight" in text
    assert "filter DISABLED" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py::test_format_monthly_summary_content -v`
Expected: FAIL — `AttributeError: ... 'format_monthly_summary'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/monthly_env_backtest.py`:

```python
def format_monthly_summary(agg_by_horizon, few_blocks=15):
    """Per-horizon realized metric tables (mean +/- cross-seed std) with block counts + caveats."""
    lines = ["Monthly environment-robustness backtest (mean +/- cross-seed std)", ""]
    for horizon in sorted(agg_by_horizon):
        a = agg_by_horizon[horizon]
        lines.append(f"== horizon {horizon}-month | blocks: {a['n_blocks']} ==")
        if a["n_blocks"] < few_blocks:
            lines.append("  (few blocks -> realized stats are DIRECTIONAL, not a verdict)")
        for arm, r in a["table"].iterrows():
            cells = "  ".join(
                f"{c}={r[f'{c}_mean']:.4f}+/-{r[f'{c}_std']:.4f}" for c in a["metric_cols"]
            )
            lines.append(f"  {arm:<16} {cells}")
        lines.append("")
    lines.append(
        "Note: technical filter DISABLED (full universe). equal_weight = 1/N anchor; an optimizer "
        "arm that fails to beat it adds no value in this regime. This filter-off MONTHLY result is "
        "confounded with the weekly->monthly change -- NOT clean evidence for removing the filter "
        "from production (that needs its own weekly with-vs-without A/B)."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/monthly_env_backtest.py tests/test_monthly_env_backtest.py
git commit -m "Add monthly-env summary formatter"
```

---

### Task 6: `write_monthly_outputs`

Write per-(horizon, seed) raw artifacts (reusing `write_backtest_outputs`), per-horizon aggregated tables, and the summary text.

**Files:**
- Modify: `experiments/monthly_env_backtest.py`
- Test: `tests/test_monthly_env_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monthly_env_backtest.py`:

```python
def test_write_monthly_outputs_creates_files(tmp_path):
    cfg = meb.build_monthly_cfg(_weekly_cfg(), horizon=1)
    seed0 = {"current": _fake_arm([0.02, 0.03, 0.01]),
             "equal_weight": _fake_arm([0.01, 0.01, 0.02]),
             "rebalance_index": [10, 11, 12]}
    agg = meb.aggregate_across_seeds({0: seed0}, cfg, rebalance_every=1)
    run_out = {1: {"cfg": cfg, "per_seed": {0: seed0}}}

    outdir = str(tmp_path / "monthly_env")
    paths = meb.write_monthly_outputs(run_out, {1: agg}, outdir)

    assert os.path.exists(paths["summary"])
    assert os.path.exists(os.path.join(outdir, "monthly_table_h1.csv"))
    assert os.path.exists(os.path.join(outdir, "h1_seed0", "backtest_summary.txt"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py::test_write_monthly_outputs_creates_files -v`
Expected: FAIL — `AttributeError: ... 'write_monthly_outputs'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/monthly_env_backtest.py`:

```python
def write_monthly_outputs(run_out, agg_by_horizon, outdir):
    """Write per-(horizon, seed) raw CSVs, per-horizon aggregated tables, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    paths = {"summary": os.path.join(outdir, "monthly_env_summary.txt")}

    for horizon, hd in run_out.items():
        cfg_h = hd["cfg"]
        for seed, res in hd["per_seed"].items():
            sub = os.path.join(outdir, f"h{horizon}_seed{seed}")
            write_backtest_outputs(res, cfg_h, horizon, sub)
        tpath = os.path.join(outdir, f"monthly_table_h{horizon}.csv")
        agg_by_horizon[horizon]["table"].to_csv(tpath)
        paths[f"table_h{horizon}"] = tpath

    with open(paths["summary"], "w") as f:
        f.write(format_monthly_summary(agg_by_horizon))
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/monthly_env_backtest.py tests/test_monthly_env_backtest.py
git commit -m "Add monthly-env output writers"
```

---

### Task 7: `build_arg_parser` + `main`

Wire the CLI: load weekly cfg, read monthly data, run, aggregate per horizon, write, print.

**Files:**
- Modify: `experiments/monthly_env_backtest.py`
- Test: `tests/test_monthly_env_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_monthly_env_backtest.py`:

```python
def test_arg_parser_parses_lists():
    parser = meb.build_arg_parser()
    args = parser.parse_args(
        ["--horizons", "1,6", "--seeds", "0,100", "--oos-periods", "60",
         "--n-runs", "75", "--mc-draws", "1000", "--spreads", "4,8"]
    )
    assert [int(x) for x in args.horizons.split(",")] == [1, 6]
    assert [int(x) for x in args.seeds.split(",")] == [0, 100]
    assert [float(x) for x in args.spreads.split(",")] == [4.0, 8.0]
    assert args.oos_periods == 60
    assert args.n_runs == 75
    assert args.mc_draws == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py::test_arg_parser_parses_lists -v`
Expected: FAIL — `AttributeError: ... 'build_arg_parser'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/monthly_env_backtest.py`:

```python
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--horizons", type=str, default="1,6",
                        help="Comma-separated forecast horizons in months (default 1,6)")
    parser.add_argument("--seeds", type=str, default="0,100",
                        help="Comma-separated base seeds (default 0,100)")
    parser.add_argument("--oos-periods", type=int, default=60,
                        help="Out-of-sample months held out (default 60)")
    parser.add_argument("--n-runs", type=int, default=75,
                        help="Transformer runs per rebalance (default 75)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="K parametric Monte-Carlo draws (default 1000)")
    parser.add_argument("--spreads", type=str, default="4,8",
                        help="Comma-separated parametric spreads s (default 4,8)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "monthly_env"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    weekly_cfg = load_config()
    prices = pd.read_csv(MONTHLY_PRICES, index_col=0)
    rets = pd.read_csv(MONTHLY_RETURNS, index_col=0)

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    spreads = [float(x) for x in args.spreads.split(",")]

    print(f"Monthly universe: {rets.shape[1]} names | months: {rets.shape[0]} | "
          f"horizons={horizons} | seeds={seeds} | oos={args.oos_periods} | "
          f"n_runs={args.n_runs} | K={args.mc_draws} | spreads={spreads} | filter=OFF",
          flush=True)

    run_out = run_monthly_env(
        prices, rets, weekly_cfg, horizons=horizons, seeds=seeds,
        oos_periods=args.oos_periods, n_runs=args.n_runs, mc_draws=args.mc_draws,
        spreads=spreads,
    )

    agg_by_horizon = {
        h: aggregate_across_seeds(run_out[h]["per_seed"], run_out[h]["cfg"], h)
        for h in horizons
    }
    paths = write_monthly_outputs(run_out, agg_by_horizon, args.outdir)

    print()
    print(format_monthly_summary(agg_by_horizon))
    print("\nSaved:")
    for p in sorted(set(paths.values())):
        print(f"       {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test file**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_monthly_env_backtest.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/monthly_env_backtest.py tests/test_monthly_env_backtest.py
git commit -m "Add monthly-env CLI orchestration"
```

---

### Task 8: Full-suite check + fetch + tiny real smoke run

Confirm the repo is green, fetch real monthly data once, and run a trivial real backtest before the production launch.

**Files:** none (verification only).

- [ ] **Step 1: Run the whole test suite**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest -q`
Expected: PASS — all prior tests plus the new fetch (1) and monthly-env (7) tests green.

- [ ] **Step 2: Fetch real monthly data (network, run once)**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/fetch_monthly_data.py`
Expected: prints the ticker count, the "Monthly universe / weekly / overlap" line, the monthly point count (~120), and "Saved:" with both paths. Confirm `experiments/data_monthly/01_prices.csv` and `01_returns.csv` exist. **Report the actual month count and universe size to the controller** — if months differ a lot from ~120, the oos=60 split may need revisiting before the production run.

- [ ] **Step 3: Tiny real smoke backtest (NOT the production run)**

Run:
```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/monthly_env_backtest.py --horizons 1 --seeds 0 --oos-periods 6 --n-runs 3 --mc-draws 20 --spreads 4 --outdir experiments/results/monthly_env_smoke
```
Expected: prints the universe banner, the horizon-1 table (arms current / parametric_s4 / equal_weight), and a "Saved:" list. Confirm `experiments/results/monthly_env_smoke/monthly_env_summary.txt` exists, the table has all three arms, and `equal_weight`'s `avg_names` equals the full universe size (filter is off). At 1 seed the `_std` columns are 0.0 — expected.

- [ ] **Step 4: Report and proceed**

No commit needed (smoke output and monthly data are gitignored under `experiments/`). Report the fetch diagnostics and smoke outcome to the controller, who decides the production launch.

---

## Production run (after implementation + smoke pass)

Launch in the background (~10–11 h: ~5 h/seed, K=1000 optimization dominant):

```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/monthly_env_backtest.py \
  --horizons 1,6 --seeds 0,100 --oos-periods 60 --n-runs 75 --mc-draws 1000 --spreads 4,8 \
  --outdir experiments/results/monthly_env
```

Then read `experiments/results/monthly_env/monthly_env_summary.txt`: per horizon, does any optimizer arm (current / s4 / s8) beat `equal_weight` on realized Sharpe, and how does the s4-vs-s8-vs-current ranking compare to the weekly result? Read the 1-month horizon as the primary signal (~60 blocks) and the 6-month as directional (~10 blocks). Cross-seed std indicates whether the ranking is trustworthy.

---

## Self-Review Notes

- **Spec coverage:** isolated monthly fetch (Task 1); override cfg (Task 2 `build_monthly_cfg`); filter disabled (Task 2 `select_all`, wired in Task 3); horizons {1,6} × seeds {0,100} × 4 arms via run_backtest reused unchanged (Task 3 + Task 7 defaults); cross-seed mean±std (Task 4); block counts + few-block caveat + equal_weight/filter-off notes (Task 5); per-(horizon,seed) raw + aggregated tables + summary outputs (Task 6); CLI defaults n_runs=75/K=1000/oos=60 (Task 7); cost + run command + fetch diagnostics (Tasks 7–8 + Production section). Covered.
- **Out-of-scope respected:** no params.yaml/data/ mutation (fetch writes only experiments/data_monthly/), no pipeline edits, no backtest_allocation edits, no net-of-cost computation (turnover saved for the free follow-up), no with-vs-without-filter A/B.
- **Type consistency:** arm labels `current` / `parametric_s4` / `parametric_s8` / `equal_weight` come from `backtest_allocation.label_of` and are used consistently across Tasks 3–6; `METRIC_COLS` defined in Task 2 and reused in Tasks 4–5; the `{horizon: {"cfg", "per_seed"}}` run_out shape and the `{"table","n_blocks","metric_cols"}` agg shape are consistent between `run_monthly_env`/`aggregate_across_seeds` (Tasks 3–4) and their consumers (Tasks 5–7) and the test fakes.
