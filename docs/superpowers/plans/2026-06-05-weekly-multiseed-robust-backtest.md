# Weekly Multi-Seed Robustness Backtest (Plan 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a weekly walk-forward backtest over horizons {4,24} × seeds {0,100,200} comparing current(msr)/s4/s8/equal_weight/sp500 (filter off), reusing `backtest_allocation.run_backtest` unchanged, with cheap post-hoc analytics: net-of-cost + break-even bps, and a paired block-bootstrap on the (s4−msr) difference.

**Architecture:** Two new experiment files reusing `experiments/backtest_allocation.py` (run_backtest, summarize_arm, annualized_stats, write_backtest_outputs) and `experiments/monthly_env_backtest.py` (aggregate_across_seeds, select_all) unchanged: `experiments/fetch_sp500_weekly.py` (isolated `^GSPC` fetch) and `experiments/weekly_robust_backtest.py` (driver + SP500 splice + analytics). Pure functions with DI seams so the loop is testable without torch or network.

**Tech Stack:** Python, numpy, pandas, scikit-learn (via run_backtest), yfinance (fetch only), pytest. Torch only at real-run time (stubbed in tests).

Spec: `docs/superpowers/specs/2026-06-05-weekly-multiseed-robust-backtest-design.md`.

**Conventions:**
- Test runner: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`
- Experiment-only and **local** — do NOT push; commit to the feature branch. No AI attribution in commit messages.
- The production weekly data `data/01_{prices,returns}.csv` is READ (never written).

---

## File Structure

- Create: `experiments/fetch_sp500_weekly.py` — `^GSPC` weekly fetch (pure transform + IO main).
- Create: `experiments/weekly_robust_backtest.py` — driver, SP500 splice, analytics, CLI.
- Create: `tests/test_fetch_sp500_weekly.py` — pure-transform test (no network).
- Create: `tests/test_weekly_robust_backtest.py` — torch-free driver/analytics tests via stubs.
- Reuse (no edits): `experiments/backtest_allocation.py`, `experiments/monthly_env_backtest.py`, `pipeline/config.py`.

---

### Task 1: `fetch_sp500_weekly.py` — pure weekly-returns transform + fetch/main

**Files:**
- Create: `experiments/fetch_sp500_weekly.py`
- Test: `tests/test_fetch_sp500_weekly.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetch_sp500_weekly.py`:

```python
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import fetch_sp500_weekly as fsp


def test_weekly_returns_from_close():
    idx = pd.date_range("2020-01-05", periods=4, freq="W")
    close = pd.Series([100.0, 110.0, 121.0, 121.0], index=idx)
    rets = fsp.weekly_returns_from_close(close, period_freq="W")
    assert len(rets) == 3                       # one row lost to pct_change
    assert rets.iloc[0] == np.float64(0.10)
    assert rets.iloc[1] == np.float64(0.10)
    assert isinstance(rets.index[0], str)       # period-end string index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_fetch_sp500_weekly.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch_sp500_weekly'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/fetch_sp500_weekly.py`:

```python
"""
Fetch ^GSPC (S&P 500) WEEKLY returns into an experiment-local dir, isolated from the production
pipeline (never touches data/ or params.yaml). Used as a market benchmark arm in the weekly
multi-seed robustness backtest.

Run:  python experiments/fetch_sp500_weekly.py
Output: experiments/data_weekly_bench/sp500_returns.csv

See docs/superpowers/specs/2026-06-05-weekly-multiseed-robust-backtest-design.md.
"""

import datetime
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from config import load_config

BENCH_DIR = os.path.join(BASE_DIR, "experiments", "data_weekly_bench")
SP500_RETURNS = os.path.join(BENCH_DIR, "sp500_returns.csv")


def weekly_returns_from_close(close, period_freq="W"):
    """Convert a weekly Close Series (DatetimeIndex) to period-end-string-indexed returns.

    Mirrors the universe's index convention (PeriodIndex -> 'a/b' -> 'b' end-date string) so the
    SP500 returns align positionally with data/01_returns.csv. Pure (no network).
    """
    s = close.copy()
    s.index = s.index.to_period(freq=period_freq)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    rets = s.pct_change().iloc[1:]
    rets.index = rets.index.astype("str").str.split("/").str[-1]
    return rets


def download_sp500_close(days_of_data):
    """Download ^GSPC weekly adjusted close (network). Returns a Series."""
    import yfinance as yf
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_of_data)
    raw = yf.download(
        "^GSPC", interval="1wk", start=start_date, end=end_date, auto_adjust=True
    )["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    return raw


def main():
    cfg = load_config()
    close = download_sp500_close(cfg["days_of_data"])
    rets = weekly_returns_from_close(close, period_freq="W")
    os.makedirs(BENCH_DIR, exist_ok=True)
    rets.to_frame("sp500").to_csv(SP500_RETURNS, index_label="date")
    print(f"SP500 weekly points: {len(rets)} | range {rets.index[0]}..{rets.index[-1]}")
    print(f"Saved: {SP500_RETURNS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_fetch_sp500_weekly.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/fetch_sp500_weekly.py tests/test_fetch_sp500_weekly.py
git commit -m "Add SP500 weekly benchmark fetch (isolated)"
```

---

### Task 2: `weekly_robust_backtest.py` scaffold + `build_weekly_cfg`

**Files:**
- Create: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_weekly_robust_backtest.py`:

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

import weekly_robust_backtest as wrb


def _weekly_cfg():
    return {
        "rf_rate": 0.11, "max_weight": 0.5, "min_weight": 0.05,
        "periods_per_year": 54, "interval": "1wk",
        "time_window": 54, "periods_to_forecast": 4,
        "rf_period": (1 + 0.11) ** (1 / 54) - 1,
    }


def test_build_weekly_cfg_changes_only_horizon():
    cfg = wrb.build_weekly_cfg(_weekly_cfg(), horizon=24)
    assert cfg["periods_to_forecast"] == 24
    assert cfg["periods_per_year"] == 54        # unchanged (still weekly)
    assert cfg["interval"] == "1wk"
    assert cfg["time_window"] == 54
    # input not mutated
    original = _weekly_cfg()
    wrb.build_weekly_cfg(original, horizon=24)
    assert original["periods_to_forecast"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'weekly_robust_backtest'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/weekly_robust_backtest.py`:

```python
"""
Weekly multi-seed robustness backtest (Plan 1): realized walk-forward over horizons {4,24} x seeds
{0,100,200} comparing current(msr) / parametric s4 / parametric s8 / equal_weight / sp500, with the
technical filter DISABLED, in the weekly production environment. Reuses
backtest_allocation.run_backtest unchanged; adds an SP500 benchmark arm and cheap post-hoc analytics
(net-of-cost + break-even bps, paired block-bootstrap on s4-vs-msr).

This is the seed-robust realized verdict; it stays an experiment (does not change the deployment).
The 24-week horizon is directional (~10 blocks). SP500 is a USD market anchor (currency/composition
mismatch vs the global universe). Filter-off here is forward-looking, not a prod-filter evaluation.

Run:  python experiments/weekly_robust_backtest.py --horizons 4,24 --seeds 0,100,200 \
        --oos-periods 250 --n-runs 75 --mc-draws 1000 --spreads 4,8 --cost-bps 10,25
Requires data/01_{prices,returns}.csv (production weekly) and experiments/data_weekly_bench/
sp500_returns.csv (run fetch_sp500_weekly.py first).

See docs/superpowers/specs/2026-06-05-weekly-multiseed-robust-backtest-design.md.
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

from config import load_config, PATHS
from backtest_allocation import (
    run_backtest, summarize_arm, annualized_stats, write_backtest_outputs,
)
from monthly_env_backtest import aggregate_across_seeds, select_all
from fetch_sp500_weekly import SP500_RETURNS

METRIC_COLS = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_dd",
               "mean_turnover", "avg_names", "hit_rate"]


def build_weekly_cfg(weekly_cfg, horizon):
    """Copy the weekly production cfg, changing only the forecast horizon.

    Everything stays weekly (periods_per_year=54, interval=1wk, time_window, rf_period); only
    periods_to_forecast (= the rebalance block length) becomes `horizon`. Does not mutate the input.
    """
    cfg = dict(weekly_cfg)
    cfg["periods_to_forecast"] = horizon
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add weekly-robust scaffold and build_weekly_cfg"
```

---

### Task 3: SP500 benchmark splice (`sp500_block_returns` + `splice_sp500_arm`)

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def test_sp500_block_returns_compounds_aligned_blocks():
    # universe index of 6 weekly dates; sp500 returns aligned to it
    rets_index = [f"2020-{i+1:02d}" for i in range(6)]
    sp = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.05], index=rets_index)
    # two non-overlapping 2-period blocks starting at t=2 and t=4
    blocks = wrb.sp500_block_returns(sp, rets_index, rebalance_index=[2, 4], rebalance_every=2)
    assert blocks[0] == pytest.approx((1 - 0.01) * (1 + 0.03) - 1)
    assert blocks[1] == pytest.approx((1 + 0.00) * (1 + 0.05) - 1)


def test_splice_sp500_arm_adds_benchmark_label():
    results = {"current": {"block_returns": [0.01, 0.02], "turnover": [None, 0.1],
                           "n_held": [5, 5], "weights": [], "dates": ["2020-05", "2020-06"]},
               "rebalance_index": [2, 4]}
    out = wrb.splice_sp500_arm(results, [0.015, 0.04], ["2020-05", "2020-06"])
    assert out["sp500"]["block_returns"] == [0.015, 0.04]
    assert out["sp500"]["n_held"] == [1, 1]
    assert out["sp500"]["turnover"][0] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -k sp500 -v`
Expected: FAIL — `AttributeError: ... 'sp500_block_returns'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def sp500_block_returns(sp500_rets, rets_index, rebalance_index, rebalance_every):
    """Compounded SP500 buy-and-hold return for each rebalance block.

    sp500_rets is aligned (reindexed) to the universe's date index, then for each rebalance start t
    the block return is the compounded SP500 return over rets_index[t:t+rebalance_every]. Returns a
    list aligned to rebalance_index. Pure (no network).
    """
    aligned = sp500_rets.reindex(rets_index).fillna(0.0).values
    out = []
    for t in rebalance_index:
        block = aligned[t:t + rebalance_every]
        out.append(float(np.prod(1.0 + block) - 1.0))
    return out


def splice_sp500_arm(results, sp500_block_rets, dates):
    """Add an 'sp500' benchmark arm to a run_backtest result dict (turnover ~0, one synthetic name).

    The benchmark holds the index, so it has no portfolio weights over the universe; we record a
    single synthetic 'SP500' name at weight 1.0 so the existing writers/aggregators treat it as a
    one-name arm. Mutates and returns `results`.
    """
    n = len(sp500_block_rets)
    results["sp500"] = {
        "block_returns": list(sp500_block_rets),
        "turnover": [None] + [0.0] * (n - 1),
        "n_held": [1] * n,
        "weights": [pd.Series([1.0], index=["SP500"]) for _ in range(n)],
        "dates": list(dates),
    }
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add SP500 benchmark block returns and arm splice"
```

---

### Task 4: `run_weekly_robust` core loop

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def _stub_runs_fn(n_stocks):
    rng = np.random.default_rng(0)

    def _runs_fn(rets, cfg, n_runs=None, verbose=False):
        h = cfg["periods_to_forecast"]
        return [pd.DataFrame(rng.normal(0.01, 0.005, size=(h, n_stocks)),
                             columns=list(rets.columns)) for _ in range(n_runs)]
    return _runs_fn


def _mean_period_mu(preds_df):
    return preds_df.mean(axis=0)


def test_run_weekly_robust_structure_and_sp500_arm():
    cols = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.03, size=(40, 6)), columns=cols)
    prices = pd.DataFrame(100 * (1 + rets).cumprod().values, columns=cols)
    sp500 = pd.Series(rng.normal(0.004, 0.02, size=40), index=list(rets.index))

    out = wrb.run_weekly_robust(
        prices, rets, sp500, _weekly_cfg(),
        horizons=[4], seeds=[0], oos_periods=8, n_runs=2, mc_draws=8, spreads=[4.0, 8.0],
        runs_fn=_stub_runs_fn(6), period_mu_fn=_mean_period_mu, seed_fn=lambda s: None,
    )

    res = out[4]["per_seed"][0]
    labels = [k for k in res if k != "rebalance_index"]
    assert labels == ["current", "parametric_s4", "parametric_s8", "equal_weight", "sp500"]
    assert res["equal_weight"]["n_held"][0] == 6        # filter off -> full universe
    assert res["sp500"]["n_held"][0] == 1
    assert len(res["sp500"]["block_returns"]) == len(res["current"]["block_returns"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py::test_run_weekly_robust_structure_and_sp500_arm -v`
Expected: FAIL — `AttributeError: ... 'run_weekly_robust'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def run_weekly_robust(prices, rets, sp500_rets, weekly_cfg, horizons, seeds, oos_periods,
                      n_runs, mc_draws, spreads, runs_fn=None, period_mu_fn=None, seed_fn=None):
    """Run the weekly backtest over horizons x seeds; filter off; splice the SP500 benchmark arm.

    Arms = current + one parametric arm per spread + equal_weight, then sp500 appended post-hoc.
    Returns {horizon: {"cfg": cfg_h, "per_seed": {seed: results}}}. The *_fn args are DI seams
    forwarded to run_backtest so the loop runs without torch.
    """
    arms = ["current"] + [("parametric", s) for s in spreads] + ["equal_weight"]
    out = {}
    for horizon in horizons:
        cfg_h = build_weekly_cfg(weekly_cfg, horizon)
        per_seed = {}
        for seed in seeds:
            res = run_backtest(
                prices, rets, cfg_h, oos_periods=oos_periods, rebalance_every=horizon,
                n_runs=n_runs, mc_draws=mc_draws, spreads=spreads, seed=seed, arms=arms,
                select_fn=select_all, runs_fn=runs_fn, period_mu_fn=period_mu_fn, seed_fn=seed_fn,
            )
            dates = res[arms[0] if isinstance(arms[0], str) else "current"]["dates"]
            sp_blocks = sp500_block_returns(
                sp500_rets, list(rets.index), res["rebalance_index"], horizon
            )
            res = splice_sp500_arm(res, sp_blocks, dates)
            per_seed[seed] = res
        out[horizon] = {"cfg": cfg_h, "per_seed": per_seed}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add weekly-robust run loop with SP500 splice"
```

---

### Task 5: Net-of-cost (`net_of_cost_table` + `break_even_bps`)

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def _arm(block_returns, turnover):
    n = len(block_returns)
    return {"block_returns": block_returns, "turnover": turnover,
            "n_held": [5] * n, "weights": [], "dates": [f"d{i}" for i in range(n)]}


def test_net_of_cost_reduces_sharpe_for_higher_turnover():
    cfg = wrb.build_weekly_cfg(_weekly_cfg(), horizon=4)
    # msr: higher turnover; s4: lower turnover, same gross returns
    per_seed = {0: {
        "current": _arm([0.02, 0.03, 0.01, 0.02], [None, 0.8, 0.8, 0.8]),
        "parametric_s4": _arm([0.02, 0.03, 0.01, 0.02], [None, 0.2, 0.2, 0.2]),
        "rebalance_index": [0, 1, 2, 3],
    }}
    table = wrb.net_of_cost_table(per_seed, cfg, rebalance_every=4, bps_levels=[0, 50])
    # at 0 bps both equal; at 50 bps msr's net sharpe drops more than s4's
    assert table.loc["current", "sharpe_0bps"] == pytest.approx(table.loc["parametric_s4", "sharpe_0bps"])
    drop_msr = table.loc["current", "sharpe_0bps"] - table.loc["current", "sharpe_50bps"]
    drop_s4 = table.loc["parametric_s4", "sharpe_0bps"] - table.loc["parametric_s4", "sharpe_50bps"]
    assert drop_msr > drop_s4


def test_break_even_bps_finds_crossover_or_none():
    cfg = wrb.build_weekly_cfg(_weekly_cfg(), horizon=4)
    per_seed = {0: {
        "current": _arm([0.02, 0.02, 0.02, 0.02], [None, 0.8, 0.8, 0.8]),
        "parametric_s4": _arm([0.018, 0.018, 0.018, 0.018], [None, 0.2, 0.2, 0.2]),
        "rebalance_index": [0, 1, 2, 3],
    }}
    be = wrb.break_even_bps(per_seed, cfg, rebalance_every=4,
                            arm_a="parametric_s4", arm_b="current", max_bps=500)
    # s4 starts below (lower gross) but lower turnover -> overtakes at some positive bps
    assert be is None or be > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -k "net_of_cost or break_even" -v`
Expected: FAIL — `AttributeError: ... 'net_of_cost_table'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def _net_block_returns(block_returns, turnover, bps):
    """Gross block returns minus per-block turnover cost (turnover None -> 0)."""
    cost = bps / 1e4
    return [br - (0.0 if t is None else t) * cost for br, t in zip(block_returns, turnover)]


def _arm_net_sharpe(per_seed, label, bps, blocks_per_year, rf_block):
    """Mean across seeds of the net-of-cost annualized Sharpe for one arm at one cost level."""
    vals = []
    for s in per_seed:
        d = per_seed[s][label]
        net = _net_block_returns(d["block_returns"], d["turnover"], bps)
        vals.append(annualized_stats(net, blocks_per_year, rf_block)["sharpe"])
    return float(np.nanmean(vals))


def net_of_cost_table(per_seed, cfg, rebalance_every, bps_levels):
    """Net-of-cost annualized Sharpe per arm at each bps level (mean across seeds).

    Returns a DataFrame indexed by arm with a sharpe_<bps>bps column per level.
    """
    blocks_per_year = cfg["periods_per_year"] / rebalance_every
    rf_block = (1.0 + cfg["rf_period"]) ** rebalance_every - 1.0
    seeds = list(per_seed.keys())
    labels = [k for k in per_seed[seeds[0]] if k != "rebalance_index"]
    rows = []
    for lab in labels:
        row = {"arm": lab}
        for bps in bps_levels:
            row[f"sharpe_{int(bps)}bps"] = _arm_net_sharpe(
                per_seed, lab, bps, blocks_per_year, rf_block
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def break_even_bps(per_seed, cfg, rebalance_every, arm_a, arm_b, max_bps=500, step=1.0):
    """Smallest bps at which arm_a's net Sharpe reaches/exceeds arm_b's; None if not within max_bps.

    Scans cost from 0 to max_bps; returns the first bps where (sharpe_a - sharpe_b) flips from
    negative to >= 0 (linear-interpolated crossing), or None.
    """
    blocks_per_year = cfg["periods_per_year"] / rebalance_every
    rf_block = (1.0 + cfg["rf_period"]) ** rebalance_every - 1.0

    def diff(bps):
        return (_arm_net_sharpe(per_seed, arm_a, bps, blocks_per_year, rf_block)
                - _arm_net_sharpe(per_seed, arm_b, bps, blocks_per_year, rf_block))

    prev_bps, prev_d = 0.0, diff(0.0)
    if prev_d >= 0:
        return 0.0
    bps = step
    while bps <= max_bps:
        d = diff(bps)
        if d >= 0:
            # linear interpolation of the zero crossing between prev_bps and bps
            return float(prev_bps + (0.0 - prev_d) * (bps - prev_bps) / (d - prev_d))
        prev_bps, prev_d = bps, d
        bps += step
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add net-of-cost table and break-even bps"
```

---

### Task 6: `paired_bootstrap`

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def test_paired_bootstrap_significant_positive():
    # arm_a beats arm_b by +0.01 every block, across two seeds -> P(diff>0) ~ 1
    per_seed = {
        0: {"parametric_s4": _arm([0.03, 0.03, 0.03], [None, 0, 0]),
            "current": _arm([0.02, 0.02, 0.02], [None, 0, 0]),
            "rebalance_index": [0, 1, 2]},
        1: {"parametric_s4": _arm([0.03, 0.03, 0.03], [None, 0, 0]),
            "current": _arm([0.02, 0.02, 0.02], [None, 0, 0]),
            "rebalance_index": [0, 1, 2]},
    }
    res = wrb.paired_bootstrap(per_seed, "parametric_s4", "current",
                               n_boot=500, rng=np.random.default_rng(0))
    assert res["mean_diff"] == pytest.approx(0.01, abs=1e-9)
    assert res["p_gt_0"] == pytest.approx(1.0)
    assert res["ci_low"] == pytest.approx(0.01, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py::test_paired_bootstrap_significant_positive -v`
Expected: FAIL — `AttributeError: ... 'paired_bootstrap'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def paired_bootstrap(per_seed, arm_a, arm_b, n_boot, rng, ci=0.90):
    """Bootstrap the mean paired per-block realized difference (arm_a - arm_b), pooled across seeds.

    Within a seed both arms are scored on the same blocks with the same per-rebalance forecast, so
    the per-block returns are paired; the per-block difference isolates the allocation effect.
    Resamples the pooled differences with replacement n_boot times. Returns
    {"mean_diff", "ci_low", "ci_high", "p_gt_0", "n_blocks"}.
    """
    diffs = []
    for s in per_seed:
        a = np.asarray(per_seed[s][arm_a]["block_returns"], dtype=float)
        b = np.asarray(per_seed[s][arm_b]["block_returns"], dtype=float)
        diffs.extend((a - b).tolist())
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = (1.0 - ci) / 2.0
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.quantile(boot_means, lo)),
        "ci_high": float(np.quantile(boot_means, 1.0 - lo)),
        "p_gt_0": float((boot_means > 0).mean()),
        "n_blocks": n,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add paired block-bootstrap on s4-vs-msr difference"
```

---

### Task 7: `format_weekly_summary`

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def test_format_weekly_summary_content():
    cfg = wrb.build_weekly_cfg(_weekly_cfg(), horizon=4)
    per_seed = {0: {
        "current": _arm([0.02, 0.03, 0.01], [None, 0.8, 0.8]),
        "parametric_s4": _arm([0.02, 0.02, 0.02], [None, 0.2, 0.2]),
        "parametric_s8": _arm([0.01, 0.01, 0.01], [None, 0.1, 0.1]),
        "equal_weight": _arm([0.015, 0.015, 0.015], [None, 0.0, 0.0]),
        "sp500": _arm([0.012, 0.012, 0.012], [None, 0.0, 0.0]),
        "rebalance_index": [0, 1, 2],
    }}
    agg = aggregate_for_horizon(per_seed, cfg, 4)   # helper defined in this task
    text = wrb.format_weekly_summary(
        {4: agg}, {4: {"per_seed": per_seed, "cfg": cfg}},
        bps_levels=[10, 25], n_boot=200, seed=0,
    )
    assert "horizon 4" in text
    assert "sp500" in text
    assert "net-of-cost" in text.lower()
    assert "break-even" in text.lower()
    assert "bootstrap" in text.lower()
    assert "filter OFF" in text or "filter off" in text.lower()
```

Add this helper near the top of the test file (after the imports):

```python
from monthly_env_backtest import aggregate_across_seeds as aggregate_for_horizon
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py::test_format_weekly_summary_content -v`
Expected: FAIL — `AttributeError: ... 'format_weekly_summary'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def format_weekly_summary(agg_by_horizon, run_by_horizon, bps_levels, n_boot, seed, few_blocks=15):
    """Per-horizon realized tables (mean +/- cross-seed std) + net-of-cost + break-even + bootstrap."""
    lines = ["Weekly multi-seed robustness backtest (Plan 1)", "filter OFF | arms: current/s4/s8/equal_weight/sp500", ""]
    for horizon in sorted(agg_by_horizon):
        a = agg_by_horizon[horizon]
        cfg_h = run_by_horizon[horizon]["cfg"]
        per_seed = run_by_horizon[horizon]["per_seed"]
        lines.append(f"== horizon {horizon} weeks | blocks: {a['n_blocks']} ==")
        if a["n_blocks"] < few_blocks:
            lines.append("  (few blocks -> DIRECTIONAL, not a verdict)")
        for arm, r in a["table"].iterrows():
            cells = "  ".join(f"{c}={r[f'{c}_mean']:.4f}+/-{r[f'{c}_std']:.4f}" for c in a["metric_cols"])
            lines.append(f"  {arm:<16} {cells}")

        noc = net_of_cost_table(per_seed, cfg_h, horizon, [0] + list(bps_levels))
        lines.append("  -- net-of-cost annualized Sharpe --")
        lines.append("  " + noc.to_string().replace("\n", "\n  "))
        be = break_even_bps(per_seed, cfg_h, horizon, "parametric_s4", "current")
        lines.append(f"  break-even bps (s4 overtakes msr net Sharpe): "
                     f"{'never within 500' if be is None else f'{be:.1f}'}")

        bs = paired_bootstrap(per_seed, "parametric_s4", "current",
                              n_boot=n_boot, rng=np.random.default_rng(seed))
        lines.append(f"  paired bootstrap (s4-msr mean block return): mean {bs['mean_diff']:+.5f} "
                     f"90% CI [{bs['ci_low']:+.5f}, {bs['ci_high']:+.5f}] P(>0)={bs['p_gt_0']:.2f} "
                     f"(n_blocks={bs['n_blocks']})")
        lines.append("")
    lines.append(
        "Caveats: SP500 = USD market anchor (currency/composition mismatch vs the global universe); "
        "24-week horizon is thin (~10 blocks) and strains a 4-step-tuned model; filter-off here is "
        "forward-looking, NOT a clean prod-filter-removal evaluation; 3 seeds is the coarse floor "
        "(the paired bootstrap is the firmer significance read)."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add weekly-robust summary formatter (net-of-cost + bootstrap)"
```

---

### Task 8: `write_weekly_outputs`

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def test_write_weekly_outputs_creates_files(tmp_path):
    cfg = wrb.build_weekly_cfg(_weekly_cfg(), horizon=4)
    per_seed = {0: {
        "current": {"block_returns": [0.02, 0.03], "turnover": [None, 0.8], "n_held": [5, 5],
                    "weights": [pd.Series([1.0], index=["A"]), pd.Series([1.0], index=["A"])],
                    "dates": ["2020-05", "2020-06"]},
        "sp500": {"block_returns": [0.01, 0.01], "turnover": [None, 0.0], "n_held": [1, 1],
                  "weights": [pd.Series([1.0], index=["SP500"]), pd.Series([1.0], index=["SP500"])],
                  "dates": ["2020-05", "2020-06"]},
        "rebalance_index": [10, 11],
    }}
    agg = aggregate_for_horizon(per_seed, cfg, 4)
    run_out = {4: {"cfg": cfg, "per_seed": per_seed}}

    outdir = str(tmp_path / "weekly_robust")
    paths = wrb.write_weekly_outputs(run_out, {4: agg}, outdir, bps_levels=[10, 25], n_boot=100, seed=0)

    assert os.path.exists(paths["summary"])
    assert os.path.exists(os.path.join(outdir, "weekly_table_h4.csv"))
    assert os.path.exists(os.path.join(outdir, "h4_seed0", "backtest_summary.txt"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py::test_write_weekly_outputs_creates_files -v`
Expected: FAIL — `AttributeError: ... 'write_weekly_outputs'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def write_weekly_outputs(run_out, agg_by_horizon, outdir, bps_levels, n_boot, seed):
    """Write per-(horizon,seed) raw CSVs, per-horizon aggregated tables, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    paths = {"summary": os.path.join(outdir, "weekly_robust_summary.txt")}
    for horizon, hd in run_out.items():
        cfg_h = hd["cfg"]
        for seed_k, res in hd["per_seed"].items():
            sub = os.path.join(outdir, f"h{horizon}_seed{seed_k}")
            write_backtest_outputs(res, cfg_h, horizon, sub)
        tpath = os.path.join(outdir, f"weekly_table_h{horizon}.csv")
        agg_by_horizon[horizon]["table"].to_csv(tpath)
        paths[f"table_h{horizon}"] = tpath
    with open(paths["summary"], "w") as f:
        f.write(format_weekly_summary(agg_by_horizon, run_out, bps_levels, n_boot, seed))
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add weekly-robust output writers"
```

---

### Task 9: `build_arg_parser` + `main`

**Files:**
- Modify: `experiments/weekly_robust_backtest.py`
- Test: `tests/test_weekly_robust_backtest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weekly_robust_backtest.py`:

```python
def test_arg_parser_defaults():
    parser = wrb.build_arg_parser()
    args = parser.parse_args([])
    assert args.horizons == "4,24"
    assert args.seeds == "0,100,200"
    assert args.oos_periods == 250
    assert args.n_runs == 75
    assert args.mc_draws == 1000
    assert args.spreads == "4,8"
    assert args.cost_bps == "10,25"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py::test_arg_parser_defaults -v`
Expected: FAIL — `AttributeError: ... 'build_arg_parser'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/weekly_robust_backtest.py`:

```python
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--horizons", type=str, default="4,24")
    parser.add_argument("--seeds", type=str, default="0,100,200")
    parser.add_argument("--oos-periods", type=int, default=250)
    parser.add_argument("--n-runs", type=int, default=75)
    parser.add_argument("--mc-draws", type=int, default=1000)
    parser.add_argument("--spreads", type=str, default="4,8")
    parser.add_argument("--cost-bps", type=str, default="10,25",
                        help="Comma-separated net-of-cost levels in bps (default 10,25)")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Paired-bootstrap resamples (default 10000)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "weekly_robust"))
    return parser


def main():
    args = build_arg_parser().parse_args()
    weekly_cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)
    sp500 = pd.read_csv(SP500_RETURNS, index_col=0)["sp500"]

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    spreads = [float(x) for x in args.spreads.split(",")]
    bps_levels = [float(x) for x in args.cost_bps.split(",")]

    print(f"Weekly universe: {rets.shape[1]} names | weeks: {rets.shape[0]} | "
          f"horizons={horizons} | seeds={seeds} | oos={args.oos_periods} | n_runs={args.n_runs} "
          f"| K={args.mc_draws} | spreads={spreads} | filter=OFF", flush=True)

    run_out = run_weekly_robust(
        prices, rets, sp500, weekly_cfg, horizons=horizons, seeds=seeds,
        oos_periods=args.oos_periods, n_runs=args.n_runs, mc_draws=args.mc_draws, spreads=spreads,
    )
    agg_by_horizon = {h: aggregate_across_seeds(run_out[h]["per_seed"], run_out[h]["cfg"], h)
                      for h in horizons}
    paths = write_weekly_outputs(run_out, agg_by_horizon, args.outdir,
                                 bps_levels=bps_levels, n_boot=args.n_boot, seed=seeds[0])

    print()
    print(format_weekly_summary(agg_by_horizon, run_out, bps_levels, args.n_boot, seeds[0]))
    print("\nSaved:")
    for p in sorted(set(paths.values())):
        print(f"       {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test file**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_weekly_robust_backtest.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/weekly_robust_backtest.py tests/test_weekly_robust_backtest.py
git commit -m "Add weekly-robust CLI orchestration"
```

---

### Task 10: Full-suite check + fetch + tiny smoke

**Files:** none (verification only).

- [ ] **Step 1: Run the whole test suite**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest -q`
Expected: PASS — all prior tests plus the new fetch (1) and weekly-robust (10) tests green.

- [ ] **Step 2: Fetch the SP500 benchmark (network, run once)**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/fetch_sp500_weekly.py`
Expected: prints the SP500 weekly point count + range and "Saved:"; `experiments/data_weekly_bench/sp500_returns.csv` exists. Confirm the point count is comparable to the universe's ~522 weeks. If the fetch fails (network), report DONE_WITH_CONCERNS — the controller handles it.

- [ ] **Step 3: Tiny real smoke backtest (NOT the production run)**

Run:
```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/weekly_robust_backtest.py --horizons 4 --seeds 0 --oos-periods 8 --n-runs 3 --mc-draws 20 --spreads 4,8 --cost-bps 10,25 --n-boot 200 --outdir experiments/results/weekly_robust_smoke
```
Expected: prints the banner + the horizon-4 table with all five arms (current/parametric_s4/parametric_s8/equal_weight/sp500), the net-of-cost block, the break-even line, and the paired-bootstrap line. Confirm `experiments/results/weekly_robust_smoke/weekly_robust_summary.txt` exists, equal_weight `avg_names` = full universe, sp500 `avg_names` = 1.

- [ ] **Step 4: Report (no commit — smoke output gitignored)**

Report the fetch diagnostics and smoke outcome to the controller.

---

## Production run (later session — the user runs this)

Launch in the background (~48 h):

```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/weekly_robust_backtest.py \
  --horizons 4,24 --seeds 0,100,200 --oos-periods 250 --n-runs 75 --mc-draws 1000 \
  --spreads 4,8 --cost-bps 10,25 --n-boot 10000 --outdir experiments/results/weekly_robust
```

Then read `experiments/results/weekly_robust/weekly_robust_summary.txt`: per horizon, does s4/s8 beat current(msr) and the equal_weight/sp500 benchmarks on realized Sharpe; the net-of-cost table + break-even bps (does s4's lower turnover earn its keep); and the paired bootstrap P(s4>msr) and CI (is any gap real or noise). Read h4 (~62 blocks) as primary, h24 (~10) as directional.

---

## Self-Review Notes

- **Spec coverage:** isolated SP500 fetch (Task 1); build_weekly_cfg horizon-only override (Task 2); SP500 benchmark splice (Task 3); horizons×seeds loop, filter off via select_all, 4 optimizer arms via run_backtest reused unchanged + sp500 5th arm (Task 4); net-of-cost + break-even (Task 5); paired bootstrap (Task 6); summary with block counts + net-of-cost + break-even + bootstrap + caveats (Task 7); outputs per-(h,seed) + aggregated tables (Task 8); CLI defaults horizons 4,24 / seeds 0,100,200 / oos 250 / n 75 / K 1000 / spreads 4,8 / cost 10,25 (Task 9); cost + run command (Production section). Covered.
- **Out-of-scope respected:** reads but never writes data/ or params.yaml; no pipeline / backtest_allocation edits; no filter-ON sweep; no baked-in regime split (free post-hoc); no productionisation.
- **Type consistency:** arm labels current / parametric_s4 / parametric_s8 / equal_weight / sp500 (from backtest_allocation.label_of + the splice) used consistently Tasks 3–9; aggregate_across_seeds reused from monthly_env_backtest (returns {"table","n_blocks","metric_cols"}); run_out shape {horizon:{"cfg","per_seed"}} consistent across Tasks 4/7/8/9; `_arm_net_sharpe` shared by net_of_cost_table and break_even_bps (Task 5).
