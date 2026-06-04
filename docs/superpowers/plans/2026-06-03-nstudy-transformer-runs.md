# n_transformer_runs Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/nstudy_transformer_runs.py` — an in-sample harness that sweeps `n_transformer_runs` for the `current` (msr) and parametric-Michaud `s∈{4,8}` arms (historical Σ), reusing the train-once/prefix-average trick, and reports composition + value-dispersion metrics vs `n` as mean ± cross-seed spread.

**Architecture:** A new experiment script that reuses the allocation/metric helpers from `experiments/measure_allocation_stability.py` and the prefix-average loop pattern from `experiments/sweep_transformer_runs.py`. Pure functions with dependency-injection seams (forecaster/select/seed) so the whole loop is unit-testable without torch. One seed at a time → aggregate across seeds.

**Tech Stack:** Python, numpy, pandas, scikit-learn (LedoitWolf), pytest. Torch only at runtime via lazy import (stubbed in tests).

Spec: `docs/superpowers/specs/2026-06-03-nstudy-transformer-runs-design.md`.

**Conventions:**
- Test runner: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`
- Experiment-only and **local** per the push-scope rule — do NOT push; commit to local main.
- No AI attribution in commit messages.

---

## File Structure

- Create: `experiments/nstudy_transformer_runs.py` — the study (helpers + per-seed runner + aggregation + CLI).
- Create: `tests/test_nstudy_transformer_runs.py` — torch-free unit tests.
- Reuse (no edits): `experiments/measure_allocation_stability.py`, `experiments/sweep_transformer_runs.py`.

The module is built up one function per task. Every task is TDD: failing test → run → implement → run → commit.

---

### Task 1: Module scaffold + `prefix_forecast`

`prefix_forecast` averages the first-`n` transformer runs into one per-stock μ vector — the core of the prefix trick.

**Files:**
- Create: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_nstudy_transformer_runs.py`:

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

import nstudy_transformer_runs as ns


def _identity_winsorize(preds_df, rets):
    return preds_df


def _mean_period_mu(preds_df):
    return preds_df.mean(axis=0)


def test_prefix_forecast_averages_first_n_runs():
    # 3 runs, 2 forecast periods, 2 stocks
    runs = np.array([
        [[1.0, 2.0], [1.0, 2.0]],   # run 0
        [[3.0, 4.0], [3.0, 4.0]],   # run 1
        [[9.0, 9.0], [9.0, 9.0]],   # run 2 (excluded when n=2)
    ])
    rets = pd.DataFrame(np.zeros((5, 2)), columns=["A", "B"])
    mu = ns.prefix_forecast(runs, 2, rets, _identity_winsorize, _mean_period_mu)
    # first 2 runs averaged: A=(1+3)/2=2, B=(2+4)/2=3; period-mean leaves them unchanged
    assert mu["A"] == pytest.approx(2.0)
    assert mu["B"] == pytest.approx(3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py::test_prefix_forecast_averages_first_n_runs -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nstudy_transformer_runs'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/nstudy_transformer_runs.py`:

```python
"""
n_transformer_runs study: sweep n for the current (msr) arm and parametric-Michaud
s in {4, 8} (historical Sigma), reusing the train-once / first-n prefix-average trick.

Reports composition stability (turnover, Jaccard, overlap) and value dispersion
(ret/vol/sharpe CoV) vs n, aggregated across seeds as mean +/- cross-seed std.

This is in-sample: value is scored against the same averaged mu the current arm is the
Sharpe-argmax of, so read the composition metrics and the value CoVs, NOT the value
levels. The realized verdict is a separate (backtest) study.

Run:  python experiments/nstudy_transformer_runs.py --seeds 0,100 --iterations 10 \
        --grid 10,25,50,75,100 --spreads 4,8 --mc-draws 1000
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

See docs/superpowers/specs/2026-06-03-nstudy-transformer-runs-design.md.
"""

import argparse
import math
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
sys.path.insert(0, HERE)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS
from measure_allocation_stability import (
    select_stocks, allocate_msr, sample_mu_draws, resampled_allocate,
    portfolio_metrics, mean_turnover, mean_jaccard, overlap_stats,
    metric_dispersion, seed_everything,
)


def prefix_forecast(runs, n, rets, winsorize_fn, period_mu_fn):
    """Average the first `n` transformer runs into one per-stock mu vector.

    runs: ndarray (n_runs, periods_to_forecast, n_stocks). Mirrors the prefix trick in
    sweep_transformer_runs.py: average the first-n runs, winsorise to history, then reduce
    the forecast path to one per-stock number via period_mu_fn. Returns a Series over
    rets.columns.
    """
    prefix = runs[:n].mean(axis=0)
    preds_df = winsorize_fn(pd.DataFrame(prefix, columns=rets.columns), rets)
    return period_mu_fn(preds_df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy prefix_forecast helper"
```

---

### Task 2: `parametric_arm_draws` (paired draws across spreads)

Generate the parametric Michaud draws for every spread from the **same** `z`, so the s-arms differ only by the `s` scaling (`chol(s^2 Sigma/T) = s * chol(Sigma/T)`).

**Files:**
- Modify: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nstudy_transformer_runs.py`:

```python
def test_parametric_arm_draws_are_paired_across_spreads():
    names = ["A", "B", "C"]
    mu = pd.Series([0.01, 0.02, 0.03], index=names)
    cov = pd.DataFrame(np.eye(3) * 0.04, index=names, columns=names)
    draws = ns.parametric_arm_draws(
        mu, cov, n_periods=100, n_draws=5, spreads=[4.0, 8.0], rng_seed=(7, 25)
    )
    # same z reused -> s=8 perturbation is exactly 2x the s=4 perturbation
    for k in range(5):
        d4 = draws[4.0][k] - mu
        d8 = draws[8.0][k] - mu
        assert np.allclose(d8.values, 2.0 * d4.values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py::test_parametric_arm_draws_are_paired_across_spreads -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'parametric_arm_draws'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/nstudy_transformer_runs.py`:

```python
def parametric_arm_draws(mu_sel, cov_sel, n_periods, n_draws, spreads, rng_seed):
    """K Monte-Carlo mu draws per spread, all sharing the same z for a paired s-comparison.

    A fresh rng is re-seeded from rng_seed for every spread, so the standard-normal draws
    are identical across spreads and the only difference is the s scaling. Returns
    {spread: [Series, ...]} preserving mu_sel's index.
    """
    out = {}
    for s in spreads:
        rng = np.random.default_rng(rng_seed)
        out[s] = sample_mu_draws(mu_sel, cov_sel, n_periods, n_draws, s, rng)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy paired parametric draws across spreads"
```

---

### Task 3: `run_nstudy_seed` (one-seed core loop)

Run `iterations` passes for one seed; per iteration train `max(grid)` runs once, then for each `n` derive μ̄(n) and allocate the `current` + each `s` arm, all scored against μ̄(n).

**Files:**
- Modify: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nstudy_transformer_runs.py`:

```python
@pytest.fixture
def tiny_cfg():
    return {
        "rf_period": 0.0,
        "max_weight": 0.5,
        "min_weight": 0.05,
        "periods_per_year": 54,
    }


def _make_stub_runs_fn(n_stocks, periods=4):
    """Deterministic per-iteration runs: shape (n_runs, periods, n_stocks)."""
    rng = np.random.default_rng(0)

    def _runs_fn(rets, cfg, n_runs=None, verbose=False):
        return rng.normal(0.01, 0.005, size=(n_runs, periods, n_stocks))

    return _runs_fn


def test_run_nstudy_seed_shapes_and_arms(tiny_cfg):
    cols = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.02, size=(40, 5)), columns=cols)
    prices = pd.DataFrame(np.ones((40, 5)), columns=cols)  # unused (select_fn injected)

    result = ns.run_nstudy_seed(
        prices, rets, tiny_cfg,
        grid=[2, 3], iterations=2, seed=0, spreads=[4.0, 8.0], n_draws=8,
        train_runs_fn=_make_stub_runs_fn(5),
        winsorize_fn=_identity_winsorize,
        period_mu_fn=_mean_period_mu,
        select_fn=lambda p, r, c: cols,
        seed_fn=lambda s: None,
    )

    assert result["arms"] == ["current", "s4", "s8"]
    assert result["grid"] == [2, 3]
    # weights: iterations x selected names; metrics: iterations x [ret, vol, sharpe]
    w = result["data"]["s4"][3]["weights"]
    m = result["data"]["s4"][3]["metrics"]
    assert w.shape == (2, 5)
    assert list(m.columns) == ["ret", "vol", "sharpe"]
    assert m.shape == (2, 3)
    assert result["data"]["current"][2]["weights"].shape == (2, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py::test_run_nstudy_seed_shapes_and_arms -v`
Expected: FAIL — `AttributeError: ... 'run_nstudy_seed'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/nstudy_transformer_runs.py`:

```python
def run_nstudy_seed(prices, rets, cfg, grid, iterations, seed, spreads, n_draws,
                    train_runs_fn=None, winsorize_fn=None, period_mu_fn=None,
                    select_fn=None, seed_fn=None, verbose=False):
    """Run `iterations` passes for one seed and collect per-(arm, n) weights + metrics.

    Sigma (Ledoit-Wolf) and the selected names are deterministic and computed once. Per
    iteration, train max(grid) runs once (train_runs_fn) and derive mu(n) from the first-n
    prefix for each n. The current arm allocates mu(n) via the msr elimination loop; each s
    arm draws K parametric mu vectors (paired across spreads) and forms the resampled
    consensus. All arms are scored against the same mu(n). The *_fn args are
    dependency-injection seams (lazy real defaults) so the loop runs without torch.

    Returns {"selected": [...], "arms": [...], "grid": [...],
             "data": {arm: {n: {"weights": DataFrame, "metrics": DataFrame}}}}.
    """
    if train_runs_fn is None or winsorize_fn is None or period_mu_fn is None:
        from transformer_model import train_runs, winsorize_to_history, weighted_mean_return
        train_runs_fn = train_runs_fn or train_runs
        winsorize_fn = winsorize_fn or winsorize_to_history
        period_mu_fn = period_mu_fn or weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    rf = cfg["rf_period"]
    grid = sorted(grid)
    max_n = max(grid)
    T = len(rets)

    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    arms = ["current"] + [f"s{int(s)}" for s in spreads]
    rec = {arm: {n: {"w": [], "m": []} for n in grid} for arm in arms}

    start = time.time()
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        runs = train_runs_fn(rets, cfg, n_runs=max_n)
        for n in grid:
            mu_sel = prefix_forecast(runs, n, rets, winsorize_fn, period_mu_fn).loc[selected]

            w_cur = allocate_msr(mu_sel, cov_sel, cfg)
            rec["current"][n]["w"].append(w_cur)
            rec["current"][n]["m"].append(
                portfolio_metrics(w_cur[w_cur.abs() > 1e-9], mu_sel, cov_sel, rf)
            )

            draws_by_s = parametric_arm_draws(
                mu_sel, cov_sel, T, n_draws, spreads, (seed + i, n)
            )
            for s in spreads:
                w_s, _ = resampled_allocate(draws_by_s[s], cov_sel, cfg)
                arm = f"s{int(s)}"
                rec[arm][n]["w"].append(w_s)
                rec[arm][n]["m"].append(
                    portfolio_metrics(w_s[w_s.abs() > 1e-9], mu_sel, cov_sel, rf)
                )
        if verbose:
            elapsed = time.time() - start
            eta = elapsed / i * (iterations - i)
            print(f"[seed {seed}] iter {i}/{iterations} | elapsed {elapsed:6.0f}s "
                  f"| ETA {eta:6.0f}s", flush=True)

    data = {
        arm: {
            n: {
                "weights": pd.DataFrame(rec[arm][n]["w"]).reset_index(drop=True),
                "metrics": pd.DataFrame(rec[arm][n]["m"]),
            }
            for n in grid
        }
        for arm in arms
    }
    return {"selected": selected, "arms": arms, "grid": grid, "data": data}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy per-seed core loop"
```

---

### Task 4: `seed_arm_metrics` + `summarize_nstudy` + `first_flattening_n`

Reduce each seed's raw weights/metrics to per-(arm, n) scalars, aggregate across seeds as mean ± std, and add the advisory "where does it flatten" helper.

**Files:**
- Modify: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nstudy_transformer_runs.py`:

```python
def _fake_seed_result(arms, grid, weights_by_arm_n, metrics_by_arm_n):
    data = {
        arm: {
            n: {
                "weights": weights_by_arm_n[arm][n],
                "metrics": metrics_by_arm_n[arm][n],
            }
            for n in grid
        }
        for arm in arms
    }
    return {"selected": list(weights_by_arm_n[arms[0]][grid[0]].columns),
            "arms": arms, "grid": grid, "data": data}


def test_summarize_nstudy_mean_and_std_across_seeds():
    arms, grid = ["current"], [10]
    names = ["A", "B"]
    metrics = pd.DataFrame({"ret": [0.02, 0.02], "vol": [0.1, 0.1], "sharpe": [0.2, 0.2]})

    # seed 0: weights churn between iterations -> turnover 1.0
    w_churn = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=names)
    # seed 1: identical weights -> turnover 0.0
    w_same = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], columns=names)

    r0 = _fake_seed_result(arms, grid, {"current": {10: w_churn}}, {"current": {10: metrics}})
    r1 = _fake_seed_result(arms, grid, {"current": {10: w_same}}, {"current": {10: metrics}})

    summary = ns.summarize_nstudy({0: r0, 1: r1})
    mean = summary["mean"]
    std = summary["std"]
    # turnover mean over seeds = (1.0 + 0.0)/2 = 0.5; population std = 0.5
    assert mean.loc[("current", 10), "turnover"] == pytest.approx(0.5)
    assert std.loc[("current", 10), "turnover"] == pytest.approx(0.5)


def test_first_flattening_n_finds_first_small_delta():
    grid = [10, 25, 50, 75, 100]
    # big drops early, then flat from 75 on
    values = {10: 0.84, 25: 0.74, 50: 0.66, 75: 0.65, 100: 0.648}
    assert ns.first_flattening_n(values, grid, tol=0.05) == 75
    # never flattens
    steep = {10: 0.84, 25: 0.6, 50: 0.4, 75: 0.25, 100: 0.1}
    assert ns.first_flattening_n(steep, grid, tol=0.05) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -k "summarize or flattening" -v`
Expected: FAIL — `AttributeError: ... 'summarize_nstudy'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/nstudy_transformer_runs.py`:

```python
def seed_arm_metrics(seed_result):
    """Per-(arm, n) scalar metrics for one seed (one row per arm x n)."""
    rows = []
    for arm in seed_result["arms"]:
        for n in seed_result["grid"]:
            w = seed_result["data"][arm][n]["weights"]
            m = seed_result["data"][arm][n]["metrics"]
            ov = overlap_stats(w)
            disp = metric_dispersion(m)
            rows.append({
                "arm": arm, "n": n,
                "turnover": mean_turnover(w),
                "jaccard": mean_jaccard(w),
                "overlap_fraction": ov["fraction"],
                "held": ov["held"],
                "ret_mean": disp["ret"]["mean"], "ret_cov": disp["ret"]["cov"],
                "vol_mean": disp["vol"]["mean"], "vol_cov": disp["vol"]["cov"],
                "sharpe_mean": disp["sharpe"]["mean"], "sharpe_cov": disp["sharpe"]["cov"],
            })
    return pd.DataFrame(rows)


METRIC_COLS = [
    "turnover", "jaccard", "overlap_fraction", "held",
    "ret_mean", "ret_cov", "vol_mean", "vol_cov", "sharpe_mean", "sharpe_cov",
]


def summarize_nstudy(results_by_seed):
    """Aggregate per-(arm, n) scalar metrics across seeds as mean and population std.

    Returns {"mean": DataFrame, "std": DataFrame, "metric_cols": [...], "arms": [...],
    "grid": [...]} where mean/std are indexed by (arm, n).
    """
    per_seed = [seed_arm_metrics(r) for r in results_by_seed.values()]
    allm = pd.concat(per_seed, ignore_index=True)
    grouped = allm.groupby(["arm", "n"])[METRIC_COLS]
    any_result = next(iter(results_by_seed.values()))
    return {
        "mean": grouped.mean(),
        "std": grouped.std(ddof=0),
        "metric_cols": METRIC_COLS,
        "arms": any_result["arms"],
        "grid": any_result["grid"],
    }


def first_flattening_n(values_by_n, grid, tol=0.05):
    """First n in grid whose value changed < tol (relative) vs the previous grid point.

    Advisory only -- a hint at where a metric stops moving, not a verdict. Returns None
    if no consecutive pair falls under tol.
    """
    grid = sorted(grid)
    for prev, n in zip(grid[:-1], grid[1:]):
        base = values_by_n.get(prev)
        if base is None or (isinstance(base, float) and math.isnan(base)) or base == 0:
            continue
        cur = values_by_n.get(n)
        if cur is None or (isinstance(cur, float) and math.isnan(cur)):
            continue
        if abs(cur - base) / abs(base) < tol:
            return n
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy cross-seed aggregation and flattening helper"
```

---

### Task 5: `format_nstudy_summary`

Render the per-arm metric-vs-n tables (mean ± std) plus the advisory flattening note for the primary arm.

**Files:**
- Modify: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nstudy_transformer_runs.py`:

```python
def test_format_nstudy_summary_has_arms_and_advisory():
    arms, grid = ["current", "s4"], [10, 25]
    names = ["A", "B"]
    metrics = pd.DataFrame({"ret": [0.02, 0.02], "vol": [0.1, 0.1], "sharpe": [0.2, 0.2]})
    # churning weights -> turnover is non-zero (1.0) but identical across n -> flat
    w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=names)
    weights = {a: {n: w for n in grid} for a in arms}
    mets = {a: {n: metrics for n in grid} for a in arms}
    r = _fake_seed_result(arms, grid, weights, mets)
    summary = ns.summarize_nstudy({0: r})

    text = ns.format_nstudy_summary(summary, primary_arm="s4", tol=0.05)
    assert "current" in text
    assert "s4" in text
    assert "turnover" in text
    # s4 turnover is flat across n (both = 1.0) -> advisory fires at n=25
    assert "turnover flattens at n=25" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py::test_format_nstudy_summary_has_arms_and_advisory -v`
Expected: FAIL — `AttributeError: ... 'format_nstudy_summary'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/nstudy_transformer_runs.py`:

```python
def _fmt_mean_std(mean, std):
    def _one(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.4f}"
    return f"{_one(mean)}+/-{_one(std)}"


def format_nstudy_summary(summary, primary_arm="s4", tol=0.05):
    """Per-arm metric-vs-n tables (mean +/- cross-seed std) plus the advisory note."""
    mean, std = summary["mean"], summary["std"]
    cols = summary["metric_cols"]
    grid = summary["grid"]

    lines = ["n_transformer_runs study (mean +/- cross-seed std)", ""]
    for arm in summary["arms"]:
        lines.append(f"== arm: {arm} ==")
        header = f"  {'n':>5}  " + "  ".join(f"{c:>20}" for c in cols)
        lines.append(header)
        for n in grid:
            cells = [f"{_fmt_mean_std(mean.loc[(arm, n), c], std.loc[(arm, n), c]):>20}"
                     for c in cols]
            lines.append(f"  {n:>5}  " + "  ".join(cells))
        lines.append("")

    if primary_arm in summary["arms"]:
        turnover_by_n = {n: mean.loc[(primary_arm, n), "turnover"] for n in grid}
        jacc_by_n = {n: mean.loc[(primary_arm, n), "jaccard"] for n in grid}
        t_flat = first_flattening_n(turnover_by_n, grid, tol)
        j_flat = first_flattening_n(jacc_by_n, grid, tol)
        lines.append(
            f"Advisory ({primary_arm}, tol={tol:.0%}): turnover flattens at "
            f"n={t_flat}; Jaccard flattens at n={j_flat}. "
            "A hint at where added runs stop buying stability -- the final pick is yours."
        )
    lines.append("")
    lines.append(
        "Note: in-sample. Read composition (turnover/Jaccard/overlap) and value CoVs, "
        "NOT value levels (current is the in-sample argmax of mu(n)). Realized verdict = "
        "the backtest study."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy summary formatter"
```

---

### Task 6: `write_nstudy_outputs`

Write per-seed raw weights/metrics CSVs (indexed by n + iteration), per-arm aggregated tables, and the summary text.

**Files:**
- Modify: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nstudy_transformer_runs.py`:

```python
def test_write_nstudy_outputs_creates_files(tmp_path):
    arms, grid = ["current", "s4"], [10, 25]
    names = ["A", "B"]
    metrics = pd.DataFrame({"ret": [0.02, 0.02], "vol": [0.1, 0.1], "sharpe": [0.2, 0.2]})
    w = pd.DataFrame([[1.0, 0.0], [0.6, 0.4]], columns=names)
    weights = {a: {n: w for n in grid} for a in arms}
    mets = {a: {n: metrics for n in grid} for a in arms}
    r = _fake_seed_result(arms, grid, weights, mets)
    summary = ns.summarize_nstudy({0: r})

    outdir = str(tmp_path / "nstudy")
    paths = ns.write_nstudy_outputs({0: r}, summary, outdir)

    assert os.path.exists(paths["summary"])
    assert os.path.exists(os.path.join(outdir, "nstudy_current_seed0_weights.csv"))
    assert os.path.exists(os.path.join(outdir, "nstudy_table_s4.csv"))
    # raw weights reload: 2 iterations x 2 grid points = 4 rows, with n + iteration cols
    raw = pd.read_csv(os.path.join(outdir, "nstudy_s4_seed0_weights.csv"))
    assert len(raw) == 4
    assert {"n", "iteration"}.issubset(raw.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py::test_write_nstudy_outputs_creates_files -v`
Expected: FAIL — `AttributeError: ... 'write_nstudy_outputs'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/nstudy_transformer_runs.py`:

```python
def _stack_over_grid(arm_data, grid, value="weights"):
    """Concatenate a single arm's per-n frames into one frame with n + iteration columns."""
    frames = []
    for n in grid:
        df = arm_data[n][value].copy()
        df.index.name = "iteration"
        df = df.reset_index()
        df.insert(0, "n", n)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_nstudy_outputs(results_by_seed, summary, outdir):
    """Write per-seed raw CSVs, per-arm aggregated tables, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    grid = summary["grid"]
    arms = summary["arms"]
    paths = {"summary": os.path.join(outdir, "nstudy_summary.txt")}

    for seed, result in results_by_seed.items():
        for arm in arms:
            wpath = os.path.join(outdir, f"nstudy_{arm}_seed{seed}_weights.csv")
            mpath = os.path.join(outdir, f"nstudy_{arm}_seed{seed}_metrics.csv")
            _stack_over_grid(result["data"][arm], grid, "weights").to_csv(wpath, index=False)
            _stack_over_grid(result["data"][arm], grid, "metrics").to_csv(mpath, index=False)
            paths[f"{arm}_seed{seed}_weights"] = wpath

    for arm in arms:
        table = pd.DataFrame(index=pd.Index(grid, name="n"))
        for c in summary["metric_cols"]:
            table[f"{c}_mean"] = [summary["mean"].loc[(arm, n), c] for n in grid]
            table[f"{c}_std"] = [summary["std"].loc[(arm, n), c] for n in grid]
        tpath = os.path.join(outdir, f"nstudy_table_{arm}.csv")
        table.to_csv(tpath)
        paths[f"table_{arm}"] = tpath

    with open(paths["summary"], "w") as f:
        f.write(format_nstudy_summary(summary))
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy output writers"
```

---

### Task 7: `build_arg_parser` + `main` (CLI wiring)

Wire the CLI, loop seeds, aggregate, write, and print. The runner internals are already tested; this task adds a parser test and the orchestration.

**Files:**
- Modify: `experiments/nstudy_transformer_runs.py`
- Test: `tests/test_nstudy_transformer_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nstudy_transformer_runs.py`:

```python
def test_arg_parser_parses_lists():
    parser = ns.build_arg_parser()
    args = parser.parse_args(
        ["--seeds", "0,100", "--grid", "10,25,50", "--spreads", "4,8",
         "--iterations", "10", "--mc-draws", "1000"]
    )
    assert [int(x) for x in args.seeds.split(",")] == [0, 100]
    assert [int(x) for x in args.grid.split(",")] == [10, 25, 50]
    assert [float(x) for x in args.spreads.split(",")] == [4.0, 8.0]
    assert args.iterations == 10
    assert args.mc_draws == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py::test_arg_parser_parses_lists -v`
Expected: FAIL — `AttributeError: ... 'build_arg_parser'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/nstudy_transformer_runs.py`:

```python
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=str, default="0,100",
                        help="Comma-separated base seeds (default 0,100; spaced >= iterations)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="predict->allocate passes per seed (default 10)")
    parser.add_argument("--grid", type=str, default="10,25,50,75,100",
                        help="Comma-separated n_transformer_runs values")
    parser.add_argument("--spreads", type=str, default="4,8",
                        help="Comma-separated parametric-Michaud spreads s (default 4,8)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="K Monte-Carlo mu draws per consensus (default 1000)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "nstudy"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    seeds = [int(x) for x in args.seeds.split(",")]
    grid = [int(x) for x in args.grid.split(",")]
    spreads = [float(x) for x in args.spreads.split(",")]

    print(f"Universe: {rets.shape[1]} stocks | seeds={seeds} | iterations={args.iterations} "
          f"| grid={grid} | spreads={spreads} | K={args.mc_draws}", flush=True)

    results_by_seed = {}
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        results_by_seed[seed] = run_nstudy_seed(
            prices, rets, cfg,
            grid=grid, iterations=args.iterations, seed=seed,
            spreads=spreads, n_draws=args.mc_draws, verbose=True,
        )

    summary = summarize_nstudy(results_by_seed)
    paths = write_nstudy_outputs(results_by_seed, summary, args.outdir)

    print()
    print(format_nstudy_summary(summary))
    print("\nSaved:")
    for p in sorted(set(paths.values())):
        print(f"       {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test file**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_nstudy_transformer_runs.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/nstudy_transformer_runs.py tests/test_nstudy_transformer_runs.py
git commit -m "Add nstudy CLI orchestration"
```

---

### Task 8: Full-suite check + tiny real smoke run

Confirm the new tests don't break the repo and the script runs end-to-end on real data at a trivial budget before the long run.

**Files:** none (verification only).

- [ ] **Step 1: Run the whole test suite**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest -q`
Expected: PASS — all prior tests plus the 8 new ones green.

- [ ] **Step 2: Tiny real smoke run (NOT the production run)**

Run:
```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/nstudy_transformer_runs.py --seeds 0 --iterations 1 --grid 5,10 --spreads 4,8 --mc-draws 20 --outdir experiments/results/nstudy_smoke
```
Expected: prints the universe banner, a per-seed ETA line, the per-arm tables, and a "Saved:" list. Confirm `experiments/results/nstudy_smoke/nstudy_summary.txt` exists and the `current`/`s4`/`s8` tables have rows for n=5 and n=10.

- [ ] **Step 3: Spot-check the smoke summary**

Read `experiments/results/nstudy_smoke/nstudy_summary.txt`. Confirm: three arm blocks, the advisory line, and that weights sum sensibly (each consensus arm holds a handful of names). Expected at this trivial budget: with one seed the cross-seed std is `0.0` for computable metrics (population std of a single value); with iterations=1 the turnover/Jaccard cells are `N/A` (need >=2 iterations). Both resolve at the production budget (2 seeds, 10 iterations).

- [ ] **Step 4: Commit smoke note (no code change)**

No commit needed (smoke output is gitignored under `experiments/results/`). Report results to the user and proceed to the production launch decision.

---

## Production run (after the plan is implemented and smoke passes)

Launch in the background (~8.3 h: ~5 h training + ~3.3 h optimization):

```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/nstudy_transformer_runs.py \
  --seeds 0,100 --iterations 10 --grid 10,25,50,75,100 --spreads 4,8 --mc-draws 1000 \
  --outdir experiments/results/nstudy
```

Then read `experiments/results/nstudy/nstudy_summary.txt`: for the `s4` arm, find the `n` where turnover/Jaccard/overlap stop improving meaningfully and the value CoVs are acceptable; cross-check the `s8` arm to see whether the fair-`n` shifts with spread; confirm the cross-seed std is small enough to trust the ranking. The chosen `n` feeds the later (pushable) productionisation + the Plan 1 multi-seed realized backtest.

---

## Self-Review Notes

- **Spec coverage:** grid/iterations/seeds/spreads/K/Σ (Task 7 defaults + Task 3 loop); prefix trick (Task 1); paired draws (Task 2); current + s4/s8 arms scored vs μ̄(n) (Task 3); composition + value-dispersion metrics (Task 4); cross-seed mean±std (Task 4); advisory flattening (Tasks 4–5); outputs (Task 6); in-sample caveat (Task 5 note); DI/torch-free tests (all tasks); cost + run command (Production section). Covered.
- **Out-of-scope respected:** no forecast-Σ arm, no s=2, no backtest scoring, no pipeline edits.
- **Type consistency:** arm keys `current`/`s4`/`s8` (`f"s{int(s)}"`) used consistently across Tasks 3–7; `METRIC_COLS` defined in Task 4 and reused in Tasks 5–6; seed-result dict shape (`selected`/`arms`/`grid`/`data`) consistent between `run_nstudy_seed` (Task 3) and the `_fake_seed_result` test helper (Task 4).
