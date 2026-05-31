# Allocation Stability Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, self-contained instrument that runs the predict→allocate path many times and quantifies how much portfolio *composition* moves relative to how much portfolio *value* moves.

**Architecture:** A single script `experiments/measure_allocation_stability.py` (Approach B — re-implements the technical-filter selection and the MSR elimination loop locally rather than refactoring the pipeline). Pure helper functions do the allocation, metric computation, and aggregation, so they are unit-testable with synthetic data and no GPU. The heavy `train_and_predict` (and its `torch` import) is resolved lazily and dependency-injected so the loop can be tested without training. Covariance and the selected-name set are deterministic and computed once; only `μ` varies per iteration.

**Tech Stack:** Python, numpy, pandas, scikit-learn (`LedoitWolf`), scipy (via `risk_kit.msr_tuned`), pytest. Reuses `src/risk_kit.py`, `src/transformer_model.py`, `pipeline/config.py`.

**Spec:** `docs/superpowers/specs/2026-05-31-allocation-stability-measurement-design.md`. This plan covers **Phase 1 (measurement) only**; Phase 2 (compound-annualization assessment) is a separate cycle — see the spec and `MEMORY.md`.

---

## File Structure

- **Create:** `experiments/measure_allocation_stability.py` — the instrument. One module, organized as: module header + imports → pure allocation/metric helpers → `run_experiment` orchestrator → output/reporting → `main()` CLI.
- **Create:** `tests/test_measure_allocation_stability.py` — unit tests for every pure helper plus a torch-free `run_experiment` loop test using injected stubs.
- **Reuse (no changes):** `src/risk_kit.py` (`msr_tuned`, `portfolio_return`, `portfolio_vol`, `technical_indicators`), `src/transformer_model.py` (`train_and_predict`, `annualize_expected_returns`), `pipeline/config.py` (`load_config`, `PATHS`, `BASE_DIR`).
- **Output dir (created at runtime):** `experiments/results/`.

**Conventions to follow** (from `experiments/compare_training_universe.py` and `tests/test_risk_kit.py`):
- Experiment scripts set `BASE_DIR` then `sys.path.insert` for `src` and `pipeline`, then import `config` / `transformer_model` / `risk_kit` by bare name.
- Tests insert the project root on `sys.path`, use synthetic in-test data only (no files/network/GPU), group with classes, run via `python -m pytest tests/<file> -v`.
- Commit messages: plain imperative, no AI attribution (sole author Jumarti96).

---

### Task 1: Module scaffold + `allocate_msr` (replicates step 4)

**Files:**
- Create: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_measure_allocation_stability.py`:

```python
"""
Tests for experiments/measure_allocation_stability.py

Run with:  python -m pytest tests/test_measure_allocation_stability.py -v

Pure helpers are tested with synthetic data only — no files, network, or GPU.
The run_experiment loop is tested with injected stubs so torch is never imported.
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.measure_allocation_stability import allocate_msr


CFG = {"rf_rate": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}


@pytest.fixture
def three_assets():
    """Annualised returns + a diagonal covariance for three assets."""
    returns = pd.Series({"A": 0.20, "B": 0.12, "C": 0.02})
    covmat = pd.DataFrame(
        np.diag([0.04, 0.05, 0.06]), index=returns.index, columns=returns.index
    )
    return returns, covmat


@pytest.fixture
def five_assets_one_dominated():
    """Four equivalent good assets + one dominated asset (E).

    Equal returns/variance among A-D give ~0.25 each; E's strongly negative
    return forces it to ~0, so the elimination loop drops it (cumulative weight
    < min_weight) and re-optimises the remaining four (len > 2) back to weights
    that still sum to 1. With only three assets, dropping one would trip the
    `len(optimal) <= 2` guard before re-optimising, so >=5 assets are needed to
    exercise the elimination path cleanly.
    """
    returns = pd.Series({"A": 0.15, "B": 0.15, "C": 0.15, "D": 0.15, "E": -0.50})
    covmat = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04, 0.04, 0.04]),
        index=returns.index, columns=returns.index,
    )
    return returns, covmat


class TestAllocateMsr:
    def test_weights_sum_to_one(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_index_preserved(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert list(w.index) == list(returns.index)

    def test_respects_max_weight(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert (w <= CFG["max_weight"] + 1e-6).all()

    def test_no_negative_weights(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert (w >= -1e-8).all()

    def test_dominated_asset_eliminated(self, five_assets_one_dominated):
        # E has a strongly negative return; the optimizer zeroes it, the
        # elimination loop drops it (weight exactly 0), and the remaining four
        # re-optimise to weights that still sum to 1.
        returns, covmat = five_assets_one_dominated
        w = allocate_msr(returns, covmat, CFG)
        assert w["E"] == 0.0
        assert abs(w.sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'allocate_msr'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/measure_allocation_stability.py` with the header, imports, and the first function:

```python
"""
Measure portfolio allocation stability across repeated predict->allocate runs.

Only mu (the Transformer expected returns) changes between runs; the Ledoit-Wolf
covariance and the technical-filter selection are deterministic and computed once.
This instrument runs N iterations and reports how much the *composition* moves
(selection frequency, turnover, Jaccard, weight std) relative to how much the
*value* moves (return/vol/Sharpe dispersion), plus an input->output amplification.

Self-contained (Approach B): the selection and MSR elimination logic are
re-implemented here to mirror pipeline/03_filter.py and pipeline/04_allocate.py.
Keep them in sync if those steps change.

Run:  python experiments/measure_allocation_stability.py --iterations 30 --transformer-runs 10
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

Phase 1 (measurement) only. Phase 2 assesses compound-annualisation of mu — see
docs/superpowers/specs/2026-05-31-allocation-stability-measurement-design.md.
"""

import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import risk_kit as rk
from config import load_config, PATHS


def allocate_msr(returns, covmat, cfg):
    """Sharpe-maximising weights with the step-4 batch-elimination loop.

    Mirrors pipeline/04_allocate.py. Returns a Series over `returns.index`;
    names eliminated for failing the min-weight floor get weight 0.0.
    """
    rf = cfg["rf_rate"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    names = list(returns.index)

    w0 = rk.msr_tuned(
        riskfree_rate=rf, returns=returns, covmat=covmat,
        max_weight=max_w, periods_per_year=ppy, debug=False,
    )
    optimal = (
        pd.DataFrame(w0, index=returns.index, columns=["Weights"])
        .sort_values("Weights")
    )

    while optimal["Weights"].sum() >= 0.9999:
        cum_weights = optimal["Weights"].cumsum()
        failing_mask = cum_weights < min_w
        if not failing_mask.any():
            break
        optimal = optimal[~failing_mask]
        if len(optimal) <= 2:
            break
        w = rk.msr_tuned(
            riskfree_rate=rf, returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
        )
        optimal = (
            pd.DataFrame(w, index=optimal.index, columns=["Weights"])
            .sort_values("Weights")
        )

    weights = pd.Series(0.0, index=names)
    weights[optimal.index] = optimal["Weights"]
    return weights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add allocate_msr to allocation-stability instrument"
```

---

### Task 2: `portfolio_metrics`

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import portfolio_metrics


class TestPortfolioMetrics:
    def test_known_values(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.20})
        covmat = pd.DataFrame(
            np.diag([0.04, 0.09]), index=["A", "B"], columns=["A", "B"]
        )
        m = portfolio_metrics(weights, returns, covmat, rf=0.02)
        assert abs(m["ret"] - 0.15) < 1e-10
        # vol = sqrt(0.25*0.04 + 0.25*0.09) = sqrt(0.0325)
        assert abs(m["vol"] - math.sqrt(0.0325)) < 1e-10
        assert abs(m["sharpe"] - (0.15 - 0.02) / math.sqrt(0.0325)) < 1e-10

    def test_uses_only_weight_index_names(self):
        # returns/covmat carry an extra name C that weights omit; it must be ignored.
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.20, "C": 9.0})
        covmat = pd.DataFrame(
            np.diag([0.04, 0.09, 1.0]), index=["A", "B", "C"], columns=["A", "B", "C"]
        )
        m = portfolio_metrics(weights, returns, covmat, rf=0.0)
        assert abs(m["ret"] - 0.15) < 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py::TestPortfolioMetrics -v`
Expected: FAIL with `ImportError: cannot import name 'portfolio_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py` (after `allocate_msr`):

```python
def portfolio_metrics(weights, returns, covmat, rf):
    """Portfolio return/vol/Sharpe over the names in `weights.index`.

    Computed exactly as msr_tuned's objective does (annualised mu against the
    per-period covariance) so the numbers match the optimiser's convention.
    """
    names = list(weights.index)
    w = weights.values
    r = returns.loc[names].values
    C = covmat.loc[names, names].values
    ret = float(rk.portfolio_return(w, r))
    vol = float(rk.portfolio_vol(w, C))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return {"ret": ret, "vol": vol, "sharpe": sharpe}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py::TestPortfolioMetrics -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add portfolio_metrics to allocation-stability instrument"
```

---

### Task 3: `selection_frequency` and `weight_dispersion`

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import (
    selection_frequency,
    weight_dispersion,
)


@pytest.fixture
def weights_df():
    """3 iterations x 3 names. C is held in 1 of 3 runs."""
    return pd.DataFrame(
        [
            {"A": 0.6, "B": 0.4, "C": 0.0},
            {"A": 0.5, "B": 0.5, "C": 0.0},
            {"A": 0.4, "B": 0.3, "C": 0.3},
        ]
    )


class TestSelectionFrequency:
    def test_fraction_held(self, weights_df):
        freq = selection_frequency(weights_df)
        assert freq["A"] == 1.0
        assert freq["B"] == 1.0
        assert abs(freq["C"] - 1 / 3) < 1e-12

    def test_returns_series_over_all_names(self, weights_df):
        freq = selection_frequency(weights_df)
        assert set(freq.index) == {"A", "B", "C"}


class TestWeightDispersion:
    def test_zero_for_constant_column(self):
        df = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.1, 0.2, 0.3]})
        disp = weight_dispersion(df)
        assert disp["A"] == 0.0
        assert disp["B"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "SelectionFrequency or WeightDispersion" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py`:

```python
def selection_frequency(weights_df, eps=1e-9):
    """Fraction of iterations in which each name is held (weight > eps)."""
    return (weights_df.abs() > eps).mean(axis=0)


def weight_dispersion(weights_df):
    """Population std of each name's weight across iterations."""
    return weights_df.std(axis=0, ddof=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "SelectionFrequency or WeightDispersion" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add selection_frequency and weight_dispersion metrics"
```

---

### Task 4: `mean_turnover` and `mean_jaccard` (with <2-iteration N/A)

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import mean_turnover, mean_jaccard


class TestMeanTurnover:
    def test_two_disjoint_rows(self):
        # rows [1,0] and [0,1]: turnover = 0.5*(|1|+|1|) = 1.0
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        assert abs(mean_turnover(df) - 1.0) < 1e-12

    def test_identical_rows_zero(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}])
        assert mean_turnover(df) == 0.0

    def test_single_row_returns_none(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}])
        assert mean_turnover(df) is None


class TestMeanJaccard:
    def test_identical_sets_is_one(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"A": 0.4, "B": 0.6}])
        assert abs(mean_jaccard(df) - 1.0) < 1e-12

    def test_disjoint_sets_is_zero(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        assert mean_jaccard(df) == 0.0

    def test_single_row_returns_none(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}])
        assert mean_jaccard(df) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "Turnover or Jaccard" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py`:

```python
def mean_turnover(weights_df):
    """Mean over all iteration pairs of 0.5 * L1 weight distance.

    None if fewer than 2 iterations.
    """
    if len(weights_df) < 2:
        return None
    W = weights_df.values
    n = len(W)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0.5 * np.abs(W[i] - W[j]).sum()
            count += 1
    return total / count


def mean_jaccard(weights_df, eps=1e-9):
    """Mean pairwise Jaccard similarity of the held-name sets.

    None if fewer than 2 iterations.
    """
    if len(weights_df) < 2:
        return None
    held = weights_df.abs() > eps
    rows = [set(held.columns[held.iloc[i].values]) for i in range(len(held))]
    sims = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(sims))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "Turnover or Jaccard" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add mean_turnover and mean_jaccard metrics"
```

---

### Task 5: `metric_dispersion` and `amplification_factor`

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import (
    metric_dispersion,
    amplification_factor,
)


class TestMetricDispersion:
    def test_mean_std_cov(self):
        df = pd.DataFrame({"ret": [0.10, 0.20], "vol": [0.05, 0.05], "sharpe": [1.0, 3.0]})
        disp = metric_dispersion(df)
        assert abs(disp["ret"]["mean"] - 0.15) < 1e-12
        assert disp["vol"]["std"] == 0.0
        # CoV = std / |mean| ; sharpe mean=2, std(ddof=0)=1 -> 0.5
        assert abs(disp["sharpe"]["cov"] - 0.5) < 1e-12


class TestAmplificationFactor:
    def test_ratio_of_mean_stds(self):
        # mu std per name: A->0, B->0.5 ; mean = 0.25
        mu_df = pd.DataFrame({"A": [0.10, 0.10], "B": [0.10, 1.10]})
        # weight std per name: A->0.05, B->0.05 ; mean = 0.05
        w_df = pd.DataFrame({"A": [0.45, 0.55], "B": [0.55, 0.45]})
        amp = amplification_factor(mu_df, w_df)
        assert abs(amp - (0.05 / 0.25)) < 1e-9

    def test_zero_input_std_is_nan(self):
        mu_df = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, 0.2]})
        w_df = pd.DataFrame({"A": [0.4, 0.6], "B": [0.6, 0.4]})
        assert math.isnan(amplification_factor(mu_df, w_df))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "MetricDispersion or Amplification" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py`:

```python
def metric_dispersion(metrics_df):
    """mean / population-std / coefficient-of-variation for each metric column."""
    out = {}
    for col in metrics_df.columns:
        s = metrics_df[col]
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        cov = std / abs(mean) if mean != 0 else float("nan")
        out[col] = {"mean": mean, "std": std, "cov": cov}
    return out


def amplification_factor(mu_df, weights_df):
    """Mean per-name weight std divided by mean per-name mu std.

    A rough indicator of how much the optimiser magnifies forecast jitter.
    NaN if the input (mu) dispersion is zero.
    """
    mu_disp = float(mu_df.std(axis=0, ddof=0).mean())
    w_disp = float(weights_df.std(axis=0, ddof=0).mean())
    return w_disp / mu_disp if mu_disp > 0 else float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "MetricDispersion or Amplification" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add metric_dispersion and amplification_factor"
```

---

### Task 6: `select_stocks` and `seed_everything` (replicates step 3)

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import select_stocks


@pytest.fixture
def synthetic_prices_returns():
    """60 weekly points for 4 tickers: 2 uptrending, 2 downtrending."""
    np.random.seed(7)
    n = 60
    idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
    up = lambda drift: 100 * np.cumprod(1 + np.random.normal(drift, 0.01, n))
    prices = pd.DataFrame(
        {
            "UP1": up(0.01), "UP2": up(0.008),
            "DN1": up(-0.01), "DN2": up(-0.008),
        },
        index=idx,
    )
    returns = prices.pct_change().dropna()
    return prices, returns


class TestSelectStocks:
    def test_subset_of_columns(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        selected = select_stocks(prices, returns, cfg)
        assert set(selected).issubset(set(prices.columns))

    def test_deterministic(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        assert select_stocks(prices, returns, cfg) == select_stocks(prices, returns, cfg)

    def test_returns_only_names_present_in_returns(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        selected = select_stocks(prices, returns, cfg)
        assert all(name in returns.columns for name in selected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py::TestSelectStocks -v`
Expected: FAIL with `ImportError: cannot import name 'select_stocks'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py`:

```python
def select_stocks(prices, rets, cfg):
    """Technical-filter selection — mirrors pipeline/03_filter.py.

    Keeps tickers with at least cfg['signal_min_count'] positive signals out of
    SMA/EMA/MACD/PRC. Returns the kept names that also exist in `rets`.
    """
    ma_terms = cfg["ma_terms"]
    rows = []
    for ticker in prices.columns:
        sig = rk.technical_indicators(
            prices[ticker],
            indicators=["SMA", "EMA", "MACD", "PRC"],
            ma_terms=ma_terms,
            macd_params=[12, 26, 9],
            return_df=True,
            plot=False,
            signal_tolerance=0.975,
        ).iloc[-1]
        sig_df = pd.DataFrame(sig).T
        sig_df.index = [ticker]
        sig_df.rename(columns={ticker: "Price"}, inplace=True)
        rows.append(sig_df)

    signals = pd.concat(rows, axis=0)
    keep = signals[
        np.int64(signals["MACD Signal"])
        + np.int64(signals[f"SMA{ma_terms} Signal"])
        + np.int64(signals[f"EMA{ma_terms} Signal"])
        + np.int64(signals["PRC Signal"])
        >= cfg["signal_min_count"]
    ]
    return [t for t in keep.index if t in rets.columns]


def seed_everything(seed):
    """Seed numpy and torch (torch imported lazily to keep helpers light)."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py::TestSelectStocks -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add select_stocks and seed_everything"
```

---

### Task 7: `run_experiment` (orchestration loop, torch-free via injected stubs)

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import run_experiment


class TestRunExperiment:
    def _inputs(self):
        np.random.seed(1)
        n = 40
        idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
        cols = ["A", "B", "C", "D"]
        rets = pd.DataFrame(np.random.normal(0.002, 0.02, (n, 4)), index=idx, columns=cols)
        prices = (1 + rets).cumprod() * 100
        cfg = {
            "rf_rate": 0.0, "max_weight": 0.6, "min_weight": 0.05,
            "periods_per_year": 12,
        }
        return prices, rets, cfg

    def _stubs(self):
        # predict_fn: returns 2 forecast periods that drift each call so mu varies.
        state = {"k": 0}

        def predict_fn(rets, cfg, n_runs=None, verbose=True):
            state["k"] += 1
            base = np.array([0.01, 0.008, 0.006, 0.002])
            shift = 0.001 * state["k"]
            vals = np.vstack([base + shift, base + shift])
            return pd.DataFrame(vals, columns=rets.columns)

        def annualize_fn(preds, ppy):
            return preds.mean(axis=0)

        def select_fn(prices, rets, cfg):
            return ["A", "B", "C", "D"]

        def seed_fn(seed):
            pass

        return predict_fn, annualize_fn, select_fn, seed_fn

    def test_output_shapes(self):
        prices, rets, cfg = self._inputs()
        predict_fn, annualize_fn, select_fn, seed_fn = self._stubs()
        result = run_experiment(
            prices, rets, cfg, iterations=5, transformer_runs=2, seed=0,
            predict_fn=predict_fn, annualize_fn=annualize_fn,
            select_fn=select_fn, seed_fn=seed_fn,
        )
        assert result["weights"].shape == (5, 4)
        assert result["mu"].shape == (5, 4)
        assert set(result["metrics"].columns) == {"ret", "vol", "sharpe"}
        assert len(result["metrics"]) == 5
        assert result["selected"] == ["A", "B", "C", "D"]

    def test_weights_rows_sum_to_one(self):
        prices, rets, cfg = self._inputs()
        predict_fn, annualize_fn, select_fn, seed_fn = self._stubs()
        result = run_experiment(
            prices, rets, cfg, iterations=3, transformer_runs=2, seed=0,
            predict_fn=predict_fn, annualize_fn=annualize_fn,
            select_fn=select_fn, seed_fn=seed_fn,
        )
        sums = result["weights"].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py::TestRunExperiment -v`
Expected: FAIL with `ImportError: cannot import name 'run_experiment'`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py`:

```python
def run_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                   predict_fn=None, annualize_fn=None, select_fn=None, seed_fn=None):
    """Run `iterations` predict->allocate passes and collect the raw evidence.

    Covariance (Ledoit-Wolf) and the selected-name set are deterministic and
    computed once; only mu changes per iteration. The four *_fn arguments are
    dependency-injection seams (default to the real implementations) so the loop
    can be tested without importing torch.

    Returns dict with DataFrames: 'mu' and 'weights' (iterations x selected),
    'metrics' (iterations x [ret, vol, sharpe]), and 'selected' (list).
    """
    if predict_fn is None or annualize_fn is None:
        from transformer_model import train_and_predict, annualize_expected_returns
        predict_fn = predict_fn or train_and_predict
        annualize_fn = annualize_fn or annualize_expected_returns
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    ppy = cfg["periods_per_year"]
    rf = cfg["rf_rate"]

    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    mu_records, weight_records, metric_records = [], [], []
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        preds = predict_fn(rets, cfg, n_runs=transformer_runs, verbose=False)
        mu = annualize_fn(preds, ppy)
        mu_sel = mu.loc[selected]
        weights = allocate_msr(mu_sel, cov_sel, cfg)
        survivors = weights[weights.abs() > 1e-9]
        metrics = portfolio_metrics(survivors, mu_sel, cov_sel, rf)

        mu_records.append(mu_sel)
        weight_records.append(weights)
        metric_records.append(metrics)

    return {
        "mu": pd.DataFrame(mu_records).reset_index(drop=True),
        "weights": pd.DataFrame(weight_records).reset_index(drop=True),
        "metrics": pd.DataFrame(metric_records),
        "selected": selected,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py::TestRunExperiment -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add run_experiment orchestration loop"
```

---

### Task 8: `format_summary` and `write_outputs`

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import format_summary, write_outputs


def _sample_result(iterations=3):
    names = ["A", "B", "C"]
    weights = pd.DataFrame(
        [
            {"A": 0.6, "B": 0.4, "C": 0.0},
            {"A": 0.5, "B": 0.5, "C": 0.0},
            {"A": 0.4, "B": 0.3, "C": 0.3},
        ][:iterations]
    )
    mu = pd.DataFrame(
        [
            {"A": 0.20, "B": 0.12, "C": 0.05},
            {"A": 0.21, "B": 0.11, "C": 0.06},
            {"A": 0.19, "B": 0.13, "C": 0.04},
        ][:iterations]
    )
    metrics = pd.DataFrame(
        [
            {"ret": 0.16, "vol": 0.10, "sharpe": 1.6},
            {"ret": 0.17, "vol": 0.10, "sharpe": 1.7},
            {"ret": 0.15, "vol": 0.11, "sharpe": 1.4},
        ][:iterations]
    )
    return {"mu": mu, "weights": weights, "metrics": metrics, "selected": names}


class TestFormatSummary:
    def test_contains_key_sections(self):
        text = format_summary(_sample_result())
        assert "Composition instability" in text
        assert "Value stability" in text
        assert "Selection frequency" in text

    def test_single_iteration_shows_na(self):
        text = format_summary(_sample_result(iterations=1))
        assert "N/A" in text  # turnover / jaccard undefined for 1 iteration


class TestWriteOutputs:
    def test_writes_three_files(self, tmp_path):
        paths = write_outputs(_sample_result(), str(tmp_path))
        for key in ("weights", "metrics", "summary"):
            assert os.path.exists(paths[key])

    def test_weights_csv_roundtrips(self, tmp_path):
        result = _sample_result()
        paths = write_outputs(result, str(tmp_path))
        reloaded = pd.read_csv(paths["weights"], index_col=0)
        assert reloaded.shape == result["weights"].shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "FormatSummary or WriteOutputs" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `experiments/measure_allocation_stability.py`:

```python
def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def format_summary(result):
    """Build the human-readable stability report from a run_experiment result."""
    weights_df = result["weights"]
    metrics_df = result["metrics"]
    mu_df = result["mu"]

    freq = selection_frequency(weights_df)
    wstd = weight_dispersion(weights_df)
    turnover = mean_turnover(weights_df)
    jacc = mean_jaccard(weights_df)
    disp = metric_dispersion(metrics_df)
    amp = amplification_factor(mu_df, weights_df)

    lines = []
    lines.append(f"Iterations: {len(weights_df)} | Selected names: {weights_df.shape[1]}")
    lines.append("")
    lines.append("== Composition instability ==")
    lines.append(f"Mean pairwise turnover: {_fmt(turnover)}")
    lines.append(f"Mean pairwise Jaccard:  {_fmt(jacc)}")
    lines.append("")
    lines.append("Selection frequency (fraction of iterations held):")
    for name, v in freq.sort_values(ascending=False).items():
        lines.append(f"  {name:<14} {v:6.1%}   weight std {wstd[name]:.4f}")
    lines.append("")
    lines.append("== Value stability ==")
    for col in ["ret", "vol", "sharpe"]:
        d = disp[col]
        lines.append(
            f"  {col:<7} mean {d['mean']:.4f}  std {d['std']:.4f}  CoV {_fmt(d['cov'])}"
        )
    lines.append("")
    lines.append(f"Amplification (mean weight std / mean mu std): {_fmt(amp)}")
    lines.append("")
    lines.append(
        "Note: fewer transformer-runs => noisier mu => MORE instability. This is a "
        "conservative upper bound on production (100-run) instability; re-run with "
        "--transformer-runs 100 for the production figure."
    )
    return "\n".join(lines)


def write_outputs(result, outdir):
    """Write weights/metrics CSVs and the summary text; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "weights": os.path.join(outdir, "stability_weights.csv"),
        "metrics": os.path.join(outdir, "stability_metrics.csv"),
        "summary": os.path.join(outdir, "stability_summary.txt"),
    }
    result["weights"].to_csv(paths["weights"], index_label="iteration")
    result["metrics"].to_csv(paths["metrics"], index_label="iteration")
    with open(paths["summary"], "w") as f:
        f.write(format_summary(result))
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_allocation_stability.py -k "FormatSummary or WriteOutputs" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add format_summary and write_outputs reporting"
```

---

### Task 9: `main()` CLI wiring + full-suite + manual smoke verification

**Files:**
- Modify: `experiments/measure_allocation_stability.py`

- [ ] **Step 1: Add the CLI entry point**

Add to the end of `experiments/measure_allocation_stability.py`:

```python
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of predict->allocate runs (default 30)")
    parser.add_argument("--transformer-runs", type=int, default=10,
                        help="n_runs passed to train_and_predict per iteration (default 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; iteration i uses seed+i (default 0)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results"),
                        help="Directory for output CSVs and summary")
    args = parser.parse_args()

    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    print(f"Universe: {rets.shape[1]} stocks | iterations={args.iterations} | "
          f"transformer-runs={args.transformer_runs} | seed={args.seed}")

    result = run_experiment(
        prices, rets, cfg,
        iterations=args.iterations,
        transformer_runs=args.transformer_runs,
        seed=args.seed,
    )
    paths = write_outputs(result, args.outdir)

    print()
    print(format_summary(result))
    print(f"\nSaved: {paths['weights']}\n       {paths['metrics']}\n       {paths['summary']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full unit-test suite**

Run: `python -m pytest tests/test_measure_allocation_stability.py -v`
Expected: PASS (all tests from Tasks 1-8, ~28 tests).

- [ ] **Step 3: Manual smoke run (fast settings)**

Run: `python experiments/measure_allocation_stability.py --iterations 3 --transformer-runs 2`
Expected: prints the summary report; creates `experiments/results/stability_weights.csv`, `stability_metrics.csv`, `stability_summary.txt`. (Requires `data/01_prices.csv` and `data/01_returns.csv`.)

- [ ] **Step 4: Manual determinism check**

Run the same fast command twice writing to two dirs, then compare the weight matrices:

```bash
python experiments/measure_allocation_stability.py --iterations 3 --transformer-runs 2 --seed 1 --outdir experiments/results/_det_a
python experiments/measure_allocation_stability.py --iterations 3 --transformer-runs 2 --seed 1 --outdir experiments/results/_det_b
python -c "import pandas as pd; a=pd.read_csv('experiments/results/_det_a/stability_weights.csv'); b=pd.read_csv('experiments/results/_det_b/stability_weights.csv'); import numpy as np; print('IDENTICAL' if np.allclose(a.values, b.values) else 'DIFFERS')"
```

Expected: `IDENTICAL` (confirms per-iteration seeding makes the experiment reproducible). If it prints `DIFFERS`, the GPU/AMP path may introduce nondeterminism — note it in the summary rather than treating it as a test failure. Clean up the temp dirs afterward (`rm -rf experiments/results/_det_a experiments/results/_det_b`).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py
git commit -m "Add main CLI entry point for allocation-stability instrument"
```

---

## Self-Review

**1. Spec coverage:**
- Reusable self-contained instrument under `experiments/` → Tasks 1-9. ✓
- Approach B (local re-implementation of selection + elimination) → `select_stocks` (Task 6), `allocate_msr` (Task 1), with docstrings flagging the mirror of steps 3/4. ✓
- CLI args `--iterations` (default 30), `--transformer-runs` (default 10), `--seed`, `--outdir` → Task 9. ✓
- Computed-once covariance + selection; per-iteration seed/predict/annualize/restrict/allocate/metrics → `run_experiment` (Task 7). ✓
- Metrics: selection frequency, turnover, Jaccard, weight std, value dispersion (ret/vol/sharpe CoV), amplification factor → Tasks 3-5, surfaced in `format_summary` (Task 8). ✓
- Outputs `stability_weights.csv`, `stability_metrics.csv`, `stability_summary.txt` → `write_outputs` (Task 8). ✓
- Numbers-only (no plots) → no plotting code anywhere. ✓
- Fidelity preserved (annualized mu vs per-period Σ not "corrected") → `portfolio_metrics` docstring + uses `rk.portfolio_return`/`portfolio_vol` directly (Task 2). ✓
- Testing: determinism check (Task 9 Step 4), iterations=1 N/A handling (Task 4 + Task 8 tests), fidelity/selection determinism (Task 6 tests). ✓
- Interpretation note (10-run = conservative upper bound) → baked into `format_summary` (Task 8). ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step shows complete code; every test step shows complete asserts. ✓

**3. Type consistency:** `run_experiment` returns `{"mu","weights","metrics","selected"}`; `format_summary`/`write_outputs` consume exactly those keys. `allocate_msr` returns a Series over `returns.index`; `run_experiment` collects those into `weights` DataFrame and passes survivor-filtered Series to `portfolio_metrics(weights, returns, covmat, rf)` — signature matches Task 2. `metric_dispersion` returns `{col: {"mean","std","cov"}}`; `format_summary` reads `d["mean"]/d["std"]/d["cov"]`. Metric columns `ret/vol/sharpe` consistent across Tasks 2, 7, 8. ✓

No gaps found.
