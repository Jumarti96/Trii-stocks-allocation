# Allocation Walk-Forward Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a walk-forward backtest that scores six allocation arms (current, parametric Michaud s=1/2/4, empirical Michaud, equal-weight) on realized out-of-sample returns, to decide which is closest to reality.

**Architecture:** One new module `experiments/backtest_allocation.py` holds pure metric helpers, an arm dispatcher, the walk-forward driver, output writers, and a CLI. It *reuses* the existing Phase-3b functions (`allocate_msr`, `resampled_allocate`, `sample_mu_draws`, `select_stocks`, `train_runs_as_preds`, `seed_everything`) from `experiments/measure_allocation_stability.py` and `load_config`/`PATHS` from `pipeline/config.py` — no duplication. At each monthly rebalance the transformer is retrained on data up to that date (expanding window) and every arm reuses that one forecast; realized returns come from buy-and-holding each arm's weights over the next block of *actual* returns.

**Tech Stack:** Python, numpy, pandas, scipy (via `risk_kit.msr_tuned`), scikit-learn (`LedoitWolf`), pytest. Tests are torch-free via dependency-injected stubs (the real forecaster is exercised only by the actual run).

**Conventions:**
- Run tests with the project interpreter: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`.
- Commit messages: imperative, capitalised, **no AI attribution** (no `Co-Authored-By`, no "Generated with").
- All new code in `experiments/backtest_allocation.py`; all new tests in `tests/test_backtest_allocation.py`.
- Work on branch `enh-allocation-backtest` (already created; the design spec is committed there).
- Local-only (experiments/tests/docs are not pushed).

---

## File Structure

- **Create:** `experiments/backtest_allocation.py` — the whole feature:
  - module header + imports (Task 1),
  - metric helpers `realized_block_return`, `pairwise_turnover`, `max_drawdown` (Task 1),
  - `annualized_stats`, `summarize_arm` (Task 2),
  - `equal_weight`, `label_of`, `compute_arm_weights` (Task 3),
  - `run_backtest` (Task 4),
  - `format_backtest_summary`, `write_backtest_outputs` (Task 5),
  - `parse_spreads`, `build_arg_parser`, `main` (Task 6).
- **Create:** `tests/test_backtest_allocation.py` — one test class per task.
- **Reference (do not modify):** `experiments/measure_allocation_stability.py` (reused functions), `pipeline/config.py`, `src/risk_kit.py`.

**Reused signatures (already exist on `main`, do not change):**
- `allocate_msr(returns, covmat, cfg) -> Series`
- `resampled_allocate(per_run_mu, covmat, cfg, eliminate_per_draw=False, eps=1e-9) -> (Series, DataFrame)`
- `sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng) -> list[Series]`
- `select_stocks(prices, rets, cfg) -> list[str]`
- `train_runs_as_preds(rets, cfg, n_runs=None, verbose=False) -> list[DataFrame]`
- `seed_everything(seed)`

---

### Task 1: Module header + metric helpers

Pure, torch-free helpers: realized buy-and-hold block return, target-to-target turnover, and max drawdown.

**Files:**
- Create: `experiments/backtest_allocation.py`
- Test: `tests/test_backtest_allocation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest_allocation.py`:

```python
import math
import os

import numpy as np
import pandas as pd
import pytest

from experiments.backtest_allocation import (
    realized_block_return, pairwise_turnover, max_drawdown,
)

CFG = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6,
       "min_weight": 0.05, "periods_per_year": 12}


class TestMetricHelpers:
    def test_block_return_single_asset(self):
        w = pd.Series({"A": 1.0})
        block = pd.DataFrame({"A": [0.1, 0.1]})
        assert abs(realized_block_return(w, block) - (1.1 * 1.1 - 1)) < 1e-9

    def test_block_return_two_assets(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        block = pd.DataFrame({"A": [0.1, 0.0], "B": [0.0, 0.0]})
        # A compounds to 0.10, B to 0.0 -> 0.5*0.1 + 0.5*0 = 0.05
        assert abs(realized_block_return(w, block) - 0.05) < 1e-9

    def test_block_return_ignores_unheld_columns(self):
        w = pd.Series({"A": 1.0})
        block = pd.DataFrame({"A": [0.0], "B": [0.5]})
        assert abs(realized_block_return(w, block) - 0.0) < 1e-9

    def test_turnover_identical_zero(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        assert pairwise_turnover(w, w) == 0.0

    def test_turnover_disjoint_one(self):
        assert abs(pairwise_turnover(pd.Series({"A": 1.0}), pd.Series({"B": 1.0})) - 1.0) < 1e-9

    def test_turnover_partial(self):
        assert abs(pairwise_turnover(pd.Series({"A": 1.0}),
                                     pd.Series({"A": 0.5, "B": 0.5})) - 0.5) < 1e-9

    def test_max_drawdown_monotonic_zero(self):
        assert max_drawdown([0.1, 0.1, 0.1]) == 0.0

    def test_max_drawdown_decline(self):
        # equity [1.5, 0.75]; peak 1.5; trough drawdown = 0.5
        assert abs(max_drawdown([0.5, -0.5]) - 0.5) < 1e-9

    def test_max_drawdown_empty(self):
        assert max_drawdown([]) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestMetricHelpers -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.backtest_allocation'`.

- [ ] **Step 3: Create the module with header + helpers**

Create `experiments/backtest_allocation.py`:

```python
"""
Walk-forward backtest comparing allocation methods on realized out-of-sample returns.

At each monthly rebalance the transformer is retrained on data up to that date (expanding
window); every arm reuses that one forecast. Arms: current, parametric Michaud (s sweep),
empirical Michaud, equal-weight. Frictionless headline; per-rebalance turnover and weights
are saved so net-of-cost analysis is a free follow-up.

Run:  python experiments/backtest_allocation.py --oos-periods 162 --rebalance-every 4 --n-runs 50
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).
"""

import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "experiments"))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import load_config, PATHS
from measure_allocation_stability import (
    allocate_msr, resampled_allocate, sample_mu_draws,
    select_stocks, train_runs_as_preds, seed_everything,
)


def realized_block_return(weights, block_rets):
    """Buy-and-hold realized return of `weights` over a block of actual per-period returns.

    weights: Series of target weights over held names. block_rets: DataFrame (periods x names)
    of ACTUAL per-period simple returns for the block. Each held name compounds over the block;
    the portfolio return is the weighted sum of per-name compounded returns. Only names present
    in both weights and block_rets are used. Returns a float.
    """
    names = [n for n in weights.index if n in block_rets.columns]
    compounded = (1.0 + block_rets[names]).prod(axis=0) - 1.0
    return float((weights[names] * compounded).sum())


def pairwise_turnover(w_prev, w_new):
    """Half the L1 distance between two target-weight Series.

    Aligns on the union of names (missing = 0). Returns a float in [0, 1].
    """
    names = w_prev.index.union(w_new.index)
    a = w_prev.reindex(names, fill_value=0.0)
    b = w_new.reindex(names, fill_value=0.0)
    return float(0.5 * (a - b).abs().sum())


def max_drawdown(block_returns):
    """Maximum drawdown of the equity curve built by compounding `block_returns`.

    Returns the max peak-to-trough decline as a non-negative fraction (0.2 = 20% drop).
    0.0 if the curve never declines or the input is empty.
    """
    r = np.asarray(block_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / running_max
    return float(drawdowns.max())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestMetricHelpers -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/backtest_allocation.py tests/test_backtest_allocation.py
git commit -m "Add backtest metric helpers (block return, turnover, max drawdown)"
```

---

### Task 2: Annualized stats + per-arm summary

`annualized_stats` turns a series of per-block returns into cum/annualized return, vol, Sharpe; `summarize_arm` adds max DD, mean turnover, avg names held, and hit rate.

**Files:**
- Modify: `experiments/backtest_allocation.py`
- Test: `tests/test_backtest_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_allocation.py`:

```python
from experiments.backtest_allocation import annualized_stats, summarize_arm


class TestAnnualizedStats:
    def test_zero_vol_constant_returns(self):
        out = annualized_stats([0.02] * 12, blocks_per_year=12, rf_period=0.0)
        assert abs(out["cum_return"] - (1.02 ** 12 - 1)) < 1e-9
        assert abs(out["ann_return"] - (1.02 ** 12 - 1)) < 1e-9   # n == blocks_per_year
        assert out["ann_vol"] == 0.0
        assert math.isnan(out["sharpe"])

    def test_vol_and_sharpe(self):
        out = annualized_stats([0.1, -0.1], blocks_per_year=4, rf_period=0.0)
        assert abs(out["ann_vol"] - 0.2) < 1e-9       # std 0.1 * sqrt(4)
        assert abs(out["sharpe"] - 0.0) < 1e-9        # mean 0
        assert abs(out["cum_return"] - (1.1 * 0.9 - 1)) < 1e-9

    def test_empty(self):
        out = annualized_stats([], blocks_per_year=12, rf_period=0.0)
        assert math.isnan(out["ann_return"])


class TestSummarizeArm:
    def test_full_row(self):
        out = summarize_arm([0.1, -0.1, 0.1], [None, 0.2, 0.4], [3, 4, 3],
                            blocks_per_year=12, rf_period=0.0)
        assert set(out) >= {"cum_return", "ann_return", "ann_vol", "sharpe",
                            "max_dd", "mean_turnover", "avg_names", "hit_rate"}
        assert abs(out["mean_turnover"] - 0.3) < 1e-9      # mean of 0.2, 0.4 (None ignored)
        assert abs(out["avg_names"] - (10 / 3)) < 1e-9
        assert abs(out["hit_rate"] - (2 / 3)) < 1e-9       # 2 of 3 blocks positive
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestAnnualizedStats tests/test_backtest_allocation.py::TestSummarizeArm -v`
Expected: FAIL with `ImportError: cannot import name 'annualized_stats'`.

- [ ] **Step 3: Implement the stats functions**

Add to `experiments/backtest_allocation.py` (after `max_drawdown`):

```python
def annualized_stats(block_returns, blocks_per_year, rf_period):
    """Annualized return/vol/Sharpe from a series of per-block simple returns.

    block_returns: per-block returns (each block spans one rebalance). blocks_per_year:
    blocks per year (periods_per_year / rebalance_every). rf_period: the per-BLOCK
    risk-free return for the Sharpe excess. Returns dict {cum_return, ann_return (geometric),
    ann_vol, sharpe}. NaN entries when the series is empty or has zero volatility.
    """
    r = np.asarray(block_returns, dtype=float)
    n = len(r)
    if n == 0:
        return {"cum_return": float("nan"), "ann_return": float("nan"),
                "ann_vol": float("nan"), "sharpe": float("nan")}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann_return = float((1.0 + cum) ** (blocks_per_year / n) - 1.0)
    vol_block = float(r.std(ddof=0))
    ann_vol = vol_block * math.sqrt(blocks_per_year)
    if vol_block > 0:
        sharpe = ((float(r.mean()) - rf_period) / vol_block) * math.sqrt(blocks_per_year)
    else:
        sharpe = float("nan")
    return {"cum_return": cum, "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe}


def summarize_arm(block_returns, turnovers, n_held, blocks_per_year, rf_period):
    """Full per-arm metric row: annualized stats + max DD + turnover + composition.

    turnovers may contain None (e.g. the first rebalance) -- None entries are ignored.
    n_held: per-block count of held names. Returns dict with cum_return, ann_return,
    ann_vol, sharpe, max_dd, mean_turnover, avg_names, hit_rate.
    """
    out = dict(annualized_stats(block_returns, blocks_per_year, rf_period))
    r = np.asarray(block_returns, dtype=float)
    valid_turn = [t for t in turnovers if t is not None]
    out["max_dd"] = max_drawdown(block_returns)
    out["mean_turnover"] = float(np.mean(valid_turn)) if valid_turn else float("nan")
    out["avg_names"] = float(np.mean(n_held)) if len(n_held) else float("nan")
    out["hit_rate"] = float((r > 0).mean()) if len(r) else float("nan")
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestAnnualizedStats tests/test_backtest_allocation.py::TestSummarizeArm -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/backtest_allocation.py tests/test_backtest_allocation.py
git commit -m "Add annualized stats and per-arm metric summary"
```

---

### Task 3: Arm dispatcher + equal-weight + label helper

`compute_arm_weights` maps an arm identifier to a target-weight Series, reusing the existing allocators; `equal_weight` is the 1/N benchmark; `label_of` gives each arm a stable string label.

**Files:**
- Modify: `experiments/backtest_allocation.py`
- Test: `tests/test_backtest_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_allocation.py`:

```python
from experiments.backtest_allocation import equal_weight, label_of, compute_arm_weights


@pytest.fixture
def cov5():
    names = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.diag([0.04] * 5), index=names, columns=names)


def _mu5(vals):
    return pd.Series(dict(zip(["A", "B", "C", "D", "E"], vals)))


class TestLabelOf:
    def test_labels(self):
        assert label_of("current") == "current"
        assert label_of("empirical") == "empirical"
        assert label_of(("parametric", 2.0)) == "parametric_s2"
        assert label_of(("parametric", 1.0)) == "parametric_s1"


class TestComputeArmWeights:
    def test_current_sums_to_one(self, cov5):
        w = compute_arm_weights("current", _mu5([0.2, 0.02, 0.02, 0.02, 0.02]), None,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
        assert abs(w.sum() - 1.0) < 1e-6

    def test_equal_weight(self, cov5):
        w = compute_arm_weights("equal_weight", _mu5([0.1] * 5), None,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
        assert (abs(w - 0.2) < 1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-9

    def test_empirical_sums_to_one(self, cov5):
        per_run = [_mu5([0.2, 0.02, 0.02, 0.02, 0.02]), _mu5([0.02, 0.2, 0.02, 0.02, 0.02])]
        w = compute_arm_weights("empirical", _mu5([0.1] * 5), per_run,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
        assert abs(w.sum() - 1.0) < 1e-6

    def test_parametric_sums_to_one(self, cov5):
        w = compute_arm_weights(("parametric", 2.0), _mu5([0.2, 0.02, 0.02, 0.02, 0.02]),
                                None, cov5, CFG, 100, 500, np.random.default_rng(0))
        assert abs(w.sum() - 1.0) < 1e-6

    def test_unknown_arm_raises(self, cov5):
        with pytest.raises(ValueError):
            compute_arm_weights("bogus", _mu5([0.1] * 5), None,
                                cov5, CFG, 100, 50, np.random.default_rng(0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestLabelOf tests/test_backtest_allocation.py::TestComputeArmWeights -v`
Expected: FAIL with `ImportError: cannot import name 'equal_weight'`.

- [ ] **Step 3: Implement the arm dispatcher, equal-weight, and label helper**

Add to `experiments/backtest_allocation.py` (after `summarize_arm`):

```python
def equal_weight(names):
    """Equal (1/N) weight Series over `names`."""
    names = list(names)
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series([1.0 / n] * n, index=names)


def label_of(arm):
    """Stable string label for an arm identifier."""
    if isinstance(arm, tuple) and arm[0] == "parametric":
        return f"parametric_s{arm[1]:g}"
    return arm


def compute_arm_weights(arm, mu_bar, per_run_mu, covmat, cfg, n_periods, mc_draws, rng):
    """Map an arm identifier to a target-weight Series over the eligible names.

    arm: "current" | "empirical" | "equal_weight" | ("parametric", s). mu_bar: averaged
    per-period mu over the eligible names. per_run_mu: list of per-run mu Series (eligible
    names) for the empirical arm. covmat: Ledoit-Wolf cov over the eligible names. n_periods:
    T backing covmat (parametric Sigma/T scale). mc_draws: K parametric draws. rng: numpy
    Generator. Reuses allocate_msr / resampled_allocate / sample_mu_draws. Returns a Series
    over mu_bar.index (dropped names = 0.0).
    """
    if arm == "current":
        return allocate_msr(mu_bar, covmat, cfg)
    if arm == "equal_weight":
        return equal_weight(list(mu_bar.index))
    if arm == "empirical":
        consensus, _ = resampled_allocate(per_run_mu, covmat, cfg)
        return consensus
    if isinstance(arm, tuple) and arm[0] == "parametric":
        draws = sample_mu_draws(mu_bar, covmat, n_periods, mc_draws, arm[1], rng)
        consensus, _ = resampled_allocate(draws, covmat, cfg)
        return consensus
    raise ValueError(f"unknown arm: {arm!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestLabelOf tests/test_backtest_allocation.py::TestComputeArmWeights -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/backtest_allocation.py tests/test_backtest_allocation.py
git commit -m "Add arm dispatcher, equal-weight benchmark, and label helper"
```

---

### Task 4: Walk-forward driver `run_backtest`

The orchestration: step monthly through the OOS window, retrain once per rebalance (expanding window), build every arm's weights from that shared forecast, and realize buy-and-hold returns over the next block.

**Files:**
- Modify: `experiments/backtest_allocation.py`
- Test: `tests/test_backtest_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_allocation.py`:

```python
from experiments.backtest_allocation import run_backtest


class TestRunBacktest:
    def _data(self):
        np.random.seed(0)
        n = 40
        idx = pd.date_range("2020-01-01", periods=n, freq="W-SUN")
        cols = ["A", "B", "C", "D", "E"]
        rets = pd.DataFrame(np.random.normal(0.001, 0.02, (n, 5)), index=idx, columns=cols)
        prices = (1 + rets).cumprod() * 100
        return prices, rets, dict(CFG)

    def _stubs(self, seen=None):
        cols = ["A", "B", "C", "D", "E"]

        def runs_fn(rets, cfg, n_runs=None, verbose=True):
            if seen is not None:
                seen.append(len(rets))
            n_runs = n_runs or 3
            return [pd.DataFrame(np.full((2, 5), 0.01), columns=cols) for _ in range(n_runs)]

        def period_mu_fn(preds):
            return preds.mean(axis=0)

        def select_fn(prices, rets, cfg):
            return cols

        def seed_fn(seed):
            pass

        return runs_fn, period_mu_fn, select_fn, seed_fn

    def _run(self, seen=None, spreads=(1.0, 2.0, 4.0)):
        prices, rets, cfg = self._data()
        runs_fn, period_mu_fn, select_fn, seed_fn = self._stubs(seen)
        return run_backtest(
            prices, rets, cfg, oos_periods=12, rebalance_every=4, n_runs=3,
            mc_draws=200, spreads=list(spreads), seed=0,
            runs_fn=runs_fn, period_mu_fn=period_mu_fn, select_fn=select_fn, seed_fn=seed_fn,
        )

    def test_rebalance_count_and_arms(self):
        res = self._run()
        assert res["rebalance_index"] == [28, 32, 36]
        for label in ["current", "parametric_s1", "parametric_s2", "parametric_s4",
                      "empirical", "equal_weight"]:
            assert len(res[label]["block_returns"]) == 3

    def test_no_lookahead(self):
        seen = []
        self._run(seen=seen)
        # forecaster called once per rebalance on the expanding history up to t (< T=40)
        assert seen == [28, 32, 36]

    def test_weights_sum_to_one(self):
        res = self._run()
        for label in ["current", "parametric_s2", "empirical", "equal_weight"]:
            for w in res[label]["weights"]:
                held = w[w.abs() > 1e-9]
                assert abs(held.sum() - 1.0) < 1e-6

    def test_turnover_first_is_none(self):
        res = self._run()
        assert res["current"]["turnover"][0] is None
        assert res["current"]["turnover"][1] is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestRunBacktest -v`
Expected: FAIL with `ImportError: cannot import name 'run_backtest'`.

- [ ] **Step 3: Implement `run_backtest`**

Add to `experiments/backtest_allocation.py` (after `compute_arm_weights`):

```python
def run_backtest(prices, rets, cfg, oos_periods, rebalance_every, n_runs, mc_draws,
                 spreads, seed, arms=None, runs_fn=None, period_mu_fn=None,
                 select_fn=None, seed_fn=None):
    """Walk-forward backtest: retrain once per rebalance and score all arms on realized returns.

    Steps t from len(rets)-oos_periods by rebalance_every while a full block (t+rebalance_every)
    fits. At each t: train on rets[:t] (expanding), filter on prices[:t], Sigma=Ledoit-Wolf(rets[:t]),
    build each arm's weights over the eligible names, then realize buy-and-hold over
    rets[t:t+rebalance_every]. arms defaults to the 6-arm set built from `spreads`. The four *_fn
    are dependency-injection seams (default to the real torch-backed implementations).

    Returns dict label -> {block_returns, weights (list of Series), turnover (list, first None),
    n_held (list), dates (list)}, plus "rebalance_index" -> list of t indices.
    """
    if arms is None:
        arms = (["current"]
                + [("parametric", s) for s in spreads]
                + ["empirical", "equal_weight"])
    if runs_fn is None:
        runs_fn = train_runs_as_preds
    if period_mu_fn is None:
        from transformer_model import weighted_mean_return
        period_mu_fn = weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    T = len(rets)
    start = T - oos_periods
    rebalance_index = list(range(start, T - rebalance_every + 1, rebalance_every))

    labels = [label_of(a) for a in arms]
    results = {lab: {"block_returns": [], "weights": [], "turnover": [],
                     "n_held": [], "dates": []} for lab in labels}
    prev_weights = {lab: None for lab in labels}

    for k, t in enumerate(rebalance_index):
        seed_fn(seed + k)
        rets_hist = rets.iloc[:t]
        prices_hist = prices.iloc[:t]
        runs = runs_fn(rets_hist, cfg, n_runs=n_runs, verbose=False)
        selected = select_fn(prices_hist, rets_hist, cfg)
        covmat = pd.DataFrame(
            LedoitWolf().fit(rets_hist).covariance_,
            index=rets_hist.columns, columns=rets_hist.columns,
        )
        cov_sel = covmat.loc[selected, selected]
        preds_avg = sum(r.values for r in runs) / len(runs)
        preds_avg = pd.DataFrame(preds_avg, columns=runs[0].columns)
        mu_bar = period_mu_fn(preds_avg).loc[selected]
        per_run_mu = [period_mu_fn(r).loc[selected] for r in runs]

        block = rets.iloc[t:t + rebalance_every]
        rng = np.random.default_rng(seed + k)
        for arm, lab in zip(arms, labels):
            w = compute_arm_weights(arm, mu_bar, per_run_mu, cov_sel, cfg,
                                    len(rets_hist), mc_draws, rng)
            held = w[w.abs() > 1e-9]
            tn = pairwise_turnover(prev_weights[lab], w) if prev_weights[lab] is not None else None
            results[lab]["block_returns"].append(realized_block_return(held, block))
            results[lab]["weights"].append(w)
            results[lab]["turnover"].append(tn)
            results[lab]["n_held"].append(int((w.abs() > 1e-9).sum()))
            results[lab]["dates"].append(rets.index[t])
            prev_weights[lab] = w

    results["rebalance_index"] = rebalance_index
    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestRunBacktest -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/backtest_allocation.py tests/test_backtest_allocation.py
git commit -m "Add walk-forward run_backtest driver"
```

---

### Task 5: Summary table + output writers

`format_backtest_summary` builds the headline metric table; `write_backtest_outputs` writes the returns/turnover/per-arm-weights CSVs and the summary text.

**Files:**
- Modify: `experiments/backtest_allocation.py`
- Test: `tests/test_backtest_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_allocation.py`:

```python
from experiments.backtest_allocation import format_backtest_summary, write_backtest_outputs


def _toy_results():
    dates = list(pd.date_range("2020-01-01", periods=3, freq="W-SUN"))
    w = [pd.Series({"A": 0.5, "B": 0.5}) for _ in range(3)]

    def arm(brets, turns, nheld):
        return {"block_returns": brets, "turnover": turns, "n_held": nheld,
                "weights": w, "dates": dates}

    return {
        "current": arm([0.02, -0.01, 0.03], [None, 0.2, 0.1], [2, 2, 2]),
        "equal_weight": arm([0.01, 0.0, 0.01], [None, 0.0, 0.0], [2, 2, 2]),
        "rebalance_index": [10, 14, 18],
    }


class TestSummaryAndOutputs:
    def test_summary_contains_arms_and_metrics(self):
        text = format_backtest_summary(_toy_results(), CFG, rebalance_every=4)
        assert "current" in text and "equal_weight" in text
        assert "sharpe" in text and "mean_turnover" in text

    def test_writes_all_files(self, tmp_path):
        paths = write_backtest_outputs(_toy_results(), CFG, 4, str(tmp_path))
        for key in ["returns", "turnover", "summary",
                    "weights_current", "weights_equal_weight"]:
            assert os.path.exists(paths[key])

    def test_returns_roundtrip(self, tmp_path):
        paths = write_backtest_outputs(_toy_results(), CFG, 4, str(tmp_path))
        df = pd.read_csv(paths["returns"], index_col=0)
        assert list(df.columns) == ["current", "equal_weight"]
        assert len(df) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestSummaryAndOutputs -v`
Expected: FAIL with `ImportError: cannot import name 'format_backtest_summary'`.

- [ ] **Step 3: Implement the summary and writers**

Add to `experiments/backtest_allocation.py` (after `run_backtest`):

```python
def format_backtest_summary(results, cfg, rebalance_every):
    """Headline metric table (one row per arm) from a run_backtest result."""
    blocks_per_year = cfg["periods_per_year"] / rebalance_every
    rf_block = (1.0 + cfg["rf_period"]) ** rebalance_every - 1.0
    labels = [k for k in results if k != "rebalance_index"]
    rows = []
    for lab in labels:
        d = results[lab]
        m = summarize_arm(d["block_returns"], d["turnover"], d["n_held"],
                          blocks_per_year, rf_block)
        m["arm"] = lab
        rows.append(m)
    cols = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_dd",
            "mean_turnover", "avg_names", "hit_rate"]
    df = pd.DataFrame(rows).set_index("arm")[cols]
    n_blocks = len(results[labels[0]]["block_returns"])
    header = (f"Walk-forward backtest | blocks: {n_blocks} | "
              f"rebalance_every: {rebalance_every} | blocks/yr: {blocks_per_year:g}\n"
              f"(frictionless / gross realized returns)\n")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    return header + "\n" + df.to_string()


def write_backtest_outputs(results, cfg, rebalance_every, outdir):
    """Write returns/turnover/per-arm-weights CSVs and the summary text; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    labels = [k for k in results if k != "rebalance_index"]
    dates = results[labels[0]]["dates"]
    paths = {}

    ret_df = pd.DataFrame({lab: results[lab]["block_returns"] for lab in labels}, index=dates)
    paths["returns"] = os.path.join(outdir, "backtest_returns.csv")
    ret_df.to_csv(paths["returns"], index_label="date")

    turn_df = pd.DataFrame({lab: results[lab]["turnover"] for lab in labels}, index=dates)
    paths["turnover"] = os.path.join(outdir, "backtest_turnover.csv")
    turn_df.to_csv(paths["turnover"], index_label="date")

    for lab in labels:
        w_df = pd.DataFrame(results[lab]["weights"], index=dates).fillna(0.0)
        p = os.path.join(outdir, f"backtest_weights_{lab}.csv")
        w_df.to_csv(p, index_label="date")
        paths[f"weights_{lab}"] = p

    paths["summary"] = os.path.join(outdir, "backtest_summary.txt")
    with open(paths["summary"], "w") as f:
        f.write(format_backtest_summary(results, cfg, rebalance_every))
    return paths
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestSummaryAndOutputs -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/backtest_allocation.py tests/test_backtest_allocation.py
git commit -m "Add backtest summary table and output writers"
```

---

### Task 6: CLI + `main`

`parse_spreads` parses the `--spreads` string; `build_arg_parser` defines the flags with the settled defaults; `main` wires the run end to end.

**Files:**
- Modify: `experiments/backtest_allocation.py`
- Test: `tests/test_backtest_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_allocation.py`:

```python
from experiments.backtest_allocation import build_arg_parser, parse_spreads


class TestCliWiring:
    def test_defaults(self):
        args = build_arg_parser().parse_args([])
        assert args.oos_periods == 162
        assert args.rebalance_every == 4
        assert args.n_runs == 50
        assert args.mc_draws == 1000
        assert args.spreads == "1,2,4"
        assert args.seed == 0

    def test_parse_spreads(self):
        assert parse_spreads("1,2,4") == [1.0, 2.0, 4.0]
        assert parse_spreads("2") == [2.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestCliWiring -v`
Expected: FAIL with `ImportError: cannot import name 'build_arg_parser'`.

- [ ] **Step 3: Implement the CLI**

Add to `experiments/backtest_allocation.py` (after `write_backtest_outputs`):

```python
def parse_spreads(text):
    """Parse a comma-separated spreads string into a list of floats."""
    return [float(x) for x in text.split(",") if x.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--oos-periods", type=int, default=162,
                        help="Out-of-sample window length in periods (default 162 ~ 3yr weekly)")
    parser.add_argument("--rebalance-every", type=int, default=4,
                        help="Rebalance/holding period in periods (default 4 = forecast horizon)")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Transformer runs per rebalance (default 50)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="Parametric Monte-Carlo mu draws K (default 1000)")
    parser.add_argument("--spreads", type=str, default="1,2,4",
                        help="Comma-separated parametric spread values (default '1,2,4')")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; rebalance k uses seed+k (default 0)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "backtest"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)
    spreads = parse_spreads(args.spreads)

    print(f"Universe: {rets.shape[1]} stocks | oos_periods={args.oos_periods} | "
          f"rebalance_every={args.rebalance_every} | n_runs={args.n_runs} | "
          f"mc_draws={args.mc_draws} | spreads={spreads} | seed={args.seed}")

    results = run_backtest(
        prices, rets, cfg, oos_periods=args.oos_periods,
        rebalance_every=args.rebalance_every, n_runs=args.n_runs,
        mc_draws=args.mc_draws, spreads=spreads, seed=args.seed,
    )
    paths = write_backtest_outputs(results, cfg, args.rebalance_every, args.outdir)
    print()
    print(format_backtest_summary(results, cfg, args.rebalance_every))
    print("\nSaved:")
    for p in paths.values():
        print(f"       {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the CLI tests, then the whole file**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py::TestCliWiring -v`
Expected: PASS (2 passed).

Run the full new suite: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_backtest_allocation.py -v`
Expected: PASS (28 passed: 9 + 4 + 6 + 4 + 3 + 2).

Run the whole repo test suite to confirm no regressions: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/ -q`
Expected: PASS (the existing 68 + 28 new = 96 green).

- [ ] **Step 5: Commit**

```bash
git add experiments/backtest_allocation.py tests/test_backtest_allocation.py
git commit -m "Add backtest CLI and main entry point"
```

---

## Post-implementation validation (manual, not a code task)

After Task 6, run the real backtest (slow — ~40 rebalances × 50 trainings ≈ 2000 trainings on the full universe; use `run_in_background`):

```
python experiments/backtest_allocation.py --oos-periods 162 --rebalance-every 4 \
  --n-runs 50 --mc-draws 1000 --spreads 1,2,4 --seed 0 \
  --outdir experiments/results/backtest
```

Then read `experiments/results/backtest/backtest_summary.txt`:
1. Does any Michaud arm match/beat **current** on *gross* realized Sharpe? (If yes, lower turnover only strengthens its net case.)
2. How does each arm compare to **equal_weight**? (The "is Michaud just disguised 1/N?" check.)
3. Turnover ranking across arms (input to the deferred net-of-cost analysis — free from `backtest_turnover.csv` + `backtest_weights_*.csv`).

Record the read-off in `project_allocation_stability.md`. Net-of-cost analysis (applying bps to the saved turnover/weights) is a free follow-up; productionising any winner into `pipeline/04_allocate.py` is a separate, pushable phase.

---

## Self-Review notes

- **Spec coverage:** walk-forward loop with expanding window + retrain-per-rebalance + forecast reuse across arms (Task 4); monthly cadence / OOS window / n_runs / mc_draws as CLI defaults (Task 6); 6 arms incl. parametric frontier, empirical, equal-weight, all deferred elimination (Task 3); no-lookahead enforced by `iloc[:t]` slicing and verified by `test_no_lookahead` (Task 4); buy-and-hold realized returns + target-to-target turnover (Tasks 1, 4); frictionless headline with turnover + per-arm weights saved for free cost analysis (Task 5); metrics realized Sharpe / cum / ann / vol / max DD / turnover / avg names / hit rate (Tasks 2, 5); outputs to `experiments/results/backtest/` (Task 5). Out-of-scope items (cost sweep, shrinkage arm, per-draw elimination, intra-block drift) are intentionally absent.
- **Placeholder scan:** none — every code/test step is complete with exact commands and expected counts.
- **Type consistency:** `compute_arm_weights(arm, mu_bar, per_run_mu, covmat, cfg, n_periods, mc_draws, rng)` (Task 3) is called with matching positional args in `run_backtest` (Task 4). `label_of` (Task 3) is used for result keys in Task 4 and consumed in Tasks 5-6. `run_backtest` returns `{label: {block_returns, weights, turnover, n_held, dates}, "rebalance_index": [...]}`, consumed consistently by `summarize_arm` / `format_backtest_summary` / `write_backtest_outputs`. `annualized_stats`/`summarize_arm` keys match the `cols` list in `format_backtest_summary`. Reused signatures match what exists on `main`.
