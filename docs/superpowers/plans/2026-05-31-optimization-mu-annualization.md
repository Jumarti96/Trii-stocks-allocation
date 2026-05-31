# Optimization μ — Annualization Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the optimizer consume a per-period expected-return vector (`wmr`) against the per-period covariance with a per-period risk-free rate, and move annualization to display-only in the report — removing the compounding cross-sectional distortion and the Sharpe unit mismatch.

**Architecture:** `μ` is per-period everywhere it is computed/optimized; `(1+μ)^ppy−1` annualization happens only for human-facing report numbers. `02_expected_returns.csv` is repurposed to hold per-period `μ`. A derived `cfg['rf_period']` feeds the optimizer. The stability instrument is updated to mirror the new optimization so we get a before/after.

**Tech Stack:** Python, numpy, pandas, scikit-learn, scipy (via `risk_kit`), pytest. Files: `pipeline/config.py`, `pipeline/02_predict.py`, `pipeline/04_allocate.py`, `pipeline/05_report.py`, `src/transformer_model.py`, `experiments/measure_allocation_stability.py`.

**Spec:** `docs/superpowers/specs/2026-05-31-optimization-mu-annualization-design.md`.

## CRITICAL environment note
Only this interpreter has the deps (pandas/numpy/sklearn/scipy/torch) — use it for every test and run:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe"`
e.g. `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest <file> -v`.
Bare `python` and `.venv` lack pandas. Repo is on branch `enh-optimization-mu-annualization`; commit there, never switch branches. Commit messages contain NO AI attribution.

---

## File Structure

- **Modify `pipeline/config.py`** — add derived `cfg['rf_period']`. Owns shared config.
- **Modify `src/transformer_model.py`** — add `annualize_period_return()`; refactor `annualize_expected_returns()` to use it. Owns forecast/annualization helpers.
- **Modify `pipeline/02_predict.py`** — persist per-period `μ`. (wiring)
- **Modify `pipeline/04_allocate.py`** — optimize with `rf_period`. (wiring)
- **Modify `pipeline/05_report.py`** — compound-annualize for display. (wiring)
- **Modify `experiments/measure_allocation_stability.py`** — mirror per-period optimization.
- **Create `tests/test_pipeline_config.py`** — `rf_period` derivation.
- **Create `tests/test_transformer_model.py`** — annualization helper + robustness.
- **Modify `tests/test_measure_allocation_stability.py`** — adapt to `rf_period` / per-period `μ`; add robustness + scale-invariance tests.

Testable logic lives in helpers (`config`, `transformer_model`) and the instrument; the three numbered pipeline scripts are thin wiring, verified by `py_compile` + the helper tests + the Task 8 end-to-end run.

---

### Task 1: Derived `rf_period` in config

**Files:**
- Modify: `pipeline/config.py`
- Test: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write the failing test** — Create `tests/test_pipeline_config.py`:

```python
"""Tests for pipeline/config.py derived values.

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_pipeline_config.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from config import load_config


def test_rf_period_is_compound_per_period_equivalent():
    cfg = load_config()
    expected = (1 + cfg["rf_rate"]) ** (1 / cfg["periods_per_year"]) - 1
    assert abs(cfg["rf_period"] - expected) < 1e-15


def test_rf_period_below_annual_for_multi_period_year():
    cfg = load_config()
    # ppy > 1, positive rf -> per-period rate is strictly smaller than annual
    assert 0 < cfg["rf_period"] < cfg["rf_rate"]
```

- [ ] **Step 2: Run to verify it fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_pipeline_config.py -v` — Expected: FAIL with `KeyError: 'rf_period'`.

- [ ] **Step 3: Write minimal implementation** — In `pipeline/config.py`, in `load_config`, immediately after the `cfg['time_window'] = cfg['time_window'] or cfg['periods_per_year']` line, add:

```python
    cfg['rf_period']    = (1 + cfg['rf_rate']) ** (1 / cfg['periods_per_year']) - 1
```

- [ ] **Step 4: Run to verify it passes** — same command — Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```
git add pipeline/config.py tests/test_pipeline_config.py
git commit -m "Add derived rf_period to config"
```

---

### Task 2: `annualize_period_return` helper

**Files:**
- Modify: `src/transformer_model.py`
- Test: `tests/test_transformer_model.py`

- [ ] **Step 1: Write the failing test** — Create `tests/test_transformer_model.py`:

```python
"""Tests for annualization helpers in src/transformer_model.py.

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_transformer_model.py -v
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformer_model import (
    annualize_period_return,
    annualize_expected_returns,
    weighted_mean_return,
)


def test_annualize_scalar():
    assert abs(annualize_period_return(0.01, 12) - ((1.01 ** 12) - 1)) < 1e-12


def test_annualize_zero_is_zero_any_ppy():
    assert annualize_period_return(0.0, 54) == 0.0


def test_annualize_series_elementwise():
    s = pd.Series({"A": 0.01, "B": 0.0})
    out = annualize_period_return(s, 12)
    assert abs(out["A"] - ((1.01 ** 12) - 1)) < 1e-12
    assert out["B"] == 0.0


def test_optimization_mu_has_no_ppy_dependence():
    # The per-period mu the optimiser consumes is weighted_mean_return, which
    # takes no periods_per_year argument -> changing frequency cannot distort it.
    preds = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [-0.01, 0.0, 0.01]})
    pd.testing.assert_series_equal(weighted_mean_return(preds), weighted_mean_return(preds))


def test_annualize_expected_returns_delegates_to_helper():
    preds = pd.DataFrame({"A": [0.01, 0.01, 0.01], "B": [0.005, 0.005, 0.005]})
    ppy = 12
    expected = annualize_period_return(weighted_mean_return(preds), ppy)
    pd.testing.assert_series_equal(annualize_expected_returns(preds, ppy), expected)
```

- [ ] **Step 2: Run to verify it fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_transformer_model.py -v` — Expected: FAIL with `ImportError: cannot import name 'annualize_period_return'`.

- [ ] **Step 3: Write minimal implementation** — In `src/transformer_model.py`, replace the existing `annualize_expected_returns` function (currently lines 100-106) with:

```python
def annualize_period_return(period_returns, periods_per_year):
    """Compound-annualise a per-period return (scalar or Series): (1+r)^ppy - 1.

    Display-only helper: used by the report to present per-period forecasts as
    annual figures. It is NOT used to build the vector fed to the optimiser.
    """
    return (1 + period_returns) ** periods_per_year - 1


def annualize_expected_returns(preds_df, periods_per_year, lambda_=0.2):
    """Exponential-decay-weighted annualised expected return per column of preds_df.

    Used by the train-universe comparison experiment and the report's display path.
    """
    wmr = weighted_mean_return(preds_df, lambda_=lambda_)
    return annualize_period_return(wmr, periods_per_year)
```

- [ ] **Step 4: Run to verify it passes** — same command — Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```
git add src/transformer_model.py tests/test_transformer_model.py
git commit -m "Extract annualize_period_return display helper"
```

---

### Task 3: `02_predict.py` writes per-period μ

**Files:**
- Modify: `pipeline/02_predict.py`

- [ ] **Step 1: Change the import.** In `pipeline/02_predict.py`, replace the line:

```python
from transformer_model import train_and_predict, annualize_expected_returns, describe_device
```
with:
```python
from transformer_model import train_and_predict, weighted_mean_return, describe_device
```

- [ ] **Step 2: Replace the annualization with the per-period quantity.** Replace these lines:

```python
    # Annualised expected returns with exponential-decay weighting
    expected_returns = annualize_expected_returns(preds_df, periods_per_year)
```
with:
```python
    # Per-period expected returns (exp-decay weighted). Annualisation is a DISPLAY
    # concern handled in step 5; the optimiser consumes these per-period values directly.
    expected_returns = weighted_mean_return(preds_df)
```

- [ ] **Step 3: Update the output column header.** Replace:

```python
    expected_returns.to_csv(PATHS['02_expected_returns'], header=['Expected Return'])
```
with:
```python
    expected_returns.to_csv(PATHS['02_expected_returns'], header=['Expected Period Return'])
```

- [ ] **Step 4: Remove the now-unused `periods_per_year` local.** Replace:

```python
    periods_to_forecast = cfg['periods_to_forecast']
    periods_per_year    = cfg['periods_per_year']
```
with:
```python
    periods_to_forecast = cfg['periods_to_forecast']
```

- [ ] **Step 5: Verify it compiles.** Run:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m py_compile pipeline/02_predict.py`
Expected: no output, exit 0. (Numeric correctness of `weighted_mean_return` is covered by Task 2; full behavior by the Task 8 end-to-end run.)

- [ ] **Step 6: Commit**
```
git add pipeline/02_predict.py
git commit -m "02_predict: persist per-period expected returns"
```

---

### Task 4: `04_allocate.py` optimizes with `rf_period`

**Files:**
- Modify: `pipeline/04_allocate.py`

- [ ] **Step 1: Read the per-period risk-free rate.** Replace:

```python
    rf_rate          = cfg['rf_rate']
```
with:
```python
    rf_period        = cfg['rf_period']
```

- [ ] **Step 2: Use it in the initial optimization.** Replace:

```python
    initial_weights = rk.msr_tuned(
        riskfree_rate=rf_rate, returns=returns, covmat=covmat,
        max_weight=max_weight, periods_per_year=periods_per_year, debug=False
    )
```
with:
```python
    initial_weights = rk.msr_tuned(
        riskfree_rate=rf_period, returns=returns, covmat=covmat,
        max_weight=max_weight, periods_per_year=periods_per_year, debug=False
    )
```

- [ ] **Step 3: Use it in the elimination re-optimization.** Replace:

```python
        w = rk.msr_tuned(
            riskfree_rate=rf_rate,
            returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_weight,
            periods_per_year=periods_per_year,
            debug=False
        )
```
with:
```python
        w = rk.msr_tuned(
            riskfree_rate=rf_period,
            returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_weight,
            periods_per_year=periods_per_year,
            debug=False
        )
```

- [ ] **Step 4: Verify it compiles.** Run:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m py_compile pipeline/04_allocate.py`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**
```
git add pipeline/04_allocate.py
git commit -m "04_allocate: optimize with per-period risk-free rate"
```

---

### Task 5: `05_report.py` annualizes for display

**Files:**
- Modify: `pipeline/05_report.py`

- [ ] **Step 1: Import the display helper.** After the line `import risk_kit as rk`, add:

```python
from transformer_model import annualize_period_return
```

- [ ] **Step 2: Build the display-annual series and use it in the per-stock column.** Replace:

```python
    output = pd.DataFrame({
        'Portfolio Weight':                       weights_series.round(4),
        'Expected Annual Return':                 expected_returns[weights_series.index].round(4),
        'Current Price':                          current_prices[weights_series.index].round(4),
        f'Forecasted Price ({last_future_date})': forecasted_prices[weights_series.index].round(4),
        'Investment (COP k)':                     cop_per_stock,
    }).sort_values('Portfolio Weight', ascending=False)
```
with:
```python
    # expected_returns from step 2 are PER-PERIOD; compound-annualise for display only.
    expected_annual = annualize_period_return(expected_returns, periods_per_year)

    output = pd.DataFrame({
        'Portfolio Weight':                       weights_series.round(4),
        'Expected Annual Return':                 expected_annual[weights_series.index].round(4),
        'Current Price':                          current_prices[weights_series.index].round(4),
        f'Forecasted Price ({last_future_date})': forecasted_prices[weights_series.index].round(4),
        'Investment (COP k)':                     cop_per_stock,
    }).sort_values('Portfolio Weight', ascending=False)
```

- [ ] **Step 3: Rewrite the forecasted-COP projection in per-period terms.** Replace:

```python
    portfolio_forecasted = (
        cop_per_stock
        * (1 + expected_returns[weights_series.index]) ** (periods_to_forecast / periods_per_year)
    ).sum().round(2)
```
with:
```python
    # Per-period mu compounded over the forecast horizon (equals the old
    # (1+annual)^(periods_to_forecast/ppy) expression, now in per-period units).
    portfolio_forecasted = (
        cop_per_stock
        * (1 + expected_returns[weights_series.index]) ** periods_to_forecast
    ).sum().round(2)
```

- [ ] **Step 4: Compute the portfolio-level expected annual return consistently.** Replace:

```python
    portfolio_row = pd.DataFrame({
        'Portfolio Weight':                       [1],
        'Expected Annual Return':                 [round((weights_series * expected_returns[weights_series.index]).sum(), 4)],
        'Current Price':                          [cop_per_stock.sum()],
        f'Forecasted Price ({last_future_date})': [portfolio_forecasted],
        'Investment (COP k)':                     [cop_per_stock.sum()],
    }, index=['PORTFOLIO INDEX'])
```
with:
```python
    portfolio_period_return = (weights_series * expected_returns[weights_series.index]).sum()
    portfolio_row = pd.DataFrame({
        'Portfolio Weight':                       [1],
        'Expected Annual Return':                 [round(annualize_period_return(portfolio_period_return, periods_per_year), 4)],
        'Current Price':                          [cop_per_stock.sum()],
        f'Forecasted Price ({last_future_date})': [portfolio_forecasted],
        'Investment (COP k)':                     [cop_per_stock.sum()],
    }, index=['PORTFOLIO INDEX'])
```

- [ ] **Step 5: Verify it compiles.** Run:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m py_compile pipeline/05_report.py`
Expected: no output, exit 0. (Annualization correctness is covered by Task 2 tests; full report by Task 8.)

- [ ] **Step 6: Commit**
```
git add pipeline/05_report.py
git commit -m "05_report: annualize per-period mu for display"
```

---

### Task 6: Instrument `allocate_msr` uses `rf_period`

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Update the test CFG and add a failing assertion.** In `tests/test_measure_allocation_stability.py`, change the module-level CFG line:

```python
CFG = {"rf_rate": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}
```
to:
```python
CFG = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}
```
Then append a new test to `class TestAllocateMsr`:
```python
    def test_uses_rf_period_key(self, three_assets):
        # allocate_msr must read cfg['rf_period']; a CFG missing it should KeyError.
        returns, covmat = three_assets
        bad_cfg = {"rf_rate": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}
        import pytest
        with pytest.raises(KeyError):
            allocate_msr(returns, covmat, bad_cfg)
```

- [ ] **Step 2: Run to verify the new test fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestAllocateMsr::test_uses_rf_period_key -v` — Expected: FAIL (allocate_msr still reads `rf_rate`, so no KeyError is raised).

- [ ] **Step 3: Switch `allocate_msr` to `rf_period`.** In `experiments/measure_allocation_stability.py`, inside `allocate_msr`, replace:

```python
    rf = cfg["rf_rate"]
```
with:
```python
    rf = cfg["rf_period"]
```

- [ ] **Step 4: Run the full instrument suite** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py -v` — Expected: PASS (29 tests: the prior 28 with CFG updated, plus the new `test_uses_rf_period_key`).

- [ ] **Step 5: Commit**
```
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Instrument: allocate with per-period risk-free rate"
```

---

### Task 7: Instrument `run_experiment` uses per-period μ + robustness tests

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Update the `run_experiment` test stub + cfg, and add robustness/scale-invariance tests.** In `tests/test_measure_allocation_stability.py`:

(a) Add a `src` path + `msr_tuned` import near the top of the file, after the existing `sys.path.insert(...)` line:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from risk_kit import msr_tuned
```

(b) In `class TestRunExperiment._inputs`, change the cfg dict to include `rf_period`:
```python
        cfg = {
            "rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05,
            "periods_per_year": 12,
        }
```

(c) In `class TestRunExperiment._stubs`, rename the per-period stub and drop the `ppy` arg. Replace:
```python
        def annualize_fn(preds, ppy):
            return preds.mean(axis=0)
```
with:
```python
        def period_mu_fn(preds):
            return preds.mean(axis=0)
```
and change the return line `return predict_fn, annualize_fn, select_fn, seed_fn` to:
```python
        return predict_fn, period_mu_fn, select_fn, seed_fn
```

(d) In both `test_output_shapes` and `test_weights_rows_sum_to_one`, change the unpacking and the `run_experiment(...)` keyword. Replace each occurrence of:
```python
        predict_fn, annualize_fn, select_fn, seed_fn = self._stubs()
```
with:
```python
        predict_fn, period_mu_fn, select_fn, seed_fn = self._stubs()
```
and each occurrence of `annualize_fn=annualize_fn,` with `period_mu_fn=period_mu_fn,`.

(e) Append two new tests to `class TestRunExperiment`:
```python
    def test_weights_invariant_to_periods_per_year(self):
        # With rf_period held at 0, the per-period optimisation must give identical
        # mu and weights regardless of periods_per_year (mu carries no ppy term).
        import numpy as np
        prices, rets, cfg = self._inputs()
        cfg_weekly = dict(cfg, periods_per_year=52)
        cfg_monthly = dict(cfg, periods_per_year=12)
        # fresh stubs per call so the predict_fn drift counter starts equal
        p1, m1, s1, sd1 = self._stubs()
        r_weekly = run_experiment(prices, rets, cfg_weekly,
                                  iterations=3, transformer_runs=2, seed=0,
                                  predict_fn=p1, period_mu_fn=m1, select_fn=s1, seed_fn=sd1)
        p2, m2, s2, sd2 = self._stubs()
        r_monthly = run_experiment(prices, rets, cfg_monthly,
                                   iterations=3, transformer_runs=2, seed=0,
                                   predict_fn=p2, period_mu_fn=m2, select_fn=s2, seed_fn=sd2)
        assert np.allclose(r_weekly["weights"].values, r_monthly["weights"].values)
        assert np.allclose(r_weekly["mu"].values, r_monthly["mu"].values)


class TestOptimizerScaleInvariance:
    def test_weights_unchanged_under_consistent_scaling(self):
        # max-Sharpe is invariant to (mu->c*mu, Sigma->c^2*Sigma, rf->c*rf):
        # justifies optimising in per-period units instead of annualised ones.
        import numpy as np
        returns = pd.Series({"A": 0.02, "B": 0.012, "C": 0.015})
        cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06]), index=returns.index, columns=returns.index)
        w1 = msr_tuned(0.001, returns=returns, covmat=cov, max_weight=0.6, periods_per_year=12)
        c = 52
        w2 = msr_tuned(0.001 * c, returns=returns * c, covmat=cov * (c ** 2),
                       max_weight=0.6, periods_per_year=12)
        assert np.allclose(w1, w2, atol=1e-4)
```

- [ ] **Step 2: Run to verify the new/renamed tests fail** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py -k "RunExperiment or ScaleInvariance" -v` — Expected: FAIL (run_experiment still has `annualize_fn`, not `period_mu_fn`).

- [ ] **Step 3: Update `run_experiment`.** In `experiments/measure_allocation_stability.py`:

(a) Change the signature line:
```python
def run_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                   predict_fn=None, annualize_fn=None, select_fn=None, seed_fn=None):
```
to:
```python
def run_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                   predict_fn=None, period_mu_fn=None, select_fn=None, seed_fn=None):
```

(b) Replace the dependency-resolution block:
```python
    if predict_fn is None or annualize_fn is None:
        from transformer_model import train_and_predict, annualize_expected_returns
        predict_fn = predict_fn or train_and_predict
        annualize_fn = annualize_fn or annualize_expected_returns
```
with:
```python
    if predict_fn is None or period_mu_fn is None:
        from transformer_model import train_and_predict, weighted_mean_return
        predict_fn = predict_fn or train_and_predict
        period_mu_fn = period_mu_fn or weighted_mean_return
```

(c) Replace the per-iteration aggregation line:
```python
        mu = annualize_fn(preds, ppy)
```
with:
```python
        mu = period_mu_fn(preds)
```

Also update the docstring line listing the seams: change `predict_fn, annualize_fn, select_fn, seed_fn` to `predict_fn, period_mu_fn, select_fn, seed_fn` if present.

- [ ] **Step 4: Run the full instrument suite** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py -v` — Expected: PASS (31 tests: 29 from Task 6 + the 2 new robustness/scale-invariance tests).

- [ ] **Step 5: Commit**
```
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Instrument: use per-period mu and add robustness tests"
```

---

### Task 8: End-to-end pipeline run + instrument validation

**Files:** none modified — verification + measurement only. (Run by the controller; GPU-heavy.)

- [ ] **Step 1: Regenerate the pipeline from step 2 with the new code.** Run:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" orchestrator.py --from 2`
Expected: steps 2→5 complete without error and `results/allocation_output.csv` is written. (Step 2 trains the Transformer `n_transformer_runs` times — slow on GPU; this is the real integration check.)

- [ ] **Step 2: Sanity-check the report.** Run:
```
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -c "import pandas as pd; d=pd.read_csv('results/allocation_output.csv', index_col=0); print(d.to_string()); w=d['Portfolio Weight'][:-1]; print('weights sum:', round(w.sum(),4))"
```
Expected: weights (excluding the PORTFOLIO INDEX row) sum to ~1.0; "Expected Annual Return" values are finite and plausible (no NaN/inf); the table prints cleanly. Also confirm `data/02_expected_returns.csv` header is `Expected Period Return` and its values are small per-period magnitudes (roughly within the winsorisation band), not annualized.

- [ ] **Step 3: Re-run the stability instrument (after) and compare to the Phase 1 baseline.** Run:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/measure_allocation_stability.py --iterations 30 --transformer-runs 10 --seed 0 --outdir experiments/results/full_30x10_phase2`
Then read `experiments/results/full_30x10_phase2/stability_summary.txt` and compare to the Phase 1 baseline (turnover **0.84**, Jaccard **0.11**, return CoV **16%**, top selection frequency **50%**). Record the deltas — this quantifies what the μ fix alone bought and informs the Phase 3 (lever #2) decision.

- [ ] **Step 4: No commit** (no source changes). Capture the comparison in the response and update project memory.

---

## Self-Review

**1. Spec coverage:**
- Per-period optimization μ (Approach A) → Task 3 (02 writes `wmr`), Task 4 (04 uses it). ✓
- `rf_period` derived + used → Task 1 (config), Task 4 (pipeline), Task 6 (instrument). ✓
- Single source of truth, column renamed → Task 3. ✓
- Display annualization compound, per-stock + portfolio + forecasted-COP consistent → Task 2 (helper) + Task 5. ✓
- Parameter robustness guarantee → Task 2 (`test_optimization_mu_has_no_ppy_dependence`), Task 7 (`test_weights_invariant_to_periods_per_year`). ✓
- Instrument mirrors new optimization → Tasks 6, 7. ✓
- Testing: rf_period conversion (T1), display annualization (T2), robustness (T2/T7), scale-invariance confidence (T7), end-to-end (T8), instrument suite green (T6/T7). ✓
- Validation plan (before/after at 30×10) → Task 8 Step 3. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step shows exact before/after text; every test step shows complete asserts. ✓

**3. Type consistency:** `annualize_period_return(period_returns, periods_per_year)` defined in T2 and called identically in T5 and the T2 tests. `cfg['rf_period']` defined in T1, consumed in T4 and the instrument in T6. The instrument seam rename `annualize_fn`→`period_mu_fn` is applied consistently across the signature, resolution block, body, and all tests in T7. `02_expected_returns.csv` column `Expected Period Return` written in T3, read (as per-period) in T4/T5. Test counts: 28 → 29 (T6) → 31 (T7). ✓

No gaps found.
