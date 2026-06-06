# PR-2: Filter Removal + Top-N Compute Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the technical-indicator filter (step 03), rewire step 4 to read its universe from step 2 outputs, and add a configurable top-N pre-selection to bound optimizer compute for large universes.

**Architecture:** A new pure function `select_top_n` in `src/allocation.py` ranks stocks by per-stock Sharpe proxy (mu/sigma) or raw return and slices the top-N before calling the existing `allocate`. Step 03 is deleted entirely; step 04 reads the full universe from `02_expected_returns.csv` and applies `select_top_n` before optimising. Two new params.yaml keys (`allocation_top_n`, `allocation_ranking`) control the cap; `n=null` is a no-op for users running the small universe.

**Tech Stack:** Python, pandas, numpy, scipy (SLSQP via risk_kit), pytest

**Python executable:** `"C:/Python projects/Finance/Scripts/python.exe"` (project venv)

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/allocation.py` | Modify | Add `select_top_n` function |
| `tests/test_allocation.py` | Modify | Add 6 tests for `select_top_n` |
| `pipeline/04_allocate.py` | Modify | Import `select_top_n`, remove step-3 read, add top-N call |
| `params.yaml` | Modify | Remove `ma_terms`/`signal_min_count`, add `allocation_top_n`/`allocation_ranking` |
| `pipeline/config.py` | Modify | Remove 3 PATHS entries for step-3 outputs |
| `pipeline/03_filter.py` | Delete | Gone entirely |

---

### Task 1: Write failing tests for `select_top_n`

**Files:**
- Modify: `tests/test_allocation.py`

- [ ] **Step 1: Add the import and test class to `tests/test_allocation.py`**

Append to the bottom of `tests/test_allocation.py` (after the existing `TestAllocateDispatcher` class):

```python
from allocation import select_top_n


def _universe5():
    """5-stock universe with known Sharpe and return rankings.

    mu:    A=0.10  B=0.05  C=0.20  D=0.08  E=0.15
    vol:   A=0.30  B=0.10  C=0.50  D=0.10  E=0.20
    Sharpe:  0.333   0.500   0.400   0.800   0.750
    Sharpe rank: D > E > B > C > A  -> top-3: {D, E, B}
    Return rank: C > E > A > D > B  -> top-3: {C, E, A}
    """
    tickers = ["A", "B", "C", "D", "E"]
    mu = pd.Series({"A": 0.10, "B": 0.05, "C": 0.20, "D": 0.08, "E": 0.15})
    vols = {"A": 0.30, "B": 0.10, "C": 0.50, "D": 0.10, "E": 0.20}
    cov_arr = np.diag([vols[t] ** 2 for t in tickers])
    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
    return mu, cov


class TestSelectTopN:
    def test_sharpe_ranking_selects_correct_names(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=3, metric="sharpe")
        assert set(mu_out.index) == {"D", "E", "B"}

    def test_return_ranking_selects_correct_names(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=3, metric="return")
        assert set(mu_out.index) == {"C", "E", "A"}

    def test_null_n_returns_full_universe(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=None, metric="sharpe")
        assert list(mu_out.index) == list(mu.index)
        assert cov_out.shape == cov.shape

    def test_n_exceeds_universe_returns_full(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=1000, metric="sharpe")
        assert list(mu_out.index) == list(mu.index)
        assert cov_out.shape == cov.shape

    def test_covmat_index_matches_mu_index(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=3, metric="sharpe")
        assert list(cov_out.index) == list(mu_out.index)
        assert list(cov_out.columns) == list(mu_out.index)

    def test_unknown_metric_raises(self):
        mu, cov = _universe5()
        with pytest.raises(ValueError):
            select_top_n(mu, cov, n=3, metric="bogus")
```

- [ ] **Step 2: Run tests to confirm they fail**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_allocation.py::TestSelectTopN -v
```

Expected: 6 failures with `ImportError: cannot import name 'select_top_n'`

---

### Task 2: Implement `select_top_n` in `src/allocation.py`

**Files:**
- Modify: `src/allocation.py`

- [ ] **Step 1: Add `select_top_n` above the `allocate` dispatcher**

Insert the following block into `src/allocation.py` immediately before the `def allocate(` line (currently line 131):

```python
def select_top_n(mu, covmat, n, metric="sharpe"):
    """Pre-select the top-n candidates by metric before optimization.

    metric='sharpe': rank by mu / sqrt(diag(covmat))  (default)
    metric='return': rank by mu only
    n=None or n >= len(mu): no-op, returns full universe unchanged.
    """
    if n is None or n >= len(mu):
        return mu, covmat
    if metric == "sharpe":
        vol = pd.Series(
            np.sqrt(np.diag(covmat.values)), index=mu.index
        ).clip(lower=1e-8)
        score = mu / vol
    elif metric == "return":
        score = mu
    else:
        raise ValueError(f"unknown allocation_ranking: {metric!r}")
    top = score.nlargest(n).index
    return mu[top], covmat.loc[top, top]

```

- [ ] **Step 2: Run tests to confirm they pass**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_allocation.py::TestSelectTopN -v
```

Expected: 6 PASSED

- [ ] **Step 3: Run the full test suite**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -v
```

Expected: all existing tests still pass (no regressions).

- [ ] **Step 4: Commit**

```
git add src/allocation.py tests/test_allocation.py
git commit -m "Add select_top_n: pre-select top-N candidates by Sharpe/return before optimizer"
```

---

### Task 3: Update `params.yaml`

**Files:**
- Modify: `params.yaml`

- [ ] **Step 1: Remove step-3 keys and add step-4 top-N keys**

In `params.yaml`, remove these two lines (under the `# Technical signal filtering` comment block):
```yaml
ma_terms: 10
signal_min_count: 3
```

Also remove the comment header for that block:
```yaml
# Technical signal filtering (Step 2)
```

Then add two lines under the `# Portfolio optimisation (Step 4)` section, right before `allocation_method`:
```yaml
allocation_top_n: 150        # candidates fed to optimizer; null = no cap (full universe)
allocation_ranking: sharpe   # "sharpe" (mu/sigma) | "return" (mu only)
```

The Portfolio optimisation block should look like this after the edit:
```yaml
# Portfolio optimisation (Step 4)
allocation_top_n: 150        # candidates fed to optimizer; null = no cap (full universe)
allocation_ranking: sharpe   # "sharpe" (mu/sigma) | "return" (mu only)
allocation_method: parametric_michaud   # "parametric_michaud" | "msr"
michaud_spread: 4.0
michaud_mc_draws: 1000
michaud_seed: 0
rf_rate: 0.11
max_weight: 0.15
min_weight: 0.05
```

- [ ] **Step 2: Run the full test suite**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -v
```

Expected: all tests pass (params.yaml changes don't break tests since tests use their own CFG dict).

- [ ] **Step 3: Commit**

```
git add params.yaml
git commit -m "Remove step-3 params, add allocation_top_n and allocation_ranking to params.yaml"
```

---

### Task 4: Rewire `pipeline/04_allocate.py`

**Files:**
- Modify: `pipeline/04_allocate.py`

- [ ] **Step 1: Replace the full `main()` function**

Replace the entire content of `pipeline/04_allocate.py` with:

```python
"""
Step 4 - Portfolio Allocation

Dispatches on cfg['allocation_method']:
  - "parametric_michaud" (default): resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
    raw msr per draw, average the weights, one min_weight floor (src/allocation.resampled_michaud).
  - "msr": the legacy Sharpe-max + batch-elimination loop (src/allocation.msr_eliminate).

Before optimising, the full-universe mu and covmat from step 2 are pre-filtered to the
top allocation_top_n stocks (ranked by allocation_ranking) to keep optimizer compute tractable
for large universes. Set allocation_top_n: null to disable the cap.

Reads  (data/): 01_returns.csv (for T), 02_expected_returns.csv, 02_covmat.csv
Outputs (data/):
    04_weights.csv - optimal weight per held stock
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from config import load_config, PATHS
from allocation import allocate, select_top_n


def main():
    cfg = load_config()

    print("\n=== Step 4: Portfolio Allocation ===")

    returns  = pd.read_csv(PATHS['02_expected_returns'], index_col=0).iloc[:, 0]
    covmat   = pd.read_csv(PATHS['02_covmat'], index_col=0)
    n_periods = len(pd.read_csv(PATHS['01_returns'], index_col=0))

    top_n  = cfg.get('allocation_top_n')
    metric = cfg.get('allocation_ranking', 'sharpe')
    returns, covmat = select_top_n(returns, covmat, top_n, metric)

    method = cfg.get('allocation_method', 'parametric_michaud')
    print(f"Method: {method} | Universe: {len(returns)} stock(s) "
          f"(top_n={top_n}, ranking={metric})")

    weights = allocate(returns, covmat, cfg, n_periods)

    held    = weights[weights.abs() > 1e-9]
    optimal = held.sort_values().to_frame('Weights')

    print(f"\nFinal portfolio: {len(optimal)} stocks")
    print(optimal.sort_values('Weights', ascending=False).to_string())

    optimal.to_csv(PATHS['04_weights'])
    print(f"\nSaved: {PATHS['04_weights']}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the full test suite**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```
git add pipeline/04_allocate.py
git commit -m "Rewire step 4 to read full universe from step 2, apply select_top_n before optimizer"
```

---

### Task 5: Delete step 03 and clean up `pipeline/config.py`

**Files:**
- Delete: `pipeline/03_filter.py`
- Modify: `pipeline/config.py`

- [ ] **Step 1: Delete `pipeline/03_filter.py`**

```
git rm pipeline/03_filter.py
```

- [ ] **Step 2: Remove the three step-3 PATHS entries from `pipeline/config.py`**

In `pipeline/config.py`, find and remove these three lines from the PATHS dict:

```python
    # Step 3 outputs (technical filter / allocation gate)
    '03_selected_returns': os.path.join(DATA_DIR, '03_selected_returns.csv'),
    '03_selected_prices':  os.path.join(DATA_DIR, '03_selected_prices.csv'),
    '03_signals':          os.path.join(DATA_DIR, '03_signals.csv'),
```

The PATHS dict comment block for step 3 should disappear entirely. The dict jumps directly from the step-2 entries to the step-4 entry:

```python
PATHS = {
    # Step 1 outputs
    '01_prices':           os.path.join(DATA_DIR, '01_prices.csv'),
    '01_returns':          os.path.join(DATA_DIR, '01_returns.csv'),
    # Step 2 outputs (Transformer prediction, full universe)
    '02_expected_returns': os.path.join(DATA_DIR, '02_expected_returns.csv'),
    '02_covmat':           os.path.join(DATA_DIR, '02_covmat.csv'),
    '02_predictions':      os.path.join(DATA_DIR, '02_predictions.csv'),
    '02_metadata':         os.path.join(DATA_DIR, '02_metadata.json'),
    # Step 4 outputs
    '04_weights':          os.path.join(DATA_DIR, '04_weights.csv'),
    # Step 5 output — set dynamically by load_config() from cfg['output_path']
    '05_report':           os.path.join(BASE_DIR, 'results', 'allocation_output.csv'),
}
```

- [ ] **Step 3: Run the full test suite**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```
git add pipeline/config.py
git commit -m "Remove step 03 (technical filter) and its PATHS entries from config"
```

---

### Task 6: Build and push the clean PR branch

**Goal:** Reconstruct a clean branch off `origin/main` containing only the 6 functional files, then push.

- [ ] **Step 1: Verify local main is clean**

```
git status
git diff --stat HEAD
```

Expected: clean working tree, no uncommitted changes.

- [ ] **Step 2: Create clean branch from `origin/main`**

```
git checkout -b feat-pr2-filter-removal origin/main
```

- [ ] **Step 3: Apply the 6 functional changes from local main**

```
git checkout main -- src/allocation.py
git checkout main -- pipeline/04_allocate.py
git checkout main -- pipeline/config.py
git checkout main -- params.yaml
git checkout main -- tests/test_allocation.py
git rm pipeline/03_filter.py
```

- [ ] **Step 4: Verify the staged diff — exactly 6 files, nothing else**

```
git diff --cached --name-status origin/main
```

Expected output (order may vary):
```
D       pipeline/03_filter.py
M       pipeline/04_allocate.py
M       pipeline/config.py
M       params.yaml
M       src/allocation.py
M       tests/test_allocation.py
```

If any unexpected files appear, unstage them with `git restore --staged <file>` before continuing.

- [ ] **Step 5: Run the full test suite on this branch**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

```
git commit -m "Remove technical filter (step 03), add top-N compute cap for large universes"
git push -u origin feat-pr2-filter-removal
```

- [ ] **Step 7: Return to local main**

```
git checkout main
```

- [ ] **Step 8: Open the PR on GitHub**

```
gh pr create --title "Remove technical filter + top-N optimizer cap for large universes" --body "$(cat <<'EOF'
## Summary
- Deletes `pipeline/03_filter.py` (SMA/EMA/MACD/PRC technical filter) — look-ahead-prone allocation gate validated off in the monthly-env backtest
- Rewires step 4 to read its candidate universe from step 2 outputs directly (no longer depends on step 3)
- Adds `select_top_n` in `src/allocation.py`: pre-selects top-N stocks by Sharpe proxy (mu/sigma) or raw return before the optimizer, bounding compute for large universes (SLSQP scales O(N^3); 500-stock K=1000 takes ~3.6h; 150-stock takes ~2min)
- Adds `allocation_top_n` (default 150) and `allocation_ranking` (default "sharpe") to params.yaml; `null` = no cap for users running the small universe

## Test plan
- [ ] `pytest tests/test_allocation.py::TestSelectTopN` — 6 new tests for select_top_n
- [ ] `pytest -v` — full suite green
- [ ] Verify `git diff --cached --name-status origin/main` shows exactly 6 files before committing the PR branch
EOF
)"
```
