# Productionise Parametric Michaud (Step 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add parametric Michaud as the production step-4 allocation method behind a config toggle (default it), keeping the current msr+elimination as a fallback, by extracting both into a new pushable `src/allocation.py` and making `pipeline/04_allocate.py` a thin dispatcher.

**Architecture:** New `src/allocation.py` holds the allocation *policies* (`msr_eliminate`, `sample_mu_draws`, `apply_consensus_floor`, `resampled_michaud`, `allocate` dispatcher) as pure functions reusing `risk_kit.msr_tuned`. `04_allocate.py` loads μ̄/Σ/selected + `T=len(01_returns)`, calls `allocate(...)`, writes `04_weights.csv` unchanged. Config gains `allocation_method` (default `parametric_michaud`), `michaud_spread`, `michaud_mc_draws`, `michaud_seed`.

**Tech Stack:** Python, numpy, pandas, scipy (`risk_kit.msr_tuned`), pytest. Tests are torch-free.

**Conventions:**
- Run tests with the project interpreter: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`.
- Commit messages: imperative, capitalised, **no AI attribution**.
- Work on branch `enh-allocation-prod-michaud` (already created; spec committed there).
- This is the first **pushable** phase: `src/allocation.py`, `pipeline/04_allocate.py`, `params.yaml`, **and `tests/test_allocation.py`** get pushed at cutover (tests now ship with the code). Spec/plan/docs stay local. **Merge/push is gated** on the robustness backtest confirming — build only here.
- **Option A** on `max_weight`: keep the validated consensus-floor logic (a top consensus weight may marginally exceed `max_weight`); do NOT add a post-floor cap.

---

## File Structure

- **Create:** `src/allocation.py` — allocation policies (all functions below). Self-inserts its own dir on `sys.path` so `import risk_kit` resolves regardless of caller.
- **Create:** `tests/test_allocation.py` — torch-free unit tests (one class per task).
- **Modify:** `pipeline/04_allocate.py` — thin dispatcher (Task 4).
- **Modify:** `params.yaml` — four new keys (Task 4).
- **Reference (do not modify):** `src/risk_kit.py` (`msr_tuned(riskfree_rate, returns, covmat, max_weight, periods_per_year, debug)`), `pipeline/config.py` (raw keys flow through `load_config`).

**Port source (validated):** `msr_eliminate` == experiment `allocate_msr`; `apply_consensus_floor`, `sample_mu_draws`, and the consensus logic == experiment `measure_allocation_stability.py` (Phase 3/3b). Production gets its own copy (experiments stay local).

---

### Task 1: `src/allocation.py` module + `msr_eliminate`

Establish the module (imports + path self-insert) and port the current msr+batch-elimination loop as a testable function. It must reproduce today's `04_allocate` weights.

**Files:**
- Create: `src/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_allocation.py`:

```python
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from allocation import msr_eliminate

CFG = {
    "rf_period": 0.0, "rf_rate": 0.0,
    "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12,
    "allocation_method": "parametric_michaud",
    "michaud_spread": 1.0, "michaud_mc_draws": 200, "michaud_seed": 0,
}


def _cov(names, var=0.04):
    return pd.DataFrame(np.diag([var] * len(names)), index=names, columns=names)


class TestMsrEliminate:
    def test_sums_to_one(self):
        names = ["A", "B", "C"]
        mu = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_returns_series_over_full_index(self):
        names = ["A", "B", "C"]
        mu = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert list(w.index) == names

    def test_respects_max_weight(self):
        names = ["A", "B", "C", "D", "E"]
        mu = pd.Series(dict(zip(names, [0.30, 0.05, 0.05, 0.05, 0.05])))
        w = msr_eliminate(mu, _cov(names), CFG)
        assert (w <= CFG["max_weight"] + 1e-6).all()

    def test_two_assets_both_held(self):
        names = ["A", "B"]
        mu = pd.Series({"A": 0.20, "B": 0.10})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert (w.abs() > 0).sum() == 2
        assert abs(w.sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py::TestMsrEliminate -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'allocation'`.

- [ ] **Step 3: Create the module with `msr_eliminate`**

Create `src/allocation.py`:

```python
"""
Portfolio allocation policies for step 4.

Two methods behind cfg['allocation_method']:
  - "msr"                : Sharpe-max + batch-elimination (the legacy method).
  - "parametric_michaud" : resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
                           raw msr per draw, average the weight vectors, one min_weight floor.

Reuses risk_kit.msr_tuned for the per-portfolio optimisation. Pure functions (no I/O);
pipeline/04_allocate.py does the file reading/writing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import risk_kit as rk


def msr_eliminate(returns, covmat, cfg):
    """Sharpe-maximising weights with the batch-elimination loop (the legacy step-4 method).

    Iteratively drops names whose cumulative weight (sorted ascending) is below min_weight and
    re-optimises, until all survivors pass or <=2 remain. Returns a Series over returns.index;
    eliminated names get 0.0.
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    names = list(returns.index)

    w0 = rk.msr_tuned(
        riskfree_rate=rf, returns=returns, covmat=covmat,
        max_weight=max_w, periods_per_year=ppy, debug=False,
    )
    optimal = pd.DataFrame(w0, index=returns.index, columns=["Weights"]).sort_values("Weights")

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
        optimal = pd.DataFrame(w, index=optimal.index, columns=["Weights"]).sort_values("Weights")

    weights = pd.Series(0.0, index=names)
    weights[optimal.index] = optimal["Weights"]
    return weights
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py::TestMsrEliminate -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/allocation.py tests/test_allocation.py
git commit -m "Add src/allocation.py with msr_eliminate (legacy step-4 method)"
```

---

### Task 2: `apply_consensus_floor` + `sample_mu_draws`

The two parametric building blocks: the min-weight floor on an averaged consensus, and the Monte-Carlo μ draw generator.

**Files:**
- Modify: `src/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_allocation.py`:

```python
from allocation import apply_consensus_floor, sample_mu_draws


class TestApplyConsensusFloor:
    def test_sums_to_one_and_preserves_index(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert list(out.index) == ["A", "B", "C"]
        assert abs(out.sum() - 1.0) < 1e-9

    def test_drops_small_tail_and_renormalises(self):
        w = pd.Series({"A": 0.60, "B": 0.38, "C": 0.02})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert out["C"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-9
        assert abs(out["A"] - 0.60 / 0.98) < 1e-9

    def test_no_drop_when_all_above_floor(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert (out == pd.Series({"A": 0.5, "B": 0.5})).all()

    def test_stops_at_two_survivors(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.9)
        assert (out.abs() > 0).sum() == 2
        assert abs(out.sum() - 1.0) < 1e-9


def _cov3():
    names = ["A", "B", "C"]
    return pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.04, 0.01], [0.00, 0.01, 0.04]],
        index=names, columns=names,
    )


def _mu3():
    return pd.Series({"A": 0.01, "B": 0.02, "C": 0.015})


class TestSampleMuDraws:
    def test_returns_n_draws_over_index(self):
        draws = sample_mu_draws(_mu3(), _cov3(), 10, 5, 1.0, np.random.default_rng(0))
        assert len(draws) == 5
        for d in draws:
            assert list(d.index) == ["A", "B", "C"]

    def test_spread_zero_returns_copies(self):
        mu = _mu3()
        draws = sample_mu_draws(mu, _cov3(), 10, 4, 0.0, np.random.default_rng(0))
        for d in draws:
            assert (d == mu).all()

    def test_large_k_mean_approx_mu(self):
        mu = _mu3()
        draws = sample_mu_draws(mu, _cov3(), 10, 50000, 1.0, np.random.default_rng(1))
        assert np.allclose(pd.DataFrame(draws).mean(axis=0).values, mu.values, atol=2e-3)

    def test_seeded_reproducible(self):
        a = sample_mu_draws(_mu3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        b = sample_mu_draws(_mu3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        assert np.allclose(pd.DataFrame(a).values, pd.DataFrame(b).values)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py::TestApplyConsensusFloor tests/test_allocation.py::TestSampleMuDraws -v`
Expected: FAIL with `ImportError: cannot import name 'apply_consensus_floor'`.

- [ ] **Step 3: Implement the two functions**

Add to `src/allocation.py` (after `msr_eliminate`):

```python
def apply_consensus_floor(weights, min_weight):
    """Enforce the min-weight floor on an averaged consensus, without re-optimising.

    Drops names whose cumulative weight (sorted ascending) is below min_weight and renormalises
    the survivors to sum to 1, iterating until every survivor passes; the len<=2 guard keeps the
    book from emptying. Returns a Series over the original index (dropped = 0.0).
    """
    names = list(weights.index)
    w = weights[weights > 0].sort_values().copy()
    while len(w) > 2:
        cum = w.cumsum()
        failing = cum < min_weight
        if not failing.any():
            break
        w = w[~failing]
        w = (w / w.sum()).sort_values()
    out = pd.Series(0.0, index=names)
    out[w.index] = w
    return out


def sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng):
    """Draw n_draws mu vectors from the canonical Michaud law N(mu_bar, spread**2 * Sigma / T).

    spread=0 returns n_draws exact copies of mu_bar. Sampled via one Cholesky of the scaled
    covariance. Returns a list of Series over mu_bar.index (name order preserved).
    """
    names = list(mu_bar.index)
    if spread == 0:
        return [mu_bar.copy() for _ in range(n_draws)]
    scale = spread ** 2 / n_periods
    cov_scaled = covmat.loc[names, names].values * scale
    chol = np.linalg.cholesky(cov_scaled)
    z = rng.standard_normal((n_draws, len(names)))
    samples = mu_bar.values + z @ chol.T
    return [pd.Series(samples[k], index=names) for k in range(n_draws)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py::TestApplyConsensusFloor tests/test_allocation.py::TestSampleMuDraws -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/allocation.py tests/test_allocation.py
git commit -m "Add consensus floor and parametric mu-draw generator"
```

---

### Task 3: `resampled_michaud` + `allocate` dispatcher

Tie the parametric pieces together (seeded MC draws → raw msr per draw → average → floor), and add the method dispatcher.

**Files:**
- Modify: `src/allocation.py`
- Test: `tests/test_allocation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_allocation.py`:

```python
from allocation import resampled_michaud, allocate


@pytest.fixture
def cov5():
    names = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.diag([0.04] * 5), index=names, columns=names)


def _mu5(vals):
    return pd.Series(dict(zip(["A", "B", "C", "D", "E"], vals)))


class TestResampledMichaud:
    def test_consensus_sums_to_one(self, cov5):
        w = resampled_michaud(_mu5([0.20, 0.02, 0.02, 0.02, 0.02]), cov5, CFG, n_periods=100)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_deterministic_with_seed(self, cov5):
        mu = _mu5([0.20, 0.05, 0.02, 0.02, 0.02])
        w1 = resampled_michaud(mu, cov5, CFG, n_periods=100)
        w2 = resampled_michaud(mu, cov5, CFG, n_periods=100)
        assert np.allclose(w1.values, w2.values)

    def test_null_seed_still_valid(self, cov5):
        cfg = dict(CFG)
        cfg["michaud_seed"] = None
        w = resampled_michaud(_mu5([0.20, 0.05, 0.02, 0.02, 0.02]), cov5, cfg, n_periods=100)
        assert abs(w.sum() - 1.0) < 1e-6


class TestAllocateDispatcher:
    def test_routes_to_msr(self, cov5):
        cfg = dict(CFG); cfg["allocation_method"] = "msr"
        mu = _mu5([0.20, 0.10, 0.05, 0.05, 0.05])
        got = allocate(mu, cov5, cfg, n_periods=100)
        want = msr_eliminate(mu, cov5, cfg)
        assert np.allclose(got.values, want.values)

    def test_routes_to_parametric(self, cov5):
        cfg = dict(CFG); cfg["allocation_method"] = "parametric_michaud"
        mu = _mu5([0.20, 0.05, 0.02, 0.02, 0.02])
        got = allocate(mu, cov5, cfg, n_periods=100)
        want = resampled_michaud(mu, cov5, cfg, n_periods=100)
        assert np.allclose(got.values, want.values)

    def test_unknown_method_raises(self, cov5):
        cfg = dict(CFG); cfg["allocation_method"] = "bogus"
        with pytest.raises(ValueError):
            allocate(_mu5([0.1] * 5), cov5, cfg, n_periods=100)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py::TestResampledMichaud tests/test_allocation.py::TestAllocateDispatcher -v`
Expected: FAIL with `ImportError: cannot import name 'resampled_michaud'`.

- [ ] **Step 3: Implement `resampled_michaud` and `allocate`**

Add to `src/allocation.py` (after `sample_mu_draws`):

```python
def resampled_michaud(returns, covmat, cfg, n_periods):
    """Parametric Michaud consensus: draw K mu ~ N(mu_bar, s^2*Sigma/T), raw msr per draw,
    average the weight vectors, one min_weight floor.

    Reads cfg['michaud_spread'] (s), cfg['michaud_mc_draws'] (K), cfg['michaud_seed'] (int for a
    reproducible draw set, or None for fresh draws). max_weight is enforced per draw by msr_tuned.
    Returns a Series over returns.index (dropped names = 0.0).
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    spread = cfg.get("michaud_spread", 1.0)
    n_draws = cfg.get("michaud_mc_draws", 1000)
    seed = cfg.get("michaud_seed", 0)

    rng = np.random.default_rng(seed)
    draws = sample_mu_draws(returns, covmat, n_periods, n_draws, spread, rng)

    rows = []
    for mu_i in draws:
        arr = rk.msr_tuned(
            riskfree_rate=rf, returns=mu_i, covmat=covmat.loc[mu_i.index, mu_i.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
        )
        rows.append(pd.Series(arr, index=mu_i.index))

    raw = pd.DataFrame(rows).reset_index(drop=True)
    return apply_consensus_floor(raw.mean(axis=0), min_w)


def allocate(returns, covmat, cfg, n_periods):
    """Dispatch to the configured allocation method (cfg['allocation_method'])."""
    method = cfg.get("allocation_method", "parametric_michaud")
    if method == "msr":
        return msr_eliminate(returns, covmat, cfg)
    if method == "parametric_michaud":
        return resampled_michaud(returns, covmat, cfg, n_periods)
    raise ValueError(f"unknown allocation_method: {method!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py::TestResampledMichaud tests/test_allocation.py::TestAllocateDispatcher -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/allocation.py tests/test_allocation.py
git commit -m "Add resampled_michaud consensus and allocate dispatcher"
```

---

### Task 4: `params.yaml` keys + rewrite `04_allocate.py`

Wire the config and make step 4 a thin dispatcher. Default to parametric Michaud.

**Files:**
- Modify: `params.yaml`
- Modify: `pipeline/04_allocate.py`
- Test: full suite + a manual two-method run (script, validated by running).

- [ ] **Step 1: Add the config keys**

In `params.yaml`, replace the "Portfolio optimisation (Step 4)" block:

```yaml
# Portfolio optimisation (Step 4)
allocation_method: parametric_michaud   # "parametric_michaud" | "msr"
michaud_spread: 1.0                      # s; draw scale on Sigma/T (1.0 = canonical)
michaud_mc_draws: 1000                   # K Monte-Carlo mu draws
michaud_seed: 0                          # int = reproducible; null = fresh draws
rf_rate: 0.11                 # Risk-free rate (10-Y Colombian bond yield ~11.2%)
max_weight: 0.15              # Maximum portfolio weight per stock
min_weight: 0.05              # Minimum portfolio weight per stock
```

(Keep the existing `rf_rate`/`max_weight`/`min_weight` values; just prepend the four new keys.)

- [ ] **Step 2: Rewrite `04_allocate.py`**

Replace the entire body of `pipeline/04_allocate.py` with:

```python
"""
Step 4 - Portfolio Allocation

Dispatches on cfg['allocation_method']:
  - "parametric_michaud" (default): resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
    raw msr per draw, average the weights, one min_weight floor (src/allocation.resampled_michaud).
  - "msr": the legacy Sharpe-max + batch-elimination loop (src/allocation.msr_eliminate).

The technical filter (step 3) is the allocation gate: full-universe predictions from step 2 are
restricted to the selected names before allocating.

Reads  (data/): 01_returns.csv (for T), 02_expected_returns.csv, 02_covmat.csv, 03_selected_returns.csv
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
from allocation import allocate


def main():
    cfg = load_config()

    print("\n=== Step 4: Portfolio Allocation ===")

    expected_returns = pd.read_csv(PATHS['02_expected_returns'], index_col=0).iloc[:, 0]
    covmat = pd.read_csv(PATHS['02_covmat'], index_col=0)
    n_periods = len(pd.read_csv(PATHS['01_returns'], index_col=0))

    # Allocation gate: restrict the full-universe predictions to the step-3 selection.
    selected = pd.read_csv(PATHS['03_selected_returns'], index_col=0, nrows=0).columns
    selected = [s for s in selected if s in expected_returns.index and s in covmat.index]

    method = cfg.get('allocation_method', 'parametric_michaud')
    print(f"Method: {method} | Allocation universe: {len(selected)} selected stock(s).")

    returns = expected_returns[selected]
    covmat = covmat.loc[selected, selected]

    weights = allocate(returns, covmat, cfg, n_periods)

    held = weights[weights.abs() > 1e-9]
    optimal = held.sort_values().to_frame('Weights')

    print(f"\nFinal portfolio: {len(optimal)} stocks")
    print(optimal.sort_values('Weights', ascending=False).to_string())

    optimal.to_csv(PATHS['04_weights'])
    print(f"\nSaved: {PATHS['04_weights']}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run the full test suite**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_allocation.py -v`
Expected: PASS (18 passed: 4 + 8 + 6).

Run the whole repo suite to confirm no regressions: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/ -q`
Expected: PASS (all green).

- [ ] **Step 4: Manual validation (both methods on real data)**

This validates the thin script + the legacy-equivalence requirement; it is not a unit test (matches the pipeline convention). Requires `data/01_returns.csv`, `data/02_expected_returns.csv`, `data/02_covmat.csv`, `data/03_selected_returns.csv` to exist (run steps 1-3 first if needed).

First capture the legacy weights from the current `main` (before this change is exercised) is not possible post-edit, so instead verify msr-equivalence by running step 4 with `allocation_method: msr` and confirming it reproduces the committed `data/04_weights.csv` if present, OR that it matches the legacy logic by spot-check. Run:

```bash
# msr method (legacy): should match prior 04_weights values per ticker
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -c "import yaml,io; c=yaml.safe_load(open('params.yaml')); c['allocation_method']='msr'; open('params.yaml','w').write(yaml.safe_dump(c, sort_keys=False))"
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" pipeline/04_allocate.py

# parametric method (new default): runs and produces a valid book
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -c "import yaml; c=yaml.safe_load(open('params.yaml')); c['allocation_method']='parametric_michaud'; open('params.yaml','w').write(yaml.safe_dump(c, sort_keys=False))"
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" pipeline/04_allocate.py
```

Expected: both runs complete and write `data/04_weights.csv`; weights sum to ~1; the msr run reproduces the legacy held names/weights (compare values per ticker against a saved baseline copy of `04_weights.csv` taken before this task). Confirm `params.yaml` ends with `allocation_method: parametric_michaud` (the intended default) after validation.

- [ ] **Step 5: Commit**

```bash
git add params.yaml pipeline/04_allocate.py
git commit -m "Make step 4 dispatch allocation method; default parametric Michaud"
```

---

## Post-implementation (gating, not a code task)

- **Do NOT merge or push yet.** Hold until the robustness backtest (oos=250) confirms parametric wins.
- On confirmation: ff-merge to local `main`, then push the functional changes **and their tests** (`src/allocation.py`, `pipeline/04_allocate.py`, `params.yaml`, `tests/test_allocation.py`) to a remote branch → PR (mirror PR #21). Spec/plan/docs stay local.
- If robustness contradicts: set `allocation_method: msr` as the default and reassess.

---

## Self-Review notes

- **Spec coverage:** new `src/allocation.py` with all five functions (Tasks 1-3); `04_allocate` thin dispatcher reading μ̄/Σ/selected + `T=len(01_returns)` (Task 4); config toggle defaulting to parametric (Task 4); seed handling (int → reproducible, `None` → fresh) in `resampled_michaud` + test (Task 3); Option A (no post-floor cap) — `resampled_michaud` does not cap, and the resampled test does not assert `max_weight` (Task 3); msr-equivalence pinned by the legacy port (Task 1) + manual validation (Task 4); tests pushed with code (header note + gating section). Out-of-scope items (step-2 seeding, Option B cap, s-sweep, diagnostic output) absent.
- **Placeholder scan:** none — every code/test step is complete; the manual step gives exact commands.
- **Type consistency:** `allocate(returns, covmat, cfg, n_periods)` defined in Task 3, called identically in `04_allocate` (Task 4). `resampled_michaud(returns, covmat, cfg, n_periods)` and `sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng)` consistent across Tasks 2-3. `msr_eliminate`/`apply_consensus_floor` signatures match the experiment originals and `risk_kit.msr_tuned`'s keyword args. Config keys (`michaud_spread/mc_draws/seed`, `allocation_method`) consistent between `params.yaml` (Task 4) and `resampled_michaud`/`allocate` (Task 3).
