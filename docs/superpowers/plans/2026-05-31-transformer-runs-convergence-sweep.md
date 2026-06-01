# Transformer-Runs Convergence Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how composition stability (turnover, Jaccard) and per-stock forecast variance converge as `n_transformer_runs` grows, by sweeping a grid and reusing prefix-averaged forecasts.

**Architecture:** A behavior-preserving refactor of `train_and_predict` exposes the raw per-run forecasts (`train_runs`) and the winsorization step (`winsorize_to_history`). A new harness `experiments/sweep_transformer_runs.py` trains `max(grid)` runs once per iteration and derives `μ(n)` from the first-`n` prefix, reusing the stability instrument's pure helpers for allocation and metrics.

**Tech Stack:** Python, numpy, pandas, scikit-learn, scipy (via `risk_kit`), torch (only for the real training path), pytest.

**Spec:** `docs/superpowers/specs/2026-05-31-transformer-runs-convergence-sweep-design.md`.

## CRITICAL environment note
Only this interpreter has the deps — use it everywhere:
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe"` (e.g. `... -m pytest <file> -v`).
Bare `python`/`.venv` lack pandas. Repo is on branch `enh-transformer-runs-sweep`; commit there, never switch branches. Commit messages: NO AI attribution.

---

## File Structure

- **Modify `src/transformer_model.py`** — split `train_and_predict` into `train_runs` + `winsorize_to_history` + a thin `train_and_predict`. Behavior-preserving.
- **Create `experiments/sweep_transformer_runs.py`** — `run_sweep` + `summarize_sweep` + `format_sweep_summary` + `write_sweep_outputs` + `main`. Reuses instrument helpers.
- **Modify `tests/test_transformer_model.py`** — add refactor/shape/winsorize tests.
- **Create `tests/test_sweep_transformer_runs.py`** — sweep logic tests (torch-free via injected stubs).

The sweep reuses these instrument helpers (already in `experiments/measure_allocation_stability.py`): `select_stocks`, `allocate_msr`, `mean_turnover`, `mean_jaccard`, `weight_dispersion`, `seed_everything`.

---

### Task 1: Expose per-run forecasts (behavior-preserving refactor)

**Files:**
- Modify: `src/transformer_model.py`
- Test: `tests/test_transformer_model.py`

- [ ] **Step 1: Write the failing tests** — Append to `tests/test_transformer_model.py`:

```python
import numpy as np
import torch
from transformer_model import train_runs, winsorize_to_history, train_and_predict


def _tiny_cfg():
    return {"time_window": 6, "periods_to_forecast": 2, "n_transformer_runs": 2}


def _tiny_rets(seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.normal(0, 0.02, (30, 3)), columns=["A", "B", "C"])


def test_train_runs_shape():
    runs = train_runs(_tiny_rets(1), _tiny_cfg(), n_runs=2, verbose=False)
    # (n_runs, periods_to_forecast, n_stocks)
    assert runs.shape == (2, 2, 3)


def test_winsorize_to_history_clips_to_percentiles():
    rets = pd.DataFrame({"A": [-0.10, 0.0, 0.10, 0.05, -0.05]})
    preds = pd.DataFrame({"A": [0.5, -0.5, 0.0]})
    out = winsorize_to_history(preds, rets)
    lo = np.percentile(rets.values, 1)
    hi = np.percentile(rets.values, 99)
    assert out["A"].max() <= hi + 1e-12
    assert out["A"].min() >= lo - 1e-12


def test_train_and_predict_composes_from_train_runs():
    # Under the same seed, train_and_predict == winsorize(mean(train_runs)).
    cfg = _tiny_cfg()
    rets = _tiny_rets(0)

    def seeded(fn):
        torch.manual_seed(123)
        np.random.seed(123)
        return fn()

    out_direct = seeded(lambda: train_and_predict(rets, cfg, n_runs=2, verbose=False))
    runs = seeded(lambda: train_runs(rets, cfg, n_runs=2, verbose=False))
    out_compose = winsorize_to_history(
        pd.DataFrame(runs.mean(axis=0), columns=rets.columns), rets
    )
    pd.testing.assert_frame_equal(out_direct, out_compose)
```

- [ ] **Step 2: Run to verify it fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_transformer_model.py -k "train_runs or winsorize or composes" -v` — Expected: FAIL with `ImportError: cannot import name 'train_runs'`.

- [ ] **Step 3: Write the refactor** — In `src/transformer_model.py`, replace the ENTIRE existing `train_and_predict` function (from its `def train_and_predict(returns_df, cfg, n_runs=None, verbose=True):` line through its final `return preds_df`) with these THREE functions:

```python
def train_runs(returns_df, cfg, n_runs=None, verbose=True):
    """Train n_runs Transformers and return the raw per-run forecasts.

    Returns an np.ndarray of shape (n_runs, periods_to_forecast, n_stocks) — no
    averaging, no winsorisation. train_and_predict composes averaging + winsorisation
    on top; the convergence sweep averages prefixes of these runs.
    """
    time_window         = cfg['time_window']
    periods_to_forecast = cfg['periods_to_forecast']
    if n_runs is None:
        n_runs = cfg['n_transformer_runs']

    data = returns_df.values
    X, Y = create_dataset(data, time_window)
    if verbose:
        print(f"Training shapes - X: {X.shape}, Y: {Y.shape}")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    dataset  = TensorDataset(X_tensor, Y_tensor)

    # Prediction input: append a dummy row so create_dataset captures the last window
    data_preds = np.concatenate((data, np.expand_dims(np.zeros_like(data[-1]), axis=0)))
    X_pred, _  = create_dataset(data_preds, time_window)

    all_preds_runs = []
    for run in range(n_runs):
        if verbose:
            print(f"  Training run {run + 1}/{n_runs}...")
        model      = TransformerModel(input_shape=(time_window, X.shape[2])).to(device)
        optimizer  = optim.Adam(model.parameters(), lr=1e-4)
        criterion  = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        scaler     = torch.cuda.amp.GradScaler() if use_amp else None

        model.train()
        for _ in range(50):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = model(batch_x)
                        loss   = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch_x)
                    loss   = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

        model.eval()
        run_preds   = []
        pred_inputs = torch.tensor(X_pred[-1], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(periods_to_forecast):
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred = model(pred_inputs)
                else:
                    pred = model(pred_inputs)
                run_preds.append(pred[0].cpu().numpy())
                pred_inputs = torch.cat((pred_inputs[:, 1:, :], pred.unsqueeze(1)), dim=1)

        all_preds_runs.append(np.array(run_preds))

    return np.array(all_preds_runs)


def winsorize_to_history(preds_df, returns_df):
    """Clip forecasts to the 1st-99th percentile of historical returns."""
    lower_w = np.percentile(returns_df.values, 1)
    upper_w = np.percentile(returns_df.values, 99)
    return preds_df.clip(lower=lower_w, upper=upper_w)


def train_and_predict(returns_df, cfg, n_runs=None, verbose=True):
    """Train n_runs Transformers on returns_df and return averaged, winsorised forecasts.

    Pure with respect to the filesystem: no reads/writes, no date handling. The caller
    supplies returns_df (rows = periods, columns = stocks) and assigns dates to the result.
    """
    runs = train_runs(returns_df, cfg, n_runs=n_runs, verbose=verbose)
    if verbose:
        print(f"Predictions averaged across {runs.shape[0]} runs.")
    preds_df = pd.DataFrame(runs.mean(axis=0), columns=returns_df.columns)
    return winsorize_to_history(preds_df, returns_df)
```

- [ ] **Step 4: Run to verify it passes** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_transformer_model.py -v` — Expected: PASS (the 5 prior Phase-2 tests + the 3 new ones = 8). The composition test trains tiny models twice (~10s).

- [ ] **Step 5: Commit**
```
git add src/transformer_model.py tests/test_transformer_model.py
git commit -m "Expose per-run forecasts via train_runs refactor"
```

---

### Task 2: Sweep scaffold + `run_sweep`

**Files:**
- Create: `experiments/sweep_transformer_runs.py`
- Test: `tests/test_sweep_transformer_runs.py`

- [ ] **Step 1: Write the failing test** — Create `tests/test_sweep_transformer_runs.py`:

```python
"""Tests for experiments/sweep_transformer_runs.py

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py -v

The sweep loop is tested with injected stubs so torch is never imported.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.sweep_transformer_runs import run_sweep


def _inputs():
    rng = np.random.RandomState(2)
    cols = ["A", "B", "C", "D"]
    rets = pd.DataFrame(rng.normal(0.001, 0.02, (40, 4)), index=range(40), columns=cols)
    prices = (1 + rets).cumprod() * 100
    cfg = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05,
           "periods_per_year": 12}
    return prices, rets, cfg


def _stubs():
    # train_runs_fn returns runs where run k is the constant k across (periods, stocks),
    # so the mean of the first n runs is (n-1)/2 everywhere -> prefix-averaging is checkable.
    def train_runs_fn(rets, cfg, n_runs=None):
        periods, stocks = 2, rets.shape[1]
        return np.stack([np.full((periods, stocks), float(k)) for k in range(n_runs)])

    def winsorize_fn(preds_df, rets):
        return preds_df

    def period_mu_fn(preds_df):
        return preds_df.mean(axis=0)

    def select_fn(prices, rets, cfg):
        return ["A", "B", "C", "D"]

    def seed_fn(seed):
        pass

    return train_runs_fn, winsorize_fn, period_mu_fn, select_fn, seed_fn


class TestRunSweep:
    def test_structure_and_shapes(self):
        prices, rets, cfg = _inputs()
        tr, wz, pm, sel, sd = _stubs()
        res = run_sweep(prices, rets, cfg, iterations=3, grid=[10, 20, 50], seed=0,
                        train_runs_fn=tr, winsorize_fn=wz, period_mu_fn=pm,
                        select_fn=sel, seed_fn=sd, verbose=False)
        assert sorted(res["by_n"].keys()) == [10, 20, 50]
        assert res["selected"] == ["A", "B", "C", "D"]
        for n in (10, 20, 50):
            assert res["by_n"][n]["mu"].shape == (3, 4)
            assert res["by_n"][n]["weights"].shape == (3, 4)

    def test_prefix_averaging_uses_first_n_runs(self):
        # mu(n) = mean over the first n runs (= (n-1)/2 for the stub) -> identical per stock.
        prices, rets, cfg = _inputs()
        tr, wz, pm, sel, sd = _stubs()
        res = run_sweep(prices, rets, cfg, iterations=2, grid=[10, 20], seed=0,
                        train_runs_fn=tr, winsorize_fn=wz, period_mu_fn=pm,
                        select_fn=sel, seed_fn=sd, verbose=False)
        assert np.allclose(res["by_n"][10]["mu"].values, (10 - 1) / 2)
        assert np.allclose(res["by_n"][20]["mu"].values, (20 - 1) / 2)
```

- [ ] **Step 2: Run to verify it fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py -v` — Expected: FAIL with `ModuleNotFoundError` / cannot import `run_sweep`.

- [ ] **Step 3: Write minimal implementation** — Create `experiments/sweep_transformer_runs.py`:

```python
"""
Sweep n_transformer_runs and measure how composition stability and per-stock forecast
variance converge as more runs are averaged.

Approach: train max(grid) runs once per iteration, then derive mu(n) for each grid point
by averaging the first n runs (controlled comparison). Reuses the stability instrument's
allocation + metric helpers.

Run:  python experiments/sweep_transformer_runs.py --iterations 30 --grid 10,20,30,40,50,100
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

See docs/superpowers/specs/2026-05-31-transformer-runs-convergence-sweep-design.md.
"""

import argparse
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
    select_stocks, allocate_msr, mean_turnover, mean_jaccard,
    weight_dispersion, seed_everything,
)


def run_sweep(prices, rets, cfg, iterations, grid, seed,
              train_runs_fn=None, winsorize_fn=None, period_mu_fn=None,
              select_fn=None, seed_fn=None, verbose=True):
    """Run `iterations` passes; per pass, train max(grid) runs once and derive mu(n)
    from the first-n prefix for each n in grid. Returns
    {"selected": [...], "by_n": {n: {"mu": DataFrame, "weights": DataFrame}}}.

    The five *_fn arguments are dependency-injection seams (lazy real defaults) so the
    loop is unit-testable without torch. With verbose=True, prints per-iteration elapsed
    time and ETA (streams to the log when run in the background).
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

    grid = sorted(grid)
    max_n = max(grid)

    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    mu_recs = {n: [] for n in grid}
    w_recs = {n: [] for n in grid}
    start = time.time()
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        runs = train_runs_fn(rets, cfg, n_runs=max_n)
        for n in grid:
            prefix = runs[:n].mean(axis=0)
            preds_df = winsorize_fn(pd.DataFrame(prefix, columns=rets.columns), rets)
            mu = period_mu_fn(preds_df)
            mu_sel = mu.loc[selected]
            weights = allocate_msr(mu_sel, cov_sel, cfg)
            mu_recs[n].append(mu_sel)
            w_recs[n].append(weights)
        if verbose:
            elapsed = time.time() - start
            eta = elapsed / i * (iterations - i)
            print(f"[iter {i}/{iterations}] elapsed {elapsed:6.0f}s | ETA {eta:6.0f}s", flush=True)

    by_n = {
        n: {
            "mu": pd.DataFrame(mu_recs[n]).reset_index(drop=True),
            "weights": pd.DataFrame(w_recs[n]).reset_index(drop=True),
        }
        for n in grid
    }
    return {"selected": selected, "by_n": by_n}
```

- [ ] **Step 4: Run to verify it passes** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py -v` — Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```
git add experiments/sweep_transformer_runs.py tests/test_sweep_transformer_runs.py
git commit -m "Add run_sweep: prefix-averaged transformer-runs sweep"
```

---

### Task 3: `summarize_sweep`

**Files:**
- Modify: `experiments/sweep_transformer_runs.py`
- Test: `tests/test_sweep_transformer_runs.py`

- [ ] **Step 1: Write the failing test** — Append to `tests/test_sweep_transformer_runs.py`:

```python
from experiments.sweep_transformer_runs import summarize_sweep


def _mk(weights_rows, mu_rows):
    return {"weights": pd.DataFrame(weights_rows), "mu": pd.DataFrame(mu_rows)}


def _decreasing_result():
    # mu across-iteration std strictly shrinks with n; metrics keep changing >5% each
    # step, so no point "converges" (converged_n is None).
    by_n = {
        10: _mk([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 0.0}],
                [{"A": 0.20, "B": 0.05}, {"A": 0.05, "B": 0.20}, {"A": 0.20, "B": 0.05}]),
        20: _mk([{"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, {"A": 0.55, "B": 0.45}],
                [{"A": 0.120, "B": 0.100}, {"A": 0.100, "B": 0.120}, {"A": 0.120, "B": 0.100}]),
        30: _mk([{"A": 0.52, "B": 0.48}, {"A": 0.50, "B": 0.50}, {"A": 0.51, "B": 0.49}],
                [{"A": 0.112, "B": 0.108}, {"A": 0.108, "B": 0.112}, {"A": 0.112, "B": 0.108}]),
    }
    return {"selected": ["A", "B"], "by_n": by_n}


def _converged_result():
    # n=20 and n=30 have IDENTICAL weights/mu -> every metric matches at the 20->30 step
    # (relative delta 0), so converged_n is 30. n=10 differs enough that 20 is not converged.
    n10 = _mk([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}, {"A": 1.0, "B": 0.0}],
              [{"A": 0.20, "B": 0.05}, {"A": 0.05, "B": 0.20}, {"A": 0.20, "B": 0.05}])
    common = _mk([{"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, {"A": 0.55, "B": 0.45}],
                 [{"A": 0.12, "B": 0.10}, {"A": 0.10, "B": 0.12}, {"A": 0.11, "B": 0.11}])
    common2 = _mk(common["weights"].copy(), common["mu"].copy())
    return {"selected": ["A", "B"], "by_n": {10: n10, 20: common, 30: common2}}


class TestSummarizeSweep:
    def test_table_columns_and_index(self):
        t = summarize_sweep(_decreasing_result())["table"]
        assert list(t.index) == [10, 20, 30]
        for c in ("turnover", "jaccard", "mean_mu_std", "d_turnover", "d_jaccard", "d_mu_std"):
            assert c in t.columns

    def test_mu_std_decreases_with_n(self):
        t = summarize_sweep(_decreasing_result())["table"]
        assert t.loc[10, "mean_mu_std"] > t.loc[20, "mean_mu_std"] > t.loc[30, "mean_mu_std"]

    def test_converged_n_detected(self):
        # 20 and 30 are identical -> all three metrics unchanged at the 20->30 step.
        assert summarize_sweep(_converged_result())["converged_n"] == 30

    def test_steady_descent_does_not_converge(self):
        # Every step still changes >5% -> no converged point.
        assert summarize_sweep(_decreasing_result())["converged_n"] is None

    def test_single_grid_point_returns_none(self):
        one = {"selected": ["A", "B"], "by_n": {
            10: _mk([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}],
                    [{"A": 0.2, "B": 0.0}, {"A": 0.0, "B": 0.2}])}}
        assert summarize_sweep(one)["converged_n"] is None
```

- [ ] **Step 2: Run to verify it fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py::TestSummarizeSweep -v` — Expected: FAIL with `ImportError: cannot import name 'summarize_sweep'`.

- [ ] **Step 3: Write minimal implementation** — Add to `experiments/sweep_transformer_runs.py` (after `run_sweep`):

```python
def summarize_sweep(result, rel_tol=0.05):
    """Per-n turnover / Jaccard / mean per-stock mu std, with consecutive-n deltas and the
    first n where all three change < rel_tol relative to the previous grid point."""
    grid = sorted(result["by_n"].keys())
    cols = ["turnover", "jaccard", "mean_mu_std"]

    data = {}
    for n in grid:
        wdf = result["by_n"][n]["weights"]
        mdf = result["by_n"][n]["mu"]
        data[n] = {
            "turnover": float(mean_turnover(wdf)),
            "jaccard": float(mean_jaccard(wdf)),
            "mean_mu_std": float(weight_dispersion(mdf).mean()),
        }
    table = pd.DataFrame(data).T[cols]
    table.index.name = "n"
    for c in cols:
        table[f"d_{c}"] = table[c].diff()

    converged_n = None
    for prev, n in zip(grid[:-1], grid[1:]):
        ok = True
        for c in cols:
            base = table.loc[prev, c]
            if base == 0:
                continue
            if abs(table.loc[n, c] - base) / abs(base) >= rel_tol:
                ok = False
                break
        if ok:
            converged_n = n
            break

    return {"table": table, "converged_n": converged_n, "rel_tol": rel_tol}
```

- [ ] **Step 4: Run to verify it passes** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py::TestSummarizeSweep -v` — Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```
git add experiments/sweep_transformer_runs.py tests/test_sweep_transformer_runs.py
git commit -m "Add summarize_sweep: convergence table + deltas"
```

---

### Task 4: `format_sweep_summary` and `write_sweep_outputs`

**Files:**
- Modify: `experiments/sweep_transformer_runs.py`
- Test: `tests/test_sweep_transformer_runs.py`

- [ ] **Step 1: Write the failing test** — Append to `tests/test_sweep_transformer_runs.py`:

```python
from experiments.sweep_transformer_runs import format_sweep_summary, write_sweep_outputs


class TestSweepOutputs:
    def test_format_mentions_converged_n(self):
        text = format_sweep_summary(summarize_sweep(_converged_result()))
        assert "n=30" in text

    def test_format_handles_no_convergence(self):
        text = format_sweep_summary(summarize_sweep(_decreasing_result()))
        assert "No grid point" in text

    def test_write_outputs_creates_files(self, tmp_path):
        paths = write_sweep_outputs(summarize_sweep(_decreasing_result()), str(tmp_path))
        assert os.path.exists(paths["metrics"])
        assert os.path.exists(paths["summary"])
        reloaded = pd.read_csv(paths["metrics"], index_col=0)
        assert list(reloaded.columns) == ["turnover", "jaccard", "mean_mu_std"]
        assert list(reloaded.index) == [10, 20, 30]
```

- [ ] **Step 2: Run to verify it fails** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py::TestSweepOutputs -v` — Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation** — Add to `experiments/sweep_transformer_runs.py` (after `summarize_sweep`):

```python
def format_sweep_summary(summary):
    """Human-readable convergence report from a summarize_sweep result."""
    table = summary["table"]
    conv = summary["converged_n"]
    tol = summary["rel_tol"]

    lines = ["Transformer-runs convergence sweep", ""]
    lines.append(table.to_string(float_format=lambda v: f"{v:.5f}"))
    lines.append("")
    if conv is None:
        lines.append(f"No grid point met the <{tol:.0%} relative-delta convergence criterion.")
    else:
        lines.append(
            f"Converged at n={conv} (turnover, jaccard, and mean_mu_std each changed "
            f"<{tol:.0%} vs the previous grid point)."
        )
    return "\n".join(lines)


def write_sweep_outputs(summary, outdir):
    """Write sweep_metrics.csv and sweep_summary.txt; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "metrics": os.path.join(outdir, "sweep_metrics.csv"),
        "summary": os.path.join(outdir, "sweep_summary.txt"),
    }
    summary["table"][["turnover", "jaccard", "mean_mu_std"]].to_csv(paths["metrics"])
    with open(paths["summary"], "w") as f:
        f.write(format_sweep_summary(summary))
    return paths
```

- [ ] **Step 4: Run to verify it passes** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py -v` — Expected: PASS (all sweep tests: 2 + 5 + 3 = 10).

- [ ] **Step 5: Commit**
```
git add experiments/sweep_transformer_runs.py tests/test_sweep_transformer_runs.py
git commit -m "Add sweep reporting: format + write outputs"
```

---

### Task 5: `main()` CLI + full suite + manual sweep run

**Files:**
- Modify: `experiments/sweep_transformer_runs.py`

- [ ] **Step 1: Add the CLI entry point** — Add to the END of `experiments/sweep_transformer_runs.py`:

```python
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--grid", type=str, default="10,20,30,40,50,100",
                        help="Comma-separated n_transformer_runs values")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results", "sweep"))
    args = parser.parse_args()

    grid = [int(x) for x in args.grid.split(",")]
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    print(f"Sweep grid={grid} | iterations={args.iterations} | seed={args.seed}")
    result = run_sweep(prices, rets, cfg, iterations=args.iterations, grid=grid, seed=args.seed)
    summary = summarize_sweep(result)
    paths = write_sweep_outputs(summary, args.outdir)

    print()
    print(format_sweep_summary(summary))
    print(f"\nSaved: {paths['metrics']}\n       {paths['summary']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_sweep_transformer_runs.py tests/test_transformer_model.py tests/test_measure_allocation_stability.py -v` — Expected: PASS (sweep 10 + transformer_model 8 + instrument 31 = 49).

- [ ] **Step 3: Verify the CLI parses** — `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/sweep_transformer_runs.py --help` — Expected: usage with `--iterations`, `--grid`, `--seed`, `--outdir`; exit 0.

- [ ] **Step 4: Manual sweep run (controller, GPU-heavy ~3000 trainings).** Run in the background
with a log so progress is trackable (`run_sweep` prints `[iter i/30] elapsed Xs | ETA Ys` per
iteration; `Read` the log any time to report status):
`"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/sweep_transformer_runs.py --iterations 30 --grid 10,20,30,40,50,100 --seed 0 --outdir experiments/results/sweep > experiments/results/sweep_run.log 2>&1`
Then read `experiments/results/sweep/sweep_summary.txt`. Expected: `mean_mu_std` decreases monotonically with `n`; turnover/Jaccard trend toward a plateau; a converged-`n` is reported (or "no grid point" if the descent is still steep at 100). Report the table and the convergence point.

- [ ] **Step 5: Commit**
```
git add experiments/sweep_transformer_runs.py
git commit -m "Add main CLI for transformer-runs sweep"
```

---

## Self-Review

**1. Spec coverage:**
- Behavior-preserving refactor exposing per-run forecasts → Task 1 (`train_runs`, `winsorize_to_history`, composed `train_and_predict`). ✓
- Prefix-averaging from one max-run training per iteration → Task 2 (`run_sweep`). ✓
- Reuse instrument helpers + DI seams for torch-free tests → Task 2 imports + the five `*_fn` seams. ✓
- Per-n metrics (turnover, Jaccard, mean per-stock mu std) + consecutive deltas + converged-n at rel_tol → Task 3 (`summarize_sweep`). ✓
- Numbers-only outputs `sweep_metrics.csv` + `sweep_summary.txt` → Task 4. ✓
- CLI `--iterations`/`--grid`/`--seed`/`--outdir` → Task 5. ✓
- Tests: refactor equivalence + shape + winsorize (T1); run_sweep structure + prefix-averaging (T2); summarize incl. converged + none cases (T3); outputs (T4); manual full sweep (T5). ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step shows full code; every test step shows complete asserts. ✓

**3. Type consistency:** `run_sweep` returns `{"selected", "by_n": {n: {"mu","weights"}}}`; `summarize_sweep` consumes `result["by_n"][n]["weights"|"mu"]` and returns `{"table","converged_n","rel_tol"}`; `format_sweep_summary`/`write_sweep_outputs` consume exactly those keys and `table` columns `turnover/jaccard/mean_mu_std`. `train_runs` returns `(n_runs, periods, stocks)`, consumed as `runs[:n].mean(axis=0)` in Task 2. Seam names (`train_runs_fn`, `winsorize_fn`, `period_mu_fn`, `select_fn`, `seed_fn`) are identical across Task 2 code and tests. ✓

No gaps found.
