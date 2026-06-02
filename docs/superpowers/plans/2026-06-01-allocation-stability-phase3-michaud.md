# Phase 3 — Michaud Resampled Efficiency (Validation Harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Michaud resampled-efficiency allocation path and a paired current-vs-Michaud comparison to the existing stability instrument, to validate (in the experiment harness only) whether resampling reduces composition instability without degrading value.

**Architecture:** Each per-draw μ (one Transformer run) is optimised by raw `msr_tuned` (no elimination); the per-draw weight vectors are averaged into a consensus, then a single floor pass enforces `min_weight` (average-then-threshold). A new paired loop derives both the current method (average μ → one allocation) and the Michaud consensus from the **same** N runs each iteration, so the before/after comparison is exactly paired at zero extra training cost. All changes are additive and local-only; `pipeline/` and `src/` are untouched (productionisation is a later phase).

**Tech Stack:** Python, pandas, numpy, scipy (`scipy.optimize` via `risk_kit.msr_tuned`), pytest. Tests are torch-free via dependency-injected stubs, matching the existing `tests/test_measure_allocation_stability.py`.

**Conventions:**
- Run tests with the project interpreter: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`.
- Commit messages: imperative, capitalised, **no AI attribution** (no `Co-Authored-By`, no "Generated with"); match the repo style (e.g. "Refactor train_and_predict to expose per-run forecasts").
- All new code goes in `experiments/measure_allocation_stability.py`; all new tests in `tests/test_measure_allocation_stability.py`.
- The existing single-arm `run_experiment` / `format_summary` / `write_outputs` and their tests stay **unchanged** — the paired path is added alongside.

---

## File Structure

- **Modify:** `experiments/measure_allocation_stability.py` — add `apply_consensus_floor`, `resampled_allocate`, `overlap_stats`, `run_paired_experiment`, `format_paired_summary`, `write_paired_outputs`, `train_runs_as_preds` (default runs_fn), and a `--mode paired` CLI branch. Reuses the existing `allocate_msr`, `portfolio_metrics`, `selection_frequency`, `weight_dispersion`, `mean_turnover`, `mean_jaccard`, `metric_dispersion`, `select_stocks`, `seed_everything`, `_fmt`.
- **Modify:** `tests/test_measure_allocation_stability.py` — add test classes for the new functions. Existing tests are not touched.
- **Reference (do not modify):** `src/risk_kit.py` (`msr_tuned`, `portfolio_return`, `portfolio_vol`), `src/transformer_model.py` (`train_runs`, `winsorize_to_history`, `weighted_mean_return`), `pipeline/04_allocate.py` (the elimination semantics being mirrored).

---

### Task 1: `apply_consensus_floor` — floor the averaged consensus (no re-optimisation)

The consensus is an average of weight vectors, not an `msr` solution, so there is no single μ to re-optimise against. The floor must therefore **drop names whose cumulative weight (ascending) is below `min_weight` and renormalise**, iterating until all survivors pass — mirroring `04_allocate.py`'s batch-elimination *rule* but renormalising instead of re-optimising.

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import apply_consensus_floor


class TestApplyConsensusFloor:
    def test_sums_to_one_and_preserves_index(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert list(out.index) == ["A", "B", "C"]
        assert abs(out.sum() - 1.0) < 1e-9

    def test_drops_small_tail_and_renormalises(self):
        # C (0.02) sits below the 0.05 floor on cumulative weight; it is dropped
        # and the survivors renormalise to sum 1.
        w = pd.Series({"A": 0.60, "B": 0.38, "C": 0.02})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert out["C"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-9
        assert abs(out["A"] - 0.60 / 0.98) < 1e-9

    def test_no_drop_when_all_above_floor(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert (out == pd.Series({"A": 0.5, "B": 0.5})).all()

    def test_zero_weight_names_stay_zero(self):
        w = pd.Series({"A": 0.7, "B": 0.3, "C": 0.0})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert out["C"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-9

    def test_stops_at_two_survivors(self):
        # Even if both remaining names are below the floor, the len<=2 guard stops
        # the loop so we never empty the book.
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.9)
        assert (out.abs() > 0).sum() == 2
        assert abs(out.sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestApplyConsensusFloor -v`
Expected: FAIL with `ImportError: cannot import name 'apply_consensus_floor'`.

- [ ] **Step 3: Implement `apply_consensus_floor`**

Add to `experiments/measure_allocation_stability.py` (after `allocate_msr`, before `portfolio_metrics`):

```python
def apply_consensus_floor(weights, min_weight):
    """Enforce the min-weight floor on an averaged consensus, without re-optimising.

    Mirrors pipeline/04_allocate.py's batch-elimination *rule* (drop names whose
    cumulative weight, sorted ascending, is below min_weight) but renormalises the
    survivors to sum to 1 instead of re-running msr_tuned -- the consensus is an
    average of weight vectors, not an msr solution, so there is no single mu to
    optimise against. Iterates until every survivor passes; the len<=2 guard keeps
    the book from emptying. Returns a Series over the original index (dropped = 0.0).
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestApplyConsensusFloor -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add consensus min-weight floor for Michaud resampling"
```

---

### Task 2: `resampled_allocate` — per-draw raw msr + average-then-floor consensus

Optimise each per-run μ with raw `msr_tuned` (no elimination), average the per-draw weight vectors, then floor once via `apply_consensus_floor`. Also emit the per-name diagnostic (cross-draw selection frequency + mean raw weight) — the conviction-gradient view. An `eliminate_per_draw=True` toggle swaps raw `msr_tuned` for the full `allocate_msr` loop (the optional comparison arm).

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import resampled_allocate


@pytest.fixture
def five_asset_cov():
    names = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.diag([0.04] * 5), index=names, columns=names)


def _mu(values):
    return pd.Series(dict(zip(["A", "B", "C", "D", "E"], values)))


class TestResampledAllocate:
    def test_consensus_sums_to_one_and_covers_all_names(self, five_asset_cov):
        # Three draws, each favouring a different leader -> diffuse but valid consensus.
        mus = [
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.20, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.02, 0.20, 0.02, 0.02]),
        ]
        consensus, diag = resampled_allocate(mus, five_asset_cov, CFG)
        assert abs(consensus.sum() - 1.0) < 1e-6
        assert set(consensus.index) == {"A", "B", "C", "D", "E"}

    def test_consensus_respects_max_weight_implicitly(self, five_asset_cov):
        # Averaging corner solutions cannot exceed the per-draw max_weight.
        mus = [_mu([0.20, 0.02, 0.02, 0.02, 0.02])] * 3
        consensus, _ = resampled_allocate(mus, five_asset_cov, CFG)
        assert (consensus <= CFG["max_weight"] + 1e-6).all()

    def test_diagnostic_reports_freq_and_mean_weight(self, five_asset_cov):
        # A leads every draw -> freq 1.0 and the largest mean raw weight.
        mus = [
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
        ]
        _, diag = resampled_allocate(mus, five_asset_cov, CFG)
        assert set(diag.columns) == {"freq", "mean_raw_weight"}
        assert abs(diag.loc["A", "freq"] - 1.0) < 1e-9
        assert diag["mean_raw_weight"].idxmax() == "A"

    def test_consensus_smooths_rotating_leaders(self, five_asset_cov):
        # No single name dominates when the leader rotates: the top consensus weight
        # is well below what a single-draw corner solution would give that name.
        mus = [
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.20, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.02, 0.20, 0.02, 0.02]),
        ]
        single, _ = resampled_allocate([mus[0]], five_asset_cov, CFG)
        consensus, _ = resampled_allocate(mus, five_asset_cov, CFG)
        assert consensus.max() < single.max()

    def test_eliminate_per_draw_toggle_runs(self, five_asset_cov):
        # The comparison arm uses the full allocate_msr loop per draw; still valid.
        mus = [_mu([0.20, 0.02, 0.02, 0.02, 0.02])] * 3
        consensus, _ = resampled_allocate(mus, five_asset_cov, CFG, eliminate_per_draw=True)
        assert abs(consensus.sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestResampledAllocate -v`
Expected: FAIL with `ImportError: cannot import name 'resampled_allocate'`.

- [ ] **Step 3: Implement `resampled_allocate`**

Add to `experiments/measure_allocation_stability.py` (after `apply_consensus_floor`):

```python
def resampled_allocate(per_run_mu, covmat, cfg, eliminate_per_draw=False, eps=1e-9):
    """Michaud resampled-efficiency consensus over a list of per-run mu vectors.

    Each draw is optimised by raw msr_tuned (no elimination) so the continuous
    conviction signal survives into the average; with eliminate_per_draw=True the
    full allocate_msr loop is used per draw instead (the optional comparison arm).
    The per-draw weight vectors are averaged and floored once via apply_consensus_floor
    -- average-then-threshold, never threshold-then-average.

    All mu vectors must share the same index (the selected names). covmat must cover
    those names. Returns (consensus: Series over those names with dropped names at 0.0,
    diagnostic: DataFrame indexed by name with columns 'freq' = fraction of draws that
    gave the name a nonzero raw weight and 'mean_raw_weight' = mean raw weight).
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    ppy = cfg["periods_per_year"]
    min_w = cfg["min_weight"]

    rows = []
    for mu_i in per_run_mu:
        cov_i = covmat.loc[mu_i.index, mu_i.index]
        if eliminate_per_draw:
            w_i = allocate_msr(mu_i, cov_i, cfg)
        else:
            arr = rk.msr_tuned(
                riskfree_rate=rf, returns=mu_i, covmat=cov_i,
                max_weight=max_w, periods_per_year=ppy, debug=False,
            )
            w_i = pd.Series(arr, index=mu_i.index)
        rows.append(w_i)

    raw = pd.DataFrame(rows).reset_index(drop=True)
    consensus = apply_consensus_floor(raw.mean(axis=0), min_w)
    diagnostic = pd.DataFrame({
        "freq": (raw.abs() > eps).mean(axis=0),
        "mean_raw_weight": raw.mean(axis=0),
    })
    return consensus, diagnostic
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestResampledAllocate -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add resampled_allocate Michaud consensus over per-run forecasts"
```

---

### Task 3: `overlap_stats` — the intuitive "shared-of-held" metric

Reports the mean pairwise count of shared held names, the mean held-name count per portfolio, and their ratio — the "7 of ~9 names shared" read that makes "consistent vs still-random" legible (Jaccard alone does not).

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import overlap_stats


class TestOverlapStats:
    def test_identical_books(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5, "C": 0.0},
                           {"A": 0.4, "B": 0.6, "C": 0.0}])
        s = overlap_stats(df)
        assert s["shared"] == 2.0
        assert s["held"] == 2.0
        assert abs(s["fraction"] - 1.0) < 1e-12

    def test_partial_overlap(self):
        # Books {A,B} and {B,C}: shared = 1, held per row = 2.
        df = pd.DataFrame([{"A": 0.5, "B": 0.5, "C": 0.0},
                           {"A": 0.0, "B": 0.5, "C": 0.5}])
        s = overlap_stats(df)
        assert s["shared"] == 1.0
        assert s["held"] == 2.0
        assert abs(s["fraction"] - 0.5) < 1e-12

    def test_disjoint_books(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        s = overlap_stats(df)
        assert s["shared"] == 0.0

    def test_single_row_shared_is_none(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}])
        s = overlap_stats(df)
        assert s["shared"] is None
        assert s["held"] == 2.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestOverlapStats -v`
Expected: FAIL with `ImportError: cannot import name 'overlap_stats'`.

- [ ] **Step 3: Implement `overlap_stats`**

Add to `experiments/measure_allocation_stability.py` (after `mean_jaccard`):

```python
def overlap_stats(weights_df, eps=1e-9):
    """Mean pairwise count of shared held names, mean held count, and their ratio.

    The intuitive 'shared of held' read (e.g. 7 of ~9). 'shared' and 'fraction' are
    None when there are fewer than 2 portfolios to compare.
    """
    held = weights_df.abs() > eps
    mean_held = float(held.sum(axis=1).mean())
    if len(weights_df) < 2:
        return {"shared": None, "held": mean_held, "fraction": None}
    rows = [set(held.columns[held.iloc[i].values]) for i in range(len(held))]
    shared = [
        len(rows[i] & rows[j])
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    ]
    mean_shared = float(np.mean(shared))
    fraction = mean_shared / mean_held if mean_held > 0 else float("nan")
    return {"shared": mean_shared, "held": mean_held, "fraction": fraction}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestOverlapStats -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add overlap_stats shared-of-held composition metric"
```

---

### Task 4: `run_paired_experiment` — both arms from the same draws

Per iteration: train N runs once (via an injectable `runs_fn` returning a list of per-run period-forecast DataFrames); derive the **current** arm (average runs → μ → `allocate_msr`) and the **Michaud** arm (`resampled_allocate` over per-run μ) from those same runs. Value metrics for **both** arms are scored against the **same** averaged μ (the best single return estimate), so the value comparison is apples-to-apples. Σ (Ledoit-Wolf) and the selected set are computed once.

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import run_paired_experiment


class TestRunPairedExperiment:
    def _inputs(self):
        np.random.seed(2)
        n = 40
        idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
        cols = ["A", "B", "C", "D", "E"]
        rets = pd.DataFrame(np.random.normal(0.002, 0.02, (n, 5)), index=idx, columns=cols)
        prices = (1 + rets).cumprod() * 100
        cfg = {
            "rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05,
            "periods_per_year": 12,
        }
        return prices, rets, cfg

    def _stubs(self):
        # runs_fn returns a LIST of N per-run forecast DataFrames (periods x stocks).
        # Each run favours a different leader and the leadership rotates per iteration,
        # so the current (averaged) and Michaud (consensus) arms genuinely differ.
        state = {"k": 0}
        cols = ["A", "B", "C", "D", "E"]

        def runs_fn(rets, cfg, n_runs=None, verbose=True):
            state["k"] += 1
            n_runs = n_runs or 3
            runs = []
            for r in range(n_runs):
                vals = np.full(5, 0.004)
                leader = (state["k"] + r) % 5
                vals[leader] = 0.02
                runs.append(pd.DataFrame(np.vstack([vals, vals]), columns=cols))
            return runs

        def period_mu_fn(preds):
            return preds.mean(axis=0)

        def select_fn(prices, rets, cfg):
            return ["A", "B", "C", "D", "E"]

        def seed_fn(seed):
            pass

        return runs_fn, period_mu_fn, select_fn, seed_fn

    def _run(self, iterations=4, transformer_runs=3):
        prices, rets, cfg = self._inputs()
        runs_fn, period_mu_fn, select_fn, seed_fn = self._stubs()
        return run_paired_experiment(
            prices, rets, cfg, iterations=iterations, transformer_runs=transformer_runs,
            seed=0, runs_fn=runs_fn, period_mu_fn=period_mu_fn,
            select_fn=select_fn, seed_fn=seed_fn,
        )

    def test_returns_both_arms_and_selected(self):
        result = self._run()
        assert set(result.keys()) == {"current", "michaud", "selected"}
        assert result["selected"] == ["A", "B", "C", "D", "E"]

    def test_current_arm_shapes(self):
        result = self._run(iterations=4)
        assert result["current"]["weights"].shape == (4, 5)
        assert len(result["current"]["metrics"]) == 4
        assert set(result["current"]["metrics"].columns) == {"ret", "vol", "sharpe"}

    def test_michaud_arm_shapes_and_diagnostic(self):
        result = self._run(iterations=4)
        assert result["michaud"]["weights"].shape == (4, 5)
        assert len(result["michaud"]["metrics"]) == 4
        assert set(result["michaud"]["diagnostic"].columns) == {"freq", "mean_raw_weight"}

    def test_both_arms_weights_sum_to_one(self):
        result = self._run()
        for arm in ("current", "michaud"):
            sums = result[arm]["weights"].sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_arms_differ_with_rotating_leaders(self):
        # The whole point: with rotating per-run leaders the consensus is not identical
        # to the averaged-mu allocation, so the two weight matrices must differ.
        result = self._run()
        assert not np.allclose(
            result["current"]["weights"].values, result["michaud"]["weights"].values
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestRunPairedExperiment -v`
Expected: FAIL with `ImportError: cannot import name 'run_paired_experiment'`.

- [ ] **Step 3: Implement `run_paired_experiment`**

Add to `experiments/measure_allocation_stability.py` (after `run_experiment`). It reuses `allocate_msr`, `resampled_allocate`, `portfolio_metrics`:

```python
def run_paired_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                          runs_fn=None, period_mu_fn=None, select_fn=None,
                          seed_fn=None, eliminate_per_draw=False):
    """Run `iterations` passes, deriving the current and Michaud arms from the SAME draws.

    runs_fn(rets, cfg, n_runs, verbose) -> list of N per-run period-forecast DataFrames
    (rows = periods, columns = stocks), already winsorised. The current arm averages the
    runs into one mu and allocates once (today's pipeline); the Michaud arm optimises each
    run's mu and averages the weights. Both arms' value metrics are scored against the same
    averaged mu. The four *_fn arguments are dependency-injection seams (default to the real
    implementations) so the loop runs without importing torch.

    Returns {'current': {weights, metrics}, 'michaud': {weights, metrics, diagnostic},
    'selected': [...]} where each weights frame is iterations x selected and metrics is
    iterations x [ret, vol, sharpe]. 'diagnostic' is the per-name mean over iterations of
    the cross-draw selection frequency and mean raw weight (the conviction-gradient view).
    """
    if runs_fn is None:
        runs_fn = train_runs_as_preds
    if period_mu_fn is None:
        from transformer_model import weighted_mean_return
        period_mu_fn = weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    rf = cfg["rf_period"]
    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    cur_w, cur_m = [], []
    mic_w, mic_m, mic_diag = [], [], []
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        runs = runs_fn(rets, cfg, n_runs=transformer_runs, verbose=False)

        # Current arm: average runs -> one mu -> allocate once (today's pipeline).
        preds_avg = sum(r.values for r in runs) / len(runs)
        preds_avg = pd.DataFrame(preds_avg, columns=runs[0].columns)
        mu_avg = period_mu_fn(preds_avg).loc[selected]
        w_cur = allocate_msr(mu_avg, cov_sel, cfg)
        s_cur = w_cur[w_cur.abs() > 1e-9]
        cur_w.append(w_cur)
        cur_m.append(portfolio_metrics(s_cur, mu_avg, cov_sel, rf))

        # Michaud arm: per-run mu -> resampled consensus. Score against the SAME mu_avg.
        per_run_mu = [period_mu_fn(r).loc[selected] for r in runs]
        w_mic, diag = resampled_allocate(
            per_run_mu, cov_sel, cfg, eliminate_per_draw=eliminate_per_draw
        )
        s_mic = w_mic[w_mic.abs() > 1e-9]
        mic_w.append(w_mic)
        mic_m.append(portfolio_metrics(s_mic, mu_avg, cov_sel, rf))
        mic_diag.append(diag)

    diagnostic = sum(d for d in mic_diag) / len(mic_diag)
    return {
        "current": {
            "weights": pd.DataFrame(cur_w).reset_index(drop=True),
            "metrics": pd.DataFrame(cur_m),
        },
        "michaud": {
            "weights": pd.DataFrame(mic_w).reset_index(drop=True),
            "metrics": pd.DataFrame(mic_m),
            "diagnostic": diagnostic,
        },
        "selected": selected,
    }
```

Note: `train_runs_as_preds` is the default `runs_fn`, implemented in Task 7; the tests inject a stub so this task does not depend on it at test time.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestRunPairedExperiment -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add paired current-vs-Michaud experiment loop from shared draws"
```

---

### Task 5: `format_paired_summary` — side-by-side report

Reuses `selection_frequency`, `mean_turnover`, `mean_jaccard`, `metric_dispersion`, `overlap_stats`, and `_fmt` to print current and Michaud blocks plus the Michaud conviction-gradient table.

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import format_paired_summary


def _sample_paired_result():
    cur_w = pd.DataFrame([{"A": 0.6, "B": 0.4, "C": 0.0},
                          {"A": 0.5, "B": 0.5, "C": 0.0},
                          {"A": 0.4, "B": 0.3, "C": 0.3}])
    mic_w = pd.DataFrame([{"A": 0.5, "B": 0.3, "C": 0.2},
                          {"A": 0.45, "B": 0.35, "C": 0.2},
                          {"A": 0.4, "B": 0.35, "C": 0.25}])
    metrics = pd.DataFrame([{"ret": 0.16, "vol": 0.10, "sharpe": 1.6},
                            {"ret": 0.17, "vol": 0.10, "sharpe": 1.7},
                            {"ret": 0.15, "vol": 0.11, "sharpe": 1.4}])
    diag = pd.DataFrame({"freq": {"A": 1.0, "B": 0.9, "C": 0.5},
                         "mean_raw_weight": {"A": 0.45, "B": 0.33, "C": 0.22}})
    return {
        "current": {"weights": cur_w, "metrics": metrics},
        "michaud": {"weights": mic_w, "metrics": metrics, "diagnostic": diag},
        "selected": ["A", "B", "C"],
    }


class TestFormatPairedSummary:
    def test_contains_both_arms_and_overlap(self):
        text = format_paired_summary(_sample_paired_result())
        assert "CURRENT" in text
        assert "MICHAUD" in text
        assert "Overlap" in text

    def test_contains_conviction_gradient_table(self):
        text = format_paired_summary(_sample_paired_result())
        assert "Conviction" in text
        assert "freq" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestFormatPairedSummary -v`
Expected: FAIL with `ImportError: cannot import name 'format_paired_summary'`.

- [ ] **Step 3: Implement `format_paired_summary`**

Add to `experiments/measure_allocation_stability.py` (after `format_summary`):

```python
def _arm_block(title, weights_df, metrics_df):
    """One arm's composition + value block for the paired summary."""
    freq = selection_frequency(weights_df)
    turnover = mean_turnover(weights_df)
    jacc = mean_jaccard(weights_df)
    ov = overlap_stats(weights_df)
    disp = metric_dispersion(metrics_df)

    lines = [f"== {title} =="]
    lines.append(f"Mean pairwise turnover: {_fmt(turnover)}")
    lines.append(f"Mean pairwise Jaccard:  {_fmt(jacc)}")
    lines.append(
        f"Overlap: {_fmt(ov['shared'])} of {_fmt(ov['held'])} names shared "
        f"(fraction {_fmt(ov['fraction'])})"
    )
    lines.append("Selection frequency (held fraction across iterations):")
    for name, v in freq.sort_values(ascending=False).items():
        lines.append(f"  {name:<14} {v:6.1%}")
    for col in ["ret", "vol", "sharpe"]:
        d = disp[col]
        lines.append(
            f"  {col:<7} mean {d['mean']:.4f}  std {d['std']:.4f}  CoV {_fmt(d['cov'])}"
        )
    return lines


def format_paired_summary(result):
    """Side-by-side current-vs-Michaud stability report from a run_paired_experiment result."""
    cur, mic = result["current"], result["michaud"]
    n_iter = len(cur["weights"])
    n_names = cur["weights"].shape[1]

    lines = [f"Iterations: {n_iter} | Selected names: {n_names}", ""]
    lines += _arm_block("CURRENT (average mu -> one allocation)", cur["weights"], cur["metrics"])
    lines.append("")
    lines += _arm_block("MICHAUD (resampled consensus)", mic["weights"], mic["metrics"])
    lines.append("")
    lines.append("== MICHAUD conviction gradient (mean across iterations) ==")
    diag = mic["diagnostic"].sort_values("mean_raw_weight", ascending=False)
    lines.append(f"  {'name':<14} {'freq':>6} {'mean_raw_weight':>16}")
    for name, row in diag.iterrows():
        lines.append(f"  {name:<14} {row['freq']:6.1%} {row['mean_raw_weight']:16.4f}")
    lines.append("")
    lines.append(
        "Note: both arms use the SAME N draws per iteration; value metrics for both are "
        "scored against the averaged mu. Fewer transformer-runs => noisier mu => more "
        "instability (conservative upper bound vs production 100-run)."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestFormatPairedSummary -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add side-by-side paired summary with conviction-gradient table"
```

---

### Task 6: `write_paired_outputs` — per-arm CSVs + summary

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import write_paired_outputs


class TestWritePairedOutputs:
    def test_writes_all_files(self, tmp_path):
        paths = write_paired_outputs(_sample_paired_result(), str(tmp_path))
        for key in ("current_weights", "current_metrics", "michaud_weights",
                    "michaud_metrics", "michaud_diagnostic", "summary"):
            assert os.path.exists(paths[key])

    def test_michaud_weights_roundtrip(self, tmp_path):
        result = _sample_paired_result()
        paths = write_paired_outputs(result, str(tmp_path))
        reloaded = pd.read_csv(paths["michaud_weights"], index_col=0)
        assert reloaded.shape == result["michaud"]["weights"].shape
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestWritePairedOutputs -v`
Expected: FAIL with `ImportError: cannot import name 'write_paired_outputs'`.

- [ ] **Step 3: Implement `write_paired_outputs`**

Add to `experiments/measure_allocation_stability.py` (after `write_outputs`):

```python
def write_paired_outputs(result, outdir):
    """Write per-arm weights/metrics CSVs, the Michaud diagnostic, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "current_weights": os.path.join(outdir, "paired_current_weights.csv"),
        "current_metrics": os.path.join(outdir, "paired_current_metrics.csv"),
        "michaud_weights": os.path.join(outdir, "paired_michaud_weights.csv"),
        "michaud_metrics": os.path.join(outdir, "paired_michaud_metrics.csv"),
        "michaud_diagnostic": os.path.join(outdir, "paired_michaud_diagnostic.csv"),
        "summary": os.path.join(outdir, "paired_summary.txt"),
    }
    result["current"]["weights"].to_csv(paths["current_weights"], index_label="iteration")
    result["current"]["metrics"].to_csv(paths["current_metrics"], index_label="iteration")
    result["michaud"]["weights"].to_csv(paths["michaud_weights"], index_label="iteration")
    result["michaud"]["metrics"].to_csv(paths["michaud_metrics"], index_label="iteration")
    result["michaud"]["diagnostic"].to_csv(paths["michaud_diagnostic"], index_label="name")
    with open(paths["summary"], "w") as f:
        f.write(format_paired_summary(result))
    return paths
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestWritePairedOutputs -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add per-arm output writer for the paired comparison"
```

---

### Task 7: Default `runs_fn` + `--mode paired` CLI + n=100 docs

Wire the real per-run forecaster as the default `runs_fn`, add a `paired` CLI mode, and document the n=100 reproducibility configuration. `train_runs_as_preds` and the CLI use torch / the real pipeline, so they are exercised by a real run rather than unit tests; one lightweight arg-parsing test guards the CLI wiring.

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import build_arg_parser


class TestCliWiring:
    def test_mode_defaults_to_measure(self):
        args = build_arg_parser().parse_args([])
        assert args.mode == "measure"

    def test_paired_mode_and_flag_parse(self):
        args = build_arg_parser().parse_args(
            ["--mode", "paired", "--iterations", "2", "--transformer-runs", "100",
             "--eliminate-per-draw"]
        )
        assert args.mode == "paired"
        assert args.iterations == 2
        assert args.transformer_runs == 100
        assert args.eliminate_per_draw is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestCliWiring -v`
Expected: FAIL with `ImportError: cannot import name 'build_arg_parser'`.

- [ ] **Step 3: Implement the default runs_fn, the arg parser, and the paired CLI branch**

Add `train_runs_as_preds` to `experiments/measure_allocation_stability.py` (after `select_stocks`):

```python
def train_runs_as_preds(rets, cfg, n_runs=None, verbose=False):
    """Default runs_fn for the paired experiment: real per-run forecasts, winsorised.

    Wraps transformer_model.train_runs (the un-averaged per-run forecasts) and
    winsorises each run to the history, returning a list of (periods x stocks)
    DataFrames -- one per Transformer run. Torch is imported lazily here so the rest
    of the module (and its tests) stay torch-free.
    """
    from transformer_model import train_runs, winsorize_to_history
    runs = train_runs(rets, cfg, n_runs=n_runs, verbose=verbose)
    out = []
    for r in range(runs.shape[0]):
        preds = pd.DataFrame(runs[r], columns=rets.columns)
        out.append(winsorize_to_history(preds, rets))
    return out
```

Refactor the bottom of the file: extract `build_arg_parser`, and dispatch on `--mode`. Replace the existing `main()` and the argument-parsing block inside it with:

```python
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["measure", "paired"], default="measure",
                        help="'measure' = single-arm Phase-1 instrument; "
                             "'paired' = current-vs-Michaud comparison (Phase 3)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of predict->allocate runs (default 30). For the "
                             "n=100 reproducibility check use --mode paired --iterations 2 "
                             "--transformer-runs 100.")
    parser.add_argument("--transformer-runs", type=int, default=10,
                        help="n_runs passed to the forecaster per iteration (default 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; iteration i uses seed+i (default 0)")
    parser.add_argument("--eliminate-per-draw", action="store_true",
                        help="(paired mode) use per-draw elimination instead of the "
                             "deferred consensus floor -- the comparison arm")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    print(f"Universe: {rets.shape[1]} stocks | mode={args.mode} | "
          f"iterations={args.iterations} | transformer-runs={args.transformer_runs} | "
          f"seed={args.seed}")

    if args.mode == "paired":
        result = run_paired_experiment(
            prices, rets, cfg,
            iterations=args.iterations, transformer_runs=args.transformer_runs,
            seed=args.seed, eliminate_per_draw=args.eliminate_per_draw,
        )
        paths = write_paired_outputs(result, args.outdir)
        print()
        print(format_paired_summary(result))
    else:
        result = run_experiment(
            prices, rets, cfg,
            iterations=args.iterations, transformer_runs=args.transformer_runs,
            seed=args.seed,
        )
        paths = write_outputs(result, args.outdir)
        print()
        print(format_summary(result))

    print("\nSaved:")
    for p in paths.values():
        print(f"       {p}")
```

(The old `main()` body that built its own `argparse.ArgumentParser` is fully replaced by `build_arg_parser` + the dispatch above. Leave the `if __name__ == "__main__": main()` line intact.)

Update the module docstring's "Run:" line to document the paired mode and the n=100 check:

```python
# In the module docstring near the existing "Run:" example, add:
#   Paired Michaud comparison (Phase 3):
#     python experiments/measure_allocation_stability.py --mode paired --iterations 30 --transformer-runs 10
#   Production-budget reproducibility check (both arms):
#     python experiments/measure_allocation_stability.py --mode paired --iterations 2 --transformer-runs 100
```

- [ ] **Step 4: Run the tests to verify they pass, then the whole file**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestCliWiring -v`
Expected: PASS (2 passed).

Run the full suite to confirm nothing regressed: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py -v`
Expected: PASS (all existing + new tests green).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add paired CLI mode, default per-run forecaster, and n=100 docs"
```

---

## Post-implementation validation (manual, not a code task)

After Task 7, run the harness for real to gather the Phase-3 evidence (these are slow — torch on the full universe):

1. **Primary paired comparison:** `--mode paired --iterations 30 --transformer-runs 10 --seed 0`. Read `paired_summary.txt`: does Michaud cut turnover and raise Jaccard / overlap vs current, while the conviction core stays high-frequency and Sharpe / return CoV do not degrade? (Decision rule A in the spec.)
2. **n=100 reproducibility:** `--mode paired --iterations 2 --transformer-runs 100 --seed 0`. Read the overlap line for both arms ("X of ~9 shared").
3. **If the Michaud diagnostic shows extreme-and-rare draws** (a few names with low `freq` but large `mean_raw_weight` swings), reach for the lead contingency lever from the spec (robust aggregation: trimmed mean / median in `resampled_allocate`) — that is a follow-up change, not part of this plan.

Results land in `experiments/results/` (gitignored); record the read-off in the project memory. Productionising the winning recipe into `pipeline/04_allocate.py` is a separate, later phase.

---

## Self-Review notes

- **Spec coverage:** resampled allocation + deferred floor (Tasks 1–2), empirical-ensemble draws via per-run μ (Task 4), average-then-floor rule A (Tasks 1–2), paired same-draws loop (Task 4), overlap metric (Task 3), value guardrail scored against shared μ (Task 4), conviction-gradient diagnostic (Tasks 2,4,5), n=100 check + CLI (Task 7), `eliminate_per_draw` comparison arm (Tasks 2,4,7), contingency levers noted as out-of-scope follow-ups (post-impl section). No `pipeline/`/`src/` edits — matches scope.
- **Type consistency:** `resampled_allocate` returns `(Series, DataFrame[freq, mean_raw_weight])`; consumed consistently in Tasks 4–6. `run_paired_experiment` returns `{current:{weights,metrics}, michaud:{weights,metrics,diagnostic}, selected}`; consumed consistently in Tasks 5–6 and `main`. `overlap_stats` returns `{shared,held,fraction}`; consumed in `_arm_block`. `build_arg_parser` used in Task 7 test and `main`.
- **No placeholders:** every code step is complete and runnable.
