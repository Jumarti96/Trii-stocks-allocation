# Phase 3b — Parametric Michaud (Mechanism B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parametric Michaud draw mechanism — sample μ from `N(μ̄, s²·Σ/T)` and feed the existing resampled-consensus machinery — so the Michaud arm integrates over a principled, controllable μ-uncertainty instead of one noise-dominated draw per transformer run.

**Architecture:** One new pure function (`sample_mu_draws`) generates the parametric draws; a `draw_mechanism` toggle on the existing `run_paired_experiment` swaps the Michaud arm's per-draw μ source from "one per transformer run" (empirical, Phase 3) to "K Monte-Carlo samples" (parametric, Phase 3b). Everything downstream — `resampled_allocate`, the consensus floor, paired scoring against the shared μ̄, metrics, diagnostic, outputs — is reused verbatim. New CLI flags expose it. All additive, local-only; no `pipeline/` or `src/` edits.

**Tech Stack:** Python, numpy (`default_rng`, Cholesky), pandas, scipy (via `risk_kit.msr_tuned`), pytest. Tests are torch-free (the new function takes an explicit numpy `Generator`; the wiring tests use the existing dependency-injection stubs).

**Conventions:**
- Run tests with the project interpreter: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`.
- Commit messages: imperative, capitalised, **no AI attribution** (no `Co-Authored-By`, no "Generated with"); match repo style.
- All new code goes in `experiments/measure_allocation_stability.py`; all new tests in `tests/test_measure_allocation_stability.py`. Existing functions and their tests stay **unchanged** — the parametric path is added alongside, default behaviour is `empirical`.
- Work on branch `enh-allocation-phase3b-parametric-michaud` (already created; the design spec is committed there).

---

## File Structure

- **Modify:** `experiments/measure_allocation_stability.py`
  - Add `sample_mu_draws(...)` after `resampled_allocate` (they are conceptually paired).
  - Add `draw_mechanism` / `spread` / `n_draws` params to `run_paired_experiment` and switch the Michaud arm's draw source on them.
  - Add `--draw-mechanism` / `--spread` / `--mc-draws` to `build_arg_parser` and thread them through `main`.
- **Modify:** `tests/test_measure_allocation_stability.py` — add `TestSampleMuDraws`, `TestParametricDrawMechanism`, and two CLI assertions. Existing tests untouched.
- **Reference (do not modify):** `resampled_allocate`, `allocate_msr`, `portfolio_metrics` (consumed unchanged); `src/risk_kit.py` (`msr_tuned`).

**Relevant existing signatures (do not change):**
- `resampled_allocate(per_run_mu, covmat, cfg, eliminate_per_draw=False, eps=1e-9)` → `(consensus Series, diagnostic DataFrame[freq, mean_raw_weight])`.
- `run_paired_experiment(prices, rets, cfg, iterations, transformer_runs, seed, runs_fn=None, period_mu_fn=None, select_fn=None, seed_fn=None, eliminate_per_draw=False)`.
- `build_arg_parser()` returns an `argparse.ArgumentParser` with `--mode/--iterations/--transformer-runs/--seed/--eliminate-per-draw/--outdir`.

---

### Task 1: `sample_mu_draws` — parametric Michaud draw generator

The whole novelty: draw `n_draws` μ vectors from `N(μ̄, spread²·Σ/T)` via one Cholesky factor of the scaled covariance. Pure and seedable (explicit numpy `Generator`), torch-free. `spread=0` short-circuits to exact copies of μ̄ (the scaled covariance is then the zero matrix, which has no Cholesky factor).

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_measure_allocation_stability.py`:

```python
from experiments.measure_allocation_stability import sample_mu_draws


def _cov3():
    names = ["A", "B", "C"]
    return pd.DataFrame(
        [[0.04, 0.01, 0.00],
         [0.01, 0.04, 0.01],
         [0.00, 0.01, 0.04]],
        index=names, columns=names,
    )


def _mu_bar3():
    return pd.Series({"A": 0.01, "B": 0.02, "C": 0.015})


class TestSampleMuDraws:
    def test_returns_n_draws_series_over_index(self):
        rng = np.random.default_rng(0)
        draws = sample_mu_draws(_mu_bar3(), _cov3(), n_periods=10,
                                n_draws=5, spread=1.0, rng=rng)
        assert len(draws) == 5
        for d in draws:
            assert list(d.index) == ["A", "B", "C"]

    def test_spread_zero_returns_exact_copies(self):
        rng = np.random.default_rng(0)
        mu = _mu_bar3()
        draws = sample_mu_draws(mu, _cov3(), n_periods=10,
                                n_draws=4, spread=0.0, rng=rng)
        for d in draws:
            assert (d == mu).all()

    def test_large_k_mean_approx_mu_bar(self):
        rng = np.random.default_rng(1)
        mu = _mu_bar3()
        draws = sample_mu_draws(mu, _cov3(), n_periods=10,
                                n_draws=50000, spread=1.0, rng=rng)
        mat = pd.DataFrame(draws)
        assert np.allclose(mat.mean(axis=0).values, mu.values, atol=2e-3)

    def test_large_k_cov_approx_scaled_sigma(self):
        rng = np.random.default_rng(2)
        cov = _cov3()
        draws = sample_mu_draws(_mu_bar3(), cov, n_periods=10,
                                n_draws=50000, spread=1.0, rng=rng)
        mat = pd.DataFrame(draws).values
        emp = np.cov(mat, rowvar=False)
        target = cov.values * (1.0 ** 2 / 10)
        assert np.allclose(emp, target, atol=3e-4)

    def test_larger_spread_more_dispersion(self):
        mu, cov = _mu_bar3(), _cov3()
        d1 = pd.DataFrame(sample_mu_draws(mu, cov, 10, 20000, 1.0, np.random.default_rng(3)))
        d2 = pd.DataFrame(sample_mu_draws(mu, cov, 10, 20000, 2.0, np.random.default_rng(3)))
        assert (d2.std(axis=0).values > d1.std(axis=0).values).all()

    def test_seeded_rng_reproducible(self):
        a = sample_mu_draws(_mu_bar3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        b = sample_mu_draws(_mu_bar3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        assert np.allclose(pd.DataFrame(a).values, pd.DataFrame(b).values)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestSampleMuDraws -v`
Expected: FAIL with `ImportError: cannot import name 'sample_mu_draws'`.

- [ ] **Step 3: Implement `sample_mu_draws`**

Add to `experiments/measure_allocation_stability.py` immediately after `resampled_allocate` (before `portfolio_metrics`):

```python
def sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng):
    """Draw n_draws mu vectors from the canonical Michaud law N(mu_bar, spread**2 * Sigma / T).

    The parametric (mechanism B) alternative to using one transformer run per draw: perturb a
    well-estimated centre mu_bar by the classical standard error of an estimated mean, scaled by
    the tunable knob `spread` (s). Reuses the optimiser's own Ledoit-Wolf Sigma -- nothing extra
    to estimate.

    mu_bar:    Series over the selected names (the averaged per-period forecast).
    covmat:    DataFrame Ledoit-Wolf covariance over those same names (the optimiser's Sigma).
    n_periods: T, the number of return periods backing Sigma (sets the sampling-error scale).
    n_draws:   K, how many mu vectors to sample.
    spread:    the knob s; spread=0 returns n_draws exact copies of mu_bar.
    rng:       a numpy Generator (explicit for reproducibility and torch-free testing).

    Sampled via one Cholesky factor of the scaled covariance. Returns a list of Series over
    mu_bar.index, preserving name order.
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

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestSampleMuDraws -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add sample_mu_draws parametric Michaud draw generator"
```

---

### Task 2: `draw_mechanism` toggle on `run_paired_experiment`

Add `draw_mechanism` (`"empirical"` default | `"parametric"`), `spread`, and `n_draws` params. When `parametric`, the Michaud arm's per-draw μ list comes from `sample_mu_draws(μ̄, Σ_sel, T=len(rets), n_draws, spread, rng)` using a per-iteration `default_rng(seed + i)`; otherwise it stays the Phase-3 per-run list. The current arm, scoring against the shared `mu_avg`, metrics, diagnostic, and outputs are all unchanged.

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_measure_allocation_stability.py`:

```python
class TestParametricDrawMechanism:
    def _setup(self):
        np.random.seed(3)
        n = 40
        idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
        cols = ["A", "B", "C", "D", "E"]
        vols = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        rets = pd.DataFrame(np.random.normal(0.0, 1.0, (n, 5)) * vols + 0.002,
                            index=idx, columns=cols)
        prices = (1 + rets).cumprod() * 100
        cfg = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6,
               "min_weight": 0.05, "periods_per_year": 12}

        def runs_fn(rets, cfg, n_runs=None, verbose=True):
            n_runs = n_runs or 3
            return [pd.DataFrame(np.full((2, 5), 0.005), columns=cols)
                    for _ in range(n_runs)]

        def period_mu_fn(preds):
            return preds.mean(axis=0)

        def select_fn(prices, rets, cfg):
            return cols

        def seed_fn(seed):
            pass

        return prices, rets, cfg, runs_fn, period_mu_fn, select_fn, seed_fn

    def _run(self, **kw):
        prices, rets, cfg, runs_fn, period_mu_fn, select_fn, seed_fn = self._setup()
        return run_paired_experiment(
            prices, rets, cfg,
            iterations=kw.pop("iterations", 3),
            transformer_runs=kw.pop("transformer_runs", 3),
            seed=0, runs_fn=runs_fn, period_mu_fn=period_mu_fn,
            select_fn=select_fn, seed_fn=seed_fn, **kw,
        )

    def test_parametric_returns_same_structure(self):
        result = self._run(draw_mechanism="parametric", n_draws=200, spread=1.0)
        assert set(result.keys()) == {"current", "michaud", "selected"}
        assert result["michaud"]["weights"].shape == (3, 5)
        assert set(result["michaud"]["diagnostic"].columns) == {"freq", "mean_raw_weight"}

    def test_parametric_weights_sum_to_one(self):
        result = self._run(draw_mechanism="parametric", n_draws=200, spread=1.0)
        for arm in ("current", "michaud"):
            sums = result[arm]["weights"].sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_parametric_differs_from_current(self):
        result = self._run(draw_mechanism="parametric", n_draws=200, spread=1.0)
        assert not np.allclose(
            result["current"]["weights"].values, result["michaud"]["weights"].values
        )

    def test_empirical_is_the_default(self):
        # Constant runs => empirical draws are identical => consensus is a single corner.
        # The parametric arm (spread>0) perturbs mu, so the two arms must differ. This
        # confirms the default path is still the Phase-3 empirical one.
        default = self._run()
        empirical = self._run(draw_mechanism="empirical")
        assert np.allclose(
            default["michaud"]["weights"].values, empirical["michaud"]["weights"].values
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestParametricDrawMechanism -v`
Expected: FAIL with `TypeError: run_paired_experiment() got an unexpected keyword argument 'draw_mechanism'`.

- [ ] **Step 3: Update the `run_paired_experiment` signature**

In `experiments/measure_allocation_stability.py`, change the function definition from:

```python
def run_paired_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                          runs_fn=None, period_mu_fn=None, select_fn=None,
                          seed_fn=None, eliminate_per_draw=False):
```

to:

```python
def run_paired_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                          runs_fn=None, period_mu_fn=None, select_fn=None,
                          seed_fn=None, eliminate_per_draw=False,
                          draw_mechanism="empirical", spread=1.0, n_draws=1000):
```

- [ ] **Step 4: Switch the Michaud arm's draw source**

In the loop body of `run_paired_experiment`, replace this block:

```python
        # Michaud arm: per-run mu -> resampled consensus. Score against the SAME mu_avg.
        per_run_mu = [period_mu_fn(r).loc[selected] for r in runs]
        w_mic, diag = resampled_allocate(
            per_run_mu, cov_sel, cfg, eliminate_per_draw=eliminate_per_draw
        )
```

with:

```python
        # Michaud arm: draws -> resampled consensus. Score against the SAME mu_avg.
        # 'empirical' = one draw per transformer run (Phase 3); 'parametric' = K Monte-Carlo
        # draws from N(mu_avg, spread**2 * Sigma / T) (Phase 3b).
        if draw_mechanism == "parametric":
            rng = np.random.default_rng(seed + i)
            per_run_mu = sample_mu_draws(
                mu_avg, cov_sel, n_periods=len(rets),
                n_draws=n_draws, spread=spread, rng=rng,
            )
        else:
            per_run_mu = [period_mu_fn(r).loc[selected] for r in runs]
        w_mic, diag = resampled_allocate(
            per_run_mu, cov_sel, cfg, eliminate_per_draw=eliminate_per_draw
        )
```

(`mu_avg` is already computed just above in the current-arm block; `cov_sel`, `selected`, and the loop index `i` are already in scope.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestParametricDrawMechanism -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add parametric draw_mechanism toggle to run_paired_experiment"
```

---

### Task 3: CLI flags + `main` wiring

Expose the parametric path: `--draw-mechanism`, `--spread`, `--mc-draws` in `build_arg_parser`, threaded into the `run_paired_experiment` call in `main`. Defaults preserve existing behaviour.

**Files:**
- Modify: `experiments/measure_allocation_stability.py`
- Test: `tests/test_measure_allocation_stability.py`

- [ ] **Step 1: Write the failing tests**

Append to the existing `TestCliWiring` class in `tests/test_measure_allocation_stability.py` (it already imports `build_arg_parser`):

```python
    def test_draw_mechanism_defaults(self):
        args = build_arg_parser().parse_args([])
        assert args.draw_mechanism == "empirical"
        assert args.spread == 1.0
        assert args.mc_draws == 1000

    def test_parametric_flags_parse(self):
        args = build_arg_parser().parse_args(
            ["--mode", "paired", "--draw-mechanism", "parametric",
             "--spread", "2.0", "--mc-draws", "500"]
        )
        assert args.draw_mechanism == "parametric"
        assert args.spread == 2.0
        assert args.mc_draws == 500
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestCliWiring -v`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'draw_mechanism'`.

- [ ] **Step 3: Add the CLI arguments**

In `build_arg_parser`, add these three arguments immediately after the existing `--eliminate-per-draw` argument (before `--outdir`):

```python
    parser.add_argument("--draw-mechanism", choices=["empirical", "parametric"],
                        default="empirical",
                        help="(paired mode) Michaud draw source: 'empirical' = one draw per "
                             "transformer run (Phase 3); 'parametric' = N(mu_bar, s^2*Sigma/T) "
                             "Monte-Carlo draws (Phase 3b)")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="(parametric) spread knob s scaling Sigma/T (default 1.0 = canonical)")
    parser.add_argument("--mc-draws", type=int, default=1000,
                        help="(parametric) number of Monte-Carlo mu draws K (default 1000)")
```

- [ ] **Step 4: Thread the flags into `main`**

In `main`, change the `paired`-mode call from:

```python
        result = run_paired_experiment(
            prices, rets, cfg,
            iterations=args.iterations, transformer_runs=args.transformer_runs,
            seed=args.seed, eliminate_per_draw=args.eliminate_per_draw,
        )
```

to:

```python
        result = run_paired_experiment(
            prices, rets, cfg,
            iterations=args.iterations, transformer_runs=args.transformer_runs,
            seed=args.seed, eliminate_per_draw=args.eliminate_per_draw,
            draw_mechanism=args.draw_mechanism, spread=args.spread, n_draws=args.mc_draws,
        )
```

- [ ] **Step 5: Update the module docstring "Run:" block**

In the module docstring near the existing paired-mode examples, add:

```python
#   Parametric Michaud (Phase 3b, canonical s=1):
#     python experiments/measure_allocation_stability.py --mode paired \
#       --draw-mechanism parametric --spread 1.0 --mc-draws 1000 \
#       --transformer-runs 100 --iterations 3 --outdir experiments/results/parametric_3x100
```

- [ ] **Step 6: Run the CLI tests, then the whole file**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py::TestCliWiring -v`
Expected: PASS (existing CLI tests + 2 new = all green).

Run the full suite to confirm nothing regressed: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_measure_allocation_stability.py -v`
Expected: PASS (all existing 56 + 12 new = 68 green).

- [ ] **Step 7: Commit**

```bash
git add experiments/measure_allocation_stability.py tests/test_measure_allocation_stability.py
git commit -m "Add parametric Michaud CLI flags (--draw-mechanism/--spread/--mc-draws)"
```

---

## Post-implementation validation (manual, not a code task)

After Task 3, run the canonical first pass for real (slow — torch on the full universe, ~300 trainings; consider `run_in_background`):

```
python experiments/measure_allocation_stability.py --mode paired \
  --draw-mechanism parametric --spread 1.0 --mc-draws 1000 \
  --transformer-runs 100 --iterations 3 --seed 0 \
  --outdir experiments/results/parametric_3x100
```

Then read `experiments/results/parametric_3x100/paired_summary.txt` against **Decision Rule A**:

1. **Does the current method stabilise at n=100?** Read its overlap line ("X of ~9 shared") — 7-of-9 → consistent at production budget; 2–3-of-9 → still essentially random.
2. **Does parametric Michaud beat current on stability without the value crater?** Compare turnover / Jaccard / overlap **and** the per-arm return / Sharpe / value-CoV, versus both the in-run current arm and the saved 30×10 empirical baseline (`experiments/results/paired_summary.txt`).
3. **Conviction core** still high-frequency in the Michaud diagnostic.

Record the read-off in project memory (`project_allocation_stability.md`). If parametric at s=1 still degrades value, the built-in knob makes the follow-up trivial — sweep `--spread` over e.g. {0.25, 0.5, 1.0, 2.0} (reusing the same trainings is a later harness tweak) to map the stability-vs-value frontier. Productionising the winning recipe into `pipeline/04_allocate.py` remains a separate, pushable phase.

---

## Self-Review notes

- **Spec coverage:** uncertainty model `N(μ̄, s²·Σ/T)` reusing the optimiser's Σ (Task 1); `spread` knob built in, default 1.0, single canonical run (Tasks 1–3 + validation); draw-count K decoupled from run-count N via `n_draws`/`--mc-draws` (Tasks 1–3); maximal reuse of `resampled_allocate` + paired loop (Task 2); `sample_mu_draws` as the only new surface, torch-free/seedable (Task 1); `draw_mechanism` toggle defaulting to empirical so Phase 3 is untouched (Tasks 2–3); CLI flags (Task 3); validation config, double-duty read, comparison vs saved baseline, Decision Rule A (validation section). Out-of-scope items (s-sweep run protocol, cross-run-variance model, backtest, productionisation) are intentionally not implemented.
- **Placeholder scan:** none — every code/test step shows complete content and exact commands.
- **Type consistency:** `sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng) -> list[Series]` is defined in Task 1 and called with matching keyword args in Task 2. `run_paired_experiment`'s new params (`draw_mechanism`, `spread`, `n_draws`) defined in Task 2 match the CLI mapping in Task 3 (`args.draw_mechanism`, `args.spread`, `args.mc_draws`). The returned result dict shape consumed by the tests matches the existing `run_paired_experiment` contract (unchanged).
