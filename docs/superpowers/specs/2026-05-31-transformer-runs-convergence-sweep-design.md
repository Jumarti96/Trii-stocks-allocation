# Transformer-Runs Convergence Sweep — Design

## Problem

Composition instability is driven by run-to-run noise in `μ`, which shrinks ~`1/√n` as the
number of averaged Transformer runs (`n_transformer_runs`) grows. Before investing in structural
optimizer remedies (μ-shrinkage / resampling / HRP), we want to know how far simple averaging
gets us: **at what `n` do the per-stock forecast variance and the composition metrics (turnover,
Jaccard) stop changing meaningfully?** This also answers the standing "is 100 overkill?" question.

## Goal

A reusable experiment that sweeps `n_transformer_runs` across a grid and reports, per `n`, how
much the composition churns (mean turnover, mean Jaccard) and how noisy the per-stock forecast is
(mean per-stock across-iteration `μ` std), with consecutive-`n` deltas so the convergence point is
visible.

## Scope

- **In scope:** a behavior-preserving refactor of `train_and_predict` to expose per-run forecasts,
  and a new sweep harness `experiments/sweep_transformer_runs.py`.
- **Out of scope:** structural instability remedies (Phase 3 proper); plots (numbers-only); value
  CoV (ret/Sharpe) per `n` (trivial to add later, omitted by YAGNI); changing the pipeline's
  default `n_transformer_runs`.
- Settled parameters: grid `{10, 20, 30, 40, 50, 100}` (10–50 range plus a 100 anchor to confirm
  any 40→50 plateau is real); **30 iterations**; seed 0.

## Approach — prefix-averaging from one max-run training per iteration

Train `max(grid)=100` runs once per iteration, then derive `μ(n)` for every grid point by
averaging the **first `n`** of those runs. Cost is `iterations × max(grid) = 30 × 100 = 3000`
trainings (independent of grid size), versus `30 × Σ(grid) = 7500` for independent per-`n` runs.
It is also a **controlled comparison**: for a given iteration, `μ(10)` and `μ(50)` share the same
first 10 runs, so each iteration's `μ(n)` converges smoothly toward its own limit and the curve is
not confounded by independent sampling noise — exactly what a convergence study needs.

## Component 1 — expose per-run forecasts (behavior-preserving refactor)

In `src/transformer_model.py`, split `train_and_predict` into reusable pieces:

- **`train_runs(returns_df, cfg, n_runs=None, verbose=True)`** — the existing training loop,
  returning the raw per-run forecasts as an `np.ndarray` of shape
  `(n_runs, periods_to_forecast, n_stocks)` (no averaging, no winsorizing).
- **`winsorize_to_history(preds_df, returns_df)`** — the existing clip to the 1st/99th percentile
  of `returns_df`, extracted from the inline code.
- **`train_and_predict`** becomes a thin composition:
  `runs = train_runs(...); preds_df = pd.DataFrame(runs.mean(axis=0), columns=returns_df.columns);
  return winsorize_to_history(preds_df, returns_df)`.

`train_and_predict`'s output is **identical** to today's for the same seed; `pipeline/02_predict.py`
is unaffected. The sweep forms each grid point as
`μ(n) = weighted_mean_return(winsorize_to_history(pd.DataFrame(runs[:n].mean(axis=0), columns=rets.columns), rets))`,
matching pipeline semantics (average → winsorize → weighted mean) over the first `n` runs.

## Component 2 — the sweep harness

New file `experiments/sweep_transformer_runs.py`, reusing the instrument's pure helpers
(`select_stocks`, `allocate_msr`, `mean_turnover`, `mean_jaccard`, `weight_dispersion`,
`seed_everything`) and the Component-1 functions.

**`run_sweep(prices, rets, cfg, iterations, grid, seed, train_runs_fn=None, winsorize_fn=None,
period_mu_fn=None, select_fn=None, seed_fn=None)`**
- Setup once (deterministic): Ledoit-Wolf `Σ`; `selected = select_fn(prices, rets, cfg)`;
  `cov_sel = Σ.loc[selected, selected]`.
- For `i` in `1..iterations`: `seed_fn(seed + i)`; `runs = train_runs_fn(rets, cfg, n_runs=max(grid))`;
  for each `n` in `grid`: `prefix = runs[:n].mean(axis=0)`;
  `mu = period_mu_fn(winsorize_fn(pd.DataFrame(prefix, columns=rets.columns), rets))`;
  `mu_sel = mu.loc[selected]`; `weights = allocate_msr(mu_sel, cov_sel, cfg)`; record `mu_sel` and
  `weights` under `n`.
- Returns `{"selected": [...], "by_n": {n: {"mu": DataFrame(iters×selected), "weights": DataFrame(iters×selected)}}}`.
- The five `*_fn` arguments are dependency-injection seams (lazy real defaults: `train_runs`,
  `winsorize_to_history`, `weighted_mean_return`, `select_stocks`, `seed_everything`) so the loop
  is unit-testable without torch.

**`summarize_sweep(result, rel_tol=0.05)`** — for each `n` (ascending): `mean_turnover(weights_df)`,
`mean_jaccard(weights_df)`, and `mean_mu_std = weight_dispersion(mu_df).mean()` (mean per-stock
across-iteration `μ` std). Returns a table (DataFrame indexed by `n`) plus consecutive-`n` deltas
(`d_turnover`, `d_jaccard`, `d_mu_std`) and the first `n` at which all three relative deltas vs the
previous grid point are below `rel_tol` (the practical "converged" point; `None` if never).

**`write_sweep_outputs(result, outdir)`** — writes `sweep_metrics.csv` (rows = `n`; columns
turnover, jaccard, mean_mu_std) and `sweep_summary.txt` (the table + deltas + converged-`n` flag,
also printed). Numbers-only, no plots.

**`main()`** — CLI: `--iterations` (default 30), `--grid` (default "10,20,30,40,50,100"),
`--seed` (default 0), `--outdir` (default `experiments/results/sweep`). Loads `cfg`, `01_prices`,
`01_returns`, runs the sweep, writes + prints the summary. Requires `data/01_*` to exist.

## Data flow

```
per iteration i (x30):
  seed(i) -> train_runs(100) -> runs[ (100, periods, stocks) ]
     for n in {10,20,30,40,50,100}:
        runs[:n].mean -> winsorize -> weighted_mean_return -> mu(n)
        allocate_msr(mu(n)|selected, Sigma|selected) -> weights(n)
collect per n: mu_df(n) [30 x selected], weights_df(n) [30 x selected]
summarize -> turnover(n), jaccard(n), mean_mu_std(n), deltas, converged-n
```

## Testing

- **Component 1:** `train_runs` returns shape `(n_runs, periods, stocks)`; `winsorize_to_history`
  clips to the 1st/99th percentile of history; **refactor equivalence** — under the same seed,
  `winsorize_to_history(pd.DataFrame(train_runs(...).mean(0), columns=...), rets)` equals
  `train_and_predict(...)` (tiny synthetic data, `n_runs=2`, a few seconds with torch).
- **`run_sweep`** (torch-free via injected stubs): a crafted `train_runs_fn` returning a
  `(max_n, periods, stocks)` array whose prefix-averages converge (later runs add shrinking noise)
  → asserts per-`n` `mu`/`weights` shapes, correct prefix-averaging (μ(n) uses exactly the first
  `n` runs), and that `mean_mu_std` decreases monotonically with `n`.
- **`summarize_sweep`**: on synthetic per-`n` matrices, asserts the table values, the deltas, and
  the converged-`n` flag (including the `None` case when deltas never fall below tolerance).
- **`write_sweep_outputs`**: both files are written and reload to the expected shapes.
- **Manual run:** the full sweep (30 iterations, grid `{10..100}`) executes and produces a
  monotone-ish descending `mean_mu_std` and turnover/Jaccard table; controller runs it (GPU-heavy,
  ~3000 trainings) and reports the convergence point.

## Risks / trade-offs

- **Refactor risk:** `train_and_predict` is production code (used by `02_predict`). Mitigated by the
  refactor-equivalence test and keeping the split purely structural (no logic change).
- **Controlled-comparison caveat:** prefix-averaging means `μ(n)` for small `n` is a sub-average of
  the larger ones within an iteration. This is the intended design (clean convergence) and matches
  how more runs would actually be accumulated; it is not a confound.
- **Interpretation:** "converged" is judged by a relative-delta tolerance on a finite-iteration
  estimate; the flagged `n` is a practical guide, not a hard statistical threshold. The 100 anchor
  guards against declaring premature convergence inside the 10–50 range.
