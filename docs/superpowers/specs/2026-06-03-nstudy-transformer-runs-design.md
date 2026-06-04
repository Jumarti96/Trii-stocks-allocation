# n_transformer_runs study for the parametric-Michaud allocation — design

Date: 2026-06-03
Status: approved (pending spec review)

## Purpose

Pick a **fair** `n_transformer_runs` for the deployed parametric-Michaud (s4) allocation —
one that balances composition stability against compute cost so smaller-compute users get a
defensible default — and gather **rough** evidence on how that stability/value tradeoff shifts
with the spread `s` (including `s=8`) as `n` grows.

This is **not** a strict optimization. The earlier transformer-runs sweep already established
that per-stock μ-std decays as `c/√n` with no knee, so there is no magic `n`; we are looking for
where the composition metrics flatten "enough" that more runs stop buying meaningful stability.
A small budget (10 iterations, 2 seeds) is deliberate — the goal is to *locate* an acceptable
`n` in a few tries, not to estimate it precisely.

This study is **in-sample**: value is scored against the forecast μ̄(n) that the `current` arm
is the Sharpe-argmax of, so forecast Sharpe/return *levels* are biased toward `current`. Read the
**composition metrics** (turnover, Jaccard, overlap) and the **value dispersion** (CoVs), not the
value levels. The realized out-of-sample verdict is a separate study (Plan 1).

## Scope

In scope:
- A new experiment script that sweeps `n` for the `current` (msr) arm and the parametric-Michaud
  arm at `s ∈ {4, 8}`, using the historical Ledoit-Wolf Σ.
- The train-once / prefix-average efficiency trick (one training set of `max(grid)` runs per
  iteration, reused across all grid points).
- Multi-seed aggregation (mean ± cross-seed spread).

Out of scope (explicitly dropped during brainstorming):
- A forecast-derived covariance variant (the "Q2" idea) — dropped.
- `s=2` — dropped: not a production candidate (the backtest set the low-spread arms aside), and
  the `current` arm already anchors the low-stability end, so s=2 sits redundantly between them.
- Any realized/backtest scoring — that is Plan 1, a separate study.
- Productionising a chosen `n` — a later, pushable step once `n` is picked.
- Modifying `sweep_transformer_runs.py`, `measure_allocation_stability.py`, or any pipeline code.

## Configuration

| Knob | Value | Rationale |
|---|---|---|
| grid (`n`) | `[10, 25, 50, 75, 100]` | log/doubling-spaced; 75 added to probe the 50→100 middle. Free under the prefix trick (cost set by `max(grid)=100`). |
| iterations | `10` | enough to *locate* a fair `n`; the n→stability direction is already known. |
| seeds | `[0, 100]` | 2 disjoint base seeds (spaced ≥ iterations so per-iteration streams `seed+i` never overlap) to expose cross-seed spread cheaply. |
| spreads `s` | `[4, 8]` | s4 is deployed; s8 is the aggressive bracket showing how much more stability a bigger `s` buys as `n` grows. The `current` arm anchors the low-stability end. |
| `K` (mc_draws) | `1000` | matches the deployed `michaud_mc_draws`; keeps the subtle n-to-n consensus deltas clean. |
| Σ | historical Ledoit-Wolf (on `01_returns`) | the deployed risk model; the forecast-Σ variant was dropped. |
| arms | `current` + `parametric@{4,8}` | one msr baseline + two spreads, all scored against the same μ̄(n). |

## Approach (Approach A — new file)

New file `experiments/nstudy_transformer_runs.py`. It **reuses**:
- from `sweep_transformer_runs.py`: the train-once-per-iteration + first-`n`-prefix-average loop
  pattern (it does not import the msr-only `summarize_sweep`).
- from `measure_allocation_stability.py`: `select_stocks`, `allocate_msr`, `sample_mu_draws`,
  `resampled_allocate`, `portfolio_metrics`, `mean_turnover`, `mean_jaccard`, `overlap_stats`,
  `metric_dispersion`, `selection_frequency`, `seed_everything`, and the lazy `train_runs` /
  `winsorize_to_history` / `weighted_mean_return` forecaster seams.

`sweep_transformer_runs.py` and `measure_allocation_stability.py` are left untouched.

### Core loop

For each `seed` in `[0, 100]`:
  - Compute the deterministic, once-per-run objects: Ledoit-Wolf Σ over `01_returns`, the
    technical-filter `selected` names, and `cov_sel = Σ.loc[selected, selected]`, `T = len(rets)`.
  - For each `iteration i` in `1..10`:
    - `seed_everything(seed + i)`; `runs = train_runs(rets, cfg, n_runs=100)` (one training set).
    - For each `n` in the grid:
      - `prefix = runs[:n].mean(axis=0)`; `μ̄(n) = weighted_mean_return(winsorize(prefix)).loc[selected]`.
      - **current arm:** `w = allocate_msr(μ̄(n), cov_sel, cfg)`; record weights + `portfolio_metrics`
        scored against `μ̄(n)`.
      - **parametric arms:** for each `s` in `[4, 8]`:
        - draws = `sample_mu_draws(μ̄(n), cov_sel, n_periods=T, n_draws=K, spread=s, rng)` where `rng`
          is re-seeded **identically for all three spreads at this `(seed, i, n)`** (e.g.
          `np.random.default_rng((seed + i, n))`). Because `chol(s²Σ/T) = s·chol(Σ/T)`, identical `z`
          across spreads ⇒ the three s-draws are perfectly paired (same noise, scaled).
        - `consensus, _ = resampled_allocate(draws, cov_sel, cfg)`; record weights + `portfolio_metrics`
          scored against the same `μ̄(n)`.

The `current` arm reproduces today's pipeline-minus-Michaud; each parametric arm is the deployed
resampled-consensus at one spread. All arms at a given `(seed, i, n)` are scored against the same
μ̄(n), so the comparison is paired before/after at zero extra training cost.

### Aggregation

Per `(seed, n, arm)`, over the 10 iterations' weight vectors and metric rows:
- composition: `mean_turnover`, `mean_jaccard`, `overlap_stats` (shared / held / fraction),
  mean held count, and `selection_frequency`.
- value dispersion: `metric_dispersion` → mean, std, CoV for ret / vol / sharpe.

Then across the 2 seeds: **mean ± cross-seed std** for every scalar metric (2 points → coarse
spread, by design).

## Outputs (`experiments/results/nstudy/`)

- Per-seed raw CSVs: `nstudy_<arm>_<seed>_weights.csv` and `_metrics.csv` (arm ∈
  `current, s4, s8`), indexed by `(iteration, n)`.
- A metric-vs-`n` table per arm: rows = `n`, columns = the composition + value-dispersion metrics,
  each as `mean ± cross-seed std`, written as CSV + rendered in the summary.
- `nstudy_summary.txt`: the per-arm metric-vs-n tables, plus an **advisory** note flagging the
  first `n` where the s4 composition metrics change less than a relative tolerance vs the previous
  grid point (a hint, not a verdict — the pick is the user's judgment).

## Testing

Mirror the existing experiment scripts' dependency-injection pattern so the loop is unit-testable
without torch: the core runner takes injectable `train_runs_fn` / `winsorize_fn` / `period_mu_fn` /
`select_fn` / `seed_fn` seams defaulting to the real (lazy-torch) implementations. Tests use stub
forecasters returning deterministic per-run arrays and assert: the prefix trick averages the first
`n` runs, all four arms are produced per `(seed, n)`, the three spreads share `z` (paired draws),
the aggregation reports mean ± cross-seed std, and the output writers produce the expected files.
The metric helpers themselves are already covered by `tests/test_measure_allocation_stability.py`.

## Cost

- Transformer trainings: 2 seeds × 10 iter × 100 runs = **2,000** (~9 s each ⇒ ~5 h).
- K=1000 optimizations: 2 seeds × 10 iter × 5 n × 2 spreads = **200** consensus batches
  (~55–60 s each ⇒ ~3.3 h). Each batch ≈ 7–8 transformer trainings; per iteration the 100
  trainings (~15 min) modestly exceed the MC optimizations (~10 min).
- **Total ≈ 8.3 h** — a single background/overnight run. `current`-arm `allocate_msr` cost is
  negligible.

## Conventions

Experiment-only and **local** per the push-scope rule (not pushed to the remote). Run in the
background (silent; empty stdout ≠ stall). Results land in the gitignored
`experiments/results/nstudy/`.
