# Optimization μ — Annualization Redesign (Phase 2)

## Problem

The optimizer is fed a **compound-annualized** expected-return vector
`μ = (1 + wmr)^periods_per_year − 1` (`src/transformer_model.py:annualize_expected_returns`),
where `wmr` is the per-period exp-decay-weighted forecast, while the covariance `Σ` it is
paired against is **per-period** (Ledoit-Wolf on the raw returns). This creates two defects:

1. **Cross-sectional distortion.** Compounding is a convex, nonlinear per-stock transform. It
   stretches the spread of `μ` and amplifies forecast noise unevenly (more for high-return
   names), feeding the mean-variance "error maximizer" and worsening composition instability.
2. **Unit mismatch.** Annualized `μ` over per-period `Σ` inflates the reported Sharpe by
   `√periods_per_year`. (This is a constant rescaling, so it does not change weights, but it
   makes the reported Sharpe uninterpretable.)

Phase 1 measured the downstream instability (30×10 run, seed 0): mean pairwise turnover **0.84**,
Jaccard **0.11**, no name held in >50% of runs, against a return CoV of ~16%. A side-finding
showed the mean annualized return fell from 2.28 (2 transformer-runs) to 0.92 (10 runs) —
direct evidence of a Jensen/convexity **upward bias** injected by compounding noisy forecasts.

## Goal

Make the `μ` the optimizer consumes **dimensionally consistent and free of compounding
distortion**, by optimizing entirely in per-period units and treating annualization as a
**display-only** transform in the report.

## Scope

- **In scope (Phase 2 = lever #1 only):** how `μ` is computed, persisted, consumed by the
  optimizer, and annualized for reporting. Update the stability instrument to mirror the new
  optimization so we get a clean before/after.
- **Out of scope (deferred to a possible Phase 3 = lever #2):** structural instability remedies
  — `μ`-shrinkage toward a prior, Michaud resampled efficiency, Hierarchical Risk Parity. Whether
  any of these is warranted depends on what lever #1 alone buys us.
- **Also out of scope:** changing the `weighted_mean_return` decay weighting (`lambda_=0.2`), the
  Transformer, the technical filter, the covariance estimator, and the notebooks (they already
  lag the pipeline per the README).

## Chosen approach — optimize in per-period units (Approach A)

Feed the optimizer `μ_opt = wmr` (per-period), the per-period `Σ` unchanged, and a per-period
risk-free rate `rf_period = (1 + rf_annual)^(1/periods_per_year) − 1` (the convention already used
in `risk_kit.sharpe_ratio`). The Sharpe ratio is then computed entirely in per-period units.

Rejected alternatives:
- **Arithmetic annualization** (`μ→ppy·wmr`, `Σ→ppy·Σ`, arithmetic `rf`): yields mathematically
  identical weights to Approach A but requires also rescaling `Σ` and keeping that consistent —
  equivalent result, more moving parts.
- **Keep compound + de-bias/calibrate:** leaves the convex cross-sectional distortion and the
  unit mismatch in place; patches a symptom. Rejected.

## Architecture & data flow

`μ` lives in per-period units everywhere it is computed and optimized; annualization is a
display-only transform applied in the report.

```
02_predict:  wmr ──► 02_expected_returns.csv  (PER-PERIOD; column "Expected Period Return")
                        ├──► 04_allocate:  rf_period ; msr_tuned(period μ, period Σ, rf_period)
                        └──► 05_report:    annualize per-period μ for display (compound)
```

**Single source of truth (D1):** `02_expected_returns.csv` is repurposed to hold the **per-period**
expected return (`wmr` per stock), column renamed `Expected Period Return`. There is no second
`μ` file — both consumers read this one; the report annualizes on the fly. (Two `μ` files would
invite the drift this redesign exists to remove.)

## Parameter robustness (design guarantee)

The user must be free to change `periods_per_year` (frequency) and `periods_to_forecast`
(horizon) without silently distorting the allocation. The new design guarantees this:

- **`periods_to_forecast`:** `μ_opt = wmr` is a per-period weighted mean. Changing the horizon
  changes only *which / how many* per-period predictions are averaged; the result stays in
  per-period units. The optimizer is agnostic to horizon length.
- **`periods_per_year`:** in the new design `ppy` enters the optimization **only** through
  `rf_period`. It no longer sits inside a nonlinear annualization of `μ`. Switching weekly↔monthly
  recomputes `rf_period` correctly; `μ` and `Σ` are already native to the data frequency.
- **Consistency rule (already enforced):** `ppy` must reflect the true data frequency — derived
  from `interval` by `config.py`; `Σ` comes from returns at that same frequency.

## Detailed changes

**`pipeline/config.py`**
- Add derived value `cfg['rf_period'] = (1 + cfg['rf_rate']) ** (1 / cfg['periods_per_year']) − 1`,
  shared by the allocator and the instrument (DRY).

**`pipeline/02_predict.py`**
- Replace `expected_returns = annualize_expected_returns(preds_df, ppy)` with the per-period
  quantity `expected_returns_period = weighted_mean_return(preds_df, lambda_=0.2)`.
- Write it to `02_expected_returns.csv` with header `['Expected Period Return']`.
- Metadata (`forecasted_prices`, winsorization bounds) unchanged.
- `annualize_expected_returns` remains in `transformer_model.py` (still used for display in 05 and
  by `experiments/compare_training_universe.py`).

**`pipeline/04_allocate.py`**
- Read `02_expected_returns.csv` (now per-period).
- Use `cfg['rf_period']` as the risk-free rate in `msr_tuned` (per-period μ, per-period Σ).
- Elimination loop unchanged. (`msr_tuned`'s `periods_per_year` arg is unused on this code path
  since `returns`/`covmat` are passed directly; leave the call signature as-is.)

**`pipeline/05_report.py`**
- Read per-period `μ` from `02_expected_returns.csv`.
- Per-stock "Expected Annual Return" = `(1 + μ_period)^ppy − 1` (compound, display only).
- Portfolio-level expected return computed consistently: `port_period = (weights·μ_period).sum()`,
  displayed as `(1 + port_period)^ppy − 1`.
- Forecasted-COP projection rewritten in per-period terms: `cop_per_stock·(1 + μ_period)^periods_to_forecast`
  (algebraically equal to the current `(1+annual)^(periods_to_forecast/ppy)`, just expressed in the
  new units).

**`experiments/measure_allocation_stability.py`** (must mirror the new optimization)
- `run_experiment`: aggregate predictions to **per-period** `μ` (`weighted_mean_return`) instead
  of annualizing.
- `allocate_msr`: use `cfg['rf_period']` (not annual `rf_rate`).
- This lets us re-run the instrument and compare turnover/Jaccard/CoV against the saved Phase 1
  "before" numbers. (The pre-change instrument version remains in git history if an exact "before"
  re-run is ever needed.)

## Testing

- **`rf_period` conversion:** `(1+rf)^(1/ppy)−1` correct for sample rates/frequencies.
- **Parameter robustness:** the value written by `02_predict` equals `weighted_mean_return(preds)`
  and is unchanged when `ppy` changes (no `ppy` term in `μ_opt`).
- **Display annualization:** compound helper `(1+μ_period)^ppy−1` matches expected values, and a
  per-period `μ` of 0 annualizes to 0.
- **Optimizer scale-invariance (confidence check):** `msr_tuned` returns the same weights for
  `(μ, Σ, rf)` and `(c·μ, c²·Σ, c·rf)` — confirms optimizing in any consistent unit is equivalent.
- **End-to-end:** the pipeline runs steps 2→5 cleanly and produces a report; the report's
  per-stock "Expected Annual Return" values are positive/sane and the table is internally
  consistent.
- **Instrument:** unit tests still pass after the `rf_period` change; an instrument smoke run
  produces the three output files.

## Validation plan

Re-run the (updated) stability instrument at seed 0, 30 iterations × 10 transformer-runs, and
compare to the Phase 1 baseline (turnover 0.84, Jaccard 0.11, return CoV 16%, top selection
frequency 50%). Record the deltas. This quantifies what the `μ` fix alone achieves and informs
the Phase 3 (lever #2) decision.

## Risks / trade-offs

- **The production allocation will change.** Today's weights were computed from a distorted `μ`;
  the new weights are better-justified but different. Expected and intended.
- **`02_expected_returns.csv` changes meaning** (annual → per-period). The column rename signals
  it; any external reader expecting annual values (e.g. notebooks) must be updated. A stale file
  from a prior run would be misread by the new `05_report`, so re-run from step 2 after the change.
- **Reported "Expected Annual Return" stays compound** and therefore still carries the Jensen
  upward bias as a *display* figure — acceptable because it no longer drives weights, and de-biasing
  the displayed forecast level is a separate concern (related to the calibration lever, out of scope).
- This change does not by itself make composition stable; it removes one distortion and makes the
  pipeline correct/robust. The structural instability remedies remain available for Phase 3.
