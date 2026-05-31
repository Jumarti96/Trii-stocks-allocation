# Allocation Stability Measurement — Design

## Problem

Across repeated end-to-end runs of the pipeline, the **per-stock forecasts** and the
**portfolio-level forecast** are relatively stable, but the **portfolio composition** (which
names are held and at what weight) varies noticeably. We want to quantify this instability to
decide whether it is a benign cosmetic artifact or a sign that mean-variance allocation is a
poor fit for this problem.

### Diagnosis (why this happens)

Only one input to the allocator changes between runs: the expected-returns vector `μ` from the
Transformer (no seed; random init/batching, averaged over `n_transformer_runs`). The Ledoit-Wolf
covariance `Σ` and the technical filter are both deterministic, so the candidate universe and
`Σ` are identical every run. Small, stable jitter in `μ` produces large swings in composition
because:

1. **Mean-variance optimization is an error maximizer** (Michaud, 1989) — weights ∝ `Σ⁻¹μ`
   amplify estimation noise in `μ`.
2. **The objective is flat near the optimum** — for correlated equities, many different weight
   vectors yield nearly the same Sharpe ratio. The `argmax` (composition) slides along the ridge
   while the value (portfolio Sharpe) barely moves. This is the exact signature observed.
3. **Near-substitutes** — correlated names with similar `μ` are nearly interchangeable.
4. **`Σ` is ill-conditioned** — wide eigenvalue spread makes some weight directions poorly
   determined.
5. **The min/max-weight elimination loop is a hard threshold** — it converts small continuous
   weight changes into discrete in/out membership flips, amplifying the *appearance* of
   instability.

## Goal

A **reusable, self-contained instrument** that runs the predict → allocate path many times and
reports how much the composition moves relative to how much the portfolio-level value moves.
Primary use now: test the flat-ridge hypothesis. Future use: re-run after model/constraint
changes, or sweep `n_transformer_runs` to decide whether the production value (100) is overkill.

## Scope

- **In scope:** measurement only — generate samples, compute and report stability metrics.
- **Out of scope:** any fix for the instability (seeding, `μ` shrinkage, resampled efficiency,
  HRP, etc.). Those are a separate brainstorming cycle, informed by these measurements.

## Architecture — Approach B (self-contained)

The instrument re-implements the selection and allocation logic locally rather than refactoring
the pipeline into shared functions. This keeps the pipeline untouched now; if/when the test
becomes a recurring need, we revisit and refactor `03_filter.py` / `04_allocate.py` into
importable functions. The instrument imports only the model/optimizer primitives it needs:
`train_and_predict`, `annualize_expected_returns` (from `src/transformer_model.py`) and
`msr_tuned`, `portfolio_return`, `portfolio_vol`, `technical_indicators` (from `src/risk_kit.py`).

**Fidelity requirement:** the locally re-implemented selection and elimination logic must match
`03_filter.py` and `04_allocate.py` exactly, so the instrument measures the real pipeline's
behavior. This duplication is the accepted trade-off of Approach B.

## Location & Invocation

`experiments/measure_allocation_stability.py`

```bash
python experiments/measure_allocation_stability.py --iterations 30 --transformer-runs 10
```

CLI arguments:

| Flag | Default | Meaning |
|---|---|---|
| `--iterations` | 30 | Number of end-to-end predict→allocate runs |
| `--transformer-runs` | 10 | `n_runs` passed to `train_and_predict` per iteration |
| `--seed` | 0 | Base seed; iteration `i` uses `seed + i` |
| `--outdir` | `experiments/results` | Where CSVs and summary are written |

Requires `data/01_prices.csv` and `data/01_returns.csv` to already exist (step 1 already run).
The instrument does not download data or run the orchestrator.

## Flow

**Setup (once, deterministic):**

1. `cfg = load_config()`; read `data/01_prices.csv`, `data/01_returns.csv`.
2. Compute Ledoit-Wolf covariance `Σ` over the full universe (identical to `02_predict.py`).
3. Compute the **selected name set** by replicating `03_filter.py`: per ticker, call
   `rk.technical_indicators(..., indicators=['SMA','EMA','MACD','PRC'], ma_terms, macd_params=[12,26,9], signal_tolerance=0.975)`, take the last row, keep tickers whose positive-signal
   count ≥ `signal_min_count`. Selection is fixed for the whole experiment.

**Per iteration `i` in `1..N`:**

1. Seed `torch.manual_seed(seed+i)` and `np.random.seed(seed+i)` for whole-experiment
   reproducibility while still varying run to run.
2. `preds_df = train_and_predict(rets, cfg, n_runs=transformer_runs)`.
3. `μ = annualize_expected_returns(preds_df, periods_per_year)`.
4. Restrict `μ` and `Σ` to the selected names.
5. Allocate by replicating `04_allocate.py`: `msr_tuned(rf, μ_sel, Σ_sel, max_weight, periods_per_year)`,
   then the batch-elimination loop (drop names whose ascending cumulative weight < `min_weight`,
   re-optimize, repeat) until stable. Eliminated names get weight 0.
6. Record the weight vector over the selected names and the portfolio metrics.

**Portfolio metrics per iteration** — computed identically to `msr_tuned`'s internal objective
for fidelity (this intentionally preserves the pipeline's existing convention of pairing an
**annualized** `μ` with a **per-period** `Σ`; it is not "corrected" here):

- `ret_i = portfolio_return(w_i, μ_sel)`  (annualized μ)
- `vol_i = portfolio_vol(w_i, Σ_sel)`     (per-period covariance)
- `sharpe_i = (ret_i - rf) / vol_i`

## Metrics & Outputs

All iterations share the same column space (the fixed selected-name set), so weight vectors and
membership sets are directly comparable. Eliminated/absent names = weight 0.

**Composition instability (expected high):**

- **Selection frequency** — per stock, fraction of iterations with weight > 0. Names in the
  ~20–80% band indicate unstable membership.
- **Mean pairwise turnover** — `0.5 · Σ|wᵢᵃ − wᵢᵇ|` averaged over all iteration pairs. Headline
  "how much would I trade between two runs" number.
- **Mean pairwise Jaccard** of the survivor name-sets (1.0 = identical membership every run).
- **Per-stock weight std** across iterations.

**Value stability (expected low):**

- Std and coefficient of variation of `ret_i`, `vol_i`, `sharpe_i` across iterations.

**Amplification factor:**

- Mean per-stock input noise (std of `μ` across iterations) vs. mean per-stock output weight std
  — how much the optimizer magnifies the forecast jitter.

**Interpretation note (printed in the report):** fewer transformer runs ⇒ noisier `μ` per
iteration ⇒ *more* composition instability. A 10-run measurement is a **conservative upper
bound** on production instability at 100. If composition looks stable at 10, production is
tighter; if it looks unstable at 10, re-run with `--transformer-runs 100` for the true
production figure before concluding. Sweeping `--transformer-runs` (10/25/50/100) answers "is
100 overkill?" with no extra code.

**Output files (to `--outdir`):**

| File | Contents |
|---|---|
| `stability_weights.csv` | iterations × selected-names weight matrix (raw evidence) |
| `stability_metrics.csv` | per-iteration `ret`, `vol`, `sharpe` |
| `stability_summary.txt` | the computed summary metrics above (also printed to stdout) |

No plots in this version (numbers-only).

## Testing

- **Determinism check:** running the instrument twice with the same `--seed` and
  `--iterations` produces identical `stability_weights.csv` (confirms the per-iteration seeding
  is wired correctly and the experiment is reproducible).
- **Degenerate-input sanity:** with `--iterations 1` the summary metrics that require ≥2
  samples (turnover, Jaccard, std) are reported as N/A rather than crashing.
- **Fidelity spot-check:** for a single seeded iteration, the selected name set and final
  weights match a normal pipeline run of `03_filter.py` + `04_allocate.py` under the same seed
  and `n_transformer_runs`, confirming the Approach-B re-implementation matches the pipeline.

## Phase 2 — Forecast annualization (follow-on, do not forget)

This spec covers **Phase 1 only** (measurement). A finding uncovered during design must be
assessed once Phase 1 confirms the instability:

`annualize_expected_returns` forms `μ = (1 + wmr)^periods_per_year − 1`
(`src/transformer_model.py:106`) — a **compound, convex, per-stock** transform. Unlike
arithmetic annualization (`ppy · wmr`, under which max-Sharpe is horizon-invariant and weights
are unchanged), compounding (a) stretches the cross-sectional spread of `μ` and (b) amplifies
per-stock forecast noise by ~`ppy·(1+r)^(ppy−1)` (~60×), **more for high-return names** — i.e.
exactly the names most likely to be selected and heavily weighted. This plausibly *aggravates*
the composition instability Phase 1 measures, so it is a prime stability lever, not just a
cosmetic detail.

**Phase 2 gets its own brainstorm → spec → plan cycle.** Likely shape: extend the instrument (or
add a variant) to measure stability under compound vs. arithmetic vs. per-period `μ`, then decide
whether to change the pipeline's annualization. Out of scope for the Phase 1 build.

Separately and cosmetically: `msr_tuned` maximizes annualized `μ` against a **per-period** `Σ`,
so the *reported* Sharpe is inflated by `√periods_per_year` (≈ 7.35×). This is a constant
rescaling of the objective and therefore **does not change the optimal weights** — it only makes
the printed Sharpe values uninterpretable in absolute terms. Low-priority cleanup, tracked here
so it is not rediscovered from scratch.

## Risks / Trade-offs

- **Logic drift (Approach B):** the duplicated selection/elimination logic can diverge from the
  pipeline over time. Mitigated by the fidelity spot-check test and a code comment pointing at
  the source steps. Accepted deliberately to avoid touching the pipeline now.
- **Cost:** 30 iterations × 10 transformer runs = 300 trainings. Acceptable on GPU; the
  `--transformer-runs` and `--iterations` knobs let the user trade speed for precision.
