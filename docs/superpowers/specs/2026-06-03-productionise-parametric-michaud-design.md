# Productionise Parametric Michaud (Step 4) — Design

## Problem

The walk-forward backtest showed **parametric Michaud beats the current msr+elimination method
out-of-sample** (seed 0, ~3yr OOS, n=50, frictionless): realized Sharpe current 1.17 vs parametric
s1/s2/s4 ~1.51–1.53, with parametric s1 best on growth (cum 1.43). The current pipeline `04_allocate`
still uses the old method. Juan wants to start using parametric Michaud (s=1) in production.

This is the **first pushable phase** of the allocation-stability investigation: the proven logic
(`sample_mu_draws` + average-then-floor consensus) currently lives only in the local-only experiment
module and must move into pushable production code.

## Goal

Add parametric Michaud as the production allocation method in `pipeline/04_allocate.py`, behind a
config toggle, defaulting to it — while keeping the current msr method as a fallback. Build now; gate
the merge/cutover on the in-flight robustness backtest confirming the win.

## Settled decisions (from brainstorming)

- **Sequencing:** build everything on a branch now (in parallel with the running robustness backtest);
  **hold the merge** until that run lands and confirms parametric wins. With `parametric_michaud` as
  the default, merging *is* the cutover, so the merge is the gate.
- **Toggle, default parametric:** new config `allocation_method: parametric_michaud | msr`, default
  **`parametric_michaud`**; `msr` kept as a permanent, instantly-revertible fallback.
- **Reproducibility:** seed step 4's Monte-Carlo draws via a config param, default a **fixed int**
  (`null` allowed for ad-hoc unseeded multi-runs). At K=1000 the seed is near-irrelevant to the
  *result* (LLN), so this is reproducibility/auditability insurance, not bias. **Scope of
  `michaud_seed`:** it governs ONLY the `sample_mu_draws` Monte-Carlo draws — `msr_tuned` is
  deterministic (no randomness), so the draws are the only stochastic element in step 4. The step-2
  transformer is NOT seeded by it (separate, out-of-scope phase). Consequence: re-running *just step 4*
  on the same `02_*` files is identical; re-running the *whole pipeline* still varies run-to-run (fresh
  μ̄), though parametric Michaud makes the weights far less sensitive to that.
- **`max_weight` after flooring — Option A:** keep the exact validated backtest logic (consensus floor
  renormalises survivors; a top name may marginally exceed `max_weight`, ~≤1%pt). This is what the
  winning backtest actually used. A strict post-floor cap (Option B) is a noted follow-up.
- **Push scope (updated):** push the functional changes **and their tests** ([[feedback-push-scope]]
  updated 2026-06-03 — tests follow the code they cover). Spec, plan, and docs stay local.

## Architecture

**New pushable module `src/allocation.py`** — portfolio-allocation *policies* as pure, testable
functions, sitting above `risk_kit` (which keeps the low-level `msr_tuned` / `portfolio_*` math):

- `msr_eliminate(returns, covmat, cfg)` — today's msr + batch-elimination loop, lifted verbatim out of
  `04_allocate.main` so it is testable and reusable.
- `sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng)` — K μ-vectors from
  `N(μ̄, spread²·Σ/T)` via one Cholesky (`spread=0` → exact copies). Ported from the validated
  experiment code.
- `apply_consensus_floor(weights, min_weight)` — drop names whose ascending-cumulative weight <
  `min_weight`, renormalise survivors; `len<=2` guard. Ported.
- `resampled_michaud(returns, covmat, cfg, n_periods)` — orchestrates: build rng from
  `cfg['michaud_seed']`; `sample_mu_draws`; raw `rk.msr_tuned` per draw (no elimination, `max_weight`
  enforced per draw); average the weight vectors; one `apply_consensus_floor`. Returns a weight Series.
- `allocate(returns, covmat, cfg, n_periods)` — thin dispatcher on `cfg.get('allocation_method',
  'parametric_michaud')`; raises `ValueError` on an unknown method.

**`04_allocate.py` becomes a thin script:** load μ̄ (`02_expected_returns`), Σ (`02_covmat`), the
selected names (`03_selected_returns`), and **T = `len(01_returns)`** (the only new input — read
directly; no change to step 2). Restrict μ̄/Σ to the selected set, call `allocate(...)`, write
`04_weights.csv` (unchanged format), keep the console reporting.

```
02_expected_returns (μ̄)  ┐
02_covmat (Σ)            ├─► restrict to 03_selected ─► allocate(method) ─► 04_weights.csv
01_returns (→ T = nrows) ┘                               │
                                          msr_eliminate  OR  resampled_michaud
```

Rationale for a new `src/` module (vs inline in `04` or bolted onto `risk_kit`): single-responsibility,
unit-testable allocation policies; `risk_kit` stays pure optimiser math; the parametric logic gets a
proper pushable home (production owns its own copy — experiments stay local per push-scope).

## Config (params.yaml)

```yaml
# Portfolio optimisation (Step 4)
allocation_method: parametric_michaud   # "parametric_michaud" | "msr"
michaud_spread: 1.0                      # s; draw scale on Σ/T (1.0 = canonical)
michaud_mc_draws: 1000                   # K Monte-Carlo μ draws
michaud_seed: 0                          # int = reproducible; null = fresh draws
```

`msr` ignores the three `michaud_*` keys. Existing `max_weight` (0.15), `min_weight` (0.05),
`rf_period`, `periods_per_year` are reused. No `config.py` change needed — raw keys flow through
`load_config`; `allocate` uses `cfg.get(...)` defaults so an older `params.yaml` still runs.

## `resampled_michaud` algorithm (the validated recipe)

1. `rng = np.random.default_rng(cfg.get('michaud_seed', 0))` — or `default_rng()` if the seed is `null`.
2. `draws = sample_mu_draws(returns, covmat, n_periods, michaud_mc_draws, michaud_spread, rng)`.
3. Each draw → `rk.msr_tuned(rf_period, μ_k, Σ, max_weight, periods_per_year)` (no elimination).
4. Average the K weight vectors → consensus.
5. One `apply_consensus_floor(consensus, min_weight)`.
6. Return the weight Series (dropped names = 0.0).

## Testing

`tests/test_allocation.py` (torch-free; **pushed** alongside `src/allocation.py`):

- `apply_consensus_floor` — sum-to-1; drops small tail + renormalises; no-drop when all pass; `len<=2`
  guard.
- `sample_mu_draws` — K series over the index; `spread=0` → exact copies; large-K mean ≈ μ̄ and cov ≈
  `s²·Σ/T`; seeded reproducibility.
- `msr_eliminate` — weights sum to 1; respects `max_weight`; eliminates below-`min_weight` names (pins
  current behaviour so the refactor is safe).
- `resampled_michaud` — consensus sums to 1; **deterministic given a fixed seed** (same seed →
  identical weights); `null` seed returns a valid book.
- `allocate` — dispatches to the right method; unknown method raises `ValueError`.

`04_allocate.py` is a thin script (load → dispatch → write), validated by an actual step-4 run on the
existing `data/02_*`/`03_*` files for **both** methods (`msr` must reproduce today's weights exactly),
not unit-tested — matching the pipeline convention.

## Rollout & gating

1. Build on the branch now (`src/allocation.py` + `tests/test_allocation.py` + `04_allocate` rewrite +
   `params.yaml` keys), TDD, reviewed.
2. Sanity-run step 4 on existing data with `msr` (reproduces today's weights) and `parametric_michaud`.
3. **Gate:** hold the merge until the robustness backtest (oos=250) confirms parametric wins. If it
   does → proceed; if it contradicts → set the default back to `msr` and reassess.
4. **Cutover (first push of this investigation):** ff-merge to local `main`, then **push the functional
   changes + their tests** (`pipeline/04_allocate.py`, `src/allocation.py`, `tests/test_allocation.py`,
   `params.yaml`) to a remote branch → PR, mirroring PR #21. Spec/plan/docs stay local.

## Out of scope (follow-ups)

- Step-2 transformer seeding (full end-to-end determinism) — separate phase.
- Strict post-floor `max_weight` cap (Option B).
- Sweeping/changing `s` (configurable; default 1.0) and the later n_runs cost/consistency analysis.
- Emitting the Michaud conviction-gradient diagnostic as a step-4 output.

## Related

- Backtest result + harness: `experiments/backtest_allocation.py`,
  `docs/superpowers/specs/2026-06-02-allocation-backtest-design.md`. Parametric draws: Phase 3b
  (`sample_mu_draws`, `resampled_allocate`). Push-scope [[feedback-push-scope]]; commit style
  [[feedback-commits]]; project [[project-allocation-stability]].
