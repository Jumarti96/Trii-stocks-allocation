# Allocation Stability Phase 3b — Parametric Michaud (Mechanism B) — Design

## Problem

Phase 3 validated **empirical-ensemble Michaud** (mechanism A: each transformer run is one μ
draw). The paired run (30 iterations × 10 transformer-runs, seed 0) gave a sharp split verdict
against Decision Rule A:

- **Stability improved strongly** — turnover 0.8406 → 0.6245 (−26%), Jaccard 0.1087 → 0.2573
  (+137%), overlap 1.7/9.0 (19%) → 5.3/13.3 (40%).
- **The conviction core survived and sharpened** — a much harder core emerged (CBU7.L 96.7%,
  SGLD.L 80%, JNJ 73.3% held), versus the old method where nothing beat NUTRESA at 50%.
- **Value degraded materially** — return mean 0.0116 → 0.0066 (−43%), Sharpe 0.5265 → 0.3319
  (−37%), Sharpe CoV 0.131 → 0.324 (worse ~2.5×); vol fell 0.0185 → 0.0140 (−24%). The book grew
  9 → 13.3 names: the **diffuse-consensus** outcome the Phase 3 spec flagged as the caveat.

### Root cause

In mechanism A, **one Michaud draw = one single transformer run's μ**, which is
**noise-dominated**: the runs sweep measured single-run per-stock μ std ≈ 0.0167, on the order of
the cross-sectional signal itself. Raw `msr_tuned` on a noise-dominated μ produces a corner bet on
essentially-random names; averaging ~10 such corners yields a diffuse book that bets a little on
whatever each noisy draw happened to like. The value cost is therefore an artifact of the **draw
definition**, not of Michaud resampling per se.

Mechanism B (parametric Michaud) was parked in the Phase 3 spec as the fallback "only if the
empirical-ensemble mechanism underperforms." It just did, so it is now warranted.

## Goal

Validate, in the experiment harness only, that **parametric Michaud** — μ draws sampled around a
well-estimated center with a *principled, controllable* spread — recovers the value lost at 30×10
while keeping the stability gains. Productionising the winning recipe into `pipeline/04_allocate.py`
remains a deliberate later phase.

## Settled decisions

- **Uncertainty model — canonical Michaud:** draw μ from `N(μ̄, s²·Σ/T)` where
  - `μ̄` = average of the N transformer runs (per-period `weighted_mean_return`), restricted to the
    selected names;
  - `Σ` = the **existing** Ledoit-Wolf return covariance, restricted to the selected names;
  - `T` = number of return periods; `s` = the tunable spread knob (default 1.0).

  This reuses the exact Σ already in the optimizer (full cross-stock structure, sampled via one
  Cholesky), so there is **nothing extra to estimate**. It is the textbook Michaud & Michaud
  resampling distribution (the standard error of an estimated mean). Chosen over estimating the
  spread from the **cross-run forecast variance** because cross-run spread measures only the model's
  training self-consistency — a confidently-wrong model has low cross-run spread — and it shrinks
  ~1/n with averaging. Cross-run forecast variance is kept as a **documented alternative** for a
  later test if `Σ/T` does not help.

- **Framing — single canonical run first.** Run textbook Michaud at `s = 1` and compare to the
  baseline. The knob is built in as a parameter (default 1.0) so a future stability-vs-value sweep
  over `s` is a one-flag follow-up, not a rewrite.

- **Draw count decoupled from run count.** `N` (transformer runs) buys a good μ̄; `K` (MC draws)
  buys a smooth consensus. Defaults: `N = 100`, `K = 1000`.

- **Maximal reuse.** Parametric draws feed the **existing**, already-tested `resampled_allocate`
  (raw `msr` per draw → average weights → one consensus floor pass) and the existing paired loop
  unchanged. All changes are additive and local-only; no `pipeline/` or `src/` edits (same scope
  rule as Phase 3, per [[feedback-push-scope]]).

## Architecture — data flow (per iteration)

```
train N runs ──► μ̄  (avg → weighted_mean_return, restricted to selected)
       │
       └────────► (kept for the cross-run-variance ALTERNATIVE; not wired up by default)

Σ (Ledoit-Wolf, restricted to selected)  +  T (#periods)  +  s (=1)
       │
       ▼
draw K μ-samples:  μ_k ~ N(μ̄, s²·Σ/T)        ◄── the only genuinely new piece
       │
       ▼
resampled_allocate(μ_1…μ_K, Σ, cfg)          ◄── EXISTING, already tested
   (raw msr per draw → average weights → one consensus floor pass)
       │
       ▼
consensus weights  +  conviction diagnostic
```

The **current** arm (μ̄ → one `allocate_msr`) runs off the *same* μ̄, so the before/after stays
perfectly paired exactly as in Phase 3.

## Components

1. **`sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng)`** *(new — the whole
   novelty).* Returns a list of `n_draws` μ Series sampled from `N(μ̄, spread²·Σ/T)` via one
   Cholesky of `Σ`. Pure and seedable (takes an explicit numpy `Generator`), torch-free → trivially
   unit-testable. `spread=0` returns `n_draws` exact copies of μ̄. Index/order preserved. The
   cross-run-variance alternative is a second documented branch, not wired up in this phase.

2. **`run_paired_experiment` — add a `draw_mechanism` toggle** (`"empirical"` | `"parametric"`,
   plus `spread`, `n_draws`). When `parametric`, the Michaud arm's per-draw μ list comes from
   `sample_mu_draws(...)` instead of the per-run μ's; **everything else in the loop is unchanged** —
   current arm, scoring against the shared μ̄, metrics, conviction diagnostic, outputs. Default stays
   `empirical` so the Phase 3 behaviour and tests are untouched.

3. **CLI** — `--draw-mechanism parametric`, `--spread 1.0`, `--mc-draws 1000`, parsed by
   `build_arg_parser` and threaded into `run_paired_experiment`. Defaults preserve existing
   behaviour.

## Validation run & comparison

**Canonical first pass:**

```
--mode paired --draw-mechanism parametric --spread 1.0 --mc-draws 1000 \
  --transformer-runs 100 --iterations 3 --seed 0 \
  --outdir experiments/results/parametric_3x100
```

- **Train once per iteration, reuse:** the 100 runs give both arms μ̄ and Σ; the 1000 parametric
  draws are the cheap MC step. ~300 trainings total — the same budget as the 30×10 run that
  completed fine.
- `iterations=3` matches the spec's n=100 reproducibility protocol. Turnover/Jaccard are coarse on
  3 points, so the **overlap "X of ~Y names shared"** count is the headline; turnover/Jaccard are
  secondary.

**This run does double duty** (both from the same draws):

1. *Baseline at production budget* — does the **current** method actually stabilise at n=100? (7 of
   ~9 shared → consistent; 2–3 of ~9 → still essentially random.)
2. *Does parametric Michaud beat it* on stability **without** the value crater seen at 30×10?

**Comparison targets:** the in-run current arm (paired, n=100) **and** the saved 30×10
empirical-Michaud baseline (`experiments/results/paired_*`), so we can see directly whether
parametric draws fix the diffuseness/value problem the single-run empirical draws caused. Verdict
read against **Decision Rule A** (stability up, conviction core survives, value not materially
degraded).

## Testing

Same torch-free, injected-stub style as the existing 56 tests in
`tests/test_measure_allocation_stability.py`.

- **`sample_mu_draws`** (the real new surface): at large K the draw mean ≈ μ̄ and draw covariance ≈
  `s²·Σ/T` (within tolerance); `spread=0` → every draw equals μ̄ exactly; larger `spread` → larger
  dispersion (monotonic); seeded rng → reproducible; index/order preserved.
- **Parametric wiring:** `run_paired_experiment(..., draw_mechanism="parametric")` returns the same
  result structure, both arms' weights sum to 1, and the two arms differ; the existing `empirical`
  path is untouched (regression check).
- **CLI:** `build_arg_parser` parses `--draw-mechanism`, `--spread`, `--mc-draws` with the right
  defaults (`empirical`, `1.0`, `1000`).

## Out of scope (future tests / phases)

- **Spread sweep** over `s` to map the full stability-vs-value frontier (the knob is built in; only
  the run protocol is deferred).
- **Cross-run forecast-variance** uncertainty model, and a **combined** model-noise + sampling-error
  model.
- **Walk-forward backtest** for true *out-of-sample* value — the only thing that fully resolves the
  in-sample-value ambiguity (the current arm is the in-sample Sharpe-argmax on the shared μ̄, so the
  harness's value comparison is structurally tilted toward it).
- **Productionisation** into `pipeline/04_allocate.py` (a separate phase; that is what gets pushed).

## Related

- Phase 3 design + plan: `docs/superpowers/specs/2026-06-01-allocation-stability-phase3-michaud-design.md`,
  `docs/superpowers/plans/2026-06-01-allocation-stability-phase3-michaud.md`.
- Phases 1/2 and the transformer-runs sweep specs are alongside under `docs/superpowers/specs/`.
- Push-scope rule [[feedback-push-scope]]: experiments/tests/docs stay local; only functional
  pipeline changes are pushed (the later productionisation phase).
