# Allocation Stability Phase 3 — Michaud Resampled Efficiency — Design

## Problem

Portfolio composition is near-random run to run (Phase 1: mean pairwise turnover 0.84, Jaccard
0.11, 61 of 69 eligible names drawn across 30 runs, zero names held in >50% of runs). This is the
classic mean-variance "error maximizer + flat-objective ridge": only `μ` (the unseeded Transformer
expected returns) moves between runs; `Σ` (Ledoit-Wolf) and the technical filter are deterministic.

Both cheap levers are now exhausted:

- **Phase 2** (per-period `μ` vs compound annualisation) was correctness-only — composition
  instability essentially unchanged (turnover 0.8407→0.8406, Jaccard 0.108→0.109).
- **The transformer-runs sweep** showed averaging buys only `1/√n` precision; composition
  stabilises far too slowly (turnover 0.83→0.71, Jaccard 0.12→0.22 across a 5× run increase), so
  stable holdings would need hundreds of runs — not viable.

A **structural** optimizer remedy is therefore warranted.

### Key insight (the thing this design is built around)

The instability is **not uniform randomness — there is a conviction gradient.** Each portfolio
holds ~9 names out of ~69 eligible, so the random base rate for any name appearing is ~13%.
NUTRESA.CL appeared in 50% of runs ≈ 3.8× base rate (~6σ above chance) → a genuine **stable core**
of high-conviction names exists. The high turnover / low Jaccard is dominated by a churning
**middle band** of near-tied marginal names. The real problem is narrow: *which marginal names
fill out the back half of the book.* A consensus method that selects by cross-draw frequency and
weights by average is the natural fit.

## Goal

Validate that **Michaud Resampled Efficiency** — Monte-Carlo over `μ`, optimize each draw, average
the weight vectors into one consensus portfolio — meaningfully reduces composition instability
**without** wrecking forecast value, using the existing stability instrument. Productionising the
winning recipe into the pipeline is a deliberate **follow-up phase**, not part of this one.

## Scope

- **In scope:** a resampled-allocation path and a paired current-vs-Michaud comparison inside the
  experiment harness (`experiments/measure_allocation_stability.py` + tests), local-only per the
  push-scope rule. A new overlap-count metric. A documented n=100 reproducibility configuration.
- **Out of scope:** any `pipeline/` or `src/` change (productionisation is a later phase); the
  contingency levers below (build only if the diffuse-consensus pathology appears); the parametric
  and bootstrap draw mechanisms (fallback only if the empirical-ensemble mechanism underperforms).
- Settled decisions (from brainstorming):
  - **Deliverable:** validate in the harness first, productionise later (Framing A).
  - **Draw mechanism:** empirical ensemble — each of the N Transformer runs is one `μ` draw
    (mechanism A). Fall back to bootstrap (B′) or parametric Michaud (B) only if N draws prove too
    few to stabilise the middle.
  - **Consensus rule:** average-then-floor (rule A) — average the per-draw weight vectors, then one
    batch-elimination pass on the consensus.
  - **Elimination timing:** **deferred** — raw `msr_tuned` per draw (no per-draw elimination),
    eliminate once on the consensus. Per-draw elimination is kept only as an optional comparison arm.
  - **Success criterion:** stability must improve *without degrading value* (criterion A);
    magnitudes read off the paired evidence, not pre-committed.

## Approach — one paired loop, two arms from the same draws

Choosing rule A has a clean consequence: within a single set of N Transformer runs, the **current
method** averages the runs → one `μ` → one allocation, while **Michaud** optimizes each run's `μ`
→ averages the weights → one consensus. *Both consume the same N runs.* So the instrument computes
current-vs-Michaud from the **same draws** every iteration — a perfectly paired before/after at zero
extra training cost.

Per iteration:

1. `train_runs(rets, cfg, n_runs=N)` → `(N, periods, n_stocks)`; winsorise to history.
2. **Current arm (unchanged pipeline):** `runs.mean(axis=0)` → `weighted_mean_return` → one
   per-period `μ`, restricted to the selected names → `allocate_msr` (the `msr_tuned` +
   batch-elimination loop). This is exactly today's `train_and_predict` → `04_allocate` path.
3. **Michaud arm:** `weighted_mean_return` applied **per run** → N per-period `μ` vectors, each
   restricted to the selected names → raw `msr_tuned` each (Σ shared, **no** elimination) →
   continuous `r_i` summing to 1.
4. **Consensus:** `r_consensus = mean_i(r_i)` → one batch-elimination pass to enforce `min_weight`
   → renormalise to sum 1.

`Σ` (Ledoit-Wolf) and the selected-name set are computed once and shared by both arms.

### Why deferred elimination (average-then-threshold), not per-draw

Threshold-then-average destroys the very signal averaging exists to exploit:

1. **Don't threshold before you aggregate.** The raw `msr` weights carry each draw's *continuous*
   conviction. Per-draw elimination is a hard discrete cut applied *before* the average sees it —
   the analogue of averaging `argmax`es instead of averaging distributions and taking one `argmax`
   at the end. Rule A is literally "average, then floor"; a per-draw floor contradicts it.
2. **The discreteness is itself the instability.** A name at the cumulative-weight floor flips
   in/out across draws; per-draw elimination amplifies that into a hard 0-vs-nonzero swing — exactly
   the churning-middle behaviour. Deferred elimination lets the average smooth it: a name marginal
   in every draw gets a stable small average and is cleanly pruned once.
3. **Cheaper** — one `msr` per draw plus one elimination loop at the end, versus a multi-pass
   re-optimising loop on every draw.

There is no "fidelity to production" argument for per-draw elimination: production never allocates
per run — it averages `μ` first and allocates once.

**Honest caveat:** raw `msr_tuned` tends toward corner solutions (a couple of names at `max_weight`,
the rest zero) because Sharpe-max is an error-maximizer. Averaging many *different* corner solutions
can yield a **diffuse** consensus, so the final book may be larger / more diversified than today's.
That is expected (diversification-through-averaging is the point of Michaud) and is measured; a
*too*-flat consensus is the signal to reach for a contingency lever.

## Component 1 — resampled allocation

A new function in the experiment module (e.g. `resampled_allocate(per_run_mu, covmat, cfg)`):

- Takes the N per-run per-period `μ` vectors (already restricted to selected names) and the shared
  `Σ`.
- Runs raw `msr_tuned` per draw (Σ shared, `max_weight` from cfg, **no** elimination) → `r_i`.
- Returns `r_consensus = mean_i(r_i)` after one batch-elimination pass + renormalisation, **plus**
  the per-name diagnostic table (selection frequency across draws + mean raw weight).

An `eliminate_per_draw: bool = False` toggle keeps per-draw elimination available as the optional
comparison arm (one-line change), but **deferred is the default**.

`allocate_msr` (the existing elimination loop) is reused verbatim for the current arm and for the
single consensus elimination pass; raw `msr_tuned` is reused for the per-draw optimisation.

## Component 2 — paired harness

Refactor `run_experiment` so each iteration trains the N runs once and derives **both** arms from
those draws, emitting two weight vectors per iteration. Across M iterations this yields two
`weights` DataFrames (current, michaud) plus two `metrics` DataFrames — paired, same draws. Σ and
the selected set stay computed once.

## Component 3 — metrics

Per arm, reuse the existing `selection_frequency / weight_dispersion / mean_turnover /
mean_jaccard / metric_dispersion`, and add:

- **Overlap count (new):** mean pairwise count of shared held names, and as a fraction of mean held
  count — the intuitive "7 of ~9 names shared" read that makes "consistent vs still-random" legible
  at a glance. (Jaccard alone does not.)
- **Value guardrail:** the existing return / vol / Sharpe dispersion, reported per arm, to confirm
  Michaud does not degrade value while improving stability.
- **Michaud diagnostic:** the per-name across-draw selection-frequency + mean-raw-weight table (the
  conviction-gradient view, and the table read to decide whether a contingency lever is needed).

Output: a side-by-side `current vs michaud` summary block plus the raw CSVs per arm.

## Decision rule (criterion A)

Michaud **passes** (→ proceed to a productionisation phase) if, across M paired iterations:

- **Stability improves meaningfully:** turnover down by a clear margin and Jaccard / overlap count
  up — exact magnitudes read off the paired run, not pre-committed; **and**
- **The conviction core survives:** NUTRESA-tier names stay high-frequency; **and**
- **Value does not degrade materially:** per-arm Sharpe / return CoV no worse than the current
  method.

Stability that improves while value craters is a **fail**, not a pass — that would just be
disguised equal-weighting. If the empirical-ensemble mechanism (A) is "too far from an improvement,"
fall back to bootstrap (B′) then parametric Michaud (B).

## n=100 reproducibility check

The same harness run with `iterations=2–3, transformer_runs=100`, reported via the overlap count for
**both** arms. Doubles as (a) the production-budget sanity check on whether the baseline is actually
consistent at n=100 (if 2–3 runs share ~7 of ~9 names it is consistent; if only 2–3 of ~9, it is
almost as random as now), and (b) the production-budget read on whether Michaud consolidates the
book. A documented configuration to run, not a separate code path.

## Contingency levers (build only if the diffuse-consensus pathology appears)

If the per-draw weight / selection-frequency table shows too many **extreme-and-rare** draws (one
draw posts an extreme weight on a name no other draw likes, contaminating the average), reach for a
lever — diagnosing from that table first, in this priority order:

1. **Robust aggregation (lead lever).** Replace `mean_i(r_i)` with a **trimmed mean** (drop each
   name's top/bottom k draws) or per-name **median**. An extreme-in-one-draw, zero-elsewhere name
   washes out automatically, with no change to the optimiser or `μ`. Surgical and cheap; the most
   direct answer to the extreme-and-rare failure mode. (Median can over-prune names held in ~4/10
   draws; a light trimmed mean is gentler.)
2. **Tighten per-draw `max_weight`.** Raw `msr_tuned` already applies `max_weight` and has no min
   floor; lowering the cap forces each draw to spread across more names so no single draw can post
   an extreme corner bet. Risk: too tight → every draw looks equal-weighted, washing out conviction.
3. **μ-shrinkage per draw.** `μ' = (1−δ)·μ + δ·μ̄` toward the cross-sectional mean before optimising
   — less extreme `μ` → less extreme corner bets. The smooth version of capping forecasted returns;
   the canonical Bayes-Stein companion to Michaud.
4. **Per-draw batch elimination** (the comparison-arm toggle) and **a ridge / diversification
   penalty on weights** in the objective (touches `src/risk_kit.py`) — heavier fallbacks, last.

## Testing

Extend `tests/test_measure_allocation_stability.py` (torch-free via injected stubs):

- The per-draw raw-allocate path (raw `msr_tuned`, no elimination).
- The average-then-floor consensus: correct mean, single elimination pass, renormalisation,
  sum-to-1.
- The new overlap-count metric (including degenerate cases: identical books, disjoint books).
- That both arms consume the **identical** draws per iteration.
- The `eliminate_per_draw` toggle selects the comparison arm.

## Related

Phase 1 measurement, Phase 2 per-period `μ`, and the transformer-runs sweep specs are alongside this
file under `docs/superpowers/specs/`. Push-scope rule: experiments/tests/docs stay local; only
functional pipeline changes are pushed (productionisation phase).
