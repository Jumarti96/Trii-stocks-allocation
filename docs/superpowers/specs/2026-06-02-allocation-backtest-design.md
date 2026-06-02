# Allocation Walk-Forward Backtest — Design

## Problem

Phases 1-3b established, **in-sample**, that composition is unstable and that parametric Michaud
trades stability for value along a tunable frontier (sweet spot ≈ s=2: overlap 33%→43% for ~1%
Sharpe). But every value number so far is **in-sample forecast-Sharpe**, scored against the averaged
forecast μ̄ — and the current method *is* the Sharpe-argmax on that μ̄, so the comparison is
structurally rigged in its favour. That metric cannot rank the methods on what actually matters:
**realized, out-of-sample performance.** Juan's standing constraint: he will not adopt a costlier
method (Michaud's extra optimizations, or any retraining) unless realized, net-of-cost performance
clearly earns it — especially at a larger stock universe. He was also explicitly unconvinced by the
s=2 knee because it rested on only 3 in-sample iterations.

## Goal

Build a **walk-forward backtest** that scores each allocation method on **realized out-of-sample
returns**, to decide which is closest to reality. Local-only experiment, same push-scope rule as
Phases 1-3b ([[feedback-push-scope]]). Productionising any winner into `pipeline/04_allocate.py`
remains a separate later phase.

## Settled decisions (from brainstorming)

- **Rebalance cadence:** every **4 periods** (≈monthly) — matches the model's `periods_to_forecast=4`
  horizon; the native cadence and the sharpest turnover-cost test.
- **Out-of-sample window:** last **~3 years** (~162 weekly periods → ~40 non-overlapping monthly
  blocks → ~40 realized observations per arm — a real sample, unlike the 3 in-sample iterations).
- **Runs per rebalance:** **50** (μ̄ reasonably de-noised at ~half the cost of the production 100).
- **Training window:** **expanding** (all history up to `t`) — matches how the pipeline trains today.
- **Transaction costs:** **frictionless headline (0 bps)** first. Triage logic: if a Michaud arm holds
  up on *gross* realized return, its lower turnover only strengthens the net case; if it is much worse
  gross, costs cannot save it. Turnover and per-rebalance weights are **recorded regardless**, so
  net-of-cost analysis is a free post-process later (no re-run).
- **Arms (6):** `current`, `parametric s=1/2/4`, `empirical Michaud`, `equal-weight`. All Michaud arms
  use **deferred elimination** (raw `msr` per draw → average weight vectors → one `min_weight` floor
  at the end). Equal-weight = `1/N` over the eligible selected set (the disguised-indexing check).
- **Explicitly out of scope (v1):** transaction-cost level sweep (deferred, data recorded), a
  `μ`-shrinkage arm, the per-draw-elimination Michaud variant (`eliminate_per_draw=True`), and
  intra-block weight drift in the turnover calc.

## Architecture — walk-forward loop

The expensive `train_runs` happens **once per rebalance**; all arms reuse that rebalance's 50 runs,
μ̄, and Σ. Stepping `t` every 4 periods through the OOS window:

```
returns[:t] ─► train_runs(50) ─► 50 per-run μ ─► μ̄_t (weighted_mean_return)
                                      │                  │
prices[:t]  ─► select_stocks (gate) ─► eligible set   Σ_t = Ledoit-Wolf(returns[:t])
                                      │                  │
   each ARM builds w_t from the SAME μ̄_t / 50 runs / Σ_t over the eligible set
                                      │
   hold w_t over t … t+4 (buy-and-hold) ─► realized block return = w_t · (actual asset returns)
                                      │
   record: w_t, realized block return, turnover vs previous target weights
```

- **No lookahead:** the forecast, filter, and Σ at rebalance `t` use only data with index `≤ t`; the
  realized return uses only data in `(t, t+4]`.
- **Selection filter re-run each rebalance** on data up to `t`, so the eligible universe evolves
  realistically. Every arm — including equal-weight — allocates over that *same* eligible set, making
  it an apples-to-apples "did the optimizer beat naive 1/N over the same names" test.
- **Realized return** = buy-and-hold the target weights over the next 4 periods using *actual*
  returns → one realized observation per block (~40 per arm).
- **Turnover** = `½·Σ|w_t − w_{t-1}|` between consecutive **target** weights (intra-block drift
  ignored — standard v1 simplification).

## Components

1. **`backtest_allocation.py`** *(new — the walk-forward driver, realized-return accounting, CLI).*
   Imports and reuses, with no duplication:
   - `train_runs`, `winsorize_to_history`, `weighted_mean_return` (from `transformer_model`),
   - `select_stocks`, `allocate_msr`, `resampled_allocate`, `sample_mu_draws`, `portfolio_metrics`
     (from `measure_allocation_stability`).
2. **Arms** — each maps `(μ̄_t, 50 per-run μ, Σ_t, eligible set)` → a target-weight Series:
   - `current`: `allocate_msr(μ̄_sel, Σ_sel, cfg)`
   - `parametric s`: `resampled_allocate(sample_mu_draws(μ̄_sel, Σ_sel, T=t, K, s, rng), Σ_sel, cfg)`
   - `empirical`: `resampled_allocate([50 per-run μ_i over selected], Σ_sel, cfg)`
   - `equal_weight`: `1/N` over the eligible selected set.
   `K` (parametric draws) defaults to 1000. The 3 parametric arms × K × ~40 rebalances are the
   second cost driver after retraining; `K` and the s-count are the easy levers if wall-time bites.
3. **Metrics** (per arm, over the ~40 realized blocks): cumulative compound return, annualized return,
   annualized vol, **realized Sharpe** (the verdict), max drawdown, mean per-rebalance turnover,
   average # names held, hit rate (% blocks positive).

## Outputs

To `experiments/results/backtest/` (gitignored):
- `backtest_returns.csv` — per-rebalance realized block return, one column per arm.
- `backtest_weights_<arm>.csv` — per-rebalance target weights per arm (so net-of-cost is a free
  re-run).
- `backtest_turnover.csv` — per-rebalance turnover per arm.
- `backtest_summary.txt` — the headline table (one row per arm): cum return, ann return, ann vol,
  realized Sharpe, max DD, mean turnover, avg #names, hit rate.

**CLI:** `--oos-periods`, `--rebalance-every`, `--n-runs`, `--mc-draws`, `--spreads`, `--seed`,
`--outdir`, defaulting to the settled values (~3yr / 4 / 50 / 1000 / "1,2,4" / 0 / the backtest dir).

## The reads this produces

1. Does any Michaud arm match/beat **current** on *gross* realized Sharpe? (If yes, lower turnover
   makes its net case even stronger.)
2. How does each arm compare to **equal-weight**? (The "is Michaud just disguised 1/N?" check.)
3. Turnover ranking across arms (the input to the deferred cost analysis).

## Testing

Torch-free, same injected-stub style as `tests/test_measure_allocation_stability.py`:
- **Walk-forward indexing:** correct number of non-overlapping blocks over the OOS window; a test that
  deliberately verifies no future row (index `> t`) leaks into a rebalance's forecast/filter/Σ inputs.
- **Realized accounting:** block return `= w · actual` computed correctly; turnover `= ½·Σ|Δw|` between
  consecutive target weights; `equal_weight` = `1/N` over the eligible set.
- **Arm wiring:** with injected stub forecasts, all six arms produce valid weight series (sum to 1 over
  held names) and the summary table has the expected rows/columns.
- **Metrics helpers:** annualized return/vol/Sharpe and max-drawdown verified on a hand-computed toy
  equity curve.

The slow real walk-forward (torch on the full universe) is validated by the actual run, not unit
tests — same pattern as the in-sample harness.

## Related

- In-sample harness + phases: `experiments/measure_allocation_stability.py`; specs/plans under
  `docs/superpowers/` for Phases 1, 2, the runs sweep, Phase 3 (empirical Michaud) and Phase 3b
  (parametric Michaud).
- Push-scope rule [[feedback-push-scope]]: experiments/tests/docs stay local; only functional pipeline
  changes get pushed (the later productionisation phase).
