# Weekly multi-seed robustness backtest (Plan 1) — design

Date: 2026-06-05
Status: approved — to be implemented and RUN in a later session.

## Purpose

The seed-robust realized verdict on the deployed allocation, in the **weekly production
environment**: does parametric Michaud **s4** (and **s8**) robustly beat the original **msr**
(`current`) and the naive benchmarks (**equal_weight**, **sp500**), averaged over multiple seeds
so the verdict is not a single-seed artifact. Mirrors the two-horizon structure of the monthly
1B run but in the weekly (production) regime, with cheap post-hoc analytics that target the
recurring "is the gap real or noise?" question.

Context: s4 is the deployed default (PR #22), reframed (2026-06-05) as a **risk-efficiency** play
(lower vol + ~40% lower turnover), NOT robust alpha — three prior stress tests (weekly oos=250
non-confirmation, seed-sensitivity, monthly 1B reversal) all weakened the alpha story. This run is
the definitive seed-robust weekly read; it stays an experiment (does not change the deployment).

## Scope

In scope:
- A weekly walk-forward backtest over horizons {4, 24} × seeds {0, 100, 200}, arms
  current / s4 / s8 / equal_weight / sp500, filter disabled, reusing `run_backtest` unchanged.
- An SP500 (`^GSPC`) benchmark spliced in as a do-nothing arm.
- Post-hoc analytics on the saved results: net-of-cost (0/10/25 bps) + break-even bps for s4-vs-msr,
  and a paired block-bootstrap significance test on the (s4 − msr) realized difference.
- Cross-seed aggregation (mean ± cross-seed std) per horizon.

Out of scope:
- Any change to the deployed pipeline / `params.yaml` / `data/` / `backtest_allocation.py`.
- A filter-ON comparison sweep (would ~double compute; the filter-off choice is deliberate and
  forward-looking — see [[project-filter-removal]]). The filter-off/prod-filter-on mismatch is a
  noted caveat, not closed here.
- A baked-in early/recent regime split — left as a free post-hoc on the saved block-returns CSV.
- Re-tuning s for any environment; productionising any result.

## Configuration

| Knob | Value | Rationale |
|---|---|---|
| environment | weekly production cfg (`load_config()`) | the deployment's real regime. |
| filter | **disabled** (`select_all` → all names) | forward-looking + matches monthly 1B; full ~83-name universe. |
| horizons | `{4, 24}` weekly (`periods_to_forecast = rebalance_every = horizon`) | short (~1mo, production, ~62 blocks) + long (~½yr, ~10 blocks, directional). |
| oos_periods | `250` | the longer/earlier window incl. the hard 2022 regime — protected (it IS the robustness); not reduced for compute. |
| seeds | `{0, 100, 200}` | 3 = the seed-robustness floor; base seed k offsets the per-rebalance stream (`seed+k`). |
| arms | current, (parametric,4.0), (parametric,8.0), equal_weight, **sp500** | msr baseline + s4(deployed) + s8(aggressive) + 1/N anchor + market anchor. |
| n_runs | `75` | the Plan-2 pick (kept; robust, no compute compromise). |
| mc_draws `K` | `1000` | deployed value. |
| trading cost | gross + net at {10, 25} bps + break-even | tests s4's surviving lower-turnover claim (post-hoc, free). |

Compute ≈ **48 h** (training ≈40 h dominates: (62+10) rebalances × 75 runs × 3 seeds; optimization
≈8 h: 2 parametric arms × K × rebalances × 3 seeds). The analytics add negligible compute.

## Approach

New files; `backtest_allocation.py` and the pipeline are reused unchanged.

### `experiments/fetch_sp500_weekly.py`

Fetch `^GSPC` weekly adjusted close for the production window, compute weekly returns, write
`experiments/data_weekly_bench/sp500_returns.csv` (a single-column Series of period-end-string
indexed weekly returns). Isolated (never touches `data/`). A pure `weekly_returns_from_close`
transform (DatetimeIndex → period-end string index → pct_change) is unit-tested; the yfinance call
and `main` are smoke-verified. Run once.

### `experiments/weekly_robust_backtest.py`

- `build_weekly_cfg(weekly_cfg, horizon)` — copy of `load_config()` with `periods_to_forecast =
  horizon` (everything else stays weekly). For horizon 4 this equals production; for 24 it is the
  long-horizon variant.
- `select_all(prices, rets, cfg)` — filter-off seam (every name eligible). (May be imported from
  `monthly_env_backtest` to avoid duplication, or re-declared; implementer's call.)
- `sp500_block_returns(sp500_rets, rets_index, rebalance_index, rebalance_every)` — align the SP500
  weekly returns to the universe's date index (reindex by position/date) and compute the compounded
  block return for each rebalance block. Returns a list aligned to the backtest's blocks. This is a
  pure function (testable without network).
- `splice_sp500_arm(results, sp500_block_rets, dates)` — add an `sp500` label to a `run_backtest`
  result dict with `block_returns = sp500_block_rets`, `turnover = [None, 0, 0, ...]`, `n_held = [1,
  ...]`, `weights = []` (benchmark, no portfolio weights). So it flows through the existing
  aggregation/summary as a fifth arm.
- `run_weekly_robust(prices, rets, sp500_rets, weekly_cfg, horizons, seeds, oos_periods, n_runs,
  mc_draws, spreads, runs_fn=..., ...)` — for each horizon × seed: call `run_backtest` (arms
  current/s4/s8/equal_weight, `select_fn=select_all`), then splice the sp500 arm. Returns
  `{horizon: {"cfg": cfg_h, "per_seed": {seed: results}}}`.
- `net_of_cost_table(per_seed, cfg, rebalance_every, bps_levels)` — per arm, recompute realized
  Sharpe with each block return reduced by `turnover * bps/1e4`, at each cost level, averaged across
  seeds. Also `break_even_bps(s4, msr)` = the cost at which s4's net Sharpe equals msr's (linear
  interpolation / root-find on the net-Sharpe-vs-bps difference; `None` if s4 never overtakes).
- `paired_bootstrap(per_seed, arm_a, arm_b, n_boot, rng)` — pool the per-block paired realized
  differences `(arm_a_block − arm_b_block)` across seeds (arms share the per-rebalance forecast, so
  blocks are paired), block-bootstrap-resample them `n_boot` times, and report the mean difference,
  a 90% CI, and `P(mean diff > 0)`. Pure given the results + an injected rng.
- Reuse `aggregate_across_seeds` + `write_backtest_outputs` (and the mean±std table idea) from
  `monthly_env_backtest` / `backtest_allocation`; a `format_weekly_summary` renders per-horizon
  tables, the net-of-cost + break-even block, and the bootstrap block, with all caveats.

### Outputs (`experiments/results/weekly_robust/`)

- Per-(horizon, seed) raw artifacts via `write_backtest_outputs` in `h{horizon}_seed{seed}/`.
- Per-horizon cross-seed aggregated table CSV (mean ± std per arm).
- `weekly_robust_summary.txt`: per-horizon metric tables (incl. sp500), the net-of-cost table +
  break-even bps, the paired-bootstrap (s4−msr) result, block counts, and the caveats below.

## Caveats (in the summary)

- **SP500 currency/composition mismatch** — `^GSPC` is USD; the universe is global/Colombian mixed
  currency. SP500 is a rough "beat the US market" anchor, not like-for-like; equal_weight is the
  cleaner same-universe naive anchor.
- **Long horizon is thin** — 24-week blocks over oos=250 give ~10 evaluations (directional), and a
  24-step weekly forecast strains a model tuned for 4-step paths.
- **Filter-off ≠ production** — production currently runs the filter ON; this run is forward-looking
  and not a clean filter-removal evaluation (that needs its own with-vs-without A/B).
- **3 seeds is the floor** — mean ± std from 3 points is coarse; the paired bootstrap is the firmer
  significance read.

## Testing

Torch-free via dependency-injection seams (stub `runs_fn`/`period_mu_fn`/`seed_fn`), mirroring the
sibling experiment tests. Cover: `build_weekly_cfg` (only periods_to_forecast changes); `select_all`;
`sp500_block_returns` alignment + compounding; `splice_sp500_arm` shape; `run_weekly_robust`
structure (5 arms incl. sp500, rebalance_every = horizon); `net_of_cost_table` + `break_even_bps`
(monotone net-Sharpe, a known crossover); `paired_bootstrap` (deterministic with a seeded rng on a
constructed paired difference); output writers. The `weekly_returns_from_close` transform is tested
without network.

## Conventions

Experiment-only and local per the push-scope rule (new files only). The big run is launched in the
background in a later session (~48 h). Results in the gitignored `experiments/results/weekly_robust/`;
SP500 data in `experiments/data_weekly_bench/`.
