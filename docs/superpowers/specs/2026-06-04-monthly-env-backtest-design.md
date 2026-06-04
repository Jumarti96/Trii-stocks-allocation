# Monthly environment-robustness backtest — design

Date: 2026-06-04
Status: approved (pending spec review)

## Purpose

Test whether the allocation-method ranking — original **msr** (`current`) vs parametric Michaud
**s4** (deployed) vs **s8** — holds up in a genuinely different environment: **monthly** data at a
**short (1-month)** and a **long (6-month)** forecast horizon, with **equal_weight** as the
beats-naive-1/N anchor. This is *environment*-robustness (frequency + horizon), complementary to
the deferred multi-seed *seed*-robustness backtest (Plan 1), at roughly one fifth the compute.

Motivation: a method that only wins in the exact weekly / 4-period-horizon setup it was discovered
in is overfit; one whose ranking survives weekly→monthly and short→long horizon is genuinely
robust. The earlier single-seed weekly backtests also proved seed-sensitive, so this cheaper
cross-environment read is a higher-information next step than re-running the same weekly setup with
more seeds.

This is a **realized out-of-sample** backtest (the verdict is realized Sharpe etc.), not an
in-sample stability run.

## Scope

In scope:
- A one-time isolated monthly-data fetch (experiment-local; never touches production data/config).
- A driver that runs `backtest_allocation.run_backtest` for horizons {1, 6} × seeds {0, 100} with
  arms `current / (parametric, 4.0) / (parametric, 8.0) / equal_weight`, the **technical filter
  disabled**, and aggregates realized metrics across seeds (mean ± cross-seed spread) per horizon.

Out of scope (explicitly):
- The technical-indicator filter — **disabled** for this test (see "Filter disabled" below).
- A dedicated with-vs-without-filter A/B to inform the production filter-removal decision — that is
  a separate future experiment in the *weekly* production environment (filter-off here is
  confounded with the monthly change and is NOT clean evidence for removing the filter from prod).
- Multi-seed seed-robustness at the weekly production setup (Plan 1) — separate.
- Net-of-cost analysis — a free later follow-up on the saved turnover/weights, not in v1.
- Any change to `params.yaml`, `data/`, `pipeline/`, or `experiments/backtest_allocation.py`.

## Configuration

| Knob | Value | Rationale |
|---|---|---|
| frequency | monthly (`interval=1mo`, `periods_per_year=12`) | the different environment under test. |
| history | `days_of_data=3650` (~10 yr, same span as weekly) | controlled: only the frequency changes → ~120 monthly points. |
| horizons | `{1, 6}` months (`periods_to_forecast = rebalance_every = horizon`) | clean short-vs-long contrast; non-overlapping blocks matching the horizon. |
| oos_periods | `60` (~50% of ~120) | initial training ~60 months; blocks ≈ 60 (1-ahead) / ≈ 10 (6-ahead). Maximizes the scarce 6-ahead evaluations. Finalized against the real point count after fetch. |
| seeds | `{0, 100}` | monthly is noisier (fewer blocks); 2 seeds expose cross-seed spread cheaply. |
| arms | `current`, `(parametric, 4.0)`, `(parametric, 8.0)`, `equal_weight` | msr baseline + deployed s4 + aggressive s8 + the 1/N anchor. |
| n_runs | `75` | the Plan-2 chosen production n. |
| mc_draws `K` | `1000` | matches deployed; kept despite leaner data (more estimation uncertainty → keep draws smooth). |
| filter | **disabled** (`select_fn` returns all names) | see below. |

Same oos_periods/horizons for both horizons keeps them a controlled comparison (only
horizon/rebalance differs).

## Filter disabled

The technical filter is dropped for this test via a `select_fn` that returns every column
(`lambda prices, rets, cfg: list(rets.columns)`). No change to `run_backtest` — it already accepts
`select_fn` as a dependency-injection seam. Reasons:
- The filter's MA/MACD windows are in *periods*, so at monthly frequency `MACD(12,26,9)` is a
  multi-year gate needing ~35 months of warm-up — it would behave erratically on data-starved
  monthly history and waste early periods. Disabling it removes that confound and cleanly isolates
  the allocation methods on the full forecast set.
- Aligns with the standing direction to remove the filter from the pipeline later.

Consequences (all minor/expected): the eligible universe per rebalance is the full ~83 names (vs
~60–69 selected); msr/parametric still prune via the min-weight elimination loop so `n_held` stays
sensible; `equal_weight` becomes 1/N over the full universe (a more naive anchor); optimization
cost rises ~20% (each msr solves over ~83 vs ~60 names). The filter-removal *production* decision
is NOT settled by this run (confounded with the monthly change) — it needs its own weekly A/B.

## Approach

Two new experiment files plus tests; `backtest_allocation.py` and the pipeline are reused
unchanged.

### `experiments/fetch_monthly_data.py`

Reuses `01_download.py`'s yfinance download + clean logic (same tickers from `stock_tickers/*.csv`,
`days_of_data=3650`, but `interval=1mo`, `period_freq=M`) and writes
`experiments/data_monthly/01_prices.csv` + `01_returns.csv`. The ~15 lines of download/clean logic
are duplicated here (not refactored out of the pushed `01_download.py`) to keep the experiment
fully local and leave the production step untouched. Run once; reused across all horizons/seeds.
Prints the actual monthly point count, the universe size, and the overlap with the weekly 83-name
universe (a sanity check that yfinance monthly availability is comparable).

### `experiments/monthly_env_backtest.py`

- `build_monthly_cfg(weekly_cfg, horizon)` — returns a copy of `load_config()` with `interval=1mo`,
  `periods_per_year=12`, `period_freq=M`, `future_freq=MS`, `date_offset=1 month`, `time_window=12`,
  `periods_to_forecast=horizon`, and `rf_period=(1+rf_rate)**(1/12)-1`. Filter/MA/weight params
  unchanged (the filter is bypassed via `select_fn`, not via config).
- Core loop: for `horizon in [1, 6]`, for `seed in [0, 100]`, call
  `run_backtest(prices_m, rets_m, build_monthly_cfg(cfg, horizon), oos_periods=60,
  rebalance_every=horizon, n_runs=75, mc_draws=1000, spreads=[4.0, 8.0], seed,
  arms=["current", ("parametric", 4.0), ("parametric", 8.0), "equal_weight"],
  select_fn=<all-names>)`.
- Aggregate the realized metrics across the two seeds per (horizon, arm) as **mean ± cross-seed
  std**. Realized metrics reuse `backtest_allocation`'s existing computations (realized Sharpe,
  cumulative & annualized return, annualized vol, max drawdown, mean turnover, hit rate, avg
  n_held), with `blocks_per_year = 12 / horizon` and the per-block rf.

### Outputs (`experiments/results/monthly_env/`)

- Per-(horizon, seed) raw artifacts via the existing `write_backtest_outputs`
  (block returns, weights, turnover), under a `h{horizon}_seed{seed}/` subdir.
- A per-horizon cross-seed table: rows = arms, columns = the realized metrics, each `mean ± std`.
- `monthly_env_summary.txt`: the two per-horizon tables, with **block counts reported prominently**
  per horizon and the caveat that the 6-ahead read (~10 blocks) is directional, not a verdict; plus
  the explicit equal_weight comparison (does any optimizer arm beat 1/N in this regime).

## Expected behavior (design notes, not bugs)

- **Wider parametric draws at monthly.** `run_backtest` passes `len(rets_hist)=T` as the `Σ/T`
  scale; monthly T (~60–115) is far smaller than weekly (~500), so `Σ/T` is larger and s4/s8 draw
  more dispersed mu vectors — i.e. they hedge more under the greater estimation uncertainty. This is
  correct Michaud behavior and part of the environment being tested.
- **Noisy forecasts.** ~60 months of initial training is thin; forecasts will be weak. The
  equal_weight anchor is how we judge whether the forecasts add any value at all — a regime where
  optimizer arms fail to beat 1/N is itself an informative result.
- **Few 6-ahead blocks.** Non-overlapping 6-month blocks over a ~60-month OOS give ~10 evaluations;
  realized Sharpe there is coarse. Reported, not hidden; read as directional.

## Testing

Torch-free via dependency-injection seams (stub `runs_fn` returning deterministic per-run arrays),
mirroring `tests/test_backtest_allocation.py`. Cover: `build_monthly_cfg` sets ppy/time_window/
rf_period/periods_to_forecast correctly; horizon drives `rebalance_every`; the four-arm set incl.
`equal_weight` is wired; the `select_fn` override yields the full universe (no filtering); cross-seed
aggregation reports mean ± std; block counts are computed and surfaced; output writers produce the
expected files. The realized-metric helpers themselves are already covered by
`tests/test_backtest_allocation.py`.

## Cost

- Monthly training is fast (~12-period sequences, ~10× smaller dataset → ~2 s/train vs ~9 s weekly).
- Per seed: 1-ahead (~60 rebalances): ~training 60×75×2 s ≈ 2.5 h + opt 60×2×K1000 ≈ 2 h. 6-ahead
  (~10 rebalances): ~0.7 h. ≈ **5 h/seed**.
- Two seeds → **~10–11 h total** — a single background/overnight run. The K=1000 optimization is the
  dominant term (training got cheap); confirm the ~2 s/train estimate in the smoke run.

## Conventions

Experiment-only and **local** per the push-scope rule (new files only; no pipeline edits). Run in
the background. Results land in the gitignored `experiments/results/monthly_env/`; monthly data in
the (also local) `experiments/data_monthly/`.
