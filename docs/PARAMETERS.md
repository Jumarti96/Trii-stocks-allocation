# Parameter reference

Every key in `params.yaml`, what reads it, and what it actually does. Ordered to match
the file. Values shown are the current defaults.

Rules of thumb before the tables:

- **Only `params.yaml` is loaded.** `pipeline/config.py::load_config` reads that one
  path. `params_trii_10y.yaml.disabled` is an inert snapshot — the suffix is what keeps
  it from being picked up, the same convention as `stock_tickers/*.csv.inactive`.
- **Do not add derived keys.** `period_freq`, `date_offset`, `future_freq` and
  `rf_period` are computed in `load_config` (`config.py:134-139`). `time_window` is the
  exception: it must be *present* (a bare `cfg['time_window']` lookup) but may be `null`.
- **Most parameters are unvalidated.** Only the transformer, universe and report blocks
  have checks. A typo in `allocation_method` fails at dispatch; a bad `max_weight` fails
  nowhere and quietly produces a book you did not intend.

---

## Your setup

| Parameter | Default | Read by | Notes |
|---|---|---|---|
| `investment` | `120000000` | step 4 | Total capital, in `report_currency`. Renamed from `investment_cop`, which now **raises** (`config.py:104`). |
| `report_currency` | `COP` | step 4 | 3-letter ISO, upper-cased on load. Prices, amounts and share counts are converted into it. |
| `rf_rate` | `0.11` | `config.py` → `rf_period`, step 4 | Risk-free rate **in `report_currency`** (0.11 = 10Y Colombian bond). Change it whenever you change the currency. |
| `universe_topn` | `500` | step 2 | How many stocks the model forecasts, ranked by FX-normalised dollar volume. `null` disables. No-op if the catalogue is smaller. |

**`universe_topn` is one of the three parameters that measurably changed results.** Do
not exceed ~600: `transformer_model.capacity_report` shows the forecast over-disperses
badly past that. At 300 the screen cut straight through the European large caps
(TotalEnergies 303, Siemens 327, LVMH 351, UBS 407), leaving an 89%-USD universe — run
`experiments/universe_profile.py` before changing it.

## Data and timing

| Parameter | Default | Read by | Notes |
|---|---|---|---|
| `periods_per_year` | `54` | steps 2-4, `rf_period` | 52 weekly, 12 monthly. |
| `interval` | `"1wk"` | step 1 | `"1wk"` or `"1mo"`; also selects `period_freq`/`future_freq`. |
| `days_of_data` | `5475` | step 1 | Download window. 15y. |
| `periods_to_forecast` | `24` | steps 2, 4 | Forecast horizon. **Must equal `transformer_forecast_window` when `transformer_loss: rank_ic`.** |

**`days_of_data` is the second parameter that changed results, and it has a trap.**
`clean_batch` drops any name missing more than 15% of the window (`missing_frac=0.15`,
not configurable), so a longer window silently deletes younger companies. Measured: at
20y, 45 of the top 300 vanished — TSLA, META, NOW, BABA, PANW, ABBV, SHOP, ANET — the
entire post-2009 IPO cohort. 15y keeps TSLA (2010) and META (2012) and loses only
post-2013 listings. Longer calendar buys walk-forward windows (13 → 20 → 30) but worsens
survivorship bias.

## Universe screen (step 2)

Picks *which* stocks to model, before forecasting, on **return-neutral criteria only**.
Never screen on past returns — see `src/data_intake.select_universe` for why.

| Parameter | Default | Notes |
|---|---|---|
| `universe_strata` | `null` | `null` = pure top-N (mega-caps only at scale). A list spreads the budget over equal liquidity bands, e.g. `[200, 60, 40]`. **Must sum exactly to `universe_topn`** (`config.py:90`). |
| `universe_price_floor` | `0.0` | Drop prices below this, in listing currency. Unvalidated. |
| `universe_min_market_cap` | `null` | Size floor **in USD**; caps are converted out of the listing currency first. ETFs report no cap and are **kept**, not dropped. |
| `unknown_currency` | `exclude` | `exclude` \| `assume_target`. What to do with a stock whose quote currency will not resolve. |

`exclude` is the right default: guessing is how a JPY-quoted name gets ranked ~150× too
high. `assume_target` exists so one unresolvable ticker cannot abort a run, but it trades
a visible gap for an invisible error.

## Portfolio optimisation (step 3)

| Parameter | Default | Notes |
|---|---|---|
| `allocation_top_n` | `150` | Candidates fed to the optimiser. `null` = no cap. **Applied only to forecast-based methods** — it ranks on the model's forecast (by `allocation_ranking`), so applying it to a model-free method would make that method quietly model-dependent. |
| `allocation_ranking` | `sharpe` | `sharpe` ranks by **mu/σ** — the forecast return divided by that stock's own volatility (covariance *diagonal* only, so correlations are ignored). `return` ranks by raw mu. Used by `allocation_top_n` and by `equal_weight_topn`. This per-stock score is **not** the portfolio Sharpe that `msr`/`parametric_michaud` maximise, which does use the full covariance. |
| `allocation_method` | `parametric_michaud` | See the table below. |
| `equal_weight_n` | *commented out*; defaults to `1/min_weight` = 20 | Names held by `equal_weight_topn`, `momentum` and `random`. The comment in `params.yaml` historically claimed 12; the real default is 20. |
| `momentum_lookback` | `24` | Periods of trailing return that `momentum` ranks on. |
| `michaud_spread` | `2.0` | Draw scale `s` on `Σ/T`. |
| `michaud_mc_draws` | `1000` | Monte-Carlo draws. The dominant cost of a Michaud run. |
| `michaud_seed` | `0` | `int` = reproducible; `null` = fresh draws. Also seeds `random`. |
| `max_weight` | `0.15` | Per-name cap. Implies a book of at least `ceil(1/max_weight)` names. |
| `min_weight` | `0.05` | Per-name floor. **Implies a book of at most `1/min_weight` = 20 names**, whatever the method. |

### The eight allocation methods

| `allocation_method` | Uses mu? | Needs | What it does |
|---|---|---|---|
| `parametric_michaud` | yes | mu + cov | Default. Draws K mu vectors, max-Sharpe per draw, averages the weights. |
| `msr` | yes | mu + cov | Legacy max-Sharpe with a batch-elimination loop. |
| `equal_weight_topn` | yes | mu + cov | Equal weights on the top `equal_weight_n` by `allocation_ranking`. |
| `equal_weight_all` | no | — | 1/n across the universe. The do-nothing baseline. |
| `gmv` | no | cov | Global minimum-variance. If it matches the model, the forecast adds nothing. |
| `inverse_vol` | no | cov | Naive risk parity; ignores correlations. |
| `momentum` | no | **returns panel** | Best trailing performers over `momentum_lookback`. |
| `random` | no | — | **A control, not a strategy.** Step 3 prints a warning. |

> **The production forms are not the backtested forms.** Every method ends with the
> `min_weight` floor, which admits at most `1/min_weight` positions. Measured on the
> 80-stock snapshot: `gmv` floors from 80 names to 11, `inverse_vol` to 17. Backtested
> `gmv` spreads across every name at well under 1% each, so **production `gmv` is "the 20
> largest GMV weights" and will not reproduce the backtest numbers.** `equal_weight_topn`
> diverges too: production ranks by `sharpe` over 20 names, the backtest by raw `mu` over
> ~10. This is deliberate — a 150-name book is untradeable at this account size — but do
> not read backtest results as predictions for the production book.

## Output (step 4)

| Parameter | Default | Notes |
|---|---|---|
| `output_path` | `results/allocation_output.csv` | Relative to the project root. Mandatory. |

## Advanced: pipeline internals

Calibrated defaults. Do not change the transformer block without reading
`src/transformer_model.py`.

| Parameter | Default | Notes |
|---|---|---|
| `transformer_arch` | `B` | `current` (autoregressive) \| `B` (direct multi-step). |
| `transformer_loss` | `rank_ic` | `rank_ic` \| `auto` (legacy MSE/Huber). Validated against that set. |
| `transformer_forecast_window` | `24` | Decode length; `null` → `periods_per_year`. Must be `>= periods_to_forecast` for `B*` archs. |
| `n_transformer_runs` | `150` | Independent runs, averaged. Linear in training time. |
| `time_window` | `null` | Input sequence length; `null` → `periods_per_year`. **Key must exist**, value may be null. |
| `transformer_epochs` | `50` | Epochs per run. |
| `transformer_warmup_epochs` | `5` | LR linear-warmup epochs. |
| `transformer_lr` | `0.0001` | Peak Adam learning rate. |
| `transformer_batch_size` | `32` | Mini-batch size. |
| `winsorization_lower_pct` | `1` | Lower percentile for forecast winsorisation. |
| `winsorization_upper_pct` | `99` | Upper percentile for forecast winsorisation. |
| `batch_size` | `500` | Tickers per download batch. |
| `download_workers` | `3` | Parallel batch workers. Also the default concurrency for the `.info` pass — **this is what trips Yahoo's rate limiter** on a 3k catalogue. |
| `download_timeout` | `30` | Per-batch yfinance timeout (s). |
| `download_warn_fraction` | `0.80` | Warn if fewer than this share of tickers download. |
| `zero_volume_warn_threshold` | `0.25` | Warn if more than this share report zero volume. |
| `liquidity_window_fraction` | `0.10` | Activity lookback = this fraction of the series (min 10 periods). |
| `liquidity_min_active_fraction` | `0.85` | Keep stocks trading in at least this share of that window. |

---

## Cross-parameter constraints

These are the only ones enforced. Everything else fails late or not at all.

| Constraint | Raised by |
|---|---|
| `transformer_loss` ∈ {`auto`, `rank_ic`} | `config.py:46` |
| `transformer_forecast_window >= periods_to_forecast` (for `B*` archs) | `config.py:52` |
| `periods_to_forecast == transformer_forecast_window` when `transformer_loss: rank_ic` | `config.py:61` |
| `universe_strata` sums exactly to `universe_topn` | `config.py:90` |
| `universe_topn` is a positive int or `null` | `config.py:87` |
| `unknown_currency` ∈ {`exclude`, `assume_target`} | `config.py:114` |
| `report_currency` is a 3-letter code | `config.py:118` |
| `investment_cop` present → error explaining the rename | `config.py:104` |

**Gap worth knowing:** the `rank_ic` horizon check is nested inside the `B*` branch, so
`transformer_loss: rank_ic` with `transformer_arch: current` is **not validated** and
will silently optimise a horizon production never consumes.

## Hidden options — real, but absent from `params.yaml`

| Key | Effect |
|---|---|
| `use_gradient` | `parametric_michaud` only. Analytic Sharpe gradient instead of finite differences: ~4.4× faster on an 80-stock problem. Left out so production keeps the historical path; worth setting for bulk sweeps. |
| `equal_weight_n` | Present but commented out. See the allocation table. |

## Constants that behave like parameters but are not configurable

| Constant | Where | Why it matters |
|---|---|---|
| `missing_frac=0.15` | `data_intake.clean_batch` | Decides which names survive a long `days_of_data`. The reason a 20-year window drops TSLA and META. |
| `lambda_=0.2` | `transformer_model.weighted_mean_return` | Exponential decay collapsing the forecast path into a single expected return. |
| `CAPACITY_WARN` / `CAPACITY_ERROR` | `transformer_model` | 4,000 / 10,000 params-per-sample. Step 2 refuses to run on `error`. |
| `MOMENTUM_LOOKBACK`, `MIN_TRAIN`, `COST_ROUND_TRIP`, `N_RANDOM` | `experiments/backtest_isin.py` | Backtest-only; `MIN_TRAIN` is exposed as `--min-train`. |

## What actually matters

Across four walk-forward runs, three parameters changed the results and one did not.

**Changed the answer:** `universe_topn` (500 vs 300 reordered every strategy),
`days_of_data` (10y vs 15y moved every level and every p-value), `allocation_method`.

**Did not:** `michaud_spread`. Three runs produced three different orderings, with every
spread taking both first and last place — see the table in
`src/allocation.resampled_michaud`. It stays at 2.0 because nothing has displaced it, not
because it has been established.

A caution that applies to all of the above: with 13–20 windows the study is underpowered
for effects this size (`n_for_80_power` runs 39–392 windows). Parameters whose measured
effect is small are not distinguishable from noise, so re-tuning them against a single
run is how a spurious calibration gets locked in.
