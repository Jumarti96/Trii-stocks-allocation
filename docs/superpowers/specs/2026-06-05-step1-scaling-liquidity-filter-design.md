# Step-1 scaling + activity liquidity filter (PR-1) — design

Date: 2026-06-05
Status: approved (pending spec review)

## Revision note (2026-06-05)

PR-1 was first built with a **relative dollar-volume within market groups** filter (keep ≥ X% of
the market-group median). Calibration on the real data showed it **over-excludes well-traded
names**: because each market group contains a few giants, the group median is enormous (~$6B/week),
so "10% of median" lands at a ~$630M/week bar and drops genuinely liquid stocks (BAP $536M/wk, JETS,
BVN, the Colombian names) purely for being small *relative to the mega-caps in their group*. A
relative-rank rule always trims the tail of every group — it "excludes for the sake of it," the
exact thing to avoid. KEY REFRAME: the deployed capital is ~115M COP (~$28k), max_weight 0.15 → a
single position is ≤ ~$4k, so liquidity *magnitude* is nearly irrelevant; the only thing that makes
a stock genuinely hard to trade is that it **rarely trades at all**. This revision replaces the
relative-magnitude filter with an **activity (frequency-of-trading) filter** — currency-free,
grouping-free, and aligned with the actual goal. The download/parallel-batch machinery (already
built) is unchanged. A price-performance ("down >90% from peak", the old PRC signal) screen was
considered and **rejected**: it is return-based universe selection (survivorship/look-ahead bias —
the same class we removed the technical filter to avoid) and overlaps the activity filter for true
junk.

## Purpose

Rework pipeline step 1 so the allocation model works at **4,000+ stocks**, not just ~123:
(a) download at scale via **parallel batches** fetching **Close + Volume**, and (b) prune the
universe **early** (before training) with an **activity-based junk filter** that drops stocks which
rarely trade (and the existing bad-data drop). Plus a **local calibration experiment** that sweeps
the filter threshold over the real ticker sources so the production default is chosen empirically.

Driver: the universe is moving from 123 tickers (`colombia_stocks_trii.csv` 36 +
`global_stocks_trii.csv` 87) to a 3,918-ISIN list (`isins_list.csv`). Pruning genuinely-inactive
names early keeps the surviving set sane for training/optimisation.

## Scope

This is **PR-1** of a two-part effort.

In scope (PR-1):
- **Production (pushed):** refactor `pipeline/01_download.py` into testable functions — robust
  ticker loading, parallel batch download of Close+Volume, the activity filter, and pruned outputs;
  new `params.yaml` keys; unit tests shipped with the code.
- **Local (not pushed):** `experiments/calibrate_liquidity_filter.py` — threshold-sweep over the
  ticker sources to pick the production default.

Out of scope:
- **PR-2 (separate later cycle):** removing the technical-indicator filter (delete step 3) and
  rewiring step 4 to optimise over the pruned universe. PR-1 leaves steps 2–4 working unchanged
  (they read the now-pruned `01_*` outputs).
- ISIN→ticker resolution (yfinance reads ISINs directly).
- Any liquidity-*magnitude* filtering, currency normalisation/FX, or market grouping (rejected — the
  activity filter is currency-free; magnitude is irrelevant at the deployed position size).
- A price-performance / drawdown-from-peak screen (rejected — return-based selection bias).
- **Compute-bounding of the surviving universe** — calibration showed an activity/threshold filter
  does not aggressively bound the count; that is a top-N cap, deferred to PR-2 (step 4 is what
  chokes). `liquidity_topn` config knob added but defaulted off.

## Decisions (settled in brainstorm)

- **Filter basis:** **activity** — `active_fraction` = the fraction of the last `liquidity_window`
  periods in which the stock has real (non-zero, non-NaN) Volume. Keep iff
  `active_fraction ≥ liquidity_min_active_fraction`. Currency-free and unitless (no grouping, no FX).
- **Data quality:** the existing **>15% missing-Close drop** (in `clean_batch`) stays as the
  bad-data screen.
- **No** `market_key` / market grouping / relative median / PRC / flatline rule.
- **`avg_dollar_volume`** is still computed but only as an **informational column** in the audit
  output (not used for any filtering decision).
- **Placement:** early — in step 1, so `01_prices/returns` are written already pruned.
- **Self-reporting:** an **activity-health diagnostic** each run (how many excluded; warn if a large
  fraction of stocks have zero/all-missing Volume, which signals a data-source problem).

## Part A — Production: `pipeline/01_download.py` rework

Refactor the thin script into focused, importable functions in `src/data_intake.py` (the
calibration experiment and tests import them; `pipeline/01_download.py` is the thin orchestrator).

Download machinery (UNCHANGED — already built):
- `load_tickers(csv_glob)` — read every `stock_tickers/*.csv` with `encoding="utf-8-sig"` (strip
  BOM), str-coerce, strip, drop empty/`nan`, dedup. Any size.
- `make_batches(tickers, batch_size)`.
- `clean_batch(close_raw, volume_raw, period_freq, missing_frac=0.15)` — period index,
  `groupby(level=0).first()`, drop >15%-missing Close, ffill/bfill, align Volume to kept names,
  string period-end index. Returns `(close, volume)`.
- `download_batch(batch, cfg)` — yfinance Close+Volume (multi-field vs single-ticker forms) →
  `clean_batch`; one light retry; `None` on empty/failure.
- `download_all(tickers, cfg, download_fn=None)` — `ThreadPoolExecutor(max_workers=download_workers)`
  over batches; concat to aligned `(close, volume)`; drops duplicate output columns (ISIN aliases);
  `RuntimeError` if all batches fail.

Filter (CHANGED — replaces `market_key` / `liquidity_filter` / `grouping_health`):
- `avg_dollar_volume(close, volume, window)` — kept (informational only): mean of Close×Volume over
  the last `window` periods, NaN→0.
- `active_fraction(volume, window)` — per ticker, the fraction of the last `window` periods with
  Volume > 0 (NaN counts as not-traded). Returns a Series in [0, 1].
- `activity_filter(close, volume, window, min_active_fraction)` — keep iff
  `active_fraction ≥ min_active_fraction`. Returns a per-ticker detail DataFrame
  `[avg_dollar_volume, active_fraction, kept]` indexed by ticker.
- `activity_health(detail)` — returns `{n_total, n_kept, n_excluded, zero_volume_fraction}`
  (`zero_volume_fraction` = share of tickers with `active_fraction == 0`, i.e. no trading/volume at
  all); drives the run-time warning + persisted audit.
- `main()` (`pipeline/01_download.py`) — `load_tickers` → `download_all` → `activity_filter` →
  `activity_health` (print n loaded/downloaded/kept; **warn if `zero_volume_fraction` is high**,
  e.g. >25%, signalling a Volume data-source problem) → returns (`pct_change().iloc[1:]`) on the kept
  Close → string index → write **pruned** `01_prices.csv` + `01_returns.csv`; also write
  `01_liquidity.csv` (kept ticker, `avg_dollar_volume`, `active_fraction`, `kept`) for inspection.

### Config (`params.yaml`)

```
batch_size: 500             # tickers per download batch
download_workers: 3         # parallel batch workers
download_timeout: 30        # per-batch yfinance timeout (s)
liquidity_window: 52        # periods for active_fraction / avg dollar-volume (weekly -> ~1yr)
liquidity_min_active_fraction: 0.90   # keep stocks trading in >= 90% of recent periods (PLACEHOLDER; set from calibration)
liquidity_topn: null        # optional compute-bound cap (future knob; null = off; deferred to PR-2)
```

(Removed: `liquidity_pct_of_median`, `liquidity_min_group_size`.)

### Data flow / compatibility

`01_prices/returns` keep the same string-period-end index format; only the column set changes
(pruned). Steps 2–4 read them unchanged. Volume is transient except for the audit `01_liquidity.csv`.
Note: `liquidity_window` is in *periods*, so on monthly data (`interval: 1mo`) `52` means 52 months —
revisit the value if the data frequency changes.

### Edge cases

- yfinance ISIN inputs: column labels may be ISINs or resolved tickers — the filter is label-agnostic.
- Batch failure/empty: logged, skipped, one retry; `RuntimeError` only if every batch fails.
  Duplicate output columns (two inputs → one symbol) are de-duplicated in `download_all`.
- A stock with all-zero / all-missing Volume → `active_fraction == 0` → excluded. If many stocks hit
  this, `activity_health` warns (likely a Volume data-source issue, not real inactivity).
- A stock with Close but genuinely sparse trading (frequent zero-volume weeks) → excluded by design
  (that is the target).

## Part B — Local: `experiments/calibrate_liquidity_filter.py`

Imports Part A's `download_all`, `avg_dollar_volume`, `active_fraction`, `activity_filter` (exercises
the real production code at the 3,918-ticker scale). Not pushed.

- For each **source set**, download Close+Volume **once**, then **sweep** `min_active_fraction` over a
  grid (default `[0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]`), reporting the **surviving count** at each
  threshold (and the `zero_volume_fraction`).
- **Source sets:** (A) `isins_list.csv` (3,918 — scale view); (B) `colombia_stocks_trii.csv` +
  `global_stocks_trii.csv` (123 — the names we've been testing).
- For set (B), also emit the **kept vs excluded named tickers** at a candidate threshold, to confirm
  the previously-wrongly-excluded well-traded names (BAP/JETS/BVN/etc.) now survive and only genuine
  low-activity names drop.
- Outputs → `experiments/results/liquidity_calibration/`: a survival-vs-threshold table per source +
  the kept/excluded name lists for set (B).
- **Purpose:** choose the production `liquidity_min_active_fraction` default empirically and confirm
  the filter keeps well-traded names while dropping only genuinely inactive ones. Run after Part A is
  revised; set the production default from it.

## Testing

Part A functions are unit-tested **torch-free and network-free** (tests ship with the PR). Carried
over (unchanged): `load_tickers`, `make_batches`, `clean_batch`, `download_all` (stub `download_fn`,
incl. the duplicate-column dedup). Changed/new:
- `avg_dollar_volume` — windowed mean of Close×Volume, NaN→0 (kept as info).
- `active_fraction` — fraction of recent periods with Volume > 0; NaN and zero count as not-traded.
- `activity_filter` — an active name (traded most weeks) kept; an inactive name (mostly
  zero/NaN volume) dropped; the detail frame has `[avg_dollar_volume, active_fraction, kept]`.
- `activity_health` — `n_excluded` and `zero_volume_fraction` correct (incl. an all-zero-volume
  stock surfaced).
Removed: the obsolete `market_key`, `liquidity_filter` (relative/group), and `grouping_health` tests.
A **smoke run** of `main()` on the small CSVs (network) confirms end-to-end.

## Conventions

Part A is a **functional pipeline change → pushed as a PR** (per push-scope), tests included.
Part B and this spec/plan stay **local**. PR-1 is currently built+merged on local main with the
*old* relative-median filter; this revision amends it before the (still-pending) clean push, so the
PR carries the activity version. Calibration results in the gitignored
`experiments/results/liquidity_calibration/`.
