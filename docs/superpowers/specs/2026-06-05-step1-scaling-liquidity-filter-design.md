# Step-1 scaling + relative-liquidity filter (PR-1) — design

Date: 2026-06-05
Status: approved (pending spec review)

## Purpose

Rework pipeline step 1 so the allocation model works at **4,000+ stocks**, not just ~123:
(a) download at scale via **parallel batches** fetching **Close + Volume**, and (b) prune the
universe **early** (before training) with a **currency-robust relative-liquidity filter** that
excludes thinly-traded "junk" and bounds downstream compute. Plus a **local calibration
experiment** that sweeps the filter threshold over the real ticker sources so the production
default is chosen empirically — the filter's purpose is to drop junk + save compute, not to
exclude for its own sake.

Driver: the universe is moving from 123 tickers (`colombia_stocks_trii.csv` 36 +
`global_stocks_trii.csv` 87) to a 3,918-ISIN list (`isins_list.csv`). The transformer trains and
the optimizer runs over the surviving universe, so pruning illiquid names early is the lever that
makes 4,000+ tractable.

## Scope

This is **PR-1** of a two-part effort.

In scope (PR-1):
- **Production (pushed):** refactor `pipeline/01_download.py` into testable functions — robust
  ticker loading, parallel batch download of Close+Volume, the relative-liquidity filter, and
  pruned outputs; new `params.yaml` keys; unit tests shipped with the code.
- **Local (not pushed):** `experiments/calibrate_liquidity_filter.py` — threshold-sweep over the
  ticker sources to pick the production default.

Out of scope:
- **PR-2 (separate later cycle):** removing the technical-indicator filter (delete step 3) and
  rewiring step 4 to optimise over the pruned universe. PR-1 leaves steps 2–4 working unchanged
  (they read the now-pruned `01_*` outputs).
- ISIN→ticker resolution (yfinance reads ISINs directly).
- FX/true-currency normalisation (the relative-within-market filter avoids needing it).
- A top-N liquidity cap (config knob added but defaulted off — threshold-only per the decision).

## Decisions (settled in brainstorm)

- **Liquidity basis:** average **dollar-volume** (Close × Volume), averaged over a recent window.
- **Currency-robustness:** **relative within market groups** — keep a stock iff its avg
  dollar-volume ≥ `pct_of_median × (median avg dollar-volume of its market group)`. No FX, no
  per-ticker currency lookups.
- **Market group key (`market_key`):** ISIN country prefix (first 2 chars of a 12-char ISIN) →
  e.g. `US`,`CO`,`KY`,`DE`; else ticker exchange suffix after `.`; else `OTHER` (catch-all bucket).
  **Caveat:** the ISIN prefix is issuer *domicile*, not always trading *currency* (e.g. `KY` ISINs
  are often USD ADRs). It correlates strongly with currency in practice; the calibration
  experiment validates that the grouping behaves sensibly before the default is trusted.
- **Placement:** early — in step 1, so `01_prices/returns` are written already pruned.
- **Cut:** threshold-only (relative). `liquidity_topn` config exists but defaults `null`.
- **Grouping robustness (conservative + self-reporting):** because the grouping may not fit an
  arbitrary user list, a group whose size is `< min_group_size` is treated as having an unreliable
  reference → **all its members are kept and flagged** (never drop stocks we can't reliably assess);
  a zero-median group keeps only its actually-trading members (`avg dollar-volume > 0`) and is
  flagged. Every run emits a **grouping-health diagnostic** (group count, fraction of tickers in the
  unparseable `default` bucket, list of flagged groups) so a list that doesn't fit the patterns is
  visible immediately. Locked with unit tests so this keeps working as ticker sources change.

## Part A — Production: `pipeline/01_download.py` rework

Refactor the thin script into focused, importable functions (the calibration experiment and tests
import them; refactoring is in-scope since the file is heavily changed).

- `load_tickers(csv_glob)` — read every `stock_tickers/*.csv` with **`encoding="utf-8-sig"`** (strip
  BOM), coerce to str, strip whitespace, drop empty/`nan`, dedup. Returns a list of any size.
- `make_batches(tickers, batch_size)` — split into `batch_size`-sized lists.
- `download_batch(batch, cfg)` — `yf.download(batch, interval, start, end, auto_adjust=True,
  threads=True, timeout=cfg['download_timeout'])`; extract the `Close` and `Volume` sub-frames
  (yfinance multi-field columns); `to_period(period_freq)`; `groupby(level=0).first()` (native, no
  lambda); drop tickers with >15% missing Close; ffill/bfill. Returns `(close_df, volume_df)` or
  `None` on empty/failure. One light retry on exception.
- `download_all(tickers, cfg)` — `ThreadPoolExecutor(max_workers=cfg['download_workers'])` over the
  batches; collect successful `(close, volume)` pairs; `pd.concat` along columns into aligned
  `close_all`, `volume_all`. Raise `RuntimeError` if all batches fail.
- `market_key(identifier)` — as defined above.
- `avg_dollar_volume(close, volume, window)` — `(close * volume)` per period, mean over the last
  `window` periods, per ticker → a Series. NaN volume treated as 0.
- `liquidity_filter(close, volume, window, pct_of_median, min_group_size, market_key_fn=market_key)`
  — compute `avg_dollar_volume`; group tickers by `market_key`; per group:
  - **size `< min_group_size`** (unreliable reference): keep all members, mark them flagged.
  - **median == 0** (degenerate): keep only members with avg dollar-volume > 0, mark flagged.
  - **otherwise**: keep members `≥ pct_of_median * group_median`.
  Returns the kept names (and the per-stock group/flag detail consumed by `grouping_health`).
- `grouping_health(detail)` — given the `liquidity_filter` detail frame, returns a report:
  per-group count + median + n_kept, the flagged groups (any small / zero-median member), the total
  group count, and the fraction of tickers landing in the `OTHER` catch-all bucket. Pure; drives the
  run-time warning and the persisted audit.
- `main()` — `load_tickers` → `download_all` → `grouping_health` (print the diagnostic: group
  count, % in the `default` bucket, flagged groups; warn loudly if the default-bucket fraction
  exceeds a sane bound, e.g. >25%, since that signals the list doesn't fit the patterns) →
  `liquidity_filter` → build returns (`pct_change().iloc[1:]`) on the kept Close → period-end string
  index → trim to window → write **pruned** `01_prices.csv` + `01_returns.csv`; also write
  `01_liquidity.csv` (kept ticker, avg dollar-volume, market group, flag reason) for inspection /
  audit. Print counts at each stage (loaded → downloaded valid → kept after liquidity) and step timer.

### Config additions (`params.yaml`)

```
batch_size: 500             # tickers per download batch
download_workers: 3         # parallel batch workers
download_timeout: 10        # per-batch yfinance timeout (s)
liquidity_window: 52        # periods for the avg dollar-volume (weekly -> ~1yr)
liquidity_pct_of_median: 0.10   # keep >= 10% of market-group median (PLACEHOLDER; set from calibration)
liquidity_min_group_size: 5     # groups smaller than this are kept-and-flagged (unreliable median)
liquidity_topn: null        # optional cap (future knob; null = off)
```

### Data flow / compatibility

`01_prices/returns` keep the same string-period-end index format, only the column set changes
(pruned). Steps 2–4 still read them and work unchanged at the new (smaller) universe size. Volume
is transient except for the inspection `01_liquidity.csv`.

### Edge cases

- yfinance ISIN inputs: column labels may come back as ISINs or resolved tickers — `market_key`
  handles both; the smoke run confirms which.
- Batch failure/empty: logged, skipped, one retry; `RuntimeError` only if every batch fails.
- Zero/NaN-volume stocks → avg dollar-volume 0 → excluded (untradeable), unless their group is
  small (`< min_group_size`), where the conservative keep-and-flag rule applies.
- Single-stock / tiny market group: `< min_group_size` → kept and flagged (never silently passed on
  an unreliable self-median).
- A list that doesn't fit the patterns (many unparseable IDs): they pool in the `default` bucket;
  `grouping_health` surfaces the high default fraction and `main` warns.

## Part B — Local: `experiments/calibrate_liquidity_filter.py`

Imports Part A's `download_all`, `avg_dollar_volume`, `market_key`, `liquidity_filter` (so it
exercises the real production code, including the 3,918-ticker parallel download). Not pushed.

- For each **source set**, download Close+Volume **once**, then **sweep** `pct_of_median` over a
  grid (default `[0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]`), reporting **surviving count — total and
  per market group** — at each threshold.
- **Source sets:** (A) `isins_list.csv` (3,918 — scale + junk-pruning view); (B)
  `colombia_stocks_trii.csv` + `global_stocks_trii.csv` (123 — the names we've been testing).
- For set (B), also emit the **kept vs excluded named tickers** at each threshold, so the dropped
  set can be eyeballed (junk, not good stocks).
- Report **grouping health per source**: the group-size distribution, the fraction of tickers in
  the `default` bucket, and the count of small/degenerate (flagged) groups — so we can see how well
  the grouping fits each list (especially the 3,918 ISINs) before trusting the production default.
- Outputs → `experiments/results/liquidity_calibration/`: a survival-vs-threshold table per source
  (CSV + summary text), the excluded/kept name lists for set (B), and the grouping-health report.
- **Purpose:** choose the production `liquidity_pct_of_median` default empirically and confirm the
  filter drops junk while saving compute; validate the `market_key` grouping behaves sensibly
  (the domicile-vs-currency caveat). Run after Part A is built; set the production default from it.

## Testing

Part A functions are unit-tested **torch-free and network-free** (tests ship with the PR):
- `load_tickers` — BOM strip, whitespace/`nan`/empty hygiene, dedup, multi-file union.
- `make_batches` — sizes/remainder.
- `market_key` — ISIN prefix, ticker suffix, plain US, unparseable → default.
- `avg_dollar_volume` — windowed mean of Close×Volume, NaN-volume → 0.
- `liquidity_filter` — relative-within-group keep/drop (two groups with different scales; a stock
  below 10% of its group median dropped, others kept); **small-group keep-and-flag** (group size
  `< min_group_size` → all kept); zero-median-group keeps only trading members; single-stock group
  kept-and-flagged.
- `grouping_health` — correct per-group counts/medians/flag reasons, total group count, and the
  default-bucket fraction (incl. a list with many unparseable IDs → high default fraction surfaced).
- `download_batch` clean transform — fed a synthetic multi-field raw frame (no network): missing
  drop, period index, groupby-first.
- A **smoke run** of `main()` on the current small CSVs (network) confirms end-to-end + the
  yfinance column-label form.
The calibration experiment is validated by its real run, not unit tests.

## Conventions

Part A is a **functional pipeline change → pushed as a PR** (per push-scope), tests included.
Part B and this spec/plan stay **local**. The calibration run (3,918-ticker download) runs in the
background; results in the gitignored `experiments/results/liquidity_calibration/`.
