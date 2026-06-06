# PR-2: Remove Technical Filter + Top-N Compute Cap

**Date:** 2026-06-06
**Status:** Approved for implementation

## Goal

Make the pipeline tractable for a 2,744-name universe by:
1. Deleting the technical-indicator filter (step 03) — a look-ahead-prone allocation gate whose removal was already validated by the monthly-env backtest (filter-OFF OOS performance is fine).
2. Rewiring step 4 to read its candidate universe from step 2 outputs directly.
3. Adding a configurable top-N pre-selection in step 4 to bound optimizer compute.

## Background and Motivation

PR-1 shipped an activity filter in step 1 that keeps ~2,744 names from the 3,918-ISIN universe. The activity filter is the right place to exclude genuinely inactive stocks (trades in <90% of recent weeks). However, it leaves ~2,744 names entering step 4, which is intractable for Michaud K=1000:

- Step 2 (Transformer training): **not a bottleneck** — training time is flat across universe sizes (GPU-bound by the fixed-depth encoder, not by n_stocks). Confirmed empirically: 72-stock and 2,744-stock universes both train at ~10s/run.
- Step 4 (SLSQP per Michaud draw): **scales O(N^3)**. Measured: 83 stocks → 0.07s/draw (K=1000 ~= 1.1 min); 500 stocks → ~14s/draw (K=1000 ~= 3.6 h). At 2,744 stocks K=1000 is completely intractable (~weeks).

Target runtime for K=1000: under ~5 minutes → requires N < ~165 stocks. A configurable `allocation_top_n` (default 150) brings the optimization universe into a tractable range while leaving the full universe available for users with more compute.

The technical filter (step 03: SMA/EMA/MACD/PRC signals) was a hard allocation gate applied before the optimizer. It introduced look-ahead bias risk and was validated-off in the monthly-env backtest. Removing it eliminates the bias and simplifies the pipeline. The top-N pre-selection replaces its compute-bounding role.

## Approach

**Approach A (chosen):** Delete step 03 entirely; add a pure `select_top_n` function to `src/allocation.py`; rewire `04_allocate.py` to read the universe from step 2 outputs directly.

Rejected alternatives:
- Approach B (inline logic in 04_allocate.py): harder to unit test, breaks the src/pipeline separation.
- Approach C (repurpose step 03 as a universe selector): confusing naming, keeps dead output files.

## Data Flow

**Before:**
```
01_prices/returns -> 03_filter.py -> 03_selected_returns.csv -> 04_allocate.py
02_expected_returns, 02_covmat ---------------------------------> 04_allocate.py
```

**After:**
```
02_expected_returns -> select_top_n(n, metric) -> allocate() -> 04_weights.csv
02_covmat ----------->     (in 04_allocate.py)
01_returns ----------> n_periods only
```

Step 03 is gone. The full-universe mu and covmat from step 2 enter step 4 directly; `select_top_n` filters to top-N before the optimizer.

## New Function: `select_top_n` in `src/allocation.py`

```python
def select_top_n(mu, covmat, n, metric="sharpe"):
    """Pre-select top-n candidates by metric before optimization.

    metric='sharpe': rank by mu / sqrt(diag(covmat))
    metric='return': rank by mu only
    n=None or n >= len(mu): no-op, full universe
    """
    if n is None or n >= len(mu):
        return mu, covmat
    if metric == "sharpe":
        vol = pd.Series(np.sqrt(np.diag(covmat.values)), index=mu.index).clip(lower=1e-8)
        score = mu / vol
    elif metric == "return":
        score = mu
    else:
        raise ValueError(f"unknown allocation_ranking: {metric!r}")
    top = score.nlargest(n).index
    return mu[top], covmat.loc[top, top]
```

**Design decisions:**
- Sharpe proxy (mu/sigma) is the default ranking: better than raw return because it doesn't bias toward high-vol names that the optimizer will downweight anyway. The correlation-ignoring drawback is acceptable at N=150 (pool is wide enough that diversifiers are likely included).
- `n=None` is the explicit no-cap signal, keeping backward compatibility for small universes (existing 83-stock users see no behavior change).
- `clip(lower=1e-8)` guards against zero-vol names without crashing.
- Returns `(mu_slice, covmat_slice)` pair — `allocate()` interface is unchanged.

## `04_allocate.py` Changes

Remove: `selected = pd.read_csv(PATHS['03_selected_returns'], ...)` and the intersection logic.

Add before the `allocate` call:
```python
top_n  = cfg.get("allocation_top_n")
metric = cfg.get("allocation_ranking", "sharpe")
returns, covmat = select_top_n(returns, covmat, top_n, metric)
```

The universe is now derived from `02_expected_returns.csv` column list (step 2 trains on the full step-1 universe, so its output columns are the right set). No explicit mu/covmat alignment check is needed: step 2 writes both files from the same DataFrame, so their indices are guaranteed to match.

## `params.yaml` Changes

Remove:
```yaml
ma_terms: 10
signal_min_count: 3
```

Add under Portfolio optimisation (Step 4):
```yaml
allocation_top_n: 150        # candidates fed to optimizer; null = no cap (full universe)
allocation_ranking: sharpe   # "sharpe" (mu/sigma) | "return" (mu only)
```

## `pipeline/config.py` Changes

Remove from PATHS dict:
```python
'03_selected_returns': ...,
'03_selected_prices':  ...,
'03_signals':          ...,
```

## Deletions

- `pipeline/03_filter.py` — deleted entirely.

## Tests (`tests/test_allocation.py`)

Six new tests for `select_top_n`:

1. `test_select_top_n_sharpe` — top-3 from 5-stock universe ranked by mu/sigma; correct names returned, covmat sliced consistently.
2. `test_select_top_n_return` — same, `metric="return"`; different ranking confirms both paths work.
3. `test_select_top_n_null` — `n=None` returns mu and covmat unchanged.
4. `test_select_top_n_exceeds_universe` — `n=1000` on 5-stock universe returns full set (safe fallback).
5. `test_select_top_n_covmat_aligned` — returned covmat index and columns exactly match returned mu index.
6. `test_select_top_n_unknown_metric` — raises `ValueError`.

Existing `test_allocation.py` tests are unaffected (allocate/msr_eliminate/resampled_michaud interfaces unchanged).

## Push Scope

Follows `[[feedback-push-scope]]`: functional changes + tests push; timing experiments stay local.

**In the PR (6 files):**
- `pipeline/03_filter.py` — deleted
- `pipeline/config.py` — remove 3 PATHS entries
- `pipeline/04_allocate.py` — rewire universe source + select_top_n call
- `src/allocation.py` — add select_top_n
- `params.yaml` — remove 2 keys, add 2 keys
- `tests/test_allocation.py` — 6 new tests

**Local only:**
- `experiments/time_step2_scale.py`
- `experiments/time_step4_scale.py`

PR branches off `origin/main` clean; verify `git diff --cached --name-status origin/main` before committing.
