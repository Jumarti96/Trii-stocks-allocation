# Activity-Filter Revision (PR-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PR-1's relative-dollar-volume-within-market-group filter (which over-excludes well-traded names) with a currency-free **activity filter** (keep stocks that trade in ≥ `min_active_fraction` of recent periods), removing the grouping machinery.

**Architecture:** Edit the already-built `src/data_intake.py`: add `active_fraction` / `activity_filter` / `activity_health`, delete `market_key` / `liquidity_filter` / `grouping_health`. Update the `pipeline/01_download.py` orchestrator, `params.yaml` config, and the calibration experiment to match. The download/parallel-batch functions are unchanged.

**Tech Stack:** Python, numpy, pandas, pytest. (yfinance only at calibration/smoke time.)

Spec: `docs/superpowers/specs/2026-06-05-step1-scaling-liquidity-filter-design.md` (revised). This amends the still-unpushed local PR-1.

**Conventions:**
- Test runner: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest`
- Production (`src/data_intake.py`, `pipeline/01_download.py`, `params.yaml`, `tests/test_data_intake.py`) is the eventual PR; the calibration experiment + spec/plan stay local. No AI attribution in commits.
- Task order keeps the suite green at every commit: add new functions → update consumers → remove obsolete functions+tests last.

---

## File Structure

- Modify: `src/data_intake.py` — add `active_fraction`, `activity_filter`, `activity_health`; remove `market_key`, `liquidity_filter`, `grouping_health`. (`avg_dollar_volume`, download funcs, hygiene unchanged.)
- Modify: `pipeline/01_download.py` — orchestrate with `activity_filter` / `activity_health`.
- Modify: `params.yaml` — swap liquidity config keys.
- Modify: `tests/test_data_intake.py` — add activity tests; remove obsolete grouping/relative tests.
- Modify: `experiments/calibrate_liquidity_filter.py` + `tests/test_calibrate_liquidity_filter.py` — sweep `min_active_fraction`.

---

### Task 1: `active_fraction` + `activity_filter`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py` (the `_frames` helper from the existing liquidity tests is reused):

```python
def test_active_fraction_counts_traded_periods():
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),     # traded every period -> 1.0
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),     # 2 of 4 -> 0.5
        "DEAD": ([10, 10, 10, 10], [0, 0, 0, 0]),     # never -> 0.0
    })
    af = di.active_fraction(volume, window=4)
    assert af["FULL"] == pytest.approx(1.0)
    assert af["HALF"] == pytest.approx(0.5)
    assert af["DEAD"] == pytest.approx(0.0)


def test_activity_filter_keeps_active_drops_inactive():
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),     # 1.0 -> kept
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),     # 0.5 -> dropped at 0.9
        "DEAD": ([10, 10, 10, 10], [0, 0, 0, 0]),     # 0.0 -> dropped
    })
    detail = di.activity_filter(close, volume, window=4, min_active_fraction=0.90)
    assert detail.loc["FULL", "kept"] == True
    assert detail.loc["HALF", "kept"] == False
    assert detail.loc["DEAD", "kept"] == False
    assert list(detail.columns) == ["avg_dollar_volume", "active_fraction", "kept"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -k "active_fraction or activity_filter" -v`
Expected: FAIL — `AttributeError: ... 'active_fraction'`.

- [ ] **Step 3: Write minimal implementation**

In `src/data_intake.py`, add these two functions immediately after `avg_dollar_volume` (keep `avg_dollar_volume` as-is):

```python
def active_fraction(volume, window):
    """Per ticker, the fraction of the last `window` periods with real (Volume > 0) trading.

    NaN volume counts as not-traded (NaN > 0 is False). Returns a Series in [0, 1] -- a currency-free,
    unitless activity measure (no grouping, no magnitude).
    """
    return (volume.iloc[-window:] > 0).mean(axis=0)


def activity_filter(close, volume, window, min_active_fraction):
    """Keep stocks that trade in at least `min_active_fraction` of the last `window` periods.

    Returns a per-ticker DataFrame [avg_dollar_volume (informational), active_fraction, kept].
    """
    adv = avg_dollar_volume(close, volume, window)
    af = active_fraction(volume, window)
    detail = pd.DataFrame({
        "avg_dollar_volume": adv,
        "active_fraction": af,
        "kept": af >= min_active_fraction,
    })
    return detail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -k "active_fraction or activity_filter" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add active_fraction and activity_filter"
```

---

### Task 2: `activity_health`

**Files:**
- Modify: `src/data_intake.py`
- Test: `tests/test_data_intake.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_intake.py`:

```python
def test_activity_health_counts_and_zero_volume_fraction():
    close, volume = _frames({
        "FULL": ([10, 10, 10, 10], [5, 5, 5, 5]),     # kept
        "HALF": ([10, 10, 10, 10], [5, 0, 5, 0]),     # dropped (not zero-volume though)
        "DEAD": ([10, 10, 10, 10], [0, 0, 0, 0]),     # dropped, zero-volume
    })
    detail = di.activity_filter(close, volume, window=4, min_active_fraction=0.90)
    health = di.activity_health(detail)
    assert health["n_total"] == 3
    assert health["n_kept"] == 1
    assert health["n_excluded"] == 2
    assert health["zero_volume_fraction"] == pytest.approx(1 / 3)   # only DEAD has af==0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py::test_activity_health_counts_and_zero_volume_fraction -v`
Expected: FAIL — `AttributeError: ... 'activity_health'`.

- [ ] **Step 3: Write minimal implementation**

In `src/data_intake.py`, add after `activity_filter`:

```python
def activity_health(detail):
    """Summarise the activity filter: counts plus the share of stocks that never trade.

    zero_volume_fraction (active_fraction == 0) is the data-source alarm: if it is high, Volume is
    probably missing from the feed rather than the stocks being genuinely inactive.
    """
    n_total = len(detail)
    n_kept = int(detail["kept"].sum())
    zero_volume_fraction = float((detail["active_fraction"] == 0).mean()) if n_total else 0.0
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_excluded": n_total - n_kept,
        "zero_volume_fraction": zero_volume_fraction,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py -v`
Expected: PASS (all data_intake tests, old + new, still green).

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Add activity_health diagnostic"
```

---

### Task 3: Update `pipeline/01_download.py` + `params.yaml`

**Files:**
- Modify: `pipeline/01_download.py`
- Modify: `params.yaml`

- [ ] **Step 1: Swap the config keys in `params.yaml`**

Replace these two lines:

```yaml
liquidity_pct_of_median: 0.10   # keep >= 10% of market-group median (set from calibration)
liquidity_min_group_size: 5     # groups smaller than this are kept-and-flagged
```

with this one line:

```yaml
liquidity_min_active_fraction: 0.90   # keep stocks trading in >= 90% of recent periods (set from calibration)
```

(Leave `batch_size`, `download_workers`, `download_timeout`, `liquidity_window`, `liquidity_topn` unchanged.)

- [ ] **Step 2: Update the orchestrator `main()` in `pipeline/01_download.py`**

Change the import line from:

```python
from data_intake import load_tickers, download_all, liquidity_filter, grouping_health
```

to:

```python
from data_intake import load_tickers, download_all, activity_filter, activity_health
```

Then replace the block from the `detail = liquidity_filter(...)` call through the `kept = ...` line (the grouping_health print, the OTHER-bucket warning, and the kept computation) with:

```python
    detail = activity_filter(
        close, volume,
        window=cfg["liquidity_window"],
        min_active_fraction=cfg["liquidity_min_active_fraction"],
    )
    health = activity_health(detail)
    print(f"Activity filter: kept {health['n_kept']}/{health['n_total']} "
          f"(excluded {health['n_excluded']}; zero-volume {health['zero_volume_fraction']:.0%})")
    if health["zero_volume_fraction"] > 0.25:
        print("  WARNING: many stocks have no Volume at all -> likely a Volume data-source problem.")

    kept = detail.index[detail["kept"]]
    print(f"Kept after activity filter: {len(kept)} / {close.shape[1]}")
```

The `detail.loc[kept].to_csv(... "01_liquidity.csv")` line below stays (it now writes the
`[avg_dollar_volume, active_fraction, kept]` columns).

- [ ] **Step 3: Verify config + module wiring**

Run:
```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -c "import yaml; c=yaml.safe_load(open('params.yaml')); print(c['liquidity_min_active_fraction'], 'pct_of_median' not in c, 'min_group_size' not in c)"
```
Expected: prints `0.9 True True`.

- [ ] **Step 4: Commit**

```bash
git add pipeline/01_download.py params.yaml
git commit -m "Switch step-1 orchestrator + config to activity filter"
```

---

### Task 4: Update calibration `experiments/calibrate_liquidity_filter.py`

**Files:**
- Modify: `experiments/calibrate_liquidity_filter.py`
- Test: `tests/test_calibrate_liquidity_filter.py`

- [ ] **Step 1: Rewrite the failing test**

Replace the body of `test_sweep_thresholds_counts_survivors` in
`tests/test_calibrate_liquidity_filter.py` with:

```python
def test_sweep_thresholds_counts_survivors():
    idx = ["p1", "p2", "p3", "p4", "p5"]
    close = pd.DataFrame({"A": [10] * 5, "B": [10] * 5}, index=idx)
    volume = pd.DataFrame({"A": [1, 1, 1, 1, 1],    # active 100%
                           "B": [1, 1, 1, 0, 0]},   # active 60%
                          index=idx)
    table = clf.sweep_thresholds(close, volume, window=5, grid=[0.5, 0.9])
    assert table.loc[0.5, "kept_total"] == 2     # both >= 50%
    assert table.loc[0.9, "kept_total"] == 1     # only A >= 90%
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_calibrate_liquidity_filter.py -v`
Expected: FAIL (the old `sweep_thresholds` signature/behavior, or a `min_active_fraction`/`kept_total` mismatch).

- [ ] **Step 3: Update the implementation**

In `experiments/calibrate_liquidity_filter.py`:

(a) Change the import from:
```python
from data_intake import load_tickers, download_all, liquidity_filter, grouping_health
```
to:
```python
from data_intake import load_tickers, download_all, activity_filter, activity_health
```

(b) Replace `sweep_thresholds` with:
```python
def sweep_thresholds(close, volume, window, grid):
    """For each min_active_fraction in grid, count how many stocks survive the activity filter.

    Returns a DataFrame indexed by min_active_fraction with a 'kept_total' column.
    """
    rows = {}
    for maf in grid:
        detail = activity_filter(close, volume, window, maf)
        rows[maf] = {"kept_total": int(detail["kept"].sum())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "min_active_fraction"
    return table
```

(c) Change the parser default grid to the activity grid:
```python
    parser.add_argument("--grid", type=str, default="0,0.5,0.7,0.8,0.9,0.95,1.0")
```

(d) In `main()`, replace the sweep + grouping-health block (the
`table = sweep_thresholds(...)` through the grouping print) with:
```python
    table = sweep_thresholds(close, volume, cfg["liquidity_window"], grid)
    tag = "_".join(os.path.splitext(os.path.basename(s))[0] for s in sources)[:60]
    table.to_csv(os.path.join(args.outdir, f"survival_{tag}.csv"))

    detail = activity_filter(close, volume, cfg["liquidity_window"],
                             cfg["liquidity_min_active_fraction"])
    health = activity_health(detail)

    print("\nSurvival vs min_active_fraction:")
    print(table.to_string())
    print(f"\nActivity: kept {health['n_kept']}/{health['n_total']} | "
          f"zero-volume {health['zero_volume_fraction']:.0%}")

    if args.list_names:
        names_path = os.path.join(args.outdir, f"names_{tag}.csv")
        detail.sort_values("active_fraction").to_csv(names_path)
        print(f"Wrote kept/excluded names: {names_path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_calibrate_liquidity_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/calibrate_liquidity_filter.py tests/test_calibrate_liquidity_filter.py
git commit -m "Switch calibration sweep to min_active_fraction"
```

---

### Task 5: Remove the obsolete relative-median / grouping code + tests

Now that no consumer references them, delete `market_key`, `liquidity_filter`, `grouping_health` and their tests.

**Files:**
- Modify: `src/data_intake.py`
- Modify: `tests/test_data_intake.py`

- [ ] **Step 1: Delete the obsolete tests**

In `tests/test_data_intake.py`, delete these test functions entirely:
- `test_market_key`
- `test_liquidity_filter_relative_within_group`
- `test_liquidity_filter_small_group_kept_and_flagged`
- `test_liquidity_filter_zero_median_group`
- `test_grouping_health_reports_default_fraction_and_flags`

Keep `test_make_batches`, `test_load_tickers_hygiene`, `test_clean_batch_*`, `test_download_all_*`,
`test_avg_dollar_volume`, and the new activity tests.

- [ ] **Step 2: Delete the obsolete source functions**

In `src/data_intake.py`, delete the three function definitions: `market_key`, `liquidity_filter`,
and `grouping_health` (and only those — leave `avg_dollar_volume`, `active_fraction`,
`activity_filter`, `activity_health`, and the download/hygiene functions intact).

- [ ] **Step 3: Run the data_intake + calibration tests**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_data_intake.py tests/test_calibrate_liquidity_filter.py -v`
Expected: PASS — no import errors, no references to the deleted functions.

- [ ] **Step 4: Grep to confirm no stragglers**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -c "import pathlib,sys; bad=[p for p in ['src/data_intake.py','pipeline/01_download.py','experiments/calibrate_liquidity_filter.py','tests/test_data_intake.py','tests/test_calibrate_liquidity_filter.py'] for n in ['market_key','liquidity_filter','grouping_health'] if n in pathlib.Path(p).read_text()]; print('stragglers:', bad); sys.exit(1 if bad else 0)"`
Expected: prints `stragglers: []` and exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/data_intake.py tests/test_data_intake.py
git commit -m "Remove obsolete market_key, liquidity_filter, grouping_health"
```

---

### Task 6: Full suite + re-run calibration to pick the threshold

**Files:** none (verification only).

- [ ] **Step 1: Run the whole test suite**

Run: `"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest -q`
Expected: PASS — all tests green (the data_intake suite now has the activity tests, the old grouping tests gone).

- [ ] **Step 2: Re-run calibration (network) on both sources**

Run:
```bash
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/calibrate_liquidity_filter.py --sources stock_tickers/colombia_stocks_trii.csv,stock_tickers/global_stocks_trii.csv --list-names
"C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" experiments/calibrate_liquidity_filter.py --sources stock_tickers/isins_list.csv
```
Expected: survival-vs-`min_active_fraction` tables under `experiments/results/liquidity_calibration/` (overwrites the old ones). The 3,918 run is slow — background it if needed.

- [ ] **Step 3: Report for threshold selection (no commit — calibration output gitignored)**

Report to the controller/user: the survival curve at each `min_active_fraction` for the 123 set and the 3,918 set; **confirm the previously-wrongly-excluded names (BAP, JETS, BVN, BVC.CL, GRUPOAVAL.CL, ENKA.CL, PFCEMARGOS.CL, ETB.CL, GOAU, CCU, CPAC) are now KEPT** at the candidate threshold; and the list of names that *do* get excluded (should be genuinely inactive). This sets the production `liquidity_min_active_fraction`.

---

## After the plan (later)

1. Pick `liquidity_min_active_fraction` from the calibration survival curves; set it in `params.yaml`.
2. Push PR-1 (now the activity version) as a reconstructed clean branch off origin/main with the
   functional files (`src/data_intake.py`, `pipeline/01_download.py`, `params.yaml`,
   `tests/test_data_intake.py`); verify `git diff --cached --name-status origin/main` first.
3. PR-2 (separate cycle): remove the technical filter + rewire step 4 + the top-N compute cap.

---

## Self-Review Notes

- **Spec coverage:** activity basis `active_fraction` + `activity_filter` (Task 1); `activity_health`
  with `zero_volume_fraction` warning (Tasks 2–3); `avg_dollar_volume` kept as info column (Task 1
  detail frame); orchestrator + config swap incl. `download_timeout` 30 already in place (Task 3);
  calibration sweeps `min_active_fraction` (Task 4); removal of `market_key`/`liquidity_filter`/
  `grouping_health` + tests (Task 5); calibration re-run confirming well-traded names survive
  (Task 6). Covered.
- **Out-of-scope respected:** no magnitude/FX/grouping, no PRC/drawdown rule, no technical-filter
  removal or step-4 rewire, `liquidity_topn` stays null (PR-2).
- **Type consistency:** `activity_filter` returns `[avg_dollar_volume, active_fraction, kept]`
  consumed identically by `activity_health`, `main`, and `sweep_thresholds` (Tasks 1–4); config key
  `liquidity_min_active_fraction` defined in Task 3 `params.yaml` and read in Tasks 3–4; the green-at-
  every-commit ordering (add → update consumers → remove) prevents broken intermediate states.
