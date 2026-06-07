# Transformer Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-stock Z-score normalisation and LR warmup + cosine decay to `train_runs` in `src/transformer_model.py`, improving forecast quality for large and mixed-volatility universes.

**Architecture:** Two private helper functions (`_normalise`, `_denormalise`) are extracted from `train_runs` for testability; `train_runs` calls them at the start (normalise inputs) and end of each run's prediction loop (denormalise outputs). The LR scheduler is wired in per-run after the optimizer is created. Both changes are transparent to all callers — `train_and_predict`, `backtest_allocation.py`, and the pipeline receive the same array shapes and value scale as before.

**Tech Stack:** Python, PyTorch (`torch.optim.lr_scheduler`), NumPy, pandas, pytest. No new dependencies.

Spec: `docs/superpowers/specs/2026-06-07-transformer-training-improvements-design.md`

**Conventions:**
- Test runner: `"C:/Python projects/Finance/Scripts/python.exe" -m pytest`
- No AI attribution in commit messages.
- `src/` and `README.md` changes are pushed; test files remain local per push-scope rule.

---

## File Structure

- Modify: `src/transformer_model.py` — add `_normalise`, `_denormalise` helpers; wire into `train_runs`; add LR scheduler per run.
- Modify: `tests/test_transformer_model.py` — add four new tests (normalise round-trip, zero-std column, output scale, scheduler lr profile).
- Modify: `README.md` — add Transformer Model section documenting training methodology.

---

### Task 1: Normalisation helper tests (torch-free)

**Files:**
- Test: `tests/test_transformer_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transformer_model.py` (after the existing imports block):

```python
from transformer_model import _normalise, _denormalise


def test_normalise_roundtrip():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (100, 4)), columns=list("ABCD"))
    df["A"] *= 10    # high-vol stock
    df["D"] *= 0.1   # low-vol stock
    data, mu, sigma = _normalise(df)
    recovered = data * sigma + mu
    np.testing.assert_allclose(recovered, df.values, atol=1e-10)


def test_normalise_zero_std_column():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "A": rng.normal(0, 0.02, 50),
        "B": np.zeros(50),  # dormant stock — zero historical variance
    })
    data, mu, sigma = _normalise(df)
    assert sigma[1] == pytest.approx(1e-8)       # clipped, not zero
    np.testing.assert_array_equal(data[:, 1], 0.0)  # all zeros → normalised to 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_transformer_model.py::test_normalise_roundtrip tests/test_transformer_model.py::test_normalise_zero_std_column -v
```

Expected: FAIL — `ImportError: cannot import name '_normalise' from 'transformer_model'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_transformer_model.py
git commit -m "Add failing tests for normalise/denormalise helpers"
```

---

### Task 2: Normalisation helper implementation

**Files:**
- Modify: `src/transformer_model.py`

- [ ] **Step 1: Add `_normalise` and `_denormalise` to `transformer_model.py`**

Insert the two functions immediately before `train_runs` (after `winsorize_to_history`):

```python
def _normalise(returns_df):
    """Per-stock Z-score normalisation. Returns (data, mu, sigma).

    sigma is clipped to 1e-8 to prevent division by zero for dormant stocks.
    Both mu and sigma are 1-D ndarrays of shape (n_stocks,).
    """
    mu = returns_df.mean().values
    sigma = returns_df.std().clip(lower=1e-8).values
    data = (returns_df.values - mu) / sigma
    return data, mu, sigma


def _denormalise(preds_arr, mu, sigma):
    """Reverse per-stock Z-score normalisation.

    preds_arr: ndarray of shape (periods_to_forecast, n_stocks) in normalised space.
    Returns an ndarray of the same shape in original return scale.
    """
    return preds_arr * sigma + mu
```

- [ ] **Step 2: Run tests to verify they pass**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_transformer_model.py::test_normalise_roundtrip tests/test_transformer_model.py::test_normalise_zero_std_column -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/transformer_model.py tests/test_transformer_model.py
git commit -m "Add _normalise/_denormalise helpers to transformer_model"
```

---

### Task 3: Wire normalisation into `train_runs`

**Files:**
- Modify: `src/transformer_model.py`
- Test: `tests/test_transformer_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transformer_model.py`:

```python
def test_train_runs_output_in_original_scale():
    # _tiny_rets produces data with std ≈ 0.02 (weekly return scale).
    # If train_runs forgets to denormalise, outputs would have std ≈ 1.0.
    rets = _tiny_rets(1)
    runs = train_runs(rets, _tiny_cfg(), n_runs=2, verbose=False)
    assert runs.std() < 0.5  # normalised scale would be ~1.0; original is ~0.02
```

- [ ] **Step 2: Run test to verify it currently passes (baseline)**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_transformer_model.py::test_train_runs_output_in_original_scale -v
```

Expected: PASS — the test is a regression guard; it passes today (no normalisation yet) and must still pass after we wire it in.

- [ ] **Step 3: Wire `_normalise` and `_denormalise` into `train_runs`**

In `src/transformer_model.py`, find the start of `train_runs` where `data` is built:

```python
    data = returns_df.values
    X, Y = create_dataset(data, time_window)
```

Replace those two lines with:

```python
    data, mu, sigma = _normalise(returns_df)
    X, Y = create_dataset(data, time_window)
```

Then find the line inside the prediction loop that appends to `all_preds_runs`:

```python
        all_preds_runs.append(np.array(run_preds))
```

Replace it with:

```python
        all_preds_runs.append(_denormalise(np.array(run_preds), mu, sigma))
```

- [ ] **Step 4: Run all transformer tests**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_transformer_model.py -v
```

Expected: all tests PASS including `test_train_runs_output_in_original_scale` and the unchanged `test_train_and_predict_composes_from_train_runs`.

- [ ] **Step 5: Commit**

```bash
git add src/transformer_model.py tests/test_transformer_model.py
git commit -m "Wire per-stock Z-score normalisation into train_runs"
```

---

### Task 4: LR scheduler test (lightweight torch)

**Files:**
- Test: `tests/test_transformer_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transformer_model.py`:

```python
def test_scheduler_lr_profile():
    # Verify warmup (rising) then cosine decay (falling to near-zero).
    # Uses a 2-parameter toy model — no training data needed.
    toy_model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(toy_model.parameters(), lr=1e-4)
    n_epochs, n_warmup = 50, 5
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warmup
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs - n_warmup, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[n_warmup]
    )
    lrs = []
    for _ in range(n_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert lrs[0] < lrs[4]    # warmup: lr rising toward epoch 5
    assert lrs[5] > lrs[49]   # cosine: lr falling after warmup
    assert lrs[49] < 1e-5     # near zero at the end
```

- [ ] **Step 2: Run test to verify it fails**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_transformer_model.py::test_scheduler_lr_profile -v
```

Expected: FAIL — `AssertionError` on `lrs[0] < lrs[4]` because no scheduler is in place yet and lr is flat.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_transformer_model.py
git commit -m "Add failing test for LR warmup + cosine decay profile"
```

---

### Task 5: LR scheduler implementation

**Files:**
- Modify: `src/transformer_model.py`

- [ ] **Step 1: Add the scheduler inside the per-run training loop**

In `src/transformer_model.py`, inside the `for run in range(n_runs):` loop, find the block that creates the optimizer, criterion, and dataloader:

```python
        model      = TransformerModel(input_shape=(time_window, X.shape[2])).to(device)
        optimizer  = optim.Adam(model.parameters(), lr=1e-4)
        criterion  = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        scaler     = torch.cuda.amp.GradScaler() if use_amp else None
```

Replace with (add the scheduler block immediately after `scaler`):

```python
        model      = TransformerModel(input_shape=(time_window, X.shape[2])).to(device)
        optimizer  = optim.Adam(model.parameters(), lr=1e-4)
        criterion  = nn.MSELoss()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        scaler     = torch.cuda.amp.GradScaler() if use_amp else None

        n_epochs  = 50
        n_warmup  = 5
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs - n_warmup, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[n_warmup]
        )
```

Then find the epoch loop:

```python
        model.train()
        for _ in range(50):
            for batch_x, batch_y in dataloader:
```

Replace `for _ in range(50):` with `for _ in range(n_epochs):` and add `scheduler.step()` after the inner batch loop closes:

```python
        model.train()
        for _ in range(n_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = model(batch_x)
                        loss   = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch_x)
                    loss   = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            scheduler.step()
```

- [ ] **Step 2: Run all transformer tests**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest tests/test_transformer_model.py -v
```

Expected: all tests PASS, including `test_scheduler_lr_profile`.

- [ ] **Step 3: Run the full test suite**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -q
```

Expected: all existing tests PASS (no regressions in allocation, pipeline, backtest, or other modules).

- [ ] **Step 4: Commit**

```bash
git add src/transformer_model.py tests/test_transformer_model.py
git commit -m "Add LR warmup + cosine decay to transformer training loop"
```

---

### Task 6: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a Transformer Model section to `README.md`**

Insert the following section between `## Covariance Estimation Methods` and `## Project Structure`:

```markdown
## Transformer Model

The Transformer Neural Network (step 2) is multivariate: at each timestep it receives the full
return cross-section — all stocks simultaneously — projected to a shared d_model=128 embedding
via self-attention. This lets the model learn cross-stock relationships directly from the return
data without requiring explicit industry tags or factor labels, similar in spirit to a Vector
Autoregressive (VAR) model but with a nonlinear attention-based architecture.

Each rebalance, the model is trained from scratch `n_transformer_runs` times with different
random initialisations. The final forecast is the average across all runs, dampening
initialisation noise.

| Technique | Detail |
|---|---|
| **Per-stock Z-score normalisation** | Each stock's return series is normalised to zero mean, unit variance before training and denormalised after prediction. This prevents high-volatility stocks from dominating the MSE loss — especially important in large universes where return scale disparity between micro-caps and large-caps can reach 10:1. |
| **LR warmup + cosine decay** | Learning rate ramps from 10% → 100% of `lr=1e-4` over the first 5 epochs (warmup), then decays via cosine annealing to near-zero over the remaining 45. This avoids large noisy gradient steps during random initialisation and allows fine-tuning toward the end of training. |
| **Winsorisation** | Predictions are clipped to the 1st–99th percentile of historical returns before being passed to the optimiser, preventing extreme outlier forecasts from distorting the allocation. |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document transformer training methodology in README"
```

---

### Task 7: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite one last time**

```
"C:/Python projects/Finance/Scripts/python.exe" -m pytest -q
```

Expected: all tests PASS. Note the total test count — it should be exactly 4 more than before this feature (the four new tests added across Tasks 1, 3, and 4).

- [ ] **Step 2: Verify git log**

```bash
git log --oneline -6
```

Expected output (in order, most recent first):
```
<hash> Document transformer training methodology in README
<hash> Add LR warmup + cosine decay to transformer training loop
<hash> Add failing test for LR warmup + cosine decay profile
<hash> Wire per-stock Z-score normalisation into train_runs
<hash> Add _normalise/_denormalise helpers to transformer_model
<hash> Add failing tests for normalise/denormalise helpers
```

---

## Self-Review

**Spec coverage:**
- Per-stock Z-score normalisation: Tasks 1 (tests), 2 (helpers), 3 (wired into `train_runs`) ✓
- LR warmup + cosine decay: Tasks 4 (test), 5 (implementation) ✓
- Existing test `test_train_and_predict_composes_from_train_runs` passes unchanged: verified in Task 3 Step 4 (full suite run) ✓
- `test_train_runs_output_in_original_scale` (denormalisation guard): Task 3 ✓
- README update: Task 6 ✓
- No changes to `02_predict.py`, `03_allocate.py`, `04_report.py`, `backtest_allocation.py`, `params.yaml`: confirmed — only `src/transformer_model.py`, `tests/test_transformer_model.py`, `README.md` are touched ✓

**Placeholder scan:** No TBDs, TODOs, or incomplete steps. All code blocks are complete. ✓

**Type consistency:**
- `_normalise(returns_df)` → `(data: ndarray, mu: ndarray, sigma: ndarray)` — defined Task 2, consumed Task 3 ✓
- `_denormalise(preds_arr, mu, sigma)` → `ndarray` — defined Task 2, consumed Task 3 ✓
- `n_epochs = 50`, `n_warmup = 5` — defined and used in Task 5; same values used in Task 4 test ✓
- `_tiny_cfg()`, `_tiny_rets()` — already defined in the existing test file; reused in Task 3 ✓
