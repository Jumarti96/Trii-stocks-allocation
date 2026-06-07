# Transformer Training Improvements — design

Date: 2026-06-07
Status: approved

## Purpose

Two targeted improvements to `src/transformer_model.py` that reduce known
optimization friction without changing the model architecture, the pipeline
interface, or any configuration parameters:

1. **Per-stock Z-score normalization** — remove input scale imbalance across
   stocks before training.
2. **LR warmup + cosine decay** — replace the fixed learning rate with a
   schedule that ramps up over the first 5 epochs and decays to near-zero over
   the remaining 45.

Both changes are motivated by better Sharpe / forecasting accuracy, not by
risk reduction. They are preconditions for scaling the transformer to a 4K+
stock universe, where return scale disparity across assets is large.

## Context

The transformer (`TransformerModel`) is multivariate: its input at each
timestep is the full return cross-section, projected to d_model=128 via
`nn.Linear(n_stocks, 128)`. This means:

- Stocks with higher return variance dominate the MSE loss and the gradient
  signal — the optimizer under-learns low-volatility stocks.
- The autoregressive prediction loop feeds predictions back as inputs across
  `periods_to_forecast` steps; scale-imbalanced inputs compound across steps.

The `norm_first=True` pre-LayerNorm inside each encoder layer normalizes
*after* the input projection, so it does not fix the input scale problem.

For the current ~83-stock weekly universe the effect is moderate. For a 4K+
universe with emerging-market micro-caps alongside large-cap developed-market
equities, the vol ratio can reach 10:1 and the problem becomes severe.

The fixed lr=1e-4 for all 50 epochs means individual training runs may still
be oscillating near epoch 50, reducing the quality of the per-run predictions
that are averaged for stability.

## Scope

In scope:
- `src/transformer_model.py`: normalization wrapper in `train_runs`; scheduler
  in the per-run training loop.
- `tests/test_transformer_model.py` (create or extend): normalization
  round-trip + zero-std guard; scheduler lr profile.
- `README.md`: add a Transformer Model section documenting the training
  methodology.

Out of scope:
- Architecture changes (d_model, n_heads, n_layers, dropout).
- Confidence-weighted position sizing (deferred to after backtest validation).
- `n_epochs` config exposure (hardcoded at 50 — no evidence yet that tuning
  this helps).
- Any change to `02_predict.py`, `03_allocate.py`, `04_report.py`,
  `backtest_allocation.py`, or `params.yaml`.

## Design

### Placement: normalization inside `train_runs`, not `train_and_predict`

The backtest harness (`backtest_allocation.py`) injects `train_runs` directly
via its `runs_fn` seam for production runs. If normalization lived only in
`train_and_predict`, the backtest would silently bypass it. Putting it in
`train_runs` ensures both the pipeline and the backtest receive normalized
training consistently.

### Normalization

At the start of `train_runs`, before `create_dataset`:

```
mu    = returns_df.mean().values          # per-stock mean, shape (n_stocks,)
sigma = returns_df.std().clip(lower=1e-8).values  # per-stock std, clipped
data  = (returns_df.values - mu) / sigma  # normalized, shape (T, n_stocks)
```

The `1e-8` clip prevents division by zero for dormant stocks (zero historical
variance). Both the training sequences (`X`, `Y`) and the prediction seed
window (`data_preds`) use this normalized `data`.

After the autoregressive prediction loop for each run, before appending to
`all_preds_runs`:

```
run_preds_arr = np.array(run_preds) * sigma + mu  # denormalize, (H, n_stocks)
```

The returned array from `train_runs` is shape `(n_runs, H, n_stocks)` in
**original return scale** — unchanged from today. `train_and_predict` averages
and winsorizes as before; `winsorize_to_history` continues to clip against the
original `returns_df` bounds.

### LR warmup + cosine decay

Inside the per-run loop, after creating the optimizer:

```python
n_epochs  = 50
n_warmup  = 5
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warmup
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs - n_warmup, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[n_warmup]
)
```

`scheduler.step()` is called once per epoch, after the inner batch loop
closes. The epoch loop uses `n_epochs` as the range argument (value unchanged
at 50).

lr profile: 1e-5 → 1e-4 (epochs 0–4), 1e-4 → ~1e-6 (epochs 5–49).

### README addition

A new **Transformer Model** section describing:
- Multivariate training (all stocks jointly, similar to a VAR).
- Per-stock Z-score normalization and why it matters at scale.
- n_runs averaging for stability (already in place, just not documented).
- LR warmup + cosine decay.
- Winsorization at the 1st–99th percentile of historical returns.

## Testing

**Existing test — no change needed:**

`tests/test_transformer_model.py:77` — `test_train_and_predict_composes_from_train_runs`
asserts `train_and_predict(rets) == winsorize(mean(train_runs(rets)))`. Because
normalization lives *inside* `train_runs` (not in `train_and_predict`), both the
direct call and the composed call receive the same `rets`, apply the same normalization
stats, train under the same seed, and denormalize before returning — the equality still
holds. No changes required.

**New normalization tests — torch-free:**

| Test | What it checks |
|---|---|
| `test_normalise_roundtrip` | `denormalize(normalize(x)) ≈ x` for a multi-column DataFrame with mixed scales |
| `test_normalise_zero_std_column` | A column with constant returns normalizes to 0.0 without raising (1e-8 clip) |

**New `train_runs` denormalization test — uses torch (tiny model, no real data):**

| Test | What it checks |
|---|---|
| `test_train_runs_output_in_original_scale` | Build a small returns DataFrame where all values are in [-0.05, 0.05]; confirm `train_runs` output values are NOT all in the normalized [-5, 5] range relative to the *normalized* distribution — i.e., predictions have been multiplied back by sigma (≈ 0.02) so they fall within the historical return magnitude, not outside it. Concretely: `assert runs.std() < 1.0` (original weekly return scale, not unit-variance scale). |

**New scheduler test — lightweight torch (no training data):**

| Test | What it checks |
|---|---|
| `test_scheduler_lr_profile` | Instantiate optimizer + scheduler on a 2-parameter toy model; step 50 times; assert `lr[0] < lr[4]` (warmup rising), `lr[5] > lr[49]` (cosine decaying), `lr[49] < 1e-5` (near zero) |

No full training loop in new tests (tiny model, no dataset).

## Conventions

- No AI attribution in commit messages.
- Experiment and test files remain local (push-scope rule); `src/` and
  `README.md` changes are pushed.
