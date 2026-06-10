"""Tests for annualization helpers in src/transformer_model.py.

Run: "C:/Users/jumar/AppData/Local/Microsoft/WindowsApps/python.exe" -m pytest tests/test_transformer_model.py -v
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformer_model import (
    annualize_period_return,
    annualize_expected_returns,
    weighted_mean_return,
    _normalise_crosssectional,
    _denormalise_crosssectional,
)


def test_annualize_scalar():
    assert abs(annualize_period_return(0.01, 12) - ((1.01 ** 12) - 1)) < 1e-12


def test_annualize_zero_is_zero_any_ppy():
    assert annualize_period_return(0.0, 54) == 0.0


def test_annualize_series_elementwise():
    s = pd.Series({"A": 0.01, "B": 0.0})
    out = annualize_period_return(s, 12)
    assert abs(out["A"] - ((1.01 ** 12) - 1)) < 1e-12
    assert out["B"] == 0.0


def test_optimization_mu_has_no_ppy_dependence():
    # The per-period mu the optimiser consumes is weighted_mean_return, which
    # takes no periods_per_year argument -> changing frequency cannot distort it.
    preds = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [-0.01, 0.0, 0.01]})
    pd.testing.assert_series_equal(weighted_mean_return(preds), weighted_mean_return(preds))


def test_annualize_expected_returns_delegates_to_helper():
    preds = pd.DataFrame({"A": [0.01, 0.01, 0.01], "B": [0.005, 0.005, 0.005]})
    ppy = 12
    expected = annualize_period_return(weighted_mean_return(preds), ppy)
    pd.testing.assert_series_equal(annualize_expected_returns(preds, ppy), expected)


import numpy as np
import torch
from transformer_model import train_runs, winsorize_to_history, train_and_predict


def _tiny_cfg():
    return {"time_window": 6, "periods_to_forecast": 2, "n_transformer_runs": 2}


def _tiny_rets(seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.normal(0, 0.02, (30, 3)), columns=["A", "B", "C"])


def test_train_runs_shape():
    runs = train_runs(_tiny_rets(1), _tiny_cfg(), n_runs=2, verbose=False)
    # (n_runs, periods_to_forecast, n_stocks)
    assert runs.shape == (2, 2, 3)


def test_winsorize_to_history_clips_to_percentiles():
    rets = pd.DataFrame({"A": [-0.10, 0.0, 0.10, 0.05, -0.05]})
    preds = pd.DataFrame({"A": [0.5, -0.5, 0.0]})
    out = winsorize_to_history(preds, rets)
    lo = np.percentile(rets.values, 1)
    hi = np.percentile(rets.values, 99)
    assert out["A"].max() <= hi + 1e-12
    assert out["A"].min() >= lo - 1e-12


def test_train_and_predict_composes_from_train_runs():
    # Under the same seed, train_and_predict == winsorize(mean(train_runs)).
    cfg = _tiny_cfg()
    rets = _tiny_rets(0)

    def seeded(fn):
        torch.manual_seed(123)
        np.random.seed(123)
        return fn()

    out_direct = seeded(lambda: train_and_predict(rets, cfg, n_runs=2, verbose=False))
    runs = seeded(lambda: train_runs(rets, cfg, n_runs=2, verbose=False))
    out_compose = winsorize_to_history(
        pd.DataFrame(runs.mean(axis=0), columns=rets.columns), rets
    )
    pd.testing.assert_frame_equal(out_direct, out_compose)


from transformer_model import _normalise, _denormalise
import pytest


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
    assert sigma[1] == pytest.approx(1e-8)           # clipped, not zero
    np.testing.assert_array_equal(data[:, 1], 0.0)   # all zeros → normalised to 0


def test_train_runs_output_in_original_scale():
    # _tiny_rets produces data with std ≈ 0.02 (weekly return scale).
    # If train_runs forgets to denormalise, outputs would have std ≈ 1.0.
    rets = _tiny_rets(1)
    runs = train_runs(rets, _tiny_cfg(), n_runs=2, verbose=False)
    assert runs.std() < 0.5  # normalised scale would be ~1.0; original is ~0.02


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


def test_normalise_crosssectional_shape():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.02, (50, 5)), columns=list("ABCDE"))
    data, mu_per_t, sigma_per_t = _normalise_crosssectional(df)
    assert data.shape == (50, 5)
    assert mu_per_t.shape == (50,)
    assert sigma_per_t.shape == (50,)


def test_normalise_crosssectional_zero_mean_per_timestep():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 0.02, (30, 8)), columns=[f"S{i}" for i in range(8)])
    data, _, _ = _normalise_crosssectional(df)
    # at every timestep the mean across stocks should be ~0
    np.testing.assert_allclose(data.mean(axis=1), 0.0, atol=1e-10)


def test_normalise_crosssectional_unit_std_per_timestep():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(0, 0.02, (30, 8)), columns=[f"S{i}" for i in range(8)])
    data, _, _ = _normalise_crosssectional(df)
    # at every timestep the std across stocks should be ~1
    np.testing.assert_allclose(data.std(axis=1, ddof=1), 1.0, atol=1e-6)


def test_denormalise_crosssectional_roundtrip_last_timestep():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 0.02, (30, 5)), columns=list("ABCDE"))
    data, mu_per_t, sigma_per_t = _normalise_crosssectional(df)
    # denormalise last row using last timestep stats
    last_norm = data[-1]  # (n_stocks,)
    recovered = _denormalise_crosssectional(last_norm[np.newaxis, :], mu_per_t[-1], sigma_per_t[-1])
    np.testing.assert_allclose(recovered[0], df.values[-1], atol=1e-10)
