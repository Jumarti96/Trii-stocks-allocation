import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from nstudy_transformer_runs_4k import (
    compute_mu_per_run,
    compute_topn_overlap,
    compute_cov_across_iters,
    compute_cov_sf_across_iters,
    aggregate_topn_overlaps,
    aggregate_sigma_decay,
    write_outputs,
)


def test_compute_mu_per_run_shape():
    # 5 runs, 4 forecast periods, 3 stocks
    rng = np.random.default_rng(0)
    preds = rng.standard_normal((5, 4, 3)).astype(np.float32)
    result = compute_mu_per_run(preds)
    assert result.shape == (5, 3)


def test_compute_mu_per_run_decay_weights():
    # When all periods have the same value v, result must equal v regardless of weights
    preds = np.ones((3, 4, 2), dtype=np.float32) * 0.05
    result = compute_mu_per_run(preds)
    assert np.allclose(result, 0.05, atol=1e-6)


def test_compute_mu_per_run_decay_weights_nontrivial():
    # Two periods, single run, single stock: values [a, b]
    # Expected: w[0]*a + w[1]*b where w = softmax of exp(-0.2 * [1, 2])
    a, b = 0.1, 0.5
    preds = np.array([[[a], [b]]], dtype=np.float32)  # shape (1, 2, 1)
    idx = np.array([1, 2])
    w = np.exp(-0.2 * idx); w /= w.sum()
    expected = w[0] * a + w[1] * b
    result = compute_mu_per_run(preds)
    assert np.allclose(result[0, 0], expected, atol=1e-6)


def test_compute_topn_overlap_full():
    # identical scores → overlap = 1.0
    scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    assert compute_topn_overlap(scores, scores, k=3) == pytest.approx(1.0)


def test_compute_topn_overlap_zero():
    # completely disjoint top-k
    scores_n   = np.array([1.0, 2.0, 0.0, 0.0, 0.0])  # top-2: indices 0,1
    scores_ref = np.array([0.0, 0.0, 1.0, 2.0, 0.0])  # top-2: indices 2,3
    assert compute_topn_overlap(scores_n, scores_ref, k=2) == pytest.approx(0.0)


def test_compute_topn_overlap_partial():
    scores_n   = np.array([3.0, 2.0, 1.0, 0.0])  # top-2: indices 0,1
    scores_ref = np.array([3.0, 0.0, 2.0, 1.0])  # top-2: indices 0,2
    # intersection size = 1 (index 0), k=2 → 0.5
    assert compute_topn_overlap(scores_n, scores_ref, k=2) == pytest.approx(0.5)


def test_compute_cov_across_iters_identical_iters():
    # If every iteration has the same mu, cross-iter std = 0 → CoV = 0 everywhere
    snapshots = [{25: np.array([0.05, 0.10, -0.03])} for _ in range(5)]
    result = compute_cov_across_iters(snapshots, checkpoints=[25])
    median, p25, p75 = result[25]
    assert median == pytest.approx(0.0, abs=1e-8)


def test_compute_cov_across_iters_known_values():
    # Two iterations: stock 0 has mu=[1.0, 3.0] → mean=2, std(ddof=1)=√2≈1.414, CoV≈0.707
    #                 stock 1 has mu=[2.0, 2.0] → mean=2, std(ddof=1)=0, CoV=0
    snap0 = {10: np.array([1.0, 2.0])}
    snap1 = {10: np.array([3.0, 2.0])}
    result = compute_cov_across_iters([snap0, snap1], checkpoints=[10])
    median, _, _ = result[10]
    # median of [0.707, 0.0] ≈ 0.354
    assert median == pytest.approx(np.sqrt(2) / 2 / 2, abs=1e-6)


def test_compute_cov_sf_shape():
    # Same structure as compute_cov_across_iters — just check output shape/keys
    sf_snaps = [{25: np.ones(4), 50: np.ones(4) * 0.5} for _ in range(3)]
    result = compute_cov_sf_across_iters(sf_snaps, checkpoints=[25, 50])
    assert set(result.keys()) == {25, 50}
    for n in [25, 50]:
        median, p25, p75 = result[n]
        assert p25 <= median <= p75


def test_aggregate_topn_overlaps_mean_std():
    # Two iterations, one checkpoint, one threshold
    topn = [{25: {150: 0.6}} , {25: {150: 0.8}}]
    result = aggregate_topn_overlaps(topn, checkpoints=[25], thresholds=[150])
    mean, std = result[(25, 150)]
    assert mean == pytest.approx(0.7, abs=1e-8)
    # std(ddof=1) of [0.6, 0.8] = sqrt(((0.6-0.7)^2 + (0.8-0.7)^2)/1) ≈ 0.1414
    assert std  == pytest.approx(np.sqrt(0.02), abs=1e-6)


def test_aggregate_sigma_decay_mean():
    mean_sf = [{25: 0.04, 50: 0.03}, {25: 0.06, 50: 0.03}]
    result = aggregate_sigma_decay(mean_sf, checkpoints=[25, 50])
    assert result[25] == pytest.approx(0.05, abs=1e-8)
    assert result[50] == pytest.approx(0.03, abs=1e-8)


def _make_stub_metrics():
    checkpoints = [25, 50]
    thresholds  = [150, 300]
    return {
        'cov_mu':         {25: (0.5, 0.3, 0.7), 50: (0.4, 0.2, 0.6)},
        'topn_lw':        {(25, 150): (0.4, 0.05), (25, 300): (0.5, 0.04),
                           (50, 150): (0.6, 0.03), (50, 300): (0.7, 0.02)},
        'sigma_decay':    {25: 0.02, 50: 0.015},
        'cov_sf':         {25: (0.3, 0.2, 0.4), 50: (0.25, 0.15, 0.35)},
        'topn_fv':        {(25, 150): (0.35, 0.06), (25, 300): (0.45, 0.05),
                           (50, 150): (0.55, 0.04), (50, 300): (0.65, 0.03)},
        'checkpoints':    checkpoints,
        'thresholds':     thresholds,
        'n_iters':        2,
        'n_runs':         50,
        'n_stocks':       100,
    }


def test_write_outputs_creates_all_files():
    metrics = _make_stub_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(metrics, tmp)
        expected = {
            'convergence_metrics.csv',
            'topn_overlap_lw.csv',
            'sigma_forecast_decay.csv',
            'cov_sigma_forecast.csv',
            'topn_overlap_fv.csv',
            'nstudy_4k_summary.txt',
        }
        assert expected == set(os.listdir(tmp))


def test_write_outputs_convergence_metrics_columns():
    metrics = _make_stub_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(metrics, tmp)
        df = pd.read_csv(os.path.join(tmp, 'convergence_metrics.csv'))
        assert list(df.columns) == ['n', 'cov_mu_median', 'cov_mu_p25', 'cov_mu_p75']
        assert len(df) == 2  # two checkpoints


def test_write_outputs_topn_overlap_lw_columns():
    metrics = _make_stub_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(metrics, tmp)
        df = pd.read_csv(os.path.join(tmp, 'topn_overlap_lw.csv'))
        assert list(df.columns) == ['n', 'k', 'overlap_mean', 'overlap_std']
        assert len(df) == 4  # 2 checkpoints × 2 thresholds
