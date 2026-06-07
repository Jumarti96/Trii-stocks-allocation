import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from nstudy_transformer_runs_4k import compute_mu_per_run, compute_topn_overlap


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
