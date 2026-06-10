"""Tests for experiments/arch_comparison.py"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from arch_comparison import (
    compute_spearman_rho,
    compute_icir,
    compute_topk_precision,
    compute_hit_rate,
    predict_zero,
    predict_momentum,
    predict_persistence,
    predict_mean_reversion,
)


def test_compute_spearman_rho_perfect():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_spearman_rho(x, x) == pytest.approx(1.0)


def test_compute_spearman_rho_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_spearman_rho(x, x[::-1]) == pytest.approx(-1.0)


def test_compute_spearman_rho_constant_returns_zero():
    x = np.ones(5)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # constant predicted → no rank signal → 0.0
    assert compute_spearman_rho(x, y) == pytest.approx(0.0)


def test_compute_icir_known_values():
    # mean=0.2, std(ddof=1)=0.1 => ICIR = 2.0
    rho_series = np.array([0.1, 0.2, 0.3])
    assert compute_icir(rho_series) == pytest.approx(2.0)


def test_compute_icir_zero_std_returns_zero():
    # constant rho → std=0 → ICIR=0 (not inf)
    assert compute_icir(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_compute_topk_precision_perfect():
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    realized  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_topk_precision(predicted, realized, k=3) == pytest.approx(1.0)


def test_compute_topk_precision_disjoint():
    # predicted top-2: indices 3,4 (values 4,5); realized top-2: indices 0,1 (values 5,4)
    predicted = np.array([0.0, 0.0, 0.0, 4.0, 5.0])
    realized  = np.array([5.0, 4.0, 0.0, 0.0, 0.0])
    assert compute_topk_precision(predicted, realized, k=2) == pytest.approx(0.0)


def test_compute_hit_rate_all_correct():
    pred = np.array([1.0, -1.0, 2.0])
    real = np.array([0.5, -0.5, 1.0])
    assert compute_hit_rate(pred, real) == pytest.approx(1.0)


def test_compute_hit_rate_all_wrong():
    pred = np.array([1.0, -1.0, 2.0])
    real = np.array([-0.5, 0.5, -1.0])
    assert compute_hit_rate(pred, real) == pytest.approx(0.0)


def test_compute_hit_rate_with_zeros_in_denominator():
    # Zero predictions count as misses; denominator is always N
    pred = np.array([1.0, 0.0, -1.0])   # middle stock: no direction
    real = np.array([0.5, 0.5,  -0.5])
    # matches: [True, False, True] → 2/3
    assert compute_hit_rate(pred, real) == pytest.approx(2.0 / 3.0)


def test_compute_spearman_rho_constant_realized_returns_zero():
    # constant realized → no rank signal → 0.0 (symmetric guard)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.ones(5)
    assert compute_spearman_rho(x, y) == pytest.approx(0.0)




def test_predict_zero_shape_and_values():
    result = predict_zero(7)
    assert result.shape == (7,)
    np.testing.assert_array_equal(result, 0.0)


def test_predict_momentum_shape():
    window = np.random.default_rng(0).normal(0, 0.02, (20, 6))
    result = predict_momentum(window)
    assert result.shape == (6,)


def test_predict_momentum_weights_recency():
    # Stock 0: spike only in most-recent row → should score highest (momentum)
    # Stock 1: spike only in oldest row       → should score lowest
    window = np.zeros((10, 2))
    window[-1, 0] = 0.10   # newest row
    window[ 0, 1] = 0.10   # oldest row
    result = predict_momentum(window)
    assert result[0] > result[1], "recent spike must outweigh old spike"


def test_predict_persistence_equals_last_row():
    window = np.random.default_rng(1).normal(0, 0.02, (15, 5))
    result = predict_persistence(window)
    np.testing.assert_array_equal(result, window[-1])


def test_predict_mean_reversion_sign():
    # Stock above cross-section → negative prediction; stock below → positive
    window = np.zeros((5, 4))
    window[-1] = [0.10, 0.01, 0.01, -0.10]  # last period: stock 0 high, stock 3 low
    result = predict_mean_reversion(window)
    assert result[0] < 0   # above cross-section → predict downward
    assert result[3] > 0   # below cross-section → predict upward
