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
    # mean=0.2, std=0.1 (ddof=1 with 3 values: 0.1, 0.2, 0.3) → ICIR=2.0
    rho_series = np.array([0.1, 0.2, 0.3])
    expected_icir = 0.2 / rho_series.std(ddof=1)
    assert compute_icir(rho_series) == pytest.approx(expected_icir)


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
