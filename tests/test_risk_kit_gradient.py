"""Tests for the opt-in analytical gradient in risk_kit.msr_tuned.

The gradient (neg_sharpe_gradient) was written but left disabled behind a
"#TODO: Pendiente a revisar". These tests are that review: they pin that the
analytical path agrees with the finite-difference path SciPy uses by default,
and that production behaviour is unchanged unless a caller opts in.

Run: .venv/Scripts/python.exe -m pytest tests/test_risk_kit_gradient.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import risk_kit as rk


def _problem(seed, n=12):
    """A well-conditioned mu/Sigma pair: Sigma = A A' + diag, so PSD and non-singular."""
    rng = np.random.default_rng(seed)
    names = [f"S{i}" for i in range(n)]
    a = rng.normal(0, 0.03, (n, n))
    cov = a @ a.T + np.diag(rng.uniform(0.001, 0.01, n))
    mu = pd.Series(rng.normal(0.008, 0.006, n), index=names)
    return mu, pd.DataFrame(cov, index=names, columns=names)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gradient_and_finite_difference_agree(seed):
    mu, cov = _problem(seed)
    kw = dict(riskfree_rate=0.001, returns=mu, covmat=cov,
              max_weight=0.35, periods_per_year=54)
    w_fd   = rk.msr_tuned(**kw)
    w_grad = rk.msr_tuned(**kw, use_gradient=True)
    np.testing.assert_allclose(w_grad, w_fd, atol=1e-4)


@pytest.mark.parametrize("use_gradient", [False, True])
def test_both_paths_respect_constraints(use_gradient):
    mu, cov = _problem(7)
    w = rk.msr_tuned(riskfree_rate=0.001, returns=mu, covmat=cov,
                     max_weight=0.35, periods_per_year=54,
                     use_gradient=use_gradient)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w.min() >= -1e-9
    assert w.max() <= 0.35 + 1e-6


def test_analytical_gradient_matches_approx_fprime():
    # Checks the derivative itself, not just the optimum it leads to -- a wrong
    # gradient can still converge to the right answer and hide the error.
    mu, cov = _problem(11, n=8)
    w = np.repeat(1 / 8, 8)
    rf = 0.001

    def neg_sharpe(weights):
        ret = rk.portfolio_return(weights, mu)
        vol = rk.portfolio_vol(weights, cov)
        return -(ret - rf) / vol

    numeric = approx_fprime(w, neg_sharpe, 1e-8)
    analytic = rk._neg_sharpe_gradient(w, rf, mu, cov)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_gradient_path_actually_reduces_objective_evaluations(monkeypatch):
    # Without this, the equivalence tests pass vacuously: returns_covmat_validation
    # reads kwargs with .get(), so an unrecognised use_gradient is silently
    # swallowed and both calls run the identical finite-difference path.
    # SLSQP needs n+1 objective evaluations per finite-difference gradient but
    # only 1 when jac is supplied, so the call count separates them decisively.
    mu, cov = _problem(5, n=12)
    kw = dict(riskfree_rate=0.001, returns=mu, covmat=cov,
              max_weight=0.35, periods_per_year=54)

    counts = {}
    real = rk.portfolio_return

    def instrument(key):
        counts[key] = 0

        def counted(weights, return_series):
            counts[key] += 1
            return real(weights, return_series)
        monkeypatch.setattr(rk, 'portfolio_return', counted)

    instrument('fd')
    rk.msr_tuned(**kw)
    instrument('grad')
    rk.msr_tuned(**kw, use_gradient=True)

    assert counts['grad'] < counts['fd'] / 3, (
        f"gradient path made {counts['grad']} objective calls vs "
        f"{counts['fd']} for finite differences - flag likely not wired")


def test_allocation_forwards_use_gradient_from_cfg(monkeypatch):
    # resampled_michaud builds its own msr_tuned calls, so setting use_gradient
    # in cfg is inert unless it is explicitly forwarded. Counting objective
    # evaluations proves the plumbing rather than trusting the kwarg was read.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from allocation import resampled_michaud

    mu, cov = _problem(21, n=10)
    base = {"rf_period": 0.001, "max_weight": 0.4, "min_weight": 0.02,
            "periods_per_year": 54, "michaud_spread": 2.0,
            "michaud_mc_draws": 5, "michaud_seed": 0}

    counts = {}
    real = rk.portfolio_return

    def instrument(key):
        counts[key] = 0

        def counted(weights, return_series):
            counts[key] += 1
            return real(weights, return_series)
        monkeypatch.setattr(rk, 'portfolio_return', counted)

    instrument('off')
    resampled_michaud(mu, cov, base, n_periods=200)
    instrument('on')
    resampled_michaud(mu, cov, {**base, 'use_gradient': True}, n_periods=200)

    assert counts['on'] < counts['off'] / 3, (
        f"use_gradient=True made {counts['on']} objective calls vs "
        f"{counts['off']} with it off - cfg value is not reaching msr_tuned")


def test_allocation_defaults_to_no_gradient(monkeypatch):
    from allocation import resampled_michaud

    mu, cov = _problem(22, n=10)
    base = {"rf_period": 0.001, "max_weight": 0.4, "min_weight": 0.02,
            "periods_per_year": 54, "michaud_spread": 2.0,
            "michaud_mc_draws": 5, "michaud_seed": 0}

    seen = []
    real_msr = rk.msr_tuned

    def spy(*a, **kw):
        seen.append(kw.get('use_gradient', False))
        return real_msr(*a, **kw)

    monkeypatch.setattr(rk, 'msr_tuned', spy)
    resampled_michaud(mu, cov, base, n_periods=200)
    assert seen and not any(seen), "production path must not enable the gradient"


def test_gradient_is_off_by_default():
    # Production must be byte-identical unless a caller opts in.
    mu, cov = _problem(3)
    kw = dict(riskfree_rate=0.001, returns=mu, covmat=cov,
              max_weight=0.35, periods_per_year=54)
    np.testing.assert_array_equal(rk.msr_tuned(**kw),
                                  rk.msr_tuned(**kw, use_gradient=False))
