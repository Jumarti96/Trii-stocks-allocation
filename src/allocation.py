"""
Portfolio allocation policies for step 4.

Two methods behind cfg['allocation_method']:
  - "msr"                : Sharpe-max + batch-elimination (the legacy method).
  - "parametric_michaud" : resampled efficiency -- draw K mu ~ N(mu_bar, s^2*Sigma/T),
                           raw msr per draw, average the weight vectors, one min_weight floor.

Reuses risk_kit.msr_tuned for the per-portfolio optimisation. Pure functions (no I/O);
pipeline/04_allocate.py does the file reading/writing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import risk_kit as rk


def msr_eliminate(returns, covmat, cfg):
    """Sharpe-maximising weights with the batch-elimination loop (the legacy step-4 method).

    Iteratively drops names whose cumulative weight (sorted ascending) is below min_weight and
    re-optimises, until all survivors pass or <=2 remain. Returns a Series over returns.index;
    eliminated names get 0.0.
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    names = list(returns.index)

    w0 = rk.msr_tuned(
        riskfree_rate=rf, returns=returns, covmat=covmat,
        max_weight=max_w, periods_per_year=ppy, debug=False,
    )
    optimal = pd.DataFrame(w0, index=returns.index, columns=["Weights"]).sort_values("Weights")

    while optimal["Weights"].sum() >= 0.9999:
        cum_weights = optimal["Weights"].cumsum()
        failing_mask = cum_weights < min_w
        if not failing_mask.any():
            break
        optimal = optimal[~failing_mask]
        if len(optimal) <= 2:
            break
        w = rk.msr_tuned(
            riskfree_rate=rf, returns=returns[optimal.index],
            covmat=covmat.loc[optimal.index, optimal.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
        )
        optimal = pd.DataFrame(w, index=optimal.index, columns=["Weights"]).sort_values("Weights")

    weights = pd.Series(0.0, index=names)
    weights[optimal.index] = optimal["Weights"]
    return weights


def apply_consensus_floor(weights, min_weight):
    """Enforce the min-weight floor on an averaged consensus, without re-optimising.

    Drops names whose cumulative weight (sorted ascending) is below min_weight and renormalises
    the survivors to sum to 1, iterating until every survivor passes; the len<=2 guard keeps the
    book from emptying. Returns a Series over the original index (dropped = 0.0).
    """
    names = list(weights.index)
    w = weights[weights > 0].sort_values().copy()
    while len(w) > 2:
        cum = w.cumsum()
        failing = cum < min_weight
        if not failing.any():
            break
        w = w[~failing]
        w = (w / w.sum()).sort_values()
    out = pd.Series(0.0, index=names)
    out[w.index] = w
    return out


def sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng):
    """Draw n_draws mu vectors from the canonical Michaud law N(mu_bar, spread**2 * Sigma / T).

    spread=0 returns n_draws exact copies of mu_bar. Sampled via one Cholesky of the scaled
    covariance. Returns a list of Series over mu_bar.index (name order preserved).
    """
    names = list(mu_bar.index)
    if spread == 0:
        return [mu_bar.copy() for _ in range(n_draws)]
    scale = spread ** 2 / n_periods
    cov_scaled = covmat.loc[names, names].values * scale
    chol = np.linalg.cholesky(cov_scaled)
    z = rng.standard_normal((n_draws, len(names)))
    samples = mu_bar.values + z @ chol.T
    return [pd.Series(samples[k], index=names) for k in range(n_draws)]


def resampled_michaud(returns, covmat, cfg, n_periods):
    """Parametric Michaud consensus: draw K mu ~ N(mu_bar, s^2*Sigma/T), raw msr per draw,
    average the weight vectors, one min_weight floor.

    Reads cfg['michaud_spread'] (s), cfg['michaud_mc_draws'] (K), cfg['michaud_seed'] (int for a
    reproducible draw set, or None for fresh draws). max_weight is enforced per draw by msr_tuned.
    Returns a Series over returns.index (dropped names = 0.0).
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    min_w = cfg["min_weight"]
    ppy = cfg["periods_per_year"]
    spread = cfg.get("michaud_spread", 1.0)
    n_draws = cfg.get("michaud_mc_draws", 1000)
    seed = cfg.get("michaud_seed", 0)

    rng = np.random.default_rng(seed)
    draws = sample_mu_draws(returns, covmat, n_periods, n_draws, spread, rng)

    rows = []
    for mu_i in draws:
        arr = rk.msr_tuned(
            riskfree_rate=rf, returns=mu_i, covmat=covmat.loc[mu_i.index, mu_i.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
        )
        rows.append(pd.Series(arr, index=mu_i.index))

    raw = pd.DataFrame(rows).reset_index(drop=True)
    return apply_consensus_floor(raw.mean(axis=0), min_w)


def select_top_n(mu, covmat, n, metric="sharpe"):
    """Pre-select the top-n candidates by metric before optimization.

    metric='sharpe': rank by mu / sqrt(diag(covmat))  (default)
    metric='return': rank by mu only
    n=None or n >= len(mu): no-op, returns full universe unchanged.
    """
    if n is None or n >= len(mu):
        return mu, covmat
    if metric == "sharpe":
        vol = pd.Series(
            np.sqrt(np.diag(covmat.values)), index=mu.index
        ).clip(lower=1e-8)
        score = mu / vol
    elif metric == "return":
        score = mu
    else:
        raise ValueError(f"unknown allocation_ranking: {metric!r}")
    top = score.nlargest(n).index
    return mu[top], covmat.loc[top, top]


def allocate(returns, covmat, cfg, n_periods):
    """Dispatch to the configured allocation method (cfg['allocation_method'])."""
    method = cfg.get("allocation_method", "parametric_michaud")
    if method == "msr":
        return msr_eliminate(returns, covmat, cfg)
    if method == "parametric_michaud":
        return resampled_michaud(returns, covmat, cfg, n_periods)
    raise ValueError(f"unknown allocation_method: {method!r}")
