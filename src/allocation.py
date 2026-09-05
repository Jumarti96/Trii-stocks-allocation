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


def _cap_weights(weights, max_weight, tol=1e-12, max_passes=100):
    """Clip to max_weight and redistribute the freed weight, until nothing exceeds it.

    Iterative because capping one name pushes its excess onto the others, which can
    carry a second name over the cap. Each pass strictly reduces the excess, so this
    converges; max_passes is a backstop, not a tuning knob.
    """
    w = weights.copy()
    for _ in range(max_passes):
        over = w > max_weight + tol
        if not over.any():
            return w
        w[over] = max_weight
        deficit = 1.0 - w.sum()
        room = ~over
        if deficit <= tol or not room.any() or w[room].sum() <= tol:
            return w
        w[room] += deficit * w[room] / w[room].sum()
    return w


def apply_consensus_floor(weights, min_weight, max_weight=1.0):
    """Enforce the min-weight floor on an averaged consensus, without re-optimising.

    Drops names whose cumulative weight (sorted ascending) is below min_weight and renormalises
    the survivors to sum to 1, iterating until every survivor passes. Returns a Series over the
    original index (dropped = 0.0).

    max_weight re-enforces the per-name cap AFTER renormalisation. msr_tuned bounds every MC
    draw, but averaging and then renormalising the survivors can push a name back over the cap
    with nothing re-checking it: on the calibration sweep, 72 of 208 allocations exceeded a 0.15
    cap, median 21% over, worst 0.285. Defaults to 1.0 so existing callers are unaffected.

    The drop loop also stops at ceil(1/max_weight) survivors, because a cap of c admits no
    feasible book with fewer than 1/c names (6 names cannot sum to 1 under a 0.15 cap). Raises
    ValueError when the universe itself is too small for the cap -- shipping a book that breaks
    a stated limit is worse than failing loudly.

    Floor first, then cap: capping only raises the uncapped names, so the surviving tail grows
    and the floor cannot be re-violated. There is no cycle between the two constraints.
    """
    names = list(weights.index)
    w = weights[weights > 0].sort_values().copy()

    min_names = max(2, int(np.ceil(1.0 / max_weight - 1e-9)))
    if len(w) < min_names:
        raise ValueError(
            f"max_weight={max_weight} needs at least {min_names} names to sum to 1, "
            f"but only {len(w)} have positive weight")

    while len(w) > min_names:
        cum = w.cumsum()
        failing = cum < min_weight
        if not failing.any():
            break
        survivors = w[~failing]
        if len(survivors) < min_names:      # dropping would breach feasibility
            break
        w = (survivors / survivors.sum()).sort_values()

    if max_weight < 1.0:
        w = _cap_weights(w, max_weight)

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
    # Opt-in analytical gradient (~4.4x faster on an 80-stock problem). Absent
    # from params.yaml, so production keeps the historical finite-difference path;
    # experiments/backtest.py sets it for its bulk optimisations.
    use_gradient = cfg.get("use_gradient", False)

    rng = np.random.default_rng(seed)
    draws = sample_mu_draws(returns, covmat, n_periods, n_draws, spread, rng)

    log_every = max(1, n_draws // 10)
    rows = []
    for i, mu_i in enumerate(draws):
        arr = rk.msr_tuned(
            riskfree_rate=rf, returns=mu_i, covmat=covmat.loc[mu_i.index, mu_i.index],
            max_weight=max_w, periods_per_year=ppy, debug=False,
            use_gradient=use_gradient,
        )
        rows.append(pd.Series(arr, index=mu_i.index))
        if (i + 1) % log_every == 0 or (i + 1) == n_draws:
            print(f"  MC draw {i + 1}/{n_draws}")

    raw = pd.DataFrame(rows).reset_index(drop=True)
    return apply_consensus_floor(raw.mean(axis=0), min_w, max_w)


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


def equal_weight_topn_alloc(returns, covmat, cfg):
    """Equal weights across the top-n candidates. No optimiser, no covariance.

    Included because it matched the Michaud consensus in the walk-forward backtest
    (net Sharpe 1.132 vs 1.131 at 24-week cadence; head-to-head p=0.949), which
    suggests the edge lives in the forecast's stock selection rather than in how
    the optimiser weights it. See docs/experiments/backtest.md.

    Reads cfg['equal_weight_n'] (defaults to the most diversified book the
    min_weight floor allows) and ranks by cfg['allocation_ranking'], reusing
    select_top_n so selection matches the pipeline's pre-filter exactly.
    """
    min_w, max_w = cfg["min_weight"], cfg["max_weight"]
    n = cfg.get("equal_weight_n")
    if not n:
        # Default: the most diversified book the min_weight floor permits.
        n = int(1.0 / min_w) if min_w > 0 else len(returns)
    n = min(n, len(returns))          # asking for more names than exist is not an error

    weight = 1.0 / n
    if weight > max_w + 1e-12:
        raise ValueError(
            f"equal_weight_n={n} gives {weight:.4f} per name, above "
            f"max_weight={max_w}. Need at least {int(np.ceil(1.0 / max_w))} names.")
    if min_w > 0 and weight < min_w - 1e-12:
        raise ValueError(
            f"equal_weight_n={n} gives {weight:.4f} per name, below "
            f"min_weight={min_w}. Use at most {int(1.0 / min_w)} names.")

    mu_top, _ = select_top_n(returns, covmat, n,
                             cfg.get("allocation_ranking", "sharpe"))
    weights = pd.Series(0.0, index=returns.index)
    weights[mu_top.index] = 1.0 / len(mu_top)
    return weights


def allocate(returns, covmat, cfg, n_periods):
    """Dispatch to the configured allocation method (cfg['allocation_method'])."""
    method = cfg.get("allocation_method", "parametric_michaud")
    if method == "msr":
        return msr_eliminate(returns, covmat, cfg)
    if method == "parametric_michaud":
        return resampled_michaud(returns, covmat, cfg, n_periods)
    if method == "equal_weight_topn":
        return equal_weight_topn_alloc(returns, covmat, cfg)
    raise ValueError(f"unknown allocation_method: {method!r}")
