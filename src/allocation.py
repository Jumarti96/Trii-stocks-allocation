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
# The capping rule used to be duplicated byte-for-byte here and in backtesting.py, so
# the rule deciding whether a book respects max_weight had two homes and nothing
# testing that they agreed. It now lives in strategies.py, shared by both.
import strategies as strat
from strategies import cap_weights


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
        w = cap_weights(w, max_weight)

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

    Why s = 2.0. Lowered from 4.0: s in {0, 1, 2} ranked above {4, 6, 8} identically across three
    independent backtest runs (different seeds, two rebalancing cadences). s=1 topped every run but
    is statistically tied with s=2 (p=0.88), so 2.0 takes the tie-break -- it holds more names and
    degrades more gracefully if the forecasts deteriorate. Re-calibrate with a walk-forward sweep
    over s, scoring net-of-cost Sharpe paired against a fixed baseline; src/backtesting.py supplies
    the schedule, the benchmark strategies and the paired statistics.

    S IS NOT IDENTIFIED BY THE AVAILABLE DATA. Three walk-forward runs on the global
    catalogue produced three different orderings, with every spread taking both the top
    and the bottom slot depending on the configuration:

                  n=500/10y      n=300/10y      n=300/15y
        s0      0.421  (2nd)   0.276  (3rd)   0.237  (1st)
        s1      0.406  (3rd)   0.261  (4th)   0.216  (2nd)
        s2      0.423  (1st)   0.344  (2nd)   0.202  (3rd)
        s4      0.361  (4th)   0.458  (1st)   0.189  (4th)

    The first two differ only in how many names the screen kept; the third adds five
    years of calendar (20 windows from 2011 instead of 13 from 2020) and is the best
    powered and best conditioned of the three -- 13 of its windows are capacity 'ok'
    and none are 'error', against 4 'error' and no 'ok' in the n=500 run.

    An earlier version of this note read the n=500 run alone as re-confirming the
    {0,1,2} over {4} ordering. It was an artifact of one configuration. 2.0 stays
    because nothing has displaced it, NOT because it has been re-established, and a
    future sweep should be treated as calibrating from scratch.

    The instability is consistent with everything else these runs show. The model beat
    equal-weight and the S&P 500 on net Sharpe every time but never significantly
    (p=0.24, 0.25, 0.39), and random_pct -- its percentile among random books of the
    same size -- ran 0.56, 0.58, then 0.52-0.55 on the best-powered run, i.e. drifting
    toward the coin flip as power improved. n_for_80_power is 204-392 windows there,
    against 20. Tuning s against samples this size fits a parameter of a component whose
    edge over chance is not demonstrated; the differences above are noise, and reading a
    ranking off them is how a spurious calibration gets locked in.

    s is calibrated against the SCALE of mu, since the draw covariance is s^2 * Sigma / T while mu
    carries its own dispersion. A forecast whose spread is several times wider than reality makes
    that perturbation negligible and collapses the consensus back onto the raw msr solution --
    which is one reason transformer_model.capacity_report caps the universe at ~600 names.
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
    # bulk sweeps running thousands of optimisations should set it.
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
    the optimiser weights it.

    Reads cfg['equal_weight_n'] (defaults to the most diversified book the
    min_weight floor allows) and ranks by cfg['allocation_ranking'], reusing
    select_top_n so selection matches the pipeline's pre-filter exactly.
    """
    min_w, max_w = cfg["min_weight"], cfg["max_weight"]
    n = _holdings_n(returns, cfg)     # asking for more names than exist is not an error

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


# ---------------------------------------------------------------------------
# Model-free methods
#
# These began as backtest benchmarks, where their job was to ask whether the
# transformer earns its keep. Over four walk-forward runs momentum led on net Sharpe
# every time and equal-weight-top-N beat the model once, so they are selectable here
# too. The rules themselves live in strategies.py, shared with the backtester; what
# these adapters add is the production contract -- a Series over the FULL input index
# with zeros for unheld names, summing to 1, respecting max_weight and the min_weight
# floor via apply_consensus_floor.
#
# THE FLOOR MAKES THEM DIFFERENT STRATEGIES FROM THE BACKTESTED ONES. gmv and
# inverse_vol naturally spread across every name at well under 1% each; floored at
# min_weight they hold at most 1/min_weight (20 at the default 0.05), so production
# gmv is "the 20 largest GMV weights" and will not reproduce the backtest numbers.
# That is deliberate -- a 150-name book is not tradeable at this account size -- but
# it must not be mistaken for the thing that was measured. See docs/PARAMETERS.md.
# ---------------------------------------------------------------------------

# Methods that never read the forecast. allocate() skips the mu-ranked pre-filter for
# these; see the comment in allocate() for why that matters.
MODEL_FREE_METHODS = {"equal_weight_all", "gmv", "inverse_vol", "momentum", "random"}


def _holdings_n(returns, cfg):
    """How many names the count-based methods hold.

    Shared by equal_weight_topn, momentum and random rather than three separate keys.
    The default is the most diversified book the floor permits, which is also the most
    names any of these can hold once apply_consensus_floor runs.
    """
    min_w = cfg["min_weight"]
    n = cfg.get("equal_weight_n")
    if not n:
        n = int(1.0 / min_w) if min_w > 0 else len(returns)
    return min(n, len(returns))


def equal_weight_all_alloc(returns, covmat, cfg):
    """1/n across the whole allocation universe. The do-nothing baseline."""
    w = strat.equal_weight_all(returns.index)
    return apply_consensus_floor(w, cfg["min_weight"], cfg["max_weight"])


def gmv_alloc(returns, covmat, cfg):
    """Global minimum-variance. Uses the covariance only -- mu never enters.

    If this matches the model's book, the forecast is contributing nothing and the
    value sits in the Ledoit-Wolf estimate.
    """
    w = strat.gmv_weights(covmat, cfg["max_weight"])
    return apply_consensus_floor(w, cfg["min_weight"], cfg["max_weight"])


def inverse_vol_alloc(returns, covmat, cfg):
    """Naive risk parity: weights proportional to 1/sigma, correlations ignored."""
    w = strat.inverse_vol_weights(covmat)
    return apply_consensus_floor(w, cfg["min_weight"], cfg["max_weight"])


def momentum_alloc(returns, covmat, cfg, hist_rets=None):
    """Equal weights on the best trailing performers over cfg['momentum_lookback'].

    The only method needing the raw returns panel rather than the forecast, so it is
    also the only one that can fail for want of an input step 3 did not used to pass.

    `hist_rets` must end at the allocation date: the rule is "what went up recently",
    and one period of future data turns it into an oracle. It is restricted to
    returns.index here so a name outside the step-2 universe cannot be allocated just
    because it rose.
    """
    if hist_rets is None:
        raise ValueError(
            "allocation_method 'momentum' needs the historical returns panel "
            "(data/01_returns.csv), which the forecast-based methods do not. Pass "
            "hist_rets to allocate(); pipeline/03_allocate.py does this for you.")
    cols = [c for c in returns.index if c in hist_rets.columns]
    missing = [c for c in returns.index if c not in hist_rets.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} allocation candidates are absent from the returns panel, "
            f"e.g. {missing[:5]}. Steps 1 and 2 are out of step -- re-run step 2.")
    w = strat.momentum_weights(hist_rets[cols], _holdings_n(returns, cfg),
                               cfg.get("momentum_lookback", 24))
    return apply_consensus_floor(w, cfg["min_weight"], cfg["max_weight"])


def random_alloc(returns, covmat, cfg):
    """Equal weights on n names drawn uniformly at random. A CONTROL, not a strategy.

    Kept selectable because it is the only honest way to ask, live, the question the
    backtest asks with random_percentile: is this book distinguishable from luck? The
    model's percentile against these sat between 0.52 and 0.59 across four runs.
    Seeded from michaud_seed so a run is reproducible.
    """
    rng = np.random.default_rng(cfg.get("michaud_seed"))
    w = strat.random_weights(list(returns.index), _holdings_n(returns, cfg), rng)
    return apply_consensus_floor(w, cfg["min_weight"], cfg["max_weight"])


def allocate(returns, covmat, cfg, n_periods, hist_rets=None):
    """Dispatch to the configured allocation method (cfg['allocation_method']).

    `returns` is mu (expected returns per name), not a returns panel; `hist_rets` is
    the panel, needed only by momentum.

    The allocation_top_n pre-filter is applied HERE rather than by the caller, because
    it ranks by the model's mu. Handing that shortlist to a model-free method would
    make it quietly model-dependent -- a "momentum" book chosen from the transformer's
    150 favourites is not momentum, and would not match the strategy the backtest
    scored. Model-free methods therefore see the whole step-2 universe.
    """
    method = cfg.get("allocation_method", "parametric_michaud")
    if method not in MODEL_FREE_METHODS:
        returns, covmat = select_top_n(returns, covmat, cfg.get("allocation_top_n"),
                                       cfg.get("allocation_ranking", "sharpe"))
    if method == "msr":
        return msr_eliminate(returns, covmat, cfg)
    if method == "parametric_michaud":
        return resampled_michaud(returns, covmat, cfg, n_periods)
    if method == "equal_weight_topn":
        return equal_weight_topn_alloc(returns, covmat, cfg)
    if method == "equal_weight_all":
        return equal_weight_all_alloc(returns, covmat, cfg)
    if method == "gmv":
        return gmv_alloc(returns, covmat, cfg)
    if method == "inverse_vol":
        return inverse_vol_alloc(returns, covmat, cfg)
    if method == "momentum":
        return momentum_alloc(returns, covmat, cfg, hist_rets=hist_rets)
    if method == "random":
        return random_alloc(returns, covmat, cfg)
    raise ValueError(
        f"unknown allocation_method: {method!r}. Valid: msr, parametric_michaud, "
        f"equal_weight_topn, equal_weight_all, gmv, inverse_vol, momentum, random")
