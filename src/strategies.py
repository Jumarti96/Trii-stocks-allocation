"""
Model-free weighting rules, shared by the backtester and the live allocator.

These were benchmarks first: src/backtesting.py used them to ask whether the
transformer earns its keep, since none of them train a model and several ignore the
forecast entirely. Four walk-forward runs later, momentum led on net Sharpe every time
and equal-weight-top-N beat the model in one, so they are allocation policies too --
src/allocation.py exposes each one through cfg['allocation_method'].

They live here rather than in either caller because a strategy defined twice drifts.
`cap_weights` is the concrete example: backtesting.py and allocation.py each carried a
byte-for-byte copy, so the rule that decides whether a book respects max_weight had two
homes and no test that they agreed.

Everything here is pure and takes no config: a Series or DataFrame in, a weight Series
out, summing to 1. The production contract -- the min_weight floor, the full-universe
index with zeros for unheld names -- is applied by allocation.py on top, because the
backtest deliberately runs these unfloored. That difference is real and is documented in
docs/PARAMETERS.md: a floored gmv book is not the gmv book the backtest scored.
"""

import numpy as np
import pandas as pd


def cap_weights(weights, max_weight, tol=1e-12, max_passes=100):
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


def equal_weight_all(names):
    """1/n across the whole universe. The do-nothing baseline."""
    names = list(names)
    return pd.Series(1.0 / len(names), index=names)


def equal_weight_topn(scores, n):
    """Equal weights across the n highest-scoring names.

    Paired against the model this separates selection skill from weighting skill:
    same picks, naive weights.

    `scores` is whatever you want to rank by -- mu, a momentum score, mu/sigma. The
    caller chooses, which is why the two callers disagree: the backtest ranks by raw mu
    over the model's own holding count, while the pipeline's equal_weight_topn method
    ranks by cfg['allocation_ranking'] over cfg['equal_weight_n'].
    """
    top = scores.nlargest(n).index
    return pd.Series(1.0 / len(top), index=top).reindex(scores.index).fillna(0.0)


def gmv_weights(covmat, max_weight=1.0):
    """Global minimum-variance weights: w proportional to Sigma^-1 * 1, normalised.

    Uses NO expected returns. If this matches the model, the forecast adds nothing
    and all the value is in the covariance estimate.

    Long-only is enforced by clipping negatives and renormalising, which keeps the
    benchmark comparable to the long-only model rather than letting it short.
    """
    names = list(covmat.index)
    ones = np.ones(len(names))
    try:
        raw = np.linalg.solve(covmat.values, ones)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(covmat.values) @ ones
    w = pd.Series(raw, index=names).clip(lower=0.0)
    if w.sum() <= 0:                      # degenerate: fall back to equal weight
        return equal_weight_all(names)
    w = w / w.sum()
    if max_weight < 1.0:
        w = cap_weights(w, max_weight)
    return w


def inverse_vol_weights(covmat):
    """Weights proportional to 1/sigma. Naive risk parity, ignores correlations."""
    vol = pd.Series(np.sqrt(np.diag(covmat.values)), index=covmat.index)
    inv = 1.0 / vol.clip(lower=1e-12)
    return inv / inv.sum()


def momentum_weights(hist_rets, n, lookback):
    """Equal weights on the n best trailing compound returns over `lookback` periods.

    The classic naive stock picker: no model, no covariance, just recent winners.

    `hist_rets` must end at the allocation date and contain nothing after it -- the
    whole rule is "what went up recently", so a single period of future data turns it
    into a look-ahead oracle.
    """
    window = hist_rets.iloc[-lookback:]
    score = (1 + window).prod() - 1
    return equal_weight_topn(score, n)


def random_weights(names, n, rng):
    """Equal weights on n names chosen uniformly at random.

    The luck control. A model that cannot beat a distribution of these has not
    demonstrated stock-picking skill, whatever its absolute return looks like. Across
    four walk-forward runs the model's random_percentile sat between 0.52 and 0.59.
    """
    names = list(names)
    picked = rng.choice(len(names), size=min(n, len(names)), replace=False)
    idx = [names[i] for i in picked]
    return pd.Series(1.0 / len(idx), index=idx).reindex(names).fillna(0.0)
