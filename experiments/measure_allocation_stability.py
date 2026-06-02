"""
Measure portfolio allocation stability across repeated predict->allocate runs.

Only mu (the Transformer expected returns) changes between runs; the Ledoit-Wolf
covariance and the technical-filter selection are deterministic and computed once.
This instrument runs N iterations and reports how much the *composition* moves
(selection frequency, turnover, Jaccard, weight std) relative to how much the
*value* moves (return/vol/Sharpe dispersion), plus an input->output amplification.

Self-contained (Approach B): the selection and MSR elimination logic are
re-implemented here to mirror pipeline/03_filter.py and pipeline/04_allocate.py.
Keep them in sync if those steps change.

Run:  python experiments/measure_allocation_stability.py --iterations 30 --transformer-runs 10
  Paired Michaud comparison (Phase 3):
    python experiments/measure_allocation_stability.py --mode paired --iterations 30 --transformer-runs 10
  Production-budget reproducibility check (both arms):
    python experiments/measure_allocation_stability.py --mode paired --iterations 2 --transformer-runs 100
Requires data/01_prices.csv and data/01_returns.csv (pipeline step 1 already run).

Phase 1 (measurement) only. Phase 2 assesses compound-annualisation of mu — see
docs/superpowers/specs/2026-05-31-allocation-stability-measurement-design.md.
"""

import argparse
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import risk_kit as rk
from config import load_config, PATHS


def allocate_msr(returns, covmat, cfg):
    """Sharpe-maximising weights with the step-4 batch-elimination loop.

    Mirrors pipeline/04_allocate.py. Returns a Series over `returns.index`;
    names eliminated for failing the min-weight floor get weight 0.0.
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
    optimal = (
        pd.DataFrame(w0, index=returns.index, columns=["Weights"])
        .sort_values("Weights")
    )

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
        optimal = (
            pd.DataFrame(w, index=optimal.index, columns=["Weights"])
            .sort_values("Weights")
        )

    weights = pd.Series(0.0, index=names)
    weights[optimal.index] = optimal["Weights"]
    return weights


def apply_consensus_floor(weights, min_weight):
    """Enforce the min-weight floor on an averaged consensus, without re-optimising.

    Mirrors pipeline/04_allocate.py's batch-elimination *rule* (drop names whose
    cumulative weight, sorted ascending, is below min_weight) but renormalises the
    survivors to sum to 1 instead of re-running msr_tuned -- the consensus is an
    average of weight vectors, not an msr solution, so there is no single mu to
    optimise against. Iterates until every survivor passes; the len<=2 guard keeps
    the book from emptying. Returns a Series over the original index (dropped = 0.0).
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


def resampled_allocate(per_run_mu, covmat, cfg, eliminate_per_draw=False, eps=1e-9):
    """Michaud resampled-efficiency consensus over a list of per-run mu vectors.

    Each draw is optimised by raw msr_tuned (no elimination) so the continuous
    conviction signal survives into the average; with eliminate_per_draw=True the
    full allocate_msr loop is used per draw instead (the optional comparison arm).
    The per-draw weight vectors are averaged and floored once via apply_consensus_floor
    -- average-then-threshold, never threshold-then-average.

    All mu vectors must share the same index (the selected names). covmat must cover
    those names. Returns (consensus: Series over those names with dropped names at 0.0,
    diagnostic: DataFrame indexed by name with columns 'freq' = fraction of draws that
    gave the name a nonzero raw weight and 'mean_raw_weight' = mean raw weight).
    """
    rf = cfg["rf_period"]
    max_w = cfg["max_weight"]
    ppy = cfg["periods_per_year"]
    min_w = cfg["min_weight"]

    rows = []
    for mu_i in per_run_mu:
        cov_i = covmat.loc[mu_i.index, mu_i.index]
        if eliminate_per_draw:
            w_i = allocate_msr(mu_i, cov_i, cfg)
        else:
            arr = rk.msr_tuned(
                riskfree_rate=rf, returns=mu_i, covmat=cov_i,
                max_weight=max_w, periods_per_year=ppy, debug=False,
            )
            w_i = pd.Series(arr, index=mu_i.index)
        rows.append(w_i)

    raw = pd.DataFrame(rows).reset_index(drop=True)
    consensus = apply_consensus_floor(raw.mean(axis=0), min_w)
    diagnostic = pd.DataFrame({
        "freq": (raw.abs() > eps).mean(axis=0),
        "mean_raw_weight": raw.mean(axis=0),
    })
    return consensus, diagnostic


def sample_mu_draws(mu_bar, covmat, n_periods, n_draws, spread, rng):
    """Draw n_draws mu vectors from the canonical Michaud law N(mu_bar, spread**2 * Sigma / T).

    The parametric (mechanism B) alternative to using one transformer run per draw: perturb a
    well-estimated centre mu_bar by the classical standard error of an estimated mean, scaled by
    the tunable knob `spread` (s). Reuses the optimiser's own Ledoit-Wolf Sigma -- nothing extra
    to estimate.

    mu_bar:    Series over the selected names (the averaged per-period forecast).
    covmat:    DataFrame Ledoit-Wolf covariance over those same names (the optimiser's Sigma).
    n_periods: T, the number of return periods backing Sigma (sets the sampling-error scale).
    n_draws:   K, how many mu vectors to sample.
    spread:    the knob s; spread=0 returns n_draws exact copies of mu_bar.
    rng:       a numpy Generator (explicit for reproducibility and torch-free testing).

    Sampled via one Cholesky factor of the scaled covariance. Returns a list of Series over
    mu_bar.index, preserving name order.
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


def portfolio_metrics(weights, returns, covmat, rf):
    """Portfolio return/vol/Sharpe over the names in `weights.index`.

    Computed exactly as msr_tuned's objective does (annualised mu against the
    per-period covariance) so the numbers match the optimiser's convention.
    """
    names = list(weights.index)
    w = weights.values
    r = returns.loc[names].values
    C = covmat.loc[names, names].values
    ret = float(rk.portfolio_return(w, r))
    vol = float(rk.portfolio_vol(w, C))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")
    return {"ret": ret, "vol": vol, "sharpe": sharpe}


def selection_frequency(weights_df, eps=1e-9):
    """Fraction of iterations in which each name is held (weight > eps)."""
    return (weights_df.abs() > eps).mean(axis=0)


def weight_dispersion(weights_df):
    """Population std of each name's weight across iterations."""
    return weights_df.std(axis=0, ddof=0)


def mean_turnover(weights_df):
    """Mean over all iteration pairs of 0.5 * L1 weight distance.

    None if fewer than 2 iterations.
    """
    if len(weights_df) < 2:
        return None
    W = weights_df.values
    n = len(W)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0.5 * np.abs(W[i] - W[j]).sum()
            count += 1
    return total / count


def mean_jaccard(weights_df, eps=1e-9):
    """Mean pairwise Jaccard similarity of the held-name sets.

    None if fewer than 2 iterations.
    """
    if len(weights_df) < 2:
        return None
    held = weights_df.abs() > eps
    rows = [set(held.columns[held.iloc[i].values]) for i in range(len(held))]
    sims = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(sims))


def overlap_stats(weights_df, eps=1e-9):
    """Mean pairwise count of shared held names, mean held count, and their ratio.

    The intuitive 'shared of held' read (e.g. 7 of ~9). 'shared' and 'fraction' are
    None when there are fewer than 2 portfolios to compare.
    """
    held = weights_df.abs() > eps
    mean_held = float(held.sum(axis=1).mean())
    if len(weights_df) < 2:
        return {"shared": None, "held": mean_held, "fraction": None}
    rows = [set(held.columns[held.iloc[i].values]) for i in range(len(held))]
    shared = [
        len(rows[i] & rows[j])
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    ]
    mean_shared = float(np.mean(shared))
    fraction = mean_shared / mean_held if mean_held > 0 else float("nan")
    return {"shared": mean_shared, "held": mean_held, "fraction": fraction}


def metric_dispersion(metrics_df):
    """mean / population-std / coefficient-of-variation for each metric column."""
    out = {}
    for col in metrics_df.columns:
        s = metrics_df[col]
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        cov = std / abs(mean) if mean != 0 else float("nan")
        out[col] = {"mean": mean, "std": std, "cov": cov}
    return out


def amplification_factor(mu_df, weights_df):
    """Mean per-name weight std divided by mean per-name mu std.

    A rough indicator of how much the optimiser magnifies forecast jitter.
    NaN if the input (mu) dispersion is zero.
    """
    mu_disp = float(mu_df.std(axis=0, ddof=0).mean())
    w_disp = float(weights_df.std(axis=0, ddof=0).mean())
    return w_disp / mu_disp if mu_disp > 0 else float("nan")


def select_stocks(prices, rets, cfg):
    """Technical-filter selection — mirrors pipeline/03_filter.py.

    Keeps tickers with at least cfg['signal_min_count'] positive signals out of
    SMA/EMA/MACD/PRC. Returns the kept names that also exist in `rets`.
    """
    ma_terms = cfg["ma_terms"]
    rows = []
    for ticker in prices.columns:
        sig = rk.technical_indicators(
            prices[ticker],
            indicators=["SMA", "EMA", "MACD", "PRC"],
            ma_terms=ma_terms,
            macd_params=[12, 26, 9],
            return_df=True,
            plot=False,
            signal_tolerance=0.975,
        ).iloc[-1]
        sig_df = pd.DataFrame(sig).T
        sig_df.index = [ticker]
        sig_df.rename(columns={ticker: "Price"}, inplace=True)
        rows.append(sig_df)

    signals = pd.concat(rows, axis=0)
    keep = signals[
        np.int64(signals["MACD Signal"])
        + np.int64(signals[f"SMA{ma_terms} Signal"])
        + np.int64(signals[f"EMA{ma_terms} Signal"])
        + np.int64(signals["PRC Signal"])
        >= cfg["signal_min_count"]
    ]
    return [t for t in keep.index if t in rets.columns]


def train_runs_as_preds(rets, cfg, n_runs=None, verbose=False):
    """Default runs_fn for the paired experiment: real per-run forecasts, winsorised.

    Wraps transformer_model.train_runs (the un-averaged per-run forecasts) and
    winsorises each run to the history, returning a list of (periods x stocks)
    DataFrames -- one per Transformer run. Torch is imported lazily here so the rest
    of the module (and its tests) stay torch-free.
    """
    from transformer_model import train_runs, winsorize_to_history
    runs = train_runs(rets, cfg, n_runs=n_runs, verbose=verbose)
    out = []
    for r in range(runs.shape[0]):
        preds = pd.DataFrame(runs[r], columns=rets.columns)
        out.append(winsorize_to_history(preds, rets))
    return out


def seed_everything(seed):
    """Seed numpy and torch (torch imported lazily to keep helpers light)."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                   predict_fn=None, period_mu_fn=None, select_fn=None, seed_fn=None):
    """Run `iterations` predict->allocate passes and collect the raw evidence.

    Covariance (Ledoit-Wolf) and the selected-name set are deterministic and
    computed once; only mu changes per iteration. The four *_fn arguments are
    dependency-injection seams (default to the real implementations) so the loop
    can be tested without importing torch.

    Returns dict with DataFrames: 'mu' and 'weights' (iterations x selected),
    'metrics' (iterations x [ret, vol, sharpe]), and 'selected' (list).
    """
    if predict_fn is None or period_mu_fn is None:
        from transformer_model import train_and_predict, weighted_mean_return
        predict_fn = predict_fn or train_and_predict
        period_mu_fn = period_mu_fn or weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    rf = cfg["rf_period"]

    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    mu_records, weight_records, metric_records = [], [], []
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        preds = predict_fn(rets, cfg, n_runs=transformer_runs, verbose=False)
        mu = period_mu_fn(preds)
        mu_sel = mu.loc[selected]
        weights = allocate_msr(mu_sel, cov_sel, cfg)
        survivors = weights[weights.abs() > 1e-9]
        metrics = portfolio_metrics(survivors, mu_sel, cov_sel, rf)

        mu_records.append(mu_sel)
        weight_records.append(weights)
        metric_records.append(metrics)

    return {
        "mu": pd.DataFrame(mu_records).reset_index(drop=True),
        "weights": pd.DataFrame(weight_records).reset_index(drop=True),
        "metrics": pd.DataFrame(metric_records),
        "selected": selected,
    }


def run_paired_experiment(prices, rets, cfg, iterations, transformer_runs, seed,
                          runs_fn=None, period_mu_fn=None, select_fn=None,
                          seed_fn=None, eliminate_per_draw=False,
                          draw_mechanism="empirical", spread=1.0, n_draws=1000):
    """Run `iterations` passes, deriving the current and Michaud arms from the SAME draws.

    runs_fn(rets, cfg, n_runs, verbose) -> list of N per-run period-forecast DataFrames
    (rows = periods, columns = stocks), already winsorised. The current arm averages the
    runs into one mu and allocates once (today's pipeline); the Michaud arm optimises each
    run's mu and averages the weights. Both arms' value metrics are scored against the same
    averaged mu. The four *_fn arguments are dependency-injection seams (default to the real
    implementations) so the loop runs without importing torch.

    Returns {'current': {weights, metrics}, 'michaud': {weights, metrics, diagnostic},
    'selected': [...]} where each weights frame is iterations x selected and metrics is
    iterations x [ret, vol, sharpe]. 'diagnostic' is the per-name mean over iterations of
    the cross-draw selection frequency and mean raw weight (the conviction-gradient view).
    """
    if runs_fn is None:
        runs_fn = train_runs_as_preds
    if period_mu_fn is None:
        from transformer_model import weighted_mean_return
        period_mu_fn = weighted_mean_return
    if select_fn is None:
        select_fn = select_stocks
    if seed_fn is None:
        seed_fn = seed_everything

    rf = cfg["rf_period"]
    covmat = pd.DataFrame(
        LedoitWolf().fit(rets).covariance_, index=rets.columns, columns=rets.columns
    )
    selected = select_fn(prices, rets, cfg)
    cov_sel = covmat.loc[selected, selected]

    cur_w, cur_m = [], []
    mic_w, mic_m, mic_diag = [], [], []
    for i in range(1, iterations + 1):
        seed_fn(seed + i)
        runs = runs_fn(rets, cfg, n_runs=transformer_runs, verbose=False)

        # Current arm: average runs -> one mu -> allocate once (today's pipeline).
        preds_avg = sum(r.values for r in runs) / len(runs)
        preds_avg = pd.DataFrame(preds_avg, columns=runs[0].columns)
        mu_avg = period_mu_fn(preds_avg).loc[selected]
        w_cur = allocate_msr(mu_avg, cov_sel, cfg)
        s_cur = w_cur[w_cur.abs() > 1e-9]
        cur_w.append(w_cur)
        cur_m.append(portfolio_metrics(s_cur, mu_avg, cov_sel, rf))

        # Michaud arm: draws -> resampled consensus. Score against the SAME mu_avg.
        # 'empirical' = one draw per transformer run (Phase 3); 'parametric' = K Monte-Carlo
        # draws from N(mu_avg, spread**2 * Sigma / T) (Phase 3b).
        if draw_mechanism == "parametric":
            rng = np.random.default_rng(seed + i)
            per_run_mu = sample_mu_draws(
                mu_avg, cov_sel, n_periods=len(rets),
                n_draws=n_draws, spread=spread, rng=rng,
            )
        else:
            per_run_mu = [period_mu_fn(r).loc[selected] for r in runs]
        w_mic, diag = resampled_allocate(
            per_run_mu, cov_sel, cfg, eliminate_per_draw=eliminate_per_draw
        )
        s_mic = w_mic[w_mic.abs() > 1e-9]
        mic_w.append(w_mic)
        mic_m.append(portfolio_metrics(s_mic, mu_avg, cov_sel, rf))
        mic_diag.append(diag)

    diagnostic = sum(d for d in mic_diag) / len(mic_diag)
    return {
        "current": {
            "weights": pd.DataFrame(cur_w).reset_index(drop=True),
            "metrics": pd.DataFrame(cur_m),
        },
        "michaud": {
            "weights": pd.DataFrame(mic_w).reset_index(drop=True),
            "metrics": pd.DataFrame(mic_m),
            "diagnostic": diagnostic,
        },
        "selected": selected,
    }


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _arm_block(title, weights_df, metrics_df):
    """One arm's composition + value block for the paired summary."""
    freq = selection_frequency(weights_df)
    turnover = mean_turnover(weights_df)
    jacc = mean_jaccard(weights_df)
    ov = overlap_stats(weights_df)
    disp = metric_dispersion(metrics_df)

    lines = [f"== {title} =="]
    lines.append(f"Mean pairwise turnover: {_fmt(turnover)}")
    lines.append(f"Mean pairwise Jaccard:  {_fmt(jacc)}")
    lines.append(
        f"Overlap: {_fmt(ov['shared'])} of {_fmt(ov['held'])} names shared "
        f"(fraction {_fmt(ov['fraction'])})"
    )
    lines.append("Selection frequency (held fraction across iterations):")
    for name, v in freq.sort_values(ascending=False).items():
        lines.append(f"  {name:<14} {v:6.1%}")
    for col in ["ret", "vol", "sharpe"]:
        d = disp[col]
        lines.append(
            f"  {col:<7} mean {d['mean']:.4f}  std {d['std']:.4f}  CoV {_fmt(d['cov'])}"
        )
    return lines


def format_paired_summary(result):
    """Side-by-side current-vs-Michaud stability report from a run_paired_experiment result."""
    cur, mic = result["current"], result["michaud"]
    n_iter = len(cur["weights"])
    n_names = cur["weights"].shape[1]

    lines = [f"Iterations: {n_iter} | Selected names: {n_names}", ""]
    lines += _arm_block("CURRENT (average mu -> one allocation)", cur["weights"], cur["metrics"])
    lines.append("")
    lines += _arm_block("MICHAUD (resampled consensus)", mic["weights"], mic["metrics"])
    lines.append("")
    lines.append("== MICHAUD Conviction gradient (mean across iterations) ==")
    diag = mic["diagnostic"].sort_values("mean_raw_weight", ascending=False)
    lines.append(f"  {'name':<14} {'freq':>6} {'mean_raw_weight':>16}")
    for name, row in diag.iterrows():
        lines.append(f"  {name:<14} {row['freq']:6.1%} {row['mean_raw_weight']:16.4f}")
    lines.append("")
    lines.append(
        "Note: both arms use the SAME N draws per iteration; value metrics for both are "
        "scored against the averaged mu. Fewer transformer-runs => noisier mu => more "
        "instability (conservative upper bound vs production 100-run)."
    )
    return "\n".join(lines)


def format_summary(result):
    """Build the human-readable stability report from a run_experiment result."""
    weights_df = result["weights"]
    metrics_df = result["metrics"]
    mu_df = result["mu"]

    freq = selection_frequency(weights_df)
    wstd = weight_dispersion(weights_df)
    turnover = mean_turnover(weights_df)
    jacc = mean_jaccard(weights_df)
    disp = metric_dispersion(metrics_df)
    amp = amplification_factor(mu_df, weights_df)

    lines = []
    lines.append(f"Iterations: {len(weights_df)} | Selected names: {weights_df.shape[1]}")
    lines.append("")
    lines.append("== Composition instability ==")
    lines.append(f"Mean pairwise turnover: {_fmt(turnover)}")
    lines.append(f"Mean pairwise Jaccard:  {_fmt(jacc)}")
    lines.append("")
    lines.append("Selection frequency (fraction of iterations held):")
    for name, v in freq.sort_values(ascending=False).items():
        lines.append(f"  {name:<14} {v:6.1%}   weight std {wstd[name]:.4f}")
    lines.append("")
    lines.append("== Value stability ==")
    for col in ["ret", "vol", "sharpe"]:
        d = disp[col]
        lines.append(
            f"  {col:<7} mean {d['mean']:.4f}  std {d['std']:.4f}  CoV {_fmt(d['cov'])}"
        )
    lines.append("")
    lines.append(f"Amplification (mean weight std / mean mu std): {_fmt(amp)}")
    lines.append("")
    lines.append(
        "Note: fewer transformer-runs => noisier mu => MORE instability. This is a "
        "conservative upper bound on production (100-run) instability; re-run with "
        "--transformer-runs 100 for the production figure."
    )
    return "\n".join(lines)


def write_outputs(result, outdir):
    """Write weights/metrics CSVs and the summary text; return their paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "weights": os.path.join(outdir, "stability_weights.csv"),
        "metrics": os.path.join(outdir, "stability_metrics.csv"),
        "summary": os.path.join(outdir, "stability_summary.txt"),
    }
    result["weights"].to_csv(paths["weights"], index_label="iteration")
    result["metrics"].to_csv(paths["metrics"], index_label="iteration")
    with open(paths["summary"], "w") as f:
        f.write(format_summary(result))
    return paths


def write_paired_outputs(result, outdir):
    """Write per-arm weights/metrics CSVs, the Michaud diagnostic, and the summary text."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "current_weights": os.path.join(outdir, "paired_current_weights.csv"),
        "current_metrics": os.path.join(outdir, "paired_current_metrics.csv"),
        "michaud_weights": os.path.join(outdir, "paired_michaud_weights.csv"),
        "michaud_metrics": os.path.join(outdir, "paired_michaud_metrics.csv"),
        "michaud_diagnostic": os.path.join(outdir, "paired_michaud_diagnostic.csv"),
        "summary": os.path.join(outdir, "paired_summary.txt"),
    }
    result["current"]["weights"].to_csv(paths["current_weights"], index_label="iteration")
    result["current"]["metrics"].to_csv(paths["current_metrics"], index_label="iteration")
    result["michaud"]["weights"].to_csv(paths["michaud_weights"], index_label="iteration")
    result["michaud"]["metrics"].to_csv(paths["michaud_metrics"], index_label="iteration")
    result["michaud"]["diagnostic"].to_csv(paths["michaud_diagnostic"], index_label="name")
    with open(paths["summary"], "w") as f:
        f.write(format_paired_summary(result))
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["measure", "paired"], default="measure",
                        help="'measure' = single-arm Phase-1 instrument; "
                             "'paired' = current-vs-Michaud comparison (Phase 3)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Number of predict->allocate runs (default 30). For the "
                             "n=100 reproducibility check use --mode paired --iterations 2 "
                             "--transformer-runs 100.")
    parser.add_argument("--transformer-runs", type=int, default=10,
                        help="n_runs passed to the forecaster per iteration (default 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; iteration i uses seed+i (default 0)")
    parser.add_argument("--eliminate-per-draw", action="store_true",
                        help="(paired mode) use per-draw elimination instead of the "
                             "deferred consensus floor -- the comparison arm")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(BASE_DIR, "experiments", "results"),
                        help="Directory for output CSVs and summary")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config()
    prices = pd.read_csv(PATHS["01_prices"], index_col=0)
    rets = pd.read_csv(PATHS["01_returns"], index_col=0)

    print(f"Universe: {rets.shape[1]} stocks | mode={args.mode} | "
          f"iterations={args.iterations} | transformer-runs={args.transformer_runs} | "
          f"seed={args.seed}")

    if args.mode == "paired":
        result = run_paired_experiment(
            prices, rets, cfg,
            iterations=args.iterations, transformer_runs=args.transformer_runs,
            seed=args.seed, eliminate_per_draw=args.eliminate_per_draw,
        )
        paths = write_paired_outputs(result, args.outdir)
        print()
        print(format_paired_summary(result))
    else:
        result = run_experiment(
            prices, rets, cfg,
            iterations=args.iterations, transformer_runs=args.transformer_runs,
            seed=args.seed,
        )
        paths = write_outputs(result, args.outdir)
        print()
        print(format_summary(result))

    print("\nSaved:")
    for p in paths.values():
        print(f"       {p}")


if __name__ == "__main__":
    main()
