"""
michaud_spread calibration study.

michaud_spread=4.0 was selected for the old configuration (Huber loss,
periods_to_forecast=4) on "most consistent across regimes, lower vol/turnover".
Switching to transformer_loss=rank_ic with periods_to_forecast=24 widened mu's
cross-sectional dispersion by ~2.2x, so the same s no longer perturbs mu by the
same relative amount and that calibration no longer holds.

Walk-forward sweep of s over two arms:
    A (control)  auto    loss, pto=4   -> should recover s ~ 4.0
    B (target)   rank_ic loss, pto=24  -> the answer

Arm A is a validation gate, not a nicety. If it does not land near 4.0 the
harness is measuring the wrong thing and arm B's number must be discarded.

Pre-condition: data/01_returns.csv must exist (run pipeline/01_download.py first).

Usage:
    .venv/Scripts/python.exe experiments/michaud_calibration.py [--n-runs N] [--draws K]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

import risk_kit as rk
from transformer_model import train_runs, weighted_mean_return
from allocation import allocate, select_top_n
from config import load_config, PATHS

# --- Experiment constants ---
SPREADS   = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
COST_ROUND_TRIP = 0.005   # round-trip transaction cost per unit turnover
COST_GRID = [0.001, 0.003, 0.005, 0.010, 0.020]   # sensitivity of the recommendation
HORIZON   = 24            # rebalance cadence; matches periods_to_forecast for arm B
MIN_TRAIN = 200           # smallest training span we trust (time_window=54 + samples)
N_RUNS    = 50            # dispersion 2.361 vs 2.335 at production 150 -- 1.1% off
ARMS = {
    'A_control': {'transformer_loss': 'auto',    'periods_to_forecast': 4},
    'B_target':  {'transformer_loss': 'rank_ic', 'periods_to_forecast': 24},
}
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "michaud_calibration")
_DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "docs", "experiments", "michaud-calibration.md")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def rebalance_splits(n_periods, horizon, min_train):
    """Split indices for non-overlapping rebalances, anchored at the end.

    Anchoring from the end keeps the most recent regime in the sample. Every
    returned split has `horizon` periods of forward data available.
    Returns an ascending list; empty when history is too short.
    """
    splits = []
    s = n_periods - horizon
    while s >= min_train:
        splits.append(s)
        s -= horizon
    return splits[::-1]


def turnover(w_prev, w_new):
    """One-way turnover: 0.5 * sum |w_new - w_prev| over the union of holdings.

    0.0 for identical books, 1.0 for fully disjoint ones.
    """
    idx = w_prev.index.union(w_new.index)
    a = w_prev.reindex(idx).fillna(0.0)
    b = w_new.reindex(idx).fillna(0.0)
    return float(0.5 * (b - a).abs().sum())


def drift_weights(weights, fwd_rets):
    """Weights after holding through fwd_rets without rebalancing.

    Turnover measured against drifted weights is the figure that costs money:
    part of the gap to the next target closes on its own as prices move.
    """
    growth = (1 + fwd_rets[weights.index]).prod()
    grown = weights * growth
    return grown / grown.sum()


def realised_returns(weights, fwd_rets):
    """Buy-and-hold portfolio return series over the holding period.

    Weights are set once and allowed to drift, matching how the book is
    actually held between rebalances. Names not held are ignored.
    """
    wealth = (1 + fwd_rets[weights.index]).cumprod()
    value = (wealth * weights).sum(axis=1)
    prev = value.shift(1).fillna(weights.sum())
    return value / prev - 1


def effective_n(weights):
    """Inverse Herfindahl: the number of equally weighted names this book resembles."""
    w = weights[weights > 0]
    return float(1.0 / (w ** 2).sum())


def regime_tertiles(dates):
    """Split an ordered date list into 3 contiguous, time-ordered groups."""
    n = len(dates)
    c1, c2 = n // 3, 2 * n // 3
    return [dates[:c1], dates[c1:c2], dates[c2:]]


def downside_deviation(returns, target=0.0):
    """RMS of shortfalls below `target`; upside contributes nothing.

    Michaud resampling buys robustness to estimation error, which shows up in the
    bad periods rather than the average. Symmetric vol charges a portfolio for
    large gains too, so it cannot see that benefit.
    """
    short = np.minimum(np.asarray(returns, dtype=float) - target, 0.0)
    return float(np.sqrt((short ** 2).mean()))


def max_drawdown(returns):
    """Deepest peak-to-trough decline of the compounded series (<= 0)."""
    wealth = (1 + pd.Series(list(returns))).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def annual_cost_drag(mean_turnover, cost, periods_per_year, horizon):
    """Annual return give-up from trading: turnover x cost x rebalances per year."""
    return mean_turnover * cost * (periods_per_year / horizon)


def net_of_cost_sharpe(ann_return, ann_vol, mean_turnover, rf, cost,
                       periods_per_year, horizon):
    """Sharpe after charging turnover at `cost` per unit.

    Realised vol and turnover both fall monotonically in michaud_spread, so
    neither can select an optimum on its own -- both rules degenerate to an
    endpoint. Charging turnover against gross return creates a genuine interior
    optimum, and it is the quantity that actually determines what the strategy
    earns.
    """
    if ann_vol <= 0 or np.isnan(ann_vol):
        return float('nan')
    net = ann_return - annual_cost_drag(mean_turnover, cost, periods_per_year, horizon)
    return (net - rf) / ann_vol


# ---------------------------------------------------------------------------
# Forecast cache
# ---------------------------------------------------------------------------

def forecast_for_split(rets, cfg, arm_name, split, cache_dir, n_runs=N_RUNS):
    """mu and Sigma from data up to `split`, cached to disk.

    mu and Sigma do not depend on s, so every spread in the sweep reuses one
    training pass. Mirrors 02_predict.py: exp-decay weighted mean of the
    predictions for mu, Ledoit-Wolf on the training window for Sigma.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # n_runs is part of the key: forecasts averaged over a different number of
    # runs have different cross-sectional dispersion, which is the exact quantity
    # this study calibrates against. A cheap smoke run must not poison a real one.
    path = os.path.join(cache_dir, f"{arm_name}_{split}_n{n_runs}.npz")
    train = rets.iloc[:split]
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        names = [str(c) for c in z['names']]
        return pd.Series(z['mu'], index=names), pd.DataFrame(z['cov'], index=names,
                                                             columns=names)

    runs = train_runs(train, cfg, n_runs=n_runs, verbose=False,
                      arch=cfg['transformer_arch'])
    preds = pd.DataFrame(runs.mean(axis=0), columns=train.columns)
    preds = preds.iloc[:cfg['periods_to_forecast']]      # as 02_predict.py:54 does
    mu = weighted_mean_return(preds)
    cov = pd.DataFrame(LedoitWolf().fit(train).covariance_,
                       index=train.columns, columns=train.columns)
    np.savez_compressed(path, mu=mu.values, cov=cov.values,
                        names=np.array(list(train.columns), dtype=object))
    return mu, cov


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def allocate_for_spread(mu, cov, cfg, n_periods, spread, use_gradient=True):
    """Weights for one spread. s=0 collapses to a single deterministic optimisation.

    sample_mu_draws short-circuits s=0 to n_draws identical copies of mu, so the
    optimiser would solve the same problem K times and average identical answers.
    Running it once is exact, not an approximation.
    """
    run_cfg = {**cfg, 'michaud_spread': spread, 'use_gradient': use_gradient}
    if spread == 0.0:
        run_cfg = {**run_cfg, 'michaud_mc_draws': 1}
    return allocate(mu, cov, run_cfg, n_periods)


def evaluate_arm(rets, cfg, arm_name, arm_overrides, splits, spreads, cache_dir,
                 n_runs=N_RUNS, verbose=True):
    """Run every spread across every rebalance date for one arm.

    Returns a long DataFrame with one row per (split, spread).
    """
    arm_cfg = {**cfg, **arm_overrides}
    rows = []
    prev_w = {s: None for s in spreads}
    for split in splits:
        mu, cov = forecast_for_split(rets, arm_cfg, arm_name, split, cache_dir, n_runs)
        mu_t, cov_t = select_top_n(mu, cov, arm_cfg.get('allocation_top_n'),
                                   arm_cfg.get('allocation_ranking', 'sharpe'))
        fwd = rets.iloc[split:split + HORIZON]
        for s in spreads:
            w = allocate_for_spread(mu_t, cov_t, arm_cfg, split, s)
            held = w[w > 1e-9]
            r = realised_returns(held, fwd)
            row = {
                'arm': arm_name, 'split': split, 'spread': s,
                'n_held': int(len(held)), 'max_weight': float(held.max()),
                'effective_n': effective_n(held),
                'realised_period_return': float((1 + r).prod() - 1),
            }
            if prev_w[s] is not None:
                pw, pfwd = prev_w[s]
                row['turnover'] = turnover(pw, held)
                row['turnover_drift'] = turnover(drift_weights(pw, pfwd), held)
            rows.append(row)
            prev_w[s] = (held, fwd)
        if verbose:
            print(f"  {arm_name} split={split} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df, rets, cfg, splits):
    """Aggregate per (arm, spread): realised vol/return/Sharpe, turnover, regime spread."""
    ppy = cfg['periods_per_year']
    rf = cfg['rf_rate']
    groups = regime_tertiles(splits)
    out = []
    for (arm, s), g in df.groupby(['arm', 'spread']):
        # Chain the per-rebalance realised returns into one continuous series.
        chain = pd.Series([r for r in g.sort_values('split')['realised_period_return']])
        ann_ret = (1 + chain).prod() ** (ppy / (len(chain) * HORIZON)) - 1
        # Per-rebalance returns are HORIZON periods long; annualise their vol.
        vol = chain.std(ddof=0) * np.sqrt(ppy / HORIZON)
        rec = {
            'arm': arm, 'spread': s,
            'ann_return': ann_ret, 'ann_vol': vol,
            'sharpe': (ann_ret - rf) / vol if vol > 0 else np.nan,
            'turnover': g['turnover'].mean(),
            'turnover_drift': g['turnover_drift'].mean(),
            'max_weight': g['max_weight'].mean(),
            'effective_n': g['effective_n'].mean(),
            'n_held': g['n_held'].mean(),
        }
        rec['cost_drag'] = annual_cost_drag(rec['turnover_drift'], COST_ROUND_TRIP,
                                            ppy, HORIZON)
        rec['net_return'] = ann_ret - rec['cost_drag']
        rec['sharpe_net'] = net_of_cost_sharpe(
            ann_ret, vol, rec['turnover_drift'], rf, COST_ROUND_TRIP, ppy, HORIZON)

        # Robustness view. Mean-based rules (vol, turnover, gross or net Sharpe)
        # are all monotone in s on this data and degenerate to an endpoint, so
        # they cannot express what resampling is for: protecting the bad periods.
        dd = downside_deviation(chain, target=0.0)
        rec['worst_period'] = float(chain.min())
        rec['max_drawdown'] = max_drawdown(chain)
        rec['downside_dev'] = dd * np.sqrt(ppy / HORIZON)
        rec['sortino_net'] = ((rec['net_return'] - rf) / rec['downside_dev']
                              if rec['downside_dev'] > 0 else np.nan)
        # Consistency across regimes: spread of per-tertile realised vol.
        tert_vols = []
        for grp in groups:
            sub = g[g['split'].isin(grp)]['realised_period_return']
            if len(sub) > 1:
                tert_vols.append(sub.std(ddof=0) * np.sqrt(ppy / HORIZON))
        rec['regime_vol_spread'] = (max(tert_vols) - min(tert_vols)) if tert_vols else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(['arm', 'spread']).reset_index(drop=True)


def cost_sensitivity(summary, cfg):
    """Optimal spread per arm across a grid of transaction-cost assumptions.

    Post-hoc arithmetic on the same sweep, so it is free. Shows whether the
    recommendation is stable or hinges on the assumed cost.
    """
    ppy, rf = cfg['periods_per_year'], cfg['rf_rate']
    rows = []
    for cost in COST_GRID:
        rec = {'cost': cost}
        for arm, g in summary.groupby('arm'):
            sn = g.apply(lambda r: net_of_cost_sharpe(
                r['ann_return'], r['ann_vol'], r['turnover_drift'],
                rf, cost, ppy, HORIZON), axis=1)
            rec[f'{arm}_best_s'] = g.loc[sn.idxmax(), 'spread']
            rec[f'{arm}_sharpe'] = sn.max()
        rows.append(rec)
    return pd.DataFrame(rows)


def run_timing_calibration(rets, cfg, splits, spreads, n_cal=3):
    """Estimate total runtime from a few real optimiser calls plus one training run."""
    train = rets.iloc[:splits[-1]]
    cov = pd.DataFrame(LedoitWolf().fit(train).covariance_,
                       index=train.columns, columns=train.columns)
    mu = pd.Series(train.mean().values, index=train.columns)
    t0 = time.time()
    for i in range(n_cal):
        rk.msr_tuned(riskfree_rate=cfg['rf_period'], returns=mu * (1 + 0.01 * i),
                     covmat=cov, max_weight=cfg['max_weight'],
                     periods_per_year=cfg['periods_per_year'], use_gradient=True)
    per_opt = (time.time() - t0) / n_cal

    t0 = time.time()
    train_runs(train, cfg, n_runs=1, verbose=False, arch=cfg['transformer_arch'])
    per_run = time.time() - t0

    n_draws = cfg.get('michaud_mc_draws', 1000)
    n_opt = (len(ARMS) * len(splits)
             * (sum(n_draws for s in spreads if s != 0) + spreads.count(0.0)))
    n_train = len(ARMS) * len(splits) * N_RUNS
    return per_opt, per_run, per_opt * n_opt + per_run * n_train, n_opt, n_train


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def write_outputs(raw, summary, sens, gate, out_dir, doc_path, n_runs, draws):
    """CSVs to experiments/results (gitignored) and a summary to docs/ (tracked)."""
    os.makedirs(out_dir, exist_ok=True)
    raw.to_csv(os.path.join(out_dir, 'raw.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    sens.to_csv(os.path.join(out_dir, 'cost_sensitivity.csv'), index=False)

    verdict = ("VALIDATED" if gate['validated'] else "NOT VALIDATED")
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    lines = [
        "# michaud_spread calibration",
        "",
        "Walk-forward sweep of `michaud_spread` on two arms. Arm A is a control on",
        "the pre-change configuration (Huber loss, `periods_to_forecast=4`) and must",
        "recover the deployed value of 4.0. If it does not, the harness is not",
        "measuring the right thing and arm B's number cannot be trusted.",
        "",
        "## Selection rule",
        "",
        "Realised vol and turnover both fall **monotonically** in `s` (larger `s`",
        "flattens weights toward equal-weight over the top-N), so neither can pick",
        "an interior optimum on its own -- and Sharpe on gross returns degenerates",
        "to `s=0`. The recommendation therefore maximises Sharpe after charging",
        f"turnover at a {COST_ROUND_TRIP:.1%} round-trip cost, which is both",
        "well-posed and the quantity that determines what the strategy earns.",
        "",
        f"Rebalance cadence {HORIZON} periods | n_runs {n_runs} | "
        f"mc_draws {draws} | spreads {SPREADS}",
        "",
        f"## Control gate: {verdict}",
        "",
        f"Arm A optimum at `s={gate['best_control']}` (deployed: 4.0).",
        "",
        (f"**Recommended `michaud_spread` for rank_ic / pto=24: "
         f"`{gate['best_target']}`**" if gate['validated'] else
         "**No recommendation** -- the control failed, so arm B's optimum "
         f"(`s={gate['best_target']}`) is not reportable."),
        "",
        "## Full sweep",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Cost sensitivity",
        "",
        "Optimal `s` per arm across transaction-cost assumptions -- shows whether",
        "the recommendation is stable or hinges on the assumed cost.",
        "",
        "```",
        sens.to_string(index=False),
        "```",
        "",
        "---",
        "Committed here rather than left in `experiments/results/`, which",
        "`.gitignore:31` excludes -- the reason the original architecture-study",
        "numbers were unrecoverable.",
    ]
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-runs', type=int, default=N_RUNS)
    ap.add_argument('--draws', type=int, default=None,
                    help="override michaud_mc_draws for the sweep")
    ap.add_argument('--yes', action='store_true', help="skip the confirmation prompt")
    args = ap.parse_args()

    cfg = load_config()
    if args.draws:
        cfg = {**cfg, 'michaud_mc_draws': args.draws}
    rets = pd.read_csv(PATHS['01_returns'], index_col=0)
    splits = rebalance_splits(len(rets), HORIZON, MIN_TRAIN)
    cache_dir = os.path.join(_OUT_DIR, "cache")

    print(f"\n=== michaud_calibration ===")
    print(f"Universe: {rets.shape[1]} stocks, {len(rets)} periods")
    print(f"Rebalances: {len(splits)} (every {HORIZON}), spreads: {SPREADS}")
    print(f"Arms: {list(ARMS)} | n_runs={args.n_runs} | "
          f"draws={cfg.get('michaud_mc_draws')}")

    per_opt, per_run, est, n_opt, n_train = run_timing_calibration(
        rets, cfg, splits, SPREADS)
    print(f"\nTiming: {per_opt*1000:.1f} ms/optimisation x {n_opt:,}, "
          f"{per_run:.1f} s/training x {n_train:,}")
    print(f"Estimated total: {est/3600:.1f}h")
    if not args.yes:
        print("Proceed? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    frames = []
    for arm_name, overrides in ARMS.items():
        t0 = time.time()
        frames.append(evaluate_arm(rets, cfg, arm_name, overrides, splits,
                                   SPREADS, cache_dir, args.n_runs))
        print(f"{arm_name} finished in {(time.time()-t0)/60:.1f} min", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    summary = summarise(raw, rets, cfg, splits)

    print("\n" + "=" * 78)
    print(summary.to_string(index=False))

    sens = cost_sensitivity(summary, cfg)
    print("\nCost sensitivity -- argmax net Sharpe per arm:")
    print(sens.to_string(index=False))

    # Several well-defined criteria rather than one canonical rule: the mean-based
    # ones are monotone in s on this data, so agreement between them is itself a
    # finding. CRITERIA maps name -> (column, take_max).
    CRITERIA = [
        ('net Sharpe',        'sharpe_net',        True),
        ('net Sortino',       'sortino_net',       True),
        ('worst period',      'worst_period',      True),
        ('max drawdown',      'max_drawdown',      True),
        ('regime consistency', 'regime_vol_spread', False),
    ]
    print("\nOptimal s by criterion:")
    picks = {}
    for arm in ('A_control', 'B_target'):
        g = summary[summary['arm'] == arm]
        row = {}
        for label, col, take_max in CRITERIA:
            idx = g[col].idxmax() if take_max else g[col].idxmin()
            row[label] = g.loc[idx, 'spread']
        picks[arm] = row
        print(f"  {arm}: " + "  ".join(f"{k}={v}" for k, v in row.items()))

    ctrl = summary[summary['arm'] == 'A_control']
    best_ctrl = ctrl.loc[ctrl['sharpe_net'].idxmax(), 'spread']
    tgt = summary[summary['arm'] == 'B_target']
    best_tgt = tgt.loc[tgt['sharpe_net'].idxmax(), 'spread']

    print(f"\nCONTROL GATE (net Sharpe, cost={COST_ROUND_TRIP:.1%}): arm A optimum "
          f"at s={best_ctrl}; deployed value is 4.0")
    validated = 2.0 <= best_ctrl <= 8.0
    if not validated:
        near = [k for k, v in picks['A_control'].items() if 2.0 <= v <= 8.0]
        if near:
            print(f"  (these criteria DO land near 4.0 for the control: {near})")
    if validated:
        print("  control recovered the deployed neighbourhood -- harness validated.")
        print(f"\nRECOMMENDED michaud_spread for rank_ic/pto=24: s={best_tgt}")
        print("  (reported only; params.yaml is unchanged)")
    else:
        print("  *** control did NOT recover s~4.0 -- harness is NOT validated. ***")
        print(f"  *** arm B's optimum (s={best_tgt}) must NOT be used. ***")

    gate = {'validated': validated, 'best_control': best_ctrl, 'best_target': best_tgt}
    write_outputs(raw, summary, sens, gate, _OUT_DIR, _DOC_PATH,
                  args.n_runs, cfg.get('michaud_mc_draws'))
    print(f"\nSaved: {_OUT_DIR}\n       {_DOC_PATH}")


if __name__ == '__main__':
    main()
