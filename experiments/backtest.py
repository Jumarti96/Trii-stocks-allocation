"""
General walk-forward backtester: model vs naive alternatives.

Answers two questions with one engine:

  1. Is the model useful, or are the wins luck? Compared against equal-weight,
     minimum-variance, momentum, inverse-vol, the S&P 500, and a distribution of
     RANDOM portfolios of the same size. A model that cannot beat random books has
     not demonstrated stock-picking skill, whatever its absolute return looks like.

  2. What michaud_spread should we use, and does it depend on rebalance cadence?
     Cadence is a swept parameter: holding period is decoupled from forecast
     horizon, so a 24-step forecast can be held for 12 weeks (roughly double the
     turnover, so plausibly a different optimal s).

Statistical note: all comparisons are PAIRED. Two long-only books from the same
universe are ~98% correlated, so comparing return levels reads a small effect
through the market's much larger swings. Differencing period-by-period cancels the
common move -- measured here, that cuts the sd from 0.159 to 0.049.

Power note: splitting a fixed calendar span into shorter windows does NOT add
power. Effect scales with window length h, sd with sqrt(h), N with 1/h, so
t = mu*sqrt(T)/sigma and the h cancels. A shorter cadence is a different strategy,
not a bigger sample.

Pre-condition: data/01_returns.csv (run pipeline/01_download.py first).

Usage:
    .venv/Scripts/python.exe experiments/backtest.py --cadence 12,24 --n-runs 50
    .venv/Scripts/python.exe experiments/backtest.py --n-runs 1 --draws 8 --yes
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "pipeline"))

import risk_kit as rk
from transformer_model import train_runs, weighted_mean_return
from allocation import allocate, select_top_n
from config import load_config, PATHS
from backtesting import (
    rebalance_schedule, equal_weight_all, equal_weight_topn, gmv_weights,
    inverse_vol_weights, momentum_weights, random_weights, random_percentile,
    paired_comparison, turnover, drift_weights, realised_returns, effective_n,
    max_drawdown, annual_cost_drag,
)

MIN_TRAIN = 200
COST_ROUND_TRIP = 0.005
N_RANDOM = 500           # random books drawn per window for the luck control
MOMENTUM_LOOKBACK = 24
_OUT_DIR = os.path.join(_HERE, "results", "backtest")
_BENCH_PATH = os.path.join(_HERE, "..", "data", "01_benchmark.csv")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_benchmark_returns(index_index, ticker="^GSPC", cfg=None):
    """S&P 500 weekly returns aligned to the universe's index; None if unavailable.

    CURRENCY CAVEAT: pipeline/01_download.py computes close.pct_change() on native
    prices -- COP, USD, CLP and CHF summed as one unit, with no FX conversion. The
    S&P 500 line is therefore only approximately comparable, and US holdings' true
    COP returns over this period were HIGHER than shown, since COP depreciated.
    """
    if not os.path.exists(_BENCH_PATH):
        return None
    s = pd.read_csv(_BENCH_PATH, index_col=0).iloc[:, 0]
    aligned = s.reindex(index_index)
    missing = aligned.isna().mean()
    if missing > 0.10:
        print(f"  WARNING: benchmark missing {missing:.0%} of dates -- skipping")
        return None
    return aligned.fillna(0.0)


def fetch_benchmark(cfg, ticker="^GSPC"):
    """Download the index once and cache it next to the other step-1 outputs."""
    from data_intake import download_batch
    res = download_batch([ticker], cfg)
    if res is None:
        raise RuntimeError(f"could not download {ticker}")
    close, _ = res
    rets = close.pct_change().iloc[1:]
    rets.columns = [ticker]
    rets.to_csv(_BENCH_PATH)
    return rets


# ---------------------------------------------------------------------------
# Forecast cache
# ---------------------------------------------------------------------------

def forecast_for_split(rets, cfg, split, cache_dir, n_runs):
    """mu and Sigma from data up to `split`, cached. Independent of strategy and s."""
    os.makedirs(cache_dir, exist_ok=True)
    key = (f"{cfg['transformer_arch']}_{cfg['transformer_loss']}_"
           f"pto{cfg['periods_to_forecast']}_n{n_runs}_{split}")
    path = os.path.join(cache_dir, key + ".npz")
    train = rets.iloc[:split]
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        names = [str(c) for c in z['names']]
        return pd.Series(z['mu'], index=names), pd.DataFrame(z['cov'], index=names,
                                                             columns=names)
    runs = train_runs(train, cfg, n_runs=n_runs, verbose=False,
                      arch=cfg['transformer_arch'])
    preds = pd.DataFrame(runs.mean(axis=0), columns=train.columns)
    preds = preds.iloc[:cfg['periods_to_forecast']]
    mu = weighted_mean_return(preds)
    cov = pd.DataFrame(LedoitWolf().fit(train).covariance_,
                       index=train.columns, columns=train.columns)
    np.savez_compressed(path, mu=mu.values, cov=cov.values,
                        names=np.array(list(train.columns), dtype=object))
    return mu, cov


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def model_weights(mu, cov, cfg, split, spread):
    """The production path: top-N pre-select, then Michaud consensus."""
    mu_t, cov_t = select_top_n(mu, cov, cfg.get('allocation_top_n'),
                               cfg.get('allocation_ranking', 'sharpe'))
    run_cfg = {**cfg, 'michaud_spread': spread, 'use_gradient': True}
    if spread == 0.0:
        run_cfg = {**run_cfg, 'michaud_mc_draws': 1}
    return allocate(mu_t, cov_t, run_cfg, split)


def build_strategies(mu, cov, cfg, split, hist, spreads, n_model_names):
    """All strategies for one window -> {label: weights}."""
    out = {}
    for s in spreads:
        out[f"model_s{s:g}"] = model_weights(mu, cov, cfg, split, s)
    out["ew_all"] = equal_weight_all(cov.index)
    out["ew_topn"] = equal_weight_topn(mu, n_model_names)
    out["gmv"] = gmv_weights(cov, cfg['max_weight'])
    out["inverse_vol"] = inverse_vol_weights(cov)
    out["momentum"] = momentum_weights(hist, n_model_names, MOMENTUM_LOOKBACK)
    return out


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_backtest(rets, cfg, cadence, spreads, n_runs, cache_dir, bench=None,
                 verbose=True):
    """One walk-forward pass. Returns (long DataFrame of per-window results,
    dict of per-window random-draw arrays)."""
    splits = rebalance_schedule(len(rets), cadence, cfg['periods_to_forecast'],
                                MIN_TRAIN)
    rows, randoms = [], {}
    prev = {}
    rng = np.random.default_rng(0)

    for split in splits:
        mu, cov = forecast_for_split(rets, cfg, split, cache_dir, n_runs)
        fwd = rets.iloc[split:split + cadence]
        hist = rets.iloc[:split]

        n_names = int((model_weights(mu, cov, cfg, split, spreads[0]) > 1e-9).sum())
        strategies = build_strategies(mu, cov, cfg, split, hist, spreads, n_names)

        # Luck control: many random books of the same size, same window.
        draws = []
        for _ in range(N_RANDOM):
            w = random_weights(list(cov.index), n_names, rng)
            held = w[w > 0]
            draws.append(float((1 + realised_returns(held, fwd)).prod() - 1))
        randoms[split] = np.array(draws)

        for label, w in strategies.items():
            held = w[w > 1e-9]
            r = realised_returns(held, fwd)
            total = float((1 + r).prod() - 1)
            row = {'cadence': cadence, 'split': split, 'strategy': label,
                   'realised_period_return': total, 'n_held': int(len(held)),
                   'max_weight': float(held.max()), 'effective_n': effective_n(held),
                   'random_percentile': random_percentile(total, randoms[split])}
            if label in prev:
                pw, pfwd = prev[label]
                row['turnover'] = turnover(pw, held)
                row['turnover_drift'] = turnover(drift_weights(pw, pfwd), held)
            rows.append(row)
            prev[label] = (held, fwd)

        if bench is not None:
            b = bench.iloc[split:split + cadence]
            rows.append({'cadence': cadence, 'split': split, 'strategy': 'sp500',
                         'realised_period_return': float((1 + b).prod() - 1),
                         'n_held': 1, 'max_weight': 1.0, 'effective_n': 1.0,
                         'turnover': 0.0, 'turnover_drift': 0.0,
                         'random_percentile': np.nan})
        if verbose:
            print(f"  cadence={cadence} split={split} done", flush=True)

    return pd.DataFrame(rows), randoms


def summarise(raw, cfg, baseline):
    """Per-strategy performance plus a PAIRED comparison against `baseline`."""
    ppy, rf = cfg['periods_per_year'], cfg['rf_rate']
    out = []
    for (cadence, label), g in raw.groupby(['cadence', 'strategy']):
        g = g.sort_values('split')
        r = g['realised_period_return'].reset_index(drop=True)
        yrs = len(r) * cadence / ppy
        cagr = (1 + r).prod() ** (1 / yrs) - 1
        vol = r.std(ddof=0) * np.sqrt(ppy / cadence)
        turn = g['turnover_drift'].mean()
        drag = annual_cost_drag(turn if pd.notna(turn) else 0.0, COST_ROUND_TRIP,
                                ppy, cadence)
        rec = {
            'cadence': cadence, 'strategy': label, 'n_windows': len(r),
            'cagr': cagr, 'ann_vol': vol,
            'sharpe': (cagr - rf) / vol if vol > 0 else np.nan,
            'net_cagr': cagr - drag,
            'net_sharpe': (cagr - drag - rf) / vol if vol > 0 else np.nan,
            'max_drawdown': max_drawdown(r), 'worst_window': float(r.min()),
            'turnover': turn, 'n_held': g['n_held'].mean(),
            'random_pct': g['random_percentile'].mean(),
        }
        base = raw[(raw['cadence'] == cadence) & (raw['strategy'] == baseline)]
        if label != baseline and len(base):
            b = base.sort_values('split')['realised_period_return'].reset_index(drop=True)
            rec.update({f'vs_{k}': v for k, v in
                        paired_comparison(r, b).items()
                        if k in ('mean_diff', 'p', 'wins', 't', 'n_for_80_power')})
        out.append(rec)
    return pd.DataFrame(out).sort_values(['cadence', 'net_sharpe'], ascending=[True, False])


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def write_outputs(raw, summary, out_dir, cfg, cadences, baseline):
    os.makedirs(out_dir, exist_ok=True)
    raw.to_csv(os.path.join(out_dir, 'raw.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

    # Results are deliberately NOT written to a tracked file. A markdown report in
    # the working tree looks current no matter how stale it is, so it misleads once
    # the model moves on. When a run drives a decision, record the numbers in the
    # commit message that changes params.yaml -- git history is timestamped and
    # immutable, so it reads unambiguously as "true as of this date".
    print("\nCaveats that apply to every number above:")
    for line in (
        "survivorship -- universe is today's catalogue filtered to full history,",
        "               so delisted names are absent; flatters stock-picking",
        "one regime   -- windows are post-2020 and mostly rising",
        "no FX        -- 01_download.py takes pct_change() on native-currency",
        "               prices; COP, USD, CLP, CHF summed as one unit",
        "power        -- bounded by calendar span, not window count: shorter",
        "               cadence gives more windows, proportionally less signal",
    ):
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cadence', default='24', help='comma-separated, e.g. 12,24')
    ap.add_argument('--spreads', default='0,1,2,4,6,8')
    ap.add_argument('--n-runs', type=int, default=50)
    ap.add_argument('--draws', type=int, default=None, help='override michaud_mc_draws')
    ap.add_argument('--baseline', default='ew_all')
    ap.add_argument('--fetch-benchmark', action='store_true',
                    help='download ^GSPC before running')
    ap.add_argument('--yes', action='store_true')
    args = ap.parse_args()

    cfg = load_config()
    if args.draws:
        cfg = {**cfg, 'michaud_mc_draws': args.draws}
    cadences = [int(c) for c in args.cadence.split(',')]
    spreads = [float(s) for s in args.spreads.split(',')]
    rets = pd.read_csv(PATHS['01_returns'], index_col=0)
    cache_dir = os.path.join(_OUT_DIR, "cache")

    if args.fetch_benchmark:
        print("Downloading ^GSPC...")
        fetch_benchmark(cfg)
    bench = load_benchmark_returns(rets.index, cfg=cfg)

    print(f"\n=== backtest ===")
    print(f"Universe: {rets.shape[1]} stocks, {len(rets)} periods")
    print(f"Cadences: {cadences} | spreads: {spreads} | baseline: {args.baseline}")
    print(f"arch={cfg['transformer_arch']} loss={cfg['transformer_loss']} "
          f"pto={cfg['periods_to_forecast']} n_runs={args.n_runs}")
    print(f"S&P 500 benchmark: {'loaded' if bench is not None else 'NOT AVAILABLE'}")
    total_splits = sum(len(rebalance_schedule(len(rets), c,
                                              cfg['periods_to_forecast'], MIN_TRAIN))
                       for c in cadences)
    print(f"Windows: {total_splits} across all cadences "
          f"({total_splits * args.n_runs} trainings if uncached)")
    if not args.yes:
        print("Proceed? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    frames = []
    for cadence in cadences:
        t0 = time.time()
        raw, _ = run_backtest(rets, cfg, cadence, spreads, args.n_runs, cache_dir,
                              bench=bench)
        frames.append(raw)
        print(f"cadence={cadence} finished in {(time.time()-t0)/60:.1f} min", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    summary = summarise(raw, cfg, args.baseline)
    write_outputs(raw, summary, _OUT_DIR, cfg, cadences, args.baseline)

    print("\n" + "=" * 100)
    print(summary.to_string(index=False))
    print(f"\nSaved: {_OUT_DIR}")


if __name__ == '__main__':
    main()
