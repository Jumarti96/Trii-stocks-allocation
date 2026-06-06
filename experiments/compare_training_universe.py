"""
Before/after comparison: does training on the full universe reduce the optimistic bias?

Trains the Transformer two ways on the SAME data and seed, then compares the annualised
expected returns for the filtered (selected) names:

  old regime: train only on the filtered selection      (the pre-change behaviour)
  new regime: train on the full universe, read off the selected names  (the new behaviour)

If the decoupling worked, the new regime's expected returns for the selected names should be
lower / less optimistic than the old regime's.

Run:  python experiments/compare_training_universe.py --runs 100

-------------------------------------------------------------------------------------------
FINDING (100 runs/regime, seed=42): the hypothesis did NOT hold. Full-universe training does
not lower the selected-name forecasts; it slightly RAISES them.
    weighted mean per-period return:  old 0.0025  ->  new 0.0033
    annualised expected return:       old 0.156   ->  new 0.206
Cross-stock spread is unchanged and per-stock effects are mixed. The feared "endpoint
selection" inflation is weak because the model trains on each stock's full history, not just
its recent uptrend. Full-universe training is kept as the cleaner/unbiased procedure, not as
an optimism-reduction technique. To actually lower optimism, calibrate the expected returns.
-------------------------------------------------------------------------------------------
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, 'pipeline'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch

from config import load_config, PATHS
from transformer_model import (
    train_and_predict, annualize_expected_returns, weighted_mean_return,
)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(name, er):
    return {'regime': name, 'mean': er.mean(), 'median': er.median(),
            'std': er.std(), 'min': er.min(), 'max': er.max()}


def report(title, old, new):
    """Print a summary table + verdict for one metric (old vs new)."""
    summary = pd.DataFrame([summarize('old (filtered)', old),
                            summarize('new (full universe)', new)]).set_index('regime')
    print(f"\n--- {title} (SELECTED names) ---")
    print(summary.to_string())
    drop    = old.mean() - new.mean()
    verdict = "LOWER (optimism reduced)" if drop > 0 else "HIGHER (no reduction)"
    print(f"  mean: old={old.mean():.4f}  new={new.mean():.4f}  "
          f"=> new is {verdict} by {drop:+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=10,
                        help='Transformer runs per regime (default 10; lower = faster)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()

    rets_full = pd.read_csv(PATHS['01_returns'], index_col=0)
    if '03_selected_returns' not in PATHS:
        raise SystemExit(
            "This experiment requires the step-03 filter outputs which were removed in PR-2. "
            "It is obsolete and can no longer be run."
        )
    selected  = pd.read_csv(PATHS['03_selected_returns'], index_col=0, nrows=0).columns
    selected  = [s for s in selected if s in rets_full.columns]
    ppy       = cfg['periods_per_year']

    print(f"Full universe: {rets_full.shape[1]} stocks | Selected: {len(selected)} | "
          f"runs={args.runs}\n")

    # OLD regime: train only on the filtered selection
    seed_everything(args.seed)
    print("=== OLD regime: training on the filtered selection only ===")
    preds_old = train_and_predict(rets_full[selected], cfg, n_runs=args.runs, verbose=False)

    # NEW regime: train on the full universe, read off the selected names
    seed_everything(args.seed)
    print("=== NEW regime: training on the full universe ===")
    preds_new = train_and_predict(rets_full, cfg, n_runs=args.runs, verbose=False)
    preds_new = preds_new[selected]

    # Primary signal: weighted mean per-period return (pre-annualisation, low noise)
    wmr_old, wmr_new = weighted_mean_return(preds_old), weighted_mean_return(preds_new)
    # Secondary: annualised expected return (what the pipeline ultimately uses)
    er_old,  er_new  = annualize_expected_returns(preds_old, ppy), annualize_expected_returns(preds_new, ppy)

    pd.set_option('display.float_format', lambda v: f"{v:.4f}")
    report("Weighted mean per-period return [PRIMARY]", wmr_old, wmr_new)
    report("Annualised expected return", er_old, er_new)

    per_stock = pd.DataFrame({
        'wmr_old': wmr_old, 'wmr_new': wmr_new, 'wmr_delta': wmr_new - wmr_old,
        'er_old': er_old, 'er_new': er_new,
    }).sort_values('wmr_delta')
    print("\n--- Per-stock, sorted by weighted-mean delta (new - old) ---")
    print(per_stock.to_string())


if __name__ == '__main__':
    main()
