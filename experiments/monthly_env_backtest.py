"""
Monthly environment-robustness backtest: run the realized walk-forward backtest in a MONTHLY
regime at short (1-month) and long (6-month) horizons, comparing the allocation methods
current (msr) / parametric s4 / parametric s8 against equal_weight, with the technical filter
DISABLED. Reports realized metrics as mean +/- cross-seed std per horizon.

Reuses experiments/backtest_allocation.run_backtest unchanged; reads MONTHLY data fetched by
experiments/fetch_monthly_data.py. This is a realized out-of-sample test (the verdict is realized
Sharpe etc.), complementary to the weekly multi-seed backtest. In-sample caveats do not apply;
the few-block 6-month horizon is directional, not a verdict.

Run:  python experiments/monthly_env_backtest.py --horizons 1,6 --seeds 0,100 \
        --oos-periods 60 --n-runs 75 --mc-draws 1000 --spreads 4,8
Requires experiments/data_monthly/01_{prices,returns}.csv (run fetch_monthly_data.py first).

See docs/superpowers/specs/2026-06-04-monthly-env-backtest-design.md.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
sys.path.insert(0, HERE)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import load_config
from backtest_allocation import run_backtest, summarize_arm, write_backtest_outputs
from fetch_monthly_data import MONTHLY_PRICES, MONTHLY_RETURNS

METRIC_COLS = ["cum_return", "ann_return", "ann_vol", "sharpe", "max_dd",
               "mean_turnover", "avg_names", "hit_rate"]


def build_monthly_cfg(weekly_cfg, horizon):
    """Copy the weekly production cfg and override the monthly + horizon fields.

    Sets interval/periods_per_year/time_window/period_freq/future_freq/date_offset to monthly,
    periods_to_forecast to the horizon, and recomputes the per-period risk-free rate for ppy=12.
    Filter/MA/weight params are left untouched (the filter is bypassed via select_all, not config).
    """
    cfg = dict(weekly_cfg)
    cfg["interval"] = "1mo"
    cfg["periods_per_year"] = 12
    cfg["time_window"] = 12
    cfg["periods_to_forecast"] = horizon
    cfg["period_freq"] = "M"
    cfg["future_freq"] = "MS"
    cfg["date_offset"] = pd.DateOffset(months=1)
    cfg["rf_period"] = (1 + cfg["rf_rate"]) ** (1 / 12) - 1
    return cfg


def select_all(prices, rets, cfg):
    """Filter-disabling selection seam: every name is eligible (no technical gate)."""
    return list(rets.columns)
