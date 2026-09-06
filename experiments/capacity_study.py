"""How large a universe can this architecture actually forecast?

Stocks are the model's feature axis, so widening the universe adds parameters (~3,224
per stock) against a FIXED number of training samples -- a sample is a time window and
every stock shares one time axis. This measures where that stops working.

Metric: out-of-sample Spearman rank IC of the cumulative `H`-period forecast against
what actually happened, i.e. the quantity rank_ic trains on and the optimiser consumes.
Two views:

  IC_own    : IC over the model's OWN universe. The real use case -- a user screens to
              n names and the model ranks those n.
  IC_top80  : IC restricted to the 80 most liquid names, held FIXED across every n.
              The controlled comparison: same stocks scored every time, so only the
              training universe changes.

Baseline: H-period trailing momentum on the same evaluation set, so an IC has a
reference point rather than floating free.

Also reports `disp`, the ratio of predicted to actual cross-sectional dispersion. This
turned out to be the cleaner signal: it is monotone in n where IC is noisy, and it
matters directly, because michaud_spread is calibrated against mu's scale.

Findings on a 3,033-stock global catalogue (4 splits, 8 runs, H=24, momentum IC +0.050):

    n=80    IC_top80 +0.146   disp 0.42
    n=150   IC_top80 +0.182   disp 0.54
    n=300   IC_top80 +0.114   disp 0.55
    n=600   IC_top80 +0.133   disp 0.61
    n=1200  IC_top80 -0.008   disp 2.08   <-- degrades here
    n=3033  IC_top80 +0.042   disp 5.35

Up to ~600 the forecast is indistinguishable from the 80-stock configuration and well
above the naive baseline. Past that it falls TO the baseline while predicting a spread
2-5x wider than reality -- overfitting. IC differences below n=600 are inside noise
(se +-0.04..0.11); the dispersion trend is not.

Env: CAPACITY_DATA (default data/), CAPACITY_SIZES, CAPACITY_RUNS, CAPACITY_SPLITS.
Run: .venv/Scripts/python.exe -u experiments/capacity_study.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Defaults to the pipeline's own data/. Point CAPACITY_DATA elsewhere to study a
# different catalogue without disturbing the working dataset.
OUT = os.environ.get("CAPACITY_DATA", os.path.join(ROOT, "data"))
RESULTS = os.path.join(ROOT, "experiments", "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import load_config
from data_intake import select_universe
from transformer_model import train_runs, capacity_report, describe_device

H = 24
N_RUNS = int(os.environ.get("CAPACITY_RUNS", 8))
SIZES = [int(x) for x in
         os.environ.get("CAPACITY_SIZES", "80,150,300,600,1200").split(",")]
SPLITS = int(os.environ.get("CAPACITY_SPLITS", 4))

cfg = load_config()
cfg.update(transformer_arch="B", transformer_loss="rank_ic",
           transformer_forecast_window=H, periods_to_forecast=H)

rets = pd.read_csv(os.path.join(OUT, "01_returns.csv"), index_col=0)
close = pd.read_csv(os.path.join(OUT, "01_prices.csv"), index_col=0)
volume = pd.read_csv(os.path.join(OUT, "01_volume.csv"), index_col=0)
cur = pd.read_csv(os.path.join(OUT, "01_currency.csv"), index_col=0)
fx = pd.read_csv(os.path.join(OUT, "01_fx.csv"), index_col=0)
cmap, ufac = cur["currency"].to_dict(), cur["unit_factor"].to_dict()

# Clamp to what exists and drop duplicates: asking for 1200 names from an 80-stock
# catalogue would otherwise run several identical arms and read as a flat curve.
SIZES = sorted({min(s, rets.shape[1]) for s in SIZES})
N_EVAL = min(80, rets.shape[1])

print(describe_device(), flush=True)
print(f"{rets.shape[1]} stocks x {rets.shape[0]} periods | H={H} "
      f"n_runs={N_RUNS} splits={SPLITS} sizes={SIZES}\n", flush=True)

os.makedirs(RESULTS, exist_ok=True)
rows = []
for split in range(SPLITS):
    end = len(rets) - split * H          # test window is [end-H, end)
    train = rets.iloc[:end - H]
    test = rets.iloc[end - H:end]
    as_of = close.index[end - H - 1]     # universe chosen with no look-ahead

    actual_all = (1 + test).prod() - 1
    ranked = select_universe(close.iloc[:end - H], volume.iloc[:end - H], max(SIZES),
                             window=52, fx=fx, cur_map=cmap, unit_factors=ufac,
                             as_of=as_of)
    top80 = ranked[:N_EVAL]

    mom = (1 + rets.iloc[end - H - H:end - H]).prod() - 1
    base_own = spearmanr(mom[ranked], actual_all[ranked]).statistic
    base_80 = spearmanr(mom[top80], actual_all[top80]).statistic
    rows.append(dict(split=split, n="momentum", ic_own=base_own, ic_top80=base_80,
                     ratio=np.nan, secs=0.0))
    print(f"[split {split}] as_of={as_of}  momentum baseline: "
          f"IC_own {base_own:+.4f}  IC_top80 {base_80:+.4f}", flush=True)

    for n in SIZES:
        uni = ranked[:n]
        sub = train[uni]
        t0 = time.time()
        preds = train_runs(sub, cfg, n_runs=N_RUNS, verbose=False, arch="B")
        secs = time.time() - t0

        pred_cum = pd.Series(preds.mean(axis=0).sum(axis=0), index=uni)
        ic_own = spearmanr(pred_cum, actual_all[uni]).statistic
        ic_80 = spearmanr(pred_cum[top80], actual_all[top80]).statistic
        ratio = float(np.std(pred_cum.values) / np.std(actual_all[uni].values))
        cap = capacity_report(len(uni), len(sub), cfg)

        rows.append(dict(split=split, n=len(uni), ic_own=ic_own, ic_top80=ic_80,
                         ratio=ratio, secs=secs, pps=cap["params_per_sample"]))
        print(f"[split {split}] n={len(uni):>5}  IC_own {ic_own:+.4f}  IC_top80 {ic_80:+.4f}"
              f"  disp_ratio {ratio:5.2f}  {cap['params_per_sample']:>6,.0f} p/s"
              f"  {secs:5.1f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "capacity_study.csv"), index=False)

print("\n=== MEAN ACROSS SPLITS ===", flush=True)
agg = df.groupby("n", dropna=False).agg(
    ic_own=("ic_own", "mean"), ic_own_sd=("ic_own", "std"),
    ic_top80=("ic_top80", "mean"), ic_top80_sd=("ic_top80", "std"),
    disp=("ratio", "mean"), secs=("secs", "mean")).reset_index()
print(agg.to_string(index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
