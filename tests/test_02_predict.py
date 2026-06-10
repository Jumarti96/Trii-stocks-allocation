"""Smoke test for pipeline/02_predict.py: it passes the configured arch and slices
the forecast to periods_to_forecast. train_and_predict is stubbed so the test is fast
and genuinely discriminating (fails on code that ignores arch or skips the slice)."""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))


def _load_script():
    path = os.path.join(ROOT, "pipeline", "02_predict.py")
    spec = importlib.util.spec_from_file_location("step02_predict", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step2_passes_arch_and_slices_to_periods_to_forecast(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.02, (60, 4)),
                        columns=[f"S{i}" for i in range(4)],
                        index=pd.date_range("2020-01-05", periods=60, freq="W-SUN").astype(str))
    prices = (1 + rets).cumprod() * 100

    mod = _load_script()

    cfg = mod.load_config()
    cfg['transformer_arch'] = 'B'
    cfg['transformer_forecast_window'] = 24
    cfg['periods_to_forecast'] = 4

    paths = {
        '01_prices': str(tmp_path / "01_prices.csv"),
        '01_returns': str(tmp_path / "01_returns.csv"),
        '02_expected_returns': str(tmp_path / "02_er.csv"),
        '02_covmat': str(tmp_path / "02_cov.csv"),
        '02_predictions': str(tmp_path / "02_pred.csv"),
        '02_metadata': str(tmp_path / "02_meta.json"),
    }
    prices.to_csv(paths['01_prices'])
    rets.to_csv(paths['01_returns'])

    captured = {}

    def fake_train_and_predict(returns_df, cfg, n_runs=None, verbose=True, arch='current'):
        # Record the arch the script passed, and emulate a direct multi-step head
        # emitting transformer_forecast_window rows (e.g. 24).
        captured['arch'] = arch
        n_rows = cfg['transformer_forecast_window']
        return pd.DataFrame(np.zeros((n_rows, returns_df.shape[1])),
                            columns=returns_df.columns)

    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "PATHS", paths)
    monkeypatch.setattr(mod, "train_and_predict", fake_train_and_predict)

    mod.main()

    # 1. The script must forward the configured arch (old code passed none -> 'current').
    assert captured['arch'] == 'B'
    # 2. The 24-row head output must be sliced down to periods_to_forecast (=4).
    preds = pd.read_csv(paths['02_predictions'], index_col=0)
    assert len(preds) == 4
