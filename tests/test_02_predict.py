"""Smoke test for pipeline/02_predict.py: B arch output is sliced to periods_to_forecast."""
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


def test_step2_slices_B_output_to_periods_to_forecast(tmp_path, monkeypatch):
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
    cfg['time_window'] = 8
    cfg['n_transformer_runs'] = 1
    cfg['transformer_epochs'] = 1
    cfg['transformer_warmup_epochs'] = 0

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

    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "PATHS", paths)

    mod.main()

    preds = pd.read_csv(paths['02_predictions'], index_col=0)
    assert len(preds) == 4   # sliced from 24 down to periods_to_forecast
