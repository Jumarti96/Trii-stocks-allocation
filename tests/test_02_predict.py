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
    # 521 periods = the production history length, giving 444 training samples.
    # (Was 60, which capacity_report correctly rejects as untrainable -- stubbing
    # the trainer had hidden that the fixture described an impossible config.)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.02, (521, 4)),
                        columns=[f"S{i}" for i in range(4)],
                        index=pd.date_range("2016-01-03", periods=521, freq="W-SUN").astype(str))
    prices = (1 + rets).cumprod() * 100

    mod = _load_script()

    cfg = mod.load_config()
    cfg['transformer_arch'] = 'B'
    cfg['transformer_forecast_window'] = 24
    cfg['periods_to_forecast'] = 4
    cfg['universe_topn'] = None      # this test is about arch forwarding, not the screen

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


def _fixture(tmp_path, n_stocks=8, n_periods=521):
    rng = np.random.default_rng(1)
    cols = [f"S{i}" for i in range(n_stocks)]
    idx = pd.date_range("2016-01-03", periods=n_periods, freq="W-SUN").astype(str)
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_periods, n_stocks)), columns=cols, index=idx)
    prices = (1 + rets).cumprod() * 100
    # Volume descending in column order, so S0 is the most liquid name. The spread
    # is 100x per rank because ADV is price*volume, and 521 weeks of random walk at
    # 2% weekly vol can move relative prices ~10x -- a narrow volume spread would
    # let price drift reorder the ranking and make the assertion flaky.
    volume = pd.DataFrame({c: [100.0 ** (n_stocks - i)] * n_periods
                           for i, c in enumerate(cols)}, index=idx)
    fx = pd.DataFrame({"USD": [1.0] * n_periods}, index=idx)
    cur = pd.DataFrame({"currency": "USD", "unit_factor": 1.0}, index=cols)

    paths = {k: str(tmp_path / f"{k}.csv") for k in
             ('01_prices', '01_returns', '01_volume', '01_fx', '01_currency',
              '02_expected_returns', '02_covmat', '02_predictions')}
    paths['02_metadata'] = str(tmp_path / "02_meta.json")
    prices.to_csv(paths['01_prices']); rets.to_csv(paths['01_returns'])
    volume.to_csv(paths['01_volume']); fx.to_csv(paths['01_fx'])
    cur.to_csv(paths['01_currency'])
    return paths, cols


def _run(mod, monkeypatch, cfg, paths):
    seen = {}

    def fake(returns_df, cfg, n_runs=None, verbose=True, arch='current'):
        seen['universe'] = list(returns_df.columns)
        return pd.DataFrame(np.zeros((cfg['transformer_forecast_window'],
                                      returns_df.shape[1])), columns=returns_df.columns)

    monkeypatch.setattr(mod, "load_config", lambda: cfg)
    monkeypatch.setattr(mod, "PATHS", paths)
    monkeypatch.setattr(mod, "train_and_predict", fake)
    mod.main()
    return seen['universe']


def test_step2_screen_is_a_noop_when_universe_topn_is_null(tmp_path, monkeypatch):
    # Gate 3: the default path must be byte-identical to the pre-screen pipeline.
    mod = _load_script()
    paths, cols = _fixture(tmp_path)
    cfg = mod.load_config()
    cfg.update(transformer_arch='B', transformer_forecast_window=24,
               periods_to_forecast=24, universe_topn=None)
    assert _run(mod, monkeypatch, cfg, paths) == cols


def test_step2_screen_narrows_universe_when_topn_set(tmp_path, monkeypatch):
    mod = _load_script()
    paths, cols = _fixture(tmp_path)
    cfg = mod.load_config()
    cfg.update(transformer_arch='B', transformer_forecast_window=24,
               periods_to_forecast=24, universe_topn=3, universe_strata=None)
    # S0..S2 are the most liquid, and mu/Sigma must be built on the subset only.
    assert _run(mod, monkeypatch, cfg, paths) == cols[:3]
    assert len(pd.read_csv(paths['02_covmat'], index_col=0)) == 3


def test_step2_refuses_a_universe_it_cannot_train(tmp_path, monkeypatch):
    import pytest
    mod = _load_script()
    # 120 periods leaves 43 training samples against ~600k parameters (14,061 each),
    # past the hard limit. Tests the wiring; the thresholds themselves are pinned in
    # test_transformer_model.test_capacity_report_verdicts_track_thresholds.
    paths, _ = _fixture(tmp_path, n_stocks=8, n_periods=120)
    cfg = mod.load_config()
    cfg.update(transformer_arch='B', transformer_forecast_window=24,
               periods_to_forecast=24, universe_topn=None)
    with pytest.raises(ValueError, match="parameters per training sample"):
        _run(mod, monkeypatch, cfg, paths)
