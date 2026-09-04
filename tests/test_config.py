"""Tests for transformer-config defaults and guard in pipeline/config.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from config import _resolve_transformer_config


def _base_cfg(**over):
    cfg = {'periods_per_year': 54, 'periods_to_forecast': 4}
    cfg.update(over)
    return cfg


def test_default_arch_is_current():
    cfg = _resolve_transformer_config(_base_cfg())
    assert cfg['transformer_arch'] == 'current'


def test_forecast_window_defaults_to_periods_per_year():
    cfg = _resolve_transformer_config(_base_cfg(transformer_arch='B'))
    assert cfg['transformer_forecast_window'] == 54


def test_explicit_forecast_window_preserved():
    cfg = _resolve_transformer_config(
        _base_cfg(transformer_arch='B', transformer_forecast_window=24))
    assert cfg['transformer_forecast_window'] == 24


def test_guard_rejects_window_below_slice():
    with pytest.raises(ValueError, match="must be >= periods_to_forecast"):
        _resolve_transformer_config(
            _base_cfg(transformer_arch='B',
                      transformer_forecast_window=2, periods_to_forecast=4))


def test_guard_ignores_current_arch():
    # current is autoregressive — forecast_window does not apply, no guard error
    cfg = _resolve_transformer_config(
        _base_cfg(transformer_arch='current', periods_to_forecast=4))
    assert cfg['transformer_arch'] == 'current'


from config import load_config


def test_real_params_load_with_B_arch():
    cfg = load_config()
    assert cfg['transformer_arch'] == 'B'
    assert cfg['transformer_forecast_window'] == 24
    assert cfg['transformer_forecast_window'] >= cfg['periods_to_forecast']


def test_transformer_loss_defaults_to_auto():
    cfg = _resolve_transformer_config(_base_cfg())
    assert cfg['transformer_loss'] == 'auto'


def test_explicit_transformer_loss_preserved():
    cfg = _resolve_transformer_config(_base_cfg(transformer_loss='rank_ic'))
    assert cfg['transformer_loss'] == 'rank_ic'


def test_transformer_loss_guard_rejects_unknown():
    with pytest.raises(ValueError):
        _resolve_transformer_config(_base_cfg(transformer_loss='nonsense'))


def test_real_params_select_rank_ic_loss():
    assert load_config()['transformer_loss'] == 'rank_ic'


def test_rank_ic_requires_consumed_horizon_to_match_decode_window():
    # rank_ic_loss ranks the cumulative return over the FULL decode window, but
    # 02_predict.py slices predictions to periods_to_forecast. If those differ,
    # the loss optimises a horizon production never consumes -- measured as a
    # real ICIR regression at h=4 (0.600 -> 0.358) while h=24 improved.
    with pytest.raises(ValueError, match="rank_ic"):
        _resolve_transformer_config(_base_cfg(
            transformer_arch='B', transformer_loss='rank_ic',
            transformer_forecast_window=24, periods_to_forecast=4))


def test_rank_ic_accepts_matching_horizons():
    cfg = _resolve_transformer_config(_base_cfg(
        transformer_arch='B', transformer_loss='rank_ic',
        transformer_forecast_window=24, periods_to_forecast=24))
    assert cfg['periods_to_forecast'] == cfg['transformer_forecast_window']


def test_auto_loss_still_allows_shorter_consumed_horizon():
    # The pointwise losses are per-step, so consuming a prefix is legitimate.
    cfg = _resolve_transformer_config(_base_cfg(
        transformer_arch='B', transformer_loss='auto',
        transformer_forecast_window=24, periods_to_forecast=4))
    assert cfg['periods_to_forecast'] == 4


def test_real_params_horizon_is_aligned():
    cfg = load_config()
    assert cfg['periods_to_forecast'] == 24
    assert cfg['periods_to_forecast'] == cfg['transformer_forecast_window']
