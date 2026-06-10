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
