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


# --- universe screen -----------------------------------------------------------

from config import _resolve_universe_config


def test_universe_topn_defaults_to_none_so_the_screen_is_off():
    # Off by default: the 80-stock universe needs no screen, and a null topn keeps
    # the pipeline byte-identical to its pre-screen behaviour.
    assert _resolve_universe_config({})['universe_topn'] is None


def test_universe_strata_must_sum_to_topn():
    with pytest.raises(ValueError, match="sum to universe_topn"):
        _resolve_universe_config({'universe_topn': 300,
                                  'universe_strata': [200, 60, 40, 25]})


def test_universe_strata_summing_correctly_is_accepted():
    cfg = _resolve_universe_config({'universe_topn': 300,
                                    'universe_strata': [200, 60, 40]})
    assert cfg['universe_strata'] == [200, 60, 40]


def test_universe_strata_ignored_when_screen_is_off():
    # strata left in params.yaml as documentation must not error while topn is null.
    cfg = _resolve_universe_config({'universe_topn': None,
                                    'universe_strata': [200, 60, 40]})
    assert cfg['universe_topn'] is None


def test_universe_topn_must_be_positive():
    with pytest.raises(ValueError, match="universe_topn"):
        _resolve_universe_config({'universe_topn': 0})


def test_real_params_universe_screen_is_valid():
    cfg = load_config()
    assert 'universe_topn' in cfg
    if cfg['universe_topn'] and cfg.get('universe_strata'):
        assert sum(cfg['universe_strata']) == cfg['universe_topn']
