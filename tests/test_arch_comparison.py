"""Tests for experiments/arch_comparison.py"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from arch_comparison import (
    compute_spearman_rho,
    compute_icir,
    compute_topk_precision,
    compute_hit_rate,
    predict_zero,
    predict_momentum,
    predict_persistence,
    predict_mean_reversion,
)


def test_compute_spearman_rho_perfect():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_spearman_rho(x, x) == pytest.approx(1.0)


def test_compute_spearman_rho_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_spearman_rho(x, x[::-1]) == pytest.approx(-1.0)


def test_compute_spearman_rho_constant_returns_zero():
    x = np.ones(5)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # constant predicted → no rank signal → 0.0
    assert compute_spearman_rho(x, y) == pytest.approx(0.0)


def test_compute_icir_known_values():
    # mean=0.2, std(ddof=1)=0.1 => ICIR = 2.0
    rho_series = np.array([0.1, 0.2, 0.3])
    assert compute_icir(rho_series) == pytest.approx(2.0)


def test_compute_icir_zero_std_returns_zero():
    # constant rho → std=0 → ICIR=0 (not inf)
    assert compute_icir(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_compute_topk_precision_perfect():
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    realized  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_topk_precision(predicted, realized, k=3) == pytest.approx(1.0)


def test_compute_topk_precision_disjoint():
    # predicted top-2: indices 3,4 (values 4,5); realized top-2: indices 0,1 (values 5,4)
    predicted = np.array([0.0, 0.0, 0.0, 4.0, 5.0])
    realized  = np.array([5.0, 4.0, 0.0, 0.0, 0.0])
    assert compute_topk_precision(predicted, realized, k=2) == pytest.approx(0.0)


def test_compute_hit_rate_all_correct():
    pred = np.array([1.0, -1.0, 2.0])
    real = np.array([0.5, -0.5, 1.0])
    assert compute_hit_rate(pred, real) == pytest.approx(1.0)


def test_compute_hit_rate_all_wrong():
    pred = np.array([1.0, -1.0, 2.0])
    real = np.array([-0.5, 0.5, -1.0])
    assert compute_hit_rate(pred, real) == pytest.approx(0.0)


def test_compute_hit_rate_with_zeros_in_denominator():
    # Zero predictions count as misses; denominator is always N
    pred = np.array([1.0, 0.0, -1.0])   # middle stock: no direction
    real = np.array([0.5, 0.5,  -0.5])
    # matches: [True, False, True] → 2/3
    assert compute_hit_rate(pred, real) == pytest.approx(2.0 / 3.0)


def test_compute_spearman_rho_constant_realized_returns_zero():
    # constant realized → no rank signal → 0.0 (symmetric guard)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.ones(5)
    assert compute_spearman_rho(x, y) == pytest.approx(0.0)




def test_predict_zero_shape_and_values():
    result = predict_zero(7)
    assert result.shape == (7,)
    np.testing.assert_array_equal(result, 0.0)


def test_predict_momentum_shape():
    window = np.random.default_rng(0).normal(0, 0.02, (20, 6))
    result = predict_momentum(window)
    assert result.shape == (6,)


def test_predict_momentum_weights_recency():
    # Stock 0: spike only in most-recent row → should score highest (momentum)
    # Stock 1: spike only in oldest row       → should score lowest
    window = np.zeros((10, 2))
    window[-1, 0] = 0.10   # newest row
    window[ 0, 1] = 0.10   # oldest row
    result = predict_momentum(window)
    assert result[0] > result[1], "recent spike must outweigh old spike"


def test_predict_persistence_equals_last_row():
    window = np.random.default_rng(1).normal(0, 0.02, (15, 5))
    result = predict_persistence(window)
    np.testing.assert_array_equal(result, window[-1])


def test_predict_mean_reversion_sign():
    # Stock above cross-section → negative prediction; stock below → positive
    window = np.zeros((5, 4))
    window[-1] = [0.10, 0.01, 0.01, -0.10]  # last period: stock 0 high, stock 3 low
    result = predict_mean_reversion(window)
    assert result[0] < 0   # above cross-section → predict downward
    assert result[3] > 0   # below cross-section → predict upward


from arch_comparison import run_one_block
from config import load_config


def _tiny_run_cfg():
    cfg = load_config()
    cfg['time_window']             = 8
    cfg['periods_to_forecast']     = 4
    cfg['transformer_epochs']      = 1
    cfg['transformer_warmup_epochs'] = 0
    return cfg


def _tiny_rets_df(seed, n_stocks=6, n_periods=40):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.02, (n_periods, n_stocks)),
        columns=[f"S{i}" for i in range(n_stocks)],
    )


def test_run_one_block_returns_expected_keys():
    cfg   = _tiny_run_cfg()
    rets  = _tiny_rets_df(0)
    train = rets.iloc[:-10]
    test  = rets.iloc[-10:]
    result = run_one_block('A_surgical', train, test, cfg, n_runs=1, horizons=[4])
    assert (4, 'rank_ic') in result
    assert (4, 'topk_precision') in result
    assert (4, 'hit_rate') in result
    # benchmark keys
    assert (4, 'rank_ic_B2_momentum') in result


def test_run_one_block_B4_unsupported_horizon_is_nan():
    cfg   = _tiny_run_cfg()
    rets  = _tiny_rets_df(1)
    train = rets.iloc[:-10]
    test  = rets.iloc[-10:]
    result = run_one_block('B_4', train, test, cfg, n_runs=1, horizons=[4, 24])
    # horizon 4 should be valid
    assert not np.isnan(result[(4, 'rank_ic')])
    # horizon 24 is beyond B_4's 4-step output
    assert np.isnan(result[(24, 'rank_ic')])


def test_run_one_block_current_smoke():
    cfg   = _tiny_run_cfg()
    rets  = _tiny_rets_df(2)
    train = rets.iloc[:-10]
    test  = rets.iloc[-10:]
    result = run_one_block('current', train, test, cfg, n_runs=1, horizons=[4])
    # result values should be finite floats in valid metric ranges
    assert -1.0 <= result[(4, 'rank_ic')] <= 1.0
    assert  0.0 <= result[(4, 'topk_precision')] <= 1.0
    assert  0.0 <= result[(4, 'hit_rate')] <= 1.0


import tempfile
from arch_comparison import aggregate_results, write_outputs


def _stub_block_results(rho_vals, horizons=None):
    """Return a list of per-block result dicts with controlled rank_ic values."""
    if horizons is None:
        horizons = [4]
    blocks = []
    for rho in rho_vals:
        r = {}
        for h in horizons:
            r[(h, 'rank_ic')]        = rho
            r[(h, 'topk_precision')] = 0.5
            r[(h, 'hit_rate')]       = 0.5
            r[(h, 'rank_ic_B1_zero')]        = 0.0
            r[(h, 'rank_ic_B2_momentum')]    = 0.1
            r[(h, 'rank_ic_B3_persistence')] = 0.05
            r[(h, 'rank_ic_B4_mean_rev')]    = -0.05
        blocks.append(r)
    return blocks


def test_aggregate_results_mean_rank_ic():
    block_results = _stub_block_results([0.1, 0.2, 0.3])
    agg = aggregate_results(block_results, horizons=[4])
    assert agg[(4, 'mean_rank_ic')] == pytest.approx(0.2)


def test_aggregate_results_std_rank_ic():
    block_results = _stub_block_results([0.1, 0.2, 0.3])
    agg = aggregate_results(block_results, horizons=[4])
    expected_std = np.array([0.1, 0.2, 0.3]).std(ddof=1)
    assert agg[(4, 'std_rank_ic')] == pytest.approx(expected_std)


def test_aggregate_results_icir():
    rho_vals = [0.1, 0.2, 0.3]
    block_results = _stub_block_results(rho_vals)
    agg = aggregate_results(block_results, horizons=[4])
    expected_icir = np.mean(rho_vals) / np.std(rho_vals, ddof=1)
    assert agg[(4, 'icir')] == pytest.approx(expected_icir)


def test_write_outputs_creates_all_files():
    # Build minimal multi-arch, multi-seed results structure
    block_results = _stub_block_results([0.1, 0.2])
    agg = aggregate_results(block_results, horizons=[4])
    results = {
        ('A_surgical', 0): agg,
        ('current',    0): agg,
    }
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(results, horizons=[4], out_dir=tmp)
        files = set(os.listdir(tmp))
    assert 'rank_ic.csv'           in files
    assert 'topk_precision.csv'    in files
    assert 'hit_rate.csv'          in files
    assert 'arch_comparison_summary.txt' in files


def test_write_outputs_rank_ic_columns():
    block_results = _stub_block_results([0.1, 0.2])
    agg = aggregate_results(block_results, horizons=[4])
    results = {('A_surgical', 0): agg}
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(results, horizons=[4], out_dir=tmp)
        df = pd.read_csv(os.path.join(tmp, 'rank_ic.csv'))
    assert set(df.columns) >= {'arch', 'seed', 'horizon',
                                'mean_rank_ic', 'std_rank_ic', 'icir'}
