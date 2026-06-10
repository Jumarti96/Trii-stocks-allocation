import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from seed_stability import (
    compute_mu_per_run,
    compute_topk_sets,
    compute_pairwise_overlaps,
    compute_stock_frequencies,
    compute_core_sets,
    write_outputs,
)


def test_compute_mu_per_run_shape():
    preds = np.ones((5, 3, 10))
    result = compute_mu_per_run(preds)
    assert result.shape == (5, 10)


def test_compute_mu_per_run_decay_weights():
    # 2 periods: w1=exp(-0.2), w2=exp(-0.4); period 1 heavier than period 2
    preds = np.zeros((1, 2, 1))
    preds[0, 0, 0] = 1.0  # period 1 only
    preds[0, 1, 0] = 0.0
    w1 = np.exp(-0.2)
    w2 = np.exp(-0.4)
    expected = w1 / (w1 + w2)
    result = compute_mu_per_run(preds)
    assert result[0, 0] == pytest.approx(expected)


def test_compute_topk_sets_keys():
    rng = np.random.default_rng(0)
    mu_per_run = rng.standard_normal((5, 10))
    sigma_lw = np.ones(10)
    result = compute_topk_sets(mu_per_run, sigma_lw, n_arms=[3, 5], thresholds=[3])
    assert set(result.keys()) == {(3, 3), (5, 3)}


def test_compute_topk_sets_set_size():
    rng = np.random.default_rng(1)
    mu_per_run = rng.standard_normal((10, 20))
    sigma_lw = np.ones(20)
    result = compute_topk_sets(mu_per_run, sigma_lw, n_arms=[5, 10], thresholds=[3, 7])
    for (n, k), s in result.items():
        assert isinstance(s, frozenset)
        assert len(s) == k


def test_compute_topk_sets_ranking():
    # highest-scoring stocks must be in top-k
    n_stocks = 10
    mu_per_run = np.zeros((3, n_stocks))
    mu_per_run[:, 0] = 10.0  # stock 0 always top
    mu_per_run[:, 1] = 9.0   # stock 1 second
    sigma_lw = np.ones(n_stocks)
    result = compute_topk_sets(mu_per_run, sigma_lw, n_arms=[3], thresholds=[2])
    assert result[(3, 2)] == frozenset([0, 1])


def test_compute_topk_sets_k_gt_n_stocks_raises():
    mu_per_run = np.ones((3, 5))
    sigma_lw = np.ones(5)
    with pytest.raises(ValueError):
        compute_topk_sets(mu_per_run, sigma_lw, n_arms=[3], thresholds=[10])


def test_compute_pairwise_overlaps_identical():
    s = frozenset([0, 1, 2])
    seeds = [{(5, 3): s}, {(5, 3): s}, {(5, 3): s}]
    result = compute_pairwise_overlaps(seeds)
    mean, std, mn, mx = result[(5, 3)]
    assert mean == pytest.approx(1.0)
    assert mn  == pytest.approx(1.0)
    assert mx  == pytest.approx(1.0)


def test_compute_pairwise_overlaps_disjoint():
    seeds = [
        {(4, 2): frozenset([0, 1])},
        {(4, 2): frozenset([2, 3])},
        {(4, 2): frozenset([4, 5])},
    ]
    result = compute_pairwise_overlaps(seeds)
    mean, std, mn, mx = result[(4, 2)]
    assert mean == pytest.approx(0.0)
    assert mx   == pytest.approx(0.0)


def test_compute_pairwise_overlaps_partial():
    # all 3 pairs share exactly 2 of 4 stocks -> overlap = 0.5
    seeds = [
        {(4, 4): frozenset([0, 1, 2, 3])},
        {(4, 4): frozenset([2, 3, 4, 5])},
        {(4, 4): frozenset([0, 2, 4, 6])},
    ]
    result = compute_pairwise_overlaps(seeds)
    mean, std, mn, mx = result[(4, 4)]
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.0)
    assert mn == pytest.approx(0.5)


def test_compute_stock_frequencies_shape():
    n_stocks = 5
    seeds = [
        {(3, 2): frozenset([0, 1])},
        {(3, 2): frozenset([0, 2])},
    ]
    freq = compute_stock_frequencies(seeds, n_stocks, n_arms=[3], thresholds=[2])
    assert freq[(3, 2)].shape == (n_stocks,)


def test_compute_stock_frequencies_values():
    n_stocks = 5
    seeds = [
        {(3, 2): frozenset([0, 1])},
        {(3, 2): frozenset([0, 1])},
        {(3, 2): frozenset([0, 2])},
    ]
    freq = compute_stock_frequencies(seeds, n_stocks, n_arms=[3], thresholds=[2])
    counts = freq[(3, 2)]
    assert counts[0] == 3
    assert counts[1] == 2
    assert counts[2] == 1
    assert counts[3] == 0
    assert counts[4] == 0


def test_compute_core_sets_values():
    # stock 0: count=3, stock 1: count=2, stock 2: count=1
    freq = {(3, 2): np.array([3, 2, 1, 0, 0])}
    result = compute_core_sets(freq, n_seeds=3)
    # 100%: ceil(1.0*3)=3 -> stock 0 only -> 1
    assert result[(3, 2, 1.0)] == 1
    # 80%: ceil(0.8*3)=ceil(2.4)=3 -> stock 0 only -> 1
    assert result[(3, 2, 0.8)] == 1
    # 60%: ceil(0.6*3)=ceil(1.8)=2 -> stocks 0 and 1 -> 2
    assert result[(3, 2, 0.6)] == 2


def _make_stub_results():
    n_stocks = 5
    n_arms   = [2, 4]
    thresholds = [2]
    n_seeds  = 3
    seed_sets = [
        {(2, 2): frozenset([0, 1]), (4, 2): frozenset([0, 1])},
        {(2, 2): frozenset([0, 2]), (4, 2): frozenset([0, 2])},
        {(2, 2): frozenset([1, 2]), (4, 2): frozenset([1, 2])},
    ]
    freq = compute_stock_frequencies(seed_sets, n_stocks, n_arms, thresholds)
    return {
        'pairwise':   compute_pairwise_overlaps(seed_sets),
        'core':       compute_core_sets(freq, n_seeds),
        'freq':       freq,
        'n_seeds':    n_seeds,
        'n_arms':     n_arms,
        'thresholds': thresholds,
        'n_stocks':   n_stocks,
    }


def test_write_outputs_creates_all_files():
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(_make_stub_results(), tmp)
        assert set(os.listdir(tmp)) == {
            'pairwise_overlap.csv', 'core_set_size.csv',
            'stock_frequency.csv', 'seed_stability_summary.txt',
        }


def test_write_outputs_pairwise_columns():
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(_make_stub_results(), tmp)
        df = pd.read_csv(os.path.join(tmp, 'pairwise_overlap.csv'))
        assert list(df.columns) == ['n', 'k', 'overlap_mean', 'overlap_std', 'overlap_min', 'overlap_max']


def test_write_outputs_core_columns():
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(_make_stub_results(), tmp)
        df = pd.read_csv(os.path.join(tmp, 'core_set_size.csv'))
        assert list(df.columns) == ['n', 'k', 'threshold_pct', 'core_size']


def test_run_one_seed_output_structure():
    from seed_stability import run_one_seed
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
    from config import load_config
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(0)
    n_stocks, n_periods = 5, 30
    rets_df = pd.DataFrame(
        rng.standard_normal((n_periods, n_stocks)) * 0.01,
        columns=[f"S{i}" for i in range(n_stocks)],
    )
    cfg = load_config()
    cfg['time_window']             = 8
    cfg['periods_to_forecast']     = 2
    cfg['transformer_epochs']      = 1
    cfg['transformer_warmup_epochs'] = 0

    sigma_lw = np.sqrt(np.diag(LedoitWolf().fit(rets_df.values).covariance_))
    n_arms   = [2, 4]
    thresholds = [3]

    result = run_one_seed(rets_df, sigma_lw, cfg, n_arms, thresholds, seed=0)

    assert set(result.keys()) == {(2, 3), (4, 3)}
    for key, s in result.items():
        assert isinstance(s, frozenset)
        assert len(s) == key[1]


def test_run_timing_calibration_returns_positive():
    from seed_stability import run_timing_calibration
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
    from config import load_config

    rng = np.random.default_rng(7)
    rets_df = pd.DataFrame(
        rng.standard_normal((30, 5)) * 0.01,
        columns=[f"S{i}" for i in range(5)],
    )
    cfg = load_config()
    cfg['time_window']             = 8
    cfg['periods_to_forecast']     = 2
    cfg['transformer_epochs']      = 1
    cfg['transformer_warmup_epochs'] = 0

    per_run, total = run_timing_calibration(rets_df, cfg, n_cal=2)
    assert per_run > 0
    assert total   > 0


def test_main_smoke(tmp_path, monkeypatch):
    """Run main() end-to-end with tiny overrides -- must write all 4 output files."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
    from config import load_config
    import seed_stability as mod

    rng = np.random.default_rng(99)
    n_stocks, n_periods = 5, 30
    rets_df = pd.DataFrame(
        rng.standard_normal((n_periods, n_stocks)) * 0.01,
        columns=[f"S{i}" for i in range(n_stocks)],
    )
    cfg = load_config()
    cfg['time_window']             = 8
    cfg['periods_to_forecast']     = 2
    cfg['transformer_epochs']      = 1
    cfg['transformer_warmup_epochs'] = 0

    monkeypatch.setattr(mod, "N_SEEDS",    3)
    monkeypatch.setattr(mod, "N_ARMS",     [2, 4])
    monkeypatch.setattr(mod, "THRESHOLDS", [3])
    monkeypatch.setattr(mod, "_OUT_DIR",   str(tmp_path))
    monkeypatch.setattr(mod, "_load_data", lambda: (rets_df, cfg))
    monkeypatch.setattr(mod, "run_timing_calibration", lambda *a, **kw: (1.0, 900.0))
    monkeypatch.setattr("builtins.input", lambda _="": "y")

    mod.main()

    assert set(os.listdir(str(tmp_path))) == {
        'pairwise_overlap.csv', 'core_set_size.csv',
        'stock_frequency.csv', 'seed_stability_summary.txt',
    }
