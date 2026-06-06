import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from allocation import msr_eliminate

CFG = {
    "rf_period": 0.0, "rf_rate": 0.0,
    "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12,
    "allocation_method": "parametric_michaud",
    "michaud_spread": 1.0, "michaud_mc_draws": 200, "michaud_seed": 0,
}


def _cov(names, var=0.04):
    return pd.DataFrame(np.diag([var] * len(names)), index=names, columns=names)


class TestMsrEliminate:
    def test_sums_to_one(self):
        names = ["A", "B", "C"]
        mu = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_returns_series_over_full_index(self):
        names = ["A", "B", "C"]
        mu = pd.Series({"A": 0.20, "B": 0.10, "C": 0.05})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert list(w.index) == names

    def test_respects_max_weight(self):
        names = ["A", "B", "C", "D", "E"]
        mu = pd.Series(dict(zip(names, [0.30, 0.05, 0.05, 0.05, 0.05])))
        w = msr_eliminate(mu, _cov(names), CFG)
        assert (w <= CFG["max_weight"] + 1e-6).all()

    def test_two_assets_both_held(self):
        names = ["A", "B"]
        mu = pd.Series({"A": 0.20, "B": 0.10})
        w = msr_eliminate(mu, _cov(names), CFG)
        assert (w.abs() > 0).sum() == 2
        assert abs(w.sum() - 1.0) < 1e-6


from allocation import apply_consensus_floor, sample_mu_draws


class TestApplyConsensusFloor:
    def test_sums_to_one_and_preserves_index(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert list(out.index) == ["A", "B", "C"]
        assert abs(out.sum() - 1.0) < 1e-9

    def test_drops_small_tail_and_renormalises(self):
        w = pd.Series({"A": 0.60, "B": 0.38, "C": 0.02})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert out["C"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-9
        assert abs(out["A"] - 0.60 / 0.98) < 1e-9

    def test_no_drop_when_all_above_floor(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert (out == pd.Series({"A": 0.5, "B": 0.5})).all()

    def test_stops_at_two_survivors(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.9)
        assert (out.abs() > 0).sum() == 2
        assert abs(out.sum() - 1.0) < 1e-9


def _cov3():
    names = ["A", "B", "C"]
    return pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.04, 0.01], [0.00, 0.01, 0.04]],
        index=names, columns=names,
    )


def _mu3():
    return pd.Series({"A": 0.01, "B": 0.02, "C": 0.015})


class TestSampleMuDraws:
    def test_returns_n_draws_over_index(self):
        draws = sample_mu_draws(_mu3(), _cov3(), 10, 5, 1.0, np.random.default_rng(0))
        assert len(draws) == 5
        for d in draws:
            assert list(d.index) == ["A", "B", "C"]

    def test_spread_zero_returns_copies(self):
        mu = _mu3()
        draws = sample_mu_draws(mu, _cov3(), 10, 4, 0.0, np.random.default_rng(0))
        for d in draws:
            assert (d == mu).all()

    def test_large_k_mean_approx_mu(self):
        mu = _mu3()
        draws = sample_mu_draws(mu, _cov3(), 10, 50000, 1.0, np.random.default_rng(1))
        assert np.allclose(pd.DataFrame(draws).mean(axis=0).values, mu.values, atol=2e-3)

    def test_seeded_reproducible(self):
        a = sample_mu_draws(_mu3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        b = sample_mu_draws(_mu3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        assert np.allclose(pd.DataFrame(a).values, pd.DataFrame(b).values)


from allocation import resampled_michaud, allocate


@pytest.fixture
def cov5():
    names = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.diag([0.04] * 5), index=names, columns=names)


def _mu5(vals):
    return pd.Series(dict(zip(["A", "B", "C", "D", "E"], vals)))


class TestResampledMichaud:
    def test_consensus_sums_to_one(self, cov5):
        w = resampled_michaud(_mu5([0.20, 0.02, 0.02, 0.02, 0.02]), cov5, CFG, n_periods=100)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_deterministic_with_seed(self, cov5):
        mu = _mu5([0.20, 0.05, 0.02, 0.02, 0.02])
        w1 = resampled_michaud(mu, cov5, CFG, n_periods=100)
        w2 = resampled_michaud(mu, cov5, CFG, n_periods=100)
        assert np.allclose(w1.values, w2.values)

    def test_null_seed_still_valid(self, cov5):
        cfg = dict(CFG)
        cfg["michaud_seed"] = None
        w = resampled_michaud(_mu5([0.20, 0.05, 0.02, 0.02, 0.02]), cov5, cfg, n_periods=100)
        assert abs(w.sum() - 1.0) < 1e-6


class TestAllocateDispatcher:
    def test_routes_to_msr(self, cov5):
        cfg = dict(CFG); cfg["allocation_method"] = "msr"
        mu = _mu5([0.20, 0.10, 0.05, 0.05, 0.05])
        got = allocate(mu, cov5, cfg, n_periods=100)
        want = msr_eliminate(mu, cov5, cfg)
        assert np.allclose(got.values, want.values)

    def test_routes_to_parametric(self, cov5):
        cfg = dict(CFG); cfg["allocation_method"] = "parametric_michaud"
        mu = _mu5([0.20, 0.05, 0.02, 0.02, 0.02])
        got = allocate(mu, cov5, cfg, n_periods=100)
        want = resampled_michaud(mu, cov5, cfg, n_periods=100)
        assert np.allclose(got.values, want.values)

    def test_unknown_method_raises(self, cov5):
        cfg = dict(CFG); cfg["allocation_method"] = "bogus"
        with pytest.raises(ValueError):
            allocate(_mu5([0.1] * 5), cov5, cfg, n_periods=100)


from allocation import select_top_n


def _universe5():
    """5-stock universe with known Sharpe and return rankings.

    mu:    A=0.10  B=0.05  C=0.20  D=0.08  E=0.15
    vol:   A=0.30  B=0.10  C=0.50  D=0.10  E=0.20
    Sharpe:  0.333   0.500   0.400   0.800   0.750
    Sharpe rank: D > E > B > C > A  -> top-3: {D, E, B}
    Return rank: C > E > A > D > B  -> top-3: {C, E, A}
    """
    tickers = ["A", "B", "C", "D", "E"]
    mu = pd.Series({"A": 0.10, "B": 0.05, "C": 0.20, "D": 0.08, "E": 0.15})
    vols = {"A": 0.30, "B": 0.10, "C": 0.50, "D": 0.10, "E": 0.20}
    cov_arr = np.diag([vols[t] ** 2 for t in tickers])
    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
    return mu, cov


class TestSelectTopN:
    def test_sharpe_ranking_selects_correct_names(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=3, metric="sharpe")
        assert set(mu_out.index) == {"D", "E", "B"}

    def test_return_ranking_selects_correct_names(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=3, metric="return")
        assert set(mu_out.index) == {"C", "E", "A"}

    def test_null_n_returns_full_universe(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=None, metric="sharpe")
        assert list(mu_out.index) == list(mu.index)
        assert cov_out.shape == cov.shape

    def test_n_exceeds_universe_returns_full(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=1000, metric="sharpe")
        assert list(mu_out.index) == list(mu.index)
        assert cov_out.shape == cov.shape

    def test_covmat_index_matches_mu_index(self):
        mu, cov = _universe5()
        mu_out, cov_out = select_top_n(mu, cov, n=3, metric="sharpe")
        assert list(cov_out.index) == list(mu_out.index)
        assert list(cov_out.columns) == list(mu_out.index)

    def test_unknown_metric_raises(self):
        mu, cov = _universe5()
        with pytest.raises(ValueError):
            select_top_n(mu, cov, n=3, metric="bogus")
