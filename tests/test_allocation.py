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


class TestConsensusFloorRespectsMaxWeight:
    """msr_tuned bounds every MC draw to max_weight, but the floor then drops
    names and renormalises the survivors to sum to 1, which can push a weight
    back over the cap with nothing re-checking it.

    Measured on the calibration sweep: 72 of 208 allocations exceeded a 0.15
    cap, median 21% over, worst 0.285 (+90%).
    """

    def test_renormalisation_after_drop_breaches_the_cap(self):
        # The regression case: a heavy name plus a tail that the floor removes.
        # Renormalising 0.14 over a 0.72 surviving base gives 0.194 > 0.15.
        w = pd.Series({"A": 0.14, "B": 0.13, "C": 0.13, "D": 0.13, "E": 0.13,
                       "F": 0.12, "G": 0.11, "H": 0.06, "I": 0.03, "J": 0.02})
        out = apply_consensus_floor(w, min_weight=0.05, max_weight=0.15)
        assert out.max() <= 0.15 + 1e-9
        assert abs(out.sum() - 1.0) < 1e-9

    def test_capping_cascades_to_second_name(self):
        # Capping A pushes its excess onto B, which must then also be capped.
        w = pd.Series({"A": 0.50, "B": 0.28, "C": 0.12, "D": 0.10})
        out = apply_consensus_floor(w, min_weight=0.0, max_weight=0.30)
        assert out.max() <= 0.30 + 1e-9
        assert abs(out.sum() - 1.0) < 1e-9
        assert (out > 0).sum() == 4

    def test_floor_leaves_enough_names_for_the_cap_to_be_satisfiable(self):
        # A 0.15 cap needs ceil(1/0.15) = 7 names; 6 x 0.15 = 0.90 < 1. The old
        # guard stopped at 2 survivors, which could strip the book past
        # feasibility. Observed once in the sweep: a 6-name book at 0.183.
        w = pd.Series({c: v for c, v in zip("ABCDEFGHIJ",
                                            [0.40, 0.30, 0.20, 0.04, 0.02, 0.01,
                                             0.01, 0.01, 0.005, 0.005])})
        out = apply_consensus_floor(w, min_weight=0.30, max_weight=0.15)
        held = (out > 0).sum()
        assert held >= 7, f"only {held} names survive; a 0.15 cap needs 7"
        assert out.max() <= 0.15 + 1e-9

    def test_raises_when_cap_is_unsatisfiable_for_the_universe(self):
        # 5 names cannot sum to 1 under a 0.15 cap. Silently shipping a book
        # that breaks a stated limit is the failure this fix exists to prevent.
        w = pd.Series({"A": 0.3, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.1})
        with pytest.raises(ValueError, match="max_weight"):
            apply_consensus_floor(w, min_weight=0.0, max_weight=0.15)

    def test_default_max_weight_preserves_old_behaviour(self):
        # Existing callers pass no cap and must be unaffected.
        w = pd.Series({"A": 0.60, "B": 0.38, "C": 0.02})
        legacy = apply_consensus_floor(w, min_weight=0.05)
        assert abs(legacy["A"] - 0.60 / 0.98) < 1e-9

    def test_capping_does_not_reintroduce_a_sub_floor_tail(self):
        # Water-filling only raises the uncapped names, so the tail it feeds can
        # never fall back below the floor -- there is no floor/cap cycle.
        w = pd.Series({"A": 0.40, "B": 0.25, "C": 0.20, "D": 0.09, "E": 0.06})
        out = apply_consensus_floor(w, min_weight=0.05, max_weight=0.25)
        held = out[out > 0].sort_values()
        assert held.cumsum().iloc[0] >= 0.05 - 1e-9
        assert out.max() <= 0.25 + 1e-9


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


from allocation import resampled_michaud, allocate, equal_weight_topn_alloc


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

    def test_routes_to_equal_weight_topn(self, cov5):
        cfg = dict(CFG)
        cfg.update(allocation_method="equal_weight_topn", equal_weight_n=3,
                   min_weight=0.05, max_weight=0.6)
        mu = _mu5([0.20, 0.10, 0.05, 0.05, 0.05])
        got = allocate(mu, cov5, cfg, n_periods=100)
        want = equal_weight_topn_alloc(mu, cov5, cfg)
        assert np.allclose(got.values, want.values)


class TestEqualWeightTopN:
    """Equal-weighting the model's picks matched the Michaud optimiser in the
    backtest (net Sharpe 1.132 vs 1.131, head-to-head p=0.949), so it is worth
    having as a switchable allocator rather than only as a benchmark.
    """

    def test_holds_exactly_n_names_at_equal_weight(self, cov5):
        cfg = dict(CFG)
        cfg.update(equal_weight_n=3, min_weight=0.05, max_weight=0.6)
        w = equal_weight_topn_alloc(_mu5([0.20, 0.10, 0.05, 0.02, 0.01]), cov5, cfg)
        held = w[w > 0]
        assert len(held) == 3
        assert np.allclose(held.values, 1 / 3)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_selects_by_sharpe_when_ranking_is_sharpe(self, cov5):
        # cov5 has unequal variances, so mu-ranking and sharpe-ranking differ.
        cfg = dict(CFG)
        cfg.update(equal_weight_n=2, allocation_ranking="sharpe",
                   min_weight=0.05, max_weight=0.6)
        mu = _mu5([0.20, 0.10, 0.05, 0.05, 0.05])
        vol = np.sqrt(np.diag(cov5.values))
        expected = set(pd.Series(mu.values / vol, index=mu.index).nlargest(2).index)
        held = equal_weight_topn_alloc(mu, cov5, cfg)
        assert set(held[held > 0].index) == expected

    def test_selects_by_return_when_ranking_is_return(self, cov5):
        cfg = dict(CFG)
        cfg.update(equal_weight_n=2, allocation_ranking="return",
                   min_weight=0.05, max_weight=0.6)
        mu = _mu5([0.20, 0.10, 0.05, 0.05, 0.05])
        held = equal_weight_topn_alloc(mu, cov5, cfg)
        assert set(held[held > 0].index) == set(mu.nlargest(2).index)

    def test_rejects_n_that_would_breach_max_weight(self, cov5):
        # 1/2 = 0.50 per name against a 0.15 cap is infeasible.
        cfg = dict(CFG)
        cfg.update(equal_weight_n=2, min_weight=0.05, max_weight=0.15)
        with pytest.raises(ValueError, match="max_weight"):
            equal_weight_topn_alloc(_mu5([0.2] * 5), cov5, cfg)

    def test_rejects_n_that_would_breach_min_weight(self, cov5):
        # 5 names at 1/5 = 0.20 each is below a 0.30 floor. Checked against the
        # EFFECTIVE n, after clamping to the universe -- asking for more names
        # than exist is not itself an error (see the clamping test below).
        cfg = dict(CFG)
        cfg.update(equal_weight_n=5, min_weight=0.30, max_weight=0.6)
        with pytest.raises(ValueError, match="min_weight"):
            equal_weight_topn_alloc(_mu5([0.2] * 5), cov5, cfg)

    def test_default_n_satisfies_both_weight_bounds(self, cov5):
        cfg = dict(CFG)
        cfg.update(min_weight=0.05, max_weight=0.6)
        cfg.pop("equal_weight_n", None)
        w = equal_weight_topn_alloc(_mu5([0.5, 0.4, 0.3, 0.2, 0.1]), cov5, cfg)
        held = w[w > 0]
        assert held.iloc[0] >= 0.05 - 1e-9
        assert held.iloc[0] <= 0.6 + 1e-9

    def test_caps_n_at_the_available_universe(self, cov5):
        cfg = dict(CFG)
        cfg.update(equal_weight_n=50, min_weight=0.0, max_weight=1.0)
        w = equal_weight_topn_alloc(_mu5([0.2] * 5), cov5, cfg)
        assert (w > 0).sum() == 5


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


# ---------------------------------------------------------------------------
# Model-free methods
#
# These were backtest benchmarks first. Momentum led on net Sharpe in all four
# walk-forward runs and equal-weight-top-N beat the model in one, so they are
# selectable in production -- but the production form is NOT the backtested form:
# every method here ends with the min_weight floor, which caps any book at
# 1/min_weight names. A floored gmv book is "the 20 largest gmv weights", not gmv.
# ---------------------------------------------------------------------------

from allocation import (equal_weight_all_alloc, gmv_alloc, inverse_vol_alloc,
                        momentum_alloc, random_alloc, MODEL_FREE_METHODS)


@pytest.fixture
def cov20():
    names = [f"S{i:02d}" for i in range(20)]
    var = [0.01 + 0.002 * i for i in range(20)]      # S00 least volatile
    return pd.DataFrame(np.diag(var), index=names, columns=names)


def _wide_cfg(**over):
    cfg = dict(CFG)
    cfg.update(min_weight=0.05, max_weight=0.15)
    cfg.update(over)
    return cfg


class TestEqualWeightAll:
    def test_sums_to_one_and_keeps_full_index(self, cov20):
        mu = pd.Series(0.1, index=cov20.index)
        w = equal_weight_all_alloc(mu, cov20, _wide_cfg())
        assert abs(w.sum() - 1.0) < 1e-9
        assert list(w.index) == list(mu.index)

    def test_universe_too_small_for_the_cap_raises(self, cov20):
        # 5 names cannot sum to 1 under a 0.15 cap. Fail loudly rather than ship a
        # book that breaks a stated limit.
        mu = pd.Series(0.1, index=cov20.index[:5])
        with pytest.raises(ValueError, match="max_weight"):
            equal_weight_all_alloc(mu, cov20.iloc[:5, :5], _wide_cfg())


class TestGmvAlloc:
    def test_keeps_the_low_vol_name_and_respects_bounds(self, cov20):
        w = gmv_alloc(pd.Series(0.1, index=cov20.index), cov20, _wide_cfg())
        assert "S00" in w[w > 0].index          # least volatile of the 20
        assert abs(w.sum() - 1.0) < 1e-9
        assert w.max() <= 0.15 + 1e-9

    def test_ignores_expected_returns(self, cov20):
        cfg = _wide_cfg()
        a = gmv_alloc(pd.Series(0.1, index=cov20.index), cov20, cfg)
        b = gmv_alloc(pd.Series(range(20), index=cov20.index, dtype=float), cov20, cfg)
        assert np.allclose(a.values, b.values)      # mu must not enter

    def test_floor_is_enforced(self, cov20):
        # The documented divergence from the backtest: unfloored gmv spreads across
        # every name, floored it cannot hold a position below min_weight.
        w = gmv_alloc(pd.Series(0.1, index=cov20.index), cov20, _wide_cfg())
        assert (w[w > 0] >= 0.05 - 1e-9).all()


class TestInverseVolAlloc:
    def test_sums_to_one_and_respects_bounds(self, cov20):
        w = inverse_vol_alloc(pd.Series(0.1, index=cov20.index), cov20, _wide_cfg())
        assert abs(w.sum() - 1.0) < 1e-9
        assert w.max() <= 0.15 + 1e-9
        assert (w[w > 0] >= 0.05 - 1e-9).all()

    def test_ignores_expected_returns(self, cov20):
        cfg = _wide_cfg()
        a = inverse_vol_alloc(pd.Series(0.1, index=cov20.index), cov20, cfg)
        b = inverse_vol_alloc(pd.Series(range(20), index=cov20.index, dtype=float),
                              cov20, cfg)
        assert np.allclose(a.values, b.values)


def _hist(names, winners):
    """Returns panel where `winners` compound up and everything else is flat."""
    idx = [f"p{i}" for i in range(40)]
    return pd.DataFrame({n: [0.05 if n in winners else 0.0] * 40 for n in names},
                        index=idx)


class TestMomentumAlloc:
    def test_picks_the_trailing_winners(self, cov20):
        names = list(cov20.index)
        winners = names[:8]
        cfg = _wide_cfg(equal_weight_n=8, momentum_lookback=24)
        w = momentum_alloc(pd.Series(0.1, index=names), cov20, cfg,
                           hist_rets=_hist(names, winners))
        assert set(w[w > 0].index) == set(winners)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_ignores_expected_returns(self, cov20):
        names = list(cov20.index)
        cfg = _wide_cfg(equal_weight_n=8, momentum_lookback=24)
        hist = _hist(names, names[:8])
        a = momentum_alloc(pd.Series(0.1, index=names), cov20, cfg, hist_rets=hist)
        # mu ordered opposite to the momentum winners: must not change the book.
        b = momentum_alloc(pd.Series(range(20), index=names, dtype=float), cov20, cfg,
                           hist_rets=hist)
        assert np.allclose(a.values, b.values)

    def test_uses_only_the_lookback_window(self, cov20):
        # Older history must not leak in: S01 wins over 40 periods but LOSES over the
        # last 24. S02 is the modest in-window gainer, so the two picks are decided by
        # momentum rather than by an arbitrary tie-break among flat names.
        names = list(cov20.index)
        idx = [f"p{i}" for i in range(40)]
        data = {n: [0.0] * 40 for n in names}
        data["S00"] = [0.0] * 16 + [0.05] * 24        # strong only inside the window
        data["S01"] = [0.9] * 16 + [-0.02] * 24       # strong only before it, then falls
        data["S02"] = [0.0] * 16 + [0.01] * 24        # mild in-window gain
        cfg = _wide_cfg(equal_weight_n=2, momentum_lookback=24, max_weight=0.6)
        w = momentum_alloc(pd.Series(0.1, index=names), cov20, cfg,
                           hist_rets=pd.DataFrame(data, index=idx))
        assert set(w[w > 0].index) == {"S00", "S02"}
        assert w["S01"] == 0

    def test_raises_a_named_error_without_history(self, cov20):
        # Momentum is the one method needing the raw panel; the error has to name
        # the missing input rather than fail on a None dereference deep inside.
        cfg = _wide_cfg(equal_weight_n=8)
        with pytest.raises(ValueError, match="01_returns"):
            momentum_alloc(pd.Series(0.1, index=cov20.index), cov20, cfg,
                           hist_rets=None)

    def test_scores_only_the_allocation_universe(self, cov20):
        # The panel carries the whole downloaded catalogue; a name outside the
        # step-2 universe must not be allocatable just because it rose.
        names = list(cov20.index)
        hist = _hist(names + ["OUTSIDER"], ["OUTSIDER"] + names[:7])
        cfg = _wide_cfg(equal_weight_n=8, momentum_lookback=24)
        w = momentum_alloc(pd.Series(0.1, index=names), cov20, cfg, hist_rets=hist)
        assert "OUTSIDER" not in w.index
        assert abs(w.sum() - 1.0) < 1e-9


class TestRandomAlloc:
    def test_is_reproducible_under_a_seed(self, cov20):
        cfg = _wide_cfg(equal_weight_n=8, michaud_seed=7)
        mu = pd.Series(0.1, index=cov20.index)
        a = random_alloc(mu, cov20, cfg)
        b = random_alloc(mu, cov20, cfg)
        assert np.allclose(a.values, b.values)
        assert abs(a.sum() - 1.0) < 1e-9

    def test_different_seeds_differ(self, cov20):
        mu = pd.Series(0.1, index=cov20.index)
        a = random_alloc(mu, cov20, _wide_cfg(equal_weight_n=8, michaud_seed=1))
        b = random_alloc(mu, cov20, _wide_cfg(equal_weight_n=8, michaud_seed=2))
        assert not np.allclose(a.values, b.values)


class TestModelFreeMethodsSkipTheModelPrefilter:
    """allocation_top_n ranks by the model's mu. Handing that shortlist to a
    model-free method would make it quietly model-dependent and stop it matching
    the strategy the backtest scored, so allocate() applies the pre-filter only to
    the methods that actually consume the forecast."""

    def test_registry_lists_the_model_free_methods(self):
        assert MODEL_FREE_METHODS == {"equal_weight_all", "gmv", "inverse_vol",
                                      "momentum", "random"}

    def test_gmv_result_does_not_depend_on_allocation_top_n(self, cov20):
        mu = pd.Series(range(20), index=cov20.index, dtype=float)
        wide = allocate(mu, cov20, _wide_cfg(allocation_method="gmv",
                                             allocation_top_n=None), n_periods=100)
        narrow = allocate(mu, cov20, _wide_cfg(allocation_method="gmv",
                                               allocation_top_n=5), n_periods=100)
        assert np.allclose(wide.values, narrow.values)

    def test_michaud_still_honours_allocation_top_n(self, cov20):
        mu = pd.Series(range(20), index=cov20.index, dtype=float) / 100
        cfg = _wide_cfg(allocation_method="parametric_michaud", allocation_top_n=4,
                        michaud_mc_draws=20, max_weight=0.6)
        w = allocate(mu, cov20, cfg, n_periods=100)
        assert set(w[w > 0].index) <= set(mu.nlargest(4).index)

    def test_dispatcher_routes_each_new_method(self, cov20):
        mu = pd.Series(0.1, index=cov20.index)
        hist = _hist(list(cov20.index), list(cov20.index)[:8])
        pairs = [("equal_weight_all", equal_weight_all_alloc),
                 ("gmv", gmv_alloc),
                 ("inverse_vol", inverse_vol_alloc)]
        for name, fn in pairs:
            cfg = _wide_cfg(allocation_method=name)
            got = allocate(mu, cov20, cfg, n_periods=100)
            assert np.allclose(got.values, fn(mu, cov20, cfg).values), name
        cfg = _wide_cfg(allocation_method="momentum", equal_weight_n=8,
                        momentum_lookback=24)
        got = allocate(mu, cov20, cfg, n_periods=100, hist_rets=hist)
        want = momentum_alloc(mu, cov20, cfg, hist_rets=hist)
        assert np.allclose(got.values, want.values)
