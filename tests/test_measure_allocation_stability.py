"""
Tests for experiments/measure_allocation_stability.py

Run with:  python -m pytest tests/test_measure_allocation_stability.py -v

Pure helpers are tested with synthetic data only — no files, network, or GPU.
The run_experiment loop is tested with injected stubs so torch is never imported.
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from risk_kit import msr_tuned

from experiments.measure_allocation_stability import allocate_msr


CFG = {"rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}


@pytest.fixture
def three_assets():
    """Annualised returns + a diagonal covariance for three assets."""
    returns = pd.Series({"A": 0.20, "B": 0.12, "C": 0.02})
    covmat = pd.DataFrame(
        np.diag([0.04, 0.05, 0.06]), index=returns.index, columns=returns.index
    )
    return returns, covmat


@pytest.fixture
def five_assets_one_dominated():
    """Four equivalent good assets + one dominated asset (E).

    Equal returns/variance among A-D give ~0.25 each; E's strongly negative
    return forces it to ~0, so the elimination loop drops it (cumulative weight
    < min_weight) and re-optimises the remaining four (len > 2) back to weights
    that still sum to 1. With only three assets, dropping one would trip the
    `len(optimal) <= 2` guard before re-optimising, so >=5 assets are needed to
    exercise the elimination path cleanly.
    """
    returns = pd.Series({"A": 0.15, "B": 0.15, "C": 0.15, "D": 0.15, "E": -0.50})
    covmat = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04, 0.04, 0.04]),
        index=returns.index, columns=returns.index,
    )
    return returns, covmat


class TestAllocateMsr:
    def test_weights_sum_to_one(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_index_preserved(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert list(w.index) == list(returns.index)

    def test_respects_max_weight(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert (w <= CFG["max_weight"] + 1e-6).all()

    def test_no_negative_weights(self, three_assets):
        returns, covmat = three_assets
        w = allocate_msr(returns, covmat, CFG)
        assert (w >= -1e-8).all()

    def test_dominated_asset_eliminated(self, five_assets_one_dominated):
        # E has a strongly negative return; the optimizer zeroes it, the
        # elimination loop drops it (weight exactly 0), and the remaining four
        # re-optimise to weights that still sum to 1.
        returns, covmat = five_assets_one_dominated
        w = allocate_msr(returns, covmat, CFG)
        assert w["E"] == 0.0
        assert abs(w.sum() - 1.0) < 1e-6

    def test_uses_rf_period_key(self, three_assets):
        # allocate_msr must read cfg['rf_period']; a CFG missing it should KeyError.
        returns, covmat = three_assets
        bad_cfg = {"rf_rate": 0.0, "max_weight": 0.6, "min_weight": 0.05, "periods_per_year": 12}
        import pytest
        with pytest.raises(KeyError):
            allocate_msr(returns, covmat, bad_cfg)


from experiments.measure_allocation_stability import portfolio_metrics


class TestPortfolioMetrics:
    def test_known_values(self):
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.20})
        covmat = pd.DataFrame(
            np.diag([0.04, 0.09]), index=["A", "B"], columns=["A", "B"]
        )
        m = portfolio_metrics(weights, returns, covmat, rf=0.02)
        assert abs(m["ret"] - 0.15) < 1e-10
        # vol = sqrt(0.25*0.04 + 0.25*0.09) = sqrt(0.0325)
        assert abs(m["vol"] - math.sqrt(0.0325)) < 1e-10
        assert abs(m["sharpe"] - (0.15 - 0.02) / math.sqrt(0.0325)) < 1e-10

    def test_uses_only_weight_index_names(self):
        # returns/covmat carry an extra name C that weights omit; it must be ignored.
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.Series({"A": 0.10, "B": 0.20, "C": 9.0})
        covmat = pd.DataFrame(
            np.diag([0.04, 0.09, 1.0]), index=["A", "B", "C"], columns=["A", "B", "C"]
        )
        m = portfolio_metrics(weights, returns, covmat, rf=0.0)
        assert abs(m["ret"] - 0.15) < 1e-10


from experiments.measure_allocation_stability import (
    selection_frequency,
    weight_dispersion,
)


@pytest.fixture
def weights_df():
    """3 iterations x 3 names. C is held in 1 of 3 runs."""
    return pd.DataFrame(
        [
            {"A": 0.6, "B": 0.4, "C": 0.0},
            {"A": 0.5, "B": 0.5, "C": 0.0},
            {"A": 0.4, "B": 0.3, "C": 0.3},
        ]
    )


class TestSelectionFrequency:
    def test_fraction_held(self, weights_df):
        freq = selection_frequency(weights_df)
        assert freq["A"] == 1.0
        assert freq["B"] == 1.0
        assert abs(freq["C"] - 1 / 3) < 1e-12

    def test_returns_series_over_all_names(self, weights_df):
        freq = selection_frequency(weights_df)
        assert set(freq.index) == {"A", "B", "C"}


class TestWeightDispersion:
    def test_zero_for_constant_column(self):
        df = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.1, 0.2, 0.3]})
        disp = weight_dispersion(df)
        assert disp["A"] == 0.0
        assert disp["B"] > 0.0


from experiments.measure_allocation_stability import mean_turnover, mean_jaccard


class TestMeanTurnover:
    def test_two_disjoint_rows(self):
        # rows [1,0] and [0,1]: turnover = 0.5*(|1|+|1|) = 1.0
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        assert abs(mean_turnover(df) - 1.0) < 1e-12

    def test_identical_rows_zero(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}])
        assert mean_turnover(df) == 0.0

    def test_single_row_returns_none(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}])
        assert mean_turnover(df) is None


class TestMeanJaccard:
    def test_identical_sets_is_one(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}, {"A": 0.4, "B": 0.6}])
        assert abs(mean_jaccard(df) - 1.0) < 1e-12

    def test_disjoint_sets_is_zero(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        assert mean_jaccard(df) == 0.0

    def test_single_row_returns_none(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}])
        assert mean_jaccard(df) is None


from experiments.measure_allocation_stability import (
    metric_dispersion,
    amplification_factor,
)


class TestMetricDispersion:
    def test_mean_std_cov(self):
        df = pd.DataFrame({"ret": [0.10, 0.20], "vol": [0.05, 0.05], "sharpe": [1.0, 3.0]})
        disp = metric_dispersion(df)
        assert abs(disp["ret"]["mean"] - 0.15) < 1e-12
        assert disp["vol"]["std"] == 0.0
        # CoV = std / |mean| ; sharpe mean=2, std(ddof=0)=1 -> 0.5
        assert abs(disp["sharpe"]["cov"] - 0.5) < 1e-12


class TestAmplificationFactor:
    def test_ratio_of_mean_stds(self):
        # mu std per name: A->0, B->0.5 ; mean = 0.25
        mu_df = pd.DataFrame({"A": [0.10, 0.10], "B": [0.10, 1.10]})
        # weight std per name: A->0.05, B->0.05 ; mean = 0.05
        w_df = pd.DataFrame({"A": [0.45, 0.55], "B": [0.55, 0.45]})
        amp = amplification_factor(mu_df, w_df)
        assert abs(amp - (0.05 / 0.25)) < 1e-9

    def test_zero_input_std_is_nan(self):
        mu_df = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, 0.2]})
        w_df = pd.DataFrame({"A": [0.4, 0.6], "B": [0.6, 0.4]})
        assert math.isnan(amplification_factor(mu_df, w_df))


from experiments.measure_allocation_stability import select_stocks


@pytest.fixture
def synthetic_prices_returns():
    """60 weekly points for 4 tickers: 2 uptrending, 2 downtrending."""
    np.random.seed(7)
    n = 60
    idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
    up = lambda drift: 100 * np.cumprod(1 + np.random.normal(drift, 0.01, n))
    prices = pd.DataFrame(
        {
            "UP1": up(0.01), "UP2": up(0.008),
            "DN1": up(-0.01), "DN2": up(-0.008),
        },
        index=idx,
    )
    returns = prices.pct_change().dropna()
    return prices, returns


class TestSelectStocks:
    def test_subset_of_columns(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        selected = select_stocks(prices, returns, cfg)
        assert set(selected).issubset(set(prices.columns))

    def test_deterministic(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        assert select_stocks(prices, returns, cfg) == select_stocks(prices, returns, cfg)

    def test_returns_only_names_present_in_returns(self, synthetic_prices_returns):
        prices, returns = synthetic_prices_returns
        cfg = {"ma_terms": 10, "signal_min_count": 3}
        selected = select_stocks(prices, returns, cfg)
        assert all(name in returns.columns for name in selected)


from experiments.measure_allocation_stability import run_experiment


class TestRunExperiment:
    def _inputs(self):
        np.random.seed(1)
        n = 40
        idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
        cols = ["A", "B", "C", "D"]
        rets = pd.DataFrame(np.random.normal(0.002, 0.02, (n, 4)), index=idx, columns=cols)
        prices = (1 + rets).cumprod() * 100
        cfg = {
            "rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05,
            "periods_per_year": 12,
        }
        return prices, rets, cfg

    def _stubs(self):
        # predict_fn: returns 2 forecast periods that drift each call so mu varies.
        state = {"k": 0}

        def predict_fn(rets, cfg, n_runs=None, verbose=True):
            state["k"] += 1
            base = np.array([0.01, 0.008, 0.006, 0.002])
            shift = 0.001 * state["k"]
            vals = np.vstack([base + shift, base + shift])
            return pd.DataFrame(vals, columns=rets.columns)

        def period_mu_fn(preds):
            return preds.mean(axis=0)

        def select_fn(prices, rets, cfg):
            return ["A", "B", "C", "D"]

        def seed_fn(seed):
            pass

        return predict_fn, period_mu_fn, select_fn, seed_fn

    def test_output_shapes(self):
        prices, rets, cfg = self._inputs()
        predict_fn, period_mu_fn, select_fn, seed_fn = self._stubs()
        result = run_experiment(
            prices, rets, cfg, iterations=5, transformer_runs=2, seed=0,
            predict_fn=predict_fn, period_mu_fn=period_mu_fn,
            select_fn=select_fn, seed_fn=seed_fn,
        )
        assert result["weights"].shape == (5, 4)
        assert result["mu"].shape == (5, 4)
        assert set(result["metrics"].columns) == {"ret", "vol", "sharpe"}
        assert len(result["metrics"]) == 5
        assert result["selected"] == ["A", "B", "C", "D"]

    def test_weights_rows_sum_to_one(self):
        prices, rets, cfg = self._inputs()
        predict_fn, period_mu_fn, select_fn, seed_fn = self._stubs()
        result = run_experiment(
            prices, rets, cfg, iterations=3, transformer_runs=2, seed=0,
            predict_fn=predict_fn, period_mu_fn=period_mu_fn,
            select_fn=select_fn, seed_fn=seed_fn,
        )
        sums = result["weights"].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_weights_invariant_to_periods_per_year(self):
        # With rf_period held at 0, the per-period optimisation must give identical
        # mu and weights regardless of periods_per_year (mu carries no ppy term).
        import numpy as np
        prices, rets, cfg = self._inputs()
        cfg_weekly = dict(cfg, periods_per_year=52)
        cfg_monthly = dict(cfg, periods_per_year=12)
        # fresh stubs per call so the predict_fn drift counter starts equal
        p1, m1, s1, sd1 = self._stubs()
        r_weekly = run_experiment(prices, rets, cfg_weekly,
                                  iterations=3, transformer_runs=2, seed=0,
                                  predict_fn=p1, period_mu_fn=m1, select_fn=s1, seed_fn=sd1)
        p2, m2, s2, sd2 = self._stubs()
        r_monthly = run_experiment(prices, rets, cfg_monthly,
                                   iterations=3, transformer_runs=2, seed=0,
                                   predict_fn=p2, period_mu_fn=m2, select_fn=s2, seed_fn=sd2)
        assert np.allclose(r_weekly["weights"].values, r_monthly["weights"].values)
        assert np.allclose(r_weekly["mu"].values, r_monthly["mu"].values)


from experiments.measure_allocation_stability import format_summary, write_outputs


def _sample_result(iterations=3):
    names = ["A", "B", "C"]
    weights = pd.DataFrame(
        [
            {"A": 0.6, "B": 0.4, "C": 0.0},
            {"A": 0.5, "B": 0.5, "C": 0.0},
            {"A": 0.4, "B": 0.3, "C": 0.3},
        ][:iterations]
    )
    mu = pd.DataFrame(
        [
            {"A": 0.20, "B": 0.12, "C": 0.05},
            {"A": 0.21, "B": 0.11, "C": 0.06},
            {"A": 0.19, "B": 0.13, "C": 0.04},
        ][:iterations]
    )
    metrics = pd.DataFrame(
        [
            {"ret": 0.16, "vol": 0.10, "sharpe": 1.6},
            {"ret": 0.17, "vol": 0.10, "sharpe": 1.7},
            {"ret": 0.15, "vol": 0.11, "sharpe": 1.4},
        ][:iterations]
    )
    return {"mu": mu, "weights": weights, "metrics": metrics, "selected": names}


class TestFormatSummary:
    def test_contains_key_sections(self):
        text = format_summary(_sample_result())
        assert "Composition instability" in text
        assert "Value stability" in text
        assert "Selection frequency" in text

    def test_single_iteration_shows_na(self):
        text = format_summary(_sample_result(iterations=1))
        assert "N/A" in text  # turnover / jaccard undefined for 1 iteration


class TestWriteOutputs:
    def test_writes_three_files(self, tmp_path):
        paths = write_outputs(_sample_result(), str(tmp_path))
        for key in ("weights", "metrics", "summary"):
            assert os.path.exists(paths[key])

    def test_weights_csv_roundtrips(self, tmp_path):
        result = _sample_result()
        paths = write_outputs(result, str(tmp_path))
        reloaded = pd.read_csv(paths["weights"], index_col=0)
        assert reloaded.shape == result["weights"].shape


class TestOptimizerScaleInvariance:
    def test_weights_unchanged_under_consistent_scaling(self):
        # max-Sharpe is invariant to (mu->c*mu, Sigma->c^2*Sigma, rf->c*rf):
        # justifies optimising in per-period units instead of annualised ones.
        import numpy as np
        returns = pd.Series({"A": 0.02, "B": 0.012, "C": 0.015})
        cov = pd.DataFrame(np.diag([0.04, 0.05, 0.06]), index=returns.index, columns=returns.index)
        w1 = msr_tuned(0.001, returns=returns, covmat=cov, max_weight=0.6, periods_per_year=12)
        c = 52
        w2 = msr_tuned(0.001 * c, returns=returns * c, covmat=cov * (c ** 2),
                       max_weight=0.6, periods_per_year=12)
        assert np.allclose(w1, w2, atol=1e-4)


from experiments.measure_allocation_stability import apply_consensus_floor


class TestApplyConsensusFloor:
    def test_sums_to_one_and_preserves_index(self):
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert list(out.index) == ["A", "B", "C"]
        assert abs(out.sum() - 1.0) < 1e-9

    def test_drops_small_tail_and_renormalises(self):
        # C (0.02) sits below the 0.05 floor on cumulative weight; it is dropped
        # and the survivors renormalise to sum 1.
        w = pd.Series({"A": 0.60, "B": 0.38, "C": 0.02})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert out["C"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-9
        assert abs(out["A"] - 0.60 / 0.98) < 1e-9

    def test_no_drop_when_all_above_floor(self):
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert (out == pd.Series({"A": 0.5, "B": 0.5})).all()

    def test_zero_weight_names_stay_zero(self):
        w = pd.Series({"A": 0.7, "B": 0.3, "C": 0.0})
        out = apply_consensus_floor(w, min_weight=0.05)
        assert out["C"] == 0.0
        assert abs(out.sum() - 1.0) < 1e-9

    def test_stops_at_two_survivors(self):
        # Even if both remaining names are below the floor, the len<=2 guard stops
        # the loop so we never empty the book.
        w = pd.Series({"A": 0.5, "B": 0.5})
        out = apply_consensus_floor(w, min_weight=0.9)
        assert (out.abs() > 0).sum() == 2
        assert abs(out.sum() - 1.0) < 1e-9


from experiments.measure_allocation_stability import resampled_allocate


@pytest.fixture
def five_asset_cov():
    names = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(np.diag([0.04] * 5), index=names, columns=names)


def _mu(values):
    return pd.Series(dict(zip(["A", "B", "C", "D", "E"], values)))


class TestResampledAllocate:
    def test_consensus_sums_to_one_and_covers_all_names(self, five_asset_cov):
        # Three draws, each favouring a different leader -> diffuse but valid consensus.
        mus = [
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.20, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.02, 0.20, 0.02, 0.02]),
        ]
        consensus, diag = resampled_allocate(mus, five_asset_cov, CFG)
        assert abs(consensus.sum() - 1.0) < 1e-6
        assert set(consensus.index) == {"A", "B", "C", "D", "E"}

    def test_consensus_respects_max_weight_implicitly(self, five_asset_cov):
        # Averaging corner solutions cannot exceed the per-draw max_weight.
        mus = [_mu([0.20, 0.02, 0.02, 0.02, 0.02])] * 3
        consensus, _ = resampled_allocate(mus, five_asset_cov, CFG)
        assert (consensus <= CFG["max_weight"] + 1e-6).all()

    def test_diagnostic_reports_freq_and_mean_weight(self, five_asset_cov):
        # A leads every draw -> freq 1.0 and the largest mean raw weight.
        mus = [
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
        ]
        _, diag = resampled_allocate(mus, five_asset_cov, CFG)
        assert set(diag.columns) == {"freq", "mean_raw_weight"}
        assert abs(diag.loc["A", "freq"] - 1.0) < 1e-9
        assert diag["mean_raw_weight"].idxmax() == "A"

    def test_consensus_smooths_rotating_leaders(self, five_asset_cov):
        # No single name dominates when the leader rotates: the top consensus weight
        # is well below what a single-draw corner solution would give that name.
        mus = [
            _mu([0.20, 0.02, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.20, 0.02, 0.02, 0.02]),
            _mu([0.02, 0.02, 0.20, 0.02, 0.02]),
        ]
        single, _ = resampled_allocate([mus[0]], five_asset_cov, CFG)
        consensus, _ = resampled_allocate(mus, five_asset_cov, CFG)
        assert consensus.max() < single.max()

    def test_eliminate_per_draw_toggle_runs(self, five_asset_cov):
        # The comparison arm uses the full allocate_msr loop per draw; still valid.
        mus = [_mu([0.20, 0.02, 0.02, 0.02, 0.02])] * 3
        consensus, _ = resampled_allocate(mus, five_asset_cov, CFG, eliminate_per_draw=True)
        assert abs(consensus.sum() - 1.0) < 1e-6


from experiments.measure_allocation_stability import overlap_stats


class TestOverlapStats:
    def test_identical_books(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5, "C": 0.0},
                           {"A": 0.4, "B": 0.6, "C": 0.0}])
        s = overlap_stats(df)
        assert s["shared"] == 2.0
        assert s["held"] == 2.0
        assert abs(s["fraction"] - 1.0) < 1e-12

    def test_partial_overlap(self):
        # Books {A,B} and {B,C}: shared = 1, held per row = 2.
        df = pd.DataFrame([{"A": 0.5, "B": 0.5, "C": 0.0},
                           {"A": 0.0, "B": 0.5, "C": 0.5}])
        s = overlap_stats(df)
        assert s["shared"] == 1.0
        assert s["held"] == 2.0
        assert abs(s["fraction"] - 0.5) < 1e-12

    def test_disjoint_books(self):
        df = pd.DataFrame([{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}])
        s = overlap_stats(df)
        assert s["shared"] == 0.0

    def test_single_row_shared_is_none(self):
        df = pd.DataFrame([{"A": 0.5, "B": 0.5}])
        s = overlap_stats(df)
        assert s["shared"] is None
        assert s["held"] == 2.0


from experiments.measure_allocation_stability import run_paired_experiment


class TestRunPairedExperiment:
    def _inputs(self):
        np.random.seed(2)
        n = 40
        idx = pd.date_range("2023-01-01", periods=n, freq="W-SUN")
        cols = ["A", "B", "C", "D", "E"]
        rets = pd.DataFrame(np.random.normal(0.002, 0.02, (n, 5)), index=idx, columns=cols)
        prices = (1 + rets).cumprod() * 100
        cfg = {
            "rf_rate": 0.0, "rf_period": 0.0, "max_weight": 0.6, "min_weight": 0.05,
            "periods_per_year": 12,
        }
        return prices, rets, cfg

    def _stubs(self):
        # runs_fn returns a LIST of N per-run forecast DataFrames (periods x stocks).
        # Each run favours a different leader and the leadership rotates per iteration,
        # so the current (averaged) and Michaud (consensus) arms genuinely differ.
        state = {"k": 0}
        cols = ["A", "B", "C", "D", "E"]

        def runs_fn(rets, cfg, n_runs=None, verbose=True):
            state["k"] += 1
            n_runs = n_runs or 3
            runs = []
            for r in range(n_runs):
                vals = np.full(5, 0.004)
                leader = (state["k"] + r) % 5
                vals[leader] = 0.02
                runs.append(pd.DataFrame(np.vstack([vals, vals]), columns=cols))
            return runs

        def period_mu_fn(preds):
            return preds.mean(axis=0)

        def select_fn(prices, rets, cfg):
            return ["A", "B", "C", "D", "E"]

        def seed_fn(seed):
            pass

        return runs_fn, period_mu_fn, select_fn, seed_fn

    def _run(self, iterations=4, transformer_runs=3):
        prices, rets, cfg = self._inputs()
        runs_fn, period_mu_fn, select_fn, seed_fn = self._stubs()
        return run_paired_experiment(
            prices, rets, cfg, iterations=iterations, transformer_runs=transformer_runs,
            seed=0, runs_fn=runs_fn, period_mu_fn=period_mu_fn,
            select_fn=select_fn, seed_fn=seed_fn,
        )

    def test_returns_both_arms_and_selected(self):
        result = self._run()
        assert set(result.keys()) == {"current", "michaud", "selected"}
        assert result["selected"] == ["A", "B", "C", "D", "E"]

    def test_current_arm_shapes(self):
        result = self._run(iterations=4)
        assert result["current"]["weights"].shape == (4, 5)
        assert len(result["current"]["metrics"]) == 4
        assert set(result["current"]["metrics"].columns) == {"ret", "vol", "sharpe"}

    def test_michaud_arm_shapes_and_diagnostic(self):
        result = self._run(iterations=4)
        assert result["michaud"]["weights"].shape == (4, 5)
        assert len(result["michaud"]["metrics"]) == 4
        assert set(result["michaud"]["diagnostic"].columns) == {"freq", "mean_raw_weight"}

    def test_both_arms_weights_sum_to_one(self):
        result = self._run()
        for arm in ("current", "michaud"):
            sums = result[arm]["weights"].sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_arms_differ_with_rotating_leaders(self):
        # The whole point: with rotating per-run leaders the consensus is not identical
        # to the averaged-mu allocation, so the two weight matrices must differ.
        result = self._run()
        assert not np.allclose(
            result["current"]["weights"].values, result["michaud"]["weights"].values
        )


from experiments.measure_allocation_stability import format_paired_summary


def _sample_paired_result():
    cur_w = pd.DataFrame([{"A": 0.6, "B": 0.4, "C": 0.0},
                          {"A": 0.5, "B": 0.5, "C": 0.0},
                          {"A": 0.4, "B": 0.3, "C": 0.3}])
    mic_w = pd.DataFrame([{"A": 0.5, "B": 0.3, "C": 0.2},
                          {"A": 0.45, "B": 0.35, "C": 0.2},
                          {"A": 0.4, "B": 0.35, "C": 0.25}])
    metrics = pd.DataFrame([{"ret": 0.16, "vol": 0.10, "sharpe": 1.6},
                            {"ret": 0.17, "vol": 0.10, "sharpe": 1.7},
                            {"ret": 0.15, "vol": 0.11, "sharpe": 1.4}])
    diag = pd.DataFrame({"freq": {"A": 1.0, "B": 0.9, "C": 0.5},
                         "mean_raw_weight": {"A": 0.45, "B": 0.33, "C": 0.22}})
    return {
        "current": {"weights": cur_w, "metrics": metrics},
        "michaud": {"weights": mic_w, "metrics": metrics, "diagnostic": diag},
        "selected": ["A", "B", "C"],
    }


class TestFormatPairedSummary:
    def test_contains_both_arms_and_overlap(self):
        text = format_paired_summary(_sample_paired_result())
        assert "CURRENT" in text
        assert "MICHAUD" in text
        assert "Overlap" in text

    def test_contains_conviction_gradient_table(self):
        text = format_paired_summary(_sample_paired_result())
        assert "Conviction" in text
        assert "freq" in text


from experiments.measure_allocation_stability import write_paired_outputs


class TestWritePairedOutputs:
    def test_writes_all_files(self, tmp_path):
        paths = write_paired_outputs(_sample_paired_result(), str(tmp_path))
        for key in ("current_weights", "current_metrics", "michaud_weights",
                    "michaud_metrics", "michaud_diagnostic", "summary"):
            assert os.path.exists(paths[key])

    def test_michaud_weights_roundtrip(self, tmp_path):
        result = _sample_paired_result()
        paths = write_paired_outputs(result, str(tmp_path))
        reloaded = pd.read_csv(paths["michaud_weights"], index_col=0)
        assert reloaded.shape == result["michaud"]["weights"].shape


from experiments.measure_allocation_stability import build_arg_parser


class TestCliWiring:
    def test_mode_defaults_to_measure(self):
        args = build_arg_parser().parse_args([])
        assert args.mode == "measure"

    def test_paired_mode_and_flag_parse(self):
        args = build_arg_parser().parse_args(
            ["--mode", "paired", "--iterations", "2", "--transformer-runs", "100",
             "--eliminate-per-draw"]
        )
        assert args.mode == "paired"
        assert args.iterations == 2
        assert args.transformer_runs == 100
        assert args.eliminate_per_draw is True


from experiments.measure_allocation_stability import sample_mu_draws


def _cov3():
    names = ["A", "B", "C"]
    return pd.DataFrame(
        [[0.04, 0.01, 0.00],
         [0.01, 0.04, 0.01],
         [0.00, 0.01, 0.04]],
        index=names, columns=names,
    )


def _mu_bar3():
    return pd.Series({"A": 0.01, "B": 0.02, "C": 0.015})


class TestSampleMuDraws:
    def test_returns_n_draws_series_over_index(self):
        rng = np.random.default_rng(0)
        draws = sample_mu_draws(_mu_bar3(), _cov3(), n_periods=10,
                                n_draws=5, spread=1.0, rng=rng)
        assert len(draws) == 5
        for d in draws:
            assert list(d.index) == ["A", "B", "C"]

    def test_spread_zero_returns_exact_copies(self):
        rng = np.random.default_rng(0)
        mu = _mu_bar3()
        draws = sample_mu_draws(mu, _cov3(), n_periods=10,
                                n_draws=4, spread=0.0, rng=rng)
        for d in draws:
            assert (d == mu).all()

    def test_large_k_mean_approx_mu_bar(self):
        rng = np.random.default_rng(1)
        mu = _mu_bar3()
        draws = sample_mu_draws(mu, _cov3(), n_periods=10,
                                n_draws=50000, spread=1.0, rng=rng)
        mat = pd.DataFrame(draws)
        assert np.allclose(mat.mean(axis=0).values, mu.values, atol=2e-3)

    def test_large_k_cov_approx_scaled_sigma(self):
        rng = np.random.default_rng(2)
        cov = _cov3()
        draws = sample_mu_draws(_mu_bar3(), cov, n_periods=10,
                                n_draws=50000, spread=1.0, rng=rng)
        mat = pd.DataFrame(draws).values
        emp = np.cov(mat, rowvar=False)
        target = cov.values * (1.0 ** 2 / 10)
        assert np.allclose(emp, target, atol=3e-4)

    def test_larger_spread_more_dispersion(self):
        mu, cov = _mu_bar3(), _cov3()
        d1 = pd.DataFrame(sample_mu_draws(mu, cov, 10, 20000, 1.0, np.random.default_rng(3)))
        d2 = pd.DataFrame(sample_mu_draws(mu, cov, 10, 20000, 2.0, np.random.default_rng(3)))
        assert (d2.std(axis=0).values > d1.std(axis=0).values).all()

    def test_seeded_rng_reproducible(self):
        a = sample_mu_draws(_mu_bar3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        b = sample_mu_draws(_mu_bar3(), _cov3(), 10, 100, 1.0, np.random.default_rng(7))
        assert np.allclose(pd.DataFrame(a).values, pd.DataFrame(b).values)
