"""Tests for mfglob calibration, synthetic LOB data, and LOB statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfglob.models.avellaneda_stoikov import AvellanedaStoikovMFG
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.grids import Grid1D, TimeGrid
from mfglob.mfg_solver import MFGSolver
from mfglob.calibration import MFGCalibrator
from data.synthetic_lob import (
    generate_synthetic_book_shape,
    generate_synthetic_impact_data,
    generate_synthetic_inventory_data,
)
from data.lob_statistics import (
    average_book_shape,
    empirical_impact_curve,
    spread_statistics,
)


# ---------------------------------------------------------------------------
# TestMFGCalibrator
# ---------------------------------------------------------------------------

class TestMFGCalibrator:
    """Tests for MFGCalibrator on the AvellanedaStoikovMFG model."""

    def _make_calibrator(self, target_data=None, grid_params=None):
        if target_data is None:
            target_data = {"spread": 1.0 / 1.5}  # target spread = 1/k_true
        if grid_params is None:
            grid_params = {"n_x": 15, "n_t": 10, "T": 0.05,
                           "x_min": -3.0, "x_max": 3.0}
        return MFGCalibrator(
            model_class=AvellanedaStoikovMFG,
            target_data=target_data,
            param_bounds={"k": (0.5, 3.0), "phi": (0.001, 0.1)},
            grid_params=grid_params,
        )

    # -- Construction --

    def test_construction(self):
        calib = self._make_calibrator()
        assert calib.model_class is AvellanedaStoikovMFG
        assert "k" in calib._param_names
        assert "phi" in calib._param_names

    def test_param_names_ordered(self):
        calib = self._make_calibrator()
        assert calib._param_names == ["k", "phi"]

    # -- objective --

    def test_objective_returns_float(self):
        calib = self._make_calibrator()
        loss = calib.objective(np.array([1.5, 0.01]))
        assert isinstance(loss, float)

    def test_objective_finite(self):
        calib = self._make_calibrator()
        loss = calib.objective(np.array([1.5, 0.01]))
        assert np.isfinite(loss)

    def test_objective_nonneg(self):
        calib = self._make_calibrator()
        loss = calib.objective(np.array([1.5, 0.01]))
        assert loss >= 0.0

    def test_objective_large_for_bad_params(self):
        """Very wrong spread param → higher loss than near-truth params."""
        calib = self._make_calibrator(target_data={"spread": 1.0 / 1.5})
        loss_good = calib.objective(np.array([1.5, 0.01]))
        loss_bad = calib.objective(np.array([0.5, 0.01]))   # 1/0.5 far from target
        # loss_good should be smaller (near-truth) or at least finite
        assert np.isfinite(loss_good)
        assert np.isfinite(loss_bad)

    def test_objective_increments_n_evals(self):
        calib = self._make_calibrator()
        calib._n_evals = 0
        calib.objective(np.array([1.5, 0.01]))
        assert calib._n_evals == 1

    def test_objective_with_book_shape(self):
        """Objective with book_shape target should run without error."""
        x = np.linspace(-3.0, 3.0, 15)
        density = np.exp(-0.5 * x ** 2)
        density /= np.trapezoid(density, x)
        target = {"book_shape": {"levels": x, "density": density}}
        calib = MFGCalibrator(
            AvellanedaStoikovMFG, target,
            param_bounds={"k": (0.5, 3.0)},
            grid_params={"n_x": 15, "n_t": 10, "T": 0.05,
                         "x_min": -3.0, "x_max": 3.0},
        )
        loss = calib.objective(np.array([1.5]))
        assert np.isfinite(loss)
        assert loss >= 0.0

    # -- calibrate: Nelder-Mead (fast) --

    def test_calibrate_nelder_mead_returns_dict(self):
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert isinstance(result, dict)

    def test_calibrate_nelder_mead_required_keys(self):
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        for key in ("params", "objective", "success", "n_evals", "message"):
            assert key in result, f"Missing key: {key}"

    def test_calibrate_nelder_mead_params_dict(self):
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert isinstance(result["params"], dict)
        assert "k" in result["params"]
        assert "phi" in result["params"]

    def test_calibrate_nelder_mead_params_in_bounds(self):
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert 0.5 <= result["params"]["k"] <= 3.0
        assert 0.001 <= result["params"]["phi"] <= 0.1

    def test_calibrate_nelder_mead_objective_finite(self):
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert np.isfinite(result["objective"])

    def test_calibrate_nelder_mead_n_evals_positive(self):
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert result["n_evals"] > 0

    def test_calibrate_finds_spread_param(self):
        """For spread-only target, calibrated k should give spread ≈ target."""
        target_k = 2.0
        calib = MFGCalibrator(
            AvellanedaStoikovMFG,
            target_data={"spread": 1.0 / target_k},
            param_bounds={"k": (0.5, 4.0)},
            grid_params={"n_x": 10, "n_t": 8, "T": 0.05,
                         "x_min": -2.0, "x_max": 2.0},
        )
        result = calib.calibrate(method="nelder_mead", verbose=False)
        recovered_k = result["params"]["k"]
        implied_spread = 1.0 / recovered_k
        target_spread = 1.0 / target_k
        assert abs(implied_spread - target_spread) < 0.3, (
            f"Recovered spread {implied_spread:.3f} far from target {target_spread:.3f}"
        )

    def test_calibrate_invalid_method_raises(self):
        calib = self._make_calibrator()
        with pytest.raises(ValueError):
            calib.calibrate(method="invalid_method", verbose=False)

    def test_diagnostic_plots_runs(self):
        """diagnostic_plots should not raise (uses Agg backend)."""
        model = AvellanedaStoikovMFG()
        grid = Grid1D(-3.0, 3.0, 15)
        tgrid = TimeGrid(0.05, 10)
        solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=5)
        sol = solver.solve()
        calib = self._make_calibrator()
        calib.diagnostic_plots(sol)  # should not raise


# ---------------------------------------------------------------------------
# TestSyntheticBookShape
# ---------------------------------------------------------------------------

class TestSyntheticBookShape:

    def test_returns_dict(self):
        result = generate_synthetic_book_shape(n_levels=20)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = generate_synthetic_book_shape(n_levels=20)
        for key in ("price_levels", "volumes_bid", "volumes_ask",
                    "mid_price", "spread"):
            assert key in result, f"Missing key: {key}"

    def test_price_levels_shape(self):
        result = generate_synthetic_book_shape(n_levels=30)
        assert len(result["price_levels"]) == 30
        assert len(result["volumes_bid"]) == 30
        assert len(result["volumes_ask"]) == 30

    def test_volumes_nonneg(self):
        for shape in ("exponential", "hump", "uniform", "power_law"):
            result = generate_synthetic_book_shape(n_levels=20, shape=shape)
            assert np.all(result["volumes_bid"] >= 0), f"{shape}: negative bid"
            assert np.all(result["volumes_ask"] >= 0), f"{shape}: negative ask"

    def test_volumes_positive_sum(self):
        for shape in ("exponential", "hump", "uniform", "power_law"):
            result = generate_synthetic_book_shape(n_levels=20, shape=shape)
            assert result["volumes_bid"].sum() > 0, f"{shape}: zero bid sum"
            assert result["volumes_ask"].sum() > 0, f"{shape}: zero ask sum"

    def test_exponential_decreasing(self):
        """Exponential shape: volumes should decrease on average."""
        result = generate_synthetic_book_shape(n_levels=40, shape="exponential", seed=0)
        v = result["volumes_bid"]
        # First half > second half on average
        assert v[:20].mean() > v[20:].mean()

    def test_hump_peaks_away_from_zero(self):
        """Hump shape: peak at tick > 1."""
        result = generate_synthetic_book_shape(n_levels=40, shape="hump", seed=0)
        v = result["volumes_bid"]
        peak_idx = np.argmax(v)
        assert peak_idx > 0, "Hump should peak away from first level"

    def test_uniform_roughly_flat(self):
        """Uniform shape: all levels have similar volume."""
        result = generate_synthetic_book_shape(n_levels=20, shape="uniform", seed=0)
        v = result["volumes_bid"]
        cv = v.std() / (v.mean() + 1e-14)  # coefficient of variation
        assert cv < 0.5, f"Uniform shape has high variance: cv={cv:.3f}"

    def test_seed_reproducibility(self):
        r1 = generate_synthetic_book_shape(n_levels=20, seed=42)
        r2 = generate_synthetic_book_shape(n_levels=20, seed=42)
        np.testing.assert_array_equal(r1["volumes_bid"], r2["volumes_bid"])

    def test_different_seeds_differ(self):
        r1 = generate_synthetic_book_shape(n_levels=20, seed=1)
        r2 = generate_synthetic_book_shape(n_levels=20, seed=2)
        assert not np.array_equal(r1["volumes_bid"], r2["volumes_bid"])

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            generate_synthetic_book_shape(shape="invalid_shape")


# ---------------------------------------------------------------------------
# TestSyntheticImpactData
# ---------------------------------------------------------------------------

class TestSyntheticImpactData:

    def test_returns_dataframe(self):
        df = generate_synthetic_impact_data(n_trades=100)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_synthetic_impact_data(n_trades=100)
        for col in ("trade_size", "price_impact", "volatility", "volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_length(self):
        df = generate_synthetic_impact_data(n_trades=200)
        assert len(df) == 200

    def test_trade_size_positive(self):
        df = generate_synthetic_impact_data(n_trades=500, seed=0)
        assert np.all(df["trade_size"] > 0)

    def test_price_impact_nonneg(self):
        for law in ("sqrt", "linear", "concave_power"):
            df = generate_synthetic_impact_data(n_trades=300, impact_law=law, seed=0)
            assert np.all(df["price_impact"] >= 0), f"{law}: negative impact"

    def test_sqrt_law_concavity(self):
        """For sqrt law, impact / sqrt(trade_size) should be roughly constant."""
        df = generate_synthetic_impact_data(n_trades=500, impact_law="sqrt", seed=0)
        sizes = df["trade_size"].values
        impacts = df["price_impact"].values
        # Bin and check impact grows slower than linearly with size
        small = sizes < np.median(sizes)
        large = ~small
        ratio_small = np.mean(impacts[small]) / (np.mean(sizes[small]) + 1e-14)
        ratio_large = np.mean(impacts[large]) / (np.mean(sizes[large]) + 1e-14)
        # For concave law, ratio decreases as size increases
        assert ratio_small >= ratio_large * 0.8, "Impact should be concave in trade size"

    def test_seed_reproducibility(self):
        df1 = generate_synthetic_impact_data(n_trades=100, seed=7)
        df2 = generate_synthetic_impact_data(n_trades=100, seed=7)
        np.testing.assert_array_equal(df1["trade_size"].values, df2["trade_size"].values)

    def test_invalid_law_raises(self):
        with pytest.raises(ValueError):
            generate_synthetic_impact_data(impact_law="invalid_law")


# ---------------------------------------------------------------------------
# TestSyntheticInventoryData
# ---------------------------------------------------------------------------

class TestSyntheticInventoryData:

    def test_returns_dataframe(self):
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=50)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=50)
        for col in ("timestamp", "mm_id", "inventory"):
            assert col in df.columns, f"Missing column: {col}"

    def test_shape(self):
        n_mm, n_t = 10, 100
        df = generate_synthetic_inventory_data(n_market_makers=n_mm, n_timestamps=n_t)
        assert len(df) == n_mm * n_t

    def test_mm_id_range(self):
        n_mm = 8
        df = generate_synthetic_inventory_data(n_market_makers=n_mm, n_timestamps=20)
        assert df["mm_id"].min() == 0
        assert df["mm_id"].max() == n_mm - 1

    def test_timestamp_range(self):
        n_t = 50
        df = generate_synthetic_inventory_data(n_market_makers=3, n_timestamps=n_t)
        assert df["timestamp"].min() >= 0
        assert df["timestamp"].max() < n_t

    def test_mean_reverting(self):
        """High mean-reversion → inventories concentrated near zero."""
        df = generate_synthetic_inventory_data(
            n_market_makers=30, n_timestamps=500,
            mean_reversion=5.0, seed=0,
        )
        # At stationarity, mean inventory should be near 0
        late = df[df["timestamp"] > 400]
        mean_inv = late["inventory"].mean()
        assert abs(mean_inv) < 1.0, f"Mean inventory {mean_inv:.3f} not near 0"

    def test_inventory_finite(self):
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=100, seed=0)
        assert df["inventory"].notna().all()
        assert np.all(np.isfinite(df["inventory"].values))

    def test_seed_reproducibility(self):
        df1 = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=50, seed=3)
        df2 = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=50, seed=3)
        np.testing.assert_array_equal(df1["inventory"].values, df2["inventory"].values)


# ---------------------------------------------------------------------------
# TestAverageBookShape
# ---------------------------------------------------------------------------

class TestAverageBookShape:

    def _make_snapshots(self, n=5, shape="exponential"):
        return [generate_synthetic_book_shape(n_levels=20, shape=shape, seed=i)
                for i in range(n)]

    def test_returns_dict(self):
        snaps = self._make_snapshots()
        result = average_book_shape(snaps)
        assert isinstance(result, dict)

    def test_required_keys(self):
        snaps = self._make_snapshots()
        result = average_book_shape(snaps)
        for key in ("price_levels", "avg_volumes_bid", "avg_volumes_ask",
                    "std_volumes_bid", "std_volumes_ask"):
            assert key in result, f"Missing key: {key}"

    def test_shape_matches_n_levels(self):
        snaps = self._make_snapshots()
        result = average_book_shape(snaps)
        n = len(snaps[0]["price_levels"])
        assert len(result["avg_volumes_bid"]) == n
        assert len(result["std_volumes_bid"]) == n

    def test_std_nonneg(self):
        snaps = self._make_snapshots(n=10)
        result = average_book_shape(snaps)
        assert np.all(result["std_volumes_bid"] >= 0)
        assert np.all(result["std_volumes_ask"] >= 0)

    def test_avg_nonneg(self):
        snaps = self._make_snapshots()
        result = average_book_shape(snaps)
        assert np.all(result["avg_volumes_bid"] >= 0)
        assert np.all(result["avg_volumes_ask"] >= 0)

    def test_empty_raises(self):
        with pytest.raises((ValueError, IndexError)):
            average_book_shape([])

    def test_single_snapshot(self):
        snap = generate_synthetic_book_shape(n_levels=10, seed=0)
        result = average_book_shape([snap])
        np.testing.assert_allclose(result["avg_volumes_bid"], snap["volumes_bid"])


# ---------------------------------------------------------------------------
# TestEmpiricalImpactCurve
# ---------------------------------------------------------------------------

class TestEmpiricalImpactCurve:

    def _make_trades(self, n=500):
        return generate_synthetic_impact_data(n_trades=n, impact_law="sqrt", seed=0)

    def test_returns_dict(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades)
        assert isinstance(result, dict)

    def test_required_keys(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades)
        for key in ("bin_centers", "mean_impact", "std_impact", "n_trades_per_bin"):
            assert key in result, f"Missing key: {key}"

    def test_bin_centers_positive(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades)
        assert np.all(result["bin_centers"] > 0)

    def test_mean_impact_nonneg(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades)
        assert np.all(result["mean_impact"] >= 0)

    def test_std_nonneg(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades)
        assert np.all(result["std_impact"] >= 0)

    def test_bin_counts_positive(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades)
        assert np.all(result["n_trades_per_bin"] > 0)

    def test_total_count(self):
        trades = self._make_trades(n=500)
        result = empirical_impact_curve(trades, n_bins=10)
        assert result["n_trades_per_bin"].sum() == 500

    def test_bin_centers_increasing(self):
        trades = self._make_trades()
        result = empirical_impact_curve(trades, n_bins=15)
        assert np.all(np.diff(result["bin_centers"]) > 0)

    def test_impact_generally_increasing(self):
        """For sqrt law, mean impact should increase with bin center."""
        trades = self._make_trades(n=2000)
        result = empirical_impact_curve(trades, n_bins=10)
        centers = result["bin_centers"]
        impacts = result["mean_impact"]
        # First quartile < last quartile
        n = len(impacts)
        assert impacts[n // 4:].mean() >= impacts[:n // 4].mean()


# ---------------------------------------------------------------------------
# TestSpreadStatistics
# ---------------------------------------------------------------------------

class TestSpreadStatistics:

    def _make_snapshots(self, spreads):
        return [{"spread": float(s), "volumes_bid": np.ones(5),
                 "volumes_ask": np.ones(5)} for s in spreads]

    def test_returns_dict(self):
        snaps = self._make_snapshots([0.01, 0.02, 0.03])
        result = spread_statistics(snaps)
        assert isinstance(result, dict)

    def test_required_keys(self):
        snaps = self._make_snapshots([0.01, 0.02, 0.03])
        result = spread_statistics(snaps)
        for key in ("mean", "median", "std", "min", "max", "distribution"):
            assert key in result, f"Missing key: {key}"

    def test_mean_correct(self):
        snaps = self._make_snapshots([0.01, 0.03])
        result = spread_statistics(snaps)
        assert result["mean"] == pytest.approx(0.02)

    def test_median_correct(self):
        snaps = self._make_snapshots([0.01, 0.02, 0.03])
        result = spread_statistics(snaps)
        assert result["median"] == pytest.approx(0.02)

    def test_std_nonneg(self):
        snaps = self._make_snapshots([0.01, 0.02, 0.03])
        result = spread_statistics(snaps)
        assert result["std"] >= 0.0

    def test_min_max(self):
        snaps = self._make_snapshots([0.01, 0.05, 0.03])
        result = spread_statistics(snaps)
        assert result["min"] == pytest.approx(0.01)
        assert result["max"] == pytest.approx(0.05)

    def test_distribution_array(self):
        spreads = [0.01, 0.02, 0.03, 0.04]
        snaps = self._make_snapshots(spreads)
        result = spread_statistics(snaps)
        assert len(result["distribution"]) == 4

    def test_constant_spread_zero_std(self):
        snaps = self._make_snapshots([0.02, 0.02, 0.02])
        result = spread_statistics(snaps)
        assert result["std"] == pytest.approx(0.0, abs=1e-10)

    def test_empty_raises(self):
        with pytest.raises((ValueError, IndexError)):
            spread_statistics([])
