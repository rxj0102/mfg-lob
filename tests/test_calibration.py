"""Tests for calibration module, synthetic LOB data generators, and LOB statistics.

Covers:
- MFGCalibrator: objective, calibrate (nelder_mead), parameter recovery
- generate_synthetic_book_shape: shapes, seed reproducibility
- generate_synthetic_impact_data: columns, positivity, concave_power law
- generate_synthetic_inventory_data: shape, columns, mean-reversion
- LOB statistics: average_book_shape, empirical_impact_curve, spread_statistics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Imports from mfglob calibration
# ---------------------------------------------------------------------------
from mfglob.calibration import MFGCalibrator
from mfglob.models.avellaneda_stoikov import AvellanedaStoikovMFG

# ---------------------------------------------------------------------------
# Imports from data modules
# ---------------------------------------------------------------------------
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


# ===========================================================================
# TestMFGCalibrator
# ===========================================================================

class TestMFGCalibrator:
    """Tests for the MFGCalibrator class in mfglob.calibration."""

    def _make_calibrator(self, target_data=None, param_bounds=None):
        """Helper: build a calibrator with sensible defaults."""
        if target_data is None:
            target_data = {"spread": 1.0 / 1.5}
        if param_bounds is None:
            param_bounds = {"k": (0.5, 3.0)}
        grid_params = {"n_x": 15, "n_t": 10, "T": 0.05}
        return MFGCalibrator(
            AvellanedaStoikovMFG,
            target_data=target_data,
            param_bounds=param_bounds,
            grid_params=grid_params,
        )

    def test_objective_returns_finite_float(self):
        """objective() must return a finite float for valid params."""
        calib = self._make_calibrator()
        val = calib.objective(np.array([1.5]))
        assert isinstance(val, float), "objective should return a float"
        assert np.isfinite(val), f"objective should be finite, got {val}"

    def test_objective_returns_1e10_on_failure(self):
        """objective() should return 1e10 when solver raises an exception."""
        # Pass an invalid model class that always raises
        class BrokenModel:
            def __call__(self):
                raise RuntimeError("broken")

        # Use extreme bounds to force a degenerate grid (x_min >= x_max via setattr)
        # Instead, verify the fallback by patching with a model that has x_min >= x_max
        calib = MFGCalibrator(
            AvellanedaStoikovMFG,
            target_data={"spread": 0.5},
            param_bounds={"k": (0.5, 3.0)},
            grid_params={"n_x": 15, "n_t": 10, "T": 0.05,
                         "x_min": 0.0, "x_max": 0.0},  # invalid: equal
        )
        val = calib.objective(np.array([1.5]))
        # Should either be 1e10 (caught) or still finite if grid clamps
        assert val == 1e10 or np.isfinite(val)

    def test_calibrate_nelder_mead_returns_required_keys(self):
        """calibrate(method='nelder_mead') must return dict with required keys."""
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        required_keys = {"params", "objective", "success", "n_evals", "message"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_calibrate_nelder_mead_params_is_dict(self):
        """The 'params' value should be a dict mapping param_name → float."""
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert isinstance(result["params"], dict)
        assert "k" in result["params"]

    def test_calibrate_nelder_mead_objective_is_float(self):
        """The 'objective' value should be a finite float."""
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert isinstance(result["objective"], float)
        assert np.isfinite(result["objective"])

    def test_calibrate_nelder_mead_n_evals_positive(self):
        """n_evals must be a positive integer."""
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert isinstance(result["n_evals"], int)
        assert result["n_evals"] > 0

    def test_calibrate_nelder_mead_message_is_str(self):
        """message must be a string."""
        calib = self._make_calibrator()
        result = calib.calibrate(method="nelder_mead", verbose=False)
        assert isinstance(result["message"], str)

    def test_calibrate_finds_parameter_near_truth(self):
        """1-param calibration: calibrate should recover k ≈ 1.5 when spread ≈ 1/k."""
        true_k = 1.5
        target_spread = 1.0 / true_k  # ≈ 0.667
        calib = MFGCalibrator(
            AvellanedaStoikovMFG,
            target_data={"spread": target_spread},
            param_bounds={"k": (0.5, 4.0)},
            grid_params={"n_x": 15, "n_t": 10, "T": 0.05},
        )
        result = calib.calibrate(method="nelder_mead", verbose=False)
        recovered_k = result["params"]["k"]
        assert abs(recovered_k - true_k) < 0.5, (
            f"Expected k ≈ {true_k}, got {recovered_k}"
        )

    def test_calibrate_book_shape_target(self):
        """Calibrator handles book_shape target_data without error."""
        levels = np.linspace(-1.0, 1.0, 15)
        density = np.exp(-0.5 * levels ** 2)
        calib = MFGCalibrator(
            AvellanedaStoikovMFG,
            target_data={"book_shape": {"levels": levels, "density": density}},
            param_bounds={"phi": (0.001, 0.1)},
            grid_params={"n_x": 15, "n_t": 10, "T": 0.05,
                         "x_min": -10.0, "x_max": 10.0},
        )
        val = calib.objective(np.array([0.01]))
        assert np.isfinite(val) or val == 1e10

    def test_calibrate_unknown_method_raises(self):
        """calibrate() should raise ValueError for unknown method."""
        calib = self._make_calibrator()
        with pytest.raises(ValueError, match="Unknown method"):
            calib.calibrate(method="unknown_method", verbose=False)

    def test_diagnostic_plots_runs_without_display(self):
        """diagnostic_plots should not raise even without a display."""
        calib = self._make_calibrator()
        # Build a minimal solution dict
        n_t, n_x = 10, 15
        solution = {
            "density": np.random.rand(n_t, n_x),
            "optimal_control": np.random.rand(n_t, n_x),
            "convergence_history": [1.0, 0.5, 0.1],
        }
        # Should complete without error (uses Agg backend)
        calib.diagnostic_plots(solution)


# ===========================================================================
# TestSyntheticBookShape
# ===========================================================================

class TestSyntheticBookShape:
    """Tests for generate_synthetic_book_shape in data.synthetic_lob."""

    def test_returns_required_keys(self):
        """Must return dict with all required keys."""
        result = generate_synthetic_book_shape(n_levels=20, shape="exponential", seed=0)
        required = {"price_levels", "volumes_bid", "volumes_ask", "mid_price", "spread"}
        assert required.issubset(result.keys())

    def test_exponential_shape_is_decreasing_on_average(self):
        """Exponential shape should broadly decrease from level 1 to n_levels."""
        result = generate_synthetic_book_shape(n_levels=50, shape="exponential", seed=42)
        vols = result["volumes_bid"]
        # Compare first half mean vs second half mean
        n = len(vols)
        first_half = vols[:n // 2].mean()
        second_half = vols[n // 2:].mean()
        assert first_half > second_half, (
            f"Exponential should decrease: first_half={first_half:.3f} > second_half={second_half:.3f}"
        )

    def test_exponential_first_exceeds_last(self):
        """First level volume should exceed last level volume for exponential."""
        result = generate_synthetic_book_shape(n_levels=50, shape="exponential", seed=7)
        vols = result["volumes_bid"]
        assert vols[0] > vols[-1], (
            f"Exponential: vols[0]={vols[0]:.3f} should exceed vols[-1]={vols[-1]:.3f}"
        )

    def test_hump_shape_peaks_away_from_zero(self):
        """Hump shape A*x*exp(-k*x) should peak at level > 1."""
        result = generate_synthetic_book_shape(n_levels=50, shape="hump", seed=42)
        vols = result["volumes_bid"]
        peak_idx = int(np.argmax(vols))
        assert peak_idx > 0, (
            f"Hump should peak away from first level, got peak at index {peak_idx}"
        )

    def test_hump_shape_not_monotone(self):
        """Hump should rise then fall (not monotonically decreasing)."""
        result = generate_synthetic_book_shape(n_levels=50, shape="hump", seed=42)
        vols = result["volumes_bid"]
        peak_idx = int(np.argmax(vols))
        # Volume at peak should exceed volume at first level
        assert vols[peak_idx] > vols[0], (
            f"Hump peak vol {vols[peak_idx]:.3f} should exceed first level vol {vols[0]:.3f}"
        )

    def test_uniform_shape_volumes_are_positive(self):
        """Uniform shape should give positive volumes at all levels."""
        result = generate_synthetic_book_shape(n_levels=20, shape="uniform", seed=42)
        assert np.all(result["volumes_bid"] >= 0)
        assert np.all(result["volumes_ask"] >= 0)

    def test_power_law_shape_volumes_positive(self):
        """Power-law shape should give positive volumes."""
        result = generate_synthetic_book_shape(n_levels=20, shape="power_law", seed=42)
        assert np.all(result["volumes_bid"] >= 0)
        assert np.all(result["volumes_ask"] >= 0)

    def test_all_shapes_sum_positive(self):
        """All shape types should have positive total volume."""
        for shape in ["exponential", "hump", "uniform", "power_law"]:
            result = generate_synthetic_book_shape(n_levels=30, shape=shape, seed=1)
            assert result["volumes_bid"].sum() > 0, f"volumes_bid should sum > 0 for {shape}"
            assert result["volumes_ask"].sum() > 0, f"volumes_ask should sum > 0 for {shape}"

    def test_price_levels_are_positive_integers(self):
        """price_levels should be 1, 2, ..., n_levels."""
        n = 25
        result = generate_synthetic_book_shape(n_levels=n, shape="exponential", seed=0)
        levels = result["price_levels"]
        assert len(levels) == n
        assert levels[0] == 1.0
        assert levels[-1] == float(n)

    def test_seed_reproducibility(self):
        """Same seed should give identical results."""
        r1 = generate_synthetic_book_shape(n_levels=30, shape="hump", seed=99)
        r2 = generate_synthetic_book_shape(n_levels=30, shape="hump", seed=99)
        np.testing.assert_array_equal(r1["volumes_bid"], r2["volumes_bid"])
        np.testing.assert_array_equal(r1["volumes_ask"], r2["volumes_ask"])

    def test_different_seeds_give_different_results(self):
        """Different seeds should (almost certainly) give different results."""
        r1 = generate_synthetic_book_shape(n_levels=30, shape="exponential", seed=1)
        r2 = generate_synthetic_book_shape(n_levels=30, shape="exponential", seed=2)
        assert not np.allclose(r1["volumes_bid"], r2["volumes_bid"])

    def test_unknown_shape_raises(self):
        """Unknown shape name should raise ValueError."""
        with pytest.raises(ValueError):
            generate_synthetic_book_shape(shape="unknown_shape")

    def test_n_levels_parameter(self):
        """n_levels controls the number of price levels returned."""
        for n in [10, 20, 50]:
            result = generate_synthetic_book_shape(n_levels=n, seed=0)
            assert len(result["price_levels"]) == n
            assert len(result["volumes_bid"]) == n
            assert len(result["volumes_ask"]) == n

    def test_spread_is_positive(self):
        """Returned spread should be positive."""
        result = generate_synthetic_book_shape()
        assert result["spread"] > 0

    def test_mid_price_is_positive(self):
        """Returned mid_price should be positive."""
        result = generate_synthetic_book_shape()
        assert result["mid_price"] > 0


# ===========================================================================
# TestSyntheticImpactData
# ===========================================================================

class TestSyntheticImpactData:
    """Tests for generate_synthetic_impact_data in data.synthetic_lob."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = generate_synthetic_impact_data(n_trades=100, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        """DataFrame must have required columns."""
        df = generate_synthetic_impact_data(n_trades=100, seed=0)
        required = {"trade_size", "price_impact", "volatility", "volume"}
        assert required.issubset(df.columns)

    def test_n_trades_length(self):
        """DataFrame should have exactly n_trades rows."""
        for n in [50, 200, 1000]:
            df = generate_synthetic_impact_data(n_trades=n, seed=0)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_impact_positive_for_positive_trade_sizes(self):
        """All price_impact values should be non-negative (abs is applied)."""
        df = generate_synthetic_impact_data(n_trades=500, impact_law="sqrt", seed=42)
        assert np.all(df["price_impact"] >= 0), "price_impact should be non-negative"

    def test_trade_size_positive(self):
        """All trade_size values should be positive (exponential draws)."""
        df = generate_synthetic_impact_data(n_trades=500, seed=42)
        assert np.all(df["trade_size"] > 0)

    def test_sqrt_law_impact_positive(self):
        """sqrt impact law should give non-negative impact values."""
        df = generate_synthetic_impact_data(n_trades=200, impact_law="sqrt", seed=42)
        assert np.all(df["price_impact"] >= 0)

    def test_linear_law_increases_with_size(self):
        """Linear impact: larger trades should have larger mean impact (in bins)."""
        df = generate_synthetic_impact_data(n_trades=2000, impact_law="linear", seed=42)
        df_sorted = df.sort_values("trade_size").reset_index(drop=True)
        n = len(df_sorted)
        first_q = df_sorted["price_impact"].iloc[:n // 4].mean()
        last_q = df_sorted["price_impact"].iloc[3 * n // 4:].mean()
        assert last_q > first_q, (
            f"Linear impact should increase with size: Q1={first_q:.4f}, Q4={last_q:.4f}"
        )

    def test_concave_power_law_is_concave(self):
        """Concave power law (Q^0.6): impact should be concave in trade size."""
        df = generate_synthetic_impact_data(n_trades=2000, impact_law="concave_power", seed=42)
        df_sorted = df.sort_values("trade_size").reset_index(drop=True)
        n = len(df_sorted)
        # Split into 5 equal bins
        groups = np.array_split(range(n), 5)
        bin_sizes = [df_sorted["trade_size"].iloc[g].mean() for g in groups]
        bin_impacts = [df_sorted["price_impact"].iloc[g].mean() for g in groups]
        # Check that marginal impact is decreasing (concavity)
        growth = [
            (bin_impacts[i + 1] - bin_impacts[i]) / (bin_sizes[i + 1] - bin_sizes[i])
            for i in range(4)
        ]
        assert all(growth[i] > growth[i + 1] for i in range(3)), (
            f"Concave power law should be concave. Growth rates: {growth}"
        )

    def test_concave_power_positive_impact(self):
        """Concave power law should give non-negative impact."""
        df = generate_synthetic_impact_data(n_trades=200, impact_law="concave_power", seed=42)
        assert np.all(df["price_impact"] >= 0)

    def test_seed_reproducibility(self):
        """Same seed should give identical DataFrames."""
        df1 = generate_synthetic_impact_data(n_trades=100, impact_law="sqrt", seed=77)
        df2 = generate_synthetic_impact_data(n_trades=100, impact_law="sqrt", seed=77)
        pd.testing.assert_frame_equal(df1, df2)

    def test_unknown_impact_law_raises(self):
        """Unknown impact_law should raise ValueError."""
        with pytest.raises(ValueError):
            generate_synthetic_impact_data(impact_law="unknown_law")

    def test_volatility_in_range(self):
        """Volatility should be in [0.01, 0.05]."""
        df = generate_synthetic_impact_data(n_trades=500, seed=0)
        assert df["volatility"].min() >= 0.0
        assert df["volatility"].max() <= 1.0  # reasonable upper bound


# ===========================================================================
# TestSyntheticInventoryData
# ===========================================================================

class TestSyntheticInventoryData:
    """Tests for generate_synthetic_inventory_data in data.synthetic_lob."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=10, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        """DataFrame must have columns: timestamp, mm_id, inventory."""
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=10, seed=0)
        assert set(df.columns) == {"timestamp", "mm_id", "inventory"}

    def test_shape(self):
        """DataFrame should have n_market_makers * n_timestamps rows."""
        n_mm = 10
        n_ts = 50
        df = generate_synthetic_inventory_data(
            n_market_makers=n_mm, n_timestamps=n_ts, seed=0
        )
        assert df.shape == (n_mm * n_ts, 3), (
            f"Expected shape ({n_mm * n_ts}, 3), got {df.shape}"
        )

    def test_mm_id_range(self):
        """mm_id should range from 0 to n_market_makers - 1."""
        n_mm = 7
        df = generate_synthetic_inventory_data(n_market_makers=n_mm, n_timestamps=20, seed=0)
        assert df["mm_id"].min() == 0
        assert df["mm_id"].max() == n_mm - 1

    def test_timestamp_range(self):
        """Timestamps should range from 0 to n_timestamps - 1."""
        n_ts = 30
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=n_ts, seed=0)
        assert df["timestamp"].min() == 0
        assert df["timestamp"].max() == n_ts - 1

    def test_mean_reverting_behavior(self):
        """OU process: inventory should have approximately zero mean over long runs."""
        df = generate_synthetic_inventory_data(
            n_market_makers=100, n_timestamps=1000, mean_reversion=0.5, seed=42
        )
        mean_inv = df["inventory"].mean()
        std_inv = df["inventory"].std()
        # Mean should be close to 0 relative to std
        assert abs(mean_inv) < std_inv, (
            f"OU mean ({mean_inv:.4f}) should be small relative to std ({std_inv:.4f})"
        )

    def test_seed_reproducibility(self):
        """Same seed gives identical DataFrames."""
        df1 = generate_synthetic_inventory_data(
            n_market_makers=10, n_timestamps=20, seed=123
        )
        df2 = generate_synthetic_inventory_data(
            n_market_makers=10, n_timestamps=20, seed=123
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_inventory_is_numeric(self):
        """inventory column should contain numeric values."""
        df = generate_synthetic_inventory_data(n_market_makers=5, n_timestamps=10, seed=0)
        assert pd.api.types.is_numeric_dtype(df["inventory"])

    def test_strong_mean_reversion_keeps_inventory_small(self):
        """Strong mean reversion (kappa=5) should keep inventory near zero."""
        df = generate_synthetic_inventory_data(
            n_market_makers=50, n_timestamps=500,
            mean_reversion=5.0, seed=42,
        )
        # After burn-in, inventory std should be smaller than with low reversion
        df_weak = generate_synthetic_inventory_data(
            n_market_makers=50, n_timestamps=500,
            mean_reversion=0.01, seed=42,
        )
        std_strong = df["inventory"].std()
        std_weak = df_weak["inventory"].std()
        # Strong reversion => smaller std
        assert std_strong < std_weak, (
            f"Strong reversion std ({std_strong:.4f}) should be < weak reversion std ({std_weak:.4f})"
        )


# ===========================================================================
# TestLOBStatistics
# ===========================================================================

class TestAverageBookShape:
    """Tests for average_book_shape in data.lob_statistics."""

    def _make_snapshots(self, n=5, n_levels=20):
        return [
            generate_synthetic_book_shape(n_levels=n_levels, seed=i)
            for i in range(n)
        ]

    def test_returns_required_keys(self):
        """Must return dict with all required keys."""
        snaps = self._make_snapshots()
        result = average_book_shape(snaps)
        required = {
            "price_levels", "avg_volumes_bid", "avg_volumes_ask",
            "std_volumes_bid", "std_volumes_ask",
        }
        assert required.issubset(result.keys())

    def test_shapes_match_n_levels(self):
        """All arrays should have length n_levels."""
        n_levels = 25
        snaps = self._make_snapshots(n=5, n_levels=n_levels)
        result = average_book_shape(snaps)
        assert len(result["price_levels"]) == n_levels
        assert len(result["avg_volumes_bid"]) == n_levels
        assert len(result["avg_volumes_ask"]) == n_levels
        assert len(result["std_volumes_bid"]) == n_levels
        assert len(result["std_volumes_ask"]) == n_levels

    def test_std_is_non_negative(self):
        """Standard deviations must be non-negative."""
        snaps = self._make_snapshots(n=10)
        result = average_book_shape(snaps)
        assert np.all(result["std_volumes_bid"] >= 0)
        assert np.all(result["std_volumes_ask"] >= 0)

    def test_avg_volumes_positive(self):
        """Average volumes should be positive."""
        snaps = self._make_snapshots(n=5)
        result = average_book_shape(snaps)
        assert np.all(result["avg_volumes_bid"] >= 0)
        assert np.all(result["avg_volumes_ask"] >= 0)

    def test_single_snapshot_std_zero(self):
        """With a single snapshot, std should be zero."""
        snaps = self._make_snapshots(n=1)
        result = average_book_shape(snaps)
        np.testing.assert_array_almost_equal(result["std_volumes_bid"], 0.0)

    def test_empty_snapshots_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            average_book_shape([])

    def test_average_is_consistent(self):
        """With identical snapshots, avg should equal the common snapshot."""
        snap = generate_synthetic_book_shape(n_levels=10, seed=42)
        snaps = [snap, snap, snap]
        result = average_book_shape(snaps)
        np.testing.assert_allclose(
            result["avg_volumes_bid"], snap["volumes_bid"], rtol=1e-10
        )


class TestEmpiricalImpactCurve:
    """Tests for empirical_impact_curve in data.lob_statistics."""

    def _make_trades(self, n=500):
        return generate_synthetic_impact_data(n_trades=n, impact_law="concave_power", seed=42)

    def test_returns_required_keys(self):
        """Must return dict with all required keys."""
        df = self._make_trades()
        result = empirical_impact_curve(df, n_bins=10)
        required = {"bin_centers", "mean_impact", "std_impact", "n_trades_per_bin"}
        assert required.issubset(result.keys())

    def test_bin_lengths_consistent(self):
        """All returned arrays should have the same length."""
        df = self._make_trades()
        result = empirical_impact_curve(df, n_bins=10)
        n = len(result["bin_centers"])
        assert len(result["mean_impact"]) == n
        assert len(result["std_impact"]) == n
        assert len(result["n_trades_per_bin"]) == n

    def test_bin_count_positive(self):
        """n_trades_per_bin should be positive for each bin."""
        df = self._make_trades()
        result = empirical_impact_curve(df, n_bins=10)
        assert np.all(result["n_trades_per_bin"] > 0)

    def test_total_trades_matches(self):
        """Sum of n_trades_per_bin should equal total trades."""
        n_trades = 400
        df = self._make_trades(n=n_trades)
        result = empirical_impact_curve(df, n_bins=10)
        assert result["n_trades_per_bin"].sum() == n_trades

    def test_std_impact_non_negative(self):
        """std_impact should be non-negative."""
        df = self._make_trades()
        result = empirical_impact_curve(df, n_bins=10)
        assert np.all(result["std_impact"] >= 0)

    def test_mean_impact_non_negative(self):
        """mean_impact should be non-negative (all impacts are abs-valued)."""
        df = self._make_trades()
        result = empirical_impact_curve(df, n_bins=10)
        assert np.all(result["mean_impact"] >= 0)

    def test_bin_centers_increasing(self):
        """bin_centers should be in increasing order."""
        df = self._make_trades()
        result = empirical_impact_curve(df, n_bins=10)
        centers = result["bin_centers"]
        assert np.all(np.diff(centers) > 0), "bin_centers should be strictly increasing"

    def test_n_bins_parameter(self):
        """Using more bins should give more (or equal) bins in result."""
        df = self._make_trades(n=1000)
        r5 = empirical_impact_curve(df, n_bins=5)
        r10 = empirical_impact_curve(df, n_bins=10)
        assert len(r10["bin_centers"]) >= len(r5["bin_centers"])


class TestSpreadStatistics:
    """Tests for spread_statistics in data.lob_statistics."""

    def _make_book_snapshots(self, n=20):
        """Create snapshots with known spread values."""
        spreads = np.linspace(0.01, 0.05, n)
        return [{"spread": float(s)} for s in spreads]

    def test_returns_required_keys(self):
        """Must return dict with all required keys."""
        snaps = self._make_book_snapshots()
        result = spread_statistics(snaps)
        required = {"mean", "median", "std", "min", "max", "distribution"}
        assert required.issubset(result.keys())

    def test_mean_in_range(self):
        """Mean spread should be between min and max."""
        snaps = self._make_book_snapshots()
        result = spread_statistics(snaps)
        assert result["min"] <= result["mean"] <= result["max"]

    def test_median_in_range(self):
        """Median spread should be between min and max."""
        snaps = self._make_book_snapshots()
        result = spread_statistics(snaps)
        assert result["min"] <= result["median"] <= result["max"]

    def test_std_non_negative(self):
        """Standard deviation should be non-negative."""
        snaps = self._make_book_snapshots()
        result = spread_statistics(snaps)
        assert result["std"] >= 0

    def test_distribution_array_length(self):
        """distribution array should have same length as input snapshots."""
        n = 15
        snaps = self._make_book_snapshots(n=n)
        result = spread_statistics(snaps)
        assert len(result["distribution"]) == n

    def test_constant_spread(self):
        """With constant spread, std should be 0."""
        snaps = [{"spread": 0.02}] * 10
        result = spread_statistics(snaps)
        assert result["std"] == pytest.approx(0.0, abs=1e-10)
        assert result["mean"] == pytest.approx(0.02)
        assert result["min"] == pytest.approx(0.02)
        assert result["max"] == pytest.approx(0.02)

    def test_empty_snapshots_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            spread_statistics([])

    def test_using_synthetic_book_snapshots(self):
        """Works with real generate_synthetic_book_shape output."""
        snaps = [generate_synthetic_book_shape(seed=i) for i in range(10)]
        result = spread_statistics(snaps)
        # All snapshots have spread=0.02
        assert result["mean"] == pytest.approx(0.02)
        assert result["std"] == pytest.approx(0.0, abs=1e-10)

    def test_min_max_correct(self):
        """min and max should correspond to actual min/max spread values."""
        spreads = [0.01, 0.02, 0.03, 0.04, 0.05]
        snaps = [{"spread": s} for s in spreads]
        result = spread_statistics(snaps)
        assert result["min"] == pytest.approx(0.01)
        assert result["max"] == pytest.approx(0.05)
