"""Tests for calibration module."""
import numpy as np
import pytest

from mfg_lob.calibration.empirical import LOBSnapshot, EmpiricalLOB
from mfg_lob.calibration.calibrator import MFGCalibrator
from mfg_lob.solvers.grid import Grid


class TestLOBSnapshot:
    def test_synthetic_construction(self):
        snap = LOBSnapshot.synthetic(mid_price=100.0, spread=0.02, n_levels=10)
        assert len(snap.bid_prices) == 10
        assert len(snap.ask_prices) == 10

    def test_spread(self):
        snap = LOBSnapshot.synthetic(mid_price=100.0, spread=0.02, n_levels=5)
        assert snap.spread == pytest.approx(0.02, rel=0.1)

    def test_imbalance_range(self):
        snap = LOBSnapshot.synthetic()
        assert -1.0 <= snap.imbalance <= 1.0

    def test_depth_profile_keys(self):
        snap = LOBSnapshot.synthetic(n_levels=5)
        prof = snap.depth_profile(n_levels=5, tick_size=0.01)
        assert "levels" in prof
        assert "bid_cum" in prof
        assert "ask_cum" in prof


class TestEmpiricalLOB:
    def test_from_synthetic(self):
        emp = EmpiricalLOB.from_synthetic(n_snapshots=10)
        assert len(emp) == 10

    def test_mean_spread_positive(self):
        emp = EmpiricalLOB.from_synthetic(n_snapshots=20, spread_mean=0.02)
        assert emp.mean_spread > 0

    def test_depth_profile_shape(self):
        emp = EmpiricalLOB.from_synthetic(n_snapshots=5)
        prof = emp.mean_depth_profile(n_levels=8)
        assert len(prof["levels"]) == 8
        assert len(prof["bid_cum"]) == 8
        assert len(prof["ask_cum"]) == 8

    def test_empirical_impact_nonnegative(self):
        emp = EmpiricalLOB.from_synthetic(n_snapshots=5)
        vols = np.linspace(0, 100, 10)
        impact = emp.empirical_impact(vols, tick_size=0.01)
        assert np.all(impact >= 0)


class TestMFGCalibrator:
    def test_calibration_runs(self):
        grid = Grid.uniform(q_min=-5, q_max=5, nq=21, t_min=0, t_max=0.3, nt=31)
        emp = EmpiricalLOB.from_synthetic(n_snapshots=10, spread_mean=0.5)
        base_params = {
            "A_term": 1.0, "B_term": 0.0,
            "Lambda0": 1.0, "kappa": 1.0,
            "mu0": 0.0, "sigma0": 2.0,
            "agent_type": "market_maker",
        }
        calib = MFGCalibrator(
            empirical=emp,
            grid=grid,
            param_names=["sigma", "phi"],
            param_bounds=[(0.1, 2.0), (0.01, 1.0)],
            base_params=base_params,
            solver_kwargs={"max_iter": 5, "tol": 1e-3},
            n_depth_levels=5,
            n_impact_points=5,
        )
        result = calib.calibrate(max_iter=5)
        assert "sigma" in result.params
        assert "phi" in result.params
        assert result.loss < 1e5
        assert result.n_evaluations > 0
