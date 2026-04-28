"""Tests for analysis modules."""
import numpy as np
import pytest

from mfg_lob.solvers.grid import Grid
from mfg_lob.solvers.mfg_solver import MFGSolver
from mfg_lob.models.lob_model import LOBModel
from mfg_lob.analysis.price_impact import PriceImpactAnalyzer
from mfg_lob.analysis.lob_shape import LOBShapeAnalyzer


GRID = Grid.uniform(q_min=-5, q_max=5, nq=31, t_min=0, t_max=0.5, nt=51)


@pytest.fixture(scope="module")
def equilibrium():
    model = LOBModel(sigma=0.5, phi=0.1, alpha=0.5, A_term=1.0)
    solver = MFGSolver(model, GRID, max_iter=5, tol=1e-3, verbose=False)
    return solver.solve()


class TestPriceImpactAnalyzer:
    def test_illiquidity_shape(self, equilibrium):
        pia = PriceImpactAnalyzer(equilibrium)
        q, lam = pia.illiquidity(t=0.0)
        assert q.shape == (GRID.nq,)
        assert lam.shape == (GRID.nq,)
        assert np.all(lam > 0)

    def test_impact_curve_shape(self, equilibrium):
        pia = PriceImpactAnalyzer(equilibrium)
        curve = pia.compute(t=0.0, n_volumes=10)
        assert len(curve.volumes) == 10
        assert len(curve.impact) == 10

    def test_impact_nonnegative(self, equilibrium):
        pia = PriceImpactAnalyzer(equilibrium)
        curve = pia.compute(t=0.0, n_volumes=10)
        assert np.all(curve.impact >= 0)

    def test_impact_increasing(self, equilibrium):
        pia = PriceImpactAnalyzer(equilibrium)
        curve = pia.compute(t=0.0, n_volumes=20)
        # Impact should be weakly increasing in volume
        assert np.all(np.diff(curve.impact) >= -1e-10)


class TestLOBShapeAnalyzer:
    def test_snapshot_shape(self, equilibrium):
        lsa = LOBShapeAnalyzer(equilibrium)
        snap = lsa.snapshot(t=0.0)
        assert snap.density.shape == (GRID.nq,)
        assert snap.q.shape == (GRID.nq,)

    def test_density_nonnegative(self, equilibrium):
        lsa = LOBShapeAnalyzer(equilibrium)
        snap = lsa.snapshot(t=0.0)
        assert np.all(snap.density >= 0)

    def test_spread_proxy_positive(self, equilibrium):
        lsa = LOBShapeAnalyzer(equilibrium)
        snap = lsa.snapshot(t=0.0)
        assert snap.spread_proxy >= 0

    def test_time_evolution_count(self, equilibrium):
        lsa = LOBShapeAnalyzer(equilibrium)
        snaps = lsa.time_evolution(n_snapshots=3)
        assert len(snaps) == 3

    def test_depth_profile_keys(self, equilibrium):
        lsa = LOBShapeAnalyzer(equilibrium)
        prof = lsa.depth_profile(t=0.0, n_levels=5)
        assert "levels" in prof
        assert "bid_volume" in prof
        assert "ask_volume" in prof
        assert len(prof["levels"]) == 5
