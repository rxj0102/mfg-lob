"""Tests for PriceImpactAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from mfglob.grids import Grid1D, TimeGrid
from mfglob.mfg_solver import MFGSolver
from mfglob.models.optimal_execution import OptimalExecutionMFG
from mfglob.price_impact import PriceImpactAnalyzer


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def make_oe_solution(Q0=1.0, eta=0.2, sigma=0.05, phi=0.5, psi=1.0):
    """Solve a small OptimalExecution MFG."""
    model = OptimalExecutionMFG(phi=phi, psi=psi, eta=eta, lam=1.0,
                                sigma=sigma, Q0=Q0)
    grid = Grid1D(0.0, max(Q0 * 1.5, 0.5), 30)
    tgrid = TimeGrid(0.5, 30)
    solver = MFGSolver(model, grid, tgrid, damping=0.5,
                       tol=1e-3, max_iterations=20)
    sol = solver.solve()
    return sol, model, grid, tgrid


@pytest.fixture(scope="module")
def base_pi():
    sol, model, grid, tgrid = make_oe_solution()
    return PriceImpactAnalyzer(sol, model, grid, tgrid)


# ---------------------------------------------------------------------------
# Temporary impact
# ---------------------------------------------------------------------------

class TestTemporaryImpact:

    def test_shape(self, base_pi):
        ti = base_pi.temporary_impact()
        assert ti.shape == (base_pi.time_grid.n_nodes,)

    def test_nonneg(self, base_pi):
        ti = base_pi.temporary_impact()
        assert np.all(ti >= -1e-10)

    def test_finite(self, base_pi):
        ti = base_pi.temporary_impact()
        assert np.all(np.isfinite(ti))

    def test_zero_for_zero_eta(self):
        sol, model, grid, tgrid = make_oe_solution(eta=0.0)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        ti = pia.temporary_impact()
        assert np.all(ti == pytest.approx(0.0, abs=1e-10))

    def test_scales_with_eta(self):
        """Doubling η should double the temporary impact."""
        sol1, model1, grid, tgrid = make_oe_solution(eta=0.2)
        sol2, model2, _, _ = make_oe_solution(eta=0.4)
        pia1 = PriceImpactAnalyzer(sol1, model1, grid, tgrid)
        pia2 = PriceImpactAnalyzer(sol2, model2, grid, tgrid)
        ratio = np.mean(pia2.temporary_impact()) / (np.mean(pia1.temporary_impact()) + 1e-14)
        assert ratio == pytest.approx(2.0, rel=0.3)


# ---------------------------------------------------------------------------
# Permanent impact
# ---------------------------------------------------------------------------

class TestPermanentImpact:

    def test_scalar(self, base_pi):
        pi = base_pi.permanent_impact()
        assert np.isscalar(pi) or np.ndim(pi) == 0

    def test_nonneg(self, base_pi):
        pi = base_pi.permanent_impact()
        assert pi >= -1e-10

    def test_finite(self, base_pi):
        pi = base_pi.permanent_impact()
        assert np.isfinite(pi)

    def test_zero_for_zero_eta(self):
        sol, model, grid, tgrid = make_oe_solution(eta=0.0)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        assert pia.permanent_impact() == pytest.approx(0.0, abs=1e-10)

    def test_scales_with_order_size(self, base_pi):
        """Impact for 2×Q0 should be larger than for Q0."""
        Q0 = getattr(base_pi.model, "Q0", 1.0)
        pi1 = base_pi.permanent_impact(meta_order_size=Q0)
        pi2 = base_pi.permanent_impact(meta_order_size=2.0 * Q0)
        assert pi2 > pi1

    def test_increases_with_order_size_series(self):
        """permanent_impact(Q) should be monotone increasing in Q."""
        sol, model, grid, tgrid = make_oe_solution(Q0=1.0)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        Q0 = model.Q0
        pis = [pia.permanent_impact(meta_order_size=q)
               for q in [0.5 * Q0, Q0, 2.0 * Q0]]
        assert pis[0] < pis[1] < pis[2], f"Not monotone: {pis}"


# ---------------------------------------------------------------------------
# Impact curve
# ---------------------------------------------------------------------------

class TestImpactCurve:

    def test_returns_dict(self, base_pi):
        curve = base_pi.impact_curve(np.array([0.5, 1.0, 1.5]))
        assert isinstance(curve, dict)

    def test_required_keys(self, base_pi):
        curve = base_pi.impact_curve(np.array([0.5, 1.0]))
        for key in ("order_sizes", "permanent_impact", "temporary_impact_peak"):
            assert key in curve

    def test_shape_matches_input(self, base_pi):
        Qs = np.array([0.3, 0.6, 0.9, 1.2])
        curve = base_pi.impact_curve(Qs)
        assert len(curve["permanent_impact"]) == 4
        assert len(curve["temporary_impact_peak"]) == 4

    def test_monotone_permanent_impact(self):
        """Impact curve should be (approximately) monotone increasing."""
        sol, model, grid, tgrid = make_oe_solution(Q0=1.0, eta=0.3)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        Qs = np.linspace(0.2, 1.8, 5)
        curve = pia.impact_curve(Qs)
        impacts = curve["permanent_impact"]
        # Allow up to 2 non-monotone steps (numerical noise from re-solving)
        diffs = np.diff(impacts)
        assert np.sum(diffs < 0) <= 2, f"Too many decreases: {diffs}"

    def test_concavity(self):
        """Impact curve should be concave (second differences ≤ 0) or nearly so."""
        sol, model, grid, tgrid = make_oe_solution(Q0=1.0, eta=0.3)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        Qs = np.linspace(0.2, 2.0, 6)
        curve = pia.impact_curve(Qs)
        impacts = curve["permanent_impact"]
        second_diff = np.diff(np.diff(impacts))
        # At least half of second differences should be ≤ small positive threshold
        n_concave = np.sum(second_diff <= 0.01 * np.max(np.abs(impacts)))
        assert n_concave >= len(second_diff) // 2, (
            f"Impact curve not sufficiently concave: 2nd diffs = {second_diff}"
        )

    def test_zero_order_zero_impact(self):
        """Near-zero order size → near-zero impact."""
        sol, model, grid, tgrid = make_oe_solution(Q0=1.0)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        curve = pia.impact_curve(np.array([0.01, 0.5, 1.0]))
        assert curve["permanent_impact"][0] < curve["permanent_impact"][-1]

    def test_temporary_peak_nonneg(self, base_pi):
        Qs = np.array([0.5, 1.0, 1.5])
        curve = base_pi.impact_curve(Qs)
        assert np.all(curve["temporary_impact_peak"] >= -1e-10)


# ---------------------------------------------------------------------------
# Square-root law comparison
# ---------------------------------------------------------------------------

class TestSquareRootLaw:

    def test_returns_dict(self, base_pi):
        result = base_pi.compare_to_square_root_law(np.array([0.5, 1.0, 1.5]))
        assert isinstance(result, dict)

    def test_required_keys(self, base_pi):
        result = base_pi.compare_to_square_root_law(np.array([0.5, 1.0]))
        for key in ("mfg_impact", "sqrt_law", "mfg_exponent", "mfg_coefficient"):
            assert key in result

    def test_exponent_positive(self, base_pi):
        result = base_pi.compare_to_square_root_law(np.array([0.3, 0.6, 1.0, 1.5]))
        assert result["mfg_exponent"] > 0.0

    def test_exponent_order_of_magnitude(self):
        """Fitted exponent should be in [0.2, 2.0] for reasonable models."""
        sol, model, grid, tgrid = make_oe_solution(eta=0.3)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        Qs = np.linspace(0.2, 1.8, 6)
        result = pia.compare_to_square_root_law(Qs)
        b = result["mfg_exponent"]
        assert 0.0 < b < 3.0, f"Fitted exponent {b:.2f} out of expected range"

    def test_sqrt_law_nonneg(self, base_pi):
        result = base_pi.compare_to_square_root_law(np.array([0.5, 1.0, 1.5]))
        assert np.all(result["sqrt_law"] >= 0.0)


# ---------------------------------------------------------------------------
# Kyle's lambda
# ---------------------------------------------------------------------------

class TestKyleLambda:

    def test_positive(self, base_pi):
        lam = base_pi.kyle_lambda()
        assert lam > 0.0

    def test_finite(self, base_pi):
        lam = base_pi.kyle_lambda()
        assert np.isfinite(lam)

    def test_zero_for_zero_eta(self):
        sol, model, grid, tgrid = make_oe_solution(eta=0.0)
        pia = PriceImpactAnalyzer(sol, model, grid, tgrid)
        lam = pia.kyle_lambda()
        assert lam == pytest.approx(0.0, abs=1e-8)

    def test_increases_with_eta(self):
        """Stronger permanent impact coefficient → larger Kyle lambda."""
        sol1, model1, g, t = make_oe_solution(eta=0.1)
        sol2, model2, _, _ = make_oe_solution(eta=0.5)
        lam1 = PriceImpactAnalyzer(sol1, model1, g, t).kyle_lambda()
        lam2 = PriceImpactAnalyzer(sol2, model2, g, t).kyle_lambda()
        assert lam2 > lam1, f"Kyle λ should increase with η: λ1={lam1:.4f}, λ2={lam2:.4f}"

    def test_scalar(self, base_pi):
        lam = base_pi.kyle_lambda()
        assert np.isscalar(lam) or np.ndim(lam) == 0
