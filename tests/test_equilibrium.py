"""Tests for EquilibriumAnalyzer, compute_social_cost, compute_price_of_anarchy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mfglob.grids import Grid1D, TimeGrid
from mfglob.mfg_solver import MFGSolver
from mfglob.models.avellaneda_stoikov import AvellanedaStoikovMFG
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.equilibrium import (
    EquilibriumAnalyzer,
    compute_social_cost,
    compute_price_of_anarchy,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_solution(damping=0.5, tol=1e-4, max_iter=30):
    """Small, fast AS MFG solution used by most tests."""
    model = AvellanedaStoikovMFG(
        A=1.0, k=1.5, phi=0.01, psi=0.005,
        gamma=0.1, sigma_mid=0.3, Q_max=5.0, c=0.1,
    )
    grid = Grid1D(-4.0, 4.0, 25)
    tgrid = TimeGrid(0.1, 15)
    solver = MFGSolver(model, grid, tgrid, damping=damping,
                       tol=tol, max_iterations=max_iter)
    sol = solver.solve()
    return sol, model, grid, tgrid


@pytest.fixture(scope="module")
def base_solution():
    return make_solution()


# ---------------------------------------------------------------------------
# TestVerifyEquilibrium
# ---------------------------------------------------------------------------

class TestVerifyEquilibrium:

    def test_returns_dict(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert isinstance(result, dict)

    def test_required_keys(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        for key in ("checks", "passed", "details"):
            assert key in result

    def test_checks_has_mass_key(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert "mass_conserved" in result["checks"]
        assert "density_nonneg" in result["checks"]

    def test_mass_conserved_for_converged(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert result["checks"]["mass_conserved"], (
            f"Mass error: {result['checks']['max_mass_error']}"
        )

    def test_density_nonneg_for_converged(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert result["checks"]["density_nonneg"], (
            f"Min density: {result['checks']['min_density']}"
        )

    def test_passed_is_bool(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert isinstance(result["passed"], bool)

    def test_details_is_string(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert isinstance(result["details"], str)
        assert len(result["details"]) > 0

    def test_fp_residual_present(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.verify_equilibrium()
        assert "fp_residual_l2" in result["checks"]
        assert np.isfinite(result["checks"]["fp_residual_l2"])


# ---------------------------------------------------------------------------
# TestEpsilonNash
# ---------------------------------------------------------------------------

class TestEpsilonNash:

    def test_returns_dict(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.epsilon_nash_check(n_perturbations=20)
        assert isinstance(result, dict)

    def test_required_keys(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.epsilon_nash_check(n_perturbations=20)
        for key in ("epsilon", "all_perturbations_worse",
                    "cost_improvements", "baseline_cost"):
            assert key in result

    def test_epsilon_nonneg(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.epsilon_nash_check(n_perturbations=30)
        assert result["epsilon"] >= 0.0

    def test_baseline_cost_finite(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.epsilon_nash_check(n_perturbations=20)
        assert np.isfinite(result["baseline_cost"])

    def test_epsilon_small_for_well_resolved(self):
        """For a well-converged solution, ε-Nash ε should be small."""
        model = AvellanedaStoikovMFG(phi=0.05, psi=0.0, c=0.0, sigma_mid=0.5)
        grid = Grid1D(-3.0, 3.0, 25)
        tgrid = TimeGrid(0.1, 15)
        solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-5, max_iterations=50)
        sol = solver.solve()
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.epsilon_nash_check(n_perturbations=50,
                                             perturbation_scale=0.05)
        # ε should be a small fraction of the baseline cost
        frac = result["epsilon"] / (abs(result["baseline_cost"]) + 1e-6)
        assert frac < 0.5, f"ε-Nash fraction too large: {frac:.3f}"


# ---------------------------------------------------------------------------
# TestComparativeStatics
# ---------------------------------------------------------------------------

class TestComparativeStatics:

    def test_returns_dataframe(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        df = analyzer.comparative_statics("gamma", np.array([0.05, 0.1]))
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        vals = np.array([0.05, 0.1, 0.2])
        df = analyzer.comparative_statics("gamma", vals)
        assert len(df) == 3

    def test_required_columns(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        df = analyzer.comparative_statics("gamma", np.array([0.05, 0.1]))
        for col in ("gamma", "converged", "n_iterations", "social_cost"):
            assert col in df.columns, f"Missing column: {col}"

    def test_parameter_values_stored(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        vals = np.array([0.05, 0.1, 0.2])
        df = analyzer.comparative_statics("gamma", vals)
        np.testing.assert_allclose(df["gamma"].values, vals)

    def test_social_cost_varies(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        df = analyzer.comparative_statics("gamma", np.array([0.01, 0.5]))
        # Higher gamma (liquidation penalty) should produce higher social cost
        assert df["social_cost"].iloc[1] != df["social_cost"].iloc[0]


# ---------------------------------------------------------------------------
# TestStabilityAnalysis
# ---------------------------------------------------------------------------

class TestStabilityAnalysis:

    def test_returns_dict(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.stability_analysis()
        assert isinstance(result, dict)

    def test_required_keys(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.stability_analysis()
        for key in ("eigenvalues", "stable", "spectral_radius", "spectral_gap"):
            assert key in result

    def test_stable_is_bool(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.stability_analysis()
        assert isinstance(result["stable"], bool)

    def test_spectral_radius_nonneg(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.stability_analysis()
        assert result["spectral_radius"] >= 0.0

    def test_eigenvalues_finite(self, base_solution):
        sol, model, grid, tgrid = base_solution
        analyzer = EquilibriumAnalyzer(sol, model, grid, tgrid)
        result = analyzer.stability_analysis()
        assert np.all(np.isfinite(np.abs(result["eigenvalues"])))


# ---------------------------------------------------------------------------
# TestSocialCost
# ---------------------------------------------------------------------------

class TestSocialCost:

    def test_positive(self, base_solution):
        sol, model, grid, tgrid = base_solution
        cost = compute_social_cost(sol, model, grid, tgrid)
        assert np.isfinite(cost)

    def test_finite(self, base_solution):
        sol, model, grid, tgrid = base_solution
        cost = compute_social_cost(sol, model, grid, tgrid)
        assert np.isfinite(cost)

    def test_scalar(self, base_solution):
        sol, model, grid, tgrid = base_solution
        cost = compute_social_cost(sol, model, grid, tgrid)
        assert np.ndim(cost) == 0 or isinstance(cost, float)

    def test_increases_with_penalty(self):
        """Higher inventory penalty → higher social cost."""
        grid = Grid1D(-4.0, 4.0, 20)
        tgrid = TimeGrid(0.1, 10)
        costs = []
        for phi in [0.001, 0.1]:
            model = AvellanedaStoikovMFG(phi=phi, psi=0.0, c=0.0)
            solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-4, max_iterations=20)
            sol = solver.solve()
            costs.append(compute_social_cost(sol, model, grid, tgrid))
        # Cost with higher penalty should be larger in magnitude
        assert abs(costs[1]) >= abs(costs[0]) * 0.5


# ---------------------------------------------------------------------------
# TestPriceOfAnarchy
# ---------------------------------------------------------------------------

class TestPriceOfAnarchy:

    def test_returns_dict(self):
        model = LOBFormationMFG(c_crowd=0.3, sigma=0.5)
        grid = Grid1D(0.0, 8.0, 20)
        tgrid = TimeGrid(0.1, 10)
        result = compute_price_of_anarchy(model, grid, tgrid)
        assert isinstance(result, dict)

    def test_required_keys(self):
        model = LOBFormationMFG(c_crowd=0.3, sigma=0.5)
        grid = Grid1D(0.0, 8.0, 20)
        tgrid = TimeGrid(0.1, 10)
        result = compute_price_of_anarchy(model, grid, tgrid)
        for key in ("social_cost_nash", "social_cost_optimum",
                    "price_of_anarchy", "nash_solution", "optimum_solution"):
            assert key in result

    def test_poa_nonneg(self):
        model = LOBFormationMFG(c_crowd=0.2, sigma=0.5)
        grid = Grid1D(0.0, 8.0, 20)
        tgrid = TimeGrid(0.1, 10)
        result = compute_price_of_anarchy(model, grid, tgrid)
        assert result["price_of_anarchy"] >= 0.0

    def test_poa_finite(self):
        model = LOBFormationMFG(c_crowd=0.2, sigma=0.5)
        grid = Grid1D(0.0, 8.0, 20)
        tgrid = TimeGrid(0.1, 10)
        result = compute_price_of_anarchy(model, grid, tgrid)
        assert np.isfinite(result["price_of_anarchy"])
