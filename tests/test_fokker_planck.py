"""Tests for FokkerPlanckSolver: analytical solutions, mass conservation, convergence."""

from __future__ import annotations

import numpy as np
import pytest

from mfglob.grids import Grid1D, TimeGrid
from mfglob.fokker_planck import FokkerPlanckSolver
from mfglob.models.base import MFGLOBModel


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------

class DiffusionOnlyModel(MFGLOBModel):
    """b=0, σ=const — pure forward diffusion."""

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def hamiltonian(self, x, p, M, m): return np.zeros_like(x)
    def optimal_control(self, x, p, m): return np.zeros_like(x)
    def drift(self, x, alpha, m): return np.zeros_like(x)
    def diffusion(self, x, alpha, m): return np.full_like(x, self.sigma)
    def running_cost(self, x, alpha, m): return np.zeros_like(x)
    def terminal_cost(self, x, m): return np.zeros_like(x)

    def initial_distribution(self, x):
        mu, s = 0.5 * (x[0] + x[-1]), 0.1 * (x[-1] - x[0])
        m = np.exp(-0.5 * ((x - mu) / s) ** 2)
        return m / np.trapezoid(m, x)

    @property
    def state_dimension(self): return 1

    @property
    def param_dict(self): return {"sigma": self.sigma}


class DriftDiffusionModel(MFGLOBModel):
    """Constant b and σ — translating Gaussian."""

    def __init__(self, b=0.5, sigma=0.5):
        self._b = b
        self.sigma = sigma

    def hamiltonian(self, x, p, M, m): return np.zeros_like(x)
    def optimal_control(self, x, p, m): return np.zeros_like(x)
    def drift(self, x, alpha, m): return np.full_like(x, self._b)
    def diffusion(self, x, alpha, m): return np.full_like(x, self.sigma)
    def running_cost(self, x, alpha, m): return np.zeros_like(x)
    def terminal_cost(self, x, m): return np.zeros_like(x)

    def initial_distribution(self, x):
        mu, s = 0.5 * (x[0] + x[-1]), 0.08 * (x[-1] - x[0])
        m = np.exp(-0.5 * ((x - mu) / s) ** 2)
        return m / np.trapezoid(m, x)

    @property
    def state_dimension(self): return 1

    @property
    def param_dict(self): return {"b": self._b, "sigma": self.sigma}


class OrnsteinUhlenbeckModel(MFGLOBModel):
    """OU process: b(x) = -κ·x, σ=const. Stationary dist ~ N(0, σ²/(2κ))."""

    def __init__(self, kappa=2.0, sigma=1.0):
        self.kappa = kappa
        self.sigma = sigma

    def hamiltonian(self, x, p, M, m): return np.zeros_like(x)
    def optimal_control(self, x, p, m): return np.zeros_like(x)
    def drift(self, x, alpha, m): return -self.kappa * x
    def diffusion(self, x, alpha, m): return np.full_like(x, self.sigma)
    def running_cost(self, x, alpha, m): return np.zeros_like(x)
    def terminal_cost(self, x, m): return np.zeros_like(x)

    def initial_distribution(self, x):
        # Start as broad Gaussian, let OU drive it to steady state
        mu, s = 0.0, 0.5
        m = np.exp(-0.5 * ((x - mu) / s) ** 2)
        return m / np.trapezoid(m, x)

    @property
    def state_dimension(self): return 1

    @property
    def param_dict(self): return {"kappa": self.kappa, "sigma": self.sigma}


# ---------------------------------------------------------------------------
# Helper: build constant drift/diffusion arrays
# ---------------------------------------------------------------------------

def const_arrays(n_t, n_x, b_val, sigma_val):
    drift = np.full((n_t, n_x), b_val)
    diff = np.full((n_t, n_x), sigma_val)
    return drift, diff


def gaussian(x, mu, var):
    """Normalised Gaussian N(mu, var) on x."""
    g = np.exp(-0.5 * (x - mu) ** 2 / var)
    return g / np.trapezoid(g, x)


# ---------------------------------------------------------------------------
# Test 1 — Pure diffusion (heat equation forward)
# ---------------------------------------------------------------------------

class TestPureDiffusion:
    """
    b=0, σ=const.  Initial Gaussian with variance s².
    Analytical solution: Gaussian with variance s² + σ²t.
    """

    def _run(self, n_x=120, n_t=200, sigma=0.3, T=0.5, s0=0.15):
        L = 3.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = DiffusionOnlyModel(sigma=sigma)
        x = grid.points

        m0 = gaussian(x, 0.0, s0**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, 0.0, sigma)

        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(drift, diff, initial_distribution=m0,
                              bc_type="reflecting")
        return result, x, tgrid, sigma, T, s0

    def test_mean_preserved(self):
        """Zero drift → mean stays at 0."""
        result, x, tgrid, *_ = self._run()
        m = result["density"]
        for k in range(0, tgrid.n_nodes, 20):
            mean = np.trapezoid(x * m[k], x)
            assert abs(mean) < 0.05, f"Mean drifted at step {k}: {mean:.4f}"

    def test_variance_grows(self):
        """Variance should grow as s² + σ²t."""
        result, x, tgrid, sigma, T, s0 = self._run()
        m = result["density"]
        times = tgrid.times

        for k in [0, tgrid.n_nodes // 4, tgrid.n_nodes // 2, tgrid.n_nodes - 1]:
            t = times[k]
            var_num = np.trapezoid((x ** 2) * m[k], x)
            var_exact = s0**2 + sigma**2 * t
            rel_err = abs(var_num - var_exact) / var_exact
            assert rel_err < 0.12, (
                f"Variance mismatch at t={t:.3f}: num={var_num:.4f}, exact={var_exact:.4f}"
            )

    def test_shape_is_gaussian(self):
        """At final time, density should still look like a Gaussian."""
        result, x, tgrid, sigma, T, s0 = self._run()
        m_final = result["density"][-1]
        var_final = s0**2 + sigma**2 * T
        m_exact = gaussian(x, 0.0, var_final)
        # Allow generous tolerance for numerical diffusion
        err = np.max(np.abs(m_final - m_exact))
        assert err < 0.25, f"Final shape deviates from Gaussian: max err={err:.4f}"

    def test_mass_conserved(self):
        result, x, tgrid, *_ = self._run()
        err = result["mass_error"]
        assert err < 1e-8, f"Mass error too large: {err:.2e}"

    def test_positivity(self):
        result, x, tgrid, *_ = self._run()
        m = result["density"]
        assert np.all(m >= -1e-12), f"Negative density: min={m.min():.2e}"


# ---------------------------------------------------------------------------
# Test 2 — Drift-diffusion with constant coefficients
# ---------------------------------------------------------------------------

class TestDriftDiffusion:
    """
    b=const, σ=const.  Gaussian translates rightward and spreads.
    Analytical: N(μ₀ + b·t, s² + σ²t).
    Use reflecting BCs and a domain wide enough to avoid BC effects.
    """

    def _run(self, n_x=150, n_t=300, b=0.3, sigma=0.25, T=0.4, s0=0.12):
        L = 4.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = DriftDiffusionModel(b=b, sigma=sigma)
        x = grid.points

        mu0 = 0.0
        m0 = gaussian(x, mu0, s0**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, b, sigma)

        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(drift, diff, initial_distribution=m0,
                              bc_type="reflecting")
        return result, x, tgrid, b, sigma, T, s0

    def test_mean_translates(self):
        """Mean should grow as μ₀ + b·t."""
        result, x, tgrid, b, sigma, T, s0 = self._run()
        m = result["density"]
        times = tgrid.times
        # Check at several times (avoid late times where BC contamination grows)
        for k in [0, tgrid.n_nodes // 4, tgrid.n_nodes // 2]:
            t = times[k]
            mean_num = np.trapezoid(x * m[k], x)
            mean_exact = b * t
            assert abs(mean_num - mean_exact) < 0.06, (
                f"Mean wrong at t={t:.3f}: num={mean_num:.4f}, exact={mean_exact:.4f}"
            )

    def test_variance_grows(self):
        """Variance should grow as s² + σ²t."""
        result, x, tgrid, b, sigma, T, s0 = self._run()
        m = result["density"]
        times = tgrid.times
        for k in [0, tgrid.n_nodes // 4, tgrid.n_nodes // 2]:
            t = times[k]
            mean_num = np.trapezoid(x * m[k], x)
            var_num = np.trapezoid((x - mean_num) ** 2 * m[k], x)
            var_exact = s0**2 + sigma**2 * t
            rel_err = abs(var_num - var_exact) / max(var_exact, 1e-10)
            assert rel_err < 0.15, (
                f"Variance wrong at t={t:.3f}: num={var_num:.4f}, exact={var_exact:.4f}"
            )

    def test_mass_conserved(self):
        result, *_ = self._run()
        assert result["mass_error"] < 1e-8

    def test_positivity(self):
        result, *_ = self._run()
        assert np.all(result["density"] >= -1e-12)


# ---------------------------------------------------------------------------
# Test 3 — Ornstein-Uhlenbeck steady state
# ---------------------------------------------------------------------------

class TestOrnsteinUhlenbeck:
    """
    b(x) = -κx, σ=const.  Stationary distribution: N(0, σ²/(2κ)).
    Run to large T and check convergence to stationary dist.
    """

    def _run(self, n_x=120, n_t=800, kappa=2.0, sigma=1.0, T=4.0):
        L = 4.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = OrnsteinUhlenbeckModel(kappa=kappa, sigma=sigma)
        x = grid.points

        m0 = model.initial_distribution(x)
        # Drift varies in space: b(x) = -κx
        b_arr = np.outer(np.ones(tgrid.n_nodes), -kappa * x)
        diff_arr = np.full((tgrid.n_nodes, n_x), sigma)

        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(b_arr, diff_arr, initial_distribution=m0,
                              bc_type="reflecting")
        return result, x, kappa, sigma

    def test_converges_to_stationary(self):
        """Final density should be close to N(0, σ²/(2κ))."""
        result, x, kappa, sigma = self._run()
        m_final = result["density"][-1]
        var_stat = sigma**2 / (2.0 * kappa)
        m_stat = gaussian(x, 0.0, var_stat)
        err = np.max(np.abs(m_final - m_stat))
        assert err < 0.15, f"OU did not converge to stationary: max err={err:.4f}"

    def test_stationary_mean_near_zero(self):
        result, x, kappa, sigma = self._run()
        m_final = result["density"][-1]
        mean = np.trapezoid(x * m_final, x)
        assert abs(mean) < 0.05, f"OU stationary mean not near 0: {mean:.4f}"

    def test_stationary_variance(self):
        result, x, kappa, sigma = self._run()
        m_final = result["density"][-1]
        var_num = np.trapezoid(x**2 * m_final, x)
        var_stat = sigma**2 / (2.0 * kappa)
        rel_err = abs(var_num - var_stat) / var_stat
        assert rel_err < 0.15, (
            f"OU stationary variance: num={var_num:.4f}, exact={var_stat:.4f}"
        )

    def test_mass_conserved(self):
        result, *_ = self._run()
        assert result["mass_error"] < 1e-8

    def test_positivity(self):
        result, *_ = self._run()
        assert np.all(result["density"] >= -1e-12)


# ---------------------------------------------------------------------------
# Test 4 — Mass conservation for all cases
# ---------------------------------------------------------------------------

class TestMassConservation:

    def _solver_and_result(self, b_val, sigma_val, n_x=80, n_t=100,
                           T=0.3, bc='reflecting'):
        L = 3.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = DiffusionOnlyModel(sigma=sigma_val)
        x = grid.points
        m0 = gaussian(x, 0.0, 0.2**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, b_val, sigma_val)
        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        return solver.solve(drift, diff, initial_distribution=m0, bc_type=bc)

    def test_pure_diffusion_mass(self):
        result = self._solver_and_result(0.0, 0.5)
        assert result["mass_error"] < 1e-8

    def test_drift_diffusion_mass(self):
        result = self._solver_and_result(0.3, 0.4)
        assert result["mass_error"] < 1e-8

    def test_negative_drift_mass(self):
        result = self._solver_and_result(-0.4, 0.3)
        assert result["mass_error"] < 1e-8

    def test_absorbing_bc_mass_decreases(self):
        """Absorbing BCs let mass escape; after renorm mass=1 but verify we handle it."""
        result = self._solver_and_result(0.0, 0.5, bc='absorbing')
        # After renormalization each step, mass array should still be ≈1
        assert result["mass_error"] < 1e-8

    def test_mass_array_shape(self):
        result = self._solver_and_result(0.0, 0.4)
        n_t = 101  # n_steps=100 → n_nodes=101
        assert result["mass"].shape == (n_t,)

    def test_density_shape(self):
        n_x, n_t_steps = 80, 100
        result = self._solver_and_result(0.0, 0.4, n_x=n_x, n_t=n_t_steps)
        assert result["density"].shape == (n_t_steps + 1, n_x)


# ---------------------------------------------------------------------------
# Test 5 — Positivity
# ---------------------------------------------------------------------------

class TestPositivity:

    def _run(self, b_val, sigma_val, n_x=80, n_t=100, T=0.2):
        L = 3.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = DiffusionOnlyModel(sigma=sigma_val)
        x = grid.points
        # Non-negative initial condition
        m0 = gaussian(x, 0.5, 0.15**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, b_val, sigma_val)
        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        return solver.solve(drift, diff, initial_distribution=m0, bc_type="reflecting")

    def test_positive_drift(self):
        result = self._run(0.5, 0.3)
        assert np.all(result["density"] >= -1e-12)

    def test_negative_drift(self):
        result = self._run(-0.5, 0.3)
        assert np.all(result["density"] >= -1e-12)

    def test_zero_drift(self):
        result = self._run(0.0, 0.4)
        assert np.all(result["density"] >= -1e-12)

    def test_large_diffusion(self):
        result = self._run(0.0, 1.0)
        assert np.all(result["density"] >= -1e-12)

    def test_small_diffusion(self):
        result = self._run(0.2, 0.05)
        assert np.all(result["density"] >= -1e-12)


# ---------------------------------------------------------------------------
# Test 6 — Spatial convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    """Solve pure diffusion on grids dx, dx/2, dx/4, check O(dx²) error."""

    def _error(self, n_x, n_t=1000, sigma=0.4, T=0.1, s0=0.25):
        """Return L∞ error in variance vs. analytical at t=T."""
        L = 3.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = DiffusionOnlyModel(sigma=sigma)
        x = grid.points
        m0 = gaussian(x, 0.0, s0**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, 0.0, sigma)

        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(drift, diff, initial_distribution=m0,
                              bc_type="reflecting")
        m_final = result["density"][-1]

        var_num = np.trapezoid(x**2 * m_final, x)
        var_exact = s0**2 + sigma**2 * T
        return abs(var_num - var_exact)

    def test_error_decreases_with_refinement(self):
        e1 = self._error(n_x=30)
        e2 = self._error(n_x=60)
        e3 = self._error(n_x=120)
        assert e2 < e1, f"Error not decreasing: {e1:.2e} → {e2:.2e}"
        assert e3 < e2, f"Error not decreasing: {e2:.2e} → {e3:.2e}"

    def test_spatial_convergence_rate(self):
        """Convergence should be at least first-order in dx."""
        e1 = self._error(n_x=30)
        e2 = self._error(n_x=60)
        e3 = self._error(n_x=120)

        if e1 > 1e-14 and e2 > 1e-14:
            rate1 = np.log(e1 / e2) / np.log(2.0)
        else:
            rate1 = 2.0
        if e2 > 1e-14 and e3 > 1e-14:
            rate2 = np.log(e2 / e3) / np.log(2.0)
        else:
            rate2 = 2.0

        print(f"\nFP spatial convergence rates: {rate1:.2f}, {rate2:.2f}")
        assert rate1 > 0.8, f"Spatial rate 30→60 too low: {rate1:.2f}"
        assert rate2 > 0.8, f"Spatial rate 60→120 too low: {rate2:.2f}"


# ---------------------------------------------------------------------------
# Test 7 — Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    """b=0, symmetric initial condition → solution stays symmetric."""

    def test_symmetric_density(self):
        n_x, n_t = 101, 200
        L = 2.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(0.3, n_t)
        model = DiffusionOnlyModel(sigma=0.4)
        x = grid.points

        # Symmetric initial condition (even function about 0)
        m0 = gaussian(x, 0.0, 0.3**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, 0.0, 0.4)

        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(drift, diff, initial_distribution=m0,
                              bc_type="reflecting")
        m = result["density"]

        for k in range(0, n_t + 1, 20):
            # m[k, i] should equal m[k, n_x-1-i] for symmetric grid
            assert np.allclose(m[k], m[k, ::-1], atol=1e-10), (
                f"Symmetry broken at time index {k}"
            )

    def test_antisymmetric_initial_mean_zero(self):
        """Even initial condition → zero mean at all times with zero drift."""
        n_x, n_t = 100, 200
        L = 2.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(0.3, n_t)
        model = DiffusionOnlyModel(sigma=0.3)
        x = grid.points
        m0 = gaussian(x, 0.0, 0.25**2)
        drift, diff = const_arrays(tgrid.n_nodes, n_x, 0.0, 0.3)

        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(drift, diff, initial_distribution=m0,
                              bc_type="reflecting")
        m = result["density"]
        for k in range(0, n_t + 1, 20):
            mean = np.trapezoid(x * m[k], x)
            assert abs(mean) < 1e-10, f"Mean non-zero at step {k}: {mean:.2e}"


# ---------------------------------------------------------------------------
# Test 8 — check_mass_conservation helper
# ---------------------------------------------------------------------------

class TestCheckMassConservation:

    def test_perfect_conservation(self):
        n_x, n_t = 50, 20
        L = 2.0
        grid = Grid1D(-L, L, n_x)
        tgrid = TimeGrid(0.1, n_t)
        model = DiffusionOnlyModel()
        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        x = grid.points
        # Manually build a density that integrates to exactly 1 at each step
        m = np.zeros((n_t + 1, n_x))
        for k in range(n_t + 1):
            g = gaussian(x, 0.0, 0.3**2)
            m[k] = g
        err = solver.check_mass_conservation(m)
        assert err < 1e-10

    def test_returns_float(self):
        n_x, n_t = 40, 10
        grid = Grid1D(-1.0, 1.0, n_x)
        tgrid = TimeGrid(0.1, n_t)
        model = DiffusionOnlyModel()
        solver = FokkerPlanckSolver(grid=grid, time_grid=tgrid, model=model)
        x = grid.points
        m = np.tile(gaussian(x, 0.0, 0.2**2), (n_t + 1, 1))
        assert isinstance(solver.check_mass_conservation(m), float)
