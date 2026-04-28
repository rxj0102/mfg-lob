"""Tests for HJB, FP, and MFG solvers."""
import numpy as np
import pytest

from mfg_lob.solvers.grid import Grid
from mfg_lob.solvers.hjb_solver import HJBSolver
from mfg_lob.solvers.fp_solver import FPSolver
from mfg_lob.solvers.mfg_solver import MFGSolver
from mfg_lob.models.lob_model import LOBModel


# Small grid for fast tests
GRID = Grid.uniform(q_min=-5, q_max=5, nq=31, t_min=0, t_max=0.5, nt=51)


def make_model(**kwargs) -> LOBModel:
    defaults = dict(sigma=0.5, phi=0.1, alpha=0.5, A_term=1.0)
    defaults.update(kwargs)
    return LOBModel(**defaults)


class TestHJBSolver:
    def test_output_shape(self):
        model = make_model()
        g = GRID
        m = np.outer(model.initial_density(g.q), np.ones(g.nt))
        solver = HJBSolver(
            grid=g,
            sigma=model.sigma,
            hamiltonian=model.hamiltonian,
            running_cost=model.running_cost,
            terminal_cost=model.terminal_cost,
        )
        u = solver.solve(m)
        assert u.shape == (g.nq, g.nt)

    def test_terminal_condition(self):
        model = make_model()
        g = GRID
        m = np.outer(model.initial_density(g.q), np.ones(g.nt))
        solver = HJBSolver(
            grid=g,
            sigma=model.sigma,
            hamiltonian=model.hamiltonian,
            running_cost=model.running_cost,
            terminal_cost=model.terminal_cost,
        )
        u = solver.solve(m)
        expected = model.terminal_cost(g.q)
        np.testing.assert_allclose(u[:, -1], expected, rtol=1e-10)

    def test_finite_values(self):
        model = make_model()
        g = GRID
        m = np.outer(model.initial_density(g.q), np.ones(g.nt))
        solver = HJBSolver(
            grid=g,
            sigma=model.sigma,
            hamiltonian=model.hamiltonian,
            running_cost=model.running_cost,
            terminal_cost=model.terminal_cost,
        )
        u = solver.solve(m)
        assert np.all(np.isfinite(u))


class TestFPSolver:
    def test_output_shape(self):
        model = make_model()
        g = GRID
        m_init_2d = np.outer(model.initial_density(g.q), np.ones(g.nt))
        hjb = HJBSolver(
            grid=g, sigma=model.sigma, hamiltonian=model.hamiltonian,
            running_cost=model.running_cost, terminal_cost=model.terminal_cost,
        )
        u = hjb.solve(m_init_2d)
        fp = FPSolver(
            grid=g, sigma=model.sigma, optimal_drift=model.optimal_drift,
            initial_density=model.initial_density,
        )
        m = fp.solve(u)
        assert m.shape == (g.nq, g.nt)

    def test_initial_condition(self):
        model = make_model()
        g = GRID
        m_init_2d = np.outer(model.initial_density(g.q), np.ones(g.nt))
        hjb = HJBSolver(
            grid=g, sigma=model.sigma, hamiltonian=model.hamiltonian,
            running_cost=model.running_cost, terminal_cost=model.terminal_cost,
        )
        u = hjb.solve(m_init_2d)
        fp = FPSolver(
            grid=g, sigma=model.sigma, optimal_drift=model.optimal_drift,
            initial_density=model.initial_density,
        )
        m = fp.solve(u)
        # Initial density should be normalised
        assert np.trapezoid(m[:, 0], g.q) == pytest.approx(1.0, rel=1e-3)

    def test_mass_conserved(self):
        model = make_model()
        g = GRID
        m_init_2d = np.outer(model.initial_density(g.q), np.ones(g.nt))
        hjb = HJBSolver(
            grid=g, sigma=model.sigma, hamiltonian=model.hamiltonian,
            running_cost=model.running_cost, terminal_cost=model.terminal_cost,
        )
        u = hjb.solve(m_init_2d)
        fp = FPSolver(
            grid=g, sigma=model.sigma, optimal_drift=model.optimal_drift,
            initial_density=model.initial_density,
        )
        m = fp.solve(u)
        masses = np.array([np.trapezoid(m[:, k], g.q) for k in range(g.nt)])
        np.testing.assert_allclose(masses, 1.0, atol=1e-6)

    def test_nonnegative(self):
        model = make_model()
        g = GRID
        m_init_2d = np.outer(model.initial_density(g.q), np.ones(g.nt))
        hjb = HJBSolver(
            grid=g, sigma=model.sigma, hamiltonian=model.hamiltonian,
            running_cost=model.running_cost, terminal_cost=model.terminal_cost,
        )
        u = hjb.solve(m_init_2d)
        fp = FPSolver(
            grid=g, sigma=model.sigma, optimal_drift=model.optimal_drift,
            initial_density=model.initial_density,
        )
        m = fp.solve(u)
        assert np.all(m >= -1e-12)


class TestMFGSolver:
    def test_returns_equilibrium(self):
        model = make_model()
        solver = MFGSolver(model, GRID, max_iter=5, tol=1e-3)
        eq = solver.solve()
        assert eq.u.shape == (GRID.nq, GRID.nt)
        assert eq.m.shape == (GRID.nq, GRID.nt)

    def test_residuals_decreasing_trend(self):
        model = make_model()
        solver = MFGSolver(model, GRID, max_iter=10, tol=1e-8, damping=0.5)
        eq = solver.solve()
        # Residuals should generally decrease (not necessarily monotone with damping)
        assert len(eq.residuals) > 0
        assert eq.residuals[0] >= eq.residuals[-1] or not eq.converged

    def test_trend_follower_model(self):
        model = make_model(agent_type="trend_follower", epsilon=1.0)
        solver = MFGSolver(model, GRID, max_iter=5, tol=1e-3)
        eq = solver.solve()
        assert np.all(np.isfinite(eq.u))
        assert np.all(np.isfinite(eq.m))

    def test_equilibrium_interpolation(self):
        model = make_model()
        solver = MFGSolver(model, GRID, max_iter=3, tol=1e-2)
        eq = solver.solve()
        q_test = np.array([0.0, 1.0, -1.0])
        u_t = eq.value_function(q_test, t=0.0)
        assert u_t.shape == (3,)
        assert np.all(np.isfinite(u_t))
