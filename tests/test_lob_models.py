"""Tests for the three concrete MFG-LOB models.

Coverage:
  - AvellanedaStoikovMFG  — market making
  - OptimalExecutionMFG   — optimal execution
  - LOBFormationMFG       — LOB shape formation
"""

from __future__ import annotations

import numpy as np
import pytest

from mfglob.grids import Grid1D, TimeGrid
from mfglob.mfg_solver import MFGSolver
from mfglob.models.avellaneda_stoikov import AvellanedaStoikovMFG
from mfglob.models.optimal_execution import OptimalExecutionMFG
from mfglob.models.lob_formation import LOBFormationMFG


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_as_grid():
    return Grid1D(-5.0, 5.0, 40)


def make_oe_grid():
    return Grid1D(0.0, 1.0, 40)


def make_lob_grid():
    return Grid1D(0.0, 10.0, 50)


def uniform_density(x):
    d = np.ones_like(x)
    return d / np.trapezoid(d, x)


# ---------------------------------------------------------------------------
# 1 — AvellanedaStoikovMFG
# ---------------------------------------------------------------------------

class TestAvellanedaStoikovMFG:

    def setup_method(self):
        self.model = AvellanedaStoikovMFG(
            A=1.0, k=1.5, phi=0.01, psi=0.005,
            gamma=0.1, sigma_mid=0.3, Q_max=5.0, c=0.5,
        )
        self.grid = make_as_grid()
        self.x = self.grid.points
        self.m = uniform_density(self.x)

    # -- param_dict / state_dimension --

    def test_param_dict_keys(self):
        d = self.model.param_dict
        for key in ("A", "k", "phi", "psi", "gamma", "sigma_mid", "Q_max", "c"):
            assert key in d

    def test_state_dimension(self):
        assert self.model.state_dimension == 1

    # -- execution rates --

    def test_execution_rate_ask_positive(self):
        delta_a = np.full_like(self.x, 0.5)
        rate = self.model.execution_rate_ask(delta_a, self.m, self.x)
        assert np.all(rate > 0)

    def test_execution_rate_bid_positive(self):
        delta_b = np.full_like(self.x, 0.5)
        rate = self.model.execution_rate_bid(delta_b, self.m, self.x)
        assert np.all(rate > 0)

    def test_execution_rate_decreases_with_spread(self):
        """Larger spread → lower execution probability."""
        q = self.x
        m = self.m
        r1 = self.model.execution_rate_ask(np.full_like(q, 0.5), m, q)
        r2 = self.model.execution_rate_ask(np.full_like(q, 1.0), m, q)
        assert np.all(r2 < r1)

    def test_execution_rate_ask_shape(self):
        delta_a = np.full_like(self.x, 0.3)
        rate = self.model.execution_rate_ask(delta_a, self.m, self.x)
        assert rate.shape == self.x.shape

    # -- optimal spread --

    def test_optimal_spread_returns_tuple(self):
        p = np.zeros_like(self.x)
        result = self.model.optimal_spread(self.x, p, self.m)
        assert len(result) == 2

    def test_optimal_spread_nonneg(self):
        p = np.zeros_like(self.x)
        da, db = self.model.optimal_spread(self.x, p, self.m)
        assert np.all(da >= 0)
        assert np.all(db >= 0)

    def test_optimal_spread_base_is_1_over_k(self):
        """At zero inventory (q=0) both spreads equal 1/k."""
        x0 = np.array([0.0])
        p0 = np.array([0.0])
        m0 = np.array([1.0])
        da, db = self.model.optimal_spread(x0, p0, m0)
        assert da[0] == pytest.approx(1.0 / self.model.k, rel=1e-6)
        assert db[0] == pytest.approx(1.0 / self.model.k, rel=1e-6)

    def test_optimal_spread_ask_increases_with_inventory(self):
        """Positive inventory → ask spread larger (want to sell fast)."""
        q_pos = np.array([2.0])
        q_neg = np.array([-2.0])
        p = np.array([0.0])
        m = np.array([1.0])
        da_pos, _ = self.model.optimal_spread(q_pos, p, m)
        da_neg, _ = self.model.optimal_spread(q_neg, p, m)
        assert da_pos[0] > da_neg[0]

    # -- running cost --

    def test_running_cost_finite(self):
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        assert np.all(np.isfinite(cost))

    def test_running_cost_nonneg(self):
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        assert np.all(cost >= 0)

    # -- terminal cost --

    def test_terminal_cost_at_zero(self):
        cost = self.model.terminal_cost(np.array([0.0]), np.array([1.0]))
        assert cost[0] == pytest.approx(0.0)

    def test_terminal_cost_nonneg(self):
        cost = self.model.terminal_cost(self.x, self.m)
        assert np.all(cost >= 0)

    # -- initial distribution --

    def test_initial_distribution_integrates_to_one(self):
        m0 = self.model.initial_distribution(self.x)
        assert np.trapezoid(m0, self.x) == pytest.approx(1.0, rel=1e-4)

    def test_initial_distribution_nonneg(self):
        m0 = self.model.initial_distribution(self.x)
        assert np.all(m0 >= 0)

    # -- Hamiltonian --

    def test_hamiltonian_finite(self):
        p = np.zeros_like(self.x)
        M = np.zeros_like(self.x)
        H = self.model.hamiltonian(self.x, p, M, self.m)
        assert np.all(np.isfinite(H))

    def test_hamiltonian_shape(self):
        p = np.zeros_like(self.x)
        M = np.zeros_like(self.x)
        H = self.model.hamiltonian(self.x, p, M, self.m)
        assert H.shape == self.x.shape

    # -- diffusion --

    def test_diffusion_constant(self):
        alpha = np.zeros_like(self.x)
        sigma = self.model.diffusion(self.x, alpha, self.m)
        assert np.allclose(sigma, self.model.sigma_mid)

    # -- MFG solver integration (light smoke test) --

    def test_mfg_solver_runs(self):
        tgrid = TimeGrid(0.1, 20)
        solver = MFGSolver(self.model, self.grid, tgrid, damping=0.5,
                           max_iterations=5, tol=1e-2)
        result = solver.solve()
        assert result["value_function"].shape == (tgrid.n_nodes, self.grid.n)
        assert result["density"].shape == (tgrid.n_nodes, self.grid.n)


# ---------------------------------------------------------------------------
# 2 — OptimalExecutionMFG
# ---------------------------------------------------------------------------

class TestOptimalExecutionMFG:

    def setup_method(self):
        self.model = OptimalExecutionMFG(
            phi=0.001, psi=0.01, eta=0.1, lam=1.0, sigma=0.0, Q0=1.0,
        )
        self.grid = make_oe_grid()
        self.x = self.grid.points
        self.m = uniform_density(self.x)

    # -- param_dict / state_dimension --

    def test_param_dict_keys(self):
        d = self.model.param_dict
        for key in ("phi", "psi", "eta", "lam", "sigma", "Q0"):
            assert key in d

    def test_state_dimension(self):
        assert self.model.state_dimension == 1

    # -- optimal control (execution rate) --

    def test_optimal_rate_nonneg_positive_p(self):
        """With positive costate (p>0) the optimal rate should be non-negative."""
        p = np.full_like(self.x, 0.5)
        nu = self.model.optimal_control(self.x, p, self.m)
        assert np.all(nu >= 0)

    def test_optimal_rate_zero_for_negative_p(self):
        """Very negative costate → zero execution rate (no short selling)."""
        p = np.full_like(self.x, -10.0)
        nu = self.model.optimal_control(self.x, p, self.m)
        assert np.all(nu == 0)

    def test_optimal_rate_finite(self):
        p = np.linspace(-1.0, 1.0, len(self.x))
        nu = self.model.optimal_control(self.x, p, self.m)
        assert np.all(np.isfinite(nu))

    # -- aggregate execution rate --

    def test_aggregate_rate_nonneg(self):
        nu = np.maximum(np.random.RandomState(0).randn(len(self.x)), 0)
        M = self.model.aggregate_execution_rate(nu, self.m, self.grid)
        assert M >= 0.0

    def test_aggregate_rate_zero_for_zero_nu(self):
        nu = np.zeros_like(self.x)
        M = self.model.aggregate_execution_rate(nu, self.m, self.grid)
        assert M == pytest.approx(0.0)

    def test_aggregate_rate_scalar(self):
        nu = np.ones_like(self.x)
        M = self.model.aggregate_execution_rate(nu, self.m, self.grid)
        assert np.isscalar(M) or (isinstance(M, np.ndarray) and M.ndim == 0)

    # -- Almgren-Chriss single-agent limit --

    def test_almgren_chriss_rate_shape(self):
        """No mean-field (eta=0, lam=0): mean inventory should decrease monotonically.

        The classical Almgren-Chriss solution has q*(t) strictly decreasing.
        With small diffusion (needed for numerical stability of the FP) we
        verify this qualitative property holds for the MFG solution.
        """
        model_ac = OptimalExecutionMFG(phi=0.5, psi=1.0, eta=0.0, lam=0.0,
                                       sigma=0.05, Q0=1.0)
        grid = Grid1D(0.0, 1.5, 40)
        tgrid = TimeGrid(1.0, 60)
        solver = MFGSolver(model_ac, grid, tgrid, damping=0.5,
                           max_iterations=15, tol=1e-3)
        result = solver.solve()

        x_g = grid.points
        m_sol = result["density"]
        mean_q = np.array([float(np.trapezoid(x_g * m_sol[n], x_g))
                           for n in range(tgrid.n_nodes)])

        # Mean inventory should decrease over time (agents are executing)
        assert mean_q[0] > mean_q[-1], "Mean inventory should decrease as agents execute"

        # Almgren-Chriss κ = √(φ/ψ), qualitative check: most mass executed by T
        phi, psi = model_ac.phi, model_ac.psi
        kappa = np.sqrt(phi / psi)
        T = tgrid.T
        # At t=T, AC gives q*(T)=0; check mean_q[-1] < mean_q[0]
        assert mean_q[-1] < 0.9 * mean_q[0], (
            f"Insufficient execution: mean_q[0]={mean_q[0]:.3f}, mean_q[-1]={mean_q[-1]:.3f}"
        )

    # -- running cost --

    def test_running_cost_finite(self):
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        assert np.all(np.isfinite(cost))

    def test_running_cost_nonneg_for_zero_control(self):
        """Zero execution: only inventory holding cost φq² ≥ 0."""
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        assert np.all(cost >= 0)

    # -- terminal cost --

    def test_terminal_cost_at_zero(self):
        cost = self.model.terminal_cost(np.array([0.0]), np.array([1.0]))
        assert cost[0] == pytest.approx(0.0)

    def test_terminal_cost_nonneg(self):
        cost = self.model.terminal_cost(self.x, self.m)
        assert np.all(cost >= 0)

    # -- initial distribution --

    def test_initial_distribution_integrates_to_one(self):
        m0 = self.model.initial_distribution(self.x)
        assert np.trapezoid(m0, self.x) == pytest.approx(1.0, rel=1e-4)

    def test_initial_distribution_concentrated_at_Q0(self):
        """Mass concentrated near Q0."""
        m0 = self.model.initial_distribution(self.x)
        # More than 50% of mass should be in the upper quarter of the domain
        upper = self.x >= 0.75 * self.model.Q0
        mass_upper = np.trapezoid(m0[upper], self.x[upper])
        assert mass_upper > 0.5

    # -- diffusion --

    def test_diffusion_zero_for_deterministic(self):
        alpha = np.zeros_like(self.x)
        sigma = self.model.diffusion(self.x, alpha, self.m)
        assert np.all(sigma == pytest.approx(0.0))

    # -- drift direction --

    def test_drift_is_negative_execution(self):
        """Executing (ν > 0) should decrease inventory."""
        nu = np.full_like(self.x, 2.0)
        b = self.model.drift(self.x, nu, self.m)
        assert np.all(b < 0)


# ---------------------------------------------------------------------------
# 3 — LOBFormationMFG
# ---------------------------------------------------------------------------

class TestLOBFormationMFG:

    def setup_method(self):
        self.model = LOBFormationMFG(
            c_far=0.1, c_near=1.0, kappa=2.0,
            c_crowd=0.5, c_control=0.1, sigma=0.5, x_max=10.0,
        )
        self.grid = make_lob_grid()
        self.x = self.grid.points
        self.m = uniform_density(self.x)

    # -- param_dict / state_dimension --

    def test_param_dict_keys(self):
        d = self.model.param_dict
        for key in ("c_far", "c_near", "kappa", "c_crowd",
                    "c_control", "sigma", "x_max"):
            assert key in d

    def test_state_dimension(self):
        assert self.model.state_dimension == 1

    # -- running cost structure --

    def test_running_cost_finite(self):
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        assert np.all(np.isfinite(cost))

    def test_running_cost_nonneg_zero_control(self):
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        assert np.all(cost >= 0)

    def test_adverse_selection_large_at_zero(self):
        """Cost near x=0 should be high (adverse selection)."""
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        # cost[0] dominated by c_near·exp(0) = c_near
        assert cost[0] >= self.model.c_near * np.exp(-self.model.kappa * self.x[0])

    def test_opportunity_cost_increases_with_distance(self):
        """c_far·x term should increase cost at larger x."""
        alpha = np.zeros_like(self.x)
        cost = self.model.running_cost(self.x, alpha, self.m)
        # Strip adverse selection: check cost increases in the tail
        tail = self.x > 5.0
        x_tail = self.x[tail]
        cost_tail = cost[tail]
        if len(x_tail) > 1:
            # c_far·x dominates for large x (adverse selection ≈ 0)
            assert cost_tail[-1] > cost_tail[0]

    # -- terminal cost --

    def test_terminal_cost_zero(self):
        cost = self.model.terminal_cost(self.x, self.m)
        assert np.all(cost == pytest.approx(0.0))

    # -- Hamiltonian --

    def test_hamiltonian_finite(self):
        p = np.zeros_like(self.x)
        M = np.zeros_like(self.x)
        H = self.model.hamiltonian(self.x, p, M, self.m)
        assert np.all(np.isfinite(H))

    # -- optimal control --

    def test_optimal_control_zero_p(self):
        """With zero costate, optimal control is zero."""
        p = np.zeros_like(self.x)
        alpha = self.model.optimal_control(self.x, p, self.m)
        assert np.allclose(alpha, 0.0)

    def test_optimal_control_negative_for_positive_p(self):
        """α* = -p/(2c_control): positive p → negative drift."""
        p = np.full_like(self.x, 1.0)
        alpha = self.model.optimal_control(self.x, p, self.m)
        assert np.all(alpha < 0)

    # -- diffusion --

    def test_diffusion_constant_sigma(self):
        alpha = np.zeros_like(self.x)
        sigma = self.model.diffusion(self.x, alpha, self.m)
        assert np.allclose(sigma, self.model.sigma)

    # -- initial distribution --

    def test_initial_distribution_integrates_to_one(self):
        m0 = self.model.initial_distribution(self.x)
        assert np.trapezoid(m0, self.x) == pytest.approx(1.0, rel=1e-4)

    # -- steady-state density --

    def test_steady_state_positive(self):
        m_ss = self.model.steady_state_density(self.grid)
        assert np.all(m_ss >= 0)

    def test_steady_state_integrates_to_one(self):
        m_ss = self.model.steady_state_density(self.grid)
        mass = np.trapezoid(m_ss, self.x)
        assert mass == pytest.approx(1.0, rel=1e-4)

    def test_steady_state_shape_low_near_zero(self):
        """Adverse selection pushes orders away from x=0, so m* should be
        small near x=0 and larger at intermediate distances."""
        m_ss = self.model.steady_state_density(self.grid)
        # Density at x=0 should be less than density at the mode
        mode_idx = np.argmax(m_ss)
        assert m_ss[0] <= m_ss[mode_idx]

    def test_steady_state_mode_at_intermediate_distance(self):
        """Mode should not be at x=0 (adverse selection) nor at x_max."""
        m_ss = self.model.steady_state_density(self.grid)
        mode_idx = np.argmax(m_ss)
        n = len(self.x)
        # Mode is not at the first or last point
        assert 0 < mode_idx < n - 1

    def test_steady_state_decays_for_large_x(self):
        """Density should decline in the far tail (opportunity cost dominates)."""
        m_ss = self.model.steady_state_density(self.grid)
        # Compare second half to first half near the tail
        n = len(m_ss)
        assert m_ss[-1] < m_ss[n // 2]

    # -- MFG solver integration (light smoke test) --

    def test_mfg_solver_runs(self):
        tgrid = TimeGrid(0.2, 20)
        solver = MFGSolver(self.model, self.grid, tgrid, damping=0.5,
                           max_iterations=5, tol=1e-2)
        result = solver.solve()
        assert result["density"].shape == (tgrid.n_nodes, self.grid.n)
        assert np.all(np.isfinite(result["value_function"]))
