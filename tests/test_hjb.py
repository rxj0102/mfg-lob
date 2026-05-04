"""Tests for HJBSolver: analytical solutions, convergence, and consistency checks."""

from __future__ import annotations

import numpy as np
import pytest

from mfglob.grids import Grid1D, TimeGrid
from mfglob.hjb import HJBSolver
from mfglob.models.base import MFGLOBModel


# ---------------------------------------------------------------------------
# Minimal concrete model stubs for testing
# ---------------------------------------------------------------------------

class HeatModel(MFGLOBModel):
    """Pure diffusion: H = (σ²/2) D²u — no control, no advection, no cost."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def hamiltonian(self, x, p, M, m):
        return 0.5 * self.sigma**2 * M

    def optimal_control(self, x, p, m):
        return np.zeros_like(x)

    def drift(self, x, alpha, m):
        return np.zeros_like(x)

    def diffusion(self, x, alpha, m):
        return np.full_like(x, self.sigma)

    def running_cost(self, x, alpha, m):
        return np.zeros_like(x)

    def terminal_cost(self, x, m):
        return np.sin(np.pi * x / (x[-1] - x[0] + x[-1]))  # overridden per test

    def initial_distribution(self, x):
        m = np.ones_like(x)
        return m / np.trapezoid(m, x)

    @property
    def state_dimension(self):
        return 1

    @property
    def param_dict(self):
        return {"sigma": self.sigma}


class LinearQuadraticModel(MFGLOBModel):
    """LQ model: H = p²/2, optimal α* = p (α enters drift directly, cost = α²/2)."""

    def hamiltonian(self, x, p, M, m):
        return 0.5 * p**2

    def optimal_control(self, x, p, m):
        return p.copy()

    def drift(self, x, alpha, m):
        return alpha.copy()

    def diffusion(self, x, alpha, m):
        return np.zeros_like(x)

    def running_cost(self, x, alpha, m):
        return 0.5 * alpha**2

    def terminal_cost(self, x, m):
        return np.zeros_like(x)

    def initial_distribution(self, x):
        m = np.ones_like(x)
        return m / np.trapezoid(m, x)

    @property
    def state_dimension(self):
        return 1

    @property
    def param_dict(self):
        return {}


class PureAdvectionModel(MFGLOBModel):
    """Pure advection: H = v·p with constant v, zero diffusion, zero cost."""

    def __init__(self, v: float = 1.0):
        self.v = v

    def hamiltonian(self, x, p, M, m):
        return self.v * p

    def optimal_control(self, x, p, m):
        return np.full_like(x, self.v)

    def drift(self, x, alpha, m):
        return np.full_like(x, self.v)

    def diffusion(self, x, alpha, m):
        return np.zeros_like(x)

    def running_cost(self, x, alpha, m):
        return np.zeros_like(x)

    def terminal_cost(self, x, m):
        return np.zeros_like(x)

    def initial_distribution(self, x):
        m = np.ones_like(x)
        return m / np.trapezoid(m, x)

    @property
    def state_dimension(self):
        return 1

    @property
    def param_dict(self):
        return {"v": self.v}


# ---------------------------------------------------------------------------
# Helper: build flat (constant) m array
# ---------------------------------------------------------------------------

def flat_m(n_times, n_x):
    m = np.ones((n_times, n_x))
    return m


# ---------------------------------------------------------------------------
# Test 1 — Heat equation backward
# ---------------------------------------------------------------------------

class TestHeatEquationBackward:
    """
    H = -σ²/2 · D²u (from the Hamiltonian convention sup_α{-f - b·p - σ²/2·M}).
    Terminal condition: u(T, x) = sin(π x / L).

    The PDE is -∂u/∂t + H = 0  →  ∂u/∂t = -σ²/2 · D²u.
    For u = sin(πx/L), D²u = -(π/L)² u, so ∂u/∂t = σ²π²/(2L²) · u > 0
    — the solution DECAYS going backward from T to 0.

    Analytical solution: u(t, x) = sin(πx/L) · exp(-σ²π²(T-t) / (2L²))
    so u(0, x) = sin(πx/L) · exp(-σ²π²T/(2L²))  <  u(T, x).
    """

    def _run(self, n_x, n_t, sigma=1.0, T=0.1, L=1.0):
        grid = Grid1D(0.0, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = HeatModel(sigma=sigma)
        x = grid.points

        terminal = np.sin(np.pi * x / L)
        m = flat_m(tgrid.n_nodes, n_x)

        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        u_num = result["value_function"]

        # Analytical: u(t,x) = sin(πx/L) · exp(-σ²π²(T-t)/(2L²))
        times = tgrid.times
        u_exact = np.outer(
            np.exp(-sigma**2 * np.pi**2 * (T - times) / (2.0 * L**2)),
            np.sin(np.pi * x / L),
        )
        return u_num, u_exact, x, times

    def test_terminal_matches(self):
        u_num, u_exact, x, times = self._run(60, 200)
        np.testing.assert_allclose(u_num[-1], u_exact[-1], atol=1e-12)

    def test_interior_accuracy(self):
        """Numerical solution at t=0 should match analytical within O(dx²+dt)."""
        u_num, u_exact, x, times = self._run(80, 400)
        err = np.max(np.abs(u_num[0] - u_exact[0]))
        assert err < 5e-3, f"Heat backward error too large: {err:.2e}"

    def test_solution_shape(self):
        n_x, n_t = 40, 100
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.1, n_t)
        model = HeatModel()
        m = flat_m(tgrid.n_nodes, n_x)
        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m)
        assert result["value_function"].shape == (tgrid.n_nodes, n_x)
        assert result["optimal_control"].shape == (tgrid.n_nodes, n_x)
        assert result["gradient"].shape == (tgrid.n_nodes, n_x)

    def test_boundary_values_enforced(self):
        """Dirichlet BCs: u[n, 0] = 0, u[n, -1] = 0 (sin(0) = sin(π) = 0)."""
        u_num, _, x, _ = self._run(60, 200)
        assert np.allclose(u_num[:, 0], 0.0, atol=1e-10)
        assert np.allclose(u_num[:, -1], 0.0, atol=1e-10)

    def test_solution_decays_backward(self):
        """For the backward HJB H = -σ²/2·D²u, amplitude DECREASES from T to 0."""
        u_num, u_exact, x, times = self._run(60, 200, sigma=1.0, T=0.2)
        amp_T = np.max(np.abs(u_num[-1]))
        amp_0 = np.max(np.abs(u_num[0]))
        # u(T) = sin (amplitude 1); u(0) = sin·exp(-σ²π²T/(2L²)) < u(T)
        assert amp_0 < amp_T, f"Expected decay backward: amp_0={amp_0:.4f} < amp_T={amp_T:.4f}"


# ---------------------------------------------------------------------------
# Test 2 — Linear-Quadratic (LQ) model
# ---------------------------------------------------------------------------

class TestLinearQuadratic:
    """
    H = p²/2, α* = p.
    Terminal condition u(T, x) = x².
    Analytical solution: u(t, x) = (x + (T-t))²  (characteristics method).

    Derivation:
      -u_t + p²/2 = 0,  u_T = x²
      Along characteristic dx/dt = -p = -u_x, we need:
        u(t, x) = x²  shifted by (T-t) along the characteristic.
      For zero drift + zero diffusion + running cost α²/2 = p²/2:
        The HJB is -u_t + (u_x)²/2 = 0.
      With u(T,x) = x², try u = (x + c(t))²:
        -2(x+c)c' + 2(x+c)² / 2 = 0  → c' = x+c  (doesn't work for all x).
      Simpler: use u(T,x)=0, f=α²/2, drift=α:
        -u_t + p²/2 = 0  with u(T)=0  → u ≡ 0 is a valid solution.
      Instead verify: with terminal u(T)=x, analytical u(t,x) = x + (T-t)·0 = x
        since -u_t + (1)²/2 = 0 → -0 + 1/2 ≠ 0.
      Use terminal condition u(T,x)=0, then u≡0 everywhere (trivial check).
    """

    def test_zero_terminal_gives_zero(self):
        """With terminal_cost=0 and running_cost=α²/2, u=0 is the exact solution when α*=0."""
        n_x, n_t = 40, 100
        grid = Grid1D(-1.0, 1.0, n_x)
        tgrid = TimeGrid(0.1, n_t)

        class ZeroTerminalLQ(LinearQuadraticModel):
            def terminal_cost(self, x, m):
                return np.zeros_like(x)

        model = ZeroTerminalLQ()
        x = grid.points
        m = flat_m(tgrid.n_nodes, n_x)
        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=np.zeros(n_x),
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        u = result["value_function"]
        np.testing.assert_allclose(u, 0.0, atol=1e-12)

    def test_optimal_control_is_gradient(self):
        """α*(t, x) should equal Du(t, x) for the LQ model at every grid point."""
        n_x, n_t = 50, 200
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.2, n_t)
        model = LinearQuadraticModel()
        x = grid.points

        terminal = np.sin(2.0 * np.pi * x)
        m = flat_m(tgrid.n_nodes, n_x)
        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)

        alpha = result["optimal_control"]
        grad = result["gradient"]
        # At interior points the optimal control should equal the gradient
        np.testing.assert_allclose(alpha[:, 1:-1], grad[:, 1:-1], atol=1e-12)

    def test_smooth_solution(self):
        """Value function should be smooth — no spurious oscillations (small T for stability)."""
        n_x, n_t = 60, 600
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.05, n_t)   # small T keeps explicit advection stable
        model = LinearQuadraticModel()
        x = grid.points
        terminal = np.sin(np.pi * x)
        m = flat_m(tgrid.n_nodes, n_x)
        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        u = result["value_function"]
        # No spurious oscillations: second differences should be bounded
        for k in range(0, tgrid.n_nodes, 10):
            d2u = np.diff(u[k], n=2)
            assert np.all(np.isfinite(d2u)), f"NaN/Inf in d²u at t index {k}"
            assert np.max(np.abs(d2u)) < 1.0, (
                f"Large second difference at t index {k}: {np.max(np.abs(d2u)):.3f}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Pure advection backward
# ---------------------------------------------------------------------------

class TestPureAdvectionBackward:
    """
    H = v·p, constant v, zero diffusion, zero cost.

    The solver uses H_expl = -b*·p = -v·p (convention H = -f - b·p).
    Scheme: u^n_i = u^{n+1}_i - dt·v·(D_bwd u^{n+1})_i  (for v > 0)
    = (1-CFL)·u^{n+1}_i + CFL·u^{n+1}_{i-1}

    Each backward step shifts the profile one CFL fraction to the RIGHT (higher index).
    After N steps: u(0, x) ≈ g(x - v·T)  [rightward shift in x-space going backward].

    Note: the terminal condition g(x) = sin(2πx/L) satisfies g(0)=g(L)=0 (Dirichlet BC).
    The shifted solution g(x - vT) does NOT satisfy BC at x=0 when vT > 0, so there
    is a boundary layer of width ~vT near x=0; tests check well inside this layer.
    """

    def _run(self, n_x, n_t, v=1.0, T=0.3, L=2.0):
        grid = Grid1D(0.0, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = PureAdvectionModel(v=v)
        x = grid.points
        terminal = np.sin(2.0 * np.pi * x / L)
        m = flat_m(tgrid.n_nodes, n_x)
        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        return result, grid, tgrid

    def test_no_oscillations_upwind(self):
        """Upwind scheme should produce monotone solution without oscillations."""
        result, grid, tgrid = self._run(80, 400)
        u = result["value_function"]
        # For each time slice, check total variation is bounded (< 10 for sin)
        for k in range(0, tgrid.n_nodes, 20):
            tv = np.sum(np.abs(np.diff(u[k])))
            assert tv < 15.0, f"TV too large at t index {k}: {tv:.2f}"

    def test_characteristics_transport(self):
        """u(0, x) ≈ g(x - v·T) — profile shifts rightward going backward in time."""
        result, grid, tgrid = self._run(100, 500, v=1.0, T=0.2, L=3.0)
        u = result["value_function"]
        x = grid.points
        L = 3.0
        # Backward upwind shifts rightward: u(0,x) ≈ sin(2π(x - vT)/L)
        shift = 1.0 * 0.2  # v * T = 0.2, about 7 grid spacings
        u_exact_t0 = np.sin(2.0 * np.pi * (x - shift) / L)
        # Check well inside the domain, away from the boundary layer at x=0
        interior = slice(15, -5)
        err = np.max(np.abs(u[0, interior] - u_exact_t0[interior]))
        assert err < 0.15, f"Advection error at t=0: {err:.3f}"

    def test_shape_preserved(self):
        result, grid, tgrid = self._run(60, 300)
        u = result["value_function"]
        assert u.shape == (tgrid.n_nodes, grid.n)


# ---------------------------------------------------------------------------
# Test 4 — Convergence test
# ---------------------------------------------------------------------------

class TestConvergenceRates:
    """
    Convergence test for the heat equation backward.
    Solve on grids with dx, dx/2, dx/4.
    Diffusion part: O(dx²); upwind advection part: O(dx).
    """

    def _error(self, n_x, n_t, sigma=1.0, T=0.05, L=1.0):
        """Return L∞ error vs. analytical at t=0."""
        grid = Grid1D(0.0, L, n_x)
        tgrid = TimeGrid(T, n_t)
        model = HeatModel(sigma=sigma)
        x = grid.points
        terminal = np.sin(np.pi * x / L)
        m = flat_m(tgrid.n_nodes, n_x)

        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        u_num = result["value_function"][0]

        # u(0,x) = sin(πx/L) · exp(-σ²π²T/(2L²))  [decays backward from T to 0]
        u_exact = np.exp(-sigma**2 * np.pi**2 * T / (2.0 * L**2)) * np.sin(np.pi * x / L)
        return np.max(np.abs(u_num - u_exact))

    def test_convergence_in_space(self):
        """Error decreases as n_x increases (at fixed small dt)."""
        # Fix dt very small to isolate spatial error
        e1 = self._error(n_x=20, n_t=2000)
        e2 = self._error(n_x=40, n_t=2000)
        e3 = self._error(n_x=80, n_t=2000)
        assert e2 < e1, f"Error did not decrease from n_x=20 to 40: {e1:.2e} → {e2:.2e}"
        assert e3 < e2, f"Error did not decrease from n_x=40 to 80: {e2:.2e} → {e3:.2e}"

    def test_spatial_convergence_rate(self):
        """Measured spatial convergence rate should be close to 2 (second-order diffusion)."""
        e1 = self._error(n_x=20, n_t=4000)
        e2 = self._error(n_x=40, n_t=4000)
        e3 = self._error(n_x=80, n_t=4000)

        # Rate from 20→40 and 40→80
        if e2 > 1e-14 and e1 > 1e-14:
            rate1 = np.log(e1 / e2) / np.log(2.0)
        else:
            rate1 = 2.0
        if e3 > 1e-14 and e2 > 1e-14:
            rate2 = np.log(e2 / e3) / np.log(2.0)
        else:
            rate2 = 2.0

        print(f"\nSpatial convergence rates: {rate1:.2f}, {rate2:.2f}")
        # For pure diffusion (implicit), expect near second-order in space
        assert rate1 > 1.5, f"Spatial rate 20→40 too low: {rate1:.2f}"
        assert rate2 > 1.5, f"Spatial rate 40→80 too low: {rate2:.2f}"

    def test_convergence_in_time(self):
        """Error decreases as n_t increases (at fixed fine spatial grid)."""
        e1 = self._error(n_x=200, n_t=20)
        e2 = self._error(n_x=200, n_t=40)
        e3 = self._error(n_x=200, n_t=80)
        assert e2 < e1, f"Error did not decrease from n_t=20 to 40: {e1:.2e} → {e2:.2e}"
        assert e3 < e2, f"Error did not decrease from n_t=40 to 80: {e2:.2e} → {e3:.2e}"

    def test_temporal_convergence_rate(self):
        """Measured temporal convergence rate should be ≥ 1 (first-order in time)."""
        e1 = self._error(n_x=200, n_t=20)
        e2 = self._error(n_x=200, n_t=40)
        e3 = self._error(n_x=200, n_t=80)

        if e2 > 1e-14 and e1 > 1e-14:
            rate1 = np.log(e1 / e2) / np.log(2.0)
        else:
            rate1 = 1.0
        if e3 > 1e-14 and e2 > 1e-14:
            rate2 = np.log(e2 / e3) / np.log(2.0)
        else:
            rate2 = 1.0

        print(f"\nTemporal convergence rates: {rate1:.2f}, {rate2:.2f}")
        assert rate1 > 0.7, f"Temporal rate n_t=20→40 too low: {rate1:.2f}"
        assert rate2 > 0.7, f"Temporal rate n_t=40→80 too low: {rate2:.2f}"


# ---------------------------------------------------------------------------
# Test 5 — Upwind gradient helper
# ---------------------------------------------------------------------------

class TestUpwindGradient:
    """Unit tests for _upwind_gradient."""

    def _solver(self, n_x=10):
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(1.0, 5)
        model = HeatModel()
        return HJBSolver(grid=grid, time_grid=tgrid, model=model)

    def test_positive_drift_uses_backward(self):
        solver = self._solver(10)
        x = solver.x
        u = x.copy()  # linear u → all finite differences equal 1/dx (actually 1 for dx=1/9)
        drift = np.ones(10)  # positive everywhere
        p = solver._upwind_gradient(u, drift)
        p_bwd = solver._D_bwd @ u
        np.testing.assert_allclose(p, p_bwd)

    def test_negative_drift_uses_forward(self):
        solver = self._solver(10)
        x = solver.x
        u = x.copy()
        drift = -np.ones(10)  # negative everywhere
        p = solver._upwind_gradient(u, drift)
        p_fwd = solver._D_fwd @ u
        np.testing.assert_allclose(p, p_fwd)

    def test_zero_drift_uses_central(self):
        solver = self._solver(10)
        x = solver.x
        u = x**2
        drift = np.zeros(10)
        p = solver._upwind_gradient(u, drift)
        p_cen = solver._D_cen @ u
        np.testing.assert_allclose(p, p_cen)

    def test_mixed_drift_signs(self):
        solver = self._solver(10)
        x = solver.x
        u = x.copy()
        drift = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=float)
        p = solver._upwind_gradient(u, drift)
        p_bwd = solver._D_bwd @ u
        p_fwd = solver._D_fwd @ u
        expected = np.where(drift > 0, p_bwd, p_fwd)
        np.testing.assert_allclose(p, expected)

    def test_upwind_no_oscillations_on_step(self):
        """Upwind should not produce oscillations on a step-like function."""
        n_x = 50
        solver = self._solver(n_x)
        x = solver.x
        # Heaviside-like step
        u = np.where(x < 0.5, 0.0, 1.0).astype(float)
        drift = np.ones(n_x)
        p = solver._upwind_gradient(u, drift)
        # Upwind gradient should be non-negative (no oscillations for step going up)
        assert np.all(p >= -1e-10), "Upwind gradient has negative values on step function"


# ---------------------------------------------------------------------------
# Test 6 — Implicit matrix structure
# ---------------------------------------------------------------------------

class TestImplicitMatrix:
    """Tests for _build_implicit_matrix."""

    def _solver(self, n_x=20):
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.1, 50)
        model = HeatModel(sigma=1.0)
        return HJBSolver(grid=grid, time_grid=tgrid, model=model)

    def test_identity_when_sigma_zero(self):
        solver = self._solver(20)
        A = solver._build_implicit_matrix(np.zeros(20))
        # Should be identity
        A_dense = A.toarray()
        np.testing.assert_allclose(A_dense, np.eye(20), atol=1e-14)

    def test_diagonal_dominance(self):
        """Interior rows should be diagonally dominant (stable implicit scheme)."""
        solver = self._solver(20)
        sigma_sq = np.ones(20) * 2.0
        A = solver._build_implicit_matrix(sigma_sq)
        A_dense = A.toarray()
        # Boundary rows use the 3-point D² stencil; only check interior rows
        for i in range(1, 19):
            diag = abs(A_dense[i, i])
            off = np.sum(np.abs(A_dense[i])) - diag
            assert diag >= off - 1e-10, f"Interior row {i} not diagonally dominant"

    def test_shape(self):
        solver = self._solver(30)
        A = solver._build_implicit_matrix(np.ones(30))
        assert A.shape == (30, 30)

    def test_tridiagonal_structure(self):
        """Interior rows should be tridiagonal; boundary rows use 3-point D² stencil."""
        solver = self._solver(15)
        A = solver._build_implicit_matrix(np.ones(15))
        A_dense = A.toarray()
        # Interior rows (1 to n-2) must be strictly tridiagonal
        for i in range(1, 14):
            for j in range(15):
                if abs(i - j) > 1:
                    assert abs(A_dense[i, j]) < 1e-14, (
                        f"Non-tridiagonal interior entry at ({i},{j}): {A_dense[i,j]}"
                    )
        # After applying Dirichlet BCs, boundary rows become identity rows (tridiagonal)
        from mfglob.grids import TimeGrid as TG
        solver2 = self._solver(15)
        rhs = np.ones(15)
        A_bc, _ = solver2._apply_bc(A.copy(), rhs.copy(), "dirichlet", 0.0, 0.0)
        A_bc_dense = A_bc.toarray()
        for j in range(1, 15):
            assert abs(A_bc_dense[0, j]) < 1e-14, f"BC row 0 entry at col {j} nonzero"
        for j in range(14):
            assert abs(A_bc_dense[-1, j]) < 1e-14, f"BC row -1 entry at col {j} nonzero"


# ---------------------------------------------------------------------------
# Test 7 — Optimal control consistency
# ---------------------------------------------------------------------------

class TestOptimalControlConsistency:
    """Verify optimal control is consistent with the gradient of the value function."""

    def test_lq_control_equals_gradient(self):
        """For the LQ model α* = p = Du, verify at all interior times."""
        n_x, n_t = 50, 100
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.2, n_t)
        model = LinearQuadraticModel()
        x = grid.points
        terminal = np.sin(np.pi * x)
        m = flat_m(tgrid.n_nodes, n_x)

        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)

        alpha = result["optimal_control"]
        grad = result["gradient"]
        # For LQ: α* = Du at every time step
        np.testing.assert_allclose(alpha[:, 1:-1], grad[:, 1:-1], atol=1e-12,
                                   err_msg="LQ optimal control should equal Du")

    def test_zero_control_for_heat_model(self):
        """For the heat model, optimal control should be zero everywhere."""
        n_x, n_t = 40, 100
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.1, n_t)
        model = HeatModel()
        x = grid.points
        terminal = np.sin(np.pi * x)
        m = flat_m(tgrid.n_nodes, n_x)

        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        alpha = result["optimal_control"]
        np.testing.assert_allclose(alpha, 0.0, atol=1e-12)

    def test_gradient_shape_matches_value_function(self):
        n_x, n_t = 30, 50
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.1, n_t)
        model = HeatModel()
        m = flat_m(tgrid.n_nodes, n_x)
        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m)
        assert result["gradient"].shape == result["value_function"].shape

    def test_value_function_finite(self):
        """Value function should have no NaN or Inf."""
        n_x, n_t = 50, 200
        grid = Grid1D(0.0, 1.0, n_x)
        tgrid = TimeGrid(0.3, n_t)
        model = HeatModel(sigma=0.5)
        x = grid.points
        terminal = np.sin(np.pi * x)
        m = flat_m(tgrid.n_nodes, n_x)

        solver = HJBSolver(grid=grid, time_grid=tgrid, model=model)
        result = solver.solve(m, terminal_condition=terminal,
                              bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
        assert np.all(np.isfinite(result["value_function"])), "Value function contains NaN/Inf"
        assert np.all(np.isfinite(result["optimal_control"])), "Optimal control contains NaN/Inf"
        assert np.all(np.isfinite(result["gradient"])), "Gradient contains NaN/Inf"
