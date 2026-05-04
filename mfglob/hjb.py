"""Hamilton-Jacobi-Bellman PDE solver.

Solves the backward HJB equation:

    -∂u/∂t + H(x, Du, D²u, m) = 0     in (0,T) × Ω
    u(T, x) = g(x, m(T))

using an IMEX (implicit-explicit) scheme:
  - Diffusion (σ²/2 · D²u): implicit — unconditionally stable
  - Advection  (b* · Du):   explicit with upwind spatial differences

At each backward step (n+1 → n):

    (I - dt·σ²/2·D²) u^n = u^{n+1} + dt·(f(x,α*,m^n) + b*·D_upwind u^{n+1})  ... (*)

Wait — the HJB in standard form is:
    -∂u/∂t + H = 0   with H = sup_α{-f - b·p - σ²/2·M}

Rearranging backward step (u^n - u^{n+1})/(-dt) + H^n = 0, so:

    (u^{n+1} - u^n)/dt + H(x, Du^{n+1}, D²u^n, m^n) = 0

IMEX splitting:
    (u^n - u^{n+1})/dt = H(x, Du^{n+1}, D²u^n, m^n)

with diffusion treated implicitly at time n:
    (I - dt·σ²/2·D²) u^n = u^{n+1} + dt·(-f - b*·D_upwind u^{n+1})
                         = u^{n+1} - dt·f - dt·b*·D_upwind u^{n+1}

The tridiagonal system is solved via scipy.linalg.solve_banded (Thomas algorithm).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mfglob.grids import Grid1D, TimeGrid


class HJBSolver:
    """
    Solve the Hamilton-Jacobi-Bellman equation BACKWARD in time:

    -∂u/∂t + H(x, Du, D²u, m) = 0     in (0,T) × Ω
    u(T, x) = g(x)                      terminal condition

    For a generic Hamiltonian of the form:
    H(x, p, M, m) = sup_α { -f(x,α,m) - b(x,α,m)·p - (1/2)σ²(x,α,m)·M }

    Discretization:
    - Time: IMEX — diffusion implicit, advection explicit with upwind differences
    - Space: finite differences, upwind scheme for first derivatives

    The IMEX scheme at each backward step (from t_{n+1} to t_n):

    (I - dt·(σ²/2)·D²) u^n = u^{n+1} - dt·f(x,α*,m^n) - dt·b*·D_upwind u^{n+1}

    where D_upwind uses upwind differences based on the sign of the optimal drift.
    """

    def __init__(self, grid: Grid1D, time_grid: TimeGrid, model) -> None:
        """
        Args:
            grid: spatial grid
            time_grid: temporal grid
            model: MFGLOBModel providing hamiltonian, optimal_control, drift, etc.
        """
        self.grid = grid
        self.time_grid = time_grid
        self.model = model
        self.x = grid.points
        self.n_x = grid.n
        self.dt = time_grid.dt
        self.n_times = time_grid.n_nodes

        # Pre-compute second-derivative operator (n_x × n_x sparse matrix)
        from mfglob.operators import FiniteDifferenceOperators
        self._ops = FiniteDifferenceOperators(grid)
        self._D2 = self._ops.d2_dx2()
        self._D_fwd = self._ops.d_dx_forward()
        self._D_bwd = self._ops.d_dx_backward()
        self._D_cen = self._ops.d_dx_central()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        m: np.ndarray,
        terminal_condition: np.ndarray = None,
        bc_type: str = "dirichlet",
        bc_left: float = 0.0,
        bc_right: float = 0.0,
    ) -> dict:
        """
        Solve the HJB equation backward in time, given the population m.

        Args:
            m: population distribution, shape (n_times, n_x)
               m[k, :] is the distribution at time t_k
            terminal_condition: u(T, x), shape (n_x,)
               If None, uses model.terminal_cost(x, m[-1])
            bc_type: boundary condition type ('dirichlet', 'neumann', etc.)
            bc_left: left boundary value
            bc_right: right boundary value

        Returns:
            dict with:
            - 'value_function': u(t, x), shape (n_times, n_x)
            - 'optimal_control': α*(t, x), shape (n_times, n_x)
            - 'gradient': Du(t, x), shape (n_times, n_x)
        """
        x = self.x
        n_x = self.n_x
        dt = self.dt
        n_t = self.n_times

        u = np.zeros((n_t, n_x))
        alpha_star = np.zeros((n_t, n_x))
        grad_u = np.zeros((n_t, n_x))

        # Terminal condition
        if terminal_condition is not None:
            u[-1, :] = np.asarray(terminal_condition, dtype=float)
        else:
            u[-1, :] = self.model.terminal_cost(x, m[-1])

        # Gradient and optimal control at terminal time
        p_T = self._D_cen @ u[-1]
        grad_u[-1, :] = p_T
        alpha_star[-1, :] = self.model.optimal_control(x, p_T, m[-1])

        # Backward loop: from index n_t-2 down to 0
        for n in range(n_t - 2, -1, -1):
            u_next = u[n + 1, :]
            m_n = m[n, :]

            # Step a: compute gradient of u^{n+1} (central for the costate)
            p_central = self._D_cen @ u_next

            # Step b: optimal control from gradient
            alpha = self.model.optimal_control(x, p_central, m_n)

            # Step c: optimal drift b*(x, α*, m^n)
            b_star = self.model.drift(x, alpha, m_n)

            # Step d: diffusion σ²(x, α*, m^n)
            sigma = self.model.diffusion(x, alpha, m_n)
            sigma_sq = sigma ** 2

            # Step e-f: upwind gradient for the advection RHS term
            p_upwind = self._upwind_gradient(u_next, b_star)

            # Running cost f(x, α*, m^n)
            f_cost = self.model.running_cost(x, alpha, m_n)

            # H = -f - b*·p - σ²/2·M, so u^n = u^{n+1} - dt·H
            # = u^{n+1} - dt·f - dt·b*·p_upwind  (implicit σ²/2·D²u^n moved to LHS)
            rhs = u_next - dt * f_cost - dt * b_star * p_upwind

            # Step g: build and solve the implicit system
            # A u^n = rhs,  A = I - dt·(σ²/2)·D²
            A = self._build_implicit_matrix(sigma_sq)

            # Apply boundary conditions to A and rhs
            A, rhs = self._apply_bc(A, rhs, bc_type, bc_left, bc_right)

            # Solve tridiagonal system
            u_n = spla.spsolve(A, rhs)

            u[n, :] = u_n
            grad_u[n, :] = self._D_cen @ u_n
            alpha_star[n, :] = self.model.optimal_control(x, grad_u[n], m_n)

        return {
            "value_function": u,
            "optimal_control": alpha_star,
            "gradient": grad_u,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_implicit_matrix(self, sigma_sq: np.ndarray) -> sp.csr_matrix:
        """
        Build (I - dt · σ²/2 · D²) for the implicit diffusion step.

        This is a tridiagonal matrix solved via scipy.sparse.linalg.spsolve.
        """
        n = self.n_x
        dt = self.dt
        # Diagonal scaling: diag(σ²/2) @ D²
        S = sp.diags(0.5 * sigma_sq, 0, format="csr")
        A = sp.eye(n, format="csr") - dt * (S @ self._D2)
        return A

    def _upwind_gradient(self, u: np.ndarray, drift: np.ndarray) -> np.ndarray:
        """
        Compute the upwind approximation of Du:

        (Du)_i = D^- u_i  if drift_i > 0  (information flows from left)
        (Du)_i = D^+ u_i  if drift_i < 0  (information flows from right)
        (Du)_i = D^0 u_i  if drift_i = 0  (use central)

        This ensures the scheme is monotone → convergence to viscosity solution.
        """
        p_fwd = self._D_fwd @ u
        p_bwd = self._D_bwd @ u
        p_cen = self._D_cen @ u

        p = np.where(drift > 0, p_bwd, np.where(drift < 0, p_fwd, p_cen))
        return p

    def _apply_bc(
        self,
        A: sp.csr_matrix,
        rhs: np.ndarray,
        bc_type: str,
        bc_left: float,
        bc_right: float,
    ):
        """Apply boundary conditions to (A, rhs) in place."""
        DIRICHLET = {"dirichlet", "absorbing"}
        NEUMANN = {"neumann", "reflecting"}

        A_lil = A.tolil()

        if bc_type in DIRICHLET:
            # Left boundary: u[0] = bc_left
            A_lil[0, :] = 0.0
            A_lil[0, 0] = 1.0
            rhs[0] = bc_left
            # Right boundary: u[-1] = bc_right
            A_lil[-1, :] = 0.0
            A_lil[-1, -1] = 1.0
            rhs[-1] = bc_right
        elif bc_type in NEUMANN:
            dx0 = float(self.grid.dx) if self.grid.grid_type == "uniform" else float(self.grid._dxf[0])
            dxN = float(self.grid.dx) if self.grid.grid_type == "uniform" else float(self.grid._dxf[-1])
            A_lil[0, :] = 0.0
            A_lil[0, 0] = -1.0 / dx0
            A_lil[0, 1] = 1.0 / dx0
            rhs[0] = bc_left
            A_lil[-1, :] = 0.0
            A_lil[-1, -2] = -1.0 / dxN
            A_lil[-1, -1] = 1.0 / dxN
            rhs[-1] = bc_right

        return A_lil.tocsr(), rhs
