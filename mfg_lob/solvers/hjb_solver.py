"""Backward HJB solver for the MFG-LOB system.

Solves the Hamilton-Jacobi-Bellman equation backward in time:

    -∂_t u - (σ²/2) ∂_qq u + H(q, ∂_q u, m) + f(q, m) = 0,  (q,t) ∈ Ω × [0,T)
    u(q, T) = g(q)                                              (terminal condition)

where:
  u(q,t)   = value function of the representative agent
  σ        = volatility of inventory / order flow
  H        = Hamiltonian encoding the optimal quoting strategy
  f(q,m)   = running cost (inventory penalty, adverse selection, etc.)
  g(q)     = terminal inventory liquidation cost

Discretisation: implicit finite differences in q (Crank-Nicolson or fully implicit),
explicit in time (backward Euler sweep). The tridiagonal linear system is solved
with Thomas algorithm O(nq).
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from mfg_lob.solvers.grid import Grid


class HJBSolver:
    """Backward-in-time finite-difference solver for the HJB equation.

    Parameters
    ----------
    grid : Grid
    sigma : float
        Inventory diffusion coefficient.
    hamiltonian : callable (q, du_dq, m, t) -> float array
        Evaluates H(q, p, m) on the full inventory grid. p = ∂_q u is
        computed by central differences inside the solver.
    running_cost : callable (q, m, t) -> float array
        f(q, m, t): running cost evaluated on the inventory grid.
    terminal_cost : callable (q) -> float array
        g(q): terminal cost evaluated on the inventory grid.
    theta : float in [0,1]
        Crank-Nicolson parameter. 1 = fully implicit, 0.5 = CN.
    """

    def __init__(
        self,
        grid: Grid,
        sigma: float,
        hamiltonian,
        running_cost,
        terminal_cost,
        theta: float = 1.0,
    ) -> None:
        self.grid = grid
        self.sigma = float(sigma)
        self.hamiltonian = hamiltonian
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.theta = float(theta)

        self._build_diffusion_matrix()

    def _build_diffusion_matrix(self) -> None:
        """Pre-build the banded diffusion operator for efficiency."""
        g = self.grid
        nq, dq, dt = g.nq, g.dq, g.dt
        nu = self.sigma**2 / 2.0
        r = nu * dt / dq**2  # diffusion number

        # Banded storage for scipy: (2,1,0) = (upper, main, lower) rows
        # For the implicit part: I - theta * r * D2
        # D2 u_i = u_{i+1} - 2u_i + u_{i-1}
        # leading to tridiagonal: -theta*r on off-diags, 1+2*theta*r on main
        self._r = r

        # Store banded matrix (ab[0]=upper, ab[1]=main, ab[2]=lower)
        ab = np.zeros((3, nq))
        ab[0, 1:] = -self.theta * r          # upper diagonal
        ab[1, :] = 1.0 + 2.0 * self.theta * r  # main diagonal
        ab[2, :-1] = -self.theta * r         # lower diagonal

        # Neumann BC: zero-flux at boundaries (reflect inventory boundaries)
        # Modify corner entries
        ab[1, 0] = 1.0 + self.theta * r
        ab[1, -1] = 1.0 + self.theta * r

        self._ab = ab

    def solve(self, m: np.ndarray) -> np.ndarray:
        """Solve HJB backward in time given population density m.

        Parameters
        ----------
        m : np.ndarray, shape (nq, nt)
            Population density on the full (q, t) grid.

        Returns
        -------
        u : np.ndarray, shape (nq, nt)
            Value function on the full grid.
        """
        g = self.grid
        nq, nt = g.nq, g.nt
        u = np.zeros((nq, nt))

        # Terminal condition
        u[:, -1] = self.terminal_cost(g.q)

        r = self._r
        ab = self._ab

        for k in range(nt - 2, -1, -1):
            t_k = g.t[k]
            u_curr = u[:, k + 1]
            m_k = m[:, k]

            # Gradient: central differences, one-sided at boundaries
            du_dq = np.empty(nq)
            du_dq[1:-1] = (u_curr[2:] - u_curr[:-2]) / (2.0 * g.dq)
            du_dq[0] = (u_curr[1] - u_curr[0]) / g.dq
            du_dq[-1] = (u_curr[-1] - u_curr[-2]) / g.dq

            H = self.hamiltonian(g.q, du_dq, m_k, t_k)
            f = self.running_cost(g.q, m_k, t_k)

            # RHS: explicit diffusion + nonlinear terms
            # Backward Euler: u_k = u_{k+1} + dt*(nu*D2 u_{k+1} - H - f)
            #   but we reorganize for the implicit part.
            # Explicit contribution from diffusion:
            D2u = np.zeros(nq)
            D2u[1:-1] = (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]) / g.dq**2
            # Neumann at boundaries
            D2u[0] = (u_curr[1] - u_curr[0]) / g.dq**2
            D2u[-1] = (u_curr[-2] - u_curr[-1]) / g.dq**2

            nu = self.sigma**2 / 2.0
            rhs = u_curr + g.dt * ((1.0 - self.theta) * nu * D2u - H - f)

            u[:, k] = solve_banded((1, 1), ab, rhs)

        return u
