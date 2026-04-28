"""Forward Fokker-Planck solver for the MFG-LOB system.

Solves the Kolmogorov forward (Fokker-Planck) equation:

    ∂_t m - (σ²/2) ∂_qq m + ∂_q [b(q, u, m, t) m] = 0,  (q,t) ∈ Ω × (0,T]
    m(q, 0) = m0(q)                                         (initial distribution)

where:
  m(q,t)          = population density of agents at inventory q at time t
  b(q, u, m, t)   = optimal drift derived from the HJB value function u
                    b = -∂_H/∂p  (optimal control via Pontryagin)

Discretisation: upwind finite differences for the advection term + implicit
diffusion (Crank-Nicolson / backward Euler). Mass is conserved to machine
precision by construction.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

from mfg_lob.solvers.grid import Grid


class FPSolver:
    """Forward-in-time finite-difference solver for the Fokker-Planck equation.

    Parameters
    ----------
    grid : Grid
    sigma : float
        Inventory diffusion coefficient (same as HJB).
    optimal_drift : callable (q, u, m, t) -> float array
        b(q, u, m, t): optimal control drift on the inventory grid.
        Derived from the HJB solution as b = argmin_a { a p + L(a) }
        where p = ∂_q u. Computed externally per time step.
    initial_density : callable (q) -> float array
        m0(q): normalised initial distribution (integrates to 1 over q).
    theta : float in [0,1]
        Implicit weight (1 = fully implicit Euler).
    """

    def __init__(
        self,
        grid: Grid,
        sigma: float,
        optimal_drift,
        initial_density,
        theta: float = 1.0,
    ) -> None:
        self.grid = grid
        self.sigma = float(sigma)
        self.optimal_drift = optimal_drift
        self.initial_density = initial_density
        self.theta = float(theta)

    def _upwind_flux_matrix(self, b: np.ndarray) -> np.ndarray:
        """Build banded matrix for upwind advection ∂_q(b*m).

        Uses donor-cell upwinding: flux F_{i+1/2} = b_{i+1/2} * m_upwind.
        """
        g = self.grid
        nq = g.nq
        dq = g.dq
        dt = g.dt

        # Interface velocities (average)
        b_right = 0.5 * (b[:-1] + b[1:])   # b_{i+1/2}, shape (nq-1,)

        # Upwind: if b_{i+1/2} > 0 → flux from left cell i
        #         if b_{i+1/2} < 0 → flux from right cell i+1
        # ∂_q(bm)_i ≈ (F_{i+1/2} - F_{i-1/2}) / dq
        # Contribution to RHS requires the matrix for implicit solve.

        # Implicit advection matrix A_adv such that (A_adv @ m) = ∂_q(bm)
        ab = np.zeros((3, nq))  # banded (upper, main, lower)

        for i in range(nq):
            # Right interface i+1/2
            if i < nq - 1:
                bf_right = b_right[i]
                coeff_right_self = max(bf_right, 0.0) / dq
                coeff_right_next = -min(bf_right, 0.0) / dq
            else:
                coeff_right_self = 0.0
                coeff_right_next = 0.0

            # Left interface i-1/2
            if i > 0:
                bf_left = b_right[i - 1]
                coeff_left_self = -min(bf_left, 0.0) / dq
                coeff_left_prev = max(bf_left, 0.0) / dq
            else:
                coeff_left_self = 0.0
                coeff_left_prev = 0.0

            ab[1, i] += coeff_right_self - coeff_left_self

        # Build full upwind entries for off-diagonals
        # Upper diagonal: contribution from m_{i+1} to cell i
        for i in range(nq - 1):
            bf_right = b_right[i]
            if bf_right < 0:
                ab[0, i + 1] = -bf_right / dq  # upper: row i, col i+1

        # Lower diagonal: contribution from m_{i-1} to cell i
        for i in range(1, nq):
            bf_left = b_right[i - 1]
            if bf_left > 0:
                ab[2, i - 1] = bf_left / dq   # lower: row i, col i-1

        return ab * dt

    def solve(self, u: np.ndarray) -> np.ndarray:
        """Solve FP forward in time given value function u.

        Parameters
        ----------
        u : np.ndarray, shape (nq, nt)
            Value function from the HJB solver.

        Returns
        -------
        m : np.ndarray, shape (nq, nt)
            Population density on the full grid (normalised at each time).
        """
        g = self.grid
        nq, nt = g.nq, g.nt
        m = np.zeros((nq, nt))

        # Initial condition
        m0 = self.initial_density(g.q)
        m0 = np.maximum(m0, 0.0)
        mass = np.trapezoid(m0, g.q)
        if mass > 1e-14:
            m0 = m0 / mass
        m[:, 0] = m0

        nu = self.sigma**2 / 2.0
        r = nu * g.dt / g.dq**2

        # Implicit diffusion matrix (same structure as HJB)
        ab_diff = np.zeros((3, nq))
        ab_diff[0, 1:] = -self.theta * r
        ab_diff[1, :] = 1.0 + 2.0 * self.theta * r
        ab_diff[2, :-1] = -self.theta * r
        ab_diff[1, 0] = 1.0 + self.theta * r
        ab_diff[1, -1] = 1.0 + self.theta * r

        for k in range(nt - 1):
            t_k = g.t[k]
            m_curr = m[:, k]
            u_k = u[:, k]

            # Optimal drift from HJB gradient
            b = self.optimal_drift(g.q, u_k, m_curr, t_k)

            # Explicit diffusion term
            D2m = np.zeros(nq)
            D2m[1:-1] = (m_curr[2:] - 2.0 * m_curr[1:-1] + m_curr[:-2]) / g.dq**2
            D2m[0] = (m_curr[1] - m_curr[0]) / g.dq**2
            D2m[-1] = (m_curr[-2] - m_curr[-1]) / g.dq**2

            # Explicit advection divergence  ∂_q(b m)
            adv = np.zeros(nq)
            b_right = 0.5 * (b[:-1] + b[1:])
            # Upwind flux
            flux = np.where(b_right > 0, b_right * m_curr[:-1], b_right * m_curr[1:])
            adv[1:-1] = (flux[1:] - flux[:-1]) / g.dq
            # Boundary zero-flux
            adv[0] = flux[0] / g.dq
            adv[-1] = -flux[-1] / g.dq

            rhs = m_curr + g.dt * ((1.0 - self.theta) * nu * D2m - adv)

            m_next = solve_banded((1, 1), ab_diff, rhs)

            # Enforce non-negativity and re-normalise mass
            m_next = np.maximum(m_next, 0.0)
            mass = np.trapezoid(m_next, g.q)
            if mass > 1e-14:
                m_next = m_next / mass
            m[:, k + 1] = m_next

        return m
