"""Fokker-Planck (Kolmogorov forward) PDE solver.

Solves the forward equation in conservative form:

    ∂m/∂t + ∂J/∂x = 0
    J = b·m - (1/2)∂(σ²·m)/∂x

where J is the probability flux.  Equivalently:

    ∂m/∂t = (1/2)∂²(σ²·m)/∂x² - ∂(b·m)/∂x

IMEX scheme (forward step n → n+1):

    (I - dt/2 · L_D[σ²^n]) m^{n+1} = m^n - dt · ∂_x(b^n · m^n)_upwind

where L_D is the conservative diffusion operator built with half-point
diffusion coefficients D_{i±1/2} = (σ²_i + σ²_{i±1})/2.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mfglob.grids import Grid1D, TimeGrid


class FokkerPlanckSolver:
    """
    Solve the Fokker-Planck equation FORWARD in time:

    ∂m/∂t = (1/2)∂²(σ²·m)/∂x² - ∂(b·m)/∂x     in (0,T) × Ω
    m(0, x) = m_0(x)

    Discretization: IMEX with conservative upwind advection and
    half-point diffusion coefficients (preserves positivity and mass).
    """

    def __init__(self, grid: Grid1D, time_grid: TimeGrid, model) -> None:
        self.grid = grid
        self.time_grid = time_grid
        self.model = model
        self.x = grid.points
        self.n_x = grid.n
        self.dt = time_grid.dt
        self.n_times = time_grid.n_nodes

        if grid.grid_type == "uniform":
            self._dx = float(grid.dx)
        else:
            self._dx = np.asarray(grid.dx, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        drift: np.ndarray,
        diffusion: np.ndarray,
        initial_distribution: np.ndarray = None,
        bc_type: str = "absorbing",
    ) -> dict:
        """
        Solve the FP equation forward in time.

        Args:
            drift:      b(t, x), shape (n_times, n_x)
            diffusion:  σ(t, x), shape (n_times, n_x)
            initial_distribution: m_0(x), shape (n_x,)
            bc_type: 'absorbing', 'reflecting', or 'periodic'

        Returns dict with 'density', 'flux', 'mass', 'mass_error'.
        """
        x = self.x
        n_x = self.n_x
        dt = self.dt
        n_t = self.n_times

        m = np.zeros((n_t, n_x))
        flux = np.zeros((n_t, n_x))

        # Initial condition
        if initial_distribution is not None:
            m0 = np.asarray(initial_distribution, dtype=float).copy()
        else:
            m0 = self.model.initial_distribution(x)

        # Normalize
        mass0 = np.trapezoid(m0, x)
        if mass0 > 1e-14:
            m0 = m0 / mass0

        m[0, :] = m0

        # Forward loop
        for n in range(n_t - 1):
            b_n = drift[n]
            sig_n = diffusion[n]
            sigma_sq = sig_n ** 2
            m_n = m[n]

            # Explicit advection: ∂(b·m)/∂x
            adv = self._conservative_advection(m_n, b_n)

            # RHS
            rhs = m_n - dt * adv

            # Implicit diffusion matrix
            A = self._diffusion_matrix(sigma_sq, bc_type)

            # Apply BCs to rhs
            rhs = self._apply_bc_rhs(rhs, bc_type)

            # Solve
            m_new = spla.spsolve(A, rhs)

            # Enforce positivity and renormalize
            m_new = np.maximum(m_new, 0.0)
            mass = np.trapezoid(m_new, x)
            if mass > 1e-14:
                m_new = m_new / mass

            m[n + 1, :] = m_new

            # Record flux: J = b·m - (σ²/2)·∂m/∂x (central diff for output only)
            flux[n, :] = b_n * m_n

        # Final flux
        flux[-1, :] = drift[-1] * m[-1]

        # Mass at every time
        mass_arr = np.array([np.trapezoid(m[k], x) for k in range(n_t)])
        mass_error = float(np.max(np.abs(mass_arr - 1.0)))

        return {
            "density": m,
            "flux": flux,
            "mass": mass_arr,
            "mass_error": mass_error,
        }

    # ------------------------------------------------------------------
    # Conservative advection (Godunov)
    # ------------------------------------------------------------------

    def _conservative_advection(self, m: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute ∂(b·m)/∂x via conservative Godunov (upwind) scheme.

        Interface velocity b_{i+1/2} = (b_i + b_{i+1})/2.
        Flux:  F_{i+1/2} = max(b_{i+1/2},0)·m_i + min(b_{i+1/2},0)·m_{i+1}
        Div:   (∂(bm)/∂x)_i = (F_{i+1/2} - F_{i-1/2}) / dx
        """
        n = self.n_x
        dx = self._dx  # scalar for uniform grid

        # Interface velocities: b_{i+1/2} for i=0..n-2
        b_half = 0.5 * (b[:-1] + b[1:])  # shape (n-1,)

        # Godunov fluxes at interfaces
        F = np.maximum(b_half, 0.0) * m[:-1] + np.minimum(b_half, 0.0) * m[1:]

        # Boundary fluxes: zero-flux (no mass leaves through ghost cells)
        F_left = 0.0    # F_{-1/2}
        F_right = 0.0   # F_{n-1/2}

        div = np.empty(n)
        div[0] = (F[0] - F_left) / dx
        div[1:-1] = (F[1:] - F[:-1]) / dx
        div[-1] = (F_right - F[-1]) / dx

        return div

    # ------------------------------------------------------------------
    # Conservative diffusion matrix
    # ------------------------------------------------------------------

    def _diffusion_matrix(self, sigma_sq: np.ndarray, bc_type: str) -> sp.csr_matrix:
        """
        Build (I - dt/2 · L_D) where L_D is the conservative diffusion operator:

        (L_D m)_i = [D_{i+1/2}(m_{i+1}-m_i) - D_{i-1/2}(m_i-m_{i-1})] / dx²

        with D_{i±1/2} = (σ²_i + σ²_{i±1})/2 (half-point averages).
        """
        n = self.n_x
        dt = self.dt
        dx = self._dx  # scalar

        # Half-point diffusion coefficients
        D_half = 0.5 * (sigma_sq[:-1] + sigma_sq[1:])  # shape (n-1,)

        # Build tridiagonal entries for L_D
        # Row i (interior): coefs of m_{i-1}, m_i, m_{i+1}
        main = np.zeros(n)
        upper = np.zeros(n - 1)
        lower = np.zeros(n - 1)

        for i in range(1, n - 1):
            lower[i - 1] = D_half[i - 1] / dx**2
            upper[i] = D_half[i] / dx**2
            main[i] = -(D_half[i - 1] + D_half[i]) / dx**2

        # Boundary rows handled by BC application
        main[0] = 1.0
        main[-1] = 1.0

        L_D = sp.diags([lower, main, upper], [-1, 0, 1], shape=(n, n), format="lil")

        # Apply boundary conditions
        if bc_type in ("absorbing", "dirichlet"):
            # m=0 at both ends: identity rows → already set (main[0]=1, main[-1]=1)
            L_D[0, :] = 0.0
            L_D[0, 0] = 1.0
            L_D[-1, :] = 0.0
            L_D[-1, -1] = 1.0
        elif bc_type in ("reflecting", "neumann"):
            # No-flux: zero-Neumann on m → first/last rows use one-sided stencil
            # Row 0: D_{1/2}(m_1 - m_0)/dx² (one-sided, D_{-1/2}=0)
            L_D[0, :] = 0.0
            L_D[0, 0] = -D_half[0] / dx**2
            L_D[0, 1] = D_half[0] / dx**2
            # Row n-1: D_{n-3/2}(m_{n-2} - m_{n-1})/dx²
            L_D[-1, :] = 0.0
            L_D[-1, -2] = D_half[-1] / dx**2
            L_D[-1, -1] = -D_half[-1] / dx**2

        L_D = L_D.tocsr()
        A = sp.eye(n, format="csr") - (dt / 2.0) * L_D
        return A

    def _apply_bc_rhs(self, rhs: np.ndarray, bc_type: str) -> np.ndarray:
        """Enforce boundary values on RHS vector."""
        rhs = rhs.copy()
        if bc_type in ("absorbing", "dirichlet"):
            rhs[0] = 0.0
            rhs[-1] = 0.0
        return rhs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_mass_conservation(self, m: np.ndarray) -> float:
        """Return max |∫m dx - 1| over all time slices."""
        masses = np.array([np.trapezoid(m[k], self.x) for k in range(m.shape[0])])
        return float(np.max(np.abs(masses - 1.0)))
