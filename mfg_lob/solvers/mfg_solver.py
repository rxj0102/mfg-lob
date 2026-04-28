"""MFG fixed-point iteration coupling the HJB and Fokker-Planck solvers.

Algorithm (Lasry-Lions fixed-point iteration):
  1. Initialise density m^(0) (e.g. uniform or Gaussian).
  2. For iteration n = 0, 1, 2, ...:
     a. Solve HJB backward with density m^(n)  → u^(n+1)
     b. Solve FP  forward  with value  u^(n+1)  → m^(n+1)
     c. Update: m^(n+1) = (1-λ) m^(n) + λ m^(n+1)  (damped Picard)
     d. Check convergence: ||m^(n+1) - m^(n)||₂ / ||m^(n)||₂ < tol
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from mfg_lob.solvers.grid import Grid
from mfg_lob.solvers.hjb_solver import HJBSolver
from mfg_lob.solvers.fp_solver import FPSolver


@dataclass
class MFGEquilibrium:
    """Container for a computed MFG Nash equilibrium.

    Attributes
    ----------
    u : ndarray (nq, nt)
        Value function (HJB solution).
    m : ndarray (nq, nt)
        Population density (FP solution).
    grid : Grid
    converged : bool
    n_iterations : int
    residuals : list of float
        L2 relative density residuals per iteration.
    """

    u: np.ndarray
    m: np.ndarray
    grid: Grid
    converged: bool
    n_iterations: int
    residuals: list = field(default_factory=list)

    def value_function(self, q: np.ndarray, t: float) -> np.ndarray:
        """Interpolate u(q, t) for arbitrary q and t."""
        t_idx = np.searchsorted(self.grid.t, t)
        t_idx = np.clip(t_idx, 0, self.grid.nt - 1)
        return np.interp(q, self.grid.q, self.u[:, t_idx])

    def density(self, q: np.ndarray, t: float) -> np.ndarray:
        """Interpolate m(q, t) for arbitrary q and t."""
        t_idx = np.searchsorted(self.grid.t, t)
        t_idx = np.clip(t_idx, 0, self.grid.nt - 1)
        return np.interp(q, self.grid.q, self.m[:, t_idx])

    def optimal_control(self, q: np.ndarray, t: float, model) -> np.ndarray:
        """Return the optimal control a*(q,t) from the equilibrium."""
        t_idx = np.searchsorted(self.grid.t, t)
        t_idx = np.clip(t_idx, 0, self.grid.nt - 1)
        u_t = self.u[:, t_idx]
        m_t = self.m[:, t_idx]
        du_dq = np.gradient(u_t, self.grid.q)
        du_dq_q = np.interp(q, self.grid.q, du_dq)
        m_q = np.interp(q, self.grid.q, m_t)
        return model.optimal_control(q, du_dq_q, m_q, t)


class MFGSolver:
    """Coupled HJB-FP solver implementing the Lasry-Lions fixed-point iteration.

    Parameters
    ----------
    model : LOBModel (or any model implementing the required interface)
    grid : Grid
    tol : float
        Relative L2 convergence tolerance on the density.
    max_iter : int
        Maximum number of Picard iterations.
    damping : float in (0, 1]
        Picard damping factor λ. Smaller = more stable but slower.
    hjb_theta : float in [0,1]
        Implicit weight for HJB solver.
    fp_theta : float in [0,1]
        Implicit weight for FP solver.
    verbose : bool
    """

    def __init__(
        self,
        model,
        grid: Grid,
        tol: float = 1e-6,
        max_iter: int = 100,
        damping: float = 0.5,
        hjb_theta: float = 1.0,
        fp_theta: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.grid = grid
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping
        self.verbose = verbose

        self._hjb = HJBSolver(
            grid=grid,
            sigma=model.sigma,
            hamiltonian=model.hamiltonian,
            running_cost=model.running_cost,
            terminal_cost=model.terminal_cost,
            theta=hjb_theta,
        )
        self._fp = FPSolver(
            grid=grid,
            sigma=model.sigma,
            optimal_drift=model.optimal_drift,
            initial_density=model.initial_density,
            theta=fp_theta,
        )

    def solve(self, m_init: np.ndarray | None = None) -> MFGEquilibrium:
        """Run the fixed-point iteration to compute the MFG equilibrium.

        Parameters
        ----------
        m_init : ndarray (nq, nt), optional
            Initial guess for the population density. Defaults to
            the model's initial density broadcast across time.

        Returns
        -------
        MFGEquilibrium
        """
        g = self.grid
        model = self.model

        if m_init is None:
            m0 = model.initial_density(g.q)
            m0 = np.maximum(m0, 0.0)
            mass = np.trapezoid(m0, g.q)
            if mass > 1e-14:
                m0 /= mass
            m_prev = np.outer(m0, np.ones(g.nt))
        else:
            m_prev = m_init.copy()

        residuals = []
        converged = False
        u = np.zeros((g.nq, g.nt))

        for n in range(self.max_iter):
            # HJB backward with current density
            u = self._hjb.solve(m_prev)

            # FP forward with updated value function
            m_new = self._fp.solve(u)

            # Damped update
            m_update = (1.0 - self.damping) * m_prev + self.damping * m_new

            # Convergence check
            norm_prev = np.linalg.norm(m_prev)
            res = np.linalg.norm(m_update - m_prev) / (norm_prev + 1e-14)
            residuals.append(float(res))

            if self.verbose:
                print(f"  MFG iter {n+1:3d}: residual = {res:.3e}")

            m_prev = m_update

            if res < self.tol:
                converged = True
                if self.verbose:
                    print(f"  Converged after {n+1} iterations.")
                break

        if not converged and self.verbose:
            print(f"  Warning: did not converge after {self.max_iter} iterations. "
                  f"Final residual = {residuals[-1]:.3e}")

        return MFGEquilibrium(
            u=u,
            m=m_prev,
            grid=g,
            converged=converged,
            n_iterations=len(residuals),
            residuals=residuals,
        )
