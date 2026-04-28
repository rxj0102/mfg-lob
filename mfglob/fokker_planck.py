"""Fokker-Planck (Kolmogorov forward) PDE solver.

Solves the forward equation:

    ∂m/∂t - (1/2) ∂²(σ²·m)/∂x² + ∂(b*·m)/∂x = 0    in (0,T) × Ω
    m(0, x) = m₀(x)

using implicit upwind finite differences with exact mass conservation.

This module will be implemented in Prompt 2.
"""

from __future__ import annotations


class FokkerPlanckSolver:
    """Forward-in-time solver for the Fokker-Planck equation.

    Parameters
    ----------
    model    : MFGLOBModel
    grid     : Grid1D
    time_grid: TimeGrid
    theta    : float in [0, 1] — implicit weight
    """

    def __init__(self, model, grid, time_grid, theta: float = 1.0) -> None:
        self.model = model
        self.grid = grid
        self.time_grid = time_grid
        self.theta = float(theta)

    def solve(self, u: "np.ndarray") -> "np.ndarray":  # noqa: F821
        """Solve the Fokker-Planck equation forward in time given value u.

        Parameters
        ----------
        u : ndarray, shape (n_space, n_time)
            Value function from the HJB solver.

        Returns
        -------
        m : ndarray, shape (n_space, n_time)
            Population density (normalised at each time step).
        """
        raise NotImplementedError("FokkerPlanckSolver will be implemented in Prompt 2.")
