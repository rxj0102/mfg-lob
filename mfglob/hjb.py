"""Hamilton-Jacobi-Bellman PDE solver.

Solves the backward HJB equation:

    -∂u/∂t + H(x, Du, D²u, m) = 0     in (0,T) × Ω
    u(T, x) = g(x, m(T))

using implicit finite differences with upwind discretisation.
The tridiagonal linear system at each time step is solved via
the Thomas algorithm (O(n) per step).

This module will be implemented in Prompt 2.
"""

from __future__ import annotations


class HJBSolver:
    """Backward-in-time solver for the Hamilton-Jacobi-Bellman equation.

    Parameters
    ----------
    model    : MFGLOBModel
    grid     : Grid1D
    time_grid: TimeGrid
    theta    : float in [0, 1] — Crank-Nicolson weight (1 = fully implicit)
    """

    def __init__(self, model, grid, time_grid, theta: float = 1.0) -> None:
        self.model = model
        self.grid = grid
        self.time_grid = time_grid
        self.theta = float(theta)

    def solve(self, m: "np.ndarray") -> "np.ndarray":  # noqa: F821
        """Solve the HJB equation backward in time given density m.

        Parameters
        ----------
        m : ndarray, shape (n_space, n_time)
            Population density on the full space-time grid.

        Returns
        -------
        u : ndarray, shape (n_space, n_time)
            Value function.
        """
        raise NotImplementedError("HJBSolver will be implemented in Prompt 2.")
