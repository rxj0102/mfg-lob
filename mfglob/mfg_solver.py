"""Coupled MFG system solver (Lasry-Lions fixed-point iteration).

Alternates between:
  1. Solving HJB backward in time given current density m^(k)
  2. Solving FP  forward  in time given current value   u^(k+1)
  3. Damped update and convergence check

This module will be implemented in Prompt 2.
"""

from __future__ import annotations


class MFGSystemSolver:
    """Lasry-Lions fixed-point solver for the coupled HJB-FP system.

    Parameters
    ----------
    model       : MFGLOBModel
    grid        : Grid1D
    time_grid   : TimeGrid
    tol         : float — L∞ convergence tolerance on m
    max_iter    : int   — maximum Picard iterations
    damping     : float — λ ∈ (0, 1], Picard damping factor
    """

    def __init__(
        self,
        model,
        grid,
        time_grid,
        tol: float = 1e-6,
        max_iter: int = 100,
        damping: float = 0.5,
    ) -> None:
        self.model = model
        self.grid = grid
        self.time_grid = time_grid
        self.tol = tol
        self.max_iter = max_iter
        self.damping = damping

    def solve(self):
        """Compute the MFG Nash equilibrium.

        Returns
        -------
        Equilibrium object (to be defined in Prompt 2).
        """
        raise NotImplementedError("MFGSystemSolver will be implemented in Prompt 2.")
