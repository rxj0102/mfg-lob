"""
mfglob — Mean-Field Game solver for Limit Order Book models.

Core library providing:
  grids      — Grid1D, TimeGrid, Grid2D
  operators  — FiniteDifferenceOperators
  models     — MFGLOBModel base class and concrete LOB model implementations
  hjb        — Hamilton-Jacobi-Bellman PDE solver
  fokker_planck — Fokker-Planck PDE solver
  mfg_solver — Coupled MFG system (Lasry-Lions fixed-point)
  equilibrium   — Equilibrium analysis and verification
  price_impact  — Price impact extraction from MFG equilibrium
  calibration   — Parameter calibration to empirical LOB data
"""

from mfglob.grids import Grid1D, TimeGrid, Grid2D
from mfglob.operators import FiniteDifferenceOperators

__version__ = "0.1.0"
__all__ = [
    "Grid1D",
    "TimeGrid",
    "Grid2D",
    "FiniteDifferenceOperators",
]
