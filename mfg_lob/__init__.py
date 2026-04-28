"""
mfg-lob: Mean-Field Game solver for Limit Order Book models.

Solves the coupled HJB-Fokker-Planck PDE system characterizing Nash equilibria
in continuum trader populations. Provides comparative statics, price impact
analysis, and calibration to empirical LOB data.

Quick start
-----------
>>> from mfg_lob.models import LOBModel
>>> from mfg_lob.solvers import MFGSolver
>>> model = LOBModel(sigma=0.5, phi=0.1, alpha=1.0)
>>> grid = mfg_lob.solvers.Grid.uniform(q_min=-10, q_max=10, nq=101,
...                                      t_min=0.0, t_max=1.0, nt=201)
>>> solver = MFGSolver(model, grid)
>>> eq = solver.solve()
"""

from mfg_lob import solvers, models, analysis, calibration
from mfg_lob.solvers.grid import Grid
from mfg_lob.solvers.mfg_solver import MFGSolver, MFGEquilibrium

__version__ = "0.1.0"
__all__ = [
    "solvers",
    "models",
    "analysis",
    "calibration",
    "Grid",
    "MFGSolver",
    "MFGEquilibrium",
]
