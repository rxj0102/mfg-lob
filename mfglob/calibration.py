"""Calibration of MFG-LOB model parameters to empirical LOB data.

Minimises a composite loss over spread, depth profile, and price impact.
Optimisation: Nelder-Mead (gradient-free) in log-parameter space.

Will be implemented in Prompt 4.
"""

from __future__ import annotations
