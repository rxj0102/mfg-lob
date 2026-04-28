"""Equilibrium analysis and verification tools.

Provides diagnostics for MFG Nash equilibria:
  - Existence check via fixed-point residuals
  - Uniqueness verification (monotonicity condition)
  - Convergence rate analysis

This module will be implemented in Prompt 3.
"""

from __future__ import annotations


class EquilibriumAnalyzer:
    """Analyse and verify a computed MFG Nash equilibrium.

    Will be implemented in Prompt 3.
    """

    def __init__(self, equilibrium) -> None:
        self.eq = equilibrium
