"""Running and terminal cost functions for MFG-LOB models.

In the LOB MFG, agents face costs for:
  - Holding inventory (risk aversion / capital cost)
  - Adverse selection from the population density
  - Terminal liquidation penalty

All cost functions operate on numpy arrays over the inventory grid.
"""

from __future__ import annotations
import numpy as np


class QuadraticCost:
    """Quadratic running cost: f(q, m, t) = phi * q^2 + alpha * integral coupling.

    Parameters
    ----------
    phi : float
        Inventory aversion parameter (quadratic penalty on |q|).
    alpha : float
        Mean-field coupling coefficient (interaction cost).
    """

    def __init__(self, phi: float = 0.1, alpha: float = 1.0) -> None:
        self.phi = float(phi)
        self.alpha = float(alpha)

    def __call__(self, q: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """f(q, m, t) = phi * q^2 + alpha * mean-field term."""
        # Mean-field coupling: agents dislike crowded inventory positions
        # The MFG interaction term penalizes concentration of m near q
        mf_term = self.alpha * m  # pointwise interaction
        return self.phi * q**2 + mf_term


class ExponentialCost:
    """Exponential (CARA) running cost representing risk-averse inventory penalty.

    f(q, m, t) = phi * (exp(gamma * q^2) - 1) + alpha * m

    Parameters
    ----------
    phi : float
        Scale of inventory penalty.
    gamma : float
        Risk-aversion coefficient.
    alpha : float
        Mean-field coupling coefficient.
    """

    def __init__(self, phi: float = 0.05, gamma: float = 0.1, alpha: float = 1.0) -> None:
        self.phi = float(phi)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def __call__(self, q: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        return self.phi * (np.exp(self.gamma * q**2) - 1.0) + self.alpha * m


class LOBCostTerminal:
    """Terminal liquidation cost g(q).

    Penalises agents for holding non-zero inventory at the horizon.
    Models the cost of crossing the spread to liquidate.

    g(q) = A * q^2 + B * |q|

    Parameters
    ----------
    A : float
        Quadratic liquidation cost (e.g. permanent price impact).
    B : float
        Linear liquidation cost (e.g. bid-ask spread crossing).
    """

    def __init__(self, A: float = 1.0, B: float = 0.0) -> None:
        self.A = float(A)
        self.B = float(B)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        return self.A * q**2 + self.B * np.abs(q)
