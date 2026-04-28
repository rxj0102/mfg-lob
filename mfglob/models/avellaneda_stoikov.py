"""Avellaneda-Stoikov market-making MFG model.

The representative market maker controls her bid and ask quotes (δ_bid, δ_ask)
to maximise expected PnL subject to inventory risk.  In the MFG formulation,
the equilibrium is determined by the distribution of inventory across all
market makers.

Reference:
  Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book",
  Quantitative Finance 8(3), 217-224.
  Cardaliaguet & Lehalle (2018), "Mean field game of controls and an application
  to trade crowding", Mathematics and Financial Economics 12, 335-363.

Will be fully implemented in Prompt 2.
"""

from __future__ import annotations
import numpy as np
from mfglob.models.base import MFGLOBModel


class AvellanedaStoikovMFG(MFGLOBModel):
    """MFG extension of the Avellaneda-Stoikov market-maker model.

    State:    q ∈ [q_min, q_max]  — inventory level
    Control:  (δ_bid, δ_ask)      — bid and ask half-spreads

    Parameters
    ----------
    sigma    : float — mid-price volatility
    gamma    : float — inventory risk aversion
    Lambda0  : float — base order arrival rate
    kappa    : float — order arrival decay with distance
    A        : float — terminal inventory penalty
    """

    def __init__(
        self,
        sigma: float = 0.3,
        gamma: float = 0.1,
        Lambda0: float = 1.0,
        kappa: float = 1.5,
        A: float = 0.5,
    ) -> None:
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.Lambda0 = float(Lambda0)
        self.kappa = float(kappa)
        self.A = float(A)

    def hamiltonian(self, x, p, M, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def optimal_control(self, x, p, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def drift(self, x, alpha, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def diffusion(self, x, alpha, m):
        return np.full_like(x, self.sigma)

    def running_cost(self, x, alpha, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def terminal_cost(self, x, m):
        return self.A * x**2

    def initial_distribution(self, x):
        density = np.exp(-0.5 * x**2 / 2.0**2)
        return density / np.trapezoid(density, x)

    @property
    def state_dimension(self) -> int:
        return 1

    @property
    def param_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "gamma": self.gamma,
            "Lambda0": self.Lambda0,
            "kappa": self.kappa,
            "A": self.A,
        }
