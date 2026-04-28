"""LOB shape formation MFG (Lasry-Lions type).

Models how the limit order book shape emerges from the equilibrium of many
strategic agents choosing at which price levels to post limit orders.
The equilibrium distribution m(x, t) directly gives the LOB depth profile.

Reference:
  Lasry & Lions (2007), "Mean field games", Japanese Journal of Mathematics.
  Gomes et al. (2014), "Mean field games models — a brief survey",
  Dynamic Games and Applications.

Will be fully implemented in Prompt 2.
"""

from __future__ import annotations
import numpy as np
from mfglob.models.base import MFGLOBModel


class LOBFormationMFG(MFGLOBModel):
    """Lasry-Lions MFG model for limit order book shape formation.

    State:    x ∈ [0, x_max]  — distance of limit order from mid-price
    Control:  a                — rate of order cancellation / repositioning

    The equilibrium density m(x, t) characterises the LOB depth profile:
    m(x, t) dx = standing volume at price distance x at time t.

    Parameters
    ----------
    sigma   : float — order flow volatility
    phi     : float — inventory cost at distance x
    alpha   : float — mean-field coupling (congestion cost)
    kappa   : float — order execution rate decay with depth
    """

    def __init__(
        self,
        sigma: float = 0.5,
        phi: float = 0.1,
        alpha: float = 1.0,
        kappa: float = 1.0,
    ) -> None:
        self.sigma = float(sigma)
        self.phi = float(phi)
        self.alpha = float(alpha)
        self.kappa = float(kappa)

    def hamiltonian(self, x, p, M, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def optimal_control(self, x, p, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def drift(self, x, alpha, m):
        raise NotImplementedError("Will be implemented in Prompt 2.")

    def diffusion(self, x, alpha_ctrl, m):
        return np.full_like(x, self.sigma)

    def running_cost(self, x, alpha, m):
        return self.phi * x**2 + self.alpha * m

    def terminal_cost(self, x, m):
        return np.zeros_like(x)  # no terminal penalty (stationary model)

    def initial_distribution(self, x):
        density = np.ones_like(x)
        return density / np.trapezoid(density, x)

    @property
    def state_dimension(self) -> int:
        return 1

    @property
    def param_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "phi": self.phi,
            "alpha": self.alpha,
            "kappa": self.kappa,
        }
