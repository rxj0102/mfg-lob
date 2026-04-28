"""Hamiltonians for MFG-LOB models.

The Hamiltonian H(q, p, m) arises from the optimal control problem.
For a representative agent with control a (order submission rate), the
Lagrangian is L(q, a) = execution_cost(a) and the Hamiltonian is:

    H(q, p, m) = sup_a { -a p - L(q, a, m) }
               = sup_a { -a p - c(a) }

The sup is achieved at a* = (c')^{-1}(-p), giving:

    H(q, p, m) = -a*(p) p - c(a*(p))

Different agent archetypes lead to different Hamiltonians and drift terms.

Notation
--------
p = ∂_q u  (costate = marginal value of inventory)
a          = order submission rate (positive = buying, negative = selling)
"""

from __future__ import annotations
import numpy as np


class MarketMakerHamiltonian:
    """Hamiltonian for a symmetric market maker who posts limit orders.

    The market maker controls the depth δ at which she quotes (half-spread).
    Orders arrive at rate Λ(δ) = Λ0 * exp(-κ δ). Execution revenue per lot
    is δ (the half-spread earned). Inventory changes by ±1 per fill.

    Value process: da = ± δ  per executed order  →  inventory drift controlled
    by choosing δ independently on bid and ask sides.

    Running optimisation (per side):
        sup_δ≥0 { Λ(δ) * (δ - p) }  where p = |∂_q u|

    FOC: Λ0 * e^{-κδ} * (1 - κ(δ - p)) = 0  →  δ* = 1/κ + p

    Hamiltonian (combined bid + ask):
        H(q, p, m) = (Λ0 / κ) * exp(-1) * exp(-κ p_ask) + (Λ0 / κ) * exp(-1) * exp(-κ p_bid)

    Optimal half-spread: δ*(p) = 1/κ + p  (bid), 1/κ - p  (ask)
    Optimal drift: b*(q, p) = Λ(δ*_ask) - Λ(δ*_bid)

    Parameters
    ----------
    Lambda0 : float
        Base order arrival rate.
    kappa : float
        Order arrival rate decay with depth.
    """

    def __init__(self, Lambda0: float = 1.0, kappa: float = 1.0) -> None:
        self.Lambda0 = float(Lambda0)
        self.kappa = float(kappa)

    def __call__(self, q: np.ndarray, p: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """Evaluate H(q, p, m)."""
        L0, k = self.Lambda0, self.kappa
        e_inv = np.exp(-1.0)
        # Ask side: agent sells, costate effect is +p (inventory decreases)
        # Bid side: agent buys, costate effect is -p (inventory increases)
        H_ask = (L0 / k) * e_inv * np.exp(-k * np.maximum(p, -1.0 / k))
        H_bid = (L0 / k) * e_inv * np.exp( k * np.minimum(p,  1.0 / k))
        return H_ask + H_bid

    def optimal_spread(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (delta_ask*, delta_bid*) optimal half-spreads."""
        delta_ask = 1.0 / self.kappa + p
        delta_bid = 1.0 / self.kappa - p
        return np.maximum(delta_ask, 0.0), np.maximum(delta_bid, 0.0)

    def optimal_drift(self, q: np.ndarray, p: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """b*(q, p, m, t): net inventory drift under the optimal strategy."""
        L0, k = self.Lambda0, self.kappa
        delta_ask, delta_bid = self.optimal_spread(p)
        rate_ask = L0 * np.exp(-k * delta_ask)
        rate_bid = L0 * np.exp(-k * delta_bid)
        # buying increases inventory, selling decreases it
        return rate_bid - rate_ask


class TrendFollowerHamiltonian:
    """Hamiltonian for a momentum / trend-following trader.

    The agent submits market orders at rate a ∈ ℝ. The execution cost is
    quadratic: c(a) = (epsilon/2) a^2.  The Hamiltonian is:

        H(q, p, m) = sup_a { -a p - (epsilon/2) a^2 }
                   = p^2 / (2 epsilon)

    Optimal control: a*(p) = -p / epsilon  (buy when ∂_q u > 0, i.e. marginal
    value of inventory is positive).

    Parameters
    ----------
    epsilon : float
        Price impact / order execution cost coefficient.
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        self.epsilon = float(epsilon)

    def __call__(self, q: np.ndarray, p: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        return p**2 / (2.0 * self.epsilon)

    def optimal_drift(self, q: np.ndarray, p: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """a*(p) = -p / epsilon."""
        return -p / self.epsilon
