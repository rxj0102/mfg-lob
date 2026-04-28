"""LOBModel: the primary model class for MFG limit order book problems.

Encapsulates all model parameters and provides the callables required by
HJBSolver and FPSolver:
  - hamiltonian(q, p, m, t)
  - running_cost(q, m, t)
  - terminal_cost(q)
  - optimal_drift(q, u, m, t)
  - initial_density(q)

Two built-in configurations are provided:
  - 'market_maker'  : symmetric market makers posting limit orders
  - 'trend_follower': momentum traders submitting market orders

Parameters
----------
sigma : float
    Volatility of the inventory process (noise / order flow randomness).
phi : float
    Inventory aversion (quadratic running cost coefficient).
alpha : float
    Mean-field coupling strength (interaction between agents).
A_term : float
    Quadratic terminal liquidation cost.
B_term : float
    Linear terminal liquidation cost (spread crossing).
Lambda0 : float
    Base order arrival rate (market maker model).
kappa : float
    Order book depth decay (market maker model).
epsilon : float
    Execution cost coefficient (trend-follower model).
mu0 : float
    Mean of the Gaussian initial inventory distribution.
sigma0 : float
    Std dev of the Gaussian initial inventory distribution.
agent_type : str
    'market_maker' or 'trend_follower'.
"""

from __future__ import annotations
import numpy as np

from mfg_lob.models.costs import QuadraticCost, LOBCostTerminal
from mfg_lob.models.hamiltonians import MarketMakerHamiltonian, TrendFollowerHamiltonian


class LOBModel:
    """Full MFG-LOB model specification.

    Example
    -------
    >>> model = LOBModel(sigma=0.5, phi=0.1, alpha=1.0, agent_type='market_maker')
    """

    def __init__(
        self,
        sigma: float = 0.5,
        phi: float = 0.1,
        alpha: float = 1.0,
        A_term: float = 1.0,
        B_term: float = 0.0,
        Lambda0: float = 1.0,
        kappa: float = 1.0,
        epsilon: float = 1.0,
        mu0: float = 0.0,
        sigma0: float = 2.0,
        agent_type: str = "market_maker",
    ) -> None:
        if agent_type not in {"market_maker", "trend_follower"}:
            raise ValueError(f"agent_type must be 'market_maker' or 'trend_follower', got {agent_type!r}")

        self.sigma = float(sigma)
        self.phi = float(phi)
        self.alpha = float(alpha)
        self.A_term = float(A_term)
        self.B_term = float(B_term)
        self.Lambda0 = float(Lambda0)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)
        self.mu0 = float(mu0)
        self.sigma0 = float(sigma0)
        self.agent_type = agent_type

        self._cost = QuadraticCost(phi=phi, alpha=alpha)
        self._terminal = LOBCostTerminal(A=A_term, B=B_term)

        if agent_type == "market_maker":
            self._hamiltonian = MarketMakerHamiltonian(Lambda0=Lambda0, kappa=kappa)
        else:
            self._hamiltonian = TrendFollowerHamiltonian(epsilon=epsilon)

    # ------------------------------------------------------------------
    # Callables required by solvers
    # ------------------------------------------------------------------

    def hamiltonian(self, q: np.ndarray, p: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """H(q, p, m, t)."""
        return self._hamiltonian(q, p, m, t)

    def running_cost(self, q: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """f(q, m, t)."""
        return self._cost(q, m, t)

    def terminal_cost(self, q: np.ndarray) -> np.ndarray:
        """g(q)."""
        return self._terminal(q)

    def optimal_drift(self, q: np.ndarray, u: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """b*(q, t): optimal inventory drift from the HJB costate."""
        p = np.gradient(u, q) if len(q) > 1 else np.zeros_like(q)
        return self._hamiltonian.optimal_drift(q, p, m, t)

    def initial_density(self, q: np.ndarray) -> np.ndarray:
        """m0(q): Gaussian initial inventory distribution."""
        density = np.exp(-0.5 * ((q - self.mu0) / self.sigma0) ** 2)
        density /= np.trapezoid(density, q)
        return density

    def optimal_control(self, q: np.ndarray, p: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
        """a*(q, p, m, t): optimal control (compatible with MFGEquilibrium)."""
        return self._hamiltonian.optimal_drift(q, p, m, t)

    def __repr__(self) -> str:
        return (
            f"LOBModel(agent_type={self.agent_type!r}, sigma={self.sigma}, "
            f"phi={self.phi}, alpha={self.alpha})"
        )
