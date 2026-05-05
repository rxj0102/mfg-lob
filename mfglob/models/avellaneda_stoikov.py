"""Avellaneda-Stoikov market-making MFG model.

Reference:
  Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book",
  Quantitative Finance 8(3), 217-224.
"""

from __future__ import annotations
import numpy as np
from mfglob.models.base import MFGLOBModel


class AvellanedaStoikovMFG(MFGLOBModel):
    """MFG extension of the Avellaneda-Stoikov market-maker model.

    State:   q ∈ [-Q_max, Q_max] — inventory
    Control: (δ^a, δ^b) encoded as a scalar α = δ^a - δ^b (spread asymmetry)

    Diffusion approximation of Poisson jump dynamics:
        dq ≈ (Λ^b - Λ^a) dt + √(Λ^b + Λ^a) dW

    With symmetric base spreads the effective single scalar control is the
    indifference-price shift p_I = -γ σ² q / 2, which determines the spread:
        δ^a* = 1/k + γ σ² q / 2
        δ^b* = 1/k - γ σ² q / 2

    In the diffusion limit the representative agent's state evolves as:
        dq = α dt + σ_eff dW
    where α = Λ^b - Λ^a and σ_eff = √(Λ^b + Λ^a).

    For the MFG PDE we use a single-control representation:
        control α (scalar) = net inventory drift chosen by the agent
        b(q, α, m) = α
        σ(q, α, m) = sigma_mid  (volatility from mid-price moves)

    Running cost encodes inventory penalty and mean-field crowd aversion:
        f(q, α, m) = φ q² + ψ m(q) q² + (1/2) α²

    The quadratic control cost (1/2)α² replaces the explicit spread
    optimisation so the problem fits the standard MFG PDE template.

    Parameters
    ----------
    A         : baseline market-order arrival rate
    k         : price sensitivity (arrival decay per unit spread)
    phi       : inventory penalty coefficient
    psi       : crowd-aversion coefficient (mean-field coupling)
    gamma     : terminal liquidation penalty
    sigma_mid : mid-price volatility
    Q_max     : maximum inventory (domain boundary)
    c         : mean-field intensity for execution rates
    """

    def __init__(
        self,
        A: float = 1.0,
        k: float = 1.5,
        phi: float = 0.01,
        psi: float = 0.005,
        gamma: float = 0.1,
        sigma_mid: float = 0.3,
        Q_max: float = 10.0,
        c: float = 0.5,
    ) -> None:
        self.A = float(A)
        self.k = float(k)
        self.phi = float(phi)
        self.psi = float(psi)
        self.gamma = float(gamma)
        self.sigma_mid = float(sigma_mid)
        self.Q_max = float(Q_max)
        self.c = float(c)

    # ------------------------------------------------------------------
    # MFGLOBModel interface
    # ------------------------------------------------------------------

    def hamiltonian(self, x, p, M, m):
        # H = sup_α { -f - b·p - σ²/2·M }
        # With b=α, optimising over α: -f(α) - α·p = -φq² - ψmq² - α²/2 - αp
        # FOC: -α - p = 0  →  α* = -p
        # H = -φq² - ψmq² - (-p)²/2 - (-p)p - σ²/2·M
        #   = -φq² - ψmq² + p²/2 - σ²/2·M
        sigma_sq = self.sigma_mid ** 2
        mfg_term = self.psi * m * x ** 2
        return 0.5 * p ** 2 - self.phi * x ** 2 - mfg_term - 0.5 * sigma_sq * M

    def optimal_control(self, x, p, m):
        # α* = -p  (from FOC of Hamiltonian)
        return -p.copy()

    def drift(self, x, alpha, m):
        return alpha.copy()

    def diffusion(self, x, alpha, m):
        return np.full_like(x, self.sigma_mid)

    def running_cost(self, x, alpha, m):
        return self.phi * x ** 2 + self.psi * m * x ** 2 + 0.5 * alpha ** 2

    def terminal_cost(self, x, m):
        return self.gamma * x ** 2

    def initial_distribution(self, x):
        # Gaussian centred at q=0 with std ≈ Q_max/4
        s = max(self.Q_max / 4.0, 1e-6)
        density = np.exp(-0.5 * (x / s) ** 2)
        mass = np.trapezoid(density, x)
        return density / mass

    # ------------------------------------------------------------------
    # Model-specific helpers
    # ------------------------------------------------------------------

    def execution_rate_ask(self, delta_a, m, q):
        """Λ^a(δ^a, m) = A · exp(-k·δ^a) · (1 + c·m(q))"""
        return self.A * np.exp(-self.k * delta_a) * (1.0 + self.c * m)

    def execution_rate_bid(self, delta_b, m, q):
        """Λ^b(δ^b, m) = A · exp(-k·δ^b) · (1 + c·m(-q))

        For a 1-D grid the crowd effect at -q is approximated by
        interpolating m at the reflected position; on a symmetric grid
        the reflected index is (n-1-i).
        """
        m_reflected = m[::-1]  # m(-q) on a symmetric grid
        return self.A * np.exp(-self.k * delta_b) * (1.0 + self.c * m_reflected)

    def optimal_spread(self, q, p, m):
        """Optimal bid and ask half-spreads.

        In the diffusion approximation the indifference price gives:
            δ^a* = 1/k + γ σ² q / 2
            δ^b* = 1/k - γ σ² q / 2

        Spreads are clamped to [0, ∞) (no negative spreads).

        Returns
        -------
        (delta_a, delta_b) : tuple of (n,) arrays
        """
        base = 1.0 / self.k
        skew = 0.5 * self.gamma * self.sigma_mid ** 2 * q
        delta_a = np.maximum(base + skew, 0.0)
        delta_b = np.maximum(base - skew, 0.0)
        return delta_a, delta_b

    @property
    def state_dimension(self) -> int:
        return 1

    @property
    def param_dict(self) -> dict:
        return {
            "A": self.A,
            "k": self.k,
            "phi": self.phi,
            "psi": self.psi,
            "gamma": self.gamma,
            "sigma_mid": self.sigma_mid,
            "Q_max": self.Q_max,
            "c": self.c,
        }
