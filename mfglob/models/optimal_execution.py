"""Optimal execution MFG model (Cardaliaguet-Lehalle type).

Reference:
  Cardaliaguet & Lehalle (2018), "Mean field game of controls and an
  application to trade crowding", Mathematics and Financial Economics 12.
"""

from __future__ import annotations
import numpy as np
from mfglob.models.base import MFGLOBModel
from mfglob.grids import Grid1D


class OptimalExecutionMFG(MFGLOBModel):
    """MFG model for optimal execution with permanent price impact.

    State:   q ∈ [0, Q0] — remaining inventory
    Control: ν ≥ 0       — execution rate (shares per unit time)

    Dynamics:   dq = -ν dt + σ dW

    The σ dW term captures the stochasticity in execution (e.g. randomness
    in fill rates).  Set sigma=0 for the deterministic model.

    Running cost:
        f(q, ν, m) = φ q² + ψ ν² + λ η M(t) ν

    where M(t) = ∫ ν(t,q) m(t,q) dq is the aggregate execution rate
    (mean-field coupling through permanent price impact).

    Optimal control (from FOC):
        ν* = max(0, (−∂u/∂q − λ η M(t)) / (2ψ))
           = max(0, (−p − λ η M(t)) / (2ψ))

    Because M depends on m and ν* (which in turn depends on p), the
    mean-field interaction is encoded via the running cost / Hamiltonian
    evaluated at the current population m.

    For PDE purposes the aggregate rate M is computed once per Picard
    iteration from the current (u, m) pair.  We approximate:
        M(t) ≈ ∫ ν*(t,q,m) m(t,q) dq

    In the single-agent limit (eta=0, lam=0) the model reduces to
    Almgren-Chriss:
        ν*(t) = Q0 · κ · cosh(κ(T-t)) / sinh(κT),   κ = √(φ/ψ)

    Parameters
    ----------
    phi   : inventory holding cost
    psi   : execution cost (temporary price impact)
    eta   : permanent price impact coefficient
    lam   : price sensitivity
    sigma : execution randomness (set 0 for deterministic)
    Q0    : initial inventory (used for initial distribution)
    """

    def __init__(
        self,
        phi: float = 0.001,
        psi: float = 0.01,
        eta: float = 0.1,
        lam: float = 1.0,
        sigma: float = 0.0,
        Q0: float = 1.0,
    ) -> None:
        self.phi = float(phi)
        self.psi = float(psi)
        self.eta = float(eta)
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.Q0 = float(Q0)

    # ------------------------------------------------------------------
    # MFGLOBModel interface
    # ------------------------------------------------------------------

    def hamiltonian(self, x, p, M, m):
        # Aggregate execution rate at current population
        # (approx from current m and optimal control)
        nu_star = self._nu_star(p, m, x)
        # H = sup_ν {-f - b·p - σ²/2·M}
        #   = sup_ν {-φq² - ψν² - λη·M_agg·ν + ν·p} - σ²/2·M
        # FOC at ν*: 2ψν + λη·M_agg - p = 0  →  ν* = max(0,(p-λη·M_agg)/(2ψ))
        # H = -φq² - ψν*² - λη·M_agg·ν* + ν*·p - σ²/2·M
        M_agg = self._aggregate_rate(nu_star, m, x)
        H = (
            -self.phi * x ** 2
            - self.psi * nu_star ** 2
            - self.lam * self.eta * M_agg * nu_star
            + nu_star * p
            - 0.5 * self.sigma ** 2 * M
        )
        return H

    def optimal_control(self, x, p, m):
        return self._nu_star(p, m, x)

    def drift(self, x, alpha, m):
        # q decreases at rate ν
        return -alpha

    def diffusion(self, x, alpha, m):
        return np.full_like(x, self.sigma)

    def running_cost(self, x, alpha, m):
        M_agg = self._aggregate_rate(alpha, m, x)
        return (
            self.phi * x ** 2
            + self.psi * alpha ** 2
            + self.lam * self.eta * M_agg * alpha
        )

    def terminal_cost(self, x, m):
        # Large penalty for unexecuted inventory
        return x ** 2

    def initial_distribution(self, x):
        # All agents start concentrated near Q0
        s = max(0.05 * self.Q0, 1e-4)
        density = np.exp(-0.5 * ((x - self.Q0) / s) ** 2)
        mass = np.trapezoid(density, x)
        if mass < 1e-14:
            density = np.ones_like(x)
            mass = np.trapezoid(density, x)
        return density / mass

    # ------------------------------------------------------------------
    # Model-specific helpers
    # ------------------------------------------------------------------

    def aggregate_execution_rate(
        self, nu: np.ndarray, m: np.ndarray, grid: Grid1D
    ) -> float:
        """M(t) = ∫ ν(t, q) m(t, q) dq  via trapezoidal rule."""
        return float(np.trapezoid(nu * m, grid.points))

    def _aggregate_rate(self, nu: np.ndarray, m: np.ndarray, x: np.ndarray) -> float:
        return float(np.trapezoid(nu * m, x))

    def _nu_star(self, p: np.ndarray, m: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ν* = max(0, (−p − λ η M_agg) / (2ψ)).

        Note: p = ∂u/∂q. Since inventory decreases when executing,
        b = -ν, so the HJB gradient term is -(-ν)·p = ν·p.
        FOC: -2ψν + p - λη·M_agg = 0  →  ν* = max(0, (p - λη·M_agg)/(2ψ))
        """
        # Initial estimate ignoring M_agg (self-consistent fixed point)
        nu_0 = np.maximum(p / (2.0 * self.psi + 1e-14), 0.0)
        M_agg = float(np.trapezoid(nu_0 * m, x))
        shift = self.lam * self.eta * M_agg
        return np.maximum((p - shift) / (2.0 * self.psi + 1e-14), 0.0)

    @property
    def state_dimension(self) -> int:
        return 1

    @property
    def param_dict(self) -> dict:
        return {
            "phi": self.phi,
            "psi": self.psi,
            "eta": self.eta,
            "lam": self.lam,
            "sigma": self.sigma,
            "Q0": self.Q0,
        }
