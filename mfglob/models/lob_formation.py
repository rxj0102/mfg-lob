"""LOB shape formation MFG (Lasry-Lions type).

Models how the limit order book depth profile emerges as the Nash equilibrium
of strategic agents choosing where to post limit orders.

Reference:
  Lasry & Lions (2007), "Mean field games", Japanese Journal of Mathematics.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mfglob.models.base import MFGLOBModel
from mfglob.grids import Grid1D


class LOBFormationMFG(MFGLOBModel):
    """Lasry-Lions MFG model for limit order book shape formation.

    State:   x ∈ [0, x_max] — distance of limit order from mid-price (ticks)
    Control: α ∈ R          — drift (rate of repositioning)

    Dynamics:   dx = α dt + σ dW

    σ dW captures random mid-price movements that push all orders.

    Running cost:
        f(x, α, m) = c_far · x              (opportunity cost of distance)
                   + c_near · exp(-κ x)      (adverse selection near mid)
                   + c_crowd · m(x)          (priority loss from crowding)
                   + c_control · α²          (repositioning cost)

    The equilibrium density m*(x) gives the steady-state LOB profile.

    Parameters
    ----------
    c_far     : opportunity cost of distance from mid
    c_near    : adverse selection cost coefficient (near mid)
    kappa     : decay rate of adverse selection
    c_crowd   : congestion / priority cost
    c_control : cost of repositioning (control cost)
    sigma     : mid-price volatility
    x_max     : maximum distance from mid
    """

    def __init__(
        self,
        c_far: float = 0.1,
        c_near: float = 1.0,
        kappa: float = 2.0,
        c_crowd: float = 0.5,
        c_control: float = 0.1,
        sigma: float = 0.5,
        x_max: float = 10.0,
    ) -> None:
        self.c_far = float(c_far)
        self.c_near = float(c_near)
        self.kappa = float(kappa)
        self.c_crowd = float(c_crowd)
        self.c_control = float(c_control)
        self.sigma = float(sigma)
        self.x_max = float(x_max)

    # ------------------------------------------------------------------
    # MFGLOBModel interface
    # ------------------------------------------------------------------

    def hamiltonian(self, x, p, M, m):
        # H = sup_α {-f - b·p - σ²/2·M}
        # f = c_far·x + c_near·exp(-κx) + c_crowd·m + c_control·α²
        # b = α, so -f - α·p = -(c_far·x + c_near·e^{-κx} + c_crowd·m)
        #                       - c_control·α² - α·p
        # FOC: -2·c_control·α - p = 0  →  α* = -p/(2·c_control)
        # H = -(c_far·x + c_near·e^{-κx} + c_crowd·m)
        #     + p²/(4·c_control) - σ²/2·M
        adv_sel = self.c_near * np.exp(-self.kappa * x)
        base_cost = self.c_far * x + adv_sel + self.c_crowd * m
        return (
            -base_cost
            + p ** 2 / (4.0 * self.c_control)
            - 0.5 * self.sigma ** 2 * M
        )

    def optimal_control(self, x, p, m):
        # α* = -p / (2·c_control)
        return -p / (2.0 * self.c_control)

    def drift(self, x, alpha, m):
        return alpha.copy()

    def diffusion(self, x, alpha, m):
        return np.full_like(x, self.sigma)

    def running_cost(self, x, alpha, m):
        adv_sel = self.c_near * np.exp(-self.kappa * x)
        return (
            self.c_far * x
            + adv_sel
            + self.c_crowd * m
            + self.c_control * alpha ** 2
        )

    def terminal_cost(self, x, m):
        return np.zeros_like(x)

    def initial_distribution(self, x):
        # Uniform as starting point; solver will evolve to equilibrium
        density = np.ones_like(x)
        mass = np.trapezoid(density, x)
        return density / mass

    # ------------------------------------------------------------------
    # Stationary MFG
    # ------------------------------------------------------------------

    def steady_state_density(self, grid: Grid1D) -> np.ndarray:
        """Compute the stationary LOB-shape density via fixed-point iteration.

        Solves the ergodic (stationary) MFG system:
            λ + H(x, Du, D²u, m) = 0        (ergodic HJB)
            -(σ²/2)·m'' + (α*·m)' = 0       (stationary FP)

        Strategy: iterate between
          1. Given m, solve the stationary FP for m_new (using current α*)
          2. Update m ← damp·m_new + (1-damp)·m

        The stationary FP with constant diffusion and drift α = -u'/(2c) can
        be written as a Fokker-Planck ODE. We solve it by noting that at
        stationarity:
            J = α·m - (σ²/2)·m'  = const

        With reflecting BCs (J=0), the stationary density satisfies:
            m' = (2 α / σ²) · m
            m(x) ∝ exp(2/σ² · ∫₀ˣ α(s) ds)

        Because α depends on u (the value function) which itself depends on m,
        we iterate. For simplicity we use a fixed drift profile based on the
        cost structure alone (ignoring the value function gradient), producing
        an approximate but physically meaningful density.
        """
        x = grid.points
        dx = float(grid.dx)
        n = grid.n
        sigma_sq = self.sigma ** 2

        # The stationary FP with reflecting BCs has solution m ∝ exp(-u/D)
        # where D = σ²/(2c_control) and u is the value function.
        # Key: u'(x) ≈ c_far - c_near·κ·exp(-κx) + c_crowd·m'(x)
        # Near x=0: c_near·κ is large → u' < 0 → α* = -u'/(2c) > 0 (drift away from 0)
        # For large x: c_far dominates → u' > 0 → α* < 0 (drift toward mid)
        # This creates a peak at x* = log(c_near·κ/c_far)/κ

        # Zeroth-order approximation (no congestion):
        # log m ∝ -(2/σ²) ∫₀ˣ α_eff ds  with α_eff = -(c_far - c_near·κ·e^{-κs})/(2c)
        # ∫₀ˣ α_eff ds = -(c_far·x + c_near·(exp(-κx)-1))/(2c)
        # → log m ∝ (c_far·x + c_near·(exp(-κx)-1))/(c_control·σ²)
        integral = (self.c_far * x
                    + self.c_near * (np.exp(-self.kappa * x) - 1.0)) / (
                       self.c_control * sigma_sq)
        log_m = integral - integral.max()
        m = np.exp(log_m)
        m /= np.trapezoid(m, x)

        # Fixed-point iteration to include congestion: c_crowd·m raises cost
        damping = 0.5
        for _ in range(200):
            # Gradient of running cost (first derivative w.r.t. x)
            # ∂f/∂x = c_far - c_near·κ·exp(-κx) + c_crowd·m'
            dm_dx = np.gradient(m, x)
            cost_grad = (self.c_far
                         - self.c_near * self.kappa * np.exp(-self.kappa * x)
                         + self.c_crowd * dm_dx)
            # α_eff = -u'/(2c_control) ≈ -cost_grad/(2c_control)
            alpha_eff = -cost_grad / (2.0 * self.c_control)

            # Rebuild log-density by integrating α_eff
            log_m_new = np.zeros(n)
            for i in range(1, n):
                log_m_new[i] = log_m_new[i - 1] + (2.0 * dx / sigma_sq) * alpha_eff[i - 1]
            log_m_new -= log_m_new.max()
            m_new = np.exp(log_m_new)
            mass = np.trapezoid(m_new, x)
            if mass > 1e-14:
                m_new /= mass

            m = damping * m_new + (1.0 - damping) * m
            mass = np.trapezoid(m, x)
            if mass > 1e-14:
                m /= mass

        return m

    @property
    def state_dimension(self) -> int:
        return 1

    @property
    def param_dict(self) -> dict:
        return {
            "c_far": self.c_far,
            "c_near": self.c_near,
            "kappa": self.kappa,
            "c_crowd": self.c_crowd,
            "c_control": self.c_control,
            "sigma": self.sigma,
            "x_max": self.x_max,
        }
