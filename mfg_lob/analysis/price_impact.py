"""Price impact function from MFG equilibrium.

The price impact function I(v) measures how much a metaorder of volume v
moves the equilibrium mid-price. In the MFG-LOB framework it emerges from
the population density m(q, t): a large buyer absorbs ask-side liquidity,
shifting the balance of inventory across agents.

Empirical LOB price impact follows a concave (square-root) law:
    I(v) ≈ Y * sigma * sqrt(v / V_daily)

The MFG equilibrium predicts this from first principles via the density m.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

from mfg_lob.solvers.mfg_solver import MFGEquilibrium


@dataclass
class PriceImpactCurve:
    """Container for a computed price impact curve.

    Attributes
    ----------
    volumes : ndarray
        Metaorder volumes (in inventory units).
    impact : ndarray
        Expected price impact (in price units).
    participation_rates : ndarray
        Fraction of market volume represented by each metaorder.
    fit_params : dict
        Parameters from the power-law fit I(v) = Y * v^beta.
    """

    volumes: np.ndarray
    impact: np.ndarray
    participation_rates: np.ndarray
    fit_params: dict


class PriceImpactAnalyzer:
    """Compute and analyse the price impact function from an MFG equilibrium.

    The price impact at time t of a metaorder of volume v is approximated by:

        I(v, t) = ∫_0^v  λ(q, t) dq

    where λ(q, t) is the inverse supply density (illiquidity) derived from
    the population density:

        λ(q, t) = 1 / (m(q, t) * dq_effective)

    This is integrated to give cumulative price impact as a function of volume.

    Parameters
    ----------
    equilibrium : MFGEquilibrium
    price_tick : float
        Minimum price increment (tick size).
    """

    def __init__(self, equilibrium: MFGEquilibrium, price_tick: float = 0.01) -> None:
        self.eq = equilibrium
        self.price_tick = price_tick

    def illiquidity(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (q_grid, lambda(q, t)) — inverse density (price per unit volume).

        In the LOB, the density of standing limit orders at price level q is
        proportional to m(q, t). The marginal price impact of consuming
        volume dv at queue position q is therefore:

            dP / dv = 1 / (m(q, t) * normalisation)

        Returns
        -------
        q : ndarray — inventory grid
        lam : ndarray — illiquidity λ(q, t)
        """
        g = self.eq.grid
        t_idx = int(np.searchsorted(g.t, t))
        t_idx = np.clip(t_idx, 0, g.nt - 1)
        m_t = self.eq.m[:, t_idx]
        # Regularise: avoid division by zero at tails
        m_safe = np.maximum(m_t, 1e-10)
        lam = 1.0 / m_safe
        return g.q.copy(), lam

    def compute(
        self,
        t: float = 0.0,
        n_volumes: int = 50,
        v_max: float | None = None,
    ) -> PriceImpactCurve:
        """Compute the price impact curve I(v) at time t.

        Parameters
        ----------
        t : float
            Time at which to evaluate the impact.
        n_volumes : int
            Number of volume levels.
        v_max : float, optional
            Maximum volume. Defaults to half the total inventory range.

        Returns
        -------
        PriceImpactCurve
        """
        g = self.eq.grid
        q, lam = self.illiquidity(t)
        t_idx = int(np.searchsorted(g.t, t))
        t_idx = np.clip(t_idx, 0, g.nt - 1)
        m_t = self.eq.m[:, t_idx]

        total_mass = np.trapezoid(m_t, q)

        if v_max is None:
            v_max = 0.5 * (g.q_max - g.q_min)

        volumes = np.linspace(0.0, v_max, n_volumes)
        impact = np.zeros(n_volumes)
        participation = np.zeros(n_volumes)

        for i, v in enumerate(volumes):
            if v < 1e-10:
                continue
            # Integrate illiquidity from q=0 to q=v (ask side impact)
            # Find the q range corresponding to volume v
            # Cumulative mass from centre outward
            q_ask = q[q >= 0.0]
            m_ask = m_t[q >= 0.0]

            cum_mass = np.concatenate([[0.0], np.cumsum(np.diff(q_ask) * 0.5 *
                                        (m_ask[:-1] + m_ask[1:]))])
            # Impact = price shift needed to attract v units of supply
            if len(cum_mass) > 1 and cum_mass[-1] > 0:
                q_impact = np.interp(
                    min(v, cum_mass[-1] * 0.999),
                    cum_mass,
                    q_ask[:len(cum_mass)],
                )
                impact[i] = q_impact * self.price_tick
            participation[i] = v / (total_mass + 1e-14)

        fit_params = self._fit_power_law(volumes[1:], impact[1:])

        return PriceImpactCurve(
            volumes=volumes,
            impact=impact,
            participation_rates=participation,
            fit_params=fit_params,
        )

    def _fit_power_law(
        self, volumes: np.ndarray, impact: np.ndarray
    ) -> dict:
        """Fit I(v) = Y * v^beta via log-linear regression."""
        mask = (volumes > 0) & (impact > 0)
        if mask.sum() < 2:
            return {"Y": float("nan"), "beta": float("nan"), "r2": float("nan")}
        log_v = np.log(volumes[mask])
        log_I = np.log(impact[mask])
        A = np.column_stack([np.ones_like(log_v), log_v])
        coeffs, res, *_ = np.linalg.lstsq(A, log_I, rcond=None)
        log_Y, beta = coeffs
        Y = np.exp(log_Y)
        # R^2
        ss_res = np.sum((log_I - (log_Y + beta * log_v)) ** 2)
        ss_tot = np.sum((log_I - log_I.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-14)
        return {"Y": float(Y), "beta": float(beta), "r2": float(r2)}

    def plot(self, t: float = 0.0, ax=None, **kwargs):
        """Plot the price impact curve."""
        import matplotlib.pyplot as plt
        curve = self.compute(t=t)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(curve.volumes, curve.impact, **kwargs)
        ax.set_xlabel("Metaorder volume")
        ax.set_ylabel("Price impact")
        ax.set_title(f"MFG price impact at t={t:.2f}")
        params = curve.fit_params
        if not np.isnan(params["beta"]):
            ax.annotate(
                f"β = {params['beta']:.3f}  (R²={params['r2']:.3f})",
                xy=(0.05, 0.85),
                xycoords="axes fraction",
            )
        return ax
