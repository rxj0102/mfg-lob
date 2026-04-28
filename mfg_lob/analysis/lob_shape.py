"""LOB shape analysis from MFG equilibrium density.

The equilibrium population density m(q, t) characterises the shape of the
limit order book. At any time t the density m(·, t) represents the
distribution of agent inventory levels, which in turn determines the
available liquidity at each price level.

Specifically, the LOB volume density at price level p is proportional to
m(q(p), t) where q ↔ p via the optimal quoting strategy.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from mfg_lob.solvers.mfg_solver import MFGEquilibrium


@dataclass
class LOBSnapshot:
    """LOB shape at a single instant.

    Attributes
    ----------
    q : ndarray
        Inventory grid.
    density : ndarray
        m(q, t) — population density (proportional to LOB volume).
    bid_depth : ndarray
        Available liquidity on the bid side (q < 0).
    ask_depth : ndarray
        Available liquidity on the ask side (q > 0).
    spread_proxy : float
        Effective spread as 2 * q* where q* is the median of the
        inventory distribution.
    """

    q: np.ndarray
    density: np.ndarray
    bid_depth: np.ndarray
    ask_depth: np.ndarray
    spread_proxy: float
    t: float


class LOBShapeAnalyzer:
    """Extract LOB shape statistics from an MFG equilibrium.

    Parameters
    ----------
    equilibrium : MFGEquilibrium
    """

    def __init__(self, equilibrium: MFGEquilibrium) -> None:
        self.eq = equilibrium

    def snapshot(self, t: float) -> LOBSnapshot:
        """Return a LOB shape snapshot at time t."""
        g = self.eq.grid
        t_idx = int(np.searchsorted(g.t, t))
        t_idx = np.clip(t_idx, 0, g.nt - 1)
        m_t = self.eq.m[:, t_idx].copy()

        bid_mask = g.q <= 0.0
        ask_mask = g.q >= 0.0

        bid_depth = m_t * bid_mask
        ask_depth = m_t * ask_mask

        # Spread proxy: distance between bid and ask 'best' (highest density)
        # sides weighted median
        cdf = np.cumsum(m_t * g.dq)
        cdf /= cdf[-1] + 1e-14
        median_q = np.interp(0.5, cdf, g.q)
        spread_proxy = float(2.0 * abs(median_q))

        return LOBSnapshot(
            q=g.q.copy(),
            density=m_t,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            spread_proxy=spread_proxy,
            t=float(g.t[t_idx]),
        )

    def depth_profile(self, t: float, n_levels: int = 10) -> dict:
        """Return LOB depth at discrete price levels away from mid.

        Returns
        -------
        dict with keys 'levels', 'bid_volume', 'ask_volume'
        """
        snap = self.snapshot(t)
        g = self.eq.grid
        max_depth = min(n_levels * g.dq, (g.q_max - g.q_min) / 2.0)
        levels = np.linspace(g.dq, max_depth, n_levels)

        bid_vol = np.array([
            np.interp(-lev, g.q, snap.bid_depth) for lev in levels
        ])
        ask_vol = np.array([
            np.interp(lev, g.q, snap.ask_depth) for lev in levels
        ])
        return {"levels": levels, "bid_volume": bid_vol, "ask_volume": ask_vol}

    def time_evolution(self, n_snapshots: int = 5) -> list[LOBSnapshot]:
        """Return LOB snapshots at evenly-spaced times."""
        times = np.linspace(self.eq.grid.t_min, self.eq.grid.t_max, n_snapshots)
        return [self.snapshot(t) for t in times]

    def plot_density_evolution(self, n_snapshots: int = 5, ax=None):
        """Plot the evolution of the population density over time."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        snapshots = self.time_evolution(n_snapshots)
        cmap = plt.get_cmap("viridis")
        for i, snap in enumerate(snapshots):
            color = cmap(i / max(n_snapshots - 1, 1))
            ax.plot(snap.q, snap.density, color=color, label=f"t={snap.t:.2f}")
        ax.set_xlabel("Inventory q")
        ax.set_ylabel("Density m(q, t)")
        ax.set_title("Population density evolution (= LOB shape)")
        ax.legend(fontsize=8)
        return ax

    def plot_depth_profile(self, t: float = 0.0, ax=None):
        """Plot bid/ask depth as a bar chart (traditional LOB view)."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        prof = self.depth_profile(t)
        ax.bar(-prof["levels"], prof["bid_volume"], width=self.eq.grid.dq * 0.8,
               color="steelblue", alpha=0.7, label="Bid")
        ax.bar(prof["levels"], prof["ask_volume"], width=self.eq.grid.dq * 0.8,
               color="tomato", alpha=0.7, label="Ask")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Price distance from mid")
        ax.set_ylabel("Volume density")
        ax.set_title(f"LOB depth profile at t={t:.2f}")
        ax.legend()
        return ax
