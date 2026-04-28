"""Empirical LOB data structures and moment extraction.

This module provides tools for loading real or synthetic LOB snapshots
and computing the summary statistics needed for MFG calibration:
  - Volume profile at each price level
  - Bid-ask spread
  - Price impact function (empirical)
  - Order flow imbalance
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class LOBSnapshot:
    """Single LOB snapshot from empirical data.

    Parameters
    ----------
    bid_prices : array-like, shape (n_bid,)
        Bid price levels (descending from best bid).
    bid_volumes : array-like, shape (n_bid,)
        Standing bid volume at each price level.
    ask_prices : array-like, shape (n_ask,)
        Ask price levels (ascending from best ask).
    ask_volumes : array-like, shape (n_ask,)
        Standing ask volume at each price level.
    mid_price : float
        Mid-price = (best_bid + best_ask) / 2.
    timestamp : float, optional
    """

    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_prices: np.ndarray
    ask_volumes: np.ndarray
    mid_price: float
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.bid_prices = np.asarray(self.bid_prices, dtype=float)
        self.bid_volumes = np.asarray(self.bid_volumes, dtype=float)
        self.ask_prices = np.asarray(self.ask_prices, dtype=float)
        self.ask_volumes = np.asarray(self.ask_volumes, dtype=float)

    @property
    def best_bid(self) -> float:
        return float(self.bid_prices[0]) if len(self.bid_prices) > 0 else self.mid_price

    @property
    def best_ask(self) -> float:
        return float(self.ask_prices[0]) if len(self.ask_prices) > 0 else self.mid_price

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def imbalance(self) -> float:
        """Order flow imbalance: (V_bid - V_ask) / (V_bid + V_ask)."""
        vb = self.bid_volumes.sum()
        va = self.ask_volumes.sum()
        return (vb - va) / (vb + va + 1e-14)

    def depth_profile(self, n_levels: int = 10, tick_size: float = 0.01) -> dict:
        """Return cumulative depth profile normalised to mid-price distances."""
        levels = np.arange(1, n_levels + 1) * tick_size
        bid_cum = np.array([
            self.bid_volumes[self.bid_prices >= self.mid_price - lev].sum()
            for lev in levels
        ])
        ask_cum = np.array([
            self.ask_volumes[self.ask_prices <= self.mid_price + lev].sum()
            for lev in levels
        ])
        return {"levels": levels, "bid_cum": bid_cum, "ask_cum": ask_cum}

    @classmethod
    def synthetic(
        cls,
        mid_price: float = 100.0,
        spread: float = 0.02,
        depth_shape: str = "power_law",
        n_levels: int = 20,
        tick_size: float = 0.01,
        total_volume: float = 1000.0,
        alpha: float = 0.6,
        rng: np.random.Generator | None = None,
    ) -> "LOBSnapshot":
        """Generate a synthetic LOB snapshot.

        Parameters
        ----------
        depth_shape : 'power_law' | 'gaussian' | 'uniform'
            Shape of the LOB depth profile.
        alpha : float
            Power-law exponent for depth at level k: V_k ∝ k^{-alpha}.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        levels = np.arange(1, n_levels + 1)

        if depth_shape == "power_law":
            weights = levels.astype(float) ** (-alpha)
        elif depth_shape == "gaussian":
            weights = np.exp(-0.5 * ((levels - 3.0) / 2.0) ** 2)
        else:
            weights = np.ones(n_levels, dtype=float)

        weights /= weights.sum()
        vol = total_volume * weights

        noise = 1.0 + 0.05 * rng.standard_normal(n_levels)
        bid_vol = vol * np.maximum(noise, 0.1)
        ask_vol = vol * np.maximum(1.0 + 0.05 * rng.standard_normal(n_levels), 0.1)

        half_spread = spread / 2.0
        bid_prices = mid_price - half_spread - (levels - 1) * tick_size
        ask_prices = mid_price + half_spread + (levels - 1) * tick_size

        return cls(
            bid_prices=bid_prices,
            bid_volumes=bid_vol,
            ask_prices=ask_prices,
            ask_volumes=ask_vol,
            mid_price=mid_price,
        )


class EmpiricalLOB:
    """Collection of LOB snapshots and aggregate statistics.

    Parameters
    ----------
    snapshots : list of LOBSnapshot
    """

    def __init__(self, snapshots: Sequence[LOBSnapshot]) -> None:
        self.snapshots = list(snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def mean_spread(self) -> float:
        return float(np.mean([s.spread for s in self.snapshots]))

    @property
    def mean_imbalance(self) -> float:
        return float(np.mean([s.imbalance for s in self.snapshots]))

    def mean_depth_profile(
        self, n_levels: int = 10, tick_size: float = 0.01
    ) -> dict:
        """Average depth profile across all snapshots."""
        profiles = [s.depth_profile(n_levels, tick_size) for s in self.snapshots]
        levels = profiles[0]["levels"]
        bid_cum = np.mean([p["bid_cum"] for p in profiles], axis=0)
        ask_cum = np.mean([p["ask_cum"] for p in profiles], axis=0)
        return {"levels": levels, "bid_cum": bid_cum, "ask_cum": ask_cum}

    def empirical_impact(
        self,
        volumes: np.ndarray,
        tick_size: float = 0.01,
        n_levels: int = 20,
    ) -> np.ndarray:
        """Estimate empirical price impact for given metaorder volumes.

        Uses average cumulative depth to infer the price level that would
        be reached by a metaorder of each volume size.
        """
        prof = self.mean_depth_profile(n_levels=n_levels, tick_size=tick_size)
        levels = prof["levels"]
        ask_cum = prof["ask_cum"]

        impact = np.zeros_like(volumes, dtype=float)
        for i, v in enumerate(volumes):
            if v <= 0:
                continue
            impact[i] = np.interp(v, ask_cum, levels, right=levels[-1])

        return impact

    @classmethod
    def from_synthetic(
        cls,
        n_snapshots: int = 100,
        mid_price: float = 100.0,
        spread_mean: float = 0.02,
        spread_std: float = 0.003,
        total_volume: float = 1000.0,
        alpha: float = 0.6,
        n_levels: int = 20,
        tick_size: float = 0.01,
        seed: int = 0,
    ) -> "EmpiricalLOB":
        """Generate a collection of synthetic LOB snapshots."""
        rng = np.random.default_rng(seed)
        snapshots = []
        for i in range(n_snapshots):
            spread = max(tick_size, rng.normal(spread_mean, spread_std))
            snap = LOBSnapshot.synthetic(
                mid_price=mid_price,
                spread=spread,
                total_volume=total_volume * (1 + 0.1 * rng.standard_normal()),
                alpha=alpha,
                n_levels=n_levels,
                tick_size=tick_size,
                rng=rng,
            )
            snapshots.append(snap)
        return cls(snapshots)
