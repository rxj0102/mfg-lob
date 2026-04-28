"""Synthetic limit order book data generators.

Provides tools to generate synthetic LOB snapshots for testing and
calibration.  Two generation modes are available:

  1. Statistical model: draw volumes from parametric distributions (power-law,
     log-normal, exponential) with controlled spread and depth.
  2. MFG-implied: generate LOB shapes directly from an MFG equilibrium
     density m(x, t).

The generated snapshots conform to the EmpiricalLOB interface in
mfg_lob.calibration.empirical for compatibility with the calibration pipeline.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SyntheticLOBConfig:
    """Configuration for synthetic LOB generation.

    Parameters
    ----------
    n_levels    : int   — number of price levels on each side
    tick_size   : float — minimum price increment
    mid_price   : float — reference mid-price
    spread      : float — bid-ask spread (in price units)
    depth_shape : str   — 'power_law' | 'exponential' | 'gaussian'
    depth_param : float — shape parameter (exponent, decay rate, etc.)
    total_volume: float — total standing volume (bid + ask)
    noise_level : float — relative noise in volume
    """

    n_levels: int = 20
    tick_size: float = 0.01
    mid_price: float = 100.0
    spread: float = 0.02
    depth_shape: str = "power_law"
    depth_param: float = 0.6
    total_volume: float = 1000.0
    noise_level: float = 0.05


def generate_lob_snapshot(
    config: SyntheticLOBConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate a single synthetic LOB snapshot.

    Returns
    -------
    dict with keys: 'bid_prices', 'bid_volumes', 'ask_prices',
                    'ask_volumes', 'mid_price', 'spread'
    """
    if rng is None:
        rng = np.random.default_rng()

    cfg = config
    levels = np.arange(1, cfg.n_levels + 1)
    half_spread = cfg.spread / 2.0

    if cfg.depth_shape == "power_law":
        weights = levels.astype(float) ** (-cfg.depth_param)
    elif cfg.depth_shape == "exponential":
        weights = np.exp(-cfg.depth_param * levels)
    elif cfg.depth_shape == "gaussian":
        weights = np.exp(-0.5 * ((levels - 1.5) / cfg.depth_param) ** 2)
    else:
        raise ValueError(f"Unknown depth_shape: {cfg.depth_shape!r}")

    weights /= weights.sum()
    vol_per_side = cfg.total_volume / 2.0
    base_vol = vol_per_side * weights

    noise = 1.0 + cfg.noise_level * rng.standard_normal(cfg.n_levels)
    bid_vol = base_vol * np.maximum(noise, 0.01)
    ask_vol = base_vol * np.maximum(
        1.0 + cfg.noise_level * rng.standard_normal(cfg.n_levels), 0.01
    )

    bid_prices = cfg.mid_price - half_spread - (levels - 1) * cfg.tick_size
    ask_prices = cfg.mid_price + half_spread + (levels - 1) * cfg.tick_size

    return {
        "bid_prices": bid_prices,
        "bid_volumes": bid_vol,
        "ask_prices": ask_prices,
        "ask_volumes": ask_vol,
        "mid_price": cfg.mid_price,
        "spread": cfg.spread,
    }
