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
import pandas as pd
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


# ---------------------------------------------------------------------------
# Additional synthetic generators
# ---------------------------------------------------------------------------


def generate_synthetic_book_shape(
    n_levels: int = 50,
    shape: str = "exponential",
    seed: int | None = None,
) -> dict:
    """Generate a synthetic LOB book shape.

    Parameters
    ----------
    n_levels : number of price levels (ticks from mid)
    shape    : 'exponential', 'hump', 'uniform', 'power_law'
    seed     : random seed for reproducibility

    Returns
    -------
    dict with keys: 'price_levels', 'volumes_bid', 'volumes_ask',
                    'mid_price', 'spread'
    """
    rng = np.random.default_rng(seed)
    price_levels = np.arange(1, n_levels + 1, dtype=float)

    if shape == "exponential":
        A, k = 100.0, 0.1
        weights = A * np.exp(-k * price_levels)
    elif shape == "hump":
        A, k = 100.0, 0.08
        weights = A * price_levels * np.exp(-k * price_levels)
    elif shape == "uniform":
        weights = np.ones(n_levels) * 50.0
    elif shape == "power_law":
        A, alpha = 100.0, 1.0
        weights = A * price_levels ** (-alpha)
    else:
        raise ValueError(f"Unknown shape: {shape!r}. "
                         "Choose 'exponential', 'hump', 'uniform', 'power_law'.")

    # Add small noise
    noise_bid = 1.0 + 0.05 * rng.standard_normal(n_levels)
    noise_ask = 1.0 + 0.05 * rng.standard_normal(n_levels)
    volumes_bid = np.maximum(weights * noise_bid, 0.0)
    volumes_ask = np.maximum(weights * noise_ask, 0.0)

    mid_price = 100.0
    spread = 0.02

    return {
        "price_levels": price_levels,
        "volumes_bid": volumes_bid,
        "volumes_ask": volumes_ask,
        "mid_price": mid_price,
        "spread": spread,
    }


def generate_synthetic_impact_data(
    n_trades: int = 1000,
    impact_law: str = "sqrt",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic trade impact data.

    Parameters
    ----------
    n_trades   : number of trade observations
    impact_law : 'sqrt', 'linear', 'concave_power'
    seed       : random seed

    Returns
    -------
    DataFrame with columns: trade_size, price_impact, volatility, volume
    """
    rng = np.random.default_rng(seed)

    trade_size = rng.exponential(scale=100.0, size=n_trades)
    volatility = rng.uniform(0.01, 0.05, size=n_trades)
    volume = rng.uniform(500.0, 5000.0, size=n_trades)
    noise = rng.standard_normal(n_trades) * 0.01

    if impact_law == "sqrt":
        # impact = 0.1 * sigma * sqrt(Q/V) + noise
        price_impact = 0.1 * volatility * np.sqrt(trade_size / volume) + noise
    elif impact_law == "linear":
        price_impact = 0.01 * trade_size + noise
    elif impact_law == "concave_power":
        price_impact = 0.1 * trade_size ** 0.6 + noise
    else:
        raise ValueError(f"Unknown impact_law: {impact_law!r}. "
                         "Choose 'sqrt', 'linear', 'concave_power'.")

    # Ensure positive impact for positive trade sizes
    price_impact = np.abs(price_impact)

    return pd.DataFrame({
        "trade_size": trade_size,
        "price_impact": price_impact,
        "volatility": volatility,
        "volume": volume,
    })


def generate_synthetic_inventory_data(
    n_market_makers: int = 50,
    n_timestamps: int = 1000,
    mean_reversion: float = 0.1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate market-maker inventory paths via Ornstein-Uhlenbeck processes.

    dq = -kappa * q * dt + sigma * dW

    Parameters
    ----------
    n_market_makers : number of agents
    n_timestamps    : number of time steps
    mean_reversion  : kappa (mean-reversion speed)
    seed            : random seed

    Returns
    -------
    DataFrame with columns: timestamp, mm_id, inventory
    """
    rng = np.random.default_rng(seed)
    kappa = float(mean_reversion)
    sigma = 1.0
    dt = 1.0 / n_timestamps

    # Initialise inventories
    q = rng.standard_normal(n_market_makers)

    timestamps = []
    mm_ids = []
    inventories = []

    for t in range(n_timestamps):
        dW = rng.standard_normal(n_market_makers) * np.sqrt(dt)
        q = q - kappa * q * dt + sigma * dW

        timestamps.extend([t] * n_market_makers)
        mm_ids.extend(list(range(n_market_makers)))
        inventories.extend(q.tolist())

    return pd.DataFrame({
        "timestamp": timestamps,
        "mm_id": mm_ids,
        "inventory": inventories,
    })
