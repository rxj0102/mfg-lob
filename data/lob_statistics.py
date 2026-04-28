"""LOB summary statistics for comparison and calibration.

Computes moment-based and distributional statistics from LOB snapshots,
used as targets for the MFG calibration pipeline.
"""

from __future__ import annotations
import numpy as np


def mean_spread(snapshots: list[dict]) -> float:
    """Average bid-ask spread across snapshots."""
    return float(np.mean([s["spread"] for s in snapshots]))


def depth_profile(
    snapshots: list[dict],
    n_levels: int = 10,
    side: str = "ask",
) -> dict:
    """Average depth profile (cumulative volume vs price distance).

    Parameters
    ----------
    snapshots : list of LOB snapshot dicts (from generate_lob_snapshot)
    n_levels  : number of price levels to include
    side      : 'bid' | 'ask' | 'both'

    Returns
    -------
    dict with 'levels' (price distances) and 'cum_volume'.
    """
    all_cum = []
    for snap in snapshots:
        if side == "ask":
            vols = snap["ask_volumes"][:n_levels]
        elif side == "bid":
            vols = snap["bid_volumes"][:n_levels]
        else:
            vols = 0.5 * (snap["bid_volumes"][:n_levels] + snap["ask_volumes"][:n_levels])
        all_cum.append(np.cumsum(vols))

    levels = np.arange(1, n_levels + 1) * snapshots[0].get("tick_size", 0.01)
    cum_vol = np.mean(all_cum, axis=0)
    return {"levels": levels, "cum_volume": cum_vol}


def order_flow_imbalance(snapshots: list[dict]) -> np.ndarray:
    """Order-flow imbalance (V_bid - V_ask) / (V_bid + V_ask) per snapshot."""
    imb = []
    for s in snapshots:
        vb = s["bid_volumes"].sum()
        va = s["ask_volumes"].sum()
        imb.append((vb - va) / (vb + va + 1e-14))
    return np.array(imb)
