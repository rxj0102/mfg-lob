"""LOB summary statistics for comparison and calibration.

Computes moment-based and distributional statistics from LOB snapshots,
used as targets for the MFG calibration pipeline.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Additional LOB statistics
# ---------------------------------------------------------------------------


def average_book_shape(book_snapshots: list) -> dict:
    """Average volume at each price level across snapshots.

    Parameters
    ----------
    book_snapshots : list of dicts from generate_synthetic_book_shape()
                     Each dict must have 'price_levels', 'volumes_bid', 'volumes_ask'.

    Returns
    -------
    dict with keys: 'price_levels', 'avg_volumes_bid', 'avg_volumes_ask',
                    'std_volumes_bid', 'std_volumes_ask'
    """
    if not book_snapshots:
        raise ValueError("book_snapshots must be non-empty")

    price_levels = np.asarray(book_snapshots[0]["price_levels"], dtype=float)
    n_levels = len(price_levels)

    bid_matrix = np.zeros((len(book_snapshots), n_levels))
    ask_matrix = np.zeros((len(book_snapshots), n_levels))

    for i, snap in enumerate(book_snapshots):
        vb = np.asarray(snap["volumes_bid"], dtype=float)
        va = np.asarray(snap["volumes_ask"], dtype=float)
        # Trim or pad to n_levels
        bid_matrix[i, :len(vb)] = vb[:n_levels]
        ask_matrix[i, :len(va)] = va[:n_levels]

    return {
        "price_levels": price_levels,
        "avg_volumes_bid": np.mean(bid_matrix, axis=0),
        "avg_volumes_ask": np.mean(ask_matrix, axis=0),
        "std_volumes_bid": np.std(bid_matrix, axis=0),
        "std_volumes_ask": np.std(ask_matrix, axis=0),
    }


def empirical_impact_curve(trades: pd.DataFrame, n_bins: int = 20) -> dict:
    """Compute empirical impact curve by binning trades by size.

    Parameters
    ----------
    trades : DataFrame with columns 'trade_size', 'price_impact'
    n_bins : number of bins

    Returns
    -------
    dict with keys: 'bin_centers', 'mean_impact', 'std_impact', 'n_trades_per_bin'
    """
    sizes = np.asarray(trades["trade_size"], dtype=float)
    impacts = np.asarray(trades["price_impact"], dtype=float)

    bin_edges = np.percentile(sizes, np.linspace(0, 100, n_bins + 1))
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1

    bin_centers = []
    mean_impact = []
    std_impact = []
    n_per_bin = []

    for i in range(actual_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == actual_bins - 1:
            mask = (sizes >= lo) & (sizes <= hi)
        else:
            mask = (sizes >= lo) & (sizes < hi)
        n = int(mask.sum())
        if n > 0:
            bin_centers.append(0.5 * (lo + hi))
            mean_impact.append(float(np.mean(impacts[mask])))
            std_impact.append(float(np.std(impacts[mask])) if n > 1 else 0.0)
            n_per_bin.append(n)

    return {
        "bin_centers": np.array(bin_centers),
        "mean_impact": np.array(mean_impact),
        "std_impact": np.array(std_impact),
        "n_trades_per_bin": np.array(n_per_bin, dtype=int),
    }


def spread_statistics(book_snapshots: list) -> dict:
    """Compute spread statistics across LOB snapshots.

    Parameters
    ----------
    book_snapshots : list of dicts, each with a 'spread' key

    Returns
    -------
    dict with keys: 'mean', 'median', 'std', 'min', 'max', 'distribution'
    """
    if not book_snapshots:
        raise ValueError("book_snapshots must be non-empty")

    spreads = np.array([float(s["spread"]) for s in book_snapshots])

    return {
        "mean": float(np.mean(spreads)),
        "median": float(np.median(spreads)),
        "std": float(np.std(spreads)),
        "min": float(np.min(spreads)),
        "max": float(np.max(spreads)),
        "distribution": spreads,
    }
