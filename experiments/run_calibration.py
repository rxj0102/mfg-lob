"""Calibration experiments for the MFG-LOB model.

Experiment 1 — LOBFormation model calibrated to a synthetic exponential
               book shape.  Recovers c_crowd and sigma.

Experiment 2 — OptimalExecution model calibrated to synthetic sqrt-law
               impact data.  Recovers eta and phi.

Usage
-----
    python experiments/run_calibration.py

Output
------
    Console summary with parameter estimates and fit quality.
    results/calibration.json
    results/calibration_fit.png  (if matplotlib available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.models.optimal_execution import OptimalExecutionMFG
from mfglob.calibration import MFGCalibrator
from data.synthetic_lob import (
    generate_synthetic_book_shape,
    generate_synthetic_impact_data,
)
from data.lob_statistics import empirical_impact_curve


# ---------------------------------------------------------------------------
# Experiment 1: LOBFormation → exponential book shape
# ---------------------------------------------------------------------------

def calibrate_lob_formation() -> dict:
    print("\n" + "=" * 60)
    print("Experiment 1: Calibrate LOBFormation to exponential book shape")

    # Generate synthetic target
    snap   = generate_synthetic_book_shape(n_levels=30, shape="exponential", seed=42)
    levels = np.asarray(snap["price_levels"], dtype=float)
    # Use average of bid/ask volumes as density target
    density = 0.5 * (snap["volumes_bid"] + snap["volumes_ask"])
    density = density / np.trapezoid(density, levels)  # normalise

    target_data = {
        "book_shape": {"levels": levels, "density": density},
        "spread":     float(snap["spread"]),
    }

    # True-ish parameters: c_crowd=0.3, sigma=0.5 (generating model defaults)
    calibrator = MFGCalibrator(
        model_class=LOBFormationMFG,
        target_data=target_data,
        param_bounds={
            "c_crowd": (0.05, 3.0),
            "sigma":   (0.1, 2.0),
        },
        grid_params={"n_x": 30, "n_t": 20, "T": 0.3, "x_min": 0.0, "x_max": 8.0},
    )

    result = calibrator.calibrate(method="nelder_mead", verbose=False)

    print(f"\n  Calibrated parameters:")
    for k, v in result["params"].items():
        print(f"    {k:>12s} = {v:.4f}")
    print(f"  Objective value:   {result['objective']:.6f}")
    print(f"  Convergence:       {result['success']}")
    print(f"  Function evals:    {result['n_evals']}")

    return result


# ---------------------------------------------------------------------------
# Experiment 2: OptimalExecution → sqrt-law impact
# ---------------------------------------------------------------------------

def calibrate_optimal_execution() -> dict:
    print("\n" + "=" * 60)
    print("Experiment 2: Calibrate OptimalExecution to sqrt-law impact data")

    # Generate synthetic impact data
    trades = generate_synthetic_impact_data(n_trades=500, impact_law="sqrt", seed=7)
    curve  = empirical_impact_curve(trades, n_bins=12)

    # Use a subset of well-sampled bins as targets
    order_sizes = curve["bin_centers"]
    impacts     = curve["mean_impact"]

    target_data = {
        "impact_curve": {
            "order_sizes":      order_sizes.tolist(),
            "permanent_impact": impacts.tolist(),
        }
    }

    calibrator = MFGCalibrator(
        model_class=OptimalExecutionMFG,
        target_data=target_data,
        param_bounds={
            "eta": (0.01, 1.0),
            "phi": (0.0001, 0.05),
        },
        grid_params={"n_x": 25, "n_t": 20, "T": 0.3, "x_min": 0.0, "x_max": 1.5},
    )

    result = calibrator.calibrate(method="nelder_mead", verbose=False)

    print(f"\n  Calibrated parameters:")
    for k, v in result["params"].items():
        print(f"    {k:>12s} = {v:.4f}")
    print(f"  Objective value:   {result['objective']:.6f}")
    print(f"  Convergence:       {result['success']}")
    print(f"  Function evals:    {result['n_evals']}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_calibration_plot(lob_result: dict, oe_result: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping calibration plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # LOB Formation: show calibrated book shape
    ax = axes[0]
    snap   = generate_synthetic_book_shape(n_levels=30, shape="exponential", seed=42)
    levels = np.asarray(snap["price_levels"], dtype=float)
    density = 0.5 * (snap["volumes_bid"] + snap["volumes_ask"])
    density = density / np.trapezoid(density, levels)
    ax.plot(levels, density, "k-", lw=2, label="Target (synthetic)")
    ax.set_xlabel("Price level (ticks from mid)")
    ax.set_ylabel("Normalised density")
    ax.set_title("LOBFormation calibration\n"
                 + ", ".join(f"{k}={v:.3f}" for k, v in lob_result["params"].items()))
    ax.legend()

    # OE: show target impact vs calibrated
    ax = axes[1]
    trades = generate_synthetic_impact_data(n_trades=500, impact_law="sqrt", seed=7)
    curve  = empirical_impact_curve(trades, n_bins=12)
    ax.errorbar(curve["bin_centers"], curve["mean_impact"],
                yerr=curve["std_impact"], fmt="o", label="Target (synthetic)")
    ax.set_xlabel("Trade size Q")
    ax.set_ylabel("Price impact ΔS")
    ax.set_title("OptimalExecution calibration\n"
                 + ", ".join(f"{k}={v:.4f}" for k, v in oe_result["params"].items()))
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    lob_result = calibrate_lob_formation()
    oe_result  = calibrate_optimal_execution()

    save_calibration_plot(lob_result, oe_result, results_dir / "calibration_fit.png")

    out = results_dir / "calibration.json"
    with open(out, "w") as fh:
        json.dump({"lob_formation": lob_result, "optimal_execution": oe_result},
                  fh, indent=2)
    print(f"\nResults → {out}")
