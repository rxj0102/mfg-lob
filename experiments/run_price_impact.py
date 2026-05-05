"""Price impact analysis for the OptimalExecutionMFG model.

Computes the MFG permanent price impact curve I(Q), compares it to the
empirical square-root law I(Q) ∝ σ√(Q/V), and fits the power-law exponent δ
via log-log regression.

Also shows sensitivity of the impact curve to η (permanent impact coefficient)
and φ (inventory penalty).

Usage
-----
    python experiments/run_price_impact.py

Output
------
    Console summary tables.
    results/price_impact.json
    results/price_impact.png  (if matplotlib available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.optimal_execution import OptimalExecutionMFG
from mfglob.mfg_solver import MFGSolver
from mfglob.price_impact import PriceImpactAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_analyzer(phi: float = 0.5, eta: float = 0.2,
                  Q0: float = 1.0, sigma: float = 0.05,
                  psi: float = 1.0) -> PriceImpactAnalyzer:
    model  = OptimalExecutionMFG(phi=phi, psi=psi, eta=eta,
                                 lam=1.0, sigma=sigma, Q0=Q0)
    grid   = Grid1D(0.0, max(Q0 * 1.5, 0.5), 40)
    tgrid  = TimeGrid(0.5, 40)
    solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=30)
    sol    = solver.solve(verbose=False)
    return PriceImpactAnalyzer(sol, model, grid, tgrid)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def impact_curve_analysis() -> dict:
    print("\n" + "=" * 60)
    print("MFG Price Impact Curve  (default parameters)")
    print("  phi=0.5, psi=1.0, eta=0.2, Q0=1.0, sigma=0.05")

    pia  = make_analyzer()
    Qs   = np.linspace(0.1, 1.8, 10)
    curve = pia.impact_curve(Qs)
    perms = curve["permanent_impact"]
    temps = curve["temporary_impact_peak"]

    print(f"\n  {'Q':>8}  {'permanent':>12}  {'temp_peak':>12}")
    print("  " + "-" * 36)
    for q, pi, ti in zip(Qs, perms, temps):
        print(f"  {q:>8.3f}  {pi:>12.6f}  {ti:>12.6f}")

    # Power-law fit
    sqrl = pia.compare_to_square_root_law(Qs)
    delta = sqrl["mfg_exponent"]
    coeff = sqrl["mfg_coefficient"]
    print(f"\n  Power-law fit:  ΔS ≈ {coeff:.5f} · Q^{delta:.3f}")
    print(f"  Sqrt-law exponent (theory): 0.5")
    print(f"  Fitted exponent δ:          {delta:.3f}")

    # Kyle lambda
    lam = pia.kyle_lambda()
    print(f"  Kyle's lambda (ΔS/Q, small Q): {lam:.6f}")

    return {
        "order_sizes": Qs.tolist(),
        "permanent_impact": perms.tolist(),
        "temporary_impact_peak": temps.tolist(),
        "fitted_exponent": float(delta),
        "fitted_coefficient": float(coeff),
        "sqrt_law": sqrl["sqrt_law"].tolist(),
        "kyle_lambda": float(lam),
    }


def sensitivity_analysis() -> dict:
    print("\n" + "=" * 60)
    print("Sensitivity: impact exponent vs eta and phi")

    Qs = np.linspace(0.15, 1.5, 8)
    results = {}

    print("\n  Varying η (permanent impact coefficient):")
    print(f"  {'eta':>8}  {'exponent':>10}  {'lambda':>10}")
    print("  " + "-" * 32)
    etas   = [0.1, 0.2, 0.3, 0.4, 0.6]
    eta_rec = []
    for eta in etas:
        try:
            pia   = make_analyzer(eta=eta)
            r     = pia.compare_to_square_root_law(Qs)
            lam   = pia.kyle_lambda()
            delta = r["mfg_exponent"]
            print(f"  {eta:>8.3f}  {delta:>10.3f}  {lam:>10.6f}")
            eta_rec.append({"eta": eta, "exponent": float(delta), "lambda": float(lam)})
        except Exception as e:
            print(f"  {eta:>8.3f}  ERROR: {e}")
    results["eta_sensitivity"] = eta_rec

    print("\n  Varying φ (inventory penalty):")
    print(f"  {'phi':>8}  {'exponent':>10}  {'lambda':>10}")
    print("  " + "-" * 32)
    phis   = [0.1, 0.3, 0.5, 0.8, 1.0]
    phi_rec = []
    for phi in phis:
        try:
            pia   = make_analyzer(phi=phi)
            r     = pia.compare_to_square_root_law(Qs)
            lam   = pia.kyle_lambda()
            delta = r["mfg_exponent"]
            print(f"  {phi:>8.3f}  {delta:>10.3f}  {lam:>10.6f}")
            phi_rec.append({"phi": phi, "exponent": float(delta), "lambda": float(lam)})
        except Exception as e:
            print(f"  {phi:>8.3f}  ERROR: {e}")
    results["phi_sensitivity"] = phi_rec

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(main_res: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping impact plot")
        return

    Qs   = np.array(main_res["order_sizes"])
    pi   = np.array(main_res["permanent_impact"])
    sqrl = np.array(main_res["sqrt_law"])
    delta = main_res["fitted_exponent"]
    C     = main_res["fitted_coefficient"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(Qs, pi,   "o-",  label=f"MFG (δ={delta:.2f})")
    ax.plot(Qs, sqrl, "--",  label="Sqrt law (δ=0.5)")
    ax.plot(Qs, C * Qs ** delta, ":", label=f"Fitted power law")
    ax.set_xlabel("Order size Q")
    ax.set_ylabel("Permanent price impact ΔS")
    ax.set_title("MFG vs square-root law")
    ax.legend()

    ax = axes[1]
    ax.loglog(Qs, pi,   "o-",  label=f"MFG (δ={delta:.2f})")
    ax.loglog(Qs, sqrl, "--",  label="Sqrt law (δ=0.5)")
    ax.set_xlabel("Order size Q  (log)")
    ax.set_ylabel("ΔS  (log)")
    ax.set_title("Log-log: power-law fit")
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

    main_res  = impact_curve_analysis()
    sens_res  = sensitivity_analysis()

    save_plot(main_res, results_dir / "price_impact.png")

    out = results_dir / "price_impact.json"
    with open(out, "w") as fh:
        json.dump({"main": main_res, "sensitivity": sens_res}, fh, indent=2)
    print(f"\nResults → {out}")
