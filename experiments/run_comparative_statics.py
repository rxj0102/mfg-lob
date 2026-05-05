"""Comparative statics: how model parameters affect the MFG equilibrium.

LOBFormation model
  - vary c_crowd (congestion penalty)    : [0.1, 0.3, 0.5, 1.0, 2.0]
  - vary sigma   (volatility)            : [0.2, 0.5, 0.8, 1.2]
  - vary kappa   (adverse-selection decay): [0.5, 1.0, 2.0, 4.0]

AvellanedaStoikov model
  - vary phi (inventory penalty): [0.005, 0.01, 0.05, 0.1]
    → reports equilibrium spread proxy 1/k ± corrections

Usage
-----
    python experiments/run_comparative_statics.py

Output
------
    Console tables.
    results/comparative_statics.json
    results/comparative_statics_*.png  (if matplotlib available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.avellaneda_stoikov import AvellanedaStoikovMFG
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.mfg_solver import MFGSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def solve_lob_formation(**kwargs) -> dict:
    params = dict(c_far=0.1, c_near=1.0, kappa=2.0, c_crowd=0.3,
                  c_control=0.1, sigma=0.5, x_max=8.0)
    params.update(kwargs)
    model  = LOBFormationMFG(**params)
    grid   = Grid1D(0.0, 8.0, 60)
    tgrid  = TimeGrid(0.5, 50)
    solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=40)
    sol    = solver.solve(verbose=False)
    return sol, grid


def book_shape_stats(sol: dict, grid: Grid1D) -> dict:
    """Return mode, mean, variance, and near-zero density of time-avg m."""
    x     = grid.points
    m_avg = np.mean(sol["density"], axis=0)
    mass  = np.trapezoid(m_avg, x)
    if mass > 1e-14:
        m_avg = m_avg / mass
    mean  = float(np.trapezoid(x * m_avg, x))
    var   = float(np.trapezoid((x - mean) ** 2 * m_avg, x))
    mode  = float(x[np.argmax(m_avg)])
    near0 = float(m_avg[0])
    return {"mode": mode, "mean": mean, "std": float(np.sqrt(max(var, 0.0))), "near0": near0}


def solve_as(**kwargs) -> dict:
    params = dict(A=1.0, k=1.5, phi=0.01, psi=0.005, gamma=0.1,
                  sigma_mid=0.3, Q_max=10.0)
    params.update(kwargs)
    model  = AvellanedaStoikovMFG(**params)
    grid   = Grid1D(-10.0, 10.0, 60)
    tgrid  = TimeGrid(0.5, 50)
    solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=40)
    sol    = solver.solve(verbose=False)
    return sol, model


def spread_stats(sol: dict, model) -> dict:
    x       = sol["grid"].points if "grid" in sol else np.linspace(-10, 10, 60)
    u       = sol["value_function"]
    p       = np.gradient(u[0])  # ∂_x u at t=0 (approx)
    q_grid  = np.linspace(-10, 10, 60)
    _, ask_spread = model.optimal_spread(q_grid, p, np.ones_like(q_grid) * 0.1)
    bid_spread, _ = model.optimal_spread(q_grid, p, np.ones_like(q_grid) * 0.1)
    base_spread   = 1.0 / model.k
    return {"base_spread": base_spread,
            "mean_bid_spread": float(np.mean(bid_spread)),
            "mean_ask_spread": float(np.mean(ask_spread))}


# ---------------------------------------------------------------------------
# LOBFormation sweeps
# ---------------------------------------------------------------------------

def sweep_lob(param: str, values: list, label: str) -> list:
    print(f"\n{'='*60}")
    print(f"LOBFormation — varying {param}  ({label})")
    print(f"  {'value':>8}  {'mode':>8}  {'mean':>8}  {'std':>8}  {'m(0)':>10}")
    print("  " + "-" * 50)
    records = []
    for v in values:
        sol, grid = solve_lob_formation(**{param: v})
        st = book_shape_stats(sol, grid)
        print(f"  {v:>8.3f}  {st['mode']:>8.4f}  {st['mean']:>8.4f}  "
              f"{st['std']:>8.4f}  {st['near0']:>10.6f}")
        records.append({"value": v, **st})
    return records


def sweep_as_phi(values: list) -> list:
    print(f"\n{'='*60}")
    print("AvellanedaStoikov — varying phi (inventory penalty)")
    print(f"  {'phi':>8}  {'base_spread':>12}  {'mean_bid':>10}  {'mean_ask':>10}")
    print("  " + "-" * 46)
    records = []
    for v in values:
        sol, model = solve_as(phi=v)
        st = spread_stats(sol, model)
        print(f"  {v:>8.4f}  {st['base_spread']:>12.4f}  "
              f"{st['mean_bid_spread']:>10.4f}  {st['mean_ask_spread']:>10.4f}")
        records.append({"phi": v, **st})
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_lob_sweep_plot(records_list: list[list], param_names: list[str],
                         out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping statics plot")
        return

    fig, axes = plt.subplots(1, len(records_list), figsize=(5 * len(records_list), 4))
    if len(records_list) == 1:
        axes = [axes]

    for ax, records, pname in zip(axes, records_list, param_names):
        xs  = [r["value"] for r in records]
        ax.plot(xs, [r["mode"] for r in records], "o-", label="mode")
        ax.plot(xs, [r["mean"] for r in records], "s--", label="mean")
        ax.fill_between(xs,
                         [r["mean"] - r["std"] for r in records],
                         [r["mean"] + r["std"] for r in records],
                         alpha=0.15, label="±1 std")
        ax.set_xlabel(pname)
        ax.set_ylabel("Position x")
        ax.set_title(f"Book shape vs {pname}")
        ax.legend(fontsize=8)

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

    lob_results = {}

    rec_crowd = sweep_lob("c_crowd", [0.1, 0.3, 0.5, 1.0, 2.0],
                           "congestion penalty")
    lob_results["c_crowd"] = rec_crowd

    rec_sigma = sweep_lob("sigma", [0.2, 0.5, 0.8, 1.2], "volatility")
    lob_results["sigma"] = rec_sigma

    rec_kappa = sweep_lob("kappa", [0.5, 1.0, 2.0, 4.0],
                           "adverse-selection decay")
    lob_results["kappa"] = rec_kappa

    save_lob_sweep_plot(
        [rec_crowd, rec_sigma, rec_kappa],
        ["c_crowd", "sigma", "kappa"],
        results_dir / "comparative_statics_lob.png",
    )

    as_results = sweep_as_phi([0.005, 0.01, 0.05, 0.1])

    out = results_dir / "comparative_statics.json"
    with open(out, "w") as fh:
        json.dump({"lob_formation": lob_results, "avellaneda_stoikov": as_results},
                  fh, indent=2)
    print(f"\nResults → {out}")
