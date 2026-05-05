"""Grid convergence study for the MFG-LOB PDE system.

Solves LOBFormationMFG on successively refined grids, measures L2 error of
the terminal density against the finest-grid reference, and fits convergence
rates.  Repeats for temporal refinement.

Usage
-----
    python experiments/run_convergence_study.py

Output
------
    Prints convergence tables to stdout.
    Saves results/convergence_study.json.
    Saves results/convergence_plot.png (if matplotlib is available).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.mfg_solver import MFGSolver


# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------

def _model():
    return LOBFormationMFG(c_far=0.1, c_near=1.0, kappa=2.0,
                           c_crowd=0.3, c_control=0.1, sigma=0.5, x_max=8.0)


def solve_lob(n_x: int, n_t: int, T: float = 0.5) -> dict:
    model  = _model()
    grid   = Grid1D(0.0, 8.0, n_x)
    tgrid  = TimeGrid(T, n_t)
    solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=40)
    sol    = solver.solve(verbose=False)
    sol["grid"]  = grid
    sol["tgrid"] = tgrid
    return sol


def l2_error(sol_coarse: dict, sol_ref: dict) -> float:
    """L2 error of terminal density (interpolate coarse → fine grid)."""
    x_c  = sol_coarse["grid"].points
    x_f  = sol_ref["grid"].points
    m_c  = sol_coarse["density"][-1]
    m_f  = sol_ref["density"][-1]
    m_ci = np.interp(x_f, x_c, m_c)
    dx   = x_f[1] - x_f[0]
    return float(np.sqrt(np.sum((m_ci - m_f) ** 2) * dx))


def fit_rate(steps: list[float], errors: list[float]) -> float:
    if len(steps) < 2:
        return float("nan")
    rate, _ = np.polyfit(np.log(steps), np.log(errors), 1)
    return float(rate)


# ---------------------------------------------------------------------------
# Spatial convergence
# ---------------------------------------------------------------------------

def spatial_convergence(n_t_fixed: int = 100, T: float = 0.5) -> dict:
    grid_sizes = [50, 100, 200, 400]
    print(f"\n{'='*60}")
    print("Spatial Convergence  (n_t={n_t_fixed}, T={T})".format(
        n_t_fixed=n_t_fixed, T=T))
    print(f"{'n_x':>6}  {'dx':>9}  {'L2 error':>12}  {'rate':>7}  {'t(s)':>7}")
    print("-" * 48)

    t0  = time.time()
    ref = solve_lob(grid_sizes[-1], n_t_fixed, T)
    t_ref = time.time() - t0

    records = []
    for n_x in grid_sizes[:-1]:
        t0  = time.time()
        sol = solve_lob(n_x, n_t_fixed, T)
        elapsed = time.time() - t0
        dx  = 8.0 / (n_x - 1)
        err = l2_error(sol, ref)
        records.append({"n_x": n_x, "dx": dx, "error": err, "time": elapsed})
        print(f"{n_x:>6}  {dx:>9.5f}  {err:>12.4e}  {'—':>7}  {elapsed:>7.2f}")
    dx_ref = 8.0 / (grid_sizes[-1] - 1)
    print(f"{grid_sizes[-1]:>6}  {dx_ref:>9.5f}  {'(ref)':>12}  {'—':>7}  {t_ref:>7.2f}")

    dxs    = [r["dx"]    for r in records]
    errors = [r["error"] for r in records]
    for i in range(1, len(records)):
        r = np.log(errors[i] / errors[i-1]) / np.log(dxs[i] / dxs[i-1])
        print(f"  rate {grid_sizes[i-1]}→{grid_sizes[i]}: {r:.3f}")

    overall = fit_rate(dxs, errors)
    print(f"Overall fitted rate: {overall:.3f}  (target: 1–2 for IMEX upwind+diffusion)")
    return {"type": "spatial", "grid_sizes": grid_sizes,
            "dxs": dxs, "errors": errors, "overall_rate": overall}


# ---------------------------------------------------------------------------
# Temporal convergence
# ---------------------------------------------------------------------------

def temporal_convergence(n_x_fixed: int = 200, T: float = 0.5) -> dict:
    time_grids = [20, 40, 80, 160]
    print(f"\n{'='*60}")
    print("Temporal Convergence  (n_x={n_x}, T={T})".format(
        n_x=n_x_fixed, T=T))
    print(f"{'n_t':>6}  {'dt':>9}  {'L2 error':>12}  {'rate':>7}  {'t(s)':>7}")
    print("-" * 48)

    t0  = time.time()
    ref = solve_lob(n_x_fixed, time_grids[-1], T)
    t_ref = time.time() - t0

    records = []
    for n_t in time_grids[:-1]:
        t0  = time.time()
        sol = solve_lob(n_x_fixed, n_t, T)
        elapsed = time.time() - t0
        dt  = T / n_t
        err = l2_error(sol, ref)
        records.append({"n_t": n_t, "dt": dt, "error": err, "time": elapsed})
        print(f"{n_t:>6}  {dt:>9.5f}  {err:>12.4e}  {'—':>7}  {elapsed:>7.2f}")
    dt_ref = T / time_grids[-1]
    print(f"{time_grids[-1]:>6}  {dt_ref:>9.5f}  {'(ref)':>12}  {'—':>7}  {t_ref:>7.2f}")

    dts    = [r["dt"]    for r in records]
    errors = [r["error"] for r in records]
    for i in range(1, len(records)):
        r = np.log(errors[i] / errors[i-1]) / np.log(dts[i] / dts[i-1])
        print(f"  rate {time_grids[i-1]}→{time_grids[i]}: {r:.3f}")

    overall = fit_rate(dts, errors)
    print(f"Overall fitted rate: {overall:.3f}  (target: ≈ 1 for backward Euler)")
    return {"type": "temporal", "grid_sizes": time_grids,
            "dts": dts, "errors": errors, "overall_rate": overall}


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def save_plot(spatial: dict, temporal: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping convergence plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, step_key, label, title_fmt in [
        (axes[0], spatial,  "dxs", "Δx",
         "Spatial convergence  (rate={rate:.2f})"),
        (axes[1], temporal, "dts", "Δt",
         "Temporal convergence  (rate={rate:.2f})"),
    ]:
        steps  = data[step_key]
        errors = data["errors"]
        ax.loglog(steps, errors, "o-", label="L2 error")
        xs = np.array([steps[0], steps[-1]])
        ax.loglog(xs, errors[0] * (xs / steps[0]) ** 1.0, "--", lw=1, label="O(h¹)")
        ax.loglog(xs, errors[0] * (xs / steps[0]) ** 2.0, ":",  lw=1, label="O(h²)")
        ax.set_xlabel(label)
        ax.set_ylabel("L2 error")
        ax.set_title(title_fmt.format(rate=data["overall_rate"]))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    spatial  = spatial_convergence()
    temporal = temporal_convergence()

    save_plot(spatial, temporal, results_dir / "convergence_plot.png")

    out = results_dir / "convergence_study.json"
    with open(out, "w") as fh:
        json.dump({"spatial": spatial, "temporal": temporal}, fh, indent=2)
    print(f"\nResults → {out}")
