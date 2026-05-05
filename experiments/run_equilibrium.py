"""Compute and visualise MFG-LOB Nash equilibria for all three models.

Solves each model with default parameters, prints summary statistics, and
saves heatmap plots of u(t,x), m(t,x), and α*(t,x) when matplotlib is
available.

Usage
-----
    python experiments/run_equilibrium.py

Output
------
    Console summary tables.
    results/equilibrium_<model>.png  (if matplotlib available).
    results/equilibrium_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.avellaneda_stoikov import AvellanedaStoikovMFG
from mfglob.models.optimal_execution import OptimalExecutionMFG
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.mfg_solver import MFGSolver
from mfglob.equilibrium import EquilibriumAnalyzer


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "AvellanedaStoikov": {
        "model_fn": lambda: AvellanedaStoikovMFG(
            A=1.0, k=1.5, phi=0.01, psi=0.005, gamma=0.1,
            sigma_mid=0.3, Q_max=10.0),
        "x_min": -10.0, "x_max": 10.0, "n_x": 80, "n_t": 60, "T": 0.5,
    },
    "OptimalExecution": {
        "model_fn": lambda: OptimalExecutionMFG(
            phi=0.001, psi=0.01, eta=0.1, lam=1.0, sigma=0.05, Q0=1.0),
        "x_min": 0.0, "x_max": 1.5, "n_x": 60, "n_t": 60, "T": 0.5,
    },
    "LOBFormation": {
        "model_fn": lambda: LOBFormationMFG(
            c_far=0.1, c_near=1.0, kappa=2.0, c_crowd=0.3,
            c_control=0.1, sigma=0.5, x_max=8.0),
        "x_min": 0.0, "x_max": 8.0, "n_x": 80, "n_t": 60, "T": 0.5,
    },
}


def solve_model(name: str, cfg: dict) -> dict:
    model  = cfg["model_fn"]()
    grid   = Grid1D(cfg["x_min"], cfg["x_max"], cfg["n_x"])
    tgrid  = TimeGrid(cfg["T"], cfg["n_t"])
    solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=50)
    sol    = solver.solve(verbose=False)
    sol["model"] = model
    sol["grid"]  = grid
    sol["tgrid"] = tgrid
    sol["name"]  = name
    return sol


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(sol: dict) -> dict:
    name   = sol["name"]
    m      = sol["density"]          # (n_t+1, n_x)
    u      = sol["value_function"]   # (n_t+1, n_x)
    alpha  = sol["optimal_control"]  # (n_t+1, n_x)
    x      = sol["grid"].points
    n_iter = sol.get("n_iterations", "—")
    conv   = sol.get("converged", None)
    ch     = sol.get("convergence_history", [])
    final_res = ch[-1] if ch else float("nan")

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"  Converged:          {conv}  ({n_iter} iterations)")
    print(f"  Final residual:     {final_res:.4e}")
    print(f"  Terminal density:")
    print(f"    mass (trapz):     {float(np.trapezoid(m[-1], x)):.6f}")
    print(f"    mean:             {float(np.trapezoid(x * m[-1], x)):.4f}")
    print(f"    std:              {float(np.sqrt(np.trapezoid((x - np.trapezoid(x*m[-1],x))**2 * m[-1], x))):.4f}")
    print(f"    max density:      {float(m[-1].max()):.4f}  at x={x[np.argmax(m[-1])]:.4f}")
    print(f"  Value function u(0,·):")
    print(f"    min/max:          {float(u[0].min()):.4f} / {float(u[0].max()):.4f}")
    print(f"  Optimal control α*(0,·):")
    print(f"    min/max:          {float(alpha[0].min()):.4f} / {float(alpha[0].max()):.4f}")

    return {
        "converged": bool(conv) if conv is not None else None,
        "n_iterations": n_iter,
        "final_residual": float(final_res) if np.isfinite(final_res) else None,
        "terminal_mass": float(np.trapezoid(m[-1], x)),
        "terminal_mean": float(np.trapezoid(x * m[-1], x)),
        "terminal_std": float(np.sqrt(np.maximum(
            np.trapezoid((x - np.trapezoid(x * m[-1], x)) ** 2 * m[-1], x), 0.0))),
    }


# ---------------------------------------------------------------------------
# Extra: LOB book shape and OE execution schedule
# ---------------------------------------------------------------------------

def print_lob_book_shape(sol: dict) -> None:
    x = sol["grid"].points
    m = sol["density"]
    # Steady-state approximation: use time-average of density
    m_avg = np.mean(m, axis=0)
    mass  = np.trapezoid(m_avg, x)
    if mass > 1e-14:
        m_avg /= mass
    peak_idx = np.argmax(m_avg)
    print(f"\n  LOB Book Shape (time-averaged density):")
    print(f"    Mode at x = {x[peak_idx]:.4f}  (density = {m_avg[peak_idx]:.4f})")
    print(f"    Near-zero density m(0): {m_avg[0]:.4f}")
    print(f"    Far-field density m(x_max): {m_avg[-1]:.6f}")


def print_oe_execution_schedule(sol: dict) -> None:
    x     = sol["grid"].points
    alpha = sol["optimal_control"]
    m     = sol["density"]
    n_t   = sol["tgrid"].n_nodes
    T     = sol["tgrid"].T
    ts    = np.linspace(0, T, n_t)
    rates = [float(np.trapezoid(np.maximum(alpha[t], 0.0) * m[t], x)) for t in range(n_t)]
    cum   = np.cumsum(rates) * (T / n_t)
    print(f"\n  Optimal Execution Schedule:")
    print(f"    Aggregate rate at t=0: {rates[0]:.4f}")
    print(f"    Aggregate rate at t=T: {rates[-1]:.4f}")
    print(f"    Cumulative executed:   {cum[-1]:.4f}  (initial inventory ≈ Q0)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_heatmaps(sol: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    m      = sol["density"]
    u      = sol["value_function"]
    alpha  = sol["optimal_control"]
    x      = sol["grid"].points
    n_t    = sol["tgrid"].n_nodes
    T      = sol["tgrid"].T
    ts     = np.linspace(0, T, n_t)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, data, title in [
        (axes[0], m,     "Density  m(t,x)"),
        (axes[1], u,     "Value function  u(t,x)"),
        (axes[2], alpha, "Optimal control  α*(t,x)"),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto",
                       extent=[x[0], x[-1], ts[0], ts[-1]])
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.suptitle(sol["name"], y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmaps saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_stats = {}
    for name, cfg in CONFIGS.items():
        print(f"\nSolving {name}...")
        sol   = solve_model(name, cfg)
        stats = print_summary(sol)
        all_stats[name] = stats

        if name == "LOBFormation":
            print_lob_book_shape(sol)
        if name == "OptimalExecution":
            print_oe_execution_schedule(sol)

        save_heatmaps(sol, results_dir / f"equilibrium_{name.lower()}.png")

    out = results_dir / "equilibrium_results.json"
    with open(out, "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\nResults → {out}")
