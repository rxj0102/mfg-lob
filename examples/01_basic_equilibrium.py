"""Example 1: Computing a basic MFG-LOB equilibrium.

Demonstrates solving the coupled HJB-FP system for a symmetric market maker
and inspecting the resulting value function and population density.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfg_lob.models import LOBModel
from mfg_lob.solvers import Grid, MFGSolver

# ── Model parameters ──────────────────────────────────────────────────────────
model = LOBModel(
    sigma=0.5,       # inventory diffusion (order flow randomness)
    phi=0.1,         # inventory aversion
    alpha=0.5,       # mean-field coupling
    A_term=1.0,      # quadratic terminal cost
    Lambda0=1.0,     # base order arrival rate
    kappa=1.0,       # order arrival decay with depth
    agent_type="market_maker",
)
print(model)

# ── Grid ──────────────────────────────────────────────────────────────────────
grid = Grid.uniform(q_min=-8, q_max=8, nq=81, t_min=0.0, t_max=1.0, nt=101)

# ── Solve ─────────────────────────────────────────────────────────────────────
solver = MFGSolver(model, grid, tol=1e-5, max_iter=80, damping=0.5, verbose=True)
eq = solver.solve()

print(f"\nConverged: {eq.converged}  ({eq.n_iterations} iterations)")
print(f"Final residual: {eq.residuals[-1]:.3e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Value function at t=0 and t=T/2
axes[0].plot(grid.q, eq.u[:, 0], label="t=0")
axes[0].plot(grid.q, eq.u[:, grid.nt // 2], label="t=T/2")
axes[0].plot(grid.q, eq.u[:, -1], label="t=T (terminal)", linestyle="--")
axes[0].set_xlabel("Inventory q")
axes[0].set_ylabel("Value u(q,t)")
axes[0].set_title("Value Function")
axes[0].legend()

# Population density evolution
times = [0, grid.nt // 4, grid.nt // 2, -1]
cmap = plt.get_cmap("viridis")
for i, k in enumerate(times):
    t_label = grid.t[k]
    color = cmap(i / (len(times) - 1))
    axes[1].plot(grid.q, eq.m[:, k], color=color, label=f"t={t_label:.2f}")
axes[1].set_xlabel("Inventory q")
axes[1].set_ylabel("Density m(q,t)")
axes[1].set_title("Population Density (LOB shape)")
axes[1].legend(fontsize=8)

# Convergence residuals
axes[2].semilogy(range(1, len(eq.residuals) + 1), eq.residuals, "o-")
axes[2].set_xlabel("Picard iteration")
axes[2].set_ylabel("Relative L2 residual")
axes[2].set_title("Fixed-Point Convergence")
axes[2].axhline(solver.tol, color="r", linestyle="--", label=f"tol={solver.tol:.0e}")
axes[2].legend()

fig.tight_layout()
fig.savefig("01_basic_equilibrium.png", dpi=120, bbox_inches="tight")
print("\nSaved: 01_basic_equilibrium.png")
