"""Example 3: Comparative statics — how equilibrium changes with parameters.

Sweeps the inventory aversion phi and volatility sigma, showing how the
equilibrium LOB shape, spread, and price impact exponent respond.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfg_lob.models import LOBModel
from mfg_lob.solvers import Grid
from mfg_lob.analysis import ComparativeStatics

# Use a coarser grid for speed in the sweep
grid = Grid.uniform(q_min=-6, q_max=6, nq=61, t_min=0, t_max=0.5, nt=51)

base_model = LOBModel(sigma=0.5, phi=0.2, alpha=0.5, A_term=1.0,
                      agent_type="market_maker")

cs = ComparativeStatics(
    base_model=base_model,
    grid=grid,
    solver_kwargs={"max_iter": 40, "tol": 1e-4, "damping": 0.5},
)

# ── Sweep 1: inventory aversion phi ──────────────────────────────────────────
print("Sweeping phi (inventory aversion)...")
phi_values = np.linspace(0.05, 0.8, 8)
result_phi = cs.sweep("phi", phi_values, verbose=True)

# ── Sweep 2: volatility sigma ─────────────────────────────────────────────────
print("\nSweeping sigma (volatility)...")
sigma_values = np.linspace(0.2, 1.2, 8)
result_sigma = cs.sweep("sigma", sigma_values, verbose=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for row, (result, label) in enumerate([(result_phi, "phi"), (result_sigma, "sigma")]):
    pv = result.param_values
    axes[row, 0].plot(pv, result.spread_proxies, "o-")
    axes[row, 0].set_xlabel(label)
    axes[row, 0].set_ylabel("Spread proxy")
    axes[row, 0].set_title(f"Spread vs {label}")

    axes[row, 1].plot(pv, result.total_depths, "s-", color="steelblue")
    axes[row, 1].set_xlabel(label)
    axes[row, 1].set_ylabel("Total depth")
    axes[row, 1].set_title(f"Liquidity vs {label}")

    betas = result.impact_betas
    axes[row, 2].plot(pv, betas, "^-", color="tomato")
    axes[row, 2].axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="β=0.5")
    axes[row, 2].set_xlabel(label)
    axes[row, 2].set_ylabel("Impact exponent β")
    axes[row, 2].set_title(f"Price impact β vs {label}")
    axes[row, 2].legend(fontsize=8)

fig.tight_layout()
fig.savefig("03_comparative_statics.png", dpi=120, bbox_inches="tight")
print("\nSaved: 03_comparative_statics.png")
