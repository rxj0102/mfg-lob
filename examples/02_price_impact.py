"""Example 2: Price impact function from MFG equilibrium.

Shows that the MFG model generates a concave price impact curve I(v) ∝ v^β
consistent with the empirically observed square-root law (β ≈ 0.5).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfg_lob.models import LOBModel
from mfg_lob.solvers import Grid, MFGSolver
from mfg_lob.analysis import PriceImpactAnalyzer

# ── Solve equilibrium ─────────────────────────────────────────────────────────
model = LOBModel(sigma=0.5, phi=0.1, alpha=0.5, A_term=1.0)
grid = Grid.uniform(q_min=-8, q_max=8, nq=81, t_min=0, t_max=1, nt=101)
solver = MFGSolver(model, grid, tol=1e-5, max_iter=80, damping=0.5)
eq = solver.solve()

# ── Price impact ──────────────────────────────────────────────────────────────
pia = PriceImpactAnalyzer(eq, price_tick=0.01)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Impact curves at different times
for t, color in [(0.0, "steelblue"), (0.5, "darkorange"), (0.9, "tomato")]:
    curve = pia.compute(t=t, n_volumes=60)
    beta = curve.fit_params.get("beta", float("nan"))
    axes[0].plot(curve.volumes, curve.impact, color=color,
                 label=f"t={t:.1f}  (β={beta:.3f})")

axes[0].set_xlabel("Metaorder volume")
axes[0].set_ylabel("Price impact I(v)")
axes[0].set_title("MFG Price Impact Function")
axes[0].legend()

# Log-log plot to visualise power law
curve = pia.compute(t=0.0, n_volumes=60)
v_pos = curve.volumes[curve.volumes > 0]
I_pos = curve.impact[curve.volumes > 0]
mask = I_pos > 0
if mask.sum() > 2:
    axes[1].loglog(v_pos[mask], I_pos[mask], "o-", markersize=3, label="MFG")
    beta = curve.fit_params["beta"]
    Y = curve.fit_params["Y"]
    v_fit = np.linspace(v_pos[mask][0], v_pos[mask][-1], 100)
    axes[1].loglog(v_fit, Y * v_fit**beta, "--", color="k",
                   label=f"Power law fit: β={beta:.3f}")
    # Reference square-root law
    axes[1].loglog(v_fit, Y * v_fit**0.5, ":", color="grey", label="β=0.5 (empirical)")
axes[1].set_xlabel("log volume")
axes[1].set_ylabel("log impact")
axes[1].set_title("Log-Log: Impact Power Law")
axes[1].legend()

fig.tight_layout()
fig.savefig("02_price_impact.png", dpi=120, bbox_inches="tight")
print(f"Power-law exponent β = {curve.fit_params['beta']:.4f}  (R² = {curve.fit_params['r2']:.4f})")
print("Saved: 02_price_impact.png")
