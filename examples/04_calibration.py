"""Example 4: Calibrate the MFG-LOB model to synthetic empirical LOB data.

Generates a synthetic LOB (representing an empirically observed order book)
and fits model parameters (sigma, phi, alpha) to match its depth profile
and price impact curve.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfg_lob.calibration import EmpiricalLOB, MFGCalibrator
from mfg_lob.solvers import Grid, MFGSolver
from mfg_lob.models import LOBModel
from mfg_lob.analysis import PriceImpactAnalyzer, LOBShapeAnalyzer

# ── Generate synthetic empirical data ─────────────────────────────────────────
# Simulates an order book with power-law depth profile (alpha=0.6)
emp = EmpiricalLOB.from_synthetic(
    n_snapshots=200,
    spread_mean=0.8,
    spread_std=0.05,
    total_volume=500.0,
    alpha=0.6,
    n_levels=15,
    tick_size=0.1,
    seed=42,
)

print(f"Empirical LOB: {len(emp)} snapshots")
print(f"Mean spread: {emp.mean_spread:.4f}")
print(f"Mean imbalance: {emp.mean_imbalance:.4f}")

# ── Calibration ───────────────────────────────────────────────────────────────
grid = Grid.uniform(q_min=-6, q_max=6, nq=51, t_min=0, t_max=0.5, nt=51)

base_params = {
    "A_term": 1.0, "B_term": 0.0,
    "Lambda0": 1.0, "kappa": 1.0,
    "mu0": 0.0, "sigma0": 2.0,
    "agent_type": "market_maker",
}

calib = MFGCalibrator(
    empirical=emp,
    grid=grid,
    param_names=["sigma", "phi", "alpha"],
    param_bounds=[(0.1, 3.0), (0.01, 2.0), (0.1, 3.0)],
    base_params=base_params,
    weights={"spread": 1.0, "depth": 2.0, "impact": 1.0},
    solver_kwargs={"max_iter": 30, "tol": 1e-4, "damping": 0.5},
    tick_size=0.1,
    n_depth_levels=8,
    n_impact_points=15,
)

print("\nRunning calibration (Nelder-Mead)...")
result = calib.calibrate(max_iter=150, verbose=True)

print(f"\nCalibration {'succeeded' if result.success else 'stopped early'}:")
for k, v in result.params.items():
    if k in calib.param_names:
        print(f"  {k:12s} = {v:.6f}")
print(f"  Loss = {result.loss:.6f}  ({result.n_evaluations} evaluations)")

# ── Compare model vs empirical ────────────────────────────────────────────────
model_cal = LOBModel(**result.params)
solver = MFGSolver(model_cal, grid, max_iter=50, tol=1e-5, damping=0.5)
eq_cal = solver.solve()

pia = PriceImpactAnalyzer(eq_cal, price_tick=0.1)
lsa = LOBShapeAnalyzer(eq_cal)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Depth profile comparison
prof_model = lsa.depth_profile(t=0.0, n_levels=8)
prof_emp = emp.mean_depth_profile(n_levels=8, tick_size=0.1)

# Normalise for shape comparison
model_depth = (prof_model["bid_volume"] + prof_model["ask_volume"]) / 2
model_depth_norm = model_depth / (model_depth.mean() + 1e-14)
emp_depth_norm = prof_emp["ask_cum"] / (prof_emp["ask_cum"].mean() + 1e-14)

axes[0].plot(prof_emp["levels"], emp_depth_norm, "o-", label="Empirical")
axes[0].plot(prof_model["levels"][:8], model_depth_norm[:8], "s--", label="MFG (calibrated)")
axes[0].set_xlabel("Price distance")
axes[0].set_ylabel("Normalised depth")
axes[0].set_title("LOB Depth Profile")
axes[0].legend()

# Price impact comparison
vols = np.linspace(0, 2.0, 30)
imp_emp = emp.empirical_impact(vols, tick_size=0.1, n_levels=15)
curve_cal = pia.compute(t=0.0, n_volumes=30)
imp_cal_norm = curve_cal.impact / (curve_cal.impact.mean() + 1e-14)
imp_emp_norm = imp_emp / (imp_emp.mean() + 1e-14)

axes[1].plot(vols, imp_emp_norm, "o-", label="Empirical")
axes[1].plot(curve_cal.volumes, imp_cal_norm, "s--", label="MFG (calibrated)")
axes[1].set_xlabel("Volume")
axes[1].set_ylabel("Normalised impact")
axes[1].set_title("Price Impact")
axes[1].legend()

# Calibration loss history
axes[2].semilogy(result.loss_history, alpha=0.8)
axes[2].set_xlabel("Evaluation")
axes[2].set_ylabel("Loss")
axes[2].set_title("Calibration Convergence")

fig.tight_layout()
fig.savefig("04_calibration.png", dpi=120, bbox_inches="tight")
print("\nSaved: 04_calibration.png")
