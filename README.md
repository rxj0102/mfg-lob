# mfg-lob

**Mean-Field Game solver and analysis toolkit for Limit Order Book models.**

`mfg-lob` is a research-grade Python library for numerically solving and analysing
mean-field game (MFG) models of limit order books (LOBs).  The core mathematical
object is a coupled PDE system — a backward Hamilton-Jacobi-Bellman (HJB) equation
for the representative agent's optimal strategy, paired with a forward Fokker-Planck
(FP) equation for the population distribution.  Together they characterise the Nash
equilibrium in a continuum of interacting traders.

## Why MFG for LOBs?

| Framework | Many agents | Analytical structure | Convergence guarantees |
|-----------|:-----------:|:--------------------:|:---------------------:|
| Kyle / Glosten-Milgrom | ✗ (1–2 agents) | ✓ | — |
| Agent-based models | ✓ | ✗ | ✗ |
| **mfg-lob** | **✓** | **✓** | **✓** |

MFG provides the only tractable framework for modelling strategic interaction among
a large number of heterogeneous market participants while retaining PDE-level
mathematical structure — giving closed-form characterisation of equilibria and
verified convergence guarantees (Barles-Souganidis theorem).

---

## Features

- **Coupled HJB-FP solver** — IMEX finite-difference discretisation with Picard
  fixed-point iteration and optional Newton-Krylov acceleration
- **Three MFG-LOB models:**
  - `AvellanedaStoikovMFG` — market-maker inventory management with Poisson executions
  - `OptimalExecutionMFG` — meta-order liquidation with temporary price impact
  - `LOBFormationMFG` — equilibrium depth profile from adverse selection + congestion
- **Equilibrium analysis** — ε-Nash verification, Price of Anarchy, comparative statics, stability
- **Price impact** — extract ΔS(Q) from equilibrium; fit power-law exponent δ; compare to √Q law
- **Calibration** — fit parameters to synthetic or empirical LOB data via DE + Nelder-Mead
- **Convergence study** — verified O(Δx^{1.3–1.6}) convergence on refined grids
- **459 tests** across all modules

---

## Quick Start

Compute a Nash equilibrium in ≈ 15 lines:

```python
from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.mfg_solver import MFGSolver
from mfglob.price_impact import PriceImpactAnalyzer
import numpy as np

# 1. Define model
model = LOBFormationMFG(c_far=0.1, c_near=1.0, kappa=2.0,
                        c_crowd=0.3, c_control=0.1, sigma=0.5)

# 2. Computational grid
grid  = Grid1D(0.0, 8.0, 80)       # 80 price levels, [0, 8] ticks
tgrid = TimeGrid(T=0.5, n_t=60)    # 60 time steps, horizon T=0.5

# 3. Solve MFG equilibrium
solver = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3, max_iterations=50)
sol    = solver.solve(verbose=True)

print(f"Converged: {sol['converged']}  ({sol['n_iterations']} iters)")

# 4. Inspect equilibrium LOB shape
x     = grid.points
m_avg = np.mean(sol['density'], axis=0)
m_avg /= np.trapezoid(m_avg, x)
print(f"LOB mode (depth peak):  x = {x[np.argmax(m_avg)]:.4f} ticks")
```

---

## Optimal Execution with Price Impact

```python
from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.optimal_execution import OptimalExecutionMFG
from mfglob.mfg_solver import MFGSolver
from mfglob.price_impact import PriceImpactAnalyzer
import numpy as np

model  = OptimalExecutionMFG(phi=0.5, psi=1.0, eta=0.2, Q0=1.0, sigma=0.05)
grid   = Grid1D(0.0, 1.5, 50)
tgrid  = TimeGrid(0.5, 50)
sol    = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-3).solve()
pia    = PriceImpactAnalyzer(sol, model, grid, tgrid)

Qs     = np.linspace(0.1, 1.8, 10)
result = pia.compare_to_square_root_law(Qs)
print(f"Fitted exponent δ = {result['mfg_exponent']:.3f}  (sqrt law: 0.5)")
print(f"Kyle's lambda     = {pia.kyle_lambda():.4f}")
```

---

## Calibration

```python
from mfglob.models.lob_formation import LOBFormationMFG
from mfglob.calibration import MFGCalibrator
from data.synthetic_lob import generate_synthetic_book_shape
import numpy as np

snap    = generate_synthetic_book_shape(n_levels=30, shape="exponential", seed=42)
density = 0.5 * (snap["volumes_bid"] + snap["volumes_ask"])
levels  = np.asarray(snap["price_levels"])

cal = MFGCalibrator(
    model_class=LOBFormationMFG,
    target_data={"book_shape": {"levels": levels, "density": density}},
    param_bounds={"c_crowd": (0.05, 3.0), "sigma": (0.1, 2.0)},
    grid_params={"n_x": 30, "n_t": 20, "T": 0.3, "x_min": 0.0, "x_max": 8.0},
)
result = cal.calibrate(method="nelder_mead", verbose=False)
print(result["params"])      # {'c_crowd': ..., 'sigma': ...}
print(result["objective"])   # composite MSE loss
```

---

## Installation

```bash
git clone https://github.com/rxj0102/mfg-lob
cd mfg-lob
pip install -e ".[dev]"      # development install with test dependencies
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10

Optional: `matplotlib` for plots (experiments + notebooks).

---

## Package Layout

```
mfglob/                        ← core library
├── grids.py                   ← Grid1D, TimeGrid
├── hjb.py                     ← HJBSolver (backward IMEX)
├── fokker_planck.py           ← FokkerPlanckSolver (forward IMEX)
├── mfg_solver.py              ← MFGSolver (Picard + Newton-Krylov)
├── equilibrium.py             ← EquilibriumAnalyzer, PoA, social cost
├── price_impact.py            ← PriceImpactAnalyzer, impact curves
├── calibration.py             ← MFGCalibrator (DE + Nelder-Mead)
├── operators.py               ← finite-difference operator matrices
└── models/
    ├── base.py                ← MFGLOBModel abstract base class
    ├── avellaneda_stoikov.py  ← AvellanedaStoikovMFG
    ← optimal_execution.py    ← OptimalExecutionMFG
    └── lob_formation.py       ← LOBFormationMFG

data/
├── synthetic_lob.py           ← synthetic LOB data generators
└── lob_statistics.py          ← empirical LOB statistics

experiments/
├── run_convergence_study.py   ← grid convergence (spatial + temporal)
├── run_equilibrium.py         ← equilibrium heatmaps for all 3 models
├── run_comparative_statics.py ← parameter sensitivity tables
├── run_price_impact.py        ← impact curves + power-law fit
└── run_calibration.py         ← calibration to synthetic data

notebooks/
├── 01_mfg_primer.ipynb        ← What is an MFG?  LQ example.
├── 02_lob_model.ipynb         ← LOBFormation setup and equilibrium.
├── 03_equilibrium_analysis.ipynb  ← Comparative statics, PoA, stability.
└── 04_price_impact.ipynb      ← Impact curves, sqrt law, parameter sensitivity.

docs/
├── math_background.md         ← MFG theory, viscosity solutions, proofs
├── numerical_methods.md       ← Finite differences, Picard/Newton, convergence
└── model_specifications.md    ← Full PDE specs, parameter tables, analytics
```

---

## Running Experiments

```bash
python experiments/run_convergence_study.py   # spatial + temporal rates
python experiments/run_equilibrium.py         # heatmaps for all models
python experiments/run_comparative_statics.py # parameter sweeps
python experiments/run_price_impact.py        # impact curve + power law
python experiments/run_calibration.py         # calibration to synthetic data
```

Results are saved to `results/` as JSON (and PNG if matplotlib is installed).

---

## Tests

```bash
pytest tests/ --ignore=tests/test_models.py -q
# 459 passed
```

Test coverage spans grids, HJB solver, FP solver, MFG solver, all three models,
equilibrium analysis, price impact, calibration, synthetic data, and LOB statistics.

---

## Mathematical Background

The MFG-LOB system is the coupled PDE:

```
−∂_t u  − (σ²/2) ∂_xx u  + H(x, ∂_x u, m) = 0   (HJB, backward)
 ∂_t m  − (σ²/2) ∂_xx m  + ∂_x[b*(x,∂_x u,m)·m] = 0   (FP, forward)
```

**Notation:**
- `u(x,t)` — value function (optimal expected cost)
- `m(x,t)` — population density (distribution of agent states)
- `H(x,p,m)` — Hamiltonian (encodes the optimal strategy via Pontryagin)
- `b*(x,p,m)` — optimal drift (from the first-order condition ∂H/∂p = 0)

**Solver:** IMEX finite differences + Thomas algorithm (O(n_x) per time step) +
damped Picard fixed-point iteration, accelerated by Newton-Krylov near convergence.

See `docs/math_background.md` for full theory, `docs/numerical_methods.md` for
scheme details.

---

## References

1. **Lasry & Lions (2007).** "Mean field games."  *Japanese Journal of Mathematics* 2, 229–260.
2. **Huang, Malhamé & Caines (2006).** "Large population stochastic dynamic games."  *Communications in Information and Systems* 6, 221–252.
3. **Cardaliaguet & Lehalle (2018).** "Mean field game of controls and an application to trade crowding."  *Mathematics and Financial Economics* 12, 335–363.
4. **Avellaneda & Stoikov (2008).** "High-frequency trading in a limit order book."  *Quantitative Finance* 8, 217–224.
5. **Carmona & Delarue (2018).** *Probabilistic Theory of Mean Field Games with Applications* (2 vols).  Springer.
6. **Achdou & Capuzzo-Dolcetta (2010).** "Mean field games: numerical methods."  *SIAM J. Numerical Analysis* 48, 1136–1162.
7. **Almgren & Chriss (2001).** "Optimal execution of portfolio transactions."  *Journal of Risk* 3, 5–39.

---

## License

MIT — see `LICENSE`.
