# mfg-lob

**Mean-Field Game solver and analysis toolkit for Limit Order Book models.**

`mfg-lob` is a research-grade Python library for numerically solving and analysing
mean-field game (MFG) models of limit order books (LOBs). The core mathematical
object is a coupled PDE system — a backward Hamilton-Jacobi-Bellman (HJB) equation
for the representative agent's optimal strategy, coupled with a forward
Fokker-Planck (FP) equation for the evolution of the population distribution.
Together they characterise the Nash equilibrium in a continuum of interacting traders.

## Why MFG for LOBs?

| Framework | Many agents | Analytical structure |
|-----------|:-----------:|:--------------------:|
| Kyle / Glosten-Milgrom | ✗ (1–2 agents) | ✓ |
| Agent-based models | ✓ | ✗ |
| **MFG-LOB** | **✓** | **✓** |

MFG provides the only tractable framework for modelling strategic interaction among
a large number of heterogeneous market participants while retaining PDE-level
mathematical structure — yielding closed-form characterisation of equilibria and
verified convergence guarantees.

## Features

- **Coupled HJB-FP solver** — finite-difference discretisation with Lasry-Lions
  fixed-point (Picard) iteration; convergence diagnostics included
- **Two agent archetypes** — symmetric market makers (Avellaneda-Stoikov style)
  and trend-following traders (quadratic execution cost)
- **LOB shape analysis** — extract equilibrium depth profiles, bid/ask liquidity,
  and spread proxies directly from the population density
- **Price impact** — compute I(v) from equilibrium; fit the power-law exponent β
  (empirically β ≈ 0.5, the square-root law)
- **Comparative statics** — trace how spread, depth, and price impact change with
  σ, φ, α, κ via automated parameter sweeps
- **Calibration** — fit model parameters to empirical (or synthetic) LOB snapshots
  by minimising a composite loss over spread, depth profile, and price impact

## Installation

```bash
pip install -e ".[dev]"   # development install with test dependencies
pip install .             # production install
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7

## Quick Start

```python
from mfg_lob.models import LOBModel
from mfg_lob.solvers import Grid, MFGSolver
from mfg_lob.analysis import PriceImpactAnalyzer, LOBShapeAnalyzer

# 1. Define the model
model = LOBModel(
    sigma=0.5,       # inventory diffusion coefficient
    phi=0.1,         # inventory aversion (quadratic running cost)
    alpha=0.5,       # mean-field coupling strength
    A_term=1.0,      # quadratic terminal liquidation cost
    agent_type="market_maker",
)

# 2. Set up the computational grid
grid = Grid.uniform(q_min=-8, q_max=8, nq=81, t_min=0.0, t_max=1.0, nt=101)

# 3. Solve the MFG equilibrium
solver = MFGSolver(model, grid, tol=1e-5, max_iter=80, damping=0.5, verbose=True)
eq = solver.solve()

print(f"Converged: {eq.converged} ({eq.n_iterations} iterations)")

# 4. Analyse the equilibrium
pia = PriceImpactAnalyzer(eq, price_tick=0.01)
curve = pia.compute(t=0.0)
print(f"Price impact exponent β = {curve.fit_params['beta']:.3f}")

lsa = LOBShapeAnalyzer(eq)
snap = lsa.snapshot(t=0.0)
print(f"Equilibrium spread proxy = {snap.spread_proxy:.4f}")
```

## Mathematical Background

The MFG-LOB system is:

```
−∂_t u − (σ²/2) ∂_qq u + H(q, ∂_q u, m) + f(q, m) = 0   (HJB, backward)
∂_t m − (σ²/2) ∂_qq m + ∂_q [b*(q, ∂_q u, m) m]  = 0   (FP, forward)
```

with terminal condition `u(q,T) = g(q)` and initial condition `m(q,0) = m₀(q)`.

**Notation:**
- `u(q,t)` — value function (expected payoff of optimal strategy)
- `m(q,t)` — population density (distribution of agent inventories)
- `H(q,p,m)` — Hamiltonian encoding the optimal quoting strategy
- `f(q,m)` — running cost (inventory penalty + mean-field coupling)
- `g(q)` — terminal liquidation cost
- `b*(q,p,m)` — optimal inventory drift (from Pontryagin maximum principle)

**Solver:** Backward Euler / Crank-Nicolson finite differences + Thomas
algorithm (O(n_q) per time step) + damped Picard fixed-point iteration.

## Examples

All examples are in `examples/`:

| Script | Description |
|--------|-------------|
| `01_basic_equilibrium.py` | Solve MFG and plot u, m, convergence |
| `02_price_impact.py` | Compute and fit the price impact curve |
| `03_comparative_statics.py` | Sweep σ and φ; trace LOB shape changes |
| `04_calibration.py` | Fit model to synthetic empirical LOB data |

Run an example:

```bash
cd examples
python 01_basic_equilibrium.py
```

## Package Layout

```
mfg_lob/
├── solvers/
│   ├── grid.py          # Grid: uniform finite-difference grid
│   ├── hjb_solver.py    # HJBSolver: backward-in-time PDE solver
│   ├── fp_solver.py     # FPSolver: forward-in-time PDE solver
│   └── mfg_solver.py    # MFGSolver + MFGEquilibrium: fixed-point iteration
├── models/
│   ├── lob_model.py     # LOBModel: full model specification
│   ├── costs.py         # QuadraticCost, ExponentialCost, LOBCostTerminal
│   └── hamiltonians.py  # MarketMakerHamiltonian, TrendFollowerHamiltonian
├── analysis/
│   ├── price_impact.py      # PriceImpactAnalyzer, PriceImpactCurve
│   ├── lob_shape.py         # LOBShapeAnalyzer, LOBSnapshot
│   └── comparative_statics.py  # ComparativeStatics, StaticsResult
└── calibration/
    ├── empirical.py    # LOBSnapshot, EmpiricalLOB: data structures
    └── calibrator.py   # MFGCalibrator, CalibrationResult
```

## Tests

```bash
pytest tests/ -v
```

49 tests covering grid, models, all solvers, analysis, and calibration.

## Target Venues

- *Mathematical Finance*
- *SIAM Journal on Financial Mathematics*
- *Finance and Stochastics*
- *Quantitative Finance*

## License

MIT — see `LICENSE`.
