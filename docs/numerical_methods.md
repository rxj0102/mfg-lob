# Numerical Methods

## Finite-Difference Discretisation

### Spatial Grid

Uniform grid with n points: x_i = x_min + i·Δx, i = 0,…,n-1.

For enhanced accuracy near the inventory boundary (q = 0), a tanh-stretched
grid is available: see `Grid1D(grid_type='tanh')`.

### HJB Solver (Backward Euler)

At each backward time step k (from T to 0):

1. Evaluate H(x, D^+ u^{k+1}, D² u^{k+1}, m^k) using upwind D^+ for the
   gradient and central differences for D².
2. Solve the tridiagonal system (implicit diffusion):

   ```
   (I - Δt·(σ²/2)·D²) u^k = u^{k+1} - Δt·(H + f)
   ```

   via the Thomas algorithm in O(n).

**Monotone upwind scheme (Barles-Souganidis):**
The key insight is to use *upwind* differences for the first derivative:

```
(D^upwind u)_i = max(b_i, 0)·(D^- u)_i + min(b_i, 0)·(D^+ u)_i
```

This ensures the scheme is monotone → convergence to viscosity solution.

### FP Solver (Implicit Upwind)

At each forward time step k (from 0 to T):

1. Compute optimal drift b* from u^k via the optimal control.
2. Solve the implicit upwind system:

   ```
   (I - Δt·L^T) m^{k+1} = m^k
   ```

   where L^T is the adjoint of the HJB operator (FP operator).
   Upwind advection ensures exact mass conservation: ∫m(x,t)dx = 1 for all t.

### CFL Stability

The explicit version requires CFL condition: |b|·Δt/Δx < 1.
The fully implicit scheme (θ = 1) is unconditionally stable.
Crank-Nicolson (θ = 0.5) is second-order in time but may oscillate near steep fronts.

## Fixed-Point Iteration

The Lasry-Lions algorithm:

```
for n = 0, 1, 2, ...:
    u^{n+1} = HJB(m^n)           # backward solve
    m̃^{n+1} = FP(u^{n+1})       # forward solve
    m^{n+1} = (1-λ)m^n + λ·m̃^{n+1}  # damped update
    if ||m^{n+1} - m^n|| < tol: break
```

Convergence rate: linear, with rate ≈ (1 - λ·C) where C depends on the
coupling strength α. Damping factor λ ∈ (0.3, 0.7] is typically sufficient.

## Grid Convergence

Expected convergence rates:
- Second-order in Δx for the diffusion (D²)
- First-order in Δx for the advection (upwind D^±)
- First-order in Δt for backward Euler; second-order for Crank-Nicolson
