# Numerical Methods

## 1. Finite-Difference Discretisation

### Spatial Grid

Uniform grid with n_x points:

```
x_i = x_min + i · Δx,    i = 0, …, n_x − 1,    Δx = (x_max − x_min) / (n_x − 1)
```

For inventory-boundary problems (q ≥ 0), the domain [0, Q_max] is chosen so that
the initial density m₀ is well-supported inside the grid.

### Time Grid

Uniform grid with n_t steps:

```
t_k = k · Δt,    k = 0, …, n_t,    Δt = T / n_t
```

HJB is solved backward: k = n_t, n_t−1, …, 0.
FP is solved forward: k = 0, 1, …, n_t.

---

## 2. HJB Solver (Backward IMEX)

At each backward time step k, given m^k and u^{k+1}:

### Step 1: Upwind gradient

```
(D^+ u)_i = (u_{i+1} − u_i) / Δx           (forward difference)
(D^- u)_i = (u_i − u_{i-1}) / Δx           (backward difference)
```

For a drift coefficient b_i, the upwind first-derivative is:

```
(D^upwind u)_i = max(b_i, 0) · (D^- u)_i + min(b_i, 0) · (D^+ u)_i
```

This scheme is **monotone**: the numerical operator is non-decreasing in u_{i−1}
and u_{i+1} and non-increasing in u_i (for standard b).

### Step 2: Diffusion term (implicit)

The diffusion (σ²/2) ∂_{xx} u is handled implicitly:

```
(I − Δt · (σ²/2) · D²) u^k = rhs
```

where D² is the second-order central-difference operator.  The tridiagonal system
is solved via the Thomas algorithm in O(n_x) operations.

### Step 3: Advection and source (explicit)

```
rhs = u^{k+1} − Δt · H(x, D^upwind u^{k+1}, m^k) − Δt · f(x, m^k)
```

The IMEX (implicit–explicit) split ensures unconditional stability for the
diffusion while the advection follows the CFL restriction.

### Boundary Conditions

Dirichlet at both ends: `u(t, x_min) = u(t, x_max) = 0` (reflecting barriers,
or Neumann via ghost cells for Neumann BCs).

---

## 3. FP Solver (Forward IMEX Upwind)

At each forward time step k, given u^k and m^k:

### Upwind advection (explicit)

The optimal drift `b*_i = b(x_i, α*(x_i, ∂_x u^k_i, m^k_i), m^k_i)` is
computed from the HJB solution.  The flux is discretised as:

```
F_{i+1/2} = max(b*_{i+1/2}, 0) · m^k_i + min(b*_{i+1/2}, 0) · m^k_{i+1}
```

(upwind-in-x flux).

### Diffusion (implicit)

```
(I − Δt · (σ²/2) · D²) m^{k+1} = m^k − Δt · ∂_x F
```

Implicit diffusion ensures the scheme is unconditionally stable and preserves
mass: ∑_i m^{k+1}_i Δx = ∑_i m^k_i Δx exactly.

### Positivity

The upwind scheme satisfies a discrete maximum principle: if m^k ≥ 0, then
m^{k+1} ≥ 0, provided the CFL condition holds for the advection part.

---

## 4. Barles-Souganidis Theorem

The key convergence theorem (Barles & Souganidis 1991) states:

> A numerical scheme that is **monotone**, **stable** (in L∞), and **consistent**
> converges locally uniformly to the unique viscosity solution of the HJB equation,
> as Δx, Δt → 0.

### Verification for this scheme

- **Monotone:** The upwind discretisation of b · ∂_x u is monotone (non-decreasing
  in neighbouring values) when the sign of b is correctly tracked.
- **Stable:** The L∞ norm of u is bounded by the terminal cost g plus accumulated
  running costs, independent of Δx, Δt (for Δt/Δx² bounded).
- **Consistent:** By Taylor expansion, the truncation error is O(Δx) for upwind
  advection and O(Δx²) for the diffusion term.

Therefore the scheme converges to the viscosity solution as the grid is refined.

---

## 5. Picard Iteration (Lasry-Lions Algorithm)

The fixed-point iteration for the coupled HJB-FP system:

```
Initialise m⁰ = m₀  (broadcast to all times, or uniform)

For n = 0, 1, 2, …:
    u^{n+1}   = HJB(m^n)                    # backward solve
    m̃^{n+1}  = FP(u^{n+1})                  # forward solve
    m^{n+1}  = (1 − λ) m^n + λ m̃^{n+1}     # damped update
    err       = ‖m^{n+1} − m^n‖ / ‖m^n‖
    if err < tol: converged; break
```

### Convergence Analysis

Under the Lasry-Lions monotonicity condition, the Picard map is a contraction
in a suitable norm with contraction constant ρ < 1.  The damping factor λ ∈ (0,1]
controls the effective rate: too large (λ = 1) may overshoot; too small requires
many iterations.

Typical choice: λ = 0.5 gives geometric convergence with rate ≈ 0.5.

### Newton-Krylov Acceleration

Once Picard has brought the iterate near the fixed point, a Newton step on the
residual `F(m) = T(m) − m = 0` (where T is the Picard map) can accelerate
convergence.  We use `scipy.optimize.newton_krylov` with the LGMRES inner solver,
which avoids forming the full Jacobian by approximating its action via finite
differences of F.

Convergence is quadratic near the fixed point, making the combined Picard
(warm-start) + Newton approach highly efficient.

---

## 6. Grid Convergence Theory

### Expected Rates

| Term          | Discretisation | Expected rate |
|---------------|---------------|---------------|
| Diffusion     | Central diff  | O(Δx²)        |
| Advection     | Upwind        | O(Δx)         |
| Time stepping | Backward Euler| O(Δt)         |
| Overall       | IMEX upwind   | O(Δx + Δt)    |

For smooth solutions (no shocks), the advection truncation error can be improved
to O(Δx²) with a Lax-Wendroff correction.

### Observed Rates

Numerical experiments on the LOBFormationMFG (see `experiments/run_convergence_study.py`)
show fitted rates ≈ 1.3–1.6 for both spatial and temporal refinement.  The super-
linear rate (between 1 and 2) is explained by the fact that the density m is
smooth for these parameters: upwind error is effectively O(Δx^{1.5}) via the
Bramble-Hilbert argument when the solution is in H².

### Richardson Extrapolation

Given solutions m_h and m_{h/2} on two grids:

```
m_exact ≈ m_{h/2} + (m_{h/2} − m_h) / (2^p − 1)
```

where p is the observed convergence rate.  This provides an O(Δx^{2p}) estimate
of the exact solution from two coarse-grid runs.

---

## 7. Calibration: Composite Loss Minimisation

The `MFGCalibrator` minimises a weighted MSE composite loss:

```
L(θ) = w_book · ‖m_θ[-1] / ‖m_θ[-1]‖ − m_target / ‖m_target‖‖²
      + w_impact · ‖ΔS_θ(Q) − ΔS_target(Q)‖²
      + w_spread · (1/k_θ − spread_target)²
```

over model parameters θ = (c_crowd, sigma, eta, phi, …).

### Optimisation Methods

1. **Differential Evolution** (DE): global stochastic search over the bounded
   parameter space.  Robust to local minima; expensive (50 generations × 5 pop).

2. **Nelder-Mead simplex:** efficient local search in normalised [0,1]^d space.
   Fast per evaluation; may converge to local minima.

3. **Two-stage:** DE for global search → Nelder-Mead for refinement.  Best
   trade-off for smooth-ish objectives.

Each objective evaluation requires a full MFG solve (coarse grid: n_x=20–30,
n_t=15–20), taking ≈ 0.1–0.5 s.
