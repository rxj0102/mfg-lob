# Mathematical Background

## The MFG-LOB PDE System

The coupled Hamilton-Jacobi-Bellman / Fokker-Planck system:

```
-∂u/∂t + H(x, Du, D²u, m) = 0          (HJB, backward)   (1)
 ∂m/∂t - ½ ∂²(σ²m)/∂x² + ∂(b*m)/∂x = 0  (FP, forward)    (2)
```

with boundary conditions:
```
u(T, x) = g(x, m(T))        (terminal cost)
m(0, x) = m₀(x)             (initial distribution)
```

### Hamiltonian

For a generic agent with state x, control α ∈ A, cost f, drift b, diffusion σ:

```
H(x, p, M, m) = sup_{α ∈ A} { -f(x,α,m) - b(x,α,m)·p - ½σ²(x,α,m)·M }
```

The supremum is achieved at the **optimal control** α*(x, p, m), given
by the first-order condition (Pontryagin maximum principle):

```
∂/∂α [-f - b·p - ½σ²·M] = 0
```

### Nash Equilibrium

A pair (u, m) satisfying (1)–(2) simultaneously constitutes a **Nash equilibrium**:
no individual agent can improve their expected payoff by deviating from the
strategy implied by α*(x, Du, m), given that all other agents follow the same strategy.

### Well-posedness

Under the **Lasry-Lions monotonicity condition**:

```
∫ (f(x,α,m₁) - f(x,α,m₂)) (m₁ - m₂) dx ≥ 0
```

for all m₁, m₂, the MFG system (1)–(2) has a **unique** solution.
This condition is satisfied by the congestion-type coupling f(x,α,m) = φ·m(x)
used in LOBFormationMFG.

## Viscosity Solutions

The HJB equation (1) is solved in the viscosity sense (Crandall-Lions, 1983).
The upwind finite-difference scheme is **monotone** and hence converges to the
unique viscosity solution by the Barles-Souganidis theorem (1991).

## Regularity

Under standard assumptions (bounded Lipschitz coefficients, non-degenerate
diffusion), the solution (u, m) satisfies:
- u ∈ C^{1,2}([0,T] × Ω)  (classical solution)
- m ∈ C([0,T]; L¹(Ω)) ∩ L²([0,T]; H¹(Ω))
